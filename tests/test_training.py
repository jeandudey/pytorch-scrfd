# SPDX-FileCopyrightText: 2026 Jean-Pierre De Jesus DIAZ <me@jeandudey.tech>
#
# SPDX-License-Identifier: MIT

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from scrfd import CONFIGS, build_model, load_snapshot
from scrfd.anchors import grid_anchors
from scrfd.assign import atss_assign
from scrfd.dataset import (
    CROP_CHOICES,
    RetinaFaceDataset,
    horizontal_flip,
    load_records,
    parse_object,
    photometric_distortion,
    random_square_crop,
    resize,
)
from scrfd.evaluate import average_precision, match_image
from scrfd.loss import SCRFDLoss, diou_loss
from scrfd.train import RECIPES, Recipe, learning_rate, main

ANNOTATIONS = """\
# a.jpg 200 100
60 20 100 60 70 30 0 90 30 0 80 40 0 70 50 0 85 50 0
# b.jpg 120 160
10 10 50 60
80 90 110 150 1
"""


def write_dataset(root: Path, split: str = "train") -> Path:
    folder = root / split
    (folder / "images").mkdir(parents=True)
    (folder / "labelv2.txt").write_text(ANNOTATIONS)
    rng = np.random.default_rng(0)
    for name, (width, height) in {"a.jpg": (200, 100), "b.jpg": (120, 160)}.items():
        cv2.imwrite(
            str(folder / "images" / name), rng.integers(0, 255, (height, width, 3), np.uint8)
        )
    return folder


def test_parse_object() -> None:
    box, keypoints, ignore = parse_object([1, 2, 3, 4, *([5, 6, 0] * 4), -1, -1, -1])
    assert box.tolist() == [1, 2, 3, 4]
    assert keypoints[:, 2].tolist() == [1, 1, 1, 1, 0]
    assert not ignore
    assert parse_object([1, 2, 3, 4, 1])[2]
    assert not parse_object([1, 2, 3, 4, 0])[2]
    assert parse_object([1, 2, 3, 4])[1].shape == (5, 3)


def test_load_records(tmp_path: Path) -> None:
    path = tmp_path / "labels.txt"
    path.write_text(ANNOTATIONS)
    first, second = load_records(path)
    assert (first.name, first.width, first.height) == ("a.jpg", 200, 100)
    assert first.boxes.shape == (1, 4)
    assert first.keypoints.shape == (1, 5, 3)
    assert second.boxes.shape == (1, 4)
    assert second.ignored.tolist() == [[80, 90, 110, 150]]


def test_square_crop_keeps_box_centres() -> None:
    rng = np.random.default_rng(1)
    image = np.zeros((100, 200, 3), np.float32)
    boxes = np.array([[60, 20, 100, 60]], np.float32)
    keypoints = np.zeros((1, 5, 3), np.float32)
    keypoints[:, :, :2] = [80, 40]
    for _ in range(50):
        crop, new_boxes, new_keypoints = random_square_crop(image, boxes, keypoints, rng)
        assert crop.shape[0] == crop.shape[1]
        side = crop.shape[0]
        assert len(new_boxes) == 1
        centre = (new_boxes[0, :2] + new_boxes[0, 2:]) / 2
        assert (centre > 0).all()
        assert (centre < side).all()
        assert (new_boxes >= 0).all()
        assert (new_boxes <= side).all()
        assert new_keypoints.shape == (1, 5, 3)


def test_square_crop_weights_favour_large_crops() -> None:
    rng = np.random.default_rng(2)
    image = np.zeros((100, 100, 3), np.float32)
    boxes = np.array([[40, 40, 60, 60]], np.float32)
    keypoints = np.zeros((1, 5, 3), np.float32)
    weights = [0.0] * (len(CROP_CHOICES) - 1) + [1.0]
    for _ in range(10):
        crop, _, _ = random_square_crop(image, boxes, keypoints, rng, weights=weights)
        assert crop.shape[0] == int(CROP_CHOICES[-1] * 100)


def test_resize_and_flip() -> None:
    image = np.zeros((50, 100, 3), np.float32)
    image[:, :10] = 255
    boxes = np.array([[0, 0, 10, 50]], np.float32)
    keypoints = np.zeros((1, 5, 3), np.float32)
    keypoints[0, :, 0] = [1, 2, 3, 4, 5]
    image, boxes, keypoints = resize(image, boxes, keypoints, 20)
    assert image.shape == (20, 20, 3)
    assert boxes.tolist() == [[0, 0, 2, 20]]
    flipped, flipped_boxes, flipped_keypoints = horizontal_flip(image, boxes, keypoints)
    assert flipped_boxes.tolist() == [[18, 0, 20, 20]]
    assert flipped[0, -1, 0] == 255
    # eyes and mouth corners swap, the nose stays
    assert flipped_keypoints[0, :, 0].tolist() == pytest.approx(
        [20 - x * 0.2 for x in (2, 1, 3, 5, 4)]
    )


def test_photometric_distortion_shape() -> None:
    rng = np.random.default_rng(2)
    image = np.random.default_rng(3).uniform(0, 255, (16, 16, 3)).astype(np.float32)
    for _ in range(20):
        out = photometric_distortion(image, rng)
        assert out.shape == image.shape
        assert out.dtype == np.float32
        assert np.isfinite(out).all()


def test_dataset_sample(tmp_path: Path) -> None:
    folder = write_dataset(tmp_path)
    dataset = RetinaFaceDataset(folder / "labelv2.txt", folder / "images", size=64)
    assert len(dataset) == 2
    sample = dataset[0]
    assert sample.image.shape == (3, 64, 64)
    assert sample.boxes.shape == (1, 4)
    assert sample.keypoints.shape == (1, 5, 3)
    again = dataset[0]
    assert torch.equal(sample.image, again.image)
    dataset.epoch = 1
    assert not torch.equal(sample.image, dataset[0].image)


def test_anchor_layout() -> None:
    anchors = grid_anchors(2, 3, 8, 16, (1, 2), (1.0,), torch.device("cpu"))
    assert anchors.shape == (12, 4)
    assert anchors[0].tolist() == [-8, -8, 8, 8]
    assert anchors[1].tolist() == [-16, -16, 16, 16]
    assert anchors[2].tolist() == [0, -8, 16, 8]
    wide = grid_anchors(1, 1, 8, 8, (3,), (2.0,), torch.device("cpu"))
    assert (wide[0, 3] - wide[0, 1]) / (wide[0, 2] - wide[0, 0]) == pytest.approx(2.0)


def test_atss_assigns_nearby_anchors() -> None:
    device = torch.device("cpu")
    levels = [(8, 16), (16, 64)]
    anchors = torch.cat([grid_anchors(8, 8, s, size, (1, 2), (1.0,), device) for s, size in levels])
    counts = [8 * 8 * 2, 8 * 8 * 2]
    gt = torch.tensor([[20.0, 20.0, 50.0, 50.0]])
    assigned = atss_assign(anchors, counts, gt)
    positive = (assigned >= 0).nonzero().flatten()
    assert 0 < len(positive) <= 18
    centres = (anchors[positive, :2] + anchors[positive, 2:]) / 2
    assert (centres > gt[0, :2]).all()
    assert (centres < gt[0, 2:]).all()
    assert (atss_assign(anchors, counts, gt.new_zeros(0, 4)) == -1).all()


def test_diou_loss() -> None:
    box = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    assert diou_loss(box, box).item() == pytest.approx(0, abs=1e-5)
    assert diou_loss(box, box + 100).item() > 1


@pytest.mark.parametrize("name", ["scrfd_500m", "scrfd_2.5g_bnkps", "scrfd_crowdhuman_2.5g_bnkps"])
def test_loss_backward(name: str) -> None:
    torch.manual_seed(0)
    model = build_model(name)
    model.init_weights()
    loss_fn = SCRFDLoss(CONFIGS[name])
    images = torch.randn(2, 3, 128, 128)
    gt_boxes = [
        torch.tensor([[30.0, 30, 90, 100], [5, 5, 40, 40]]),
        torch.tensor([[50.0, 40, 100, 90]]),
    ]
    gt_keypoints = [torch.rand(2, 5, 3), torch.rand(1, 5, 3)]
    losses = loss_fn(model.predict(images), (128, 128), gt_boxes, gt_keypoints)
    assert set(losses) == {"loss_cls", "loss_bbox", *(["loss_kps"] if model.use_kps else [])}
    total = sum(losses.values())
    assert torch.isfinite(total)
    total.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


def test_learning_rate() -> None:
    recipe = Recipe(lr=0.1, lr_steps=(2, 4), warmup_iters=10)
    assert learning_rate(recipe, 0, 0) == pytest.approx(0.1 * 0.001)
    assert learning_rate(recipe, 0, 10) == pytest.approx(0.1)
    assert learning_rate(recipe, 2, 100) == pytest.approx(0.01)
    assert learning_rate(recipe, 4, 100) == pytest.approx(0.001)
    assert set(RECIPES) <= set(CONFIGS)


def test_average_precision() -> None:
    scores = np.array([0.9, 0.8, 0.7], np.float32)
    perfect = average_precision(scores, np.array([1, 1, 1], bool), np.zeros(3, bool), 3)
    assert perfect == pytest.approx(1.0)
    half = average_precision(scores, np.array([1, 0, 0], bool), np.array([0, 1, 1], bool), 2)
    assert half == pytest.approx(0.5)


def test_match_image(tmp_path: Path) -> None:
    path = tmp_path / "labels.txt"
    path.write_text(ANNOTATIONS)
    _, second = load_records(path)
    boxes = torch.tensor(
        [[10.0, 10, 50, 60], [10, 10, 50, 60], [82, 92, 108, 148], [300, 300, 340, 340]]
    )
    tp, fp = match_image(boxes, torch.tensor([0.9, 0.8, 0.7, 0.6]), second)
    assert tp.tolist() == [True, False, False, False]
    assert fp.tolist() == [False, True, False, True]


def test_train_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write_dataset(tmp_path)
    write_dataset(tmp_path, "val")
    output = tmp_path / "out"
    main(
        [
            "scrfd_500m_bnkps",
            "--data-root", str(tmp_path),
            "--output", str(output),
            "--epochs", "2",
            "--batch-size", "2",
            "--size", "128",
            "--workers", "0",
            "--device", "cpu",
            "--snapshot-interval", "1",
            "--val-interval", "2",
            "--log-interval", "1",
        ]
    )  # fmt: skip
    out = capsys.readouterr().out
    assert "loss_cls" in out
    assert "val AP@0.5" in out
    snapshot = output / "epoch_2.pth"
    assert (output / "epoch_1.pth").exists()
    recipe = json.loads((output / "recipe.json").read_text())
    assert recipe["variant"] == "scrfd_500m_bnkps" and recipe["epochs"] == 2
    assert recipe["data_root"] == str(tmp_path)
    assert (output / "latest.pth").resolve() == snapshot.resolve()
    load_snapshot(build_model("scrfd_500m_bnkps"), snapshot)
    main(
        [
            "scrfd_500m_bnkps",
            "--data-root", str(tmp_path),
            "--output", str(output),
            "--epochs", "3",
            "--batch-size", "2",
            "--size", "128",
            "--workers", "0",
            "--device", "cpu",
            "--snapshot-interval", "1",
            "--val-interval", "0",
            "--resume", str(snapshot),
        ]
    )  # fmt: skip
    assert (output / "epoch_3.pth").exists()
