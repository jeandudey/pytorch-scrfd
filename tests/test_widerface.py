# SPDX-FileCopyrightText: 2026 Jean-Pierre De Jesus DIAZ <me@jeandudey.tech>
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import numpy as np
import pytest

from scrfd.widerface import Image, load_ground_truth, match, overlaps, score, voc_ap

BOXES = np.array([[10, 10, 59, 59], [100, 100, 149, 149], [200, 200, 249, 249]], dtype=float)


def image(keep_hard: list[bool], keep_easy: list[bool] | None = None) -> Image:
    keep = np.array(keep_hard)
    return Image(
        "event",
        "a",
        BOXES,
        {
            "easy": np.array(keep_easy if keep_easy is not None else keep_hard),
            "medium": keep,
            "hard": keep,
        },
    )


def detections(boxes: np.ndarray, scores: list[float]) -> np.ndarray:
    return np.concatenate([boxes, np.array(scores)[:, None]], axis=1)


def test_overlaps() -> None:
    iou = overlaps(BOXES[:1], np.array([[10, 10, 59, 59], [35, 10, 84, 59], [500, 500, 520, 520]]))
    assert iou[0, 0] == pytest.approx(1.0)
    assert iou[0, 1] == pytest.approx(25 * 50 / (2 * 2500 - 25 * 50))
    assert iou[0, 2] == 0


def test_voc_ap() -> None:
    assert voc_ap(np.array([0.5, 1.0]), np.array([1.0, 1.0])) == pytest.approx(1.0)
    assert voc_ap(np.array([0.5]), np.array([1.0])) == pytest.approx(0.5)


def test_match() -> None:
    best_iou = np.array([0.9, 0.9, 0.9, 0.2])
    best_gt = np.array([0, 0, 1, 0])
    recall, proposal = match(best_iou, best_gt, np.array([True, False, True]))
    # the second detection duplicates face 0 and the third matches face 1, which is not kept
    assert recall.tolist() == [1, 1, 1, 1]
    assert proposal.tolist() == [True, True, False, True]


def test_match_unkept_face_is_ignored() -> None:
    recall, proposal = match(np.array([0.9, 0.9]), np.array([1, 0]), np.array([True, False]))
    assert recall.tolist() == [0, 1]
    assert proposal.tolist() == [False, True]


def test_score() -> None:
    perfect = [detections(BOXES, [0.9, 0.8, 0.7])]
    assert score([image([True] * 3)], perfect) == pytest.approx(
        {"easy": 1.0, "medium": 1.0, "hard": 1.0}, abs=1e-3
    )
    half = [detections(BOXES[:1], [0.9])]
    assert score([image([True, False, False])], half)["hard"] == pytest.approx(1.0, abs=1e-3)
    assert score([image([True] * 3)], half)["hard"] == pytest.approx(1 / 3, abs=1e-3)
    assert score([image([True] * 3)], [np.zeros((0, 5))]) == {"easy": 0, "medium": 0, "hard": 0}


def test_unkept_faces_are_not_penalised() -> None:
    # easy keeps only the first face; detecting the other two must not hurt easy
    easy_only = image([True, True, True], [True, False, False])
    ap = score([easy_only], [detections(BOXES, [0.9, 0.8, 0.7])])
    assert ap["easy"] == pytest.approx(1.0, abs=1e-3)


def test_load_ground_truth(tmp_path: Path) -> None:
    savemat = pytest.importorskip("scipy.io").savemat

    def cell(items: list[object]) -> np.ndarray:
        out = np.empty((len(items), 1), dtype=object)
        for i, item in enumerate(items):
            out[i, 0] = item
        return out

    xywh = np.array([[10, 10, 50, 50], [100, 100, 50, 50], [200, 200, 50, 50]], dtype=np.uint16)
    common = {
        "event_list": cell(["0--Parade"]),
        "file_list": cell([cell(["a"])]),
        "face_bbx_list": cell([cell([xywh])]),
    }
    savemat(tmp_path / "wider_face_val.mat", common)
    for setting, kept in {"easy": [1], "medium": [1, 2], "hard": [1, 2, 3]}.items():
        gt_list = cell([cell([np.array(kept, dtype=np.uint8).reshape(-1, 1)])])
        savemat(tmp_path / f"wider_{setting}_val.mat", {"gt_list": gt_list})

    (loaded,) = load_ground_truth(tmp_path)
    assert (loaded.event, loaded.name) == ("0--Parade", "a")
    assert loaded.boxes.tolist() == [[10, 10, 60, 60], [100, 100, 150, 150], [200, 200, 250, 250]]
    assert loaded.keep["easy"].tolist() == [True, False, False]
    assert loaded.keep["medium"].tolist() == [True, True, False]
    assert loaded.keep["hard"].tolist() == [True, True, True]
