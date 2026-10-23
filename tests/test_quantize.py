# SPDX-FileCopyrightText: 2026 Jean-Pierre De Jesus DIAZ <me@jeandudey.tech>
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pytest
import torch

from scrfd.export import export_onnx
from scrfd.quantize import calibration_images, main
from scrfd.variants import build_model
from scrfd.widerface import OnnxDetector

SIZE = 160


@pytest.fixture
def calibration(tmp_path: Path) -> Path:
    folder = tmp_path / "calibration"
    folder.mkdir()
    rng = np.random.default_rng(0)
    for i in range(6):
        cv2.imwrite(str(folder / f"{i}.jpg"), rng.integers(0, 255, (120, 200, 3), np.uint8))
    (folder / "notes.txt").write_text("not an image")
    return folder


def test_calibration_images_are_seeded(calibration: Path) -> None:
    first = calibration_images(calibration, 4, seed=1)
    assert first == calibration_images(calibration, 4, seed=1)
    assert len(first) == 4
    assert all(p.suffix == ".jpg" for p in first)
    assert len(calibration_images(calibration, 100, seed=1)) == 6
    empty = calibration.parent / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no images"):
        calibration_images(empty, 1, 0)


@pytest.mark.parametrize("method", ["minmax", "percentile"])
def test_quantize(tmp_path: Path, calibration: Path, method: str) -> None:
    snapshot = tmp_path / "model.pth"
    torch.save(build_model("scrfd_500m_bn").state_dict(), snapshot)
    output = tmp_path / "int8.onnx"
    fp32 = tmp_path / "fp32.onnx"
    main(
        [
            "scrfd_500m_bn",
            "--snapshot",
            str(snapshot),
            "--calibration",
            str(calibration),
            "--output",
            str(output),
            "--keep-fp32",
            str(fp32),
            "--num-images",
            "4",
            "--size",
            str(SIZE),
            "--method",
            method,
        ]
    )
    assert output.stat().st_size < fp32.stat().st_size
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    assert [o.name for o in session.get_outputs()] == [
        f"{k}_{s}" for k in ("score", "bbox") for s in (8, 16, 32)
    ]
    (scores, *_) = session.run(None, {"image": np.zeros((1, 3, SIZE, SIZE), np.float32)})
    assert np.asarray(scores).shape == (1, 2 * (SIZE // 8) ** 2, 1)


def test_onnx_detector_matches_pytorch(tmp_path: Path) -> None:
    model = build_model("scrfd_500m_bn").eval()
    path = tmp_path / "m.onnx"
    export_onnx(model, path, (SIZE, SIZE))
    x = torch.randn(1, 3, SIZE, SIZE)
    detector = OnnxDetector(path, model)
    with torch.no_grad():
        expected = model(x)
    outputs = detector.session.run(None, {"image": x.numpy()})
    assert len(outputs) == len(expected)
    for found, wanted in zip(outputs, expected, strict=True):
        np.testing.assert_allclose(np.asarray(found), wanted.numpy(), atol=1e-4, rtol=1e-3)
    (detections,) = detector.detect(x, 0.0, 0.4)
    assert len(detections.boxes) > 0
    assert detections.boxes.shape[1] == 4
