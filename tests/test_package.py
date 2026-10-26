# SPDX-FileCopyrightText: 2026 Jean-Pierre De Jesus DIAZ <me@jeandudey.tech>
#
# SPDX-License-Identifier: MIT

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pytest
import torch
from test_bom import RECIPE, write_split

from scrfd.package import main
from scrfd.snapshot import load_snapshot
from scrfd.variants import build_model

VARIANT = "scrfd_500m_bn"
SIZE = 160


@pytest.fixture
def trained(tmp_path: Path) -> Path:
    write_split(tmp_path, "train", ["a", "b"])
    write_split(tmp_path, "val", ["c"])
    work = tmp_path / "work"
    work.mkdir()
    # A snapshot of scrfd-train: the weights plus the optimizer state.
    snapshot = {"state_dict": build_model(VARIANT).state_dict(), "epoch": 3, "optimizer": {}}
    torch.save(snapshot, work / "epoch_3.pth")
    (work / "recipe.json").write_text(json.dumps({**RECIPE, "variant": VARIANT}))
    return tmp_path


def package(trained: Path, *extra: str) -> Path:
    output = trained / "release"
    main(
        [
            VARIANT,
            str(trained / "work" / "epoch_3.pth"),
            "--output",
            str(output),
            "--author",
            "Jane Doe",
            "--data-root",
            str(trained / "data"),
            "--size",
            str(SIZE),
            *extra,
        ]
    )
    return output


def test_package(trained: Path) -> None:
    calibration = trained / "calibration"
    calibration.mkdir()
    rng = np.random.default_rng(0)
    for i in range(3):
        cv2.imwrite(str(calibration / f"{i}.jpg"), rng.integers(0, 255, (120, 200, 3), np.uint8))
    output = package(trained, "--calibration", str(calibration), "--num-images", "3")

    names = {p.name for p in output.iterdir()}
    assert names >= {
        f"{VARIANT}.pth",
        f"{VARIANT}-fp32.onnx",
        f"{VARIANT}-int8.onnx",
        "bom.cdx.json",
        "bom.spdx3.json",
        "ATTRIBUTION.train.csv",
        "ATTRIBUTION.val.csv",
        "SHA256SUMS",
    }
    weights = torch.load(output / f"{VARIANT}.pth", weights_only=True)
    assert "optimizer" not in weights
    load_snapshot(build_model(VARIANT), output / f"{VARIANT}.pth")
    for name in (f"{VARIANT}-fp32.onnx", f"{VARIANT}-int8.onnx"):
        session = ort.InferenceSession(str(output / name), providers=["CPUExecutionProvider"])
        assert session.get_inputs()[0].shape[2:] == [SIZE, SIZE]

    sums = dict(
        line.split("  ", 1)[::-1] for line in (output / "SHA256SUMS").read_text().splitlines()
    )
    assert set(sums) == names - {"SHA256SUMS"}
    for name, digest in sums.items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    bom = json.loads((output / "bom.cdx.json").read_text())
    assert bom["components"][0]["hashes"][0]["content"] == sums[f"{VARIANT}.pth"]


def test_without_calibration_there_is_no_int8_model(trained: Path) -> None:
    output = package(trained)
    assert not (output / f"{VARIANT}-int8.onnx").exists()
    assert (output / f"{VARIANT}-fp32.onnx").exists()
