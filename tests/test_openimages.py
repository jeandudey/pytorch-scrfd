# SPDX-FileCopyrightText: 2026 Jean-Pierre De Jesus DIAZ <me@jeandudey.tech>
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import cv2
import numpy as np

from scrfd.dataset import load_records
from scrfd.openimages import HUMAN_FACE, convert

HEADER = "ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf"
ROWS = [
    f"a,xclick,{HUMAN_FACE},1,0.1,0.5,0.2,0.6,0,0,0",
    "a,xclick,/m/other,1,0.0,1.0,0.0,1.0,0,0,0",
    f"b,xclick,{HUMAN_FACE},1,0.1,0.5,0.2,0.6,0,0,0",
    f"b,xclick,{HUMAN_FACE},1,0.5,0.9,0.5,0.9,0,0,1",
    f"c,xclick,{HUMAN_FACE},1,0.1,0.5,0.2,0.6,0,0,0",
    f"d,xclick,{HUMAN_FACE},1,0.1,0.1001,0.2,0.2001,0,0,0",
]


def test_convert(tmp_path: Path) -> None:
    images = tmp_path / "raw"
    images.mkdir()
    for name in ("a", "b", "d"):  # "c" is not downloaded
        cv2.imwrite(str(images / f"{name}.jpg"), np.zeros((100, 200, 3), np.uint8))
    boxes = tmp_path / "boxes.csv"
    boxes.write_text("\n".join([HEADER, *ROWS]) + "\n")

    assert convert(boxes, images, tmp_path / "out", "train") == (1, 1)

    records = load_records(tmp_path / "out" / "train" / "labelv2.txt")
    assert [r.name for r in records] == ["a.jpg"]
    assert (records[0].width, records[0].height) == (200, 100)
    assert records[0].boxes.tolist() == [[20.0, 20.0, 100.0, 60.0]]
    assert (tmp_path / "out" / "train" / "images" / "a.jpg").exists()
