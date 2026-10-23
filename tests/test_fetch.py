# SPDX-FileCopyrightText: 2026 Jean-Pierre De Jesus DIAZ <me@jeandudey.tech>
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import cv2
import numpy as np

from scrfd.dataset import load_records
from scrfd.fetch import prepare, select
from scrfd.openimages import HUMAN_FACE

HEADER = "ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf"
METADATA_HEADER = (
    "ImageID,Subset,OriginalURL,OriginalLandingURL,License,AuthorProfileURL,Author,Title,"
    "OriginalSize,OriginalMD5,Thumbnail300KURL,Rotation"
)
BY = "https://creativecommons.org/licenses/by/2.0/"
NC = "https://creativecommons.org/licenses/by-nc/2.0/"
IDS = [f"img{i}" for i in range(6)]


def make_remote(root: Path) -> dict[str, tuple[str, str, str]]:
    """A fake Open Images server in ``root``; img4 is a group box and img5 is CC BY-NC."""
    (root / "train").mkdir(parents=True)
    rows = [f"{i},xclick,{HUMAN_FACE},1,0.1,0.5,0.2,0.6,0,0,{int(i == 'img4')}" for i in IDS]
    metadata = [
        f"{i},train,u,https://flickr.example/{i},{NC if i == 'img5' else BY},p,Author {i},T,1,m,t,0"
        for i in IDS
    ]
    (root / "boxes.csv").write_text("\n".join([HEADER, *rows]) + "\n")
    (root / "metadata.csv").write_text("\n".join([METADATA_HEADER, *metadata]) + "\n")
    for i in IDS:
        cv2.imwrite(str(root / "train" / f"{i}.jpg"), np.zeros((100, 200, 3), np.uint8))
    return {"train": ("train", (root / "boxes.csv").as_uri(), (root / "metadata.csv").as_uri())}


def test_select(tmp_path: Path) -> None:
    make_remote(tmp_path)
    boxes, metadata = tmp_path / "boxes.csv", tmp_path / "metadata.csv"
    assert select(boxes, metadata, None, 0) == ["img0", "img1", "img2", "img3"]
    subset = select(boxes, metadata, 2, 0)
    assert len(subset) == 2 and subset == select(boxes, metadata, 2, 0)
    assert subset == sorted(subset)


def test_prepare(tmp_path: Path) -> None:
    remote = tmp_path / "remote"
    urls = make_remote(remote)
    image_url = f"{remote.as_uri()}/{{bucket}}/{{image_id}}.jpg"
    output = tmp_path / "data"

    assert prepare(output, {"train": 3}, workers=2, urls=urls, image_url=image_url) == {
        "train": (3, 3)
    }
    records = load_records(output / "train" / "labelv2.txt")
    assert len(records) == 3
    assert len(list((output / "raw" / "train").glob("*.jpg"))) == 3
    credits = (output / "train" / "ATTRIBUTION.csv").read_text().splitlines()
    assert len(credits) == 4

    # a second run neither downloads nor changes anything
    for path in (remote / "train").glob("*.jpg"):
        path.unlink()
    assert prepare(output, {"train": 3}, workers=2, urls=urls, image_url=image_url) == {
        "train": (3, 3)
    }
