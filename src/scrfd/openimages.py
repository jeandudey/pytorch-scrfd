# SPDX-FileCopyrightText: 2026 Jean-Pierre De Jesus DIAZ <me@jeandudey.tech>
#
# SPDX-License-Identifier: MIT

"""``scrfd-convert-openimages``: turn Open Images face boxes into a RetinaFace-format split.

Open Images has no face landmarks, so the result only trains the variants without keypoints
(``scrfd_500m``, ``scrfd_1g``, ``scrfd_2.5g``, ...).
"""

import argparse
import csv
import importlib.util
from collections import defaultdict
from collections.abc import Collection, Sequence
from pathlib import Path

HUMAN_FACE = "/m/0dzct"
MIN_BOX_SIDE = 1.0
ATTRIBUTION_FIELDS = (
    "ImageID",
    "Title",
    "Author",
    "AuthorProfileURL",
    "OriginalLandingURL",
    "License",
)
CC_BY = "creativecommons.org/licenses/by/"


def read_boxes(
    path: Path, label: str = HUMAN_FACE
) -> tuple[dict[str, list[tuple[float, float, float, float]]], set[str]]:
    """Normalised ``(xmin, ymin, xmax, ymax)`` boxes of ``label`` per image, and group images.

    Images with a ``IsGroupOf`` box are returned in the second set: such a box stands for several
    faces, so the faces inside it would be trained on as background.
    """
    boxes: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    groups: set[str] = set()
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            if row["LabelName"] != label:
                continue
            if row["IsGroupOf"] == "1":
                groups.add(row["ImageID"])
                continue
            boxes[row["ImageID"]].append(
                (float(row["XMin"]), float(row["YMin"]), float(row["XMax"]), float(row["YMax"]))
            )
    return boxes, groups


def read_credits(path: Path, image_ids: set[str]) -> dict[str, dict[str, str]]:
    """Credit rows of the images in ``image_ids`` that are licensed under CC BY, by image ID.

    ``path`` is the Open Images image metadata CSV. Images under another license are left out,
    as their terms differ from the ones credited in the BOM.
    """
    credits = {}
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            if row["ImageID"] in image_ids and CC_BY in row["License"]:
                credits[row["ImageID"]] = {field: row[field] for field in ATTRIBUTION_FIELDS}
    return credits


def convert(
    boxes_csv: Path,
    images: Path,
    output: Path,
    split: str,
    label: str = HUMAN_FACE,
    metadata_csv: Path | None = None,
    only: Collection[str] | None = None,
) -> tuple[int, int]:
    """Write ``output/<split>/labelv2.txt`` and link ``output/<split>/images`` to ``images``.

    Returns the numbers of images and faces written. Images that are not in ``images`` are skipped,
    so a subset of Open Images can be downloaded. With ``metadata_csv``, ``ATTRIBUTION.csv`` is
    written next to the labels with the credit of every image kept, and the images that are not
    under CC BY are skipped. ``only`` restricts the conversion to these image IDs.
    """
    import cv2

    boxes, groups = read_boxes(boxes_csv, label)
    candidates = boxes.keys() - groups
    if only is not None:
        candidates &= set(only)
    credits = None
    if metadata_csv is not None:
        credits = read_credits(metadata_csv, candidates)
    folder = output / split
    folder.mkdir(parents=True, exist_ok=True)
    link = folder / "images"
    if not link.exists():
        link.symlink_to(images.resolve(), target_is_directory=True)

    lines: list[str] = []
    credited: list[dict[str, str]] = []
    num_images = num_faces = 0
    for image_id in sorted(candidates):
        if credits is not None and image_id not in credits:
            continue
        name = f"{image_id}.jpg"
        image = cv2.imread(str(images / name), cv2.IMREAD_COLOR)
        if image is None:
            continue
        height, width = image.shape[:2]
        faces = []
        for xmin, ymin, xmax, ymax in boxes[image_id]:
            x1, y1, x2, y2 = xmin * width, ymin * height, xmax * width, ymax * height
            if x2 - x1 >= MIN_BOX_SIDE and y2 - y1 >= MIN_BOX_SIDE:
                faces.append(f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}")
        if not faces:
            continue
        lines.append(f"# {name} {width} {height}")
        lines.extend(faces)
        if credits is not None:
            credited.append(credits[image_id])
        num_images += 1
        num_faces += len(faces)
    (folder / "labelv2.txt").write_text("\n".join(lines) + "\n")
    if credits is not None:
        with (folder / "ATTRIBUTION.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, ATTRIBUTION_FIELDS)
            writer.writeheader()
            writer.writerows(credited)
    return num_images, num_faces


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="scrfd-convert-openimages", description=__doc__)
    parser.add_argument("boxes", type=Path, help="Open Images bounding boxes CSV of the split")
    parser.add_argument("images", type=Path, help="directory with the <ImageID>.jpg files")
    parser.add_argument("--output", type=Path, default=Path("data/openimages"))
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Open Images image metadata CSV, to write ATTRIBUTION.csv (needed by scrfd-bom)",
    )
    parser.add_argument("--label", default=HUMAN_FACE, help="LabelName of faces (Human face)")
    args = parser.parse_args(argv)
    if importlib.util.find_spec("cv2") is None:
        msg = "scrfd-convert-openimages needs the 'train' extra: pip install 'scrfd[train]'\n"
        raise SystemExit(msg)
    num_images, num_faces = convert(
        args.boxes, args.images, args.output, args.split, args.label, args.metadata
    )
    print(f"{args.output / args.split}: {num_images} images, {num_faces} faces")


if __name__ == "__main__":
    main()
