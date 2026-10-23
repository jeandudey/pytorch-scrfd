# SPDX-FileCopyrightText: 2026 Jean-Pierre De Jesus DIAZ <me@jeandudey.tech>
#
# SPDX-License-Identifier: MIT

"""``scrfd-fetch-openimages``: download Open Images faces and prepare them for ``scrfd-train``.

Downloads the annotation and credit CSVs, picks the CC BY images with faces (a random subset
with ``--max-train`` / ``--max-val``), downloads them and converts everything with
``scrfd-convert-openimages``. Every step is skipped when its result already exists, so an
interrupted run continues where it stopped.
"""

import argparse
import importlib.util
import random
import shutil
import urllib.request
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .openimages import convert, read_boxes, read_credits

STORAGE = "https://storage.googleapis.com/openimages"
IMAGE_URL = "https://open-images-dataset.s3.amazonaws.com/{bucket}/{image_id}.jpg"
# split -> (image bucket, boxes CSV, image metadata CSV)
SPLITS = {
    "train": (
        "train",
        f"{STORAGE}/v6/oidv6-train-annotations-bbox.csv",
        f"{STORAGE}/v6/oidv6-train-images-with-labels-with-rotation.csv",
    ),
    "val": (
        "validation",
        f"{STORAGE}/v5/validation-annotations-bbox.csv",
        f"{STORAGE}/2018_04/validation/validation-images-with-rotation.csv",
    ),
}
ATTEMPTS = 3
TIMEOUT = 60


def download(url: str, path: Path) -> bool:
    """Download ``url`` to ``path`` unless it is there; ``False`` if it could not be."""
    if path.exists():
        return True
    temporary = path.with_name(path.name + ".part")
    for _ in range(ATTEMPTS):
        try:
            with (
                urllib.request.urlopen(url, timeout=TIMEOUT) as response,
                temporary.open("wb") as f,
            ):
                shutil.copyfileobj(response, f, 1 << 20)
        except OSError:
            continue
        temporary.replace(path)
        return True
    temporary.unlink(missing_ok=True)
    return False


def select(boxes_csv: Path, metadata_csv: Path, limit: int | None, seed: int) -> list[str]:
    """IDs of the CC BY images with faces and no group box, at most ``limit`` random ones."""
    boxes, groups = read_boxes(boxes_csv)
    ids = sorted(read_credits(metadata_csv, boxes.keys() - groups))
    if limit is not None and limit < len(ids):
        ids = sorted(random.Random(seed).sample(ids, limit))
    return ids


def fetch_images(
    ids: Sequence[str], bucket: str, folder: Path, workers: int, image_url: str = IMAGE_URL
) -> int:
    """Download the images that are missing from ``folder``; returns how many failed."""
    folder.mkdir(parents=True, exist_ok=True)

    def one(image_id: str) -> bool:
        url = image_url.format(bucket=bucket, image_id=image_id)
        return download(url, folder / f"{image_id}.jpg")

    with ThreadPoolExecutor(workers) as pool:
        results = []
        for done, ok in enumerate(pool.map(one, ids), 1):
            results.append(ok)
            if done % 1000 == 0 or done == len(ids):
                print(f"  {bucket}: {done}/{len(ids)} images", flush=True)
    return results.count(False)


def prepare(
    output: Path,
    limits: dict[str, int | None],
    seed: int = 0,
    workers: int = 16,
    urls: dict[str, tuple[str, str, str]] = SPLITS,
    image_url: str = IMAGE_URL,
) -> dict[str, tuple[int, int]]:
    """Fetch and convert the splits of ``limits`` into ``output``; images and faces per split."""
    cache = output / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, limit in limits.items():
        bucket, boxes_url, metadata_url = urls[split]
        boxes_csv, metadata_csv = cache / f"{split}-boxes.csv", cache / f"{split}-metadata.csv"
        for url, path in ((boxes_url, boxes_csv), (metadata_url, metadata_csv)):
            print(f"{split}: {path.name}", flush=True)
            if not download(url, path):
                msg = f"cannot download {url}"
                raise SystemExit(msg)
        ids = select(boxes_csv, metadata_csv, limit, seed)
        print(f"{split}: {len(ids)} CC BY images with faces selected", flush=True)
        failed = fetch_images(ids, bucket, output / "raw" / split, workers, image_url)
        if failed:
            print(f"{split}: {failed} images could not be downloaded, rerun to retry")
        counts[split] = convert(
            boxes_csv, output / "raw" / split, output, split, metadata_csv=metadata_csv, only=ids
        )
    return counts


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="scrfd-fetch-openimages", description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/openimages"))
    parser.add_argument("--max-train", type=int, help="random subset of this many train images")
    parser.add_argument("--max-val", type=int, help="random subset of this many val images")
    parser.add_argument("--seed", type=int, default=0, help="seed of the random subsets")
    parser.add_argument("--workers", type=int, default=16, help="parallel downloads")
    args = parser.parse_args(argv)
    if importlib.util.find_spec("cv2") is None:
        msg = "scrfd-fetch-openimages needs the 'train' extra: pip install 'scrfd[train]'\n"
        raise SystemExit(msg)
    counts = prepare(
        args.output, {"train": args.max_train, "val": args.max_val}, args.seed, args.workers
    )
    for split, (images, faces) in counts.items():
        print(f"{args.output / split}: {images} images, {faces} faces")
    print(f"next: scrfd-train <variant> --data-root {args.output}")


if __name__ == "__main__":
    main()
