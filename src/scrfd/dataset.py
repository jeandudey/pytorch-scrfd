# SPDX-FileCopyrightText: 2018-2019 Open-MMLab
# SPDX-FileCopyrightText: 2021 Jia Guo, InsightFace contributors
# SPDX-FileCopyrightText: 2026 Jean-Pierre De Jesus DIAZ <me@jeandudey.tech>
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

"""RetinaFace-format dataset (WIDER FACE and the like) with the SCRFD training augmentations.

Annotation files hold, per image, a ``# <name> <width> <height>`` line followed by one line per
object: ``x1 y1 x2 y2`` optionally followed by five ``x y visible`` keypoints (``-1 -1 -1`` for a
missing one) or by a single ignore flag.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset
from typing_extensions import override

NUM_KPS = 5
CROP_CHOICES = (0.3, 0.45, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0)
# Left and right eye, and left and right mouth corner, swap places when mirroring.
FLIP_ORDER = (1, 0, 2, 4, 3)
MIN_IMAGE_SIDE = 32
PAD_VALUE = 128.0

Array = NDArray[Any]


@dataclass
class Record:
    """Annotations of one image; ``ignored`` boxes are regions to leave out of evaluation."""

    name: str
    width: int
    height: int
    boxes: Array  # (G, 4) x1, y1, x2, y2
    keypoints: Array  # (G, 5, 3) x, y, visible
    ignored: Array  # (I, 4)


def parse_object(values: Sequence[float]) -> tuple[Array, Array, bool]:
    """Parse one object line into its box, keypoints and whether it is ignored."""
    box = np.array(values[:4], dtype=np.float32)
    keypoints = np.zeros((NUM_KPS, 3), dtype=np.float32)
    ignore = False
    if len(values) > 5:
        keypoints = np.array(values[4 : 4 + NUM_KPS * 3], dtype=np.float32).reshape(NUM_KPS, 3)
        missing = (keypoints == -1).all(axis=1)
        keypoints[:, 2] = np.where(missing, 0.0, 1.0)
    elif len(values) == 5:
        ignore = values[4] == 1
    return box, keypoints, ignore


def load_records(path: Path) -> list[Record]:
    """Read an annotation file."""
    entries: list[tuple[str, int, int, list[list[float]]]] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            name, width, height = line[1:].split()[:3]
            entries.append((name, int(width), int(height), []))
        elif entries:
            entries[-1][3].append([float(v) for v in line.split()])
        else:
            msg = f"{path}: object before the first image header"
            raise ValueError(msg)
    records = []
    for name, width, height, objects in entries:
        boxes, keypoints, ignored = [], [], []
        for values in objects:
            box, kps, ignore = parse_object(values)
            if ignore:
                ignored.append(box)
            else:
                boxes.append(box)
                keypoints.append(kps)
        records.append(
            Record(
                name,
                width,
                height,
                np.array(boxes, dtype=np.float32).reshape(-1, 4),
                np.array(keypoints, dtype=np.float32).reshape(-1, NUM_KPS, 3),
                np.array(ignored, dtype=np.float32).reshape(-1, 4),
            )
        )
    return records


def read_image(path: Path) -> Array:
    """Read an image as a float32 BGR array."""
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        msg = f"cannot read image {path}"
        raise FileNotFoundError(msg)
    return image.astype(np.float32)


def random_square_crop(
    image: Array,
    boxes: Array,
    keypoints: Array,
    rng: np.random.Generator,
    choices: Sequence[float] = CROP_CHOICES,
    clip_border: bool = True,
    attempts: int = 250,
) -> tuple[Array, Array, Array]:
    """Crop a random square whose side is a multiple of the short side of the image.

    Squares larger than the image are padded with gray. The square has to contain the centre of
    at least one box; the boxes whose centre is outside are dropped.
    """
    height, width = image.shape[:2]
    for _ in range(attempts):
        side = int(rng.choice(choices) * min(width, height))
        if side < 1:
            continue
        for _ in range(attempts):
            left = int(rng.integers(min(width - side, 0), max(width - side, 0) + 1))
            top = int(rng.integers(min(height - side, 0), max(height - side, 0) + 1))
            right, bottom = left + side, top + side
            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            keep = (
                (centers[:, 0] > left)
                & (centers[:, 1] > top)
                & (centers[:, 0] < right)
                & (centers[:, 1] < bottom)
            )
            if keep.any():
                break
        else:
            continue

        origin = np.array([left, top], dtype=np.float32)
        low, high = origin, np.array([right, bottom], dtype=np.float32)
        new_boxes = boxes[keep].copy()
        new_keypoints = keypoints[keep].copy()
        if clip_border:
            new_boxes[:, :2] = np.maximum(new_boxes[:, :2], low)
            new_boxes[:, 2:] = np.minimum(new_boxes[:, 2:], high)
            new_keypoints[..., :2] = np.clip(new_keypoints[..., :2], low, high)
        new_boxes -= np.tile(origin, 2)
        new_keypoints[..., :2] -= origin

        patch = np.full((side, side, 3), PAD_VALUE, dtype=image.dtype)
        src_x1, src_y1 = max(left, 0), max(top, 0)
        src_x2, src_y2 = min(right, width), min(bottom, height)
        dst_x1, dst_y1 = src_x1 - left, src_y1 - top
        patch[dst_y1 : dst_y1 + src_y2 - src_y1, dst_x1 : dst_x1 + src_x2 - src_x1] = image[
            src_y1:src_y2, src_x1:src_x2
        ]
        return patch, new_boxes, new_keypoints
    return image, boxes, keypoints


def resize(
    image: Array, boxes: Array, keypoints: Array, size: int, clip_border: bool = True
) -> tuple[Array, Array, Array]:
    """Resize to ``size`` x ``size`` without keeping the aspect ratio."""
    import cv2

    height, width = image.shape[:2]
    scale_x, scale_y = size / width, size / height
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    boxes = boxes * np.array([scale_x, scale_y, scale_x, scale_y], dtype=np.float32)
    keypoints = keypoints.copy()
    keypoints[..., 0] *= scale_x
    keypoints[..., 1] *= scale_y
    if clip_border:
        boxes = np.clip(boxes, 0, size)
        keypoints[..., :2] = np.clip(keypoints[..., :2], 0, size)
    return image, boxes, keypoints


def horizontal_flip(image: Array, boxes: Array, keypoints: Array) -> tuple[Array, Array, Array]:
    width = image.shape[1]
    flipped_boxes = boxes.copy()
    flipped_boxes[:, 0] = width - boxes[:, 2]
    flipped_boxes[:, 2] = width - boxes[:, 0]
    flipped_keypoints = keypoints[:, FLIP_ORDER].copy()
    flipped_keypoints[..., 0] = width - flipped_keypoints[..., 0]
    return np.ascontiguousarray(image[:, ::-1]), flipped_boxes, flipped_keypoints


def photometric_distortion(
    image: Array,
    rng: np.random.Generator,
    brightness_delta: float = 32,
    contrast_range: tuple[float, float] = (0.5, 1.5),
    saturation_range: tuple[float, float] = (0.5, 1.5),
    hue_delta: float = 18,
) -> Array:
    """Random brightness, contrast, saturation, hue and channel order, each half of the time."""
    import cv2

    image = image.copy()
    if rng.integers(2):
        image += rng.uniform(-brightness_delta, brightness_delta)
    contrast_first = bool(rng.integers(2))  # contrast goes before or after the colour changes
    if contrast_first and rng.integers(2):
        image = image * rng.uniform(*contrast_range)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if rng.integers(2):
        hsv[..., 1] = hsv[..., 1] * rng.uniform(*saturation_range)
    if rng.integers(2):
        hsv[..., 0] = (hsv[..., 0] + rng.uniform(-hue_delta, hue_delta)) % 360
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if not contrast_first and rng.integers(2):
        image = image * rng.uniform(*contrast_range)
    if rng.integers(2):
        image = image[..., rng.permutation(3)]
    return image


def to_tensor(image: Array) -> Tensor:
    """BGR image to a normalised RGB ``(3, H, W)`` tensor, ``(x - 127.5) / 128``."""
    rgb = np.ascontiguousarray(image[..., ::-1].transpose(2, 0, 1))
    return (torch.from_numpy(rgb) - 127.5) / 128.0


def letterbox(image: Array, size: int) -> tuple[Tensor, float]:
    """Fit an image into the top-left of a ``size`` x ``size`` canvas, keeping the aspect ratio."""
    import cv2

    height, width = image.shape[:2]
    scale = size / max(height, width)
    resized = cv2.resize(image, (round(width * scale), round(height * scale)))
    canvas = np.full((size, size, 3), 127.5, dtype=np.float32)
    canvas[: resized.shape[0], : resized.shape[1]] = resized
    return to_tensor(canvas), scale


@dataclass
class Sample:
    image: Tensor  # (3, S, S)
    boxes: Tensor  # (G, 4)
    keypoints: Tensor  # (G, 5, 3)


class RetinaFaceDataset(Dataset[Sample]):
    """Training samples: random square crop, resize, flip and photometric distortion.

    Images without a usable box or smaller than ``MIN_IMAGE_SIDE`` are skipped. Set ``epoch``
    each epoch so that every epoch draws different augmentations.
    """

    def __init__(
        self,
        annotations: Path,
        images: Path,
        size: int = 640,
        clip_border: bool = True,
        seed: int = 0,
    ) -> None:
        self.images = images
        self.size = size
        self.clip_border = clip_border
        self.seed = seed
        self.epoch = 0
        self.records = [
            r
            for r in load_records(annotations)
            if len(r.boxes) > 0 and min(r.width, r.height) >= MIN_IMAGE_SIDE
        ]

    def __len__(self) -> int:
        return len(self.records)

    @override
    def __getitem__(self, index: int) -> Sample:
        record = self.records[index]
        rng = np.random.default_rng([self.seed, self.epoch, index])
        image = read_image(self.images / record.name)
        boxes, keypoints = record.boxes, record.keypoints
        image, boxes, keypoints = random_square_crop(
            image, boxes, keypoints, rng, clip_border=self.clip_border
        )
        image, boxes, keypoints = resize(image, boxes, keypoints, self.size, self.clip_border)
        if rng.random() < 0.5:
            image, boxes, keypoints = horizontal_flip(image, boxes, keypoints)
        image = photometric_distortion(image, rng)
        return Sample(to_tensor(image), torch.from_numpy(boxes), torch.from_numpy(keypoints))


def collate(samples: Sequence[Sample]) -> tuple[Tensor, list[Tensor], list[Tensor]]:
    return (
        torch.stack([s.image for s in samples]),
        [s.boxes for s in samples],
        [s.keypoints for s in samples],
    )
