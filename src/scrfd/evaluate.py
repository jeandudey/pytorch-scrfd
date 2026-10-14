# SPDX-FileCopyrightText: 2018-2019 Open-MMLab
# SPDX-FileCopyrightText: 2021 Jia Guo, InsightFace contributors
# SPDX-FileCopyrightText: 2026 Jean-Pierre De Jesus DIAZ <me@jeandudey.tech>
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

"""Average precision of a detector on a RetinaFace-format validation set."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from .anchors import pairwise_iou
from .dataset import Record, letterbox, read_image
from .model import SCRFD

IOU_THRESHOLD = 0.5
SCORE_THRESHOLD = 0.02
NMS_THRESHOLD = 0.45


def match_image(
    boxes: torch.Tensor, scores: torch.Tensor, record: Record
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Flag the detections of one image, best score first, as true or false positives.

    A detection overlapping an ignored region without matching a face is dropped from both.
    """
    order = scores.argsort(descending=True)
    boxes = boxes[order]
    num_gt = len(record.boxes)
    gt = torch.from_numpy(np.concatenate([record.boxes, record.ignored]))
    tp = np.zeros(len(boxes), dtype=bool)
    fp = np.zeros(len(boxes), dtype=bool)
    if len(gt) == 0:
        fp[:] = True
        return tp, fp
    best_iou, best_gt = pairwise_iou(boxes, gt).max(dim=1)
    taken = np.zeros(len(gt), dtype=bool)
    for i, (iou, j) in enumerate(zip(best_iou.tolist(), best_gt.tolist(), strict=True)):
        if iou < IOU_THRESHOLD:
            fp[i] = True
        elif j >= num_gt:
            continue
        elif taken[j]:
            fp[i] = True
        else:
            taken[j] = True
            tp[i] = True
    return tp, fp


def average_precision(
    scores: NDArray[np.float32], tp: NDArray[np.bool_], fp: NDArray[np.bool_], num_gt: int
) -> float:
    """Area under the precision-recall curve."""
    order = np.argsort(-scores, kind="stable")
    tp_cum = np.cumsum(tp[order])
    fp_cum = np.cumsum(fp[order])
    recall = np.concatenate([[0.0], tp_cum / max(num_gt, 1), [1.0]])
    precision = np.concatenate([[0.0], tp_cum / np.maximum(tp_cum + fp_cum, 1e-12), [0.0]])
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    changed = np.nonzero(recall[1:] != recall[:-1])[0]
    return float(((recall[changed + 1] - recall[changed]) * precision[changed + 1]).sum())


@torch.no_grad()
def evaluate(
    model: SCRFD,
    records: Sequence[Record],
    images: Path,
    device: torch.device,
    size: int = 640,
) -> float:
    """AP at IoU 0.5 over ``records``, with images fitted into ``size`` x ``size``."""
    was_training = model.training
    model.eval()
    all_scores, all_tp, all_fp = [], [], []
    num_gt = 0
    for record in records:
        tensor, scale = letterbox(read_image(images / record.name), size)
        (detections,) = model.detect(tensor[None].to(device), SCORE_THRESHOLD, NMS_THRESHOLD)
        boxes, scores = detections.boxes.cpu() / scale, detections.scores.cpu()
        tp, fp = match_image(boxes, scores, record)
        all_scores.append(np.sort(scores.numpy())[::-1])
        all_tp.append(tp)
        all_fp.append(fp)
        num_gt += len(record.boxes)
    model.train(was_training)
    if not all_scores:
        return 0.0
    return average_precision(
        np.concatenate(all_scores), np.concatenate(all_tp), np.concatenate(all_fp), num_gt
    )
