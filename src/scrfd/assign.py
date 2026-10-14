# SPDX-FileCopyrightText: 2018-2019 Open-MMLab
# SPDX-FileCopyrightText: 2021 Jia Guo, InsightFace contributors
# SPDX-FileCopyrightText: 2026 Jean-Pierre De Jesus DIAZ <me@jeandudey.tech>
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

"""Adaptive training sample selection (ATSS) of anchors to ground-truth boxes."""

from collections.abc import Sequence

import torch
from torch import Tensor

from .anchors import pairwise_iou


def atss_assign(
    anchors: Tensor, level_counts: Sequence[int], gt_boxes: Tensor, topk: int = 9
) -> Tensor:
    """Assign each anchor to a ground-truth box or to the background.

    On every level the ``topk`` anchors closest to a box centre are candidates; those whose IoU
    reaches mean + std of the candidates' IoUs and whose centre lies inside the box are positive.
    An anchor claimed by several boxes goes to the one with the highest IoU.

    Returns a ``(len(anchors),)`` tensor holding the matched box index, or -1 for background.
    """
    num_anchors, num_gt = anchors.shape[0], gt_boxes.shape[0]
    assigned = anchors.new_full((num_anchors,), -1, dtype=torch.long)
    if num_gt == 0:
        return assigned

    overlaps = pairwise_iou(anchors, gt_boxes)
    gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2
    centers = (anchors[:, :2] + anchors[:, 2:]) / 2
    distances = torch.cdist(centers, gt_centers)

    candidates = []
    start = 0
    for count in level_counts:
        k = min(topk, count)
        candidates.append(distances[start : start + count].topk(k, dim=0, largest=False).indices)
        candidates[-1] += start
        start += count
    candidates_idx = torch.cat(candidates)  # (K, G)

    candidate_overlaps = overlaps.gather(0, candidates_idx)
    threshold = candidate_overlaps.mean(0) + candidate_overlaps.std(0)
    is_pos = candidate_overlaps >= threshold[None]

    # the candidate's centre has to be inside the box, relative to the box size
    cand_centers = centers[candidates_idx]  # (K, G, 2)
    inside = torch.stack(
        [
            cand_centers[..., 0] - gt_boxes[:, 0],
            cand_centers[..., 1] - gt_boxes[:, 1],
            gt_boxes[:, 2] - cand_centers[..., 0],
            gt_boxes[:, 3] - cand_centers[..., 1],
        ]
    ).amin(0)
    sizes = ((gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])).clamp(min=1e-4)
    is_pos &= inside / sizes.sqrt() > 0.001

    gt_idx = torch.arange(num_gt, device=anchors.device).expand_as(candidates_idx)[is_pos]
    anchor_idx = candidates_idx[is_pos]
    best = overlaps.new_full((num_anchors, num_gt), float("-inf"))
    best[anchor_idx, gt_idx] = overlaps[anchor_idx, gt_idx]
    best_overlap, best_gt = best.max(dim=1)
    return torch.where(best_overlap > float("-inf"), best_gt, assigned)
