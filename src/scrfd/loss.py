# SPDX-FileCopyrightText: 2018-2019 Open-MMLab
# SPDX-FileCopyrightText: 2021 Jia Guo, InsightFace contributors
# SPDX-FileCopyrightText: 2026 Jean-Pierre De Jesus DIAZ <me@jeandudey.tech>
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

"""SCRFD training objective: quality focal loss, DIoU loss and keypoint smooth-L1 loss."""

import math
from collections.abc import Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from typing_extensions import override

from .anchors import aligned_iou, grid_anchors
from .assign import atss_assign
from .model import distance2bbox
from .variants import Config

ATSS_TOPK = 9
CLS_WEIGHT = 1.0
BBOX_WEIGHT = 2.0
KPS_WEIGHT = 0.1
KPS_BETA = 1.0 / 9.0
QFL_BETA = 2.0


def reduce_mean(value: Tensor) -> Tensor:
    """Average ``value`` over all processes."""
    if dist.is_available() and dist.is_initialized():
        value = value.clone() / dist.get_world_size()
        dist.all_reduce(value)
    return value


def diou_loss(pred: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:
    """Per-box ``1 - DIoU`` between ``(n, 4)`` corner boxes."""
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]
    area_pred = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    area_target = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    iou = overlap / (area_pred + area_target - overlap + eps)

    enclose = (torch.max(pred[:, 2:], target[:, 2:]) - torch.min(pred[:, :2], target[:, :2])).clamp(
        min=0
    )
    diagonal = enclose[:, 0] ** 2 + enclose[:, 1] ** 2 + eps
    center_pred = (pred[:, :2] + pred[:, 2:]) / 2
    center_target = (target[:, :2] + target[:, 2:]) / 2
    center_distance = ((center_pred - center_target) ** 2).sum(-1)
    return 1 - (iou - center_distance / diagonal)


def smooth_l1_loss(pred: Tensor, target: Tensor, beta: float) -> Tensor:
    diff = (pred - target).abs()
    return torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)


def quality_focal_loss(logits: Tensor, is_pos: Tensor, quality: Tensor, beta: float) -> Tensor:
    """Summed QFL; negatives regress to 0, positives to ``quality`` (their predicted-box IoU).

    ``logits`` and ``is_pos`` have shape ``(B, N)``; ``quality`` holds the positives' targets.
    """
    prob = logits.sigmoid()
    negative = F.binary_cross_entropy_with_logits(
        logits, torch.zeros_like(logits), reduction="none"
    )
    loss = negative * prob.pow(beta)
    loss = loss.sum() - loss[is_pos].sum()
    pos_logits = logits[is_pos]
    pos_loss = F.binary_cross_entropy_with_logits(pos_logits, quality, reduction="none")
    return loss + (pos_loss * (quality - pos_logits.sigmoid()).abs().pow(beta)).sum()


class SCRFDLoss(nn.Module):
    """Assigns anchors with ATSS and computes the loss of a batch of ``SCRFD.predict`` outputs.

    Boxes and keypoints are predicted as distances in units of the stride from the anchor
    centre. The box and keypoint losses are weighted by the predicted score of each positive
    anchor and normalised by the sum of those weights.
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.use_kps = config.use_kps

    def anchors(
        self, image_size: tuple[int, int], device: torch.device
    ) -> tuple[Tensor, Tensor, list[int]]:
        """Anchors of every level, their strides and the per-level anchor counts."""
        config = self.config
        height, width = image_size
        anchors, strides, counts = [], [], []
        for stride, size in zip(config.strides, config.anchor_sizes, strict=True):
            level = grid_anchors(
                math.ceil(height / stride),
                math.ceil(width / stride),
                stride,
                size,
                config.anchor_scales,
                config.anchor_ratios,
                device,
            )
            anchors.append(level)
            strides.append(level.new_full((level.shape[0],), stride))
            counts.append(level.shape[0])
        return torch.cat(anchors), torch.cat(strides), counts

    @override
    def forward(
        self,
        outputs: tuple[Sequence[Tensor], Sequence[Tensor], Sequence[Tensor]],
        image_size: tuple[int, int],
        gt_boxes: Sequence[Tensor],
        gt_keypoints: Sequence[Tensor],
    ) -> dict[str, Tensor]:
        """Losses for ``outputs = (logits, bboxes, kps)`` of ``SCRFD.predict``.

        ``gt_boxes[i]`` is ``(G, 4)`` and ``gt_keypoints[i]`` is ``(G, 5, 3)`` (x, y, visible).
        """
        logits = torch.cat(list(outputs[0]), dim=1)[..., 0]  # (B, N)
        bboxes = torch.cat(list(outputs[1]), dim=1)  # (B, N, 4)
        keypoints = torch.cat(list(outputs[2]), dim=1)  # (B, N, 10)
        device = logits.device
        anchors, strides, counts = self.anchors(image_size, device)
        centers = (anchors[:, :2] + anchors[:, 2:]) / 2

        with torch.no_grad():
            assigned = torch.stack(
                [atss_assign(anchors, counts, boxes, ATSS_TOPK) for boxes in gt_boxes]
            )  # (B, N), -1 for background
        is_pos = assigned >= 0
        num_pos = sum(max(int(n), 1) for n in is_pos.sum(1).tolist())
        num_samples = reduce_mean(torch.tensor(float(num_pos), device=device)).clamp(min=1.0)

        batch_idx, anchor_idx = is_pos.nonzero(as_tuple=True)
        gt_idx = assigned[batch_idx, anchor_idx]
        flat_boxes = torch.cat(list(gt_boxes))
        first = torch.tensor([0, *(len(boxes) for boxes in gt_boxes)], device=device).cumsum(0)
        flat_idx = first[batch_idx] + gt_idx
        stride = strides[anchor_idx]
        center = centers[anchor_idx] / stride[:, None]

        pos_logits = logits[is_pos]
        weights = pos_logits.detach().sigmoid()
        target_boxes = flat_boxes[flat_idx]
        target_boxes = target_boxes / stride[:, None]
        decoded = distance2bbox(center, bboxes[is_pos])
        quality = aligned_iou(decoded.detach(), target_boxes)

        loss_cls = CLS_WEIGHT * quality_focal_loss(logits, is_pos, quality, QFL_BETA) / num_samples
        avg_factor = reduce_mean(weights.sum()).clamp(min=1e-12)
        loss_bbox = BBOX_WEIGHT * (diou_loss(decoded, target_boxes) * weights).sum() / avg_factor
        losses = {"loss_cls": loss_cls, "loss_bbox": loss_bbox}

        if self.use_kps:
            target_kps = torch.cat(list(gt_keypoints))[flat_idx]
            visible = target_kps[..., 2].mean(1)
            offsets = (target_kps[..., :2] / stride[:, None, None] - center[:, None]).flatten(1)
            kps_loss = smooth_l1_loss(keypoints[is_pos], offsets, KPS_BETA)
            losses["loss_kps"] = (
                KPS_WEIGHT * (kps_loss * (visible * weights)[:, None]).sum() / avg_factor
            )
        return losses
