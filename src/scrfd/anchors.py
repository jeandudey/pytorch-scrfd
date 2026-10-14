# SPDX-FileCopyrightText: 2018-2019 Open-MMLab
# SPDX-FileCopyrightText: 2021 Jia Guo, InsightFace contributors
# SPDX-FileCopyrightText: 2026 Jean-Pierre De Jesus DIAZ <me@jeandudey.tech>
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

"""Anchor generation and box utilities for training."""

import math
from collections.abc import Sequence

import torch
from torch import Tensor


def base_anchors(
    size: float, scales: Sequence[float], ratios: Sequence[float], device: torch.device
) -> Tensor:
    """Anchors ``(len(ratios) * len(scales), 4)`` centred on the origin, ratio-major."""
    scale = torch.tensor(scales, dtype=torch.float, device=device)
    height_ratio = torch.tensor([math.sqrt(r) for r in ratios], dtype=torch.float, device=device)
    widths = (size / height_ratio[:, None] * scale[None, :]).reshape(-1)
    heights = (size * height_ratio[:, None] * scale[None, :]).reshape(-1)
    return torch.stack([-0.5 * widths, -0.5 * heights, 0.5 * widths, 0.5 * heights], dim=-1)


def grid_anchors(
    height: int,
    width: int,
    stride: int,
    size: float,
    scales: Sequence[float],
    ratios: Sequence[float],
    device: torch.device,
) -> Tensor:
    """All anchors of one level in (row, column, anchor) order, centred on ``(x, y) * stride``."""
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij"
    )
    shifts = torch.stack([xs, ys, xs, ys], dim=-1).reshape(-1, 1, 4).float() * stride
    return (base_anchors(size, scales, ratios, device)[None] + shifts).reshape(-1, 4)


def pairwise_iou(a: Tensor, b: Tensor, eps: float = 1e-6) -> Tensor:
    """IoU matrix ``(len(a), len(b))``."""
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None] - inter).clamp(min=eps)


def aligned_iou(a: Tensor, b: Tensor, eps: float = 1e-6) -> Tensor:
    """IoU of ``a[i]`` with ``b[i]``."""
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    wh = (torch.min(a[:, 2:], b[:, 2:]) - torch.max(a[:, :2], b[:, :2])).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    return inter / (area_a + area_b - inter).clamp(min=eps)
