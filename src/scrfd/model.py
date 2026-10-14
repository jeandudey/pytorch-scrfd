# SPDX-FileCopyrightText: 2018-2019 Open-MMLab
# SPDX-FileCopyrightText: 2021 Jia Guo, InsightFace contributors
# SPDX-FileCopyrightText: 2026 Jean-Pierre De Jesus DIAZ <me@jeandudey.tech>
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

"""SCRFD detector: backbone, PAFPN neck and head."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
from torchvision.ops import nms
from typing_extensions import override

from .layers import ConvModule, DepthwiseSeparableConvModule, NormType, Scale
from .mobilenetv1 import MobileNetV1
from .pafpn import PAFPN
from .resnet import ResNetV1e

NUM_KPS = 5


class Integral(nn.Module):
    """Holds the fixed distribution support buffer ``project``.

    Unused at inference (the DFL branch is disabled in every variant); it exists
    so state dicts that carry ``integral.project`` load strictly.
    """

    def __init__(self, reg_max: int = 8) -> None:
        super().__init__()
        self.register_buffer("project", torch.linspace(0, reg_max, reg_max + 1))


class Head(nn.Module):
    """Shared-conv (or per-stride) head predicting scores, boxes and keypoints.

    Forward returns three lists (one entry per stride): sigmoid scores
    ``(B, N, 1)``, box distances ``(B, N, 4)`` and keypoint offsets
    ``(B, N, 10)``. Distances are in units of the stride.
    """

    def __init__(
        self,
        in_channels: int,
        feat_channels: int,
        stacked_convs: int,
        strides: Sequence[int] = (8, 16, 32),
        num_anchors: int = 2,
        norm: NormType = "bn",
        gn_groups: int = 32,
        strides_share: bool = True,
        scale_mode: int = 2,
        dw_conv: bool = False,
        use_kps: bool = False,
        reg_max: int = 8,
    ) -> None:
        super().__init__()
        self.strides = list(strides)
        self.num_anchors = num_anchors
        self.strides_share = strides_share
        self.use_kps = use_kps
        self.use_scale = scale_mode > 0 and (strides_share or scale_mode == 2)

        def make_conv(cin: int, cout: int) -> nn.Module:
            if dw_conv:
                return DepthwiseSeparableConvModule(
                    cin, cout, 3, padding=1, norm=norm, num_groups=gn_groups
                )
            return ConvModule(cin, cout, 3, padding=1, norm=norm, num_groups=gn_groups)

        # Keys mirror the upstream layout: '0' when shared, else str((s, s)).
        keys = ["0"] if strides_share else [str((s, s)) for s in self.strides]
        self.cls_stride_convs = nn.ModuleDict()
        self.reg_stride_convs = nn.ModuleDict()  # empty: cls and reg towers are shared
        self.stride_cls = nn.ModuleDict()
        self.stride_reg = nn.ModuleDict()
        if use_kps:
            self.stride_kps = nn.ModuleDict()
        for key in keys:
            self.cls_stride_convs[key] = nn.ModuleList(
                make_conv(in_channels if i == 0 else feat_channels, feat_channels)
                for i in range(stacked_convs)
            )
            self.reg_stride_convs[key] = nn.ModuleList()
            self.stride_cls[key] = nn.Conv2d(feat_channels, num_anchors, 3, padding=1)
            self.stride_reg[key] = nn.Conv2d(feat_channels, 4 * num_anchors, 3, padding=1)
            if use_kps:
                self.stride_kps[key] = nn.Conv2d(
                    feat_channels, NUM_KPS * 2 * num_anchors, 3, padding=1
                )
        if self.use_scale:
            self.scales = nn.ModuleList(Scale(1.0) for _ in self.strides)
        self.integral = Integral(reg_max)

    def _key(self, stride: int) -> str:
        return "0" if self.strides_share else str((stride, stride))

    def forward_single(self, x: Tensor, level: int) -> tuple[Tensor, Tensor, Tensor]:
        key = self._key(self.strides[level])
        feat = x
        convs = cast(nn.ModuleList, self.cls_stride_convs[key])
        for conv in convs:
            feat = conv(feat)
        cls_score = self.stride_cls[key](feat)
        bbox_pred = self.stride_reg[key](feat)
        if self.use_scale:
            bbox_pred = self.scales[level](bbox_pred)
        if self.use_kps:
            kps_pred = self.stride_kps[key](feat)
        else:
            kps_pred = bbox_pred.new_zeros(
                bbox_pred.shape[0], NUM_KPS * 2, bbox_pred.shape[2], bbox_pred.shape[3]
            )
        batch = cls_score.shape[0]
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(batch, -1, 1).sigmoid()
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(batch, -1, 4)
        kps_pred = kps_pred.permute(0, 2, 3, 1).reshape(batch, -1, NUM_KPS * 2)
        return cls_score, bbox_pred, kps_pred

    @override
    def forward(self, feats: Sequence[Tensor]) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        outs = [self.forward_single(f, i) for i, f in enumerate(feats)]
        scores, bboxes, kps = (list(t) for t in zip(*outs, strict=True))
        return scores, bboxes, kps


@dataclass
class Detections:
    """Detections of a single image, in network-input pixel coordinates."""

    boxes: Tensor  # (N, 4) x1, y1, x2, y2
    scores: Tensor  # (N,)
    keypoints: Tensor | None  # (N, 5, 2) or None


def distance2bbox(points: Tensor, distance: Tensor) -> Tensor:
    return torch.stack(
        [
            points[:, 0] - distance[:, 0],
            points[:, 1] - distance[:, 1],
            points[:, 0] + distance[:, 2],
            points[:, 1] + distance[:, 3],
        ],
        dim=-1,
    )


def distance2kps(points: Tensor, distance: Tensor) -> Tensor:
    offsets = distance.reshape(distance.shape[0], -1, 2)
    return points[:, None, :] + offsets


def anchor_centers(
    height: int, width: int, stride: int, num_anchors: int, device: torch.device
) -> Tensor:
    """Anchor centres in (row, column, anchor) order, matching the head layout."""
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij"
    )
    centers = torch.stack([xs, ys], dim=-1).reshape(-1, 2).float() * stride
    return centers.repeat_interleave(num_anchors, dim=0)


class SCRFD(nn.Module):
    """Sample and Computation Redistribution face detector.

    ``forward`` takes a normalised RGB batch ``(B, 3, H, W)`` (``(x - 127.5) / 128``)
    and returns, per stride, the sigmoid scores, then the box distances, then the
    keypoint offsets (only when the model predicts keypoints). Box and keypoint
    values are in units of the stride.
    """

    def __init__(
        self,
        backbone: MobileNetV1 | ResNetV1e,
        neck_channels: int,
        head_channels: int,
        stacked_convs: int,
        strides: Sequence[int] = (8, 16, 32),
        num_anchors: int = 2,
        norm: NormType = "bn",
        gn_groups: int = 32,
        strides_share: bool = True,
        scale_mode: int = 2,
        dw_conv: bool = False,
        use_kps: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = PAFPN(backbone.out_channels, neck_channels, num_outs=len(strides))
        self.bbox_head = Head(
            neck_channels,
            head_channels,
            stacked_convs,
            strides=strides,
            num_anchors=num_anchors,
            norm=norm,
            gn_groups=gn_groups,
            strides_share=strides_share,
            scale_mode=scale_mode,
            dw_conv=dw_conv,
            use_kps=use_kps,
        )

    @property
    def strides(self) -> list[int]:
        return self.bbox_head.strides

    @property
    def use_kps(self) -> bool:
        return self.bbox_head.use_kps

    @override
    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        scores, bboxes, kps = self.bbox_head(self.neck(self.backbone(x)))
        if self.use_kps:
            return (*scores, *bboxes, *kps)
        return (*scores, *bboxes)

    @torch.no_grad()
    def detect(self, x: Tensor, score_thr: float = 0.5, nms_thr: float = 0.4) -> list[Detections]:
        """Run the network and decode detections for each image in the batch."""
        return self.postprocess(self(x), (x.shape[2], x.shape[3]), score_thr, nms_thr)

    def postprocess(
        self,
        outputs: tuple[Tensor, ...],
        image_size: tuple[int, int],
        score_thr: float = 0.5,
        nms_thr: float = 0.4,
    ) -> list[Detections]:
        levels = len(self.strides)
        height, width = image_size
        batch = outputs[0].shape[0]
        per_image: list[list[tuple[Tensor, Tensor, Tensor | None]]] = [[] for _ in range(batch)]
        for level, stride in enumerate(self.strides):
            centers = anchor_centers(
                height // stride + (height % stride > 0),
                width // stride + (width % stride > 0),
                stride,
                self.bbox_head.num_anchors,
                outputs[0].device,
            )
            for b in range(batch):
                scores = outputs[level][b, :, 0]
                keep = scores >= score_thr
                pts = centers[keep]
                boxes = distance2bbox(pts, outputs[levels + level][b][keep] * stride)
                kps = (
                    distance2kps(pts, outputs[2 * levels + level][b][keep] * stride)
                    if self.use_kps
                    else None
                )
                per_image[b].append((boxes, scores[keep], kps))
        results = []
        for parts in per_image:
            boxes = torch.cat([p[0] for p in parts])
            scores = torch.cat([p[1] for p in parts])
            kps_parts = [p[2] for p in parts if p[2] is not None]
            kps = torch.cat(kps_parts) if kps_parts else None
            keep = nms(boxes, scores, nms_thr)
            results.append(
                Detections(boxes[keep], scores[keep], kps[keep] if kps is not None else None)
            )
        return results
