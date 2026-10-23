# SPDX-FileCopyrightText: 2021 Jia Guo, InsightFace contributors
# SPDX-FileCopyrightText: 2026 Jean-Pierre De Jesus DIAZ <me@jeandudey.tech>
#
# SPDX-License-Identifier: MIT

"""Constructors for the released SCRFD variants."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal

from .layers import NormType
from .mobilenetv1 import MobileNetV1
from .model import SCRFD
from .resnet import ResNetV1e


@dataclass(frozen=True)
class Config:
    """Architecture description of one SCRFD variant."""

    backbone: Literal["mobilenet", "basic", "bottleneck"]
    stage_blocks: tuple[int, ...]
    stage_planes: tuple[int, ...]
    neck_channels: int
    head_channels: int
    stacked_convs: int
    norm: NormType
    gn_groups: int = 16
    strides_share: bool = True
    scale_mode: int = 2
    dw_conv: bool = False
    use_kps: bool = False
    strides: tuple[int, ...] = (8, 16, 32)
    anchor_sizes: tuple[int, ...] = (16, 64, 256)
    anchor_scales: tuple[float, ...] = (1, 2)
    anchor_ratios: tuple[float, ...] = (1.0,)

    @property
    def num_anchors(self) -> int:
        return len(self.anchor_scales) * len(self.anchor_ratios)

    def build(self) -> SCRFD:
        backbone: MobileNetV1 | ResNetV1e
        if self.backbone == "mobilenet":
            backbone = MobileNetV1(self.stage_planes, self.stage_blocks)
        else:
            block = "BasicBlock" if self.backbone == "basic" else "Bottleneck"
            backbone = ResNetV1e(
                block, self.stage_blocks, self.stage_planes, base_channels=self.stage_planes[0]
            )
        return SCRFD(
            backbone,
            self.neck_channels,
            self.head_channels,
            self.stacked_convs,
            strides=self.strides,
            num_anchors=self.num_anchors,
            norm=self.norm,
            gn_groups=self.gn_groups,
            strides_share=self.strides_share,
            scale_mode=self.scale_mode,
            dw_conv=self.dw_conv,
            use_kps=self.use_kps,
        )


SCRFD_500M = Config(
    backbone="mobilenet",
    stage_blocks=(2, 3, 2, 6),
    stage_planes=(16, 16, 40, 72, 152, 288),
    neck_channels=16,
    head_channels=64,
    stacked_convs=2,
    norm="gn",
    dw_conv=True,
)
SCRFD_1G = Config(
    backbone="mobilenet",
    stage_blocks=(3, 2, 1, 5),
    stage_planes=(32, 48, 48, 160, 216, 312),
    neck_channels=24,
    head_channels=96,
    stacked_convs=2,
    norm="gn",
    gn_groups=8,
    dw_conv=True,
)
SCRFD_1_3G = Config(
    backbone="basic",
    stage_blocks=(3, 5, 3, 2),
    stage_planes=(16, 32, 40, 72),
    neck_channels=24,
    head_channels=48,
    stacked_convs=2,
    norm="gn",
)
SCRFD_2_5G = Config(
    backbone="basic",
    stage_blocks=(3, 5, 3, 2),
    stage_planes=(24, 48, 48, 80),
    neck_channels=24,
    head_channels=64,
    stacked_convs=2,
    norm="gn",
)
SCRFD_10G = Config(
    backbone="basic",
    stage_blocks=(3, 4, 2, 3),
    stage_planes=(56, 88, 88, 224),
    neck_channels=56,
    head_channels=80,
    stacked_convs=3,
    norm="gn",
)
SCRFD_34G = Config(
    backbone="bottleneck",
    stage_blocks=(17, 16, 2, 8),
    stage_planes=(56, 56, 144, 184),
    neck_channels=128,
    head_channels=256,
    stacked_convs=2,
    norm="gn",
    gn_groups=32,
)
SCRFD_GEN2_5G = Config(
    backbone="basic",
    stage_blocks=(3, 4, 6, 3),
    stage_planes=(16, 32, 64, 128),
    neck_channels=48,
    head_channels=96,
    stacked_convs=2,
    norm="gn",
)

# BatchNorm heads use one tower per stride; the keypoint variants add a keypoint branch.
PER_STRIDE_BN = {"norm": "bn", "strides_share": False}

CONFIGS: dict[str, Config] = {
    "scrfd_500m": SCRFD_500M,
    # Not an upstream config: scrfd_500m_bnkps without keypoints, so that Open Images can train it.
    # No GroupNorm anywhere, so every normalisation folds into its convolution.
    "scrfd_500m_bn": replace(SCRFD_500M, **PER_STRIDE_BN, scale_mode=0),
    "scrfd_500m_bnkps": replace(SCRFD_500M, **PER_STRIDE_BN, scale_mode=0, use_kps=True),
    "scrfd_1g": SCRFD_1G,
    "scrfd_1gbn": replace(SCRFD_1G, **PER_STRIDE_BN),
    "scrfd_1.3g": SCRFD_1_3G,
    "scrfd_2.5g": SCRFD_2_5G,
    "scrfd_2.5g_bnkps": replace(SCRFD_2_5G, **PER_STRIDE_BN, use_kps=True),
    "scrfd_10g": SCRFD_10G,
    "scrfd_10g_bnkps": replace(SCRFD_10G, **PER_STRIDE_BN, use_kps=True),
    "scrfd_34g": SCRFD_34G,
    "scrfdgen2.5g": SCRFD_GEN2_5G,
    "scrfd_crowdhuman_2.5g_bnkps": replace(
        SCRFD_2_5G,
        **PER_STRIDE_BN,
        use_kps=True,
        strides=(8, 16, 32, 64, 128),
        anchor_sizes=(8, 16, 32, 64, 128),
        anchor_scales=(3,),
        anchor_ratios=(2.0,),
    ),
}


def build_model(name: str) -> SCRFD:
    """Build an untrained SCRFD variant by its upstream config name."""
    try:
        return CONFIGS[name].build()
    except KeyError:
        msg = f"unknown variant {name!r}; choose from {', '.join(CONFIGS)}"
        raise ValueError(msg) from None


def _factory(name: str) -> Callable[[], SCRFD]:
    def make() -> SCRFD:
        return build_model(name)

    make.__name__ = name.replace(".", "_")
    return make


scrfd_500m = _factory("scrfd_500m")
scrfd_500m_bn = _factory("scrfd_500m_bn")
scrfd_500m_bnkps = _factory("scrfd_500m_bnkps")
scrfd_1g = _factory("scrfd_1g")
scrfd_1gbn = _factory("scrfd_1gbn")
scrfd_1_3g = _factory("scrfd_1.3g")
scrfd_2_5g = _factory("scrfd_2.5g")
scrfd_2_5g_bnkps = _factory("scrfd_2.5g_bnkps")
scrfd_10g = _factory("scrfd_10g")
scrfd_10g_bnkps = _factory("scrfd_10g_bnkps")
scrfd_34g = _factory("scrfd_34g")
scrfdgen2_5g = _factory("scrfdgen2.5g")
