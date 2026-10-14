# SPDX-FileCopyrightText: 2018-2019 Open-MMLab
# SPDX-FileCopyrightText: 2026 Jean-Pierre De Jesus DIAZ <me@jeandudey.tech>
#
# SPDX-License-Identifier: Apache-2.0

"""Basic building blocks."""

from typing import Literal

import torch
from torch import Tensor, nn
from typing_extensions import override

NormType = Literal["bn", "gn"] | None


class ConvModule(nn.Module):
    """Conv2d followed by an optional norm layer and an optional ReLU.

    The norm layer is stored under the attribute ``bn`` or ``gn`` and the
    convolution under ``conv``. The convolution has a bias only without a norm.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        norm: NormType = None,
        num_groups: int = 32,
        act: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=norm is None,
        )
        self.norm_name = norm
        if norm == "bn":
            self.bn = nn.BatchNorm2d(out_channels)
        elif norm == "gn":
            self.gn = nn.GroupNorm(num_groups, out_channels)
        self.activate = nn.ReLU(inplace=False) if act else None

    @override
    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        if self.norm_name is not None:
            out = getattr(self, self.norm_name)(out)
        if self.activate is not None:
            out = self.activate(out)
        return out


class DepthwiseSeparableConvModule(nn.Module):
    """Depthwise 3x3 ConvModule followed by a pointwise 1x1 ConvModule."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        norm: NormType = None,
        num_groups: int = 32,
    ) -> None:
        super().__init__()
        self.depthwise_conv = ConvModule(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            norm=norm,
            num_groups=num_groups,
        )
        self.pointwise_conv = ConvModule(
            in_channels, out_channels, 1, norm=norm, num_groups=num_groups
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise_conv(self.depthwise_conv(x))


class Scale(nn.Module):
    """A learnable scalar multiplier."""

    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    @override
    def forward(self, x: Tensor) -> Tensor:
        return x * self.scale
