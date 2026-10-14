# SPDX-FileCopyrightText: 2018-2019 Open-MMLab
# SPDX-FileCopyrightText: 2021 Jia Guo, InsightFace contributors
# SPDX-FileCopyrightText: 2026 Jean-Pierre De Jesus DIAZ <me@jeandudey.tech>
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

"""ResNetV1e backbone."""

from collections.abc import Sequence
from typing import Literal

from torch import Tensor, nn
from typing_extensions import override


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    @override
    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class Bottleneck(nn.Module):
    """Bottleneck block, PyTorch style (stride on the 3x3 convolution)."""

    expansion = 4

    def __init__(
        self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    @override
    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


def _make_res_layer(
    block: type[BasicBlock] | type[Bottleneck],
    inplanes: int,
    planes: int,
    num_blocks: int,
    stride: int,
) -> nn.Sequential:
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        # avg_down: pool first, then a stride-1 1x1 conv.
        downsample = nn.Sequential(
            nn.AvgPool2d(stride, stride, ceil_mode=True, count_include_pad=False),
            nn.Conv2d(inplanes, planes * block.expansion, 1, 1, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )
    layers: list[nn.Module] = [block(inplanes, planes, stride, downsample)]
    layers.extend(block(planes * block.expansion, planes) for _ in range(1, num_blocks))
    return nn.Sequential(*layers)


class ResNetV1e(nn.Module):
    """ResNetV1e: deep 3x3 stem, 2x2 max pool and average-pool downsampling.

    ``stage_planes`` are the per-stage widths before block expansion.
    """

    def __init__(
        self,
        block: Literal["BasicBlock", "Bottleneck"],
        stage_blocks: Sequence[int],
        stage_planes: Sequence[int],
        base_channels: int,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        block_cls = BasicBlock if block == "BasicBlock" else Bottleneck
        c = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c // 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(c // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // 2, c // 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // 2, c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(2, 2)
        strides = (1, 2, 2, 2)
        inplanes = c
        self.out_channels = tuple(p * block_cls.expansion for p in stage_planes)
        self.stage_names: list[str] = []
        for i, (n, planes) in enumerate(zip(stage_blocks, stage_planes, strict=True)):
            name = f"layer{i + 1}"
            self.add_module(name, _make_res_layer(block_cls, inplanes, planes, n, strides[i]))
            inplanes = planes * block_cls.expansion
            self.stage_names.append(name)

    @override
    def forward(self, x: Tensor) -> list[Tensor]:
        out = self.maxpool(self.stem(x))
        outs = []
        for name in self.stage_names:
            out = getattr(self, name)(out)
            outs.append(out)
        return outs
