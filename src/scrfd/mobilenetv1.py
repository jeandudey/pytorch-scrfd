# SPDX-FileCopyrightText: 2021 Jia Guo, InsightFace contributors
# SPDX-FileCopyrightText: 2026 Jean-Pierre De Jesus DIAZ <me@jeandudey.tech>
#
# SPDX-License-Identifier: MIT

"""MobileNetV1 backbone."""

from collections.abc import Sequence

from torch import Tensor, nn
from typing_extensions import override


class ConvBN(nn.Sequential):
    """3x3 convolution, batch norm and ReLU."""

    def __init__(self, inp: int, oup: int, stride: int) -> None:
        super().__init__(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )


class ConvDW(nn.Sequential):
    """Depthwise 3x3 then pointwise 1x1 convolution, each with batch norm and ReLU."""

    def __init__(self, inp: int, oup: int, stride: int) -> None:
        super().__init__(
            nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )


class MobileNetV1(nn.Module):
    """MobileNetV1 with configurable stage widths and depths.

    ``stage_planes`` has 6 entries (two for the stem, one per stage) and
    ``stage_blocks`` has 4. Outputs are the four stage feature maps (strides 4-32).
    """

    def __init__(self, stage_planes: Sequence[int], stage_blocks: Sequence[int]) -> None:
        super().__init__()
        if len(stage_planes) != 6 or len(stage_blocks) != 4:
            msg = "stage_planes needs 6 entries and stage_blocks needs 4"
            raise ValueError(msg)
        self.stem = nn.Sequential(
            ConvBN(3, stage_planes[0], 2),
            ConvDW(stage_planes[0], stage_planes[1], 1),
        )
        self.out_channels = tuple(stage_planes[2:])
        self.stage_names: list[str] = []
        for i, num_blocks in enumerate(stage_blocks):
            layers = [
                ConvDW(stage_planes[i + 1], stage_planes[i + 2], 2)
                if n == 0
                else ConvDW(stage_planes[i + 2], stage_planes[i + 2], 1)
                for n in range(num_blocks)
            ]
            name = f"layer{i + 1}"
            self.add_module(name, nn.Sequential(*layers))
            self.stage_names.append(name)

    @override
    def forward(self, x: Tensor) -> list[Tensor]:
        out = self.stem(x)
        outs = []
        for name in self.stage_names:
            out = getattr(self, name)(out)
            outs.append(out)
        return outs
