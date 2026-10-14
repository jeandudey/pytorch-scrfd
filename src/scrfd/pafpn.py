# SPDX-FileCopyrightText: 2018-2019 Open-MMLab
# SPDX-FileCopyrightText: 2026 Jean-Pierre De Jesus DIAZ <me@jeandudey.tech>
#
# SPDX-License-Identifier: Apache-2.0

"""Path aggregation feature pyramid network."""

from collections.abc import Sequence

import torch.nn.functional as F
from torch import Tensor, nn
from typing_extensions import override

from .layers import ConvModule


class PAFPN(nn.Module):
    """FPN with an extra bottom-up path.

    The first backbone level is skipped (``start_level=1``). With
    ``num_outs`` above the number of used levels, extra stride-2 convolutions
    are stacked on the last output.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        num_outs: int = 3,
        start_level: int = 1,
    ) -> None:
        super().__init__()
        self.in_channels = list(in_channels)
        self.num_outs = num_outs
        self.start_level = start_level
        used = len(self.in_channels) - start_level
        self.used_levels = used

        def conv(stride: int = 1, k: int = 3, cin: int = out_channels) -> ConvModule:
            return ConvModule(cin, out_channels, k, stride=stride, padding=k // 2, act=False)

        self.lateral_convs = nn.ModuleList(conv(1, 1, c) for c in self.in_channels[start_level:])
        self.fpn_convs = nn.ModuleList(conv() for _ in range(used))
        self.fpn_convs.extend(conv(2) for _ in range(num_outs - used))
        self.downsample_convs = nn.ModuleList(conv(2) for _ in range(used - 1))
        self.pafpn_convs = nn.ModuleList(conv() for _ in range(used - 1))

    @override
    def forward(self, inputs: Sequence[Tensor]) -> list[Tensor]:
        laterals = [
            lateral(inputs[i + self.start_level]) for i, lateral in enumerate(self.lateral_convs)
        ]
        for i in range(self.used_levels - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:], mode="nearest"
            )
        inter = [self.fpn_convs[i](laterals[i]) for i in range(self.used_levels)]
        for i in range(self.used_levels - 1):
            inter[i + 1] = inter[i + 1] + self.downsample_convs[i](inter[i])
        outs = [inter[0]]
        outs.extend(self.pafpn_convs[i - 1](inter[i]) for i in range(1, self.used_levels))
        for i in range(self.used_levels, self.num_outs):
            outs.append(self.fpn_convs[i](outs[-1]))
        return outs
