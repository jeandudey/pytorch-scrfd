# SPDX-FileCopyrightText: 2026 Jean-Pierre De Jesus DIAZ <me@jeandudey.tech>
#
# SPDX-License-Identifier: MIT

"""Snapshot loading."""

from os import PathLike
from typing import Any

import torch
from torch import nn


def extract_state_dict(snapshot: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Return the weights of a snapshot that may wrap them under ``state_dict``."""
    state = snapshot.get("state_dict", snapshot)
    return {k.removeprefix("module."): v for k, v in state.items()}


def load_snapshot(
    model: nn.Module, path: str | PathLike[str], map_location: str = "cpu"
) -> nn.Module:
    """Load a ``.pth`` snapshot into ``model``, requiring an exact key match."""
    snapshot = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(extract_state_dict(snapshot), strict=True)
    return model
