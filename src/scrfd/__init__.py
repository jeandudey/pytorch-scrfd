# SPDX-FileCopyrightText: 2026 Jean-Pierre De Jesus DIAZ <me@jeandudey.tech>
#
# SPDX-License-Identifier: MIT

"""SCRFD face detector in plain PyTorch."""

from .model import SCRFD, Detections
from .snapshot import load_snapshot
from .variants import CONFIGS, build_model

__all__ = ["CONFIGS", "SCRFD", "Detections", "build_model", "load_snapshot"]
