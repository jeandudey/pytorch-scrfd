# SPDX-FileCopyrightText: 2026 Jean-Pierre De Jesus DIAZ <me@jeandudey.tech>
#
# SPDX-License-Identifier: MIT

"""``scrfd-export``: export an SCRFD variant to ONNX."""

import argparse
import importlib.util
from collections.abc import Sequence
from pathlib import Path

import torch
from torch.export import Dim

from .model import SCRFD
from .snapshot import load_snapshot
from .variants import CONFIGS, build_model


def output_names(model: SCRFD) -> list[str]:
    strides = model.strides
    names = [f"score_{s}" for s in strides] + [f"bbox_{s}" for s in strides]
    if model.use_kps:
        names += [f"kps_{s}" for s in strides]
    return names


def export_onnx(
    model: SCRFD,
    output: Path,
    shape: tuple[int, int] | None = None,
    opset: int = 18,
) -> None:
    """Export ``model``; with ``shape=None`` the height, width and batch are dynamic."""
    model.eval()
    height, width = shape or (640, 640)
    dummy = torch.zeros(1 if shape else 2, 3, height, width)
    dynamic_shapes = None
    if shape is None:
        dynamic_shapes = ({0: Dim("batch"), 2: Dim("height", min=32), 3: Dim("width", min=32)},)
    torch.onnx.export(
        model,
        (dummy,),
        output,
        input_names=["image"],
        output_names=output_names(model),
        dynamic_shapes=dynamic_shapes,
        opset_version=opset,
        dynamo=True,
        external_data=False,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="scrfd-export", description=__doc__)
    parser.add_argument("variant", choices=sorted(CONFIGS), help="model variant")
    parser.add_argument("--snapshot", type=Path, help="path of a .pth snapshot to load")
    parser.add_argument("--output", type=Path, help="output file (default: <variant>.onnx)")
    parser.add_argument(
        "--shape", type=int, nargs=2, metavar=("H", "W"), help="fixed input size (default: dynamic)"
    )
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args(argv)
    if importlib.util.find_spec("onnx") is None:
        parser.exit(1, "scrfd-export needs the 'export' extra: pip install 'scrfd[export]'\n")

    model = build_model(args.variant)
    if args.snapshot is not None:
        load_snapshot(model, args.snapshot)
    output = args.output or Path(f"{args.variant}.onnx")
    export_onnx(model, output, tuple(args.shape) if args.shape else None, args.opset)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
