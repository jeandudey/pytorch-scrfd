# SPDX-FileCopyrightText: 2026 Jean-Pierre De Jesus DIAZ <me@jeandudey.tech>
#
# SPDX-License-Identifier: MIT

"""``scrfd-quantize``: post-training int8 quantization of an SCRFD variant to ONNX.

The model is exported at a fixed input size, then quantized (QDQ format, per-channel int8
weights) with activation ranges calibrated on images you provide. Use training images for
calibration, never the images you evaluate on. BatchNorm folds into the convolutions on
export, so the variants with BatchNorm heads (``scrfd_500m_bn``, ...) quantize best; score the
result with ``scrfd-eval-widerface --onnx``.
"""

import argparse
import importlib.util
import random
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from .dataset import letterbox, read_image
from .export import export_onnx
from .snapshot import load_snapshot
from .variants import CONFIGS, build_model

EXTENSIONS = {".jpg", ".jpeg", ".png"}
METHODS = ("minmax", "percentile")
PERCENTILE = 99.999


def calibration_images(folder: Path, count: int, seed: int) -> list[Path]:
    """``count`` random images of ``folder``, the same ones for the same ``seed``."""
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTENSIONS)
    if not images:
        msg = f"no images in {folder}"
        raise ValueError(msg)
    random.Random(seed).shuffle(images)
    return images[:count]


def quantize(
    fp32: Path,
    output: Path,
    images: Sequence[Path],
    size: int,
    method: str = "percentile",
) -> None:
    """Quantize the fixed-size ONNX model ``fp32`` into ``output`` to int8."""
    from onnxruntime.quantization import (
        CalibrationDataReader,
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    class Calibration(CalibrationDataReader):
        def __init__(self) -> None:
            self.paths = iter(images)

        @override
        # ONNX Runtime ends the calibration when this returns None, which its signature omits.
        def get_next(self) -> dict[str, NDArray[np.float32]] | None:  # ty: ignore[invalid-method-override]
            path = next(self.paths, None)
            if path is None:
                return None
            tensor, _ = letterbox(read_image(path), size)
            return {"image": tensor[None].numpy()}

    quantize_static(
        fp32,
        output,
        Calibration(),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=(
            CalibrationMethod.Percentile if method == "percentile" else CalibrationMethod.MinMax
        ),
        extra_options={"CalibPercentile": PERCENTILE} if method == "percentile" else {},
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="scrfd-quantize", description=__doc__)
    parser.add_argument("variant", choices=sorted(CONFIGS), help="model variant")
    parser.add_argument("--snapshot", type=Path, required=True, help="path of a .pth snapshot")
    parser.add_argument(
        "--calibration", type=Path, required=True, help="folder of calibration images"
    )
    parser.add_argument("--output", type=Path, help="int8 model (default: <variant>-int8.onnx)")
    parser.add_argument("--num-images", type=int, default=256, help="calibration images")
    parser.add_argument("--method", choices=METHODS, default="percentile")
    parser.add_argument("--size", type=int, default=640, help="fixed network input size")
    parser.add_argument("--seed", type=int, default=0, help="seed of the calibration images")
    parser.add_argument("--keep-fp32", type=Path, help="also write the float ONNX model here")
    args = parser.parse_args(argv)
    missing = [m for m in ("onnx", "onnxruntime", "cv2") if importlib.util.find_spec(m) is None]
    if missing:
        msg = "scrfd-quantize needs the 'quantize' extra: pip install 'scrfd[quantize]'\n"
        raise SystemExit(msg)

    images = calibration_images(args.calibration, args.num_images, args.seed)
    model = build_model(args.variant)
    load_snapshot(model, args.snapshot)
    output = args.output or Path(f"{args.variant}-int8.onnx")
    with tempfile.TemporaryDirectory() as folder:
        fp32 = args.keep_fp32 or Path(folder) / "fp32.onnx"
        export_onnx(model, fp32, (args.size, args.size))
        quantize(fp32, output, images, args.size, args.method)
        print(
            f"wrote {output}: {output.stat().st_size / 1e6:.2f} MB int8, "
            f"{fp32.stat().st_size / 1e6:.2f} MB float, calibrated on {len(images)} images "
            f"({args.method})"
        )


if __name__ == "__main__":
    main()
