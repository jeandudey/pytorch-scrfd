# SPDX-FileCopyrightText: 2026 Jean-Pierre De Jesus DIAZ <me@jeandudey.tech>
#
# SPDX-License-Identifier: MIT

"""``scrfd-package``: gather a trained model, its ONNX exports and its provenance for release.

Writes into one folder the weights alone (no optimizer state), the float ONNX model, optionally
the int8 one, the CycloneDX and SPDX BOMs with the image credits (``scrfd-bom``) and a
``SHA256SUMS`` file. Add a model card and the license text of the weights before uploading.
"""

import argparse
import importlib.util
from collections.abc import Sequence
from pathlib import Path

import torch

from . import bom
from .export import export_onnx
from .quantize import calibration_images, quantize
from .snapshot import extract_state_dict, load_snapshot
from .variants import CONFIGS, build_model

CHECKSUMS = "SHA256SUMS"


def write_checksums(folder: Path) -> list[Path]:
    """Write ``SHA256SUMS`` for the files of ``folder`` (``sha256sum -c`` format)."""
    files = sorted(p for p in folder.iterdir() if p.is_file() and p.name != CHECKSUMS)
    lines = [f"{bom.sha256(p)}  {p.name}" for p in files]
    (folder / CHECKSUMS).write_text("\n".join(lines) + "\n")
    return files


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="scrfd-package", description=__doc__)
    parser.add_argument("variant", choices=sorted(CONFIGS), help="model variant")
    parser.add_argument("snapshot", type=Path, help="the trained model, e.g. epoch_10.pth")
    parser.add_argument("--output", type=Path, required=True, help="folder to write")
    parser.add_argument("--author", required=True, help="who trained and publishes the model")
    parser.add_argument("--data-root", type=Path, default=Path("data/openimages"))
    parser.add_argument("--recipe", type=Path, help="recipe.json (default: next to the snapshot)")
    parser.add_argument("--ap", type=float, help="AP@0.5 on the validation split, to record")
    parser.add_argument("--license", help="SPDX license expression of the model weights")
    parser.add_argument(
        "--calibration", type=Path, help="folder of training images; also writes the int8 model"
    )
    parser.add_argument("--num-images", type=int, default=256, help="calibration images")
    parser.add_argument("--size", type=int, default=640, help="fixed ONNX input size")
    args = parser.parse_args(argv)
    if args.calibration and importlib.util.find_spec("onnxruntime") is None:
        msg = "the int8 model needs the 'quantize' extra: pip install 'scrfd[quantize]'\n"
        raise SystemExit(msg)

    args.output.mkdir(parents=True, exist_ok=True)
    model = build_model(args.variant)
    load_snapshot(model, args.snapshot)
    weights = args.output / f"{args.variant}.pth"
    torch.save(extract_state_dict(model.state_dict()), weights)
    fp32 = args.output / f"{args.variant}-fp32.onnx"
    export_onnx(model, fp32, (args.size, args.size))
    if args.calibration:
        images = calibration_images(args.calibration, args.num_images, 0)
        quantize(fp32, args.output / f"{args.variant}-int8.onnx", images, args.size)

    bom_args = [args.variant, str(weights), "--author", args.author]
    bom_args += ["--data-root", str(args.data_root), "--output", str(args.output)]
    bom_args += ["--recipe", str(args.recipe or args.snapshot.parent / "recipe.json")]
    if args.ap is not None:
        bom_args += ["--ap", str(args.ap)]
    if args.license:
        bom_args += ["--license", args.license]
    bom.main(bom_args)
    for path in write_checksums(args.output):
        print(path)


if __name__ == "__main__":
    main()
