<!--
SPDX-FileCopyrightText: 2026 Jean-Pierre De Jesus DIAZ <me@jeandudey.tech>

SPDX-License-Identifier: MIT
-->

# scrfd

[SCRFD](https://arxiv.org/abs/2105.04714) (Sample and Computation Redistribution
for Efficient Face Detection) implemented in plain PyTorch, without mmdet or mmcv.
The architecture follows the
[InsightFace](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)
reference implementation; state-dict keys are identical, so `.pth` snapshots
produced by it load strictly.

## Installation

Pick the PyTorch build matching your hardware with an extra:

```console
$ uv sync --extra cuda   # NVIDIA (CUDA 13.0)
$ uv sync --extra rocm   # AMD (ROCm 7.2)
$ uv sync --extra cpu    # CPU only
```

The `cpu`, `cuda` and `rocm` extras select the PyTorch wheel index through uv;
with other installers, install the PyTorch build you want yourself.

## Usage

```python
import torch
from scrfd import build_model, load_snapshot

model = load_snapshot(build_model("scrfd_10g_bnkps"), "snapshot.pth").eval()
# RGB image batch normalised as (x - 127.5) / 128, sides multiples of 32
detections = model.detect(torch.randn(1, 3, 640, 640), score_thr=0.5, nms_thr=0.4)
```

Variants: `scrfd_500m`, `scrfd_500m_bn` (batch norm only, no keypoints: it folds fully for
quantization), `scrfd_500m_bnkps`, `scrfd_1g`, `scrfd_1gbn`, `scrfd_1.3g`,
`scrfd_2.5g`, `scrfd_2.5g_bnkps`, `scrfd_10g`, `scrfd_10g_bnkps`, `scrfd_34g`,
`scrfdgen2.5g`, `scrfd_crowdhuman_2.5g_bnkps` (5 strides, 1 anchor). Each is also available as a function in `scrfd.variants`.

## ONNX export

```console
$ uv run --extra export scrfd-export scrfd_2.5g_bnkps --snapshot snapshot.pth --output model.onnx
$ uv run --extra export scrfd-export scrfd_500m --shape 640 640   # fixed input size
```

Without `--shape` the batch, height and width are dynamic. The outputs are
`score_*`, `bbox_*` and (for keypoint models) `kps_*`, one per stride, as in the
upstream exporter. Box and keypoint distances are in units of the stride.

## Int8 quantization

```console
$ uv run --extra quantize scrfd-quantize scrfd_500m_bn --snapshot snapshot.pth \
    --calibration data/openimages/train/images --output model-int8.onnx
$ uv run --extra eval --extra quantize scrfd-eval-widerface scrfd_500m_bn \
    --onnx model-int8.onnx --images WIDER_val/images --gt eval_tools/ground_truth
```

Post-training static quantization (QDQ, per-channel int8) with activation ranges calibrated on
256 random images of `--calibration`: use training images, never the ones you evaluate on.
BatchNorm folds into the convolutions on export, so the variants with BatchNorm heads
(`scrfd_500m_bn`, `scrfd_500m_bnkps`, ...) quantize with almost no loss; GroupNorm does not
fold. `--onnx` scores a float or quantized ONNX model on the CPU. On `scrfd_500m_bn` trained on
20k Open Images images, int8 takes the model from 2.62 to 0.94 MB and WIDER FACE val AP from
81.3 / 73.9 / 38.8 to 81.1 / 73.6 / 38.5 (Easy / Medium / Hard).

## Training

`scrfd-train` trains any variant from scratch with the reference recipe: SGD with
momentum, step learning-rate decay and linear warm-up, ATSS anchor assignment, quality
focal loss, DIoU loss and (for keypoint variants) a smooth-L1 keypoint loss, on random
square crops with flip and photometric augmentation. It needs the `train` extra.

```console
$ uv run --extra rocm --extra train scrfd-train scrfd_2.5g_bnkps --data-root data/retinaface
```

The dataset is [WIDER FACE](http://shuoyang1213.me/WIDERFACE/) with RetinaFace-style
annotations (CrowdHuman, converted the same way, for `scrfd_crowdhuman_2.5g_bnkps`):

```
data/retinaface/
    train/
        images/
        labelv2.txt
    val/
        images/
        labelv2.txt
```

Each annotation file lists, per image, a `# <name> <width> <height>` line followed by one
`x1 y1 x2 y2 [x y visible]*5` line per face (`-1 -1 -1` for an unlabelled keypoint, or a
single trailing `1` for an ignored region).

Snapshots are written to `work_dirs/<variant>/epoch_N.pth` (plus `latest.pth`) and can be
used with `--snapshot` of the other tools; `--resume` continues an interrupted run, and the
AP at IoU 0.5 on `val/` is printed every `--val-interval` epochs. Epochs, learning rate,
batch size per process and schedule default to the variant's reference values and can be
overridden. For several GPUs launch it with `torchrun`:

```console
$ uv run --extra rocm --extra train torchrun --nproc-per-node 4 -m scrfd.train scrfd_1g
```

### Open Images (no WIDER FACE licence)

[Open Images V7](https://storage.googleapis.com/openimages/web/index.html) has `Human face`
boxes under CC BY 4.0 (images: CC BY 2.0 as listed by Flickr; credit the authors). It has no
landmarks, so only the variants without `_bnkps` can be trained on it. One command downloads
the CSVs (the train ones are 2-3 GB), picks the CC BY images with faces, downloads them,
converts the boxes and writes the per-image credits to `ATTRIBUTION.csv`:

```console
$ uv run --extra rocm --extra train scrfd-fetch-openimages --max-train 20000 --max-val 2000
$ uv run --extra rocm --extra train scrfd-train scrfd_2.5g --data-root data/openimages
```

Without `--max-*` all the face images are fetched. Every step is skipped when its result is
already in `data/openimages`, so an interrupted run continues where it stopped. Images under
another license than CC BY and images with an `IsGroupOf` face box (the faces in it would be
trained on as background) are left out. `scrfd-convert-openimages` does the conversion alone
from CSVs and images you downloaded yourself. Epochs and the lr schedule are those of
WIDER FACE; tune them to the dataset size.

### Comparing with SCRFD on WIDER FACE

`scrfd-eval-widerface` reports the Easy / Medium / Hard AP that the SCRFD paper uses, with the
protocol of the official evaluation tools (IoU 0.5, 1000 score thresholds, ignored faces). It
only evaluates: the images and ground truth stay under WIDER FACE's non-commercial research
terms and must not be used for training or shipped with a model. Download `WIDER_val.zip` and
`eval_tools.zip` from the [WIDER FACE page](http://shuoyang1213.me/WIDERFACE/), unzip them, and:

```console
$ uv run --extra rocm --extra eval scrfd-eval-widerface scrfd_500m \
      --snapshot work_dirs/quick/epoch_60.pth \
      --images WIDER_val/images --gt eval_tools/ground_truth
```

Images are fitted into a 640x640 canvas (`--size`) without test-time augmentation. The published
SCRFD numbers may use a different test size or scales, so check the setting when comparing.

### Provenance and attribution (BOMs)

After training, `scrfd-bom` describes the model and the data it came from:

```console
$ uv run --extra train scrfd-bom scrfd_2.5g work_dirs/scrfd_2.5g/epoch_640.pth \
      --author "Jane Doe" --ap 0.87 --license MIT
```

It writes next to the snapshot a CycloneDX 1.6 ML-BOM (`bom.cdx.json`), an SPDX 3.0.1 document
with the AI and Dataset profiles (`bom.spdx3.json`), and a copy of each split's credits
(`ATTRIBUTION.train.csv`, `ATTRIBUTION.val.csv`). The BOMs hold the SHA-256 of the model, the
training recipe (`recipe.json`, written by `scrfd-train`), the datasets with their licenses
and the modifications made to them. It refuses to run if an image of a split has no credit.
Ship these files with the weights.

The tests check the BOMs' structure; to also validate them against the official JSON schemas,
download `bom-1.6.schema.json`, `spdx.schema.json` and `jsf-0.82.schema.json` from
[CycloneDX/specification](https://github.com/CycloneDX/specification/tree/master/schema) and
the SPDX 3.0.1 JSON schema (`spdx-json-schema.json`, saved as `spdx-schema.json`) from
<https://spdx.org/schema/3.0.1/>, and run `BOM_SCHEMAS=<folder> pytest tests/test_bom.py`.

### Packaging a release

`scrfd-package` gathers everything to publish into one folder: the weights alone (no optimizer
state), the float ONNX model, the int8 model when you pass `--calibration`, the BOMs with the
image credits, and `SHA256SUMS`.

```console
$ uv run --extra train --extra quantize scrfd-package scrfd_500m_bn work_dirs/hero/epoch_10.pth \
      --output release/scrfd_500m_bn --author "Jane Doe" --data-root data/openimages-full \
      --ap 0.85 --license Apache-2.0 --calibration data/openimages-full/train/images
```

Add a model card and the license text, then upload the folder, for example with
`uvx --from huggingface_hub hf upload <user>/<name> release/scrfd_500m_bn .`. That command
creates the repository as **public** when it doesn't exist.

## License

See `LICENSES/` and the per-file headers (this project follows the
[REUSE](https://reuse.software) specification).
