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

Variants: `scrfd_500m`, `scrfd_500m_bnkps`, `scrfd_1g`, `scrfd_1gbn`, `scrfd_1.3g`,
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

## License

See `LICENSES/` and the per-file headers (this project follows the
[REUSE](https://reuse.software) specification).
