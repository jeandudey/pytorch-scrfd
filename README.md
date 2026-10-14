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

## License

See `LICENSES/` and the per-file headers (this project follows the
[REUSE](https://reuse.software) specification).
