# SPDX-FileCopyrightText: 2026 Jean-Pierre De Jesus DIAZ <me@jeandudey.tech>
#
# SPDX-License-Identifier: MIT

import numpy as np
import onnxruntime as ort
import pytest
import torch

from scrfd import CONFIGS, build_model, load_snapshot
from scrfd.export import export_onnx, main


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_forward_shapes(name: str) -> None:
    model = build_model(name).eval()
    with torch.no_grad():
        outs = model(torch.zeros(1, 3, 128, 96))
    levels = len(model.strides)
    assert len(outs) == levels * (3 if model.use_kps else 2)
    for i, s in enumerate(model.strides):
        count = model.bbox_head.num_anchors * -(-128 // s) * -(-96 // s)
        assert outs[i].shape == (1, count, 1)
        assert outs[levels + i].shape == (1, count, 4)
        if model.use_kps:
            assert outs[2 * levels + i].shape == (1, count, 10)


def test_state_dict_keys() -> None:
    keys = build_model("scrfd_500m_bnkps").state_dict().keys()
    assert "bbox_head.cls_stride_convs.(8, 8).0.depthwise_conv.conv.weight" in keys
    assert "bbox_head.cls_stride_convs.(32, 32).1.pointwise_conv.bn.running_mean" in keys
    assert "bbox_head.stride_kps.(16, 16).bias" in keys
    assert "bbox_head.integral.project" in keys
    assert not any(k.startswith("bbox_head.scales") for k in keys)
    keys = build_model("scrfd_10g").state_dict().keys()
    assert "backbone.stem.6.weight" in keys
    assert "backbone.layer2.0.downsample.1.weight" in keys
    assert "bbox_head.cls_stride_convs.0.0.conv.weight" in keys
    assert "bbox_head.cls_stride_convs.0.0.gn.weight" in keys
    assert "bbox_head.scales.2.scale" in keys
    assert "neck.pafpn_convs.1.conv.bias" in keys


def test_snapshot_roundtrip(tmp_path) -> None:  # noqa: ANN001
    src = build_model("scrfd_2.5g_bnkps")
    path = tmp_path / "m.pth"
    torch.save({"state_dict": {f"{k}": v for k, v in src.state_dict().items()}, "meta": {}}, path)
    dst = load_snapshot(build_model("scrfd_2.5g_bnkps"), path)
    for a, b in zip(src.state_dict().values(), dst.state_dict().values(), strict=True):
        assert torch.equal(a, b)


@pytest.mark.parametrize("name", ["scrfd_2.5g_bnkps", "scrfd_1g"])
def test_onnx_export_matches(tmp_path, name: str) -> None:  # noqa: ANN001
    model = build_model(name).eval()
    path = tmp_path / "m.onnx"
    export_onnx(model, path)
    sess = ort.InferenceSession(str(path))
    for size in ((320, 320), (192, 256)):
        x = torch.randn(1, 3, *size)
        expected = model(x)
        got = [np.asarray(o) for o in sess.run(None, {"image": x.numpy()})]
        assert len(got) == len(expected)
        for g, e in zip(got, expected, strict=True):
            np.testing.assert_allclose(g, e.detach().numpy(), rtol=1e-3, atol=1e-4)


def test_cli_fixed_shape(tmp_path) -> None:  # noqa: ANN001
    out = tmp_path / "f.onnx"
    main(["scrfd_500m", "--output", str(out), "--shape", "160", "160"])
    sess = ort.InferenceSession(str(out))
    assert [o.name for o in sess.get_outputs()] == [
        f"{k}_{s}" for k in ("score", "bbox") for s in (8, 16, 32)
    ]


def test_detect() -> None:
    model = build_model("scrfd_500m_bnkps").eval()
    dets = model.detect(torch.randn(2, 3, 160, 160), score_thr=0.0)
    assert len(dets) == 2
    assert dets[0].keypoints is not None
    assert dets[0].boxes.shape[1] == 4


def test_crowdhuman_five_levels(tmp_path) -> None:  # noqa: ANN001
    model = build_model("scrfd_crowdhuman_2.5g_bnkps").eval()
    assert model.strides == [8, 16, 32, 64, 128]
    assert len(model(torch.zeros(1, 3, 256, 256))) == 15
    assert len(model.detect(torch.randn(1, 3, 256, 256), score_thr=0.0)) == 1
    export_onnx(model, tmp_path / "c.onnx")
    sess = ort.InferenceSession(str(tmp_path / "c.onnx"))
    assert len(sess.get_outputs()) == 15


def test_detect_without_detections() -> None:
    model = build_model("scrfd_500m_bnkps").eval()
    (detections,) = model.detect(torch.zeros(1, 3, 64, 64), score_thr=1.1)
    assert detections.boxes.shape == (0, 4)
    assert detections.keypoints is not None
    assert detections.keypoints.shape == (0, 5, 2)
