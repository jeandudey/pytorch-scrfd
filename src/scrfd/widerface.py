# SPDX-FileCopyrightText: 2026 Jean-Pierre De Jesus DIAZ <me@jeandudey.tech>
#
# SPDX-License-Identifier: MIT

"""``scrfd-eval-widerface``: Easy / Medium / Hard AP of a detector on the WIDER FACE val set.

Follows the protocol of the official evaluation tools: IoU 0.5, scores normalised over the
whole set, 1000 score thresholds, faces outside the subset of a difficulty ignored. It needs
the validation images and the ground truth ``.mat`` files of the official ``eval_tools``
(``wider_face_val.mat``, ``wider_easy_val.mat``, ``wider_medium_val.mat``,
``wider_hard_val.mat``), which are under WIDER FACE's own non-commercial research terms.
"""

import argparse
import importlib.util
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from numpy.typing import NDArray

from .dataset import letterbox, read_image
from .model import SCRFD, Detections
from .snapshot import load_snapshot
from .variants import CONFIGS, build_model

Array = NDArray[np.float64]
SETTINGS = ("easy", "medium", "hard")
IOU_THRESHOLD = 0.5
NUM_THRESHOLDS = 1000
SCORE_THRESHOLD = 0.01


class Detector(Protocol):
    def eval(self) -> object: ...

    def detect(self, x: torch.Tensor, score_thr: float, nms_thr: float) -> list[Detections]: ...


class OnnxDetector:
    """Runs an exported model (float or quantized) with ONNX Runtime, on the CPU.

    The network input has the fixed size the model was exported with. ``model`` only supplies
    the decoding of the outputs, so its weights do not matter.
    """

    def __init__(self, path: Path, model: SCRFD) -> None:
        import onnxruntime

        self.session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.model = model

    def eval(self) -> None:
        """Nothing to switch: an exported model is always in inference mode."""

    def detect(self, x: torch.Tensor, score_thr: float, nms_thr: float) -> list[Detections]:
        outputs = self.session.run(None, {"image": x.cpu().numpy()})
        return self.model.postprocess(
            tuple(torch.from_numpy(o) for o in outputs),
            (x.shape[2], x.shape[3]),
            score_thr,
            nms_thr,
        )


@dataclass
class Image:
    """A validation image with its faces and the faces each difficulty counts."""

    event: str
    name: str
    boxes: Array  # (G, 4) x1, y1, x2, y2
    keep: dict[str, NDArray[np.bool_]]  # setting -> (G,)


def load_ground_truth(folder: Path) -> list[Image]:
    """Read the ground truth of the official ``eval_tools/ground_truth`` folder."""
    from scipy.io import loadmat

    gt = loadmat(str(folder / "wider_face_val.mat"))
    subsets = {s: loadmat(str(folder / f"wider_{s}_val.mat"))["gt_list"] for s in SETTINGS}
    images = []
    for e, event in enumerate(gt["event_list"][:, 0]):
        files, boxes = gt["file_list"][e, 0], gt["face_bbx_list"][e, 0]
        for i in range(len(files)):
            xywh = np.asarray(boxes[i, 0], dtype=np.float64).reshape(-1, 4)
            keep = {}
            for setting, gt_list in subsets.items():
                keep[setting] = np.zeros(len(xywh), dtype=bool)
                keep[setting][np.asarray(gt_list[e, 0][i, 0], dtype=int).ravel() - 1] = True
            images.append(
                Image(
                    event=str(event[0]),
                    name=str(files[i, 0][0]),
                    boxes=np.concatenate([xywh[:, :2], xywh[:, :2] + xywh[:, 2:]], axis=1),
                    keep=keep,
                )
            )
    return images


def overlaps(boxes: Array, gt: Array) -> Array:
    """IoU of every box with every ground truth box, in pixel-inclusive coordinates."""
    iw = np.minimum(boxes[:, None, 2], gt[None, :, 2]) - np.maximum(
        boxes[:, None, 0], gt[None, :, 0]
    )
    ih = np.minimum(boxes[:, None, 3], gt[None, :, 3]) - np.maximum(
        boxes[:, None, 1], gt[None, :, 1]
    )
    inter = np.clip(iw + 1, 0, None) * np.clip(ih + 1, 0, None)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    gt_area = (gt[:, 2] - gt[:, 0] + 1) * (gt[:, 3] - gt[:, 1] + 1)
    return inter / (area[:, None] + gt_area[None, :] - inter)


def match(
    best_iou: Array, best_gt: NDArray[np.intp], keep: NDArray[np.bool_]
) -> tuple[NDArray[np.intp], NDArray[np.bool_]]:
    """Walk the detections, best first, and return per detection the faces found so far and
    whether it counts as a proposal.

    A detection whose best match is a face outside the difficulty is ignored; a second detection
    on an already found face is a false positive.
    """
    found = np.zeros(len(keep), dtype=bool)
    recall = np.zeros(len(best_iou), dtype=np.intp)
    proposal = np.ones(len(best_iou), dtype=bool)
    count = 0
    for h, (iou, g) in enumerate(zip(best_iou.tolist(), best_gt.tolist(), strict=True)):
        if iou >= IOU_THRESHOLD:
            if not keep[g]:
                proposal[h] = False
            elif not found[g]:
                found[g] = True
                count += 1
        recall[h] = count
    return recall, proposal


def curve(scores: Array, recall: NDArray[np.intp], proposal: NDArray[np.bool_]) -> Array:
    """Proposals and faces found at each of the score thresholds for one image."""
    thresholds = 1 - (np.arange(1, NUM_THRESHOLDS + 1) / NUM_THRESHOLDS)
    taken = np.searchsorted(-scores, -thresholds, side="right")  # detections >= threshold
    cumulative = np.concatenate([[0], np.cumsum(proposal)])
    found = np.concatenate([[0], recall])
    return np.stack([cumulative[taken], found[taken]], axis=1).astype(np.float64)


def voc_ap(recall: Array, precision: Array) -> float:
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    changed = np.nonzero(mrec[1:] != mrec[:-1])[0]
    return float(((mrec[changed + 1] - mrec[changed]) * mpre[changed + 1]).sum())


def score(images: Sequence[Image], detections: Sequence[Array]) -> dict[str, float]:
    """AP per difficulty; ``detections[i]`` is ``(N, 5)`` x1, y1, x2, y2, score of ``images[i]``."""
    sorted_detections = [d[np.argsort(-d[:, 4], kind="stable")] for d in detections]
    all_scores = np.concatenate([d[:, 4] for d in sorted_detections])
    low, high = (all_scores.min(), all_scores.max()) if len(all_scores) else (0.0, 1.0)
    span = high - low if high > low else 1.0

    matches = []
    for image, det in zip(images, sorted_detections, strict=True):
        if len(det) == 0 or len(image.boxes) == 0:
            matches.append(None)
            continue
        iou = overlaps(det[:, :4], image.boxes)
        matches.append((iou.max(axis=1), iou.argmax(axis=1), (det[:, 4] - low) / span))

    aps = {}
    for setting in SETTINGS:
        total = np.zeros((NUM_THRESHOLDS, 2))
        faces = sum(int(image.keep[setting].sum()) for image in images)
        for image, m in zip(images, matches, strict=True):
            if m is not None:
                best_iou, best_gt, scores = m
                recall, proposal = match(best_iou, best_gt, image.keep[setting])
                total += curve(scores, recall, proposal)
        precision = np.divide(
            total[:, 1], total[:, 0], out=np.zeros(NUM_THRESHOLDS), where=total[:, 0] > 0
        )
        aps[setting] = voc_ap(total[:, 1] / max(faces, 1), precision)
    return aps


@torch.no_grad()
def detect_all(
    model: Detector,
    images: Sequence[Image],
    root: Path,
    device: torch.device,
    size: int,
    nms_thr: float,
    progress: Callable[[int], None] | None = None,
) -> list[Array]:
    """Detections of every image, with the image fitted into ``size`` x ``size``."""
    model.eval()
    results = []
    for i, image in enumerate(images):
        tensor, scale = letterbox(read_image(root / image.event / f"{image.name}.jpg"), size)
        (found,) = model.detect(tensor[None].to(device), SCORE_THRESHOLD, nms_thr)
        boxes = found.boxes.cpu().numpy().astype(np.float64) / scale
        results.append(np.concatenate([boxes, found.scores.cpu().numpy()[:, None]], axis=1))
        if progress is not None:
            progress(i + 1)
    return results


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="scrfd-eval-widerface", description=__doc__)
    parser.add_argument("variant", choices=sorted(CONFIGS), help="model variant")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--snapshot", type=Path, help="a .pth snapshot, run with PyTorch")
    source.add_argument(
        "--onnx",
        type=Path,
        help="an ONNX model of the variant (see scrfd-quantize), run on the CPU",
    )
    parser.add_argument("--images", type=Path, required=True, help="WIDER_val/images")
    parser.add_argument("--gt", type=Path, required=True, help="eval_tools/ground_truth")
    parser.add_argument("--size", type=int, default=640, help="network input size")
    parser.add_argument("--nms-thr", type=float, default=0.45)
    parser.add_argument("--device", help="default: cuda if available, else cpu")
    args = parser.parse_args(argv)
    if importlib.util.find_spec("scipy") is None or importlib.util.find_spec("cv2") is None:
        msg = "scrfd-eval-widerface needs the 'eval' extra: pip install 'scrfd[eval]'\n"
        raise SystemExit(msg)
    if args.onnx is not None and importlib.util.find_spec("onnxruntime") is None:
        msg = "scrfd-eval-widerface --onnx needs onnxruntime: pip install 'scrfd[quantize]'\n"
        raise SystemExit(msg)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(args.variant)
    detector: Detector
    if args.onnx is not None:
        device = torch.device("cpu")
        detector = OnnxDetector(args.onnx, model)
    else:
        load_snapshot(model, args.snapshot)
        detector = model.to(device)
    images = load_ground_truth(args.gt)

    def progress(done: int) -> None:
        if done % 500 == 0 or done == len(images):
            print(f"  {done}/{len(images)} images", flush=True)

    detections = detect_all(
        detector, images, args.images, device, args.size, args.nms_thr, progress
    )
    for setting, ap in score(images, detections).items():
        print(f"{setting.capitalize():<6} val AP: {ap:.4f}")


if __name__ == "__main__":
    main()
