# SPDX-FileCopyrightText: 2026 Jean-Pierre De Jesus DIAZ <me@jeandudey.tech>
#
# SPDX-License-Identifier: MIT

"""``scrfd-webcam``: run an SCRFD variant on a webcam stream and draw the detections."""

import argparse
import importlib.util
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from .model import SCRFD, Detections
from .snapshot import load_snapshot
from .variants import CONFIGS, build_model

Frame = NDArray[Any]


def preprocess(frame: Frame, size: int, device: torch.device) -> tuple[torch.Tensor, float]:
    """Letterbox a BGR frame into a ``size`` x ``size`` normalised RGB tensor (top-left aligned)."""
    import cv2

    height, width = frame.shape[:2]
    scale = size / max(height, width)
    resized = cv2.resize(frame, (round(width * scale), round(height * scale)))
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[: resized.shape[0], : resized.shape[1]] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).to(device).permute(2, 0, 1).float()
    return ((tensor - 127.5) / 128.0).unsqueeze(0), scale


def draw(frame: Frame, detections: Detections, scale: float) -> None:
    """Draw boxes, scores and facial landmarks (in network pixels / ``scale``) onto ``frame``."""
    import cv2

    boxes = (detections.boxes.cpu() / scale).round().int().tolist()
    scores = detections.scores.cpu().tolist()
    keypoints = (
        None if detections.keypoints is None else (detections.keypoints.cpu() / scale).tolist()
    )
    for i, ((x1, y1, x2, y2), score) in enumerate(zip(boxes, scores, strict=True)):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{score:.2f}",
            (x1, max(y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        if keypoints is not None:
            for x, y in keypoints[i]:
                cv2.circle(frame, (round(x), round(y)), 2, (0, 0, 255), -1, cv2.LINE_AA)


def run(
    model: SCRFD, camera: int, size: int, score_thr: float, nms_thr: float, device: torch.device
) -> None:
    import cv2

    capture = cv2.VideoCapture(camera)
    if not capture.isOpened():
        msg = f"cannot open camera {camera}"
        raise SystemExit(msg)
    window = "scrfd"
    last = time.perf_counter()
    try:
        with torch.inference_mode():
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                tensor, scale = preprocess(frame, size, device)
                detections = model.detect(tensor, score_thr, nms_thr)[0]
                draw(frame, detections, scale)
                now = time.perf_counter()
                fps = 1.0 / max(now - last, 1e-6)
                last = now
                cv2.putText(
                    frame,
                    f"{fps:.1f} fps  {len(detections.scores)} det",
                    (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow(window, frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="scrfd-webcam", description=__doc__)
    parser.add_argument("variant", choices=sorted(CONFIGS), help="model variant")
    parser.add_argument("--snapshot", type=Path, required=True, help="path of a .pth snapshot")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")
    parser.add_argument("--size", type=int, default=640, help="network input size (default: 640)")
    parser.add_argument("--score-thr", type=float, default=0.5)
    parser.add_argument("--nms-thr", type=float, default=0.4)
    parser.add_argument("--device", help="torch device (default: cuda if available, else cpu)")
    args = parser.parse_args(argv)
    if importlib.util.find_spec("cv2") is None:
        parser.exit(1, "scrfd-webcam needs the 'webcam' extra: pip install 'scrfd[webcam]'\n")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(args.variant)
    load_snapshot(model, args.snapshot)
    model.eval().to(device)
    run(model, args.camera, args.size, args.score_thr, args.nms_thr, device)


if __name__ == "__main__":
    main()
