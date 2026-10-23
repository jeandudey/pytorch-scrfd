# SPDX-FileCopyrightText: 2018-2019 Open-MMLab
# SPDX-FileCopyrightText: 2021 Jia Guo, InsightFace contributors
# SPDX-FileCopyrightText: 2026 Jean-Pierre De Jesus DIAZ <me@jeandudey.tech>
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

"""``scrfd-train``: train an SCRFD variant on a RetinaFace-format dataset.

Run it under ``torchrun`` to train on several GPUs.
"""

import argparse
import importlib.util
import json
import os
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, Sampler
from typing_extensions import override

from .dataset import RetinaFaceDataset, collate, load_records
from .evaluate import evaluate
from .loss import SCRFDLoss
from .model import SCRFD
from .snapshot import extract_state_dict, load_snapshot
from .variants import CONFIGS, build_model

WARMUP_RATIO = 0.001
LR_DECAY = 0.1


@dataclass(frozen=True)
class Recipe:
    """Hyper-parameters of the reference training of a variant.

    ``batch_size`` is per process, as the learning rate is not scaled with the number of
    processes.
    """

    lr: float = 0.01
    epochs: int = 640
    lr_steps: tuple[int, ...] = (440, 544)
    warmup_iters: int = 1500
    batch_size: int = 8
    momentum: float = 0.9
    weight_decay: float = 5e-4
    data_root: str = "data/retinaface"
    annotations: str = "labelv2.txt"
    clip_border: bool = True


DEFAULT_RECIPE = Recipe()
RECIPES: dict[str, Recipe] = {
    "scrfd_500m_bn": replace(DEFAULT_RECIPE, batch_size=16),
    "scrfd_500m_bnkps": replace(DEFAULT_RECIPE, batch_size=16),
    "scrfd_34g": replace(DEFAULT_RECIPE, lr=0.02),
    "scrfdgen2.5g": replace(DEFAULT_RECIPE, epochs=80, lr_steps=(55, 68)),
    "scrfd_crowdhuman_2.5g_bnkps": replace(
        DEFAULT_RECIPE,
        epochs=320,
        lr_steps=(220, 272),
        warmup_iters=2500,
        data_root="data/crowdhuman",
        annotations="label_fullbody.txt",
        clip_border=False,
    ),
}


def learning_rate(recipe: Recipe, epoch: int, iteration: int) -> float:
    """Step decay per epoch with a linear warm-up over the first iterations."""
    lr = recipe.lr * LR_DECAY ** sum(epoch >= step for step in recipe.lr_steps)
    if iteration < recipe.warmup_iters:
        lr *= 1 - (1 - iteration / recipe.warmup_iters) * (1 - WARMUP_RATIO)
    return lr


class Trainer(nn.Module):
    """A model together with its loss, so that one forward pass gives the training losses."""

    def __init__(self, model: SCRFD, loss: SCRFDLoss) -> None:
        super().__init__()
        self.model = model
        self.loss = loss

    @override
    def forward(
        self, images: Tensor, gt_boxes: Sequence[Tensor], gt_keypoints: Sequence[Tensor]
    ) -> dict[str, Tensor]:
        outputs = self.model.predict(images)
        return self.loss(outputs, (images.shape[2], images.shape[3]), gt_boxes, gt_keypoints)


def save_snapshot(
    path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, iteration: int
) -> None:
    """Write ``path`` atomically and point ``latest.pth`` next to it at it."""
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "iteration": iteration,
        },
        temporary,
    )
    temporary.replace(path)
    latest = path.parent / "latest.pth"
    latest.unlink(missing_ok=True)
    latest.symlink_to(path.name)


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="scrfd-train", description=__doc__)
    parser.add_argument("variant", choices=sorted(CONFIGS), help="model variant")
    parser.add_argument(
        "--data-root",
        type=Path,
        help="directory with train/ and val/, each with images/ and the annotation file "
        "(default: the one of the variant)",
    )
    parser.add_argument(
        "--output", type=Path, help="directory for snapshots (default: work_dirs/<variant>)"
    )
    parser.add_argument("--epochs", type=int, help="number of epochs (default: the variant's)")
    parser.add_argument(
        "--batch-size", type=int, help="images per process (default: the variant's)"
    )
    parser.add_argument("--lr", type=float, help="learning rate (default: the variant's)")
    parser.add_argument(
        "--lr-steps", type=int, nargs="+", help="epochs that divide the learning rate by 10"
    )
    parser.add_argument("--warmup-iters", type=int, help="iterations of linear lr warm-up")
    parser.add_argument("--size", type=int, default=640, help="training image size")
    parser.add_argument("--workers", type=int, default=3, help="data loading workers per process")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--crop-bias",
        type=float,
        default=0.0,
        help="favour large random crops, and so small faces: odds grow with size**bias (0: equal)",
    )
    parser.add_argument("--device", help="default: cuda if available, else cpu")
    parser.add_argument("--snapshot", type=Path, help="initialise the weights from this snapshot")
    parser.add_argument("--resume", type=Path, help="continue from a snapshot written by this tool")
    parser.add_argument(
        "--snapshot-interval", type=int, default=80, help="epochs between snapshots"
    )
    parser.add_argument(
        "--val-interval", type=int, default=80, help="epochs between evaluations (0: never)"
    )
    parser.add_argument(
        "--log-interval", type=int, default=100, help="iterations between log lines"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if importlib.util.find_spec("cv2") is None:
        msg = "scrfd-train needs the 'train' extra: pip install 'scrfd[train]'\n"
        raise SystemExit(msg)

    recipe = RECIPES.get(args.variant, DEFAULT_RECIPE)
    overrides: dict[str, Any] = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lr_steps": None if args.lr_steps is None else tuple(args.lr_steps),
        "warmup_iters": args.warmup_iters,
    }
    recipe = replace(recipe, **{k: v for k, v in overrides.items() if v is not None})
    data_root = args.data_root or Path(recipe.data_root)
    output = args.output or Path("work_dirs") / args.variant

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = torch.cuda.is_available() if args.device is None else args.device.startswith("cuda")
    if args.device is not None and not args.device.startswith("cuda"):
        device = torch.device(args.device)
    elif use_cuda:
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if world_size > 1:
        dist.init_process_group(
            "nccl" if device.type == "cuda" else "gloo", timeout=timedelta(hours=2)
        )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    main_process = rank == 0

    torch.manual_seed(args.seed + rank)

    config = CONFIGS[args.variant]
    model = build_model(args.variant)
    if args.snapshot is not None:
        load_snapshot(model, args.snapshot)
    elif args.resume is None:
        model.init_weights()
    trainer = Trainer(model, SCRFDLoss(config)).to(device)
    optimizer = torch.optim.SGD(
        trainer.parameters(),
        lr=recipe.lr,
        momentum=recipe.momentum,
        weight_decay=recipe.weight_decay,
    )
    start_epoch = iteration = 0
    if args.resume is not None:
        state = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(extract_state_dict(state))
        optimizer.load_state_dict(state["optimizer"])
        start_epoch, iteration = state["epoch"], state["iteration"]
    parallel: nn.Module = trainer
    if world_size > 1:
        parallel = DistributedDataParallel(
            trainer, device_ids=[device.index] if device.type == "cuda" else None
        )

    dataset = RetinaFaceDataset(
        data_root / "train" / recipe.annotations,
        data_root / "train" / "images",
        size=args.size,
        clip_border=recipe.clip_border,
        seed=args.seed,
        crop_bias=args.crop_bias,
    )
    sampler: Sampler[Any]
    if world_size > 1:
        sampler = DistributedSampler(dataset, shuffle=True, seed=args.seed, drop_last=True)
    else:
        sampler = RandomSampler(dataset, generator=torch.Generator().manual_seed(args.seed))
    loader = DataLoader(
        dataset,
        batch_size=recipe.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        collate_fn=collate,
        drop_last=True,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    val_records = None
    if args.val_interval > 0 and main_process:
        val_annotations = data_root / "val" / recipe.annotations
        if val_annotations.exists():
            val_records = load_records(val_annotations)
    if main_process:
        output.mkdir(parents=True, exist_ok=True)
        (output / "recipe.json").write_text(
            json.dumps(
                {
                    "variant": args.variant,
                    "size": args.size,
                    "seed": args.seed,
                    "crop_bias": args.crop_bias,
                    **asdict(recipe),
                },
                indent=2,
            )
            + "\n"
        )
        print(
            f"{args.variant}: {len(dataset)} images, {len(loader)} iterations per epoch, "
            f"{world_size} process(es) x batch {recipe.batch_size}, device {device}"
        )

    parallel.train()
    for epoch in range(start_epoch, recipe.epochs):
        dataset.epoch = epoch
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)
        elif isinstance(sampler, RandomSampler):
            sampler.generator = torch.Generator().manual_seed(args.seed + epoch)
        started = time.time()
        for step, (images, gt_boxes, gt_keypoints) in enumerate(loader):
            lr = learning_rate(recipe, epoch, iteration)
            for group in optimizer.param_groups:
                group["lr"] = lr
            losses = parallel(
                images.to(device, non_blocking=True),
                [b.to(device, non_blocking=True) for b in gt_boxes],
                [k.to(device, non_blocking=True) for k in gt_keypoints],
            )
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                msg = f"non-finite loss at epoch {epoch + 1}, iteration {step + 1}: {losses}"
                raise RuntimeError(msg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iteration += 1
            if main_process and (step + 1) % args.log_interval == 0:
                values = ", ".join(f"{k}: {v.item():.4f}" for k, v in losses.items())
                print(
                    f"Epoch [{epoch + 1}][{step + 1}/{len(loader)}] lr: {lr:.5f}, "
                    f"{values}, loss: {loss.item():.4f}, "
                    f"time/iter: {(time.time() - started) / (step + 1):.3f}s",
                    flush=True,
                )
        done = epoch + 1
        if main_process and (done % args.snapshot_interval == 0 or done == recipe.epochs):
            save_snapshot(output / f"epoch_{done}.pth", model, optimizer, done, iteration)
        if (
            main_process
            and val_records is not None
            and (done % args.val_interval == 0 or done == recipe.epochs)
        ):
            ap = evaluate(model, val_records, data_root / "val" / "images", device, args.size)
            print(f"Epoch [{done}] val AP@0.5: {ap:.4f}", flush=True)
        if world_size > 1:
            dist.barrier()
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
