#!/usr/bin/env python3
"""Train the isolated V21 tri-factor IID model on one GPU."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.mixed_dataset import MixedDataset  # noqa: E402
from src.data.v21_dataset import build_v21_datasets  # noqa: E402
from src.losses.v21_loss import V21Loss  # noqa: E402
from src.models.v21_trifactor import IntrinsicDecompositionV21  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "src/configs/v21.yaml"))
    parser.add_argument("--version", default="21")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", nargs="?", const="latest", default=None)
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--init", default=None, help="Override train.init_checkpoint")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def state_dict_from_checkpoint(checkpoint: dict) -> dict[str, torch.Tensor]:
    for key in ("model_state_dict", "state_dict", "model"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]
    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint
    raise KeyError("Checkpoint has no model state dictionary")


def load_shape_compatible(model: torch.nn.Module, path: Path) -> tuple[int, int]:
    checkpoint = torch.load(path, map_location="cpu")
    incoming = state_dict_from_checkpoint(checkpoint)
    own = model.state_dict()
    filtered = {
        key: value for key, value in incoming.items()
        if key in own and own[key].shape == value.shape
    }
    model.load_state_dict(filtered, strict=False)
    print(f"Initialized {len(filtered)}/{len(own)} tensors from {path}")
    return len(filtered), len(own)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    step: int,
    config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(
        {
            "global_step": int(step),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "config": config,
        },
        temporary,
    )
    os.replace(temporary, path)


def candidate_checkpoints(directory: Path) -> list[Path]:
    candidates = list(directory.glob("checkpoint_iter_*.pth"))
    latest = directory / "checkpoint_latest.pth"
    if latest.is_file():
        candidates.append(latest)
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)


def find_latest_valid(directory: Path) -> Path | None:
    for path in candidate_checkpoints(directory):
        try:
            checkpoint = torch.load(path, map_location="cpu")
            state_dict_from_checkpoint(checkpoint)
            return path
        except Exception as exc:
            print(f"[skip corrupt checkpoint] {path}: {exc}")
    return None


def resume_training(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
) -> int:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict_from_checkpoint(checkpoint), strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    step = int(checkpoint.get("global_step", 0))
    print(f"Resumed V21 at step {step} from {path}")
    return step


def make_loaders(config: dict) -> tuple[DataLoader, DataLoader]:
    train_config = config["train"]
    datasets = build_v21_datasets(config, "train")
    mixed = MixedDataset(datasets, train_config["sampling_weights"])
    workers = int(train_config.get("num_workers", 4))
    generator = torch.Generator().manual_seed(int(train_config.get("seed", 42)))
    train_loader = DataLoader(
        mixed,
        batch_size=int(train_config.get("batch_size", 1)),
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
        worker_init_fn=worker_seed,
        generator=generator,
    )
    val_dataset = next(iter(build_v21_datasets(config, "val").values()))
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=min(2, workers),
        pin_memory=True,
    )
    return train_loader, val_loader


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    criterion: V21Loss,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
    amp_enabled: bool,
) -> float:
    model.eval()
    values: list[float] = []
    for index, raw_batch in enumerate(loader):
        if index >= max_batches:
            break
        batch = move_batch(raw_batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            output = model(batch["rgb"])
            loss, _ = criterion(output, batch)
        values.append(float(loss))
    model.train()
    return float(np.mean(values)) if values else float("nan")


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text())
    train_config = config["train"]
    output_config = config["output"]
    seed_everything(int(train_config.get("seed", 42)))

    device = torch.device(args.device if args.device != "cuda" else "cuda:0")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    amp_enabled = bool(train_config.get("amp", True)) and device.type == "cuda"
    checkpoint_dir = ROOT / output_config.get("checkpoint_dir", "checkpoints/v21")
    log_dir = ROOT / output_config.get("log_dir", "logs/v21")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    train_loader, val_loader = make_loaders(config)
    model = IntrinsicDecompositionV21(config["model"]).to(device)
    criterion = V21Loss(config["loss"]).to(device)
    parameters = [parameter for parameter in model.parameters() if not parameter.is_floating_point() or parameter.requires_grad]
    parameters = [parameter for parameter in parameters if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(train_config.get("lr", 2e-5)),
        weight_decay=float(train_config.get("weight_decay", 1e-4)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    resume_path: Path | None = None
    if args.resume or args.auto_resume:
        if args.resume and args.resume != "latest":
            resume_path = Path(args.resume).expanduser().resolve()
        else:
            resume_path = find_latest_valid(checkpoint_dir)
            if args.resume and resume_path is None:
                raise FileNotFoundError(f"No valid V21 checkpoint under {checkpoint_dir}")
    step = 0
    if resume_path is not None:
        step = resume_training(resume_path, model, optimizer, scaler)
    else:
        init_value = args.init or train_config.get("init_checkpoint")
        if init_value:
            init_path = (ROOT / init_value).resolve() if not Path(init_value).is_absolute() else Path(init_value)
            if not init_path.is_file():
                raise FileNotFoundError(f"Initialization checkpoint not found: {init_path}")
            load_shape_compatible(model, init_path)

    max_steps = int(args.max_steps or train_config.get("max_iterations", 40000))
    accumulation = int(train_config.get("grad_accum_steps", 1))
    clip_norm = float(train_config.get("grad_clip_max_norm", 1.0))
    freeze_steps = int(train_config.get("albedo_freeze_steps", 0))
    log_interval = int(train_config.get("log_interval", 50))
    val_interval = int(train_config.get("val_interval_iters", 1000))
    checkpoint_interval = int(train_config.get("checkpoint_interval_iters", 1000))
    max_val_batches = int(train_config.get("max_val_batches", 8))

    model.train()
    optimizer.zero_grad(set_to_none=True)
    running: dict[str, float] = {}
    train_iterator = iter(train_loader)
    micro_step = 0
    while step < max_steps:
        try:
            raw_batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            raw_batch = next(train_iterator)
        batch = move_batch(raw_batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            output = model(batch["rgb"])
            pair_output = model(batch["rgb2"]) if batch["pair_supervision"].sum() > 0 else None
            total, losses = criterion(output, batch, pair_output)
            scaled_total = total / accumulation

        if not torch.isfinite(total):
            raise FloatingPointError(f"Non-finite V21 loss at step {step}: {float(total)}")
        if args.dry_run:
            print(json.dumps({"total": float(total), **{key: float(value) for key, value in losses.items()}}, indent=2))
            print("V21 dry run passed")
            return 0

        scaler.scale(scaled_total).backward()
        if step < freeze_steps:
            for parameter in model.albedo_head.parameters():
                parameter.grad = None
        micro_step += 1
        for key, value in losses.items():
            running[key] = running.get(key, 0.0) + float(value.detach())
        running["total"] = running.get("total", 0.0) + float(total.detach())

        if micro_step % accumulation != 0:
            continue
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, clip_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % log_interval == 0:
            averages = {key: value / (log_interval * accumulation) for key, value in running.items()}
            print(f"step={step} lr={optimizer.param_groups[0]['lr']:.3e} " + " ".join(
                f"{key}={value:.4f}" for key, value in sorted(averages.items())
            ), flush=True)
            with (log_dir / "train.jsonl").open("a") as handle:
                handle.write(json.dumps({"step": step, **averages}) + "\n")
            running.clear()

        if step % val_interval == 0:
            value = validate(model, criterion, val_loader, device, max_val_batches, amp_enabled)
            print(f"step={step} val_total={value:.5f}", flush=True)
            with (log_dir / "val.jsonl").open("a") as handle:
                handle.write(json.dumps({"step": step, "total": value}) + "\n")

        if step % checkpoint_interval == 0:
            numbered = checkpoint_dir / f"checkpoint_iter_{step}.pth"
            save_checkpoint(numbered, model, optimizer, scaler, step, config)
            save_checkpoint(checkpoint_dir / "checkpoint_latest.pth", model, optimizer, scaler, step, config)

    save_checkpoint(checkpoint_dir / "checkpoint_latest.pth", model, optimizer, scaler, step, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
