#!/usr/bin/env python3
"""Train the V21 restorer while keeping the tri-factor base frozen."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.mixed_dataset import MixedDataset
from src.data.v21_dataset import build_v21_datasets
from src.losses.v21_loss import V21RestorerLoss
from src.models.v21_restorer import V21AlbedoRestorer
from src.models.v21_trifactor import IntrinsicDecompositionV21
from src.train_v21 import move_batch, seed_everything, state_dict_from_checkpoint, worker_seed

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "src/configs/v21_restorer.yaml"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base-checkpoint", default=None)
    parser.add_argument("--resume", nargs="?", const="latest", default=None)
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def save_checkpoint(
    path: Path,
    model: V21AlbedoRestorer,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    step: int,
    config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(
        {
            "global_step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "config": config,
        },
        temporary,
    )
    os.replace(temporary, path)


def latest_valid(directory: Path) -> Path | None:
    paths = list(directory.glob("checkpoint_iter_*.pth"))
    if (directory / "checkpoint_latest.pth").is_file():
        paths.append(directory / "checkpoint_latest.pth")
    for path in sorted(paths, key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            checkpoint = torch.load(path, map_location="cpu")
            state_dict_from_checkpoint(checkpoint)
            return path
        except Exception as exc:
            print(f"[skip corrupt restorer checkpoint] {path}: {exc}")
    return None


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    base_config_path = resolve(ROOT, config["base"]["config"])
    base_config = yaml.safe_load(base_config_path.read_text())
    base_config["train"]["input_size"] = int(config["train"].get("input_size", 512))
    base_config["train"]["sampling_weights"] = config["train"]["sampling_weights"]
    seed_everything(int(config["train"].get("seed", 42)))

    device = torch.device(args.device if args.device != "cuda" else "cuda:0")
    amp_enabled = bool(config["train"].get("amp", True)) and device.type == "cuda"
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    base_checkpoint = resolve(
        ROOT, args.base_checkpoint or config["base"]["checkpoint"]
    )
    if not base_checkpoint.is_file():
        raise FileNotFoundError(f"Frozen V21 checkpoint not found: {base_checkpoint}")
    base = IntrinsicDecompositionV21(base_config["model"]).to(device)
    base_state = state_dict_from_checkpoint(torch.load(base_checkpoint, map_location="cpu"))
    base.load_state_dict(base_state, strict=True)
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad = False
    print(f"Frozen base: {base_checkpoint}")

    restorer = V21AlbedoRestorer(config["model"]).to(device)
    criterion = V21RestorerLoss(config["loss"]).to(device)
    optimizer = torch.optim.AdamW(
        restorer.parameters(),
        lr=float(config["train"].get("lr", 1e-4)),
        weight_decay=float(config["train"].get("weight_decay", 1e-4)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    checkpoint_dir = resolve(ROOT, config["output"]["checkpoint_dir"])
    log_dir = resolve(ROOT, config["output"]["log_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    resume_path = None
    if args.resume and args.resume != "latest":
        resume_path = resolve(ROOT, args.resume)
    elif args.resume or args.auto_resume:
        resume_path = latest_valid(checkpoint_dir)
        if args.resume and resume_path is None:
            raise FileNotFoundError(f"No valid restorer checkpoint under {checkpoint_dir}")
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location="cpu")
        restorer.load_state_dict(state_dict_from_checkpoint(checkpoint), strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        step = int(checkpoint.get("global_step", 0))
        print(f"Resumed restorer at step {step}: {resume_path}")

    datasets = build_v21_datasets(base_config, "train")
    mixed = MixedDataset(datasets, config["train"]["sampling_weights"])
    workers = int(config["train"].get("num_workers", 4))
    loader = DataLoader(
        mixed,
        batch_size=int(config["train"].get("batch_size", 1)),
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
        worker_init_fn=worker_seed,
    )

    max_steps = int(args.max_steps or config["train"].get("max_iterations", 20000))
    accumulation = int(config["train"].get("grad_accum_steps", 8))
    log_interval = int(config["train"].get("log_interval", 50))
    checkpoint_interval = int(config["train"].get("checkpoint_interval_iters", 1000))
    clip = float(config["train"].get("grad_clip_max_norm", 1.0))
    iterator = iter(loader)
    micro_step = 0
    running: dict[str, float] = {}
    restorer.train()
    optimizer.zero_grad(set_to_none=True)
    while step < max_steps:
        try:
            raw_batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            raw_batch = next(iterator)
        batch = move_batch(raw_batch, device)
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp_enabled
        ):
            base_output = base(batch["rgb"])
            base_pair = base(batch["rgb2"]) if batch["pair_supervision"].sum() > 0 else None
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            output = restorer.training_forward(batch["rgb"], base_output, batch["albedo_gt"])
            pair_output = None
            if base_pair is not None:
                pair_output = restorer.training_forward(
                    batch["rgb2"], base_pair, batch["albedo_gt"]
                )
            total, losses = criterion(output, base_output, batch, pair_output)

        if not torch.isfinite(total):
            raise FloatingPointError(f"Non-finite restorer loss at step {step}")
        if args.dry_run:
            print(json.dumps({"total": float(total), **{key: float(value) for key, value in losses.items()}}, indent=2))
            print("V21 restorer dry run passed; base remained frozen")
            return 0
        scaler.scale(total / accumulation).backward()
        micro_step += 1
        for key, value in {"total": total, **losses}.items():
            running[key] = running.get(key, 0.0) + float(value.detach())
        if micro_step % accumulation:
            continue
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(restorer.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % log_interval == 0:
            averages = {key: value / (log_interval * accumulation) for key, value in running.items()}
            print(f"step={step} " + " ".join(f"{key}={value:.4f}" for key, value in sorted(averages.items())), flush=True)
            with (log_dir / "train.jsonl").open("a") as handle:
                handle.write(json.dumps({"step": step, **averages}) + "\n")
            running.clear()
        if step % checkpoint_interval == 0:
            save_checkpoint(checkpoint_dir / f"checkpoint_iter_{step}.pth", restorer, optimizer, scaler, step, config)
            save_checkpoint(checkpoint_dir / "checkpoint_latest.pth", restorer, optimizer, scaler, step, config)

    save_checkpoint(checkpoint_dir / "checkpoint_latest.pth", restorer, optimizer, scaler, step, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
