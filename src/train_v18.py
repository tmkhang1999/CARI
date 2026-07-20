"""Trainer for V18 — Physics-Grounded One-Step Colorful-Diffuse Intrinsic Diffusion.

Usage:
    python src/train_v18.py --version 18

A single SD-2.1 U-Net is fine-tuned as a ONE-STEP predictor of [albedo | π-shading],
trained end-to-end with the project's physics-loss suite (V17Loss) in image space.
VAE frozen. No V17 dependency. Clean synthetic data only (Hypersim + InteriorVerse).
"""

import argparse
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.amp import autocast, GradScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.v18_pgid import V18PGID
from losses.flexible_loss_v17 import V17Loss
from data.hypersim_dataset import get_hypersim_loader
from train_v17 import (
    _compute_lmse,
    _masked_scale_invariant_rmse,
    _compute_ssim_bounded,
    _compute_shading_ssim,
    _get_tonemap_scale,
    _vis_tonemap,
    _find_latest_checkpoint,
    _extract_iter_from_name,
    _deep_merge,
    TB_TAGS as V17_TB_TAGS,
    _log_ordered_scalars,
)


VAL_METRIC_TAGS = {
    "a_lmse": "2. Val Albedo/LMSE", "a_rmse": "2. Val Albedo/si-RMSE", "a_ssim": "2. Val Albedo/SSIM",
    "s_lmse": "2. Val Shading/LMSE", "s_rmse": "2. Val Shading/si-RMSE", "s_ssim": "2. Val Shading/SSIM",
}


# ── Diffusion targets (fixed canonical scale — no scale-match) ───────────────

def compute_diffusion_targets(batch, device, pi_floor=5e-3):
    """Fixed (A_gt, π_gt, S_d_gt, R_gt) for V17Loss — no per-image scale-match.

    A_gt = albedo_raw; S_d from diffuse-illum GT (Hypersim) else colorful I/A;
    π_gt = 1/(S_d+1); R_gt = I − A·S_d.
    """
    rgb = batch["rgb"].to(device)
    A_gt = batch["albedo_raw"].to(device).clamp(0.0, 1.0)
    A_safe = A_gt.clamp_min(1e-4)
    S_c = torch.nan_to_num(rgb / A_safe, nan=0.0, posinf=60000.0, neginf=0.0).clamp_min(0.0)

    S_d = S_c
    illum = batch.get("illum_raw", None)
    m_diffuse = batch.get("M_diffuse", None)
    if illum is not None and m_diffuse is not None:
        route = m_diffuse.view(-1, 1, 1, 1).to(device=device, dtype=rgb.dtype)
        illum = torch.nan_to_num(illum.to(device), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        S_d = route * illum + (1.0 - route) * S_c

    pi = (1.0 / (S_d + 1.0)).clamp(pi_floor, 1.0 - 1e-4)
    S_d_lin = (1.0 - pi) / pi
    R = rgb - A_gt * S_d_lin
    return {
        "A_d_star": A_gt,
        "S_d_star": S_d_lin,
        "pi_star": pi,
        "R_star": R,
        "ccr_edge_gt": None,
    }


# ── Args / config ────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train V18 one-step intrinsic diffusion")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--version", type=str, default="18")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--auto-resume", action="store_true")
    p.add_argument("--reset-lr", action="store_true")
    p.add_argument("--skip-optimizer", action="store_true")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def load_config(config_path=None, version=None):
    base_path = SRC_DIR / "configs" / "base.yaml"
    with open(base_path) as f:
        config = yaml.safe_load(f)
    if config_path is None and version is not None:
        config_path = str(SRC_DIR / "configs" / f"v{version}.yaml")
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            override = yaml.safe_load(f) or {}
        config = _deep_merge(config, override)
        print(f"Config: base.yaml <- {os.path.basename(config_path)}")
    return config


# ── Checkpoints ──────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, losses, config, path, global_step):
    tmp = path + ".tmp"
    torch.save({
        "global_step": int(global_step), "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "losses": losses, "config": config,
    }, tmp)
    os.replace(tmp, path)  # atomic on same filesystem — old file survives if killed mid-write
    print(f"Saved checkpoint → {path}")


def load_checkpoint(model, optimizer, path, device, skip_optimizer=False):
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    own = model.state_dict()
    filtered, dropped = {}, []
    for k, v in state.items():
        if k in own and own[k].shape == v.shape:
            filtered[k] = v
        elif k in own:
            dropped.append(k)
    if dropped:
        print(f"[load] reinitialising {len(dropped)} shape-mismatched tensor(s): {dropped[:6]}")
    model.load_state_dict(filtered, strict=False)
    if not skip_optimizer:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception as e:
            print(f"[warn] optimizer state not loaded: {e}")
    step = int(ckpt.get("global_step", 0))
    print(f"Loaded V18 checkpoint at step={step}: {path}")
    return step


def _resolve_resume_path(resume_arg, auto_resume, ckpt_dir):
    if auto_resume or (isinstance(resume_arg, str) and resume_arg.lower() == "latest"):
        p = _find_latest_checkpoint(ckpt_dir)
        if p is None:
            raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
        return p
    if not resume_arg:
        return None
    if not os.path.isabs(resume_arg):
        cand = str(ROOT_DIR / resume_arg)
        if os.path.exists(cand):
            return cand
    return resume_arg


# ── Train step ───────────────────────────────────────────────────────────────

def train_one_step(model, batch, criterion, device, global_step, ssi_warmup, pi_floor, lambda_lat):
    model.train()
    rgb = batch["rgb"].to(device, non_blocking=True)
    loss_mask = batch.get("loss_mask")
    if loss_mask is None:
        raise KeyError("batch missing 'loss_mask'")
    loss_mask = loss_mask.to(device, non_blocking=True)
    m_diffuse = batch.get("M_diffuse", torch.zeros(rgb.shape[0])).float().to(device, non_blocking=True)
    m_residual = batch.get("m_residual", torch.ones(rgb.shape[0])).float().to(device, non_blocking=True)

    targets = compute_diffusion_targets(batch, device, pi_floor=pi_floor)
    targets["loss_mask"] = loss_mask

    use_ssi = global_step >= ssi_warmup
    decorr_scale = min(1.0, max(0.0, (global_step - ssi_warmup) / max(1, ssi_warmup)))

    with autocast(device_type="cuda", dtype=torch.float16):
        pred = model(rgb)
        pred = {k: (v.float() if isinstance(v, torch.Tensor) else v) for k, v in pred.items()}

    losses = criterion(
        predictions=pred, targets=targets, loss_mask=loss_mask,
        m_diffuse=m_diffuse, m_residual=m_residual, rgb=rgb,
        use_ssi=use_ssi, decorr_scale=decorr_scale,
    )

    # Latent anchor (small Marigold-style MSE on x0; stabilises early, keeps prior on-manifold)
    if lambda_lat > 0:
        with torch.no_grad():
            z_tgt = model.encode_targets(targets["A_d_star"], targets["pi_star"]).float()
        l_lat = F.mse_loss(pred["x0_latent"].float(), z_tgt)
        losses["loss_lat"] = lambda_lat * l_lat
        losses["loss_total"] = losses["loss_total"] + losses["loss_lat"]

    # Multi-illumination albedo invariance (paired MID, if present) — second one-step forward
    lam_inv = float(getattr(criterion, "lambda_alb_invariance", 0.0))
    rgb2 = batch.get("rgb2", None)
    m_invariant = batch.get("m_invariant", None)
    if lam_inv > 0 and rgb2 is not None and m_invariant is not None:
        m_invariant = m_invariant.float().to(device, non_blocking=True)
        if m_invariant.sum() > 0:
            rgb2 = rgb2.to(device, non_blocking=True)
            with autocast(device_type="cuda", dtype=torch.float16):
                a2 = model(rgb2)["a_d"].float()
            inv_mask = loss_mask.float() * m_invariant.view(-1, 1, 1, 1)
            l_inv = criterion._masked_l1(pred["a_d"], a2, inv_mask)
            losses["loss_alb_invariance"] = lam_inv * l_inv
            losses["loss_total"] = losses["loss_total"] + losses["loss_alb_invariance"]

    return losses


# ── V18 TensorBoard example strip ────────────────────────────────────────────

def _log_v18_examples(
    writer, global_step, rgb, predictions, targets,
    max_items=2, sample_index=None, example_root="3. Examples",
):
    """V18-specific TB strip (2 rows × 5 tiles).

    Row 1 GT:   Input RGB | Albedo GT | S_d GT | A_gt·S_d (diffuse GT) | |R_gt|
    Row 2 Pred: blank     | Albedo    | S_d    | A·S_d (pure diffuse)  | R (physics ≥0)

    Key differences from V17's _log_val_examples:
    - 'Recon' replaced by 'A·S_d' (diffuse-only, no R) — meaningful quality signal
      because V18's full recon ≈ input by construction (R absorbs the rest).
    - Residual column shows the non-negative physics fallback: how much R is needed.
    - Shared tonemap scale between GT and Pred for each component so magnitudes
      are directly comparable.
    """
    tag = f"{example_root}/sample_{sample_index}" if sample_index is not None else example_root
    writer.add_text(
        f"{tag}/layout",
        "GT: Input | Albedo | S_d | A·S_d | |R_gt|  "
        "Pred: blank | Albedo | S_d | A·S_d | R(physics≥0)",
        global_step,
    )

    def _gamma(x):
        return torch.pow(x.clamp(0.0, 1.0), 1.0 / 3.0)

    b = min(int(rgb.shape[0]), int(max_items))
    for i in range(b):
        v_mask = targets["loss_mask"][i : i + 1]

        # GT decomposition
        a_gt    = targets["A_d_star"][i : i + 1]
        s_d_gt  = (1.0 / targets["pi_star"][i : i + 1].clamp_min(1e-6) - 1.0).clamp_min(0.0)
        r_gt    = targets["R_star"][i : i + 1]
        diff_gt = a_gt * s_d_gt

        # Pred decomposition (R is the physics fallback clamped ≥0 in _decode_decompose)
        a_p     = predictions["a_d"][i : i + 1]
        s_d_p   = predictions["shading_linear"][i : i + 1]
        r_p     = predictions["residual"][i : i + 1]
        diff_p  = a_p * s_d_p

        # Shared scales for fair GT vs Pred comparison
        scale_a = torch.max(_get_tonemap_scale(a_p, valid_mask=v_mask),
                            _get_tonemap_scale(a_gt, valid_mask=v_mask))
        scale_s = torch.max(_get_tonemap_scale(s_d_p, valid_mask=v_mask),
                            _get_tonemap_scale(s_d_gt, valid_mask=v_mask))
        scale_d = torch.max(_get_tonemap_scale(diff_p, valid_mask=v_mask),
                            _get_tonemap_scale(diff_gt, valid_mask=v_mask))
        scale_r = torch.max(_get_tonemap_scale(r_p, valid_mask=v_mask),
                            _get_tonemap_scale(r_gt.abs(), valid_mask=v_mask))

        def _t(x, sc=None):
            return _gamma(_vis_tonemap(x, scale=sc, valid_mask=v_mask))

        blank = torch.ones_like(_t(rgb[i : i + 1]))

        row1 = torch.cat([
            _t(rgb[i : i + 1]),        # Input RGB (auto-scale)
            _t(a_gt,    sc=scale_a),   # Albedo GT
            _t(s_d_gt,  sc=scale_s),   # S_d GT
            _t(diff_gt, sc=scale_d),   # A_gt·S_d (pure diffuse GT)
            _t(r_gt.abs(), sc=scale_r),# |R_gt|  (signed residual magnitude)
        ], dim=2)

        row2 = torch.cat([
            blank,                      # separator (aligned with Input slot)
            _t(a_p,    sc=scale_a),    # Albedo Pred
            _t(s_d_p,  sc=scale_s),    # S_d Pred
            _t(diff_p, sc=scale_d),    # A·S_d (pure diffuse — the key diagnostic)
            _t(r_p,    sc=scale_r),    # R physics (≥0, shows model failure)
        ], dim=2)

        strip = torch.cat([row1, row2], dim=1)
        img_tag = (
            f"{tag}/index_{sample_index}" if sample_index is not None
            else f"{tag}/sample_{i}"
        )
        writer.add_image(img_tag, strip.cpu().clamp(0.0, 1.0), global_step)


# ── Validation (one-step) ─────────────────────────────────────────────────────

def validate(model, dataloader, device, global_step, writer, pi_floor,
             val_example_images=2, val_example_indices=None, max_val_batches=None,
             num_sample_steps=1, example_root="3. Examples"):
    model.eval()
    total = {k: 0.0 for k in ("a_lmse", "a_rmse", "a_ssim", "s_lmse", "s_rmse", "s_ssim")}
    n = 0
    val_example_indices = list(val_example_indices) if val_example_indices else []
    indices = set(int(i) for i in val_example_indices)
    use_indices = len(indices) > 0
    collected = {}
    limit = len(dataloader) if max_val_batches is None else min(max_val_batches, len(dataloader))

    torch.cuda.empty_cache()   # release allocator-cached pages before the eval pass
    with torch.no_grad():
        for bi, batch in enumerate(tqdm(dataloader, desc="Validation", total=limit)):
            if bi >= limit and not (use_indices and len(collected) < len(indices)):
                break
            rgb = batch["rgb"].to(device, non_blocking=True)
            loss_mask = batch["loss_mask"].to(device, non_blocking=True)
            tgt = compute_diffusion_targets(batch, device, pi_floor=pi_floor)
            A_gt, s_gt = tgt["A_d_star"], tgt["S_d_star"]

            with autocast(device_type="cuda", dtype=torch.float16):
                out = model.sample(rgb, num_steps=num_sample_steps)
            a_pred = out["a_d"].float()
            s_pred = out["shading_linear"].float()  # linear S_d, pre-computed in _decode_decompose
            B = rgb.shape[0]
            if bi < limit:
                total["a_lmse"] += _compute_lmse(a_pred, A_gt, loss_mask).item() * B
                total["a_rmse"] += _masked_scale_invariant_rmse(a_pred, A_gt, loss_mask).item() * B
                total["a_ssim"] += _compute_ssim_bounded(a_pred, A_gt, loss_mask) * B
                total["s_lmse"] += _compute_lmse(s_pred, s_gt, loss_mask).item() * B
                total["s_rmse"] += _masked_scale_invariant_rmse(s_pred, s_gt, loss_mask).item() * B
                total["s_ssim"] += _compute_shading_ssim(s_pred, s_gt, loss_mask) * B
                n += B

            def _pt(i0, i1):
                preds = {"a_d": a_pred[i0:i1], "shading_linear": s_pred[i0:i1],
                         "residual": out["residual"][i0:i1].float()}
                tgts = {"A_d_star": A_gt[i0:i1], "pi_star": tgt["pi_star"][i0:i1],
                        "R_star": tgt["R_star"][i0:i1], "loss_mask": loss_mask[i0:i1]}
                return preds, tgts

            if use_indices:
                sidx = batch.get("sample_idx", None)
                for i in range(B):
                    gidx = int(sidx[i].item()) if sidx is not None else int(bi * dataloader.batch_size + i)
                    if gidx in indices and gidx not in collected:
                        p, t = _pt(i, i + 1)
                        collected[gidx] = {"rgb": rgb[i:i+1], "predictions": p, "targets": t}
            elif bi == 0:
                p, t = _pt(0, val_example_images)
                _log_v18_examples(writer, global_step, rgb[:val_example_images], p, t,
                                  max_items=val_example_images, example_root=example_root)

    if use_indices and collected:
        for gidx in sorted(collected):
            s = collected[gidx]
            _log_v18_examples(writer, global_step, s["rgb"], s["predictions"], s["targets"],
                               max_items=1, sample_index=gidx, example_root=example_root)

    denom = max(n, 1)
    for k in total:
        total[k] /= denom
    for k, tag in VAL_METRIC_TAGS.items():
        writer.add_scalar(tag, float(total[k]), global_step)
    return total


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    config = load_config(args.config, args.version)
    print(yaml.dump(config, default_flow_style=False))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    torch.backends.cudnn.benchmark = True

    pv = args.version if args.version else config["model"].get("version", "18")
    ckpt_dir = os.path.join(config["paths"]["checkpoint_dir"], f"v{pv}")
    log_dir = os.path.join(config["paths"]["log_dir"], f"v{pv}")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # Model
    model_cfg = dict(config["model"])
    model_cfg["input_size"] = int(config["train"].get("input_size", 384))
    model = V18PGID(model_cfg).to(device)
    pi_floor = float(model.pi_floor)
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M "
          f"(U-Net + null embedding; VAE frozen)")
    if bool(config["train"].get("grad_checkpointing", True)):
        try:
            model.denoiser.unet.enable_gradient_checkpointing()
            print("Gradient checkpointing: ON")
        except Exception as e:
            print(f"[warn] grad checkpointing: {e}")

    # Loss = the project's image-space physics suite
    criterion = V17Loss(config["loss"]).to(device)
    lambda_lat = float(config["loss"].get("lambda_latent_anchor", 0.0))
    ssi_warmup = int(config["loss"].get("ssi_warmup_iters", 3000))

    base_lr = float(config["train"].get("lr", 3e-5))
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=base_lr,
        weight_decay=float(config["train"].get("weight_decay", 1e-2)))

    # Data (clean synthetic only; MID dropped via config weights)
    def _abs(r):
        return r if os.path.isabs(r) else str(ROOT_DIR / r)
    hypersim_root = _abs(config["data"]["hypersim_root"])
    midintrinsic_root = _abs(config["data"].get("midintrinsic_root", "../datasets/MIDIntrinsics"))
    interiorverse_root = _abs(config["data"].get("interiorverse_root", "../datasets/InteriorVerse"))
    split_file = config["data"].get("hypersim_split_file", "hypersim_split.json")
    split_seed = int(config["data"].get("hypersim_split_seed", 42))
    split_ratio = float(config["data"].get("hypersim_split_ratio", 0.9))
    strict_split = bool(config["data"].get("hypersim_strict_split", True))
    hdf5_retries = int(config["data"].get("hypersim_max_hdf5_retries", 1))
    skip_corrupt = bool(config["data"].get("hypersim_skip_corrupt_samples", True))

    from src.data.mixed_dataset import get_mixed_loader

    def _build_train_loader(weights_key):
        return get_mixed_loader(
            data_roots={"hypersim": hypersim_root, "midintrinsic": midintrinsic_root, "interiorverse": interiorverse_root},
            batch_size=int(config["train"]["batch_size"]), split="train",
            num_workers=int(config["train"].get("num_workers", 4)),
            input_size=int(config["train"]["input_size"]),
            cache_max_items=int(config["data"].get("cache_max_items", 512)),
            mix_weights=config["train"].get(weights_key, {"hypersim": 1.0}),
            split_file=split_file, split_seed=split_seed, split_ratio=split_ratio,
            strict_split=strict_split, max_hdf5_retries=hdf5_retries,
            skip_corrupt_samples=skip_corrupt,
            use_mid_paired=bool(config["data"].get("use_mid_paired", False)),
            load_geometry=False)

    def infinite(dl):
        while True:
            for b in dl:
                yield b

    train_loader = _build_train_loader("sampling_weights_phase1")
    train_iter = iter(infinite(train_loader))
    val_loader = get_hypersim_loader(
        root_dir=hypersim_root, batch_size=int(config["train"]["batch_size"]), split="val",
        num_workers=max(1, int(config["data"].get("val_num_workers", 2))),
        input_size=int(config["train"]["input_size"]),
        cache_max_items=int(config["data"].get("val_cache_max_items", 64)),
        split_file=split_file, split_seed=split_seed, split_ratio=split_ratio,
        strict_split=strict_split, max_hdf5_retries=hdf5_retries, skip_corrupt_samples=skip_corrupt,
        load_geometry=False)

    # Loop config
    max_iters = int(config["train"].get("extend_iterations", 30000))
    grad_accum = max(1, int(config["train"].get("grad_accum_steps", 1)))
    grad_clip = float(config["train"].get("grad_clip_max_norm", 1.0))
    use_cosine_lr = bool(config["train"].get("use_cosine_lr", True))
    lr_eta_min = float(config["train"].get("lr_eta_min", 1e-7))
    log_interval = int(config["train"].get("log_interval", 100))
    val_interval = int(config["train"].get("val_interval_iters", 2500))
    ckpt_interval = int(config["train"].get("checkpoint_interval_iters", 1000))
    val_imgs = int(config["train"].get("val_example_images", 2))
    val_idx = config["train"].get("val_example_indices", [])
    max_val_b = config["train"].get("max_val_batches", None)
    if max_val_b is not None:
        max_val_b = int(max_val_b)
    val_steps = int(config["train"].get("val_sample_steps", 1))
    phase1 = int(config["train"].get("phase1_iterations", -1))
    phase2 = int(config["train"].get("phase2_iterations", -1))

    # Resume
    start_step = 0
    resume_path = _resolve_resume_path(args.resume, args.auto_resume, ckpt_dir)
    if resume_path:
        skip_opt = args.skip_optimizer or bool(config["train"].get("skip_optimizer", False))
        start_step = load_checkpoint(model, optimizer, resume_path, device, skip_opt) + 1
        if args.reset_lr or bool(config["train"].get("reset_lr", False)):
            for pg in optimizer.param_groups:
                pg["lr"] = base_lr

    scheduler = None
    if use_cosine_lr:
        total_opt = max(1, math.ceil(max_iters / grad_accum))
        done_opt = max(0, start_step // grad_accum)
        for pg in optimizer.param_groups:
            pg.setdefault("initial_lr", pg["lr"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_opt, eta_min=lr_eta_min, last_epoch=done_opt - 1)

    print(f"Start step: {start_step}, max step: {max_iters}")
    scaler = GradScaler("cuda")
    running = {}

    def _phase(s):
        if s < phase1:
            return 1, phase1
        if s < phase2:
            return 2, phase2
        return 3, max_iters

    cur_phase, cur_end = _phase(start_step)
    pbar = tqdm(desc=f"Phase {cur_phase}", total=cur_end - start_step, dynamic_ncols=True)

    for step in range(start_step, max_iters):
        if step == cur_end:
            pbar.close()
            cur_phase, cur_end = _phase(step)
            pbar = tqdm(desc=f"Phase {cur_phase}", total=cur_end - step, dynamic_ncols=True)
        if step == phase1:
            print(f"--- Phase 2 data mix at step {step} ---")
            del train_iter, train_loader
            import gc; gc.collect()
            train_loader = _build_train_loader("sampling_weights_phase2")
            train_iter = iter(infinite(train_loader))
        if step == phase2:
            print(f"--- Phase 3 data mix at step {step} ---")
            del train_iter, train_loader
            import gc; gc.collect()
            train_loader = _build_train_loader("sampling_weights_phase3")
            train_iter = iter(infinite(train_loader))

        batch = next(train_iter)
        losses = train_one_step(model, batch, criterion, device, step, ssi_warmup, pi_floor, lambda_lat)
        loss = losses["loss_total"] / float(grad_accum)
        scaler.scale(loss).backward()

        if (step + 1) % grad_accum == 0 or step == max_iters - 1:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        pbar.set_postfix({"loss": f"{losses['loss_total'].item():.4f}"})
        pbar.update(1)
        for k, v in losses.items():
            running[k] = running.get(k, 0.0) + (v.item() if isinstance(v, torch.Tensor) else float(v))

        if step % log_interval == 0:
            _log_ordered_scalars(writer, losses, step, tag_prefix=None)
            writer.add_scalar("0. Training/lr", float(optimizer.param_groups[0]["lr"]), step)
            if "loss_lat" in losses:
                writer.add_scalar("1. Losses/Latent_Anchor", losses["loss_lat"].item(), step)

        if (step + 1) % ckpt_interval == 0:
            avg = {k: running[k] / ckpt_interval for k in running}
            for k in running:
                running[k] = 0.0
            save_checkpoint(model, optimizer, avg, config, os.path.join(ckpt_dir, f"checkpoint_iter_{step+1}.pth"), step)
            save_checkpoint(model, optimizer, avg, config, os.path.join(ckpt_dir, "checkpoint_latest.pth"), step)
            all_ckpts = sorted([os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir)
                                if f.startswith("checkpoint_iter_") and f.endswith(".pth")], key=_extract_iter_from_name)
            while len(all_ckpts) > 2:
                try:
                    os.remove(all_ckpts.pop(0))
                except Exception:
                    pass

        if (step + 1) % val_interval == 0:
            torch.cuda.empty_cache()
            m = validate(model, val_loader, device, step + 1, writer, pi_floor,
                         val_example_images=val_imgs, val_example_indices=val_idx,
                         max_val_batches=max_val_b, num_sample_steps=val_steps, example_root="3. Examples")
            print(f"[{step+1}] val: " + ", ".join(f"{k}={v:.4f}" for k, v in m.items()))

    pbar.close()
    print("Training completed.")
    writer.close()


if __name__ == "__main__":
    main()
