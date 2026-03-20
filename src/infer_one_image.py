"""
Run Stage-1 inference for one Hypersim frame at full resolution.

This script loads one frame (pred + derived GT), computes metrics, and saves:
- all predictions
- all derived ground truths
- scalar metrics
- quick visualization strips
"""

import argparse
import inspect
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.hypersim_dataset import HypersimDataset, _load_hdf5, _tonemap_linear
from models import (
    IntrinsicDecompositionV1,
    IntrinsicDecompositionV2,
    IntrinsicDecompositionV3,
    IntrinsicDecompositionV4,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Infer one Hypersim frame with full-size input")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to Stage-1 checkpoint")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Hypersim split")
    parser.add_argument("--hypersim_root", type=str, default=None, help="Optional Hypersim root override")
    parser.add_argument("--sample_idx", type=int, default=0, help="Sample index within split")
    parser.add_argument(
        "--match",
        type=str,
        default=None,
        help="Optional substring to match in color path (overrides --sample_idx)",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on")
    parser.add_argument(
        "--max_side",
        type=int,
        default=0,
        help="Optional cap for high-res pass: if > 0 and image side is larger, resize preserving aspect ratio",
    )
    parser.add_argument(
        "--global_pass_side",
        type=int,
        default=384,
        help="Optional global pass side cap. Set 0 to disable global pass.",
    )
    parser.add_argument("--output_dir", type=str, default="outputs/infer_one", help="Directory to save outputs")
    return parser.parse_args()


def _resolve_hypersim_root(args, config):
    cfg_root = config.get("data", {}).get("hypersim_root", "../datasets/hypersim")
    root = args.hypersim_root or cfg_root
    if not os.path.isabs(root):
        root = str(ROOT_DIR / root)
    return root


def _build_stage1_model(model_cfg):
    version = int(model_cfg.get("version", 1))
    model_config = {
        "z_channels": model_cfg.get("z_channels", 1024),
        "freeze_stages": model_cfg.get("freeze_stages", [1, 2]),
        "backbone": model_cfg.get("backbone", "convnextv2_base"),
        "pretrained": model_cfg.get("pretrained", True),
        "num_seg_classes": model_cfg.get("num_seg_classes", 41),
        # Input size is unused for full-size dynamic inference, but keep compatibility.
        "input_size": int(model_cfg.get("input_size", 1024)),
    }
    model_map = {
        1: IntrinsicDecompositionV1,
        2: IntrinsicDecompositionV2,
        3: IntrinsicDecompositionV3,
        4: IntrinsicDecompositionV4,
    }
    if version not in model_map:
        raise ValueError(f"Unsupported Stage1 version: {version}")
    return model_map[version](model_config)


def _select_sample(samples, sample_idx, match):
    if match:
        matched = [s for s in samples if match in s["color"]]
        if len(matched) == 0:
            raise ValueError(f"No sample matches substring: {match}")
        if len(matched) > 1:
            paths = [m["color"] for m in matched[:5]]
            raise ValueError(
                "Match is ambiguous. Provide a more specific --match. "
                f"First matches: {paths}"
            )
        return matched[0]

    if sample_idx < 0 or sample_idx >= len(samples):
        raise IndexError(f"sample_idx out of range: {sample_idx}, valid [0, {len(samples)-1}]")
    return samples[sample_idx]


def _to_chw_tensor(arr_hwc):
    return torch.from_numpy(arr_hwc.astype(np.float32)).permute(2, 0, 1)


def _prepare_one_sample(sample):
    rgb = _load_hdf5(sample["color"], retries=1)
    alb = _load_hdf5(sample["albedo"], retries=1)
    norm = _load_hdf5(sample["normal"], retries=1) if sample.get("normal") else np.zeros_like(rgb)
    seg = _load_hdf5(sample["seg"], retries=1) if sample.get("seg") else np.zeros(rgb.shape[:2], dtype=np.int32)
    rid = _load_hdf5(sample["rid"], retries=1) if sample.get("rid") else None

    if rid is not None:
        sky_mask = (rid != -1)
    else:
        sky_mask = np.ones(rgb.shape[:2], dtype=bool)
    alb_mask = alb.min(axis=-1) > 0.01
    valid = (sky_mask & alb_mask).astype(np.float32)

    rgb_tm = _tonemap_linear(rgb)

    out = {
        "rgb": _to_chw_tensor(rgb_tm).unsqueeze(0),
        "albedo_raw": _to_chw_tensor(alb).unsqueeze(0),
        "normals": _to_chw_tensor(norm).unsqueeze(0),
        "seg": torch.from_numpy(seg.astype(np.int64)).unsqueeze(0).unsqueeze(0),
        "valid_mask": torch.from_numpy(valid).unsqueeze(0).unsqueeze(0).bool(),
        "M_diffuse": torch.tensor([1.0], dtype=torch.float32),
        "M_albedo": torch.tensor([1.0], dtype=torch.float32),
    }
    return out


def _maybe_resize_batch(batch, max_side):
    if max_side <= 0:
        return batch, None

    _, _, h, w = batch["rgb"].shape
    current_max = max(h, w)
    if current_max <= max_side:
        return batch, None

    scale = float(max_side) / float(current_max)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    resized = dict(batch)
    for key in ["rgb", "albedo_raw", "normals"]:
        resized[key] = F.interpolate(batch[key], size=(new_h, new_w), mode="bilinear", align_corners=False)
    resized["valid_mask"] = F.interpolate(batch["valid_mask"].float(), size=(new_h, new_w), mode="nearest").bool()
    resized["seg"] = F.interpolate(batch["seg"].float(), size=(new_h, new_w), mode="nearest").long()

    msg = f"Applied fallback resize from {h}x{w} to {new_h}x{new_w} due to --max_side={max_side}"
    return resized, msg


def _resize_batch_to_hw(batch, new_h, new_w):
    resized = dict(batch)
    for key in ["rgb", "albedo_raw", "normals"]:
        resized[key] = F.interpolate(batch[key], size=(new_h, new_w), mode="bilinear", align_corners=False)
    resized["valid_mask"] = F.interpolate(batch["valid_mask"].float(), size=(new_h, new_w), mode="nearest").bool()
    resized["seg"] = F.interpolate(batch["seg"].float(), size=(new_h, new_w), mode="nearest").long()
    return resized


def _upsample_predictions_to_hw(predictions, out_h, out_w):
    out = {}
    for k, v in predictions.items():
        if v.ndim == 4 and v.shape[2] > 1 and v.shape[3] > 1:
            out[k] = F.interpolate(v, size=(out_h, out_w), mode="bilinear", align_corners=False)
        else:
            out[k] = v
    return out


def _try_forward_with_shape_fallback(model, batch):
    """Run forward once; if V1 bottleneck shape mismatch appears, resize and retry."""
    try:
        pred = model(
            batch["rgb"],
            **_forward_kwargs(model, batch["M_diffuse"], batch["normals"], batch["seg"]),
        )
        return pred, None
    except RuntimeError as exc:
        msg = str(exc)
        m = re.search(r"expected \((\d+),\s*(\d+)\)", msg)
        if "bottleneck mismatch" not in msg or m is None:
            raise

        # ConvNeXt encoder bottleneck is at /32 resolution.
        exp_h = int(m.group(1)) * 32
        exp_w = int(m.group(2)) * 32
        cur_h = int(batch["rgb"].shape[-2])
        cur_w = int(batch["rgb"].shape[-1])

        if cur_h == exp_h and cur_w == exp_w:
            raise

        resized = _resize_batch_to_hw(batch, exp_h, exp_w)
        pred = model(
            resized["rgb"],
            **_forward_kwargs(model, resized["M_diffuse"], resized["normals"], resized["seg"]),
        )
        reason = (
            f"Model requires fixed size due to adapter bottleneck shape; "
            f"resized from {cur_h}x{cur_w} to {exp_h}x{exp_w}."
        )
        return pred, (resized, reason)


def scale_match(A_raw, A_pred, valid):
    v = valid.expand_as(A_raw).float()
    a = (A_raw * v).reshape(A_raw.shape[0], -1)
    b = (A_pred * v).reshape(A_pred.shape[0], -1)
    c = (a * b).sum(dim=1) / ((a * a).sum(dim=1) + 1e-6)
    c = torch.clamp_min(c, 0.05)
    return c.view(-1, 1, 1, 1)


def compute_targets(predictions, rgb, albedo_raw, valid_mask):
    eps = 1e-6
    c = scale_match(albedo_raw, predictions["a_d"].detach(), valid_mask)

    A_star = c * albedo_raw
    S_star = rgb / (A_star + eps)

    S_g_star = 0.2126 * S_star[:, 0:1] + 0.7152 * S_star[:, 1:2] + 0.0722 * S_star[:, 2:3]
    D_g_star = 1.0 / (S_g_star + 1.0)

    C_RG = S_star[:, 0:1] / (S_star[:, 1:2] + eps)
    C_BG = S_star[:, 2:3] / (S_star[:, 1:2] + eps)
    xi_star = torch.cat([1.0 / (C_RG + 1.0), 1.0 / (C_BG + 1.0)], dim=1)

    return {
        "D_g_star": D_g_star,
        "xi_star": xi_star,
        "A_d_star": A_star,
        "pi_star": 1.0 / (S_star + 1.0),
    }


def _forward_kwargs(model, m_diffuse, normals, seg):
    sig = inspect.signature(model.forward).parameters
    kwargs = {}
    if "m_diffuse" in sig:
        kwargs["m_diffuse"] = m_diffuse
    if "normals" in sig and normals is not None:
        kwargs["normals"] = normals
    if "seg" in sig and seg is not None:
        kwargs["seg"] = seg
    return kwargs


def _apply_diffuse_detach(predictions, m_diffuse):
    mask = m_diffuse.view(-1, 1, 1, 1).to(predictions["s_d"].device)
    predictions["s_d"] = predictions["s_d"] * mask + predictions["s_d"].detach() * (1.0 - mask)
    return predictions


def _masked_scale_invariant_rmse(pred, target, valid_mask, eps=1e-7):
    """
    Scale-invariant RMSE with outlier masking.
    Normalizes by mean (not median) to match standard definition, but heavily relies on masking outliers.
    """
    mask = valid_mask.bool().expand_as(pred)

    p = pred[mask]
    t = target[mask]
    if p.numel() == 0:
        return 0.0
    p = p / (p.mean() + eps)
    t = t / (t.mean() + eps)
    return float(torch.sqrt(((p - t) ** 2).mean()).item())


def compute_ssim_bounded(pred, target, valid_mask):
    """SSIM for bounded outputs (e.g., albedo in [0,1])."""
    pred_norm = torch.clamp(pred, 0.0, 1.0)
    target_norm = torch.clamp(target, 0.0, 1.0)

    mask_f = valid_mask.float().expand_as(target_norm)
    pred_norm = pred_norm * mask_f
    target_norm = target_norm * mask_f

    fn_p = pred_norm.squeeze().detach().cpu().numpy()
    fn_t = target_norm.squeeze().detach().cpu().numpy()

    if fn_p.ndim == 3:
        vals = []
        for c in range(fn_p.shape[0]):
            vals.append(ssim(fn_t[c], fn_p[c], data_range=1.0))
        return float(np.mean(vals))
    return float(ssim(fn_t, fn_p, data_range=1.0))



def _compute_lmse(pred, target, valid_mask, window_size=20, stride=10, min_valid_ratio=0.5):
    """
    Standard Local Mean Squared Error (LMSE) with sliding window.
    Based on Grosse et al. (2009) definition: local windows are rescaled
    so that their mean squared magnitude is 1.
    """
    if pred.ndim == 3: pred = pred.unsqueeze(1)
    if target.ndim == 3: target = target.unsqueeze(1)
    if valid_mask.ndim == 3: valid_mask = valid_mask.unsqueeze(1)

    B, C, H, W = pred.shape
    if H < window_size or W < window_size:
        return torch.tensor(0.0, device=pred.device)

    # 1. Zero out invalid pixels in inputs
    pred = pred * valid_mask
    target = target * valid_mask

    # 2. Extract sliding windows
    unfold = torch.nn.Unfold(kernel_size=window_size, stride=stride)
    p_u = unfold(pred)        # (B, C*K*K, L)
    t_u = unfold(target)      # (B, C*K*K, L)
    m_u = unfold(valid_mask.float())  # (B, 1*K*K, L)

    # 3. Identify valid patches (at least 50% valid pixels)
    # Note: mask is single channel, valid for all channels
    k2 = window_size * window_size
    valid_count = m_u.sum(dim=1)  # (B, L)
    valid_patch_mask = (valid_count > (min_valid_ratio * k2))  # (B, L)

    if not valid_patch_mask.any():
        return torch.tensor(0.0, device=pred.device)

    # Revert to fixed N but ensure valid_patch_mask filters properly
    N = float(C * k2)
    sqrt_N = N ** 0.5

    # Only normalize patches where BOTH pred and target have sufficient signal
    p_energy = torch.norm(p_u, p=2, dim=1, keepdim=True)
    t_energy = torch.norm(t_u, p=2, dim=1, keepdim=True)

    # Skip patches where either has near-zero energy (all-invalid or flat)
    min_energy = 1e-3
    has_signal = (p_energy.squeeze(1) > min_energy) & \
                 (t_energy.squeeze(1) > min_energy) & \
                 valid_patch_mask

    if not has_signal.any():
        return torch.tensor(0.0, device=pred.device)

    p_sc = (p_u / (p_energy + 1e-7)) * sqrt_N
    t_sc = (t_u / (t_energy + 1e-7)) * sqrt_N

    # 5. Compute MSE per patch
    # MSE = mean((p - t)^2)
    diff_sq = (p_sc - t_sc) ** 2
    mse_per_patch = diff_sq.mean(dim=1)  # (B, L)

    # 6. Average over valid patches
    # Flatten batch and L
    mse_flat = mse_per_patch[has_signal]
    return mse_flat.mean()



def _tonemap_vis(x):
    if x.ndim == 4:
        x = x[0]
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    scale = torch.quantile(torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).reshape(-1), 0.99).clamp_min(1e-6)
    return torch.clamp(x / scale, 0.0, 1.0)


def _save_visual_strip(rgb, preds, gts, out_png):
    import torchvision.utils as vutils
    from PIL import Image, ImageDraw, ImageFont

    def gamma_correct(x):
        return torch.pow(torch.clamp(x, min=0.0, max=1.0), 1.0 / 3)

    # 1. Albedo via Gamma Correction
    a_d_pred_vis = gamma_correct(preds["a_d"])
    a_d_gt_vis = gamma_correct(gts["A_d_star"])

    # 2. Shading via Inverse Domain Tonemapping (1 - D)
    s_g_pred_vis = 1.0 - preds["d_g"]
    s_g_gt_vis = 1.0 - gts["D_g_star"]
    s_d_pred_vis = 1.0 - preds["s_d"]
    s_d_gt_vis = 1.0 - gts["pi_star"]

    # 3. Colorful Shading (S_c) GT and Pred via Inverse Domain
    s_g_gt_linear = 1.0 / (gts["D_g_star"] + 1e-6) - 1.0
    c_rg_gt = 1.0 / (gts["xi_star"][:, 0:1] + 1e-6) - 1.0
    c_bg_gt = 1.0 / (gts["xi_star"][:, 1:2] + 1e-6) - 1.0
    s_c_gt_linear = torch.cat([c_rg_gt * s_g_gt_linear, s_g_gt_linear, c_bg_gt * s_g_gt_linear], dim=1)
    s_c_gt_vis = 1.0 - (1.0 / (s_c_gt_linear + 1.0))

    s_g_pred_linear = 1.0 / (preds["d_g"] + 1e-6) - 1.0
    if "s_c" in preds:
        s_c_pred_linear = 1.0 / (preds["s_c"] + 1e-6) - 1.0
    elif "xi" in preds:
        c_rg_pred = 1.0 / (preds["xi"][:, 0:1] + 1e-6) - 1.0
        c_bg_pred = 1.0 / (preds["xi"][:, 1:2] + 1e-6) - 1.0
        s_c_pred_linear = torch.cat([c_rg_pred * s_g_pred_linear, s_g_pred_linear, c_bg_pred * s_g_pred_linear], dim=1)
    else:
        s_c_pred_linear = s_g_pred_linear.repeat(1, 3, 1, 1)

    s_c_pred_vis = 1.0 - (1.0 / (s_c_pred_linear + 1.0))

    # 4. & 5. Residuals: Positive (Specularity) and Negative (Saturation)
    s_d_pred_linear = 1.0 / (preds["s_d"] + 1e-6) - 1.0
    recon_diffuse = preds["a_d"] * s_d_pred_linear
    pos_res_pred = torch.clamp(rgb - recon_diffuse, min=0.0)
    neg_res_pred = torch.clamp(recon_diffuse - rgb, min=0.0)

    # Note: s_g, s_d, s_c and a_d are already transformed to [0,1].
    # Residuals and original RGB are still in linear unbounded space, so use _tonemap_vis
    titled_tiles = [
        ("Input tonemapped RGB", _tonemap_vis(rgb)),
        ("S_g Pred", torch.clamp(s_g_pred_vis, 0.0, 1.0)),
        ("S_g GT", torch.clamp(s_g_gt_vis, 0.0, 1.0)),
        ("Colorful S_c Pred", torch.clamp(s_c_pred_vis, 0.0, 1.0)),
        ("Colorful S_c GT", torch.clamp(s_c_gt_vis, 0.0, 1.0)),
        ("Albedo Pred", _tonemap_vis(a_d_pred_vis)),
        ("Albedo GT", _tonemap_vis(a_d_gt_vis)),
        ("S_d Pred", torch.clamp(s_d_pred_vis, 0.0, 1.0)),
        ("S_d GT", torch.clamp(s_d_gt_vis, 0.0, 1.0)),
        ("Positive Residual (R+)", _tonemap_vis(pos_res_pred)),
        ("Negative Residual (R-)", _tonemap_vis(neg_res_pred)),
    ]

    tiles = []
    footer_h = 80
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 60)
    except OSError:
        font = ImageFont.load_default()
    for title, tile in titled_tiles:
        tile = tile if tile.ndim == 3 else tile[0]
        c, h, w = tile.shape
        if c == 1:
            tile = tile.repeat(3, 1, 1)
            c = 3
        canvas = torch.ones((c, h + footer_h, w), dtype=tile.dtype, device=tile.device)
        canvas[:, :h, :] = tile

        # Render title text onto footer using PIL for reliable text drawing.
        canvas_np = (canvas.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        img = Image.fromarray(canvas_np)
        draw = ImageDraw.Draw(img)
        text_w = draw.textlength(title, font=font)
        x = max(0, int((w - text_w) // 2))
        y = h + 10
        draw.text((x, y), title, fill=(0, 0, 0), font=font)

        tile_labeled = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).permute(2, 0, 1)
        tiles.append(tile_labeled)

    grid = vutils.make_grid(tiles, nrow=len(tiles), padding=4, pad_value=1.0)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    vutils.save_image(grid, out_png)


def main():
    args = parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint.get("config", {})
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = _build_stage1_model(config.get("model", {})).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    hypersim_root = _resolve_hypersim_root(args, config)
    dataset = HypersimDataset(
        root_dir=hypersim_root,
        split=args.split,
        input_size=384,
        cache_max_items=8,
        crop_mode_train="random",
        crop_mode_val="full",
        split_file=config.get("data", {}).get("hypersim_split_file", "hypersim_split.json"),
        split_seed=int(config.get("data", {}).get("hypersim_split_seed", 42)),
        split_ratio=float(config.get("data", {}).get("hypersim_split_ratio", 0.9)),
        strict_split=bool(config.get("data", {}).get("hypersim_strict_split", True)),
    )

    sample = _select_sample(dataset.samples, args.sample_idx, args.match)
    batch_full_res_cpu = _prepare_one_sample(sample)
    _, _, h_full, w_full = batch_full_res_cpu["rgb"].shape

    predictions_global_up = None
    global_resize_msg = None
    model_global_resize_msg = None

    if args.global_pass_side > 0:
        batch_global_cpu, global_resize_msg = _maybe_resize_batch(batch_full_res_cpu, args.global_pass_side)
        batch_global = {
            k: v.to(device)
            for k, v in batch_global_cpu.items()
        }
        with torch.no_grad():
            global_out, model_resize = _try_forward_with_shape_fallback(model, batch_global)
            if model_resize is not None:
                batch_global, model_global_resize_msg = model_resize
            global_out = _apply_diffuse_detach(global_out, batch_global["M_diffuse"])
            predictions_global_up = _upsample_predictions_to_hw(global_out, h_full, w_full)

    highres_resize_msg = None
    model_highres_resize_msg = None
    predictions_highres_up = None
    pred_source = None

    batch_highres_cpu, highres_resize_msg = _maybe_resize_batch(batch_full_res_cpu, args.max_side)
    batch_highres = {
        k: v.to(device)
        for k, v in batch_highres_cpu.items()
    }

    try:
        with torch.no_grad():
            highres_out, model_resize = _try_forward_with_shape_fallback(model, batch_highres)
            if model_resize is not None:
                batch_highres, model_highres_resize_msg = model_resize
            highres_out = _apply_diffuse_detach(highres_out, batch_highres["M_diffuse"])
            predictions_highres_up = _upsample_predictions_to_hw(highres_out, h_full, w_full)
            pred_source = "high_res"
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        if predictions_global_up is None:
            raise
        if device.type == "cuda":
            torch.cuda.empty_cache()
        pred_source = "global_fallback_oom"

    if predictions_highres_up is not None:
        predictions = predictions_highres_up
    elif predictions_global_up is not None:
        predictions = predictions_global_up
        if pred_source is None:
            pred_source = "global_only"
    else:
        raise RuntimeError("No predictions available. Enable --global_pass_side or use a model/input size that supports high-res pass.")

    batch = {
        k: v.to(device)
        for k, v in batch_full_res_cpu.items()
    }

    with torch.no_grad():
        targets = compute_targets(predictions, batch["rgb"], batch["albedo_raw"], batch["valid_mask"])

    # Compute metrics in inverse shading space (bounded).
    d_g_pred = predictions["d_g"]
    pi_pred = predictions["s_d"]
    d_g_gt = targets["D_g_star"]
    pi_gt = targets["pi_star"]

    valid_mask = batch["valid_mask"]

    metrics = {
        "s_g_lmse": float(_compute_lmse(d_g_pred, d_g_gt, valid_mask).item()),
        "s_g_rmse": _masked_scale_invariant_rmse(d_g_pred, d_g_gt, valid_mask),
        "s_g_ssim": compute_ssim_bounded(d_g_pred[0], d_g_gt[0], valid_mask[0]),
        "a_d_lmse": float(_compute_lmse(predictions["a_d"], targets["A_d_star"], valid_mask).item()),
        "a_d_rmse": _masked_scale_invariant_rmse(predictions["a_d"], targets["A_d_star"], valid_mask),
        "a_d_ssim": compute_ssim_bounded(predictions["a_d"][0], targets["A_d_star"][0], valid_mask[0]),
        "s_d_lmse": float(_compute_lmse(pi_pred, pi_gt, valid_mask).item()),
        "s_d_rmse": _masked_scale_invariant_rmse(pi_pred, pi_gt, valid_mask),
        "s_d_ssim": compute_ssim_bounded(pi_pred[0], pi_gt[0], valid_mask[0]),
        "xi_mse": float((((predictions["xi"] - targets["xi_star"]) ** 2) * valid_mask.expand_as(predictions["xi"]).float()).sum().item() / (valid_mask.expand_as(predictions["xi"]).float().sum().item() + 1e-7)),
    }

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    _save_visual_strip(batch["rgb"], predictions, targets, os.path.join(out_dir, "pred_gt_strip.png"))

    print("Single-image inference completed.")
    print(f"Output dir: {out_dir}")
    print(f"Sample: {sample['color']}")
    print(f"Prediction source: {pred_source}")
    if global_resize_msg:
        print(f"Global resize: {global_resize_msg}")
    if model_global_resize_msg:
        print(f"Global model resize: {model_global_resize_msg}")
    if highres_resize_msg:
        print(f"High-res resize: {highres_resize_msg}")
    if model_highres_resize_msg:
        print(f"High-res model resize: {model_highres_resize_msg}")
    print("Metrics:")
    for k in [
        "s_g_lmse", "s_g_rmse", "s_g_ssim",
        "a_d_lmse", "a_d_rmse", "a_d_ssim",
        "s_d_lmse", "s_d_rmse", "s_d_ssim",
        "xi_mse",
    ]:
        print(f"  {k}: {metrics[k]:.6f}")


if __name__ == "__main__":
    main()

# python src/infer_one_image.py --checkpoint checkpoints/v1/checkpoint_latest.pth --split val --sample_idx 0 --device cpu --output_dir outputs/infer_one_smoke