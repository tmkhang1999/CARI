"""
Training script for Stage 1 intrinsic decomposition.
"""

import argparse
import math
import inspect
import os
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import kornia
import yaml
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from torch.amp import autocast, GradScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models import (
    IntrinsicDecompositionV12,
    IntrinsicDecompositionV16,
    IntrinsicDecompositionV17,
)
from models.ccr_utils import compute_ccr
from models.iid_utils import invert, iuv_to_rgb, rgb_to_iuv
from losses.flexible_loss_v17 import V17Loss
from data.hypersim_dataset import HypersimDataset, get_hypersim_loader


TB_TAGS = {
    'loss_total': '1. Losses/Total_all',
    'loss_a': '1. Losses/A_total',
    'loss_c_l1': '1. Losses/A_MSE',
    'loss_c_msg': '1. Losses/A_MSG',
    'loss_c_dssim': '1. Losses/A_DSSIM',
    'loss_s': '1. Losses/S_total',
    'loss_shading_mse': '1. Losses/S_MSE',
    'loss_shading_msg': '1. Losses/S_MSG',
    'loss_r': '1. Losses/R_total',
    'loss_residual_mse': '1. Losses/R_MSE',
    'loss_residual_msg': '1. Losses/R_MSG',
    'loss_recon': '1. Losses/Recon_total',
    'loss_recon_l1': '1. Losses/Recon_L1',
    'loss_recon_msg': '1. Losses/Recon_MSG',
}

def _log_ordered_scalars(writer, values, global_step, tag_prefix=None):
    """Write only requested TensorBoard tags in deterministic order."""
    ordered = [
        'loss_total',
        'loss_a', 'loss_c_l1', 'loss_c_msg', 'loss_c_dssim',
        'loss_s', 'loss_shading_mse', 'loss_shading_msg',
        'loss_r', 'loss_residual_mse', 'loss_residual_msg',
        'loss_recon', 'loss_recon_l1', 'loss_recon_msg',
    ]
    for key in ordered:
        if key not in values:
            continue
        val = values[key]
        if isinstance(val, torch.Tensor):
            val = val.item()
        tag = TB_TAGS[key]
        if tag_prefix:
            tag = f"{tag_prefix}/{tag}"
        writer.add_scalar(tag, float(val), global_step)


def scale_match(A_raw, A_pred, valid, seg=None, stable_classes=[1, 2, 22]):
    """Least-squares per-image scalar c for c*A_raw ~= A_pred over valid pixels."""
    v = valid.expand_as(A_raw).float()
    
    if seg is not None and stable_classes:
        if seg.dim() == 4: seg = seg[:, 0]
        target = torch.tensor(stable_classes, device=seg.device)
        struct_mask = torch.isin(seg, target).unsqueeze(1).expand_as(A_raw).float()
        # Only use structural pixels if enough exist (>5% of image)
        struct_ratio = struct_mask.sum() / (v.sum() + 1e-7)
        if struct_ratio > 0.05:
            v = v * struct_mask

    a = (A_raw * v).reshape(A_raw.shape[0], -1)
    b = (A_pred * v).reshape(A_pred.shape[0], -1)
    c = (a * b).sum(dim=1) / ((a * a).sum(dim=1) + 1e-6)
    # Prevent degenerate target scaling that can zero-out Dec-C supervision.
    c = torch.clamp_min(c, 0.05)
    return c.reshape(-1, 1, 1, 1)


def compute_targets(predictions, batch):
    eps = 1e-6
    device = predictions['a_d'].device
    rgb = batch['rgb'].to(device)
    albedo_target = batch['albedo_raw'].to(device)

    loss_mask = batch.get('loss_mask', None)
    if loss_mask is None:
        raise KeyError("batch is missing required 'loss_mask'")
    loss_mask = loss_mask.to(device)
    seg = batch.get('seg', None)
    if seg is not None:
        seg = seg.to(device)
    
    # Scale matching parameter C is computed between predictions and truth
    c = scale_match(albedo_target, predictions['a_d'].detach(), loss_mask, seg)

    A_star = c * albedo_target
    A_star = torch.nan_to_num(A_star, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)

    # Colorful shading (S_c) is directly derived from RGB and Albedo: I / A*
    A_safe = A_star.clamp_min(1e-4) 
    S_c_star = rgb / A_safe
    S_c_star = torch.nan_to_num(S_c_star, nan=0.0, posinf=60000.0, neginf=0.0).clamp_min(0.0)

    # Diffuse shading target (S_d_star) routes from direct illumination GT if available
    S_d_star = S_c_star
    illum_raw = batch.get('illum_raw', None)
    m_diffuse = batch.get('M_diffuse', None)
    
    if illum_raw is not None and m_diffuse is not None:
        route = m_diffuse.reshape(-1, 1, 1, 1).to(device=rgb.device, dtype=rgb.dtype)
        illum = torch.nan_to_num(illum_raw.to(device), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        # --- CRITICAL PHYSICS FIX ---
        # Because A_star is scaled by 'c', S_d_star MUST be scaled by '1/c'
        # This guarantees that A_star * S_d_star = I
        illum_scaled = illum / c.clamp_min(1e-4)
        S_d_star = route * illum_scaled + (1.0 - route) * S_c_star

    # Target Residual R_star = I - A*S_d
    R_star = rgb - (A_star * S_d_star)
    R_star = torch.nan_to_num(R_star, nan=0.0, posinf=0.0, neginf=0.0)
    
    return {
        'A_d_star': A_star,
        'S_d_star': S_d_star,
        'R_star': R_star,
        'loss_mask': loss_mask,
    }



def parse_args():
    parser = argparse.ArgumentParser(description='Train Stage 1')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--version', type=str, default='14')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint path or "latest"')
    parser.add_argument('--auto-resume', action='store_true', help='Resume from latest checkpoint in version checkpoint dir')
    parser.add_argument('--reset-lr', action='store_true', help='Override checkpoint LR with value from config')
    parser.add_argument('--skip-optimizer', action='store_true', help='Resume from checkpoint but skip loading optimizer state')
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


def _deep_merge(base, override):
    merged = base.copy()
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_config(config_path=None, version=None):
    base_path = SRC_DIR / 'configs' / 'base.yaml'
    with open(base_path, 'r') as f:
        config = yaml.safe_load(f)

    if config_path is None and version is not None:
        config_path = str(SRC_DIR / 'configs' / f'v{version}.yaml')

    if config_path is not None and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            override = yaml.safe_load(f) or {}
        config = _deep_merge(config, override)
        print(f"Config: base.yaml <- {os.path.basename(config_path)}")
    else:
        print("Config: base.yaml only")
    return config


def save_checkpoint(model, optimizer, losses, config, filename, global_step):
    ckpt = {
        'global_step': int(global_step),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
        'config': config,
    }
    torch.save(ckpt, filename)
    print(f"Saved checkpoint to {filename}")


def load_checkpoint(model, optimizer, checkpoint_path, map_location=None, skip_optimizer=False):
    ckpt = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    
    if not skip_optimizer:
        try:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        except Exception as exc:
            print(
                "[warn] optimizer state not loaded (param-group mismatch). "
                f"Continuing with freshly initialized optimizer: {exc}"
            )
    else:
        print("[info] Skipping optimizer state loading as requested.")

    global_step = int(ckpt.get('global_step', 0))
    losses = ckpt.get('losses', {})
    print(f"Loaded checkpoint at global_step={global_step}: {checkpoint_path}")
    return global_step, losses


def _extract_iter_from_name(path):
    m = re.search(r'checkpoint_iter_(\d+)\.pth$', os.path.basename(path))
    return int(m.group(1)) if m else -1


def _find_latest_checkpoint(ckpt_dir):
    if not os.path.isdir(ckpt_dir):
        return None
    files = [
        os.path.join(ckpt_dir, f)
        for f in os.listdir(ckpt_dir)
        if f.startswith('checkpoint_iter_') and f.endswith('.pth')
    ]
    if not files:
        fallback = os.path.join(ckpt_dir, 'checkpoint_latest.pth')
        if os.path.isfile(fallback):
            return fallback
        return None
    files.sort(key=_extract_iter_from_name)
    return files[-1]


def _resolve_resume_path(resume_arg, auto_resume, ckpt_dir):
    if auto_resume or (isinstance(resume_arg, str) and resume_arg.lower() == 'latest'):
        latest = _find_latest_checkpoint(ckpt_dir)
        if latest is None:
            raise FileNotFoundError(
                f"Resume requested but no checkpoint found in: {ckpt_dir}"
            )
        return latest

    if not resume_arg:
        return None

    # Resolve relative paths against project root for convenience.
    if not os.path.isabs(resume_arg):
        candidate = str(ROOT_DIR / resume_arg)
        if os.path.exists(candidate):
            return candidate
    return resume_arg


def _forward_kwargs(model, m_diffuse, normals, seg, valid_mask, ccr=None):
    sig = inspect.signature(model.forward).parameters
    kwargs = {}
    if 'm_diffuse' in sig:
        kwargs['m_diffuse'] = m_diffuse
    if 'normals' in sig and normals is not None:
        kwargs['normals'] = normals
    if 'seg' in sig and seg is not None:
        kwargs['seg'] = seg
    if 'valid_mask' in sig and valid_mask is not None:
        kwargs['valid_mask'] = valid_mask
    if 'ccr' in sig and ccr is not None:
        kwargs['ccr'] = ccr
    return kwargs





def _loss_seg(model, seg):
    if 'seg' not in inspect.signature(model.forward).parameters:
        return None
    if seg is not None and seg.dim() == 4 and seg.shape[1] == 1:
        return seg[:, 0]
    return seg


def train_one_step(model, batch, criterion, device, global_step, ssi_warmup_iters):
    model.train()

    rgb = batch['rgb'].to(device, non_blocking=True)
    albedo_raw = batch['albedo_raw'].to(device, non_blocking=True)
    illum_raw = batch.get('illum_raw', None)
    if illum_raw is not None:
        illum_raw = illum_raw.to(device, non_blocking=True)
    loss_mask = batch.get('loss_mask', None)
    if loss_mask is None:
        raise KeyError("batch is missing required 'loss_mask'")
    loss_mask = loss_mask.to(device, non_blocking=True)
    m_diffuse = batch.get('M_diffuse', torch.zeros(rgb.shape[0])).float().to(device, non_blocking=True)
    m_residual = batch.get('m_residual', torch.ones(rgb.shape[0])).float().to(device, non_blocking=True)
    
    seg = batch.get('seg', None)
    if seg is not None:
        seg = seg.to(device, non_blocking=True)
    normals = batch.get('normals', None)
    if normals is not None:
        normals = normals.to(device, non_blocking=True)

    with torch.no_grad():
        ccr = compute_ccr(rgb)

    with autocast(device_type='cuda', dtype=torch.float16):
        predictions = model(rgb, **_forward_kwargs(model, m_diffuse, normals, seg, loss_mask, ccr))
        
        # Ensure predictions are float32 for stable loss and target computation
        predictions = {k: v.float() if isinstance(v, torch.Tensor) else v for k, v in predictions.items()}
        
    targets = compute_targets(predictions, batch)

    use_ssi = (global_step >= ssi_warmup_iters)

    losses = criterion(
        predictions=predictions,
        targets=targets,
        loss_mask=loss_mask,
        m_residual=m_residual,
        rgb=rgb,
        use_ssi=use_ssi,
    )

    return losses


def _compute_lmse(pred, target, valid_mask, window_size=20, stride=10):
    """
    Local Mean Squared Error (LMSE) aligned with chrislib benchmark.
    
    Key difference from naive per-channel: for RGB inputs (C=3), chrislib
    concatenates all channels into one 2D array per patch and computes ONE
    joint alpha. For gray inputs (C=1), it's equivalent to per-channel.
    
    This matches chrislib's lmse_rgb / lmse_gray exactly.
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
    p_u = unfold(pred)                # (B, C*K*K, L)
    t_u = unfold(target)              # (B, C*K*K, L)
    m_u = unfold(valid_mask.float())  # (B, 1*K*K, L)
    
    # 3. Reshape to separate Channels (C) from Spatial Pixels (K*K)
    k2 = window_size * window_size
    L = p_u.shape[-1]
    p_u = p_u.view(B, C, k2, L)       # (B, C, K*K, L)
    t_u = t_u.view(B, C, k2, L)       # (B, C, K*K, L)
    m_u = m_u.view(B, 1, k2, L)       # (B, 1, K*K, L)

    # 4. Compute ONE joint alpha across all channels per patch
    # chrislib concatenates [R, G, B] into a single 2D array, then calls
    # ssq_error which computes: alpha = sum(correct * estimate * mask) / sum(estimate^2 * mask)
    # This is equivalent to summing over both spatial (K*K) AND channel (C) dims.
    m_expanded = m_u.expand_as(p_u)    # (B, C, K*K, L) — same mask for all channels
    num = (t_u * p_u * m_expanded).sum(dim=(1, 2))   # (B, L) — joint over C and K*K
    den = (p_u ** 2 * m_expanded).sum(dim=(1, 2))     # (B, L)

    den_safe = torch.where(den > 1e-5, den, torch.ones_like(den))
    alpha = torch.where(den > 1e-5, num / den_safe, torch.zeros_like(den))
    alpha = alpha.unsqueeze(1).unsqueeze(2)            # (B, 1, 1, L) — broadcasts over C and K*K

    # 5. Compute sum squared error per patch (joint over all channels)
    diff = t_u - alpha * p_u
    diff_sq = (diff ** 2) * m_expanded
    ssq_per_patch = diff_sq.sum(dim=(1, 2))            # (B, L)

    # 6. Compute total actual squared target (joint over all channels)
    total_per_patch = ((t_u ** 2) * m_expanded).sum(dim=(1, 2))  # (B, L)

    # 7. Sum over ALL patches per image (no patch filtering)
    ssq_sum = ssq_per_patch.sum(dim=1)                 # (B,)
    total_sum = total_per_patch.sum(dim=1)             # (B,)

    # Final division with safety net
    total_safe = torch.where(total_sum > 1e-7, total_sum, torch.ones_like(total_sum))
    lmse_val = torch.where(total_sum > 1e-7, ssq_sum / total_safe, torch.zeros_like(total_sum))

    return lmse_val.mean()


def _masked_rmse(pred, target, valid_mask, eps=1e-7):
    """
    Standard RMSE matching chrislib/metrics.py rmse_error implementation.
    """
    if pred.ndim == 3: pred = pred.unsqueeze(1)
    if target.ndim == 3: target = target.unsqueeze(1)
    if valid_mask.ndim == 3: valid_mask = valid_mask.unsqueeze(1)

    mask = valid_mask.bool().expand_as(pred)

    diff_sq = ((pred - target) ** 2) * mask.float()
    valid_counts = mask.float().reshape(pred.shape[0], -1).sum(dim=1).clamp_min(1)
    rmse_per_image = torch.sqrt(diff_sq.reshape(pred.shape[0], -1).sum(dim=1) / valid_counts)
    
    return rmse_per_image.mean()


def _masked_scale_invariant_rmse(pred, target, valid_mask, eps=1e-7):
    """
    Scale-invariant RMSE.
    Matches infer_hypersim.py for scale-ambiguous targets.
    """
    if pred.ndim == 3: pred = pred.unsqueeze(1)
    if target.ndim == 3: target = target.unsqueeze(1)
    if valid_mask.ndim == 3: valid_mask = valid_mask.unsqueeze(1)

    mask_f = valid_mask.float().expand_as(pred)
    
    pred_masked = pred * mask_f
    target_masked = target * mask_f
    
    valid_counts = mask_f.reshape(pred.shape[0], -1).sum(dim=1).clamp_min(1)
    p_mean = pred_masked.reshape(pred.shape[0], -1).sum(dim=1) / valid_counts
    t_mean = target_masked.reshape(target.shape[0], -1).sum(dim=1) / valid_counts
    
    # Expand means to match spatial dimensions
    p_mean_exp = p_mean.view(-1, 1, 1, 1)
    t_mean_exp = t_mean.view(-1, 1, 1, 1)
    
    # Normalize
    p_norm = pred / (p_mean_exp + eps)
    t_norm = target / (t_mean_exp + eps)
    
    diff_sq = ((p_norm - t_norm) ** 2) * mask_f
    rmse_per_image = torch.sqrt(diff_sq.reshape(pred.shape[0], -1).sum(dim=1) / valid_counts)
    
    return rmse_per_image.mean()


def _ssim_from_dssim(criterion, pred, target):
    """Convert criterion DSSIM helper to SSIM in [0,1]."""
    dssim = criterion._compute_dssim(pred, target)
    return 1.0 - 2.0 * dssim


def _make_gaussian_window(win_size, sigma, channels, device, dtype):
    """Create a Gaussian window matching skimage SSIM defaults."""
    coords = torch.arange(win_size, dtype=dtype, device=device) - win_size // 2
    gauss_1d = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    gauss_1d = gauss_1d / gauss_1d.sum()
    gauss_2d = gauss_1d.unsqueeze(1) @ gauss_1d.unsqueeze(0)
    return gauss_2d.unsqueeze(0).unsqueeze(0).expand(channels, 1, -1, -1).contiguous()


def _compute_shading_ssim(pred, target, valid_mask, eps=1e-6, shading_cap=None):
    """Batch-safe SSIM for unbounded shading with per-image LS alignment + robust scale."""
    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
        valid_mask = valid_mask.unsqueeze(0)

    B = pred.shape[0]
    mask = valid_mask.bool().expand_as(pred)
    if shading_cap is not None:
        mask = mask & (target <= shading_cap)

    p_masked = pred * mask.float()
    t_masked = target * mask.float()

    # 1. Per-image optimal alpha (batched least-squares)
    p_flat = p_masked.reshape(B, -1)
    t_flat = t_masked.reshape(B, -1)
    num = (t_flat * p_flat).sum(dim=1)
    den = (p_flat ** 2).sum(dim=1)
    alpha = torch.where(den > 1e-5, num / den.clamp_min(1e-5), torch.zeros_like(den))
    pred_aligned = pred * alpha.view(B, 1, 1, 1)

    # 2. Per-image robust scale (mean + 2*std of valid GT pixels)
    valid_counts = mask.float().reshape(B, -1).sum(dim=1).clamp_min(1.0)
    t_mean = t_flat.sum(dim=1) / valid_counts
    t_sq_mean = (t_flat ** 2).sum(dim=1) / valid_counts
    t_var = (t_sq_mean - t_mean ** 2).clamp_min(0.0)
    t_std = torch.sqrt(t_var)
    scale = (t_mean + 2.0 * t_std).clamp_min(eps).view(B, 1, 1, 1)

    # 3. Normalize into [0,1] and mask
    pred_n = torch.clamp(pred_aligned / scale, 0.0, 1.0) * mask.float()
    tgt_n = torch.clamp(target / scale, 0.0, 1.0) * mask.float()

    return _pytorch_ssim_skimage(pred_n, tgt_n, data_range=1.0)


def _pytorch_ssim_skimage(pred, target, data_range=1.0, win_size=11, sigma=1.5):
    """GPU SSIM matching skimage.metrics.structural_similarity defaults.
    
    Uses Gaussian window (sigma=1.5, win_size=11) with Bessel-corrected
    variance, exactly matching skimage's implementation.
    """
    # Ensure batched format
    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
        
    B, C, H, W = pred.shape
    if H < win_size or W < win_size or C == 0:
        return 0.0
        
    pad = win_size // 2
    p_pad = F.pad(pred, (pad, pad, pad, pad), mode='reflect')
    t_pad = F.pad(target, (pad, pad, pad, pad), mode='reflect')
    
    # Gaussian window matching skimage defaults (sigma=1.5, win_size=11)
    weight = _make_gaussian_window(win_size, sigma, C, pred.device, pred.dtype)
    
    mu_x = F.conv2d(p_pad, weight, groups=C)
    mu_y = F.conv2d(t_pad, weight, groups=C)
    
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y
    
    sigma_x_sq = F.conv2d(p_pad * p_pad, weight, groups=C) - mu_x_sq
    sigma_y_sq = F.conv2d(t_pad * t_pad, weight, groups=C) - mu_y_sq
    sigma_xy = F.conv2d(p_pad * t_pad, weight, groups=C) - mu_xy
    
    # Bessel correction (N/(N-1)) to match skimage's unbiased variance
    NP = win_size ** 2
    # For Gaussian window, the effective NP is sum(w)^2 / sum(w^2)
    # But skimage uses a simpler correction: cov_norm = NP / (NP - 1)
    # which is applied as sigma = sigma * NP / (NP - 1)
    cov_norm = NP / (NP - 1)
    sigma_x_sq = sigma_x_sq * cov_norm
    sigma_y_sq = sigma_y_sq * cov_norm
    sigma_xy = sigma_xy * cov_norm
    
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
    
    return ssim_map.mean().item()


def _compute_ssim_bounded(pred, target, valid_mask):
    """SSIM for bounded outputs (e.g., albedo).
    """
    pred_n = torch.clamp(pred, 0.0, 1.0)
    tgt_n = torch.clamp(target, 0.0, 1.0)

    mask_f = valid_mask.float().expand_as(tgt_n)
    pred_n = pred_n * mask_f
    tgt_n = tgt_n * mask_f
    
    return _pytorch_ssim_skimage(pred_n, tgt_n, data_range=1.0)


def _get_tonemap_scale(img, percentile=90.0, target_brightness=0.8, valid_mask=None):
    """Compute scale so the given percentile maps to target_brightness."""
    if img.ndim == 4:
        img = img[0]
    if valid_mask is not None and valid_mask.ndim == 4:
        valid_mask = valid_mask[0]
        
    img = torch.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    
    if valid_mask is not None and valid_mask.any():
        pixels = img[:, valid_mask.squeeze(0).bool()].reshape(-1)
        if pixels.numel() > 10:
            q = torch.quantile(pixels, percentile / 100.0)
            return torch.clamp_min(q / target_brightness, 1e-6)
    
    q = torch.quantile(img.reshape(-1), percentile / 100.0)
    return torch.clamp_min(q / target_brightness, 1e-6)


def _vis_tonemap(img, percentile=90.0, target_brightness=0.8, eps=1e-6, scale=None, valid_mask=None):
    """Inference-style tonemap to [0,1] per sample with NaN/Inf guards.
    Use scale=1.0 for already-tonemapped RGB inputs; keep scale=None for
    diagnostic tensors (reconstruction/derived albedo) that need auto exposure.
    If valid_mask is provided, computes tonemap scale only over valid pixels."""
    if img.ndim == 4:
        img = img[0]
    if valid_mask is not None and valid_mask.ndim == 4:
        valid_mask = valid_mask[0]
        
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
        
    img = torch.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    
    if scale is None:
        scale = _get_tonemap_scale(img, percentile=percentile, target_brightness=target_brightness, valid_mask=valid_mask)
            
    scale = torch.as_tensor(scale, device=img.device, dtype=img.dtype).clamp_min(eps)
    return torch.clamp(img / scale, 0.0, 1.0)


def _log_val_examples(
    writer,
    global_step,
    rgb,
    predictions,
    targets,
    max_items=2,
    full_rgb_list=None,
    sample_index=None,
    example_root='06_Examples',
):
    """Log qualitative validation examples in the exact requested tile order.
    
    Args:
        sample_index: Optional global sample index for labeling in TensorBoard.
                     If provided, logged under Examples/{sample_index} tag.
    """
    # Tile order must match user request: original_input, cropped_input, s_g_pred, s_g_gt, a_d_pred, a_d_gt, s_d_pred, s_d_gt
    def _tile_full(i):
        if full_rgb_list is None or i >= len(full_rgb_list) or full_rgb_list[i] is None:
            return None
        return _vis_tonemap(full_rgb_list[i])

    def _normalize_tile(tile, target_hw=None):
        """Move tile to CPU CHW float for safe concatenation and TensorBoard logging."""
        if tile is None:
            return None
        if tile.ndim == 4:
            if tile.shape[0] == 0:
                raise ValueError('tile has empty batch dimension')
            tile = tile[0]
        if tile.ndim == 2:
            tile = tile.unsqueeze(0)
        if tile.ndim != 3:
            raise ValueError(f'expected CHW tile, got shape={tuple(tile.shape)}')
        if tile.shape[0] == 1:
            tile = tile.repeat(3, 1, 1)
        elif tile.shape[0] > 3:
            tile = tile[:3]

        tile = tile.detach().to(device='cpu', dtype=torch.float32)
        tile = torch.clamp(tile, 0.0, 1.0)

        if target_hw is not None and (tile.shape[1] != target_hw[0] or tile.shape[2] != target_hw[1]):
            tile = torch.nn.functional.interpolate(
                tile.unsqueeze(0),
                size=target_hw,
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)
        return tile

    layout_names = [
        'Input RGB',
        'Albedo GT',
        'Shading GT',
        'Residual GT',
        'Recon GT',
        '',
        'Albedo Pred',
        'Shading Pred',
        'Residual Pred',
        'Recon Pred',
    ]

    # Set TensorBoard tag based on sample_index
    if sample_index is not None:
        example_tag = f'{example_root}/sample_{sample_index}'
    else:
        example_tag = example_root
    
    writer.add_text(
        f'{example_tag}/layout',
        'left-to-right strip order: ' + ' | '.join(layout_names),
        global_step,
    )

    def _gamma_correct(x):
        return torch.pow(torch.clamp(x, min=0.0, max=1.0), 1.0 / 3.0)
    
    b = min(int(rgb.shape[0]), int(max_items))
    for i in range(b):
        v_mask = targets['loss_mask'][i:i+1]

        # Targets in linear
        s_gt = 1.0 / (targets['pi_star'][i:i+1] + 1e-6) - 1.0
        a_gt = targets['A_d_star'][i:i+1]
        r_gt = targets['R_star'][i:i+1]
        recon_gt = a_gt * s_gt + r_gt

        # Predictions in linear
        s_pred = predictions['shading'][i:i+1]
        a_pred = predictions['a_d'][i:i+1]
        r_pred = predictions['residual'][i:i+1]
        recon_pred = predictions.get('rgb_reconstructed', a_pred * s_pred + r_pred)[i:i+1]

        # Tonemap scales
        scale_a = torch.max(_get_tonemap_scale(a_pred, valid_mask=v_mask), _get_tonemap_scale(a_gt, valid_mask=v_mask))
        scale_s = torch.max(_get_tonemap_scale(s_pred, valid_mask=v_mask), _get_tonemap_scale(s_gt, valid_mask=v_mask))
        scale_r = torch.max(_get_tonemap_scale(r_pred.abs(), valid_mask=v_mask), _get_tonemap_scale(r_gt.abs(), valid_mask=v_mask))
        scale_recon = torch.max(_get_tonemap_scale(recon_pred, valid_mask=v_mask), _get_tonemap_scale(recon_gt, valid_mask=v_mask))

        a_pred_vis = _gamma_correct(_vis_tonemap(a_pred, scale=scale_a))
        a_gt_vis = _gamma_correct(_vis_tonemap(a_gt, scale=scale_a))
        s_pred_vis = _gamma_correct(_vis_tonemap(s_pred, scale=scale_s))
        s_gt_vis = _gamma_correct(_vis_tonemap(s_gt, scale=scale_s))
        
        # Visualize residuals as false-color or absolute magnitude
        r_pred_vis = _gamma_correct(_vis_tonemap(r_pred.abs(), scale=scale_r))
        r_gt_vis = _gamma_correct(_vis_tonemap(r_gt.abs(), scale=scale_r))
        
        recon_pred_vis = _gamma_correct(_vis_tonemap(recon_pred, scale=scale_recon))
        recon_gt_vis = _gamma_correct(_vis_tonemap(recon_gt, scale=scale_recon))

        blank_tile = torch.ones_like(a_pred_vis)

        sample_tiles = [
            ('Input RGB', _gamma_correct(_vis_tonemap(rgb[i:i+1], scale=None, valid_mask=v_mask))),
            ('Albedo GT', a_gt_vis),
            ('Shading GT', s_gt_vis),
            ('Residual GT', r_gt_vis),
            ('Recon GT', recon_gt_vis),

            ('', blank_tile),
            ('Albedo Pred', a_pred_vis),
            ('Shading Pred', s_pred_vis),
            ('Residual Pred', r_pred_vis),
            ('Recon Pred', recon_pred_vis),
        ]

        final_target_hw = None
        footer_h = 50 
        tile_scale = 0.5
        try:
            from PIL import ImageFont
            font = ImageFont.truetype("DejaVuSans.ttf", 15)
        except OSError:
            from PIL import ImageFont
            font = ImageFont.load_default()

        named_tiles = []
        for name, tile in sample_tiles:
            try:
                if tile is None:
                    continue
                # Normalize per-tile first, then force a single final HW after all
                # visualization transforms (scale + footer + text) so cat() is safe.
                norm_tile = _normalize_tile(tile, target_hw=None)
                if norm_tile is None:
                    continue
                
                # Apply tile scaling
                if tile_scale != 1.0:
                    norm_tile = F.interpolate(
                        norm_tile.unsqueeze(0),
                        scale_factor=tile_scale,
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                
                # Add footer with title
                c, h, w = norm_tile.shape
                canvas = torch.ones((c, h + footer_h, w), dtype=norm_tile.dtype, device=norm_tile.device)
                canvas[:, :h, :] = norm_tile
                
                # Render title text onto footer using PIL
                try:
                    canvas_np = (canvas.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                    from PIL import Image, ImageDraw
                    img = Image.fromarray(canvas_np)
                    draw = ImageDraw.Draw(img)
                    text_w = draw.textlength(name, font=font)
                    x = max(0, int((w - text_w) // 2))
                    y = h + 10
                    draw.text((x, y), name, fill=(0, 0, 0), font=font)
                    norm_tile = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).permute(2, 0, 1)
                except Exception as e:
                    norm_tile = canvas
                
                if final_target_hw is None:
                    final_target_hw = (int(norm_tile.shape[1]), int(norm_tile.shape[2]))
                elif (
                    norm_tile.shape[1] != final_target_hw[0]
                    or norm_tile.shape[2] != final_target_hw[1]
                ):
                    norm_tile = torch.nn.functional.interpolate(
                        norm_tile.unsqueeze(0),
                        size=final_target_hw,
                        mode='bilinear',
                        align_corners=False,
                    ).squeeze(0)
                named_tiles.append((name, norm_tile))
            except Exception as exc:
                print(f"[warn][val step {global_step}] skipped tile '{name}' for sample_{i}: {exc}")

        if not named_tiles:
            continue

        tiles = [t for _, t in named_tiles]
        if len(tiles) >= 18:
            row1_img = torch.cat(tiles[:6], dim=2)
            row2_img = torch.cat(tiles[6:12], dim=2)
            row3_img = torch.cat(tiles[12:18], dim=2)
            strip = torch.cat([row1_img, row2_img, row3_img], dim=1)
        elif len(tiles) >= 10:
            row1_img = torch.cat(tiles[:5], dim=2)
            row2_img = torch.cat(tiles[5:10], dim=2)
            strip = torch.cat([row1_img, row2_img], dim=1)
        else:
            # Fallback to single-row if some tiles were skipped.
            strip = torch.cat(tiles, dim=2)

        if sample_index is not None:
            image_tag = f'{example_tag}/index_{sample_index}'
        else:
            image_tag = f'{example_tag}/sample_{i}'
        writer.add_image(image_tag, strip, global_step)


def validate(
    model,
    dataloader,
    criterion,
    device,
    global_step,
    writer,
    val_example_images=2,
    val_example_indices=None,
    max_val_batches=None,
    compute_val_losses=True,
    example_root='3_Examples',
):
    """
    Validation on fixed Hypersim val split.
    Logs both losses and per-decoder metrics from scale-matched targets.
    """
    model.eval()
    total_loss = {}
    total_metric = {
        'a_d_lmse': 0.0,
        'a_d_rmse': 0.0,
        'a_d_ssim': 0.0,
        's_d_lmse': 0.0,
        's_d_rmse': 0.0,
        's_d_ssim': 0.0,
        'r_rmse': 0.0,
    }
    n_samples = 0
    n_s_d_samples = 0
    
    # Prepare sample collection for visualization
    if val_example_indices is None:
        val_example_indices = []
    val_example_indices = list(val_example_indices) if val_example_indices else []
    use_indices = len(val_example_indices) > 0
    use_count = not use_indices
    
    indices_to_collect = set(int(idx) for idx in val_example_indices)
    collected_samples = {}  # Maps global index to (rgb, predictions, targets)
    processed_batches = 0

    with torch.no_grad():
        metric_batches_to_process = len(dataloader) if max_val_batches is None else min(max_val_batches, len(dataloader))
        progress_total = len(dataloader) if use_indices else metric_batches_to_process

        for batch_idx, batch in enumerate(tqdm(dataloader, desc='Validation', total=progress_total)):
            metric_budget_reached = max_val_batches is not None and batch_idx >= max_val_batches
            need_more_index_samples = use_indices and (len(collected_samples) < len(indices_to_collect))
            if metric_budget_reached and not need_more_index_samples:
                break

            rgb = batch['rgb'].to(device, non_blocking=True)
            loss_mask = batch.get('loss_mask', None)
            if loss_mask is None:
                raise KeyError("batch is missing required 'loss_mask'")
            loss_mask = loss_mask.to(device, non_blocking=True)
            m_residual = batch.get('m_residual', torch.ones(rgb.shape[0])).float().to(device, non_blocking=True)
            seg = batch.get('seg', None)
            if seg is not None:
                seg = seg.to(device, non_blocking=True)
            normals = batch.get('normals', None)
            if normals is not None:
                normals = normals.to(device, non_blocking=True)
            with torch.no_grad():
                ccr = compute_ccr(rgb)
            predictions = model(rgb, **_forward_kwargs(model, m_diffuse=None, normals=normals, seg=seg, valid_mask=loss_mask, ccr=ccr))
            
            # Ensure predictions are float32 for stable metric computation
            predictions = {k: v.float() if isinstance(v, torch.Tensor) else v for k, v in predictions.items()}
            
            targets = compute_targets(predictions, batch)

            # Collect samples at specified indices.
            batch_size = rgb.shape[0]
            sample_indices = batch.get('sample_idx', None)
            if use_indices:
                for i in range(batch_size):
                    global_idx = int(sample_indices[i].item()) if sample_indices is not None else int(batch_idx * dataloader.batch_size + i)
                    if global_idx in indices_to_collect:
                        collected_samples[global_idx] = {
                            'rgb': rgb[i:i+1],
                            'predictions': {k: v[i:i+1] for k, v in predictions.items()},
                            'targets': {k: v[i:i+1] for k, v in targets.items()},
                        }
            
            # For count-based mode, log from first batch only
            if use_count and batch_idx == 0:
                _log_val_examples(
                    writer,
                    global_step,
                    rgb,
                    predictions,
                    targets,
                    max_items=val_example_images,
                    example_root=example_root,
                )

            if metric_budget_reached:
                continue

            processed_batches += 1

            if compute_val_losses:
                losses = criterion(
                    predictions=predictions,
                    targets=targets,
                    loss_mask=loss_mask,
                    m_residual=m_residual,
                    rgb=rgb,
                    use_ssi=True,
                )
                for k, v in losses.items():
                    if k not in total_loss:
                        total_loss[k] = 0.0
                    total_loss[k] += v.item()

            # Evaluate metrics
            # 1. Albedo metrics
            total_metric['a_d_lmse'] += _compute_lmse(predictions['a_d'], targets['A_d_star'], loss_mask).item() * batch_size
            total_metric['a_d_rmse'] += _masked_scale_invariant_rmse(predictions['a_d'], targets['A_d_star'], loss_mask).item() * batch_size
            total_metric['a_d_ssim'] += _compute_ssim_bounded(predictions['a_d'], targets['A_d_star'], loss_mask) * batch_size

            # 2. Diffuse Shading metrics (from Softplus Linear Shading)
            s_d_gt = targets['S_d_star']
            s_d_pred = predictions['shading']
            
            total_metric['s_d_lmse'] += _compute_lmse(s_d_pred, s_d_gt, loss_mask).item() * batch_size
            total_metric['s_d_rmse'] += _masked_scale_invariant_rmse(s_d_pred, s_d_gt, loss_mask).item() * batch_size
            total_metric['s_d_ssim'] += _compute_shading_ssim(s_d_pred, s_d_gt, loss_mask) * batch_size
            n_s_d_samples += batch_size

            # 3. Residual metrics
            r_pred = predictions['residual']
            r_gt = targets['R_star']
            total_metric['r_rmse'] += _masked_scale_invariant_rmse(r_pred, r_gt, loss_mask).item() * batch_size
            
            n_samples += batch_size

    if use_indices and collected_samples:
        for global_idx in sorted(collected_samples.keys()):
            sample = collected_samples[global_idx]
            _log_val_examples(
                writer, global_step, sample['rgb'], sample['predictions'], sample['targets'],
                max_items=1, sample_index=global_idx, example_root=example_root,
            )

    if compute_val_losses:
        denom_loss = max(processed_batches, 1)
        for k in total_loss: total_loss[k] /= denom_loss

    denom_metric = max(n_samples, 1)
    for k in total_metric:
        if k.startswith('s_d_'):
            total_metric[k] /= max(n_s_d_samples, 1)
        else:
            total_metric[k] /= denom_metric

    val_out = {}
    if compute_val_losses: val_out.update(total_loss)
    val_out.update(total_metric)
    _log_ordered_scalars(writer, val_out, global_step, tag_prefix=None)

    return val_out



def build_stage1_model(config):
    version = float(config['model'].get('version', 11))
    model_cfg = {
        'z_channels': config['model'].get('z_channels', 1024),
        'freeze_stages': config['model'].get('freeze_stages', [1, 2]),
        'backbone': config['model'].get('backbone', 'convnextv2_base'),
        'pretrained': config['model'].get('pretrained', True),
        'num_seg_classes': config['model'].get('num_seg_classes', 41),
        'input_size': int(config['train'].get('input_size', 384)),
    }

    model_map = {
        12.0: IntrinsicDecompositionV12,
        16.0: IntrinsicDecompositionV16,
        17.0: IntrinsicDecompositionV17,
    }

    if version not in model_map:
        raise ValueError(f"Unsupported Stage1 version for single dataset mode: {version}. Supported: 13, 14, 15, 16, 17")
    return model_map[version](model_cfg)


def _subset_dataset(dataset, dataset_name, max_images, seed=None):
    """Return deterministic subset for dataset-size ablations."""
    total = len(dataset)
    max_images = int(max_images)
    if max_images <= 0:
        return dataset, total, total
    if max_images >= total:
        print(
            f"[subset] {dataset_name}: requested {max_images} >= total {total}; "
            "using full dataset"
        )
        return dataset, total, total

    if seed is None:
        indices = torch.randperm(total)[:max_images].tolist()
    else:
        gen = torch.Generator()
        gen.manual_seed(int(seed))
        indices = torch.randperm(total, generator=gen)[:max_images].tolist()
    subset = Subset(dataset, indices)
    print(
        f"[subset] {dataset_name}: using {len(subset)}/{total} images "
        f"(seed={int(seed)})"
    )
    return subset, total, len(subset)


def build_optimizer_stage1(model, train_cfg, model_cfg):
    """Build Adam optimizer with optional lower LR for image encoder."""
    base_lr = float(train_cfg['lr'])
    multiplier = float(model_cfg.get('backbone_lr_multiplier', 1.0))

    if multiplier <= 0.0:
        raise ValueError(f"backbone_lr_multiplier must be > 0, got {multiplier}")

    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith('image_encoder.'):
            backbone_params.append(param)
        else:
            other_params.append(param)

    if not backbone_params or not other_params or abs(multiplier - 1.0) < 1e-12:
        params = (p for p in model.parameters() if p.requires_grad)
        return torch.optim.Adam(params, lr=base_lr)

    return torch.optim.Adam(
        [
            {'params': backbone_params, 'lr': base_lr * multiplier},
            {'params': other_params, 'lr': base_lr},
        ]
    )


def main():
    args = parse_args()
    config = load_config(args.config, args.version)
    print(yaml.dump(config, default_flow_style=False))

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    torch.backends.cudnn.benchmark = True

    # Use command-line version for paths if provided, else fall back to config's model version
    path_version = args.version if args.version is not None else config['model']['version']
    ckpt_dir = os.path.join(config['paths']['checkpoint_dir'], f'v{path_version}')
    log_dir = os.path.join(config['paths']['log_dir'], f'v{path_version}')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)
    model = build_stage1_model(config).to(device)
    criterion = FlexibleLoss(config['loss']).to(device)
    optimizer = build_optimizer_stage1(model, config['train'], config['model'])

    hypersim_root = config['data']['hypersim_root']
    if not os.path.isabs(hypersim_root):
        hypersim_root = str(ROOT_DIR / hypersim_root)

    split_file = config['data'].get('hypersim_split_file', 'hypersim_split.json')
    split_seed = int(config['data'].get('hypersim_split_seed', 42))
    split_ratio = float(config['data'].get('hypersim_split_ratio', 0.9))
    strict_split = bool(config['data'].get('hypersim_strict_split', True))
    hypersim_max_hdf5_retries = int(config['data'].get('hypersim_max_hdf5_retries', 1))
    hypersim_skip_corrupt_samples = bool(config['data'].get('hypersim_skip_corrupt_samples', True))

    cache_max_items = int(config['data'].get('cache_max_items', 512))
    crop_mode_train = str(config['data'].get('crop_mode_train', 'random'))
    crop_mode_val = str(config['data'].get('crop_mode_val', 'center'))
    subset_seed = None

    hypersim_train_max_images = int(config['data'].get('hypersim_train_max_images', 0))

    from src.data.mixed_dataset import get_mixed_loader
    def _build_train_loader(weights_key):
        return get_mixed_loader(
            data_roots={'hypersim': hypersim_root, 'midintrinsic': config['data'].get('midintrinsic_root', '../../../datasets/MIDIntrinsics')},
            batch_size=int(config['train']['batch_size']),
            split='train',
            num_workers=int(config['train'].get('num_workers', 4)),
            input_size=int(config['train']['input_size']),
            cache_max_items=cache_max_items,
            mix_weights=config['train'].get(weights_key, {'hypersim': 1.0, 'midintrinsic': 0.0})
        )

    def infinite_loader(dl):
        while True:
            for b in dl:
                yield b

    train_loader = _build_train_loader('sampling_weights_phase1')
    train_iter = iter(infinite_loader(train_loader))

    val_num_workers = max(1, int(config['data'].get('val_num_workers', config['train'].get('val_num_workers', 2))))
    val_cache_max_items = max(0, int(config['data'].get('val_cache_max_items', 64)))
    val_batch_size = int(config['train']['batch_size'])

    val_loader = get_hypersim_loader(
        root_dir=hypersim_root,
        batch_size=val_batch_size,
        split='val',
        num_workers=val_num_workers,
        input_size=int(config['train']['input_size']),
        cache_max_items=val_cache_max_items,
        crop_mode_train=crop_mode_train,
        crop_mode_val=crop_mode_val,
        split_file=split_file,
        split_seed=split_seed,
        split_ratio=split_ratio,
        strict_split=strict_split,
        max_hdf5_retries=hypersim_max_hdf5_retries,
        skip_corrupt_samples=hypersim_skip_corrupt_samples,
    )

    max_iters = int(config['train'].get('extend_iterations', 25000))
    grad_accum_steps = max(1, int(config['train'].get('grad_accum_steps', 1)))
    grad_clip_max_norm = float(config['train'].get('grad_clip_max_norm', 1.0))
    use_cosine_lr = bool(config['train'].get('use_cosine_lr', True))
    lr_eta_min = float(config['train'].get('lr_eta_min', 1.0e-7))
    val_interval_iters = int(config['train'].get('val_interval_iters', 2000))
    ckpt_interval_iters = int(config['train'].get('checkpoint_interval_iters', 5000))
    val_example_images = int(config['train'].get('val_example_images', 2))
    val_example_indices = config['train'].get('val_example_indices', [])
    max_val_batches = config['train'].get('max_val_batches', None)
    if max_val_batches is not None:
        max_val_batches = int(max_val_batches)
    compute_val_losses = bool(config['train'].get('compute_val_losses', True))

    start_step = 0
    resume_path = _resolve_resume_path(args.resume, args.auto_resume, ckpt_dir)
    if resume_path:
        skip_opt = args.skip_optimizer or config['train'].get('skip_optimizer', False)
        start_step, _ = load_checkpoint(model, optimizer, resume_path, map_location=device, skip_optimizer=skip_opt)
        start_step += 1
        
        if args.reset_lr or config['train'].get('reset_lr', False):
            base_lr = float(config['train']['lr'])
            multiplier = float(config['model'].get('backbone_lr_multiplier', 1.0))
            if len(optimizer.param_groups) == 2:
                optimizer.param_groups[0]['lr'] = base_lr * multiplier
                optimizer.param_groups[1]['lr'] = base_lr
            else:
                for pg in optimizer.param_groups:
                    pg['lr'] = base_lr
            print(f"!!! LR Reset: Manual override to {base_lr} from config")

    scheduler = None
    if use_cosine_lr:
        total_opt_steps = max(1, math.ceil(max_iters / grad_accum_steps))
        completed_opt_steps = max(0, start_step // grad_accum_steps)
        for pg in optimizer.param_groups:
            pg.setdefault('initial_lr', pg['lr'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_opt_steps, eta_min=lr_eta_min, last_epoch=completed_opt_steps - 1
        )

    print(f"Start step: {start_step}, max step: {max_iters}")

    running = {}

    train_pbar = tqdm(range(start_step, max_iters), desc='Training', total=max_iters - start_step, dynamic_ncols=True)

    scaler = GradScaler('cuda')

    phase1_iters = int(config['train'].get('phase1_iterations', -1))

    for step in train_pbar:
        # Phase switch check
        if step == phase1_iters:
            print(f"--- Switching to Phase 2 Sampling Weights at step {step} ---")
            train_loader = _build_train_loader('sampling_weights_phase2')
            train_iter = iter(infinite_loader(train_loader))

        batch = next(train_iter)
        losses = train_one_step(model, batch, criterion, device)
        loss = losses['loss_total'] / float(grad_accum_steps)
        
        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0 or step == max_iters - 1:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
        
        train_pbar.set_postfix({'loss': f"{losses['loss_total'].item():.4f}"})

        for k, v in losses.items():
            if k not in running:
                running[k] = 0.0
            running[k] += v.item()

        if step % int(config['train']['log_interval']) == 0:
            _log_ordered_scalars(writer, losses, step, tag_prefix=None)
            if len(optimizer.param_groups) > 1:
                writer.add_scalar('0. Training/lr_backbone', float(optimizer.param_groups[0]['lr']), step)
                writer.add_scalar('0. Training/lr_heads', float(optimizer.param_groups[1]['lr']), step)
            else:
                writer.add_scalar('0. Training/lr', float(optimizer.param_groups[0]['lr']), step)

        if (step + 1) % ckpt_interval_iters == 0:
            avg = {k: running[k] / ckpt_interval_iters for k in running}
            for k in running: running[k] = 0.0
            ckpt_path = os.path.join(ckpt_dir, f'checkpoint_iter_{step+1}.pth')
            save_checkpoint(model, optimizer, avg, config, ckpt_path, global_step=step)
            save_checkpoint(model, optimizer, avg, config, os.path.join(ckpt_dir, 'checkpoint_latest.pth'), global_step=step)

            # Keep maximum 2 checkpoints (exclude checkpoint_latest.pth)
            all_ckpts = [
                os.path.join(ckpt_dir, f)
                for f in os.listdir(ckpt_dir)
                if f.startswith('checkpoint_iter_') and f.endswith('.pth')
            ]
            all_ckpts.sort(key=_extract_iter_from_name)
            while len(all_ckpts) > 2:
                oldest_ckpt = all_ckpts.pop(0)
                try:
                    os.remove(oldest_ckpt)
                    print(f"Deleted old checkpoint: {oldest_ckpt}")
                except Exception as e:
                    print(f"Failed to delete {oldest_ckpt}: {e}")

        if (step + 1) % val_interval_iters == 0:
            torch.cuda.empty_cache()
            vloss = validate(
                model, val_loader, criterion, device, step + 1, writer,
                val_example_images=val_example_images, val_example_indices=val_example_indices,
                max_val_batches=max_val_batches, compute_val_losses=compute_val_losses,
                example_root='3. Examples',
            )
            pretty = ", ".join([f"{k}={v:.4f}" for k, v in vloss.items()])
            print(f"[{step+1}] val: {pretty}")

    print('Training completed')
    writer.close()


if __name__ == '__main__':
    main()

