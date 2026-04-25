"""
Training script for Stage 1 intrinsic decomposition.
"""

import argparse
import math
import inspect
import os
import random
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
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
    IntrinsicDecompositionV9,
    IntrinsicDecompositionV10,
    IntrinsicDecompositionV11Single,
    IntrinsicDecompositionV11Mix,
)
from losses.flexible_loss import FlexibleLoss
from data.hypersim_dataset import HypersimDataset, get_hypersim_loader


TB_TAGS = {
    'loss_total': '1. Losses/Total_all',
    'loss_a': '1. Losses/A_total',
    'loss_a_mse': '1. Losses/A_MSE',
    'loss_a_msg': '1. Losses/A_MSG',
    'loss_b': '1. Losses/B_total',
    'loss_b_mse': '1. Losses/B_MSE',
    'loss_b_msg': '1. Losses/B_MSG',
    'loss_c': '1. Losses/C_total',
    'loss_c_mse': '1. Losses/C_MSE',
    'loss_c_msg': '1. Losses/C_MSG',
    'loss_c_perceptual': '1. Losses/C_Perceptual',
    'loss_c_dssim': '1. Losses/C_DSSIM',
    'loss_c_tv': '1. Losses/C_TV',
    'loss_c_semvar': '1. Losses/C_SemVar',
    'loss_d': '1. Losses/D_total',
    'loss_d_mse': '1. Losses/D_MSE',
    'loss_d_msg': '1. Losses/D_MSG',
    's_g_lmse': '2. Metrics/Grayshading_Sg_LMSE',
    's_g_rmse': '2. Metrics/Grayshading_Sg_RMSE',
    's_g_ssim': '2. Metrics/Grayshading_Sg_SSIM',
    'xi_mse': '2. Metrics/Chroma_xi_MSE',
    'a_d_lmse': '2. Metrics/Albedo_Ad_LMSE',
    'a_d_rmse': '2. Metrics/Albedo_Ad_RMSE',
    'a_d_ssim': '2. Metrics/Albedo_Ad_SSIM',
    's_d_lmse': '2. Metrics/DiffuseShading_Sd_LMSE',
    's_d_rmse': '2. Metrics/DiffuseShading_Sd_RMSE',
    's_d_ssim': '2. Metrics/DiffuseShading_Sd_SSIM',
}


def _log_ordered_scalars(writer, values, global_step, tag_prefix=None):
    """Write only requested TensorBoard tags in deterministic order."""
    ordered = [
        'loss_total',
        'loss_a', 'loss_a_mse', 'loss_a_msg',
        'loss_b', 'loss_b_mse', 'loss_b_msg',
        'loss_c', 'loss_c_mse', 'loss_c_msg', 'loss_c_perceptual', 'loss_c_dssim', 'loss_c_tv',
        'loss_c_semvar',
        'loss_d', 'loss_d_mse', 'loss_d_msg',
        's_g_lmse', 's_g_rmse', 's_g_ssim',
        'xi_mse',
        'a_d_lmse', 'a_d_rmse', 'a_d_ssim',
        's_d_lmse', 's_d_rmse', 's_d_ssim',
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


def scale_match(A_raw, A_pred, valid):
    """Least-squares per-image scalar c for c*A_raw ~= A_pred over valid pixels."""
    v = valid.expand_as(A_raw).float()
    a = (A_raw * v).reshape(A_raw.shape[0], -1)
    b = (A_pred * v).reshape(A_pred.shape[0], -1)
    c = (a * b).sum(dim=1) / ((a * a).sum(dim=1) + 1e-6)
    # Prevent degenerate target scaling that can zero-out Dec-C supervision.
    c = torch.clamp_min(c, 0.05)
    return c.view(-1, 1, 1, 1)


def compute_targets(predictions, batch):
    eps = 1e-6
    device = predictions['a_d'].device
    rgb = batch['rgb'].to(device)
    albedo_raw = batch['albedo_raw'].to(device)
    valid_mask = batch['valid_mask'].to(device)
    
    # Scale matching parameter C is computed between predictions and truth
    c = scale_match(albedo_raw, predictions['a_d'].detach(), valid_mask)

    A_star = c * albedo_raw
    A_star = torch.nan_to_num(A_star, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    
    # Colorful shading (S_c) is directly derived from RGB and Albedo: I / A*
    S_c_star = rgb / (A_star + eps)
    S_c_star = torch.nan_to_num(S_c_star, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)

    # Grayscale and chroma targets depend on colorful shading (contains speculars & gloss)
    S_g_star = (
        0.299 * S_c_star[:, 0:1]
        + 0.587 * S_c_star[:, 1:2]
        + 0.114 * S_c_star[:, 2:3]
    )
    D_g_star = 1.0 / (S_g_star + 1.0)
    
    C_RG = S_c_star[:, 0:1] / (S_c_star[:, 1:2] + eps)
    C_BG = S_c_star[:, 2:3] / (S_c_star[:, 1:2] + eps)
    xi_star = torch.cat([1.0 / (C_RG + 1.0), 1.0 / (C_BG + 1.0)], dim=1)
    xi_star = torch.nan_to_num(xi_star, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    # Diffuse shading target (pi_star) routes from direct illumination GT if available
    S_d_star = S_c_star
    illum_raw = batch.get('illum_raw', None)
    m_diffuse = batch.get('M_diffuse', None)
    
    if illum_raw is not None and m_diffuse is not None:
        route = m_diffuse.view(-1, 1, 1, 1).to(device=rgb.device, dtype=rgb.dtype)
        illum = torch.nan_to_num(illum_raw.to(device), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        S_d_star = route * illum + (1.0 - route) * S_c_star
        
    pi_star = 1.0 / (S_d_star + 1.0)
    pi_star = torch.nan_to_num(pi_star, nan=1.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    D_g_star = torch.nan_to_num(D_g_star, nan=1.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        
    return {
        'D_g_star': D_g_star,
        'xi_star': xi_star,
        'A_d_star': A_star,
        'pi_star': pi_star,
        'valid_mask': valid_mask,
    }



def parse_args():
    parser = argparse.ArgumentParser(description='Train Stage 1')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--version', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint path or "latest"')
    parser.add_argument('--auto-resume', action='store_true', help='Resume from latest checkpoint in version checkpoint dir')
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


def load_checkpoint(model, optimizer, checkpoint_path, map_location=None):
    ckpt = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    try:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    except Exception as exc:
        print(
            "[warn] optimizer state not loaded (param-group mismatch). "
            f"Continuing with freshly initialized optimizer: {exc}"
        )
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


def _forward_kwargs(model, m_diffuse, normals, seg, valid_mask):
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
    return kwargs


def _apply_diffuse_detach(predictions, m_diffuse):
    """Detach only Dec-D output where diffuse supervision is unavailable."""
    mask = m_diffuse.view(-1, 1, 1, 1).to(predictions['pi'].device)
    predictions['pi'] = predictions['pi'] * mask + predictions['pi'].detach() * (1.0 - mask)
    return predictions


def _loss_seg(model, seg):
    if 'seg' not in inspect.signature(model.forward).parameters:
        return None
    if seg is not None and seg.dim() == 4 and seg.shape[1] == 1:
        return seg[:, 0]
    return seg


def train_one_step(model, batch, criterion, device):
    model.train()

    rgb = batch['rgb'].to(device, non_blocking=True)
    albedo_raw = batch['albedo_raw'].to(device, non_blocking=True)
    illum_raw = batch.get('illum_raw', None)
    if illum_raw is not None:
        illum_raw = illum_raw.to(device, non_blocking=True)
    valid_mask = batch['valid_mask'].to(device, non_blocking=True)
    m_diffuse = batch['M_diffuse'].float().to(device, non_blocking=True)
    m_albedo = batch['M_albedo'].float().to(device, non_blocking=True)
    seg = batch.get('seg', None)
    if seg is not None:
        seg = seg.to(device, non_blocking=True)
    normals = batch.get('normals', None)
    if normals is not None:
        normals = normals.to(device, non_blocking=True)

    with autocast(device_type='cuda', dtype=torch.float16):
        predictions = model(rgb, **_forward_kwargs(model, m_diffuse, normals, seg, valid_mask))
        predictions = _apply_diffuse_detach(predictions, m_diffuse)
        
        # Ensure predictions are float32 for stable loss and target computation
        predictions = {k: v.float() if isinstance(v, torch.Tensor) else v for k, v in predictions.items()}
        
    targets = compute_targets(predictions, batch)

    losses = criterion(
        predictions,
        targets,
        m_diffuse,
        m_albedo,
        valid_mask,
        _loss_seg(model, seg),
        normals=normals,
        rgb=rgb,
    )

    return losses


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


def _masked_scale_invariant_rmse(pred, target, valid_mask, eps=1e-7, shading_cap=None):
    """
    RMSE after per-sample mean normalization on valid pixels, with optional shading outlier cap.
    """
    mask = valid_mask.bool().expand_as(pred)
    if shading_cap is not None:
        outlier_mask = (target <= shading_cap)
        mask = mask & outlier_mask.expand_as(pred)

    p = pred[mask]
    t = target[mask]
    if p.numel() == 0:
        return pred.new_tensor(0.0)
    p = p / (p.mean() + eps)
    t = t / (t.mean() + eps)
    return torch.sqrt(((p - t) ** 2).mean())


def _ssim_from_dssim(criterion, pred, target):
    """Convert criterion DSSIM helper to SSIM in [0,1]."""
    dssim = criterion._compute_dssim(pred, target)
    return 1.0 - 2.0 * dssim


def _pytorch_ssim_skimage_approx(pred, target, data_range=1.0, win_size=7):
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
    
    weight = torch.ones(C, 1, win_size, win_size, device=pred.device, dtype=pred.dtype) / (win_size ** 2)
    
    mu_x = F.conv2d(p_pad, weight, groups=C)
    mu_y = F.conv2d(t_pad, weight, groups=C)
    
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y
    
    sigma_x_sq = F.conv2d(p_pad * p_pad, weight, groups=C) - mu_x_sq
    sigma_y_sq = F.conv2d(t_pad * t_pad, weight, groups=C) - mu_y_sq
    sigma_xy = F.conv2d(p_pad * t_pad, weight, groups=C) - mu_xy
    
    # Unbiased variance correction for skimage
    n = win_size ** 2
    sigma_x_sq = sigma_x_sq * n / (n - 1)
    sigma_y_sq = sigma_y_sq * n / (n - 1)
    sigma_xy = sigma_xy * n / (n - 1)
    
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
    
    return ssim_map.mean().item()

def _compute_shading_ssim(pred, target, valid_mask, eps=1e-6, shading_cap=None):
    """SSIM for unbounded shading with mean + 2*std robust scale from GT."""
    # Handle batch dim
    if pred.ndim == 4:
        pred = pred[0]
        target = target[0]
        valid_mask = valid_mask[0]

    v = valid_mask.bool().expand_as(target)
    if shading_cap is not None:
        v = v & (target <= shading_cap)

    gt_valid = target[v]
    if gt_valid.numel() < 100:
        return 0.0

    # Use mean + 2*std as scale
    gt_mean = gt_valid.mean()
    gt_std  = gt_valid.std()
    scale   = (gt_mean + 2.0 * gt_std).clamp_min(eps)

    pred_n = torch.clamp(pred / scale, 0.0, 1.0)
    tgt_n = torch.clamp(target / scale, 0.0, 1.0)

    mask_f = valid_mask.float().expand_as(tgt_n)
    pred_n = pred_n * mask_f
    tgt_n = tgt_n * mask_f

    # Use GPU-resident skimage approx
    return _pytorch_ssim_skimage_approx(pred_n, tgt_n, data_range=1.0)


def _compute_ssim_bounded(pred, target, valid_mask):
    """SSIM for bounded outputs (e.g., albedo).
    """
    pred_n = torch.clamp(pred, 0.0, 1.0)
    tgt_n = torch.clamp(target, 0.0, 1.0)

    mask_f = valid_mask.float().expand_as(tgt_n)
    pred_n = pred_n * mask_f
    tgt_n = tgt_n * mask_f
    
    # Use GPU-resident skimage approx
    return _pytorch_ssim_skimage_approx(pred_n, tgt_n, data_range=1.0)


def _get_tonemap_scale(img, percentile=99.0, valid_mask=None):
    """Compute the 99th percentile scale for an image, respecting valid_mask."""
    if img.ndim == 4:
        img = img[0]
    if valid_mask is not None and valid_mask.ndim == 4:
        valid_mask = valid_mask[0]
        
    img = torch.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    
    if valid_mask is not None and valid_mask.any():
        pixels = img[:, valid_mask.squeeze(0).bool()].reshape(-1)
        if pixels.numel() > 10:
            return torch.quantile(pixels, percentile / 100.0)
    
    return torch.quantile(img.reshape(-1), percentile / 100.0)


def _vis_tonemap(img, percentile=99.0, eps=1e-6, scale=None, valid_mask=None):
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
        scale = _get_tonemap_scale(img, percentile=percentile, valid_mask=valid_mask)
            
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
        'Gray Shading GT',
        'Colorful GT',
        'Diffuse GT',
        'Albedo GT',
        'Diffuse Recon GT',
        '',
        'Gray Shading Pred',
        'Colorful Pred',
        'Diffuse Pred',
        'Albedo Pred',
        'Diffuse Recon Pred',
        '',
        'Derived A_g = I/S_g',
        'Derived A_c = I/S_c',
        '',
        '',
        '',
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
        # Match inference visualization for direct comparability.
        return torch.pow(torch.clamp(x, min=0.0, max=1.0), 1.0 / 3.0)
    
    b = min(int(rgb.shape[0]), int(max_items))
    for i in range(b):
        v_mask = targets['valid_mask'][i:i+1]

        # 1. Linear shading components
        s_g_pred_linear = 1.0 / (predictions['d_g'][i:i+1] + 1e-6) - 1.0
        s_g_gt_linear = 1.0 / (targets['D_g_star'][i:i+1] + 1e-6) - 1.0
        s_d_pred_linear = 1.0 / (predictions['pi'][i:i+1] + 1e-6) - 1.0
        s_d_gt_linear = 1.0 / (targets['pi_star'][i:i+1] + 1e-6) - 1.0

        # 2. Colorful shading GT/pred via inverse-domain route
        c_rg_gt = (1.0 - targets['xi_star'][i:i+1, 0:1]) / (targets['xi_star'][i:i+1, 0:1] + 1e-6)
        c_bg_gt = (1.0 - targets['xi_star'][i:i+1, 1:2]) / (targets['xi_star'][i:i+1, 1:2] + 1e-6)
        denom_gt = (0.299 * c_rg_gt + 0.587 + 0.114 * c_bg_gt).clamp(1e-6)
        s_green_gt = s_g_gt_linear / denom_gt
        s_c_gt_linear = torch.cat([c_rg_gt * s_green_gt, s_green_gt, c_bg_gt * s_green_gt], dim=1)
        
        if 's_c' in predictions:
            s_c_pred_linear = predictions['s_c'][i:i+1]
        elif 'xi' in predictions:
            c_rg_pred = (1.0 - predictions['xi'][i:i+1, 0:1]) / (predictions['xi'][i:i+1, 0:1] + 1e-6)
            c_bg_pred = (1.0 - predictions['xi'][i:i+1, 1:2]) / (predictions['xi'][i:i+1, 1:2] + 1e-6)
            denom_pred = (0.299 * c_rg_pred + 0.587 + 0.114 * c_bg_pred).clamp(1e-6)
            s_green_pred = s_g_pred_linear / denom_pred
            s_c_pred_linear = torch.cat([c_rg_pred * s_green_pred, s_green_pred, c_bg_pred * s_green_pred], dim=1)
        else:
            s_c_pred_linear = s_g_pred_linear.repeat(1, 3, 1, 1)

        # 3. Diffuse reconstruction
        recon_diffuse = predictions['a_d'][i:i+1] * s_d_pred_linear
        recon_diffuse_gt = targets['A_d_star'][i:i+1] * s_d_gt_linear

        # 4. Joint Scaling only for Albedo (already scale-matched in linear space)
        scale_a_d = torch.max(_get_tonemap_scale(predictions['a_d'][i:i+1], valid_mask=v_mask), _get_tonemap_scale(targets['A_d_star'][i:i+1], valid_mask=v_mask))

        s_g_pred_vis = _vis_tonemap(s_g_pred_linear, valid_mask=v_mask)
        s_g_gt_vis = _vis_tonemap(s_g_gt_linear, valid_mask=v_mask)
        s_c_pred_vis = _vis_tonemap(s_c_pred_linear, valid_mask=v_mask)
        s_c_gt_vis = _vis_tonemap(s_c_gt_linear, valid_mask=v_mask)
        s_d_pred_vis = _vis_tonemap(s_d_pred_linear, valid_mask=v_mask)
        s_d_gt_vis = _vis_tonemap(s_d_gt_linear, valid_mask=v_mask)
        recon_vis = _vis_tonemap(recon_diffuse, valid_mask=v_mask)
        recon_gt_vis = _vis_tonemap(recon_diffuse_gt, valid_mask=v_mask)
        
        a_d_pred_vis = _gamma_correct(_vis_tonemap(predictions['a_d'][i:i+1], scale=scale_a_d))
        a_d_gt_vis = _gamma_correct(_vis_tonemap(targets['A_d_star'][i:i+1], scale=scale_a_d))

        # 5. Diagnostics
        a_g_derived_vis = _gamma_correct(_vis_tonemap(rgb[i:i+1] / (s_g_pred_linear + 1e-6), scale=None, valid_mask=v_mask))
        a_c_derived_vis = _gamma_correct(_vis_tonemap(rgb[i:i+1] / (s_c_pred_linear + 1e-6), scale=None, valid_mask=v_mask))
        blank_tile = torch.ones_like(a_g_derived_vis)

        sample_tiles = [
            ('Input RGB', _vis_tonemap(rgb[i:i+1], scale=None)), # Auto-exposure for HDR Input
            ('Gray Shading GT', s_g_gt_vis),
            ('Colorful GT', s_c_gt_vis),
            ('Diffuse GT', s_d_gt_vis),
            ('Albedo GT', a_d_gt_vis),
            ('Diffuse Recon GT', recon_gt_vis),

            ('', blank_tile),
            ('Gray Shading Pred', s_g_pred_vis),
            ('Colorful Pred', s_c_pred_vis),
            ('Diffuse Pred', s_d_pred_vis),
            ('Albedo Pred', a_d_pred_vis),
            ('Diffuse Recon Pred', recon_vis),

            ('', blank_tile),
            ('Derived A_g = I/S_g', a_g_derived_vis),
            ('Derived A_c = I/S_c', a_c_derived_vis),
            ('', blank_tile),
            ('', blank_tile),
            ('', blank_tile),
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
    if compute_val_losses:
        total_loss = {
            'loss_total': 0.0,
            'loss_a': 0.0,
            'loss_b': 0.0,
            'loss_c': 0.0,
            'loss_d': 0.0,
        }
    total_metric = {
        's_g_lmse': 0.0,
        's_g_rmse': 0.0,
        's_g_ssim': 0.0,
        'xi_mse': 0.0,
        'a_d_lmse': 0.0,
        'a_d_rmse': 0.0,
        'a_d_ssim': 0.0,
        's_d_lmse': 0.0,
        's_d_rmse': 0.0,
        's_d_ssim': 0.0,
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
            valid_mask = batch['valid_mask'].to(device, non_blocking=True)
            m_diffuse = batch['M_diffuse'].float().to(device, non_blocking=True)
            m_albedo = batch['M_albedo'].float().to(device, non_blocking=True)
            seg = batch.get('seg', None)
            if seg is not None:
                seg = seg.to(device, non_blocking=True)
            normals = batch.get('normals', None)
            if normals is not None:
                normals = normals.to(device, non_blocking=True)

            predictions = model(rgb, **_forward_kwargs(model, m_diffuse, normals, seg, valid_mask))
            predictions = _apply_diffuse_detach(predictions, m_diffuse)
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
                    predictions, targets, m_diffuse, m_albedo, valid_mask, _loss_seg(model, seg),
                    normals=normals, rgb=rgb
                )
                for k, v in losses.items():
                    if k in total_loss:
                        total_loss[k] += v.item()

            # Evaluate metrics
            d_g_star = targets['D_g_star']
            pi_star = targets['pi_star']
            d_g_pred = predictions['d_g']
            pi_pred = predictions['pi']

            for i in range(batch_size):
                vm = valid_mask[i:i+1]
                s_g_pred = 1.0 / (d_g_pred[i:i+1] + 1e-6) - 1.0
                s_g_gt = 1.0 / (d_g_star[i:i+1] + 1e-6) - 1.0
                total_metric['s_g_lmse'] += _compute_lmse(s_g_pred, s_g_gt, vm).item()
                total_metric['s_g_rmse'] += _masked_scale_invariant_rmse(s_g_pred, s_g_gt, vm).item()
                total_metric['s_g_ssim'] += _compute_ssim_bounded(s_g_pred, s_g_gt, vm)

                if 'xi' in predictions:
                    xi_v = vm.expand_as(predictions['xi'][i:i+1]).float()
                    xi_err = ((predictions['xi'][i:i+1] - targets['xi_star'][i:i+1]) ** 2 * xi_v).sum() / (xi_v.sum() + 1e-7)
                    total_metric['xi_mse'] += xi_err.item()

                total_metric['a_d_lmse'] += _compute_lmse(predictions['a_d'][i:i+1], targets['A_d_star'][i:i+1], vm).item()
                total_metric['a_d_rmse'] += _masked_scale_invariant_rmse(predictions['a_d'][i:i+1], targets['A_d_star'][i:i+1], vm).item()
                total_metric['a_d_ssim'] += _compute_ssim_bounded(predictions['a_d'][i:i+1], targets['A_d_star'][i:i+1], vm)

                if m_diffuse[i].item() > 0.5:
                    total_metric['s_d_lmse'] += _compute_lmse(pi_pred[i:i+1], pi_star[i:i+1], vm).item()
                    total_metric['s_d_rmse'] += _masked_scale_invariant_rmse(pi_pred[i:i+1], pi_star[i:i+1], vm).item()
                    total_metric['s_d_ssim'] += _compute_ssim_bounded(pi_pred[i:i+1], pi_star[i:i+1], vm)
                    n_s_d_samples += 1
                n_samples += 1

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


def _phase_schedule(train_cfg, global_step):
    """Two-phase curriculum exactly as requested by config values."""
    p1 = int(train_cfg.get('phase1_iterations', 20000))
    w1 = train_cfg.get('sampling_weights_phase1', {'hypersim': 1.0, 'midintrinsic': 0.0})
    w2 = train_cfg.get('sampling_weights_phase2', {'hypersim': 0.6, 'midintrinsic': 0.4})
    if global_step < p1:
        return 'phase1', w1
    return 'phase2', w2


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
        11.0: IntrinsicDecompositionV11Single,
    }

    if version not in model_map:
        raise ValueError(f"Unsupported Stage1 version for single dataset mode: {version}. Supported: 11")
    return model_map[version](model_cfg)


def _subset_dataset(dataset, dataset_name, max_images, seed):
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

    seed = int(config['train'].get('seed', config['train'].get('dataloader_seed', 42)))
    deterministic = bool(config['train'].get('deterministic', False))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False # Force false to avoid FIND engine error

    version = config['model']['version']
    ckpt_dir = os.path.join(config['paths']['checkpoint_dir'], f'v{version}')
    log_dir = os.path.join(config['paths']['log_dir'], f'v{version}')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)
    model = build_stage1_model(config).to(device)
    criterion = FlexibleLoss(config['loss']).to(device)
    optimizer = build_optimizer_stage1(model, config['train'], config['model'])

    hypersim_root = config['data']['hypersim_root']
    mid_root = config['data'].get('midintrinsic_root', '../datasets/MIDIntrinsics')
    if not os.path.isabs(hypersim_root):
        hypersim_root = str(ROOT_DIR / hypersim_root)
    if not os.path.isabs(mid_root):
        mid_root = str(ROOT_DIR / mid_root)

    split_file = config['data'].get('hypersim_split_file', 'hypersim_split.json')
    split_seed = int(config['data'].get('hypersim_split_seed', 42))
    split_ratio = float(config['data'].get('hypersim_split_ratio', 0.9))
    strict_split = bool(config['data'].get('hypersim_strict_split', True))
    hypersim_max_hdf5_retries = int(config['data'].get('hypersim_max_hdf5_retries', 1))
    hypersim_skip_corrupt_samples = bool(config['data'].get('hypersim_skip_corrupt_samples', True))

    enabled = set(config['data'].get('datasets', ['hypersim']))
    cache_max_items = int(config['data'].get('cache_max_items', 512))
    crop_mode_train = str(config['data'].get('crop_mode_train', 'random'))
    crop_mode_val = str(config['data'].get('crop_mode_val', 'center'))
    loader_seed = int(config['train'].get('dataloader_seed', 42))
    subset_seed = int(config['data'].get('subset_seed', loader_seed))
    hypersim_train_max_images = int(config['data'].get('hypersim_train_max_images', 0))
    midintrinsic_train_max_images = int(config['data'].get('midintrinsic_train_max_images', 0))

    train_hypersim_base = HypersimDataset(
        root_dir=hypersim_root,
        split='train',
        input_size=int(config['train']['input_size']),
        cache_max_items=cache_max_items,
        crop_mode_train=crop_mode_train,
        crop_mode_val=crop_mode_val,
        split_file=split_file,
        split_seed=split_seed,
        split_ratio=split_ratio,
        strict_split=strict_split,
        max_hdf5_retries=hypersim_max_hdf5_retries,
        skip_corrupt_samples=hypersim_skip_corrupt_samples,
    )
    train_hypersim, hypersim_total, hypersim_effective = _subset_dataset(
        train_hypersim_base,
        'hypersim',
        hypersim_train_max_images,
        subset_seed,
    )

    gen = torch.Generator()
    gen.manual_seed(loader_seed)
    train_num_workers = int(config['data'].get('hypersim_train_num_workers', config['train'].get('num_workers', 4)))
    train_prefetch_factor = int(config['data'].get('hypersim_train_prefetch_factor', config['train'].get('prefetch_factor', 4)))
    train_batch_size = int(config['train']['batch_size'])

    train_loader = DataLoader(
        train_hypersim,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=train_num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=train_prefetch_factor if train_num_workers > 0 else None,
        generator=gen,
    )

    def infinite_loader(dl):
        while True:
            for b in dl:
                yield b
    
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

    max_iters = int(config['train'].get('max_iterations', 50000))
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
        start_step, _ = load_checkpoint(model, optimizer, resume_path, map_location=device)
        start_step += 1

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

    running = {'loss_total': 0.0, 'loss_a': 0.0, 'loss_b': 0.0, 'loss_c': 0.0, 'loss_d': 0.0}

    train_pbar = tqdm(range(start_step, max_iters), desc='Training', total=max_iters - start_step, dynamic_ncols=True)

    scaler = GradScaler('cuda')

    for step in train_pbar:
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

        for k in running:
            if k in losses:
                running[k] += losses[k].item()

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

