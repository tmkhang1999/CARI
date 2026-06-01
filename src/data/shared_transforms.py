import numpy as np
import cv2
import torch
import torch.nn.functional as F
from src.data.augmentations import apply_physical_augmentations, random_segmentation_degradation

def compute_tonemap_scale(
    rgb: np.ndarray,
    percentile: float = 90.0,
    target_brightness: float = 0.8,
) -> float:
    """Compute a brightness-based scale so the given percentile maps to 0.8."""
    # Downsample for faster percentile computation if image is large
    H, W = rgb.shape[:2]
    if max(H, W) > 512:
        scale = 512.0 / max(H, W)
        rgb_small = cv2.resize(rgb, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    else:
        rgb_small = rgb

    rgb32 = np.nan_to_num(rgb_small, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    brightness = (0.3 * rgb32[..., 0]) + (0.59 * rgb32[..., 1]) + (0.11 * rgb32[..., 2])
    brightness_flat = brightness.reshape(-1)

    scale_at_percentile = float(np.percentile(brightness_flat, percentile))
    if not np.isfinite(scale_at_percentile) or scale_at_percentile <= 0.0:
        return 1e-6
    return max(float(target_brightness) / scale_at_percentile, 1e-6)


def tonemap_linear(rgb: np.ndarray, percentile: float = 90.0, scale: float | None = None) -> np.ndarray:
    """
    Compress linear HDR to [0,1] without gamma and with robust numeric guards.
    rgb: (H, W, 3) float32
    """
    rgb32 = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    if scale is None:
        scale = compute_tonemap_scale(rgb32, percentile=percentile)
    else:
        scale = max(float(scale), 1e-6)

    mapped = np.divide(rgb32, scale, out=np.zeros_like(rgb32), where=np.isfinite(rgb32))
    mapped = np.nan_to_num(mapped, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(mapped, 0.0, 1.0)


def sanitize_normals(normals: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Replace invalid normal vectors and re-normalize to unit length."""
    n = np.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    n = np.clip(n, -1.0, 1.0)
    # Faster L2 norm than np.linalg.norm
    norm = np.sqrt(np.sum(n * n, axis=-1, keepdims=True))
    safe = np.maximum(norm, eps)
    n = n / safe
    n = np.where(norm > eps, n, 0.0)
    return n


def prepare_training_tensors(
    rgb: np.ndarray,        # (H,W,3) linear HDR float32
    alb: np.ndarray,        # (H,W,3) linear float32
    illum: np.ndarray,      # (H,W,3) linear HDR float32
    norm: np.ndarray,       # (H,W,3) float32
    seg: np.ndarray,        # (H,W)   int
    crop_mode: str,         # 'random', 'center', 'full'
    input_size: int,        # default 384
    split: str,             # 'train' or 'val'
) -> dict:
    """
    Standardizes tonemapping, validity masking, cropping, resizing, and physical augmentations
    for any dataset (Hypersim, MIDIntrinsic, etc.).
    """
    # 1. Compute tonemap scale on FULL image (downsampled internally) for consistency
    tonemap_scale = compute_tonemap_scale(rgb)

    # 2. Determine crop bounds FIRST to avoid doing nan_to_num on 1080p arrays
    H, W = rgb.shape[:2]
    max_crop = min(H, W)
    min_crop = 128

    if crop_mode == 'hybrid':
        crop_mode = 'center' if np.random.rand() < 0.2 else 'random'

    if crop_mode == 'full':
        top, left, size = 0, 0, max_crop
        # If full, we just take the center square to preserve aspect ratio, 
        # or we just take the whole image. The original code did seg_crop = seg.
        # But let's stick to the original behavior:
        rgb_c = rgb
        alb_c = alb
        illum_c = illum
        norm_c = norm
        seg_c = seg
    else:
        if crop_mode == 'center':
            size = max_crop
            top = (H - size) // 2
            left = (W - size) // 2
        else:
            size = max_crop if max_crop < min_crop else np.random.randint(min_crop, max_crop + 1)
            top = np.random.randint(0, max(1, H - size + 1))
            left = np.random.randint(0, max(1, W - size + 1))

        rgb_c = rgb[top:top+size, left:left+size]
        alb_c = alb[top:top+size, left:left+size]
        illum_c = illum[top:top+size, left:left+size]
        norm_c = norm[top:top+size, left:left+size]
        seg_c = seg[top:top+size, left:left+size]

    # 3. Sanitize and tonemap the CROPPED arrays (much faster)
    rgb_c = np.nan_to_num(rgb_c, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    alb_c = np.nan_to_num(alb_c, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    alb_c = np.clip(alb_c, 0.0, None)
    illum_c = np.nan_to_num(illum_c, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    illum_c = np.clip(illum_c, 0.0, None)
    norm_c = sanitize_normals(norm_c)

    # Validity mask
    alb_mask = alb_c.mean(axis=-1) >= 0.004
    loss_crop = alb_mask.astype(np.float32)

    # Tonemap
    rgb_tm = tonemap_linear(rgb_c, scale=tonemap_scale)
    illum_norm = illum_c / tonemap_scale

    # Stack
    combined = np.concatenate([
        rgb_tm,
        alb_c,
        illum_norm,
        norm_c,
    ], axis=-1)  # (size, size, 12)

    # ── Resize ──
    t_img = torch.from_numpy(combined).permute(2, 0, 1).unsqueeze(0).float()
    t_img = F.interpolate(t_img, size=(input_size, input_size), mode='bilinear', align_corners=False).squeeze(0)

    t_loss_mask = torch.from_numpy(loss_crop[..., None]).permute(2, 0, 1).unsqueeze(0).float()
    t_loss_mask = F.interpolate(t_loss_mask, size=(input_size, input_size), mode='nearest').squeeze(0)

    t = t_img

    seg_t = torch.from_numpy(seg_c.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    seg_t = F.interpolate(seg_t, size=(input_size, input_size), mode='nearest').squeeze(0).long()

    # ── Augmentations (Train Only) ──
    if split == 'train':
        seg_t = random_segmentation_degradation(seg_t, p_degrade=0.6)
        t, seg_t = apply_physical_augmentations(t, seg_t, p_hflip=0.5, p_vflip=0.5)

    # Albedo Scaling
    alb_t = t[3:6]
    max_val_t = alb_t.max()
    if max_val_t > 1.0:
        albedo_scaled_t = alb_t / max_val_t
    else:
        albedo_scaled_t = alb_t

    return {
        'rgb':           t[0:3],
        'albedo_raw':    t[3:6],
        'albedo_scaled': albedo_scaled_t,
        'illum_raw':     t[6:9],
        'normals':       t[9:12],
        'loss_mask':     t_loss_mask.bool(),
        'seg':           seg_t,
    }
