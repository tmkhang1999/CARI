"""Albedo / shading metrics shared by training-time validation and the ARAP evaluator.

All take (B,C,H,W) predictions and targets with a (B,1,H,W) validity mask.
"""

import torch
import torch.nn.functional as F


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
