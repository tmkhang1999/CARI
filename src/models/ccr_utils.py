"""Shared CCR descriptor utility used by Stage-1 model variants."""

import torch
import torch.nn.functional as F


# UPDATE compute_ccr IN ccr_utils.py:
def compute_ccr(img, eps=1e-7):
    """Build 6-channel CCR descriptor with illumination-invariant normalization."""
    log_img = torch.log(img + eps)

    base_kernel = torch.tensor(
        [[0, 1, 0], [1, 0, -1], [0, -1, 0]],
        dtype=img.dtype,
        device=img.device,
    )
    kernel = base_kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    diffs = F.conv2d(log_img, kernel, padding=1, groups=3)

    diff_r, diff_g, diff_b = diffs[:, 0:1], diffs[:, 1:2], diffs[:, 2:3]
    m_rg = torch.clamp(diff_r - diff_g, -1.0, 1.0)
    m_rb = torch.clamp(diff_r - diff_b, -1.0, 1.0)
    m_gb = torch.clamp(diff_g - diff_b, -1.0, 1.0)
    
    ccr_log = torch.cat([m_rg, m_rb, m_gb], dim=1)
    ccr_mean = ccr_log.mean(dim=(2, 3), keepdim=True)
    ccr_log = ccr_log - ccr_mean  # Centers the ratios, removing global light color

    intensity = img[:, 0:1] + img[:, 1:2] + img[:, 2:3] + eps
    norm_rgb = img / intensity

    return torch.cat([ccr_log, norm_rgb], dim=1)