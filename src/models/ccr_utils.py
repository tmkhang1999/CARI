import numpy as np
import torch
import torch.nn.functional as F
import kornia


def compute_ccr(rgb):
    """
    6-channel CCR from linear RGB.
    Input should match the model/dataloader RGB convention, i.e. tonemapped linear [0,1].
    Supports both torch.Tensor (B,3,H,W or 3,H,W) and np.ndarray (H,W,3).
    Returns same type as input:
        If Tensor: (B,6,H,W)
        If ndarray: (H,W,6)
    """
    is_numpy = isinstance(rgb, np.ndarray)
    if is_numpy:
        img = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
    else:
        img = rgb if rgb.ndim == 4 else rgb.unsqueeze(0)

    device = img.device
    
    # 1. ROBUST EPSILON: 1e-3 is safe for 8-bit (1/255 = 0.0039). 
    # This prevents noise explosion in dark LDR shadows.
    eps = 1e-4
    
    # 2. ANTI-ALIASING: Slightly blur the image to kill JPEG/Quantization noise 
    # before computing extreme log gradients.
    # Note: Stronger blur (5x5, sigma=1.0) for V13 to suppress noise in dark pixels
    kernel_size = (3, 3)
    sigma = (0.5, 0.5)
    img_smooth = kornia.filters.gaussian_blur2d(img, kernel_size, sigma)

    kernel = torch.tensor(
        [[0, 1, 0],
        [1, 0, -1], 
        [0, -1, 0]], dtype=torch.float32, device=device
    ).view(1, 1, 3, 3)

    log_img = torch.log(img_smooth + eps)
    r, g, b = log_img[:, 0:1], log_img[:, 1:2], log_img[:, 2:3]

    def diff(ch):
        return F.conv2d(ch, kernel, padding=1)

    log_rg = torch.clamp(diff(r) - diff(g), -1.0, 1.0)
    log_rb = torch.clamp(diff(r) - diff(b), -1.0, 1.0)
    log_gb = torch.clamp(diff(g) - diff(b), -1.0, 1.0)

    intensity = img_smooth[:, 0:1] + img_smooth[:, 1:2] + img_smooth[:, 2:3] + eps
    norm_rgb = img_smooth / intensity

    ccr = torch.cat([log_rg, log_rb, log_gb, norm_rgb], dim=1)
    
    # Soft dark-region gate: suppress CCR completely in extremely dark regions
    # where sensor/quantization noise dominates the log ratio.
    # intensity = img_smooth.mean(dim=1, keepdim=True)
    # dark_gate = torch.sigmoid((intensity - 0.02) * 50.0)
    # ccr = ccr * dark_gate

    if is_numpy:
        return ccr.squeeze(0).permute(1, 2, 0).cpu().numpy()
    else:
        return ccr