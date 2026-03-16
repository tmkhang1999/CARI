"""
Cross Colour Ratio (CCR) preprocessing.
Computes illumination-invariant CCR maps from RGB images.

Combines two illumination-invariant cues (validated by PIE-Net and SIGNet):
  - Log cross-colour ratios (CCR): change only at reflectance boundaries (3ch)
  - Normalized RGB chromaticity (r,g,b / sum): illumination-invariant per-pixel
    colour, insensitive to intensity changes (3ch)

Output: 6-channel tensor [log_M_RG, log_M_RB, log_M_GB, norm_R, norm_G, norm_B]
"""

import torch
import torch.nn.functional as F


def compute_ccr(img, eps=1e-7):
    """
    Compute 6-channel illumination-invariant descriptor from linear RGB.

    Channels 0-2: Clamped log cross-colour ratios (CCR, PIE-Net / SIGNet):
      log_M_RG = conv(log R) - conv(log G)  clamped to [-1, 1]
      log_M_RB = conv(log R) - conv(log B)  clamped to [-1, 1]
      log_M_GB = conv(log G) - conv(log B)  clamped to [-1, 1]
      Clamp prevents exploding values at near-black pixels (PIE-Net validated).

    Channels 3-5: Normalized RGB chromaticity (SIGNet):
      norm_c = c / (R + G + B + eps)   in [0, 1]
      Illumination-invariant: encodes per-pixel hue regardless of intensity.

    Args:
        img: (N, 3, H, W) linear RGB, float32, values > 0
        eps: small constant to avoid log(0) and division by zero

    Returns:
        (N, 6, H, W) invariant descriptor
    """
    # ── Log CCR (channels 0-2) ────────────────────────────────────────────
    log_img = torch.log(img + eps)                      # (N, 3, H, W)
    R_log = log_img[:, 0:1]
    G_log = log_img[:, 1:2]
    B_log = log_img[:, 2:3]

    # Asymmetric cross-ratio kernel (PIE-Net / SIGNet: two opposed kernels)
    # k and -k applied separately then summed = single-pass equivalent
    kernel = torch.tensor(
        [[0,  1,  0],
         [1,  0, -1],
         [0, -1,  0]], dtype=torch.float32
    ).view(1, 1, 3, 3).to(img.device)

    def diff(ch):
        return F.conv2d(ch, kernel, padding=1)

    # Clamp to [-1, 1] — prevents exploding values at near-black pixels (PIE-Net)
    log_M_RG = torch.clamp(diff(R_log) - diff(G_log), -1.0, 1.0)
    log_M_RB = torch.clamp(diff(R_log) - diff(B_log), -1.0, 1.0)
    log_M_GB = torch.clamp(diff(G_log) - diff(B_log), -1.0, 1.0)

    # ── Normalized RGB chromaticity (channels 3-5, SIGNet) ────────────────
    intensity = img[:, 0:1] + img[:, 1:2] + img[:, 2:3] + eps   # (N,1,H,W)
    norm_rgb = img / intensity                                    # (N, 3, H, W)

    return torch.cat([log_M_RG, log_M_RB, log_M_GB, norm_rgb], dim=1)  # (N, 6, H, W)


class CCRPreprocessor:
    """
    Batch processor for CCR computation.
    Can be used for offline preprocessing or online computation.
    """
    def __init__(self, eps=1e-7):
        self.eps = eps

    def __call__(self, img):
        """
        Args:
            img: (N, 3, H, W) or (3, H, W) linear RGB tensor
        Returns:
            CCR map of same shape
        """
        if img.dim() == 3:
            img = img.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        ccr = compute_ccr(img, self.eps)

        if squeeze:
            ccr = ccr.squeeze(0)

        return ccr


def preprocess_dataset_ccr(image_paths, output_dir, device='cuda'):
    """
    Offline preprocessing: compute and save CCR maps for entire dataset.

    Args:
        image_paths: List of paths to RGB images
        output_dir: Directory to save CCR maps
        device: Device for computation
    """
    import os
    from PIL import Image
    import numpy as np
    from tqdm import tqdm

    os.makedirs(output_dir, exist_ok=True)
    preprocessor = CCRPreprocessor()

    for img_path in tqdm(image_paths, desc="Computing CCR"):
        # Load image
        img = Image.open(img_path).convert('RGB')
        img = torch.from_numpy(np.array(img)).float() / 255.0
        img = img.permute(2, 0, 1).unsqueeze(0).to(device)

        # Compute CCR
        ccr = preprocessor(img)

        # Save
        basename = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(output_dir, f"{basename}_ccr.pt")
        torch.save(ccr.cpu(), output_path)

    print(f"Saved CCR maps to {output_dir}")


if __name__ == '__main__':
    # Test
    img = torch.rand(2, 3, 256, 256) + 0.1  # Ensure positive values

    ccr = compute_ccr(img)
    print(f"Input shape: {img.shape}")
    print(f"CCR shape: {ccr.shape}")
    print(f"CCR range: [{ccr.min():.3f}, {ccr.max():.3f}]")

    # Test preprocessor
    preprocessor = CCRPreprocessor()
    ccr2 = preprocessor(img[0])
    print(f"Preprocessor output shape: {ccr2.shape}")

