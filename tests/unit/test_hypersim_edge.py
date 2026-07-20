import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import kornia

# Add src to path
ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / 'src'
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import glob
import numpy as np
from data.hypersim_dataset import _load_hdf5, _compute_tonemap_scale, _tonemap_linear
from models.iid_utils import rgb_to_iuv

def _albedo_edge_gt_v17(albedo_gt):
    """Sobel magnitude on grayscale albedo GT → per-image normalized soft target in [0,1]. (From flexible_loss_v17.py)"""
    a_gray = albedo_gt.mean(dim=1, keepdim=True)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                            dtype=a_gray.dtype, device=a_gray.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                            dtype=a_gray.dtype, device=a_gray.device).view(1, 1, 3, 3)
    gx = F.conv2d(a_gray, sobel_x, padding=1)
    gy = F.conv2d(a_gray, sobel_y, padding=1)
    mag = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-8)
    q99 = torch.quantile(mag.reshape(mag.shape[0], -1), 0.99, dim=1).view(-1, 1, 1, 1).clamp(min=1e-6)
    return (mag / q99).clamp(0.0, 1.0)

def main():
    # Load single Hypersim frame directly
    print("Loading hypersim frame...")
    hypersim_root = (ROOT_DIR / '../datasets/hypersim').resolve()
    
    color_path = hypersim_root / 'ai_001_001/images/scene_cam_00_final_hdf5/frame.0000.color.hdf5'
    if not color_path.exists():
        color_files = sorted(glob.glob(str(hypersim_root / 'ai_*/images/scene_cam_*_final_hdf5/frame.*.color.hdf5')))
        if not color_files:
            raise FileNotFoundError(f"No color HDF5 files found under {hypersim_root}")
        color_path = Path(color_files[0])
        
    albedo_path = Path(str(color_path).replace('.color.hdf5', '.diffuse_reflectance.hdf5'))
    
    print(f"Loading files from {color_path.parent.parent.parent.name}...")
    rgb_raw = _load_hdf5(str(color_path))
    alb_raw = _load_hdf5(str(albedo_path))
    
    rgb_raw = np.nan_to_num(rgb_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    alb_raw = np.nan_to_num(alb_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32).clip(0.0)
    
    tonemap_scale = _compute_tonemap_scale(rgb_raw)
    rgb_tm = _tonemap_linear(rgb_raw, scale=tonemap_scale)
    
    rgb_t = torch.from_numpy(rgb_tm).permute(2, 0, 1).unsqueeze(0)
    alb_t = torch.from_numpy(alb_raw).permute(2, 0, 1).unsqueeze(0)
    
    rgb = TF.resize(rgb_t, (384, 384), antialias=True)
    albedo_raw = TF.resize(alb_t, (384, 384), antialias=True)
    
    # Same scaling logic as train.py
    c = torch.sum(rgb * albedo_raw, dim=(1,2,3), keepdim=True) / \
        (torch.sum(albedo_raw * albedo_raw, dim=(1,2,3), keepdim=True) + 1e-6)
    
    A_star = c * albedo_raw
    A_star = torch.nan_to_num(A_star, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    
    print("Computing edges...")
    # 1. ccr_edge_gt from train.py
    alb_gray = 0.299 * albedo_raw[:, 0:1] + 0.587 * albedo_raw[:, 1:2] + 0.114 * albedo_raw[:, 2:3]
    edge_mag_train = kornia.filters.sobel(alb_gray)
    ccr_edge_gt_train = torch.clamp(edge_mag_train * 10.0, 0.0, 1.0)
    
    # 2. _albedo_edge_gt from flexible_loss_v17.py
    # Flexible loss uses A_star (the scaled target) rather than the raw unscaled
    ccr_edge_gt_v17 = _albedo_edge_gt_v17(A_star)
    
    # Convert to numpy for plotting
    rgb_np = rgb.squeeze(0).permute(1, 2, 0).cpu().numpy()
    albedo_np = A_star.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    edge_gt_train_np = ccr_edge_gt_train.squeeze(0).squeeze(0).cpu().numpy()
    edge_gt_v17_np = ccr_edge_gt_v17.squeeze(0).squeeze(0).cpu().numpy()
    
    # Plot
    print("Saving visualization...")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(rgb_np)
    axes[0].set_title('RGB')
    axes[0].axis('off')
    
    axes[1].imshow(albedo_np)
    axes[1].set_title('Scaled Albedo (A*)')
    axes[1].axis('off')
    
    axes[2].imshow(edge_gt_train_np, cmap='gray')
    axes[2].set_title('ccr_edge_gt (train.py)\nclamp(sobel(alb) * 10)')
    axes[2].axis('off')
    
    axes[3].imshow(edge_gt_v17_np, cmap='gray')
    axes[3].set_title('_albedo_edge_gt (V17 loss)\nmag / quantile(0.99)')
    axes[3].axis('off')
    
    plt.tight_layout()
    out_path = Path(__file__).parent / 'ccr_edge_comparison.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {out_path}")

if __name__ == "__main__":
    main()
