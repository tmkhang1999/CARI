import sys
from pathlib import Path
import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import kornia

# Add src to path
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / 'src'
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


import glob
import numpy as np
from data.hypersim_dataset import _load_hdf5, _compute_tonemap_scale, _tonemap_linear
from models.iid_utils import rgb_to_iuv

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
    illum_path = Path(str(color_path).replace('.color.hdf5', '.diffuse_illumination.hdf5'))
    
    print(f"Loading files from {color_path.parent.parent.parent.name}...")
    rgb_raw = _load_hdf5(str(color_path))
    alb_raw = _load_hdf5(str(albedo_path))
    ill_raw = _load_hdf5(str(illum_path))
    
    rgb_raw = np.nan_to_num(rgb_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    alb_raw = np.nan_to_num(alb_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32).clip(0.0)
    ill_raw = np.nan_to_num(ill_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32).clip(0.0)
    
    tonemap_scale = _compute_tonemap_scale(rgb_raw)
    rgb_tm = _tonemap_linear(rgb_raw, scale=tonemap_scale)
    ill_norm = ill_raw / tonemap_scale
    
    rgb_t = torch.from_numpy(rgb_tm).permute(2, 0, 1).unsqueeze(0)
    alb_t = torch.from_numpy(alb_raw).permute(2, 0, 1).unsqueeze(0)
    ill_t = torch.from_numpy(ill_norm).permute(2, 0, 1).unsqueeze(0)
    
    rgb = TF.resize(rgb_t, (384, 384), antialias=True)
    albedo_raw = TF.resize(alb_t, (384, 384), antialias=True)
    illum_raw = TF.resize(ill_t, (384, 384), antialias=True)
    
    # Same scaling logic as train.py
    c = torch.sum(rgb * albedo_raw, dim=(1,2,3), keepdim=True) / \
        (torch.sum(albedo_raw * albedo_raw, dim=(1,2,3), keepdim=True) + 1e-6)
    
    A_star = c * albedo_raw
    A_star = torch.nan_to_num(A_star, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    
    # Compute Edge GT
    print("Computing edges...")
    alb_gray = 0.299 * albedo_raw[:, 0:1] + 0.587 * albedo_raw[:, 1:2] + 0.114 * albedo_raw[:, 2:3]
    edge_mag = kornia.filters.sobel(alb_gray)
    # Soft edge target: 0.1 Sobel -> 1.0 (strong), 0.05 -> 0.5 (weak), 0 -> 0
    ccr_edge_gt = torch.clamp(edge_mag * 10.0, 0.0, 1.0)
    
    # Convert to numpy for plotting
    rgb_np = rgb.squeeze(0).permute(1, 2, 0).cpu().numpy()
    albedo_np = A_star.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    illum_np = illum_raw.squeeze(0).permute(1, 2, 0).clamp(0, 2).cpu().numpy()
    edge_mag_np = edge_mag.squeeze(0).squeeze(0).cpu().numpy()
    edge_gt_np = ccr_edge_gt.squeeze(0).squeeze(0).cpu().numpy()
    
    # Plot
    print("Saving visualization...")
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    axes[0].imshow(rgb_np)
    axes[0].set_title('RGB')
    axes[0].axis('off')
    
    axes[1].imshow(albedo_np)
    axes[1].set_title('Scaled Albedo (A*)')
    axes[1].axis('off')
    
    axes[2].imshow(illum_np / 2.0)
    axes[2].set_title('Raw Illumination')
    axes[2].axis('off')
    
    axes[3].imshow(edge_mag_np, cmap='magma')
    axes[3].set_title('Sobel Magnitude on Albedo')
    axes[3].axis('off')
    
    axes[4].imshow(edge_gt_np, cmap='gray')
    axes[4].set_title('Soft CCR Edge GT (clamp(mag*10))')
    axes[4].axis('off')
    
    plt.tight_layout()
    out_path = Path(__file__).parent / 'ccr_edge_visualization.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {out_path}")

if __name__ == "__main__":
    main()
