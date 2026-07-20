import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import sys
import argparse
from pathlib import Path

# Add src to path
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.models.ccr_utils import compute_ccr
from src.losses.flexible_loss import FlexibleLoss
from src.data.hypersim_dataset import HypersimDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Test boundary mask logic on Hypersim frames.")
    parser.add_argument("--sample_idx", type=int, default=0, help="Index of the frame in the split.")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"], help="Dataset split.")
    parser.add_argument("--out_dir", type=str, default="tests/outputs", help="Output directory for visualizations.")
    parser.add_argument("--gamma", type=float, default=2.2, help="Gamma for input visualization.")
    return parser.parse_args()

def test_mask():
    args = parse_args()
    
    hypersim_root = (ROOT_DIR / '../datasets/hypersim').resolve()
    if not hypersim_root.exists():
        print(f"Error: Hypersim root not found at {hypersim_root}")
        return

    print(f"Initializing HypersimDataset on {hypersim_root}...")
    try:
        dataset = HypersimDataset(
            root_dir=str(hypersim_root),
            split=args.split,
            input_size=384,
            cache_max_items=0,
            crop_mode_val="full",
            strict_split=False
        )
    except Exception as e:
        print(f"Error initializing HypersimDataset: {str(e)}")
        return

    if len(dataset) == 0:
        print("Error: No samples found in Hypersim dataset.")
        return

    sample_idx = min(max(0, args.sample_idx), len(dataset) - 1)
    print(f"Loaded dataset with {len(dataset)} samples. Fetching sample index {sample_idx}...")
    
    try:
        sample = dataset[sample_idx]
    except Exception as e:
        print(f"Error loading sample {sample_idx}: {str(e)}")
        return

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Prepare batch inputs
    t_img = sample['rgb'].unsqueeze(0).to(device)                 # (1, 3, 384, 384)
    valid_mask = sample['loss_mask'].unsqueeze(0).to(device)      # (1, 1, 384, 384)
    seg_map = sample['seg'].unsqueeze(0).to(device)               # (1, 1, 384, 384)
    loss_mask = sample['loss_mask'].unsqueeze(0).to(device)       # (1, 1, 384, 384)

    # 2. Compute edge_gt from GT albedo (same as train.py)
    alb_for_edge = sample.get('albedo_scaled', sample['albedo_raw']).unsqueeze(0).to(device)
    alb_gray = 0.299 * alb_for_edge[:, 0:1] + 0.587 * alb_for_edge[:, 1:2] + 0.114 * alb_for_edge[:, 2:3]
    import kornia
    edge_mag = kornia.filters.sobel(alb_gray)
    ccr_edge_gt = torch.clamp(edge_mag * 5.0, 0.0, 1.0)
    
    H, W = t_img.shape[-2:]

    # 3. Exact logic from FlexibleLoss.loss_boundary_consistency
    seg_boundary = FlexibleLoss._detect_segment_boundaries(seg_map)
    if seg_boundary.shape[-2:] != ccr_edge_gt.shape[-2:]:
        seg_boundary = F.interpolate(
            seg_boundary, size=ccr_edge_gt.shape[-2:], mode="nearest"
        )
    
    edge_map = torch.clamp(ccr_edge_gt + seg_boundary * 0.2, 0.0, 1.0)
    non_edge = (1.0 - edge_map) * valid_mask.float()

    # 4. Visualization helpers
    def to_vis(t, gamma=1.0):
        v = t.detach().squeeze().cpu().numpy()
        if v.ndim == 3:
            v = np.transpose(v, (1, 2, 0))
        v = np.clip(v, 0, 1)
        if gamma != 1.0:
            v = np.power(v, 1.0/gamma)
        return (v * 255).astype(np.uint8)

    def create_overlay(input_vis, mask_t):
        overlay = input_vis.copy()
        red = np.zeros_like(overlay)
        red[:, :] = [0, 0, 255]
        mask_np = mask_t.squeeze().cpu().numpy()
        alpha = 0.4
        overlay_final = (overlay * (1.0 - mask_np[:, :, None] * alpha) + 
                         red * (mask_np[:, :, None] * alpha)).astype(np.uint8)
        return overlay_final

    # 5. Create visualizations
    os.makedirs(args.out_dir, exist_ok=True)
    
    input_vis = cv2.cvtColor(to_vis(t_img, args.gamma), cv2.COLOR_RGB2BGR)
    alb_vis = cv2.cvtColor(to_vis(alb_for_edge, args.gamma), cv2.COLOR_RGB2BGR)
    
    edge_gt_vis = cv2.cvtColor(to_vis(ccr_edge_gt), cv2.COLOR_GRAY2BGR)
    seg_boundary_overlay = create_overlay(input_vis, seg_boundary)
    penalize_vis = cv2.cvtColor(to_vis(non_edge), cv2.COLOR_GRAY2BGR)

    # 6. Create single unified strip
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(W, H) / 500.0)
    thickness = max(1, int(font_scale * 2.0))

    strip = np.zeros((H, W * 5, 3), dtype=np.uint8)
    strip[:, 0:W] = input_vis
    strip[:, W:2*W] = alb_vis
    strip[:, 2*W:3*W] = edge_gt_vis
    strip[:, 3*W:4*W] = seg_boundary_overlay
    strip[:, 4*W:5*W] = penalize_vis

    cv2.putText(strip, "Input", (10, 30), font, font_scale, (255, 255, 255), thickness)
    cv2.putText(strip, "Albedo GT", (W + 10, 30), font, font_scale, (255, 255, 255), thickness)
    cv2.putText(strip, "ccr_edge_gt (Albedo Sobel)", (2*W + 10, 30), font, font_scale, (255, 255, 255), thickness)
    cv2.putText(strip, "Seg Boundaries", (3*W + 10, 30), font, font_scale, (255, 255, 255), thickness)
    cv2.putText(strip, "Smooth Mask (non_edge)", (4*W + 10, 30), font, font_scale, (255, 255, 255), thickness)

    out_file = os.path.join(args.out_dir, "loss_boundary_consistency_strip.png")
    cv2.imwrite(out_file, strip)

    print(f"Results saved to {args.out_dir}")
    print(f"Visualization: {out_file}")

    # 7. Verify the actual loss function instantiation and execution
    try:
        config = {
            'lambda_boundary': 1.0,
            'lambda_msg': 0.0,
            'lambda_dssim': 0.0
        }
        loss_fn = FlexibleLoss(config)
        
        # Create a dummy predicted albedo (e.g. clone of t_img)
        a_d_pred = t_img.clone()
        
        # Run boundary consistency loss with and without seg_map
        loss_fixed = loss_fn.loss_boundary_consistency(a_d_pred, ccr_edge_gt, loss_mask, seg_map=None)
        loss_adaptive = loss_fn.loss_boundary_consistency(a_d_pred, ccr_edge_gt, loss_mask, seg_map=seg_map)
        
        print("\nVerification of FlexibleLoss execution on Hypersim sample:")
        print(f"Boundary Consistency Loss (Fixed Threshold 0.25): {loss_fixed.item():.6f}")
        print(f"Boundary Consistency Loss (Adaptive Seg Map):      {loss_adaptive.item():.6f}")
    except Exception as e:
        print(f"\nWarning: Could not complete loss execution verification: {str(e)}")

if __name__ == "__main__":
    test_mask()
