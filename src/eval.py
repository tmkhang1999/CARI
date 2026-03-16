"""
Evaluation script for Stage 1 intrinsic decomposition.
Computes scale-invariant metrics following Careaga & Aksoy 2023.
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

# Make src importable
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models import (
    IntrinsicDecompositionV1,
    IntrinsicDecompositionV2,
    IntrinsicDecompositionV3,
    IntrinsicDecompositionV4,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Intrinsic Decomposition Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='hypersim',
                       choices=['hypersim', 'interiornet', 'midintrinsic'],
                       help='Dataset to evaluate on')
    parser.add_argument('--split', type=str, default='val',
                       choices=['train', 'val', 'test'],
                       help='Dataset split')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--save_outputs', action='store_true',
                       help='Save prediction images')
    parser.add_argument('--output_dir', type=str, default='outputs/eval',
                       help='Directory to save outputs')
    return parser.parse_args()


def scale_invariant_mse(pred, target):
    """
    Compute scale-invariant MSE (LMSE in log space).
    Normalizes both pred and target by their mean before comparison.
    """
    eps = 1e-7

    # Normalize by mean (scale-invariant)
    pred_norm = pred / (pred.mean() + eps)
    target_norm = target / (target.mean() + eps)

    # Compute MSE in log space
    log_pred = torch.log(pred_norm + eps)
    log_target = torch.log(target_norm + eps)

    lmse = ((log_pred - log_target) ** 2).mean()

    return lmse.item()


def compute_rmse(pred, target):
    """Compute scale-invariant RMSE."""
    eps = 1e-7
    pred_norm = pred / (pred.mean() + eps)
    target_norm = target / (target.mean() + eps)

    rmse = torch.sqrt(((pred_norm - target_norm) ** 2).mean())
    return rmse.item()


def compute_ssim(pred, target):
    """Compute SSIM on normalized images."""
    eps = 1e-7

    # Normalize
    pred_norm = pred / (pred.mean() + eps)
    target_norm = target / (target.mean() + eps)

    # Convert to numpy
    pred_np = pred_norm.cpu().numpy()
    target_np = target_norm.cpu().numpy()

    # Compute SSIM per channel, then average
    ssim_values = []
    for c in range(pred_np.shape[0]):
        s = ssim(target_np[c], pred_np[c], data_range=target_np[c].max() - target_np[c].min())
        ssim_values.append(s)

    return np.mean(ssim_values)


def build_stage1_model(model_cfg):
    version = int(model_cfg.get("version", 1))
    model_config = {
        "z_channels": model_cfg.get("z_channels", 1536),
        "freeze_stages": model_cfg.get("freeze_stages", [1, 2]),
        "backbone": model_cfg.get("backbone", "convnext_large"),
        "pretrained": model_cfg.get("pretrained", True),
    }
    model_map = {
        1: IntrinsicDecompositionV1,
        2: IntrinsicDecompositionV2,
        3: IntrinsicDecompositionV3,
        4: IntrinsicDecompositionV4,
    }
    if version not in model_map:
        raise ValueError(f"Unsupported Stage1 version: {version}")
    return model_map[version](model_config)


def evaluate_model(model, dataloader, device, save_outputs=False, output_dir=None):
    """
    Evaluate model on dataset.

    Returns:
        Dictionary of metrics
    """
    model.eval()

    metrics = {
        's_g_lmse': [],
        's_g_rmse': [],
        's_g_ssim': [],
        'a_d_lmse': [],
        'a_d_rmse': [],
        'a_d_ssim': [],
        's_d_lmse': [],
        's_d_rmse': [],
        's_d_ssim': [],
        'c_mse': []
    }

    if save_outputs:
        os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            # Move data to device
            rgb = batch['rgb'].to(device)
            albedo_gt = batch.get('albedo')
            s_g_gt = batch.get('s_g')
            c_gt = batch.get('c')
            s_d_gt = batch.get('s_d')

            m_diffuse = batch.get('M_diffuse')
            if m_diffuse is None:
                m_diffuse = batch.get('m_diffuse')
            if m_diffuse is not None:
                m_diffuse = m_diffuse.to(device)

            # Forward pass
            if m_diffuse is None:
                predictions = model(rgb)
            else:
                predictions = model(rgb, m_diffuse=m_diffuse)

            # Extract predictions
            s_g_pred = predictions['s_g']
            c_pred = predictions['c']
            a_d_pred = predictions['a_d']
            s_d_pred = predictions['s_d']

            # Compute metrics per sample
            batch_size = rgb.shape[0]
            for i in range(batch_size):
                # Gray shading metrics
                if s_g_gt is not None:
                    s_g_gt_i = s_g_gt[i].to(device)
                    metrics['s_g_lmse'].append(scale_invariant_mse(s_g_pred[i], s_g_gt_i))
                    metrics['s_g_rmse'].append(compute_rmse(s_g_pred[i], s_g_gt_i))
                    metrics['s_g_ssim'].append(compute_ssim(s_g_pred[i], s_g_gt_i))

                # Albedo metrics
                if albedo_gt is not None:
                    albedo_gt_i = albedo_gt[i].to(device)
                    metrics['a_d_lmse'].append(scale_invariant_mse(a_d_pred[i], albedo_gt_i))
                    metrics['a_d_rmse'].append(compute_rmse(a_d_pred[i], albedo_gt_i))
                    metrics['a_d_ssim'].append(compute_ssim(a_d_pred[i], albedo_gt_i))

                # Diffuse shading metrics
                if s_d_gt is not None:
                    s_d_gt_i = s_d_gt[i].to(device)
                    metrics['s_d_lmse'].append(scale_invariant_mse(s_d_pred[i], s_d_gt_i))
                    metrics['s_d_rmse'].append(compute_rmse(s_d_pred[i], s_d_gt_i))
                    metrics['s_d_ssim'].append(compute_ssim(s_d_pred[i], s_d_gt_i))

                # Chroma metrics
                if c_gt is not None:
                    c_gt_i = c_gt[i].to(device)
                    metrics['c_mse'].append(((c_pred[i] - c_gt_i) ** 2).mean().item())

            # Save outputs if requested
            if save_outputs and batch_idx < 10:  # Save first 10 batches
                save_predictions(
                    rgb, predictions, batch,
                    output_dir, batch_idx
                )

    # Compute average metrics
    avg_metrics = {}
    for key, values in metrics.items():
        if len(values) > 0:
            avg_metrics[key] = np.mean(values)
            avg_metrics[f'{key}_std'] = np.std(values)

    return avg_metrics


def save_predictions(rgb, predictions, ground_truths, output_dir, batch_idx):
    """Save prediction visualizations."""
    import torchvision.utils as vutils
    from PIL import Image

    batch_size = rgb.shape[0]

    for i in range(batch_size):
        # Apply gamma for visualization
        rgb_vis = torch.clamp(rgb[i].cpu(), 0, 1) ** (1/2.2)
        a_d_vis = torch.clamp(predictions['a_d'][i].cpu(), 0, 1) ** (1/2.2)
        s_g_vis = torch.clamp(predictions['s_g'][i].cpu(), 0, 1).repeat(3, 1, 1) ** (1/2.2)
        s_c_vis = torch.clamp(predictions['s_c'][i].cpu(), 0, 1) ** (1/2.2)
        s_d_vis = torch.clamp(predictions['s_d'][i].cpu(), 0, 1) ** (1/2.2)

        # Create grid
        images = [rgb_vis, a_d_vis, s_g_vis, s_c_vis, s_d_vis]
        grid = vutils.make_grid(images, nrow=5, padding=10, pad_value=1.0)

        # Save
        save_path = os.path.join(output_dir, f'batch_{batch_idx:04d}_sample_{i:02d}.png')
        vutils.save_image(grid, save_path)


def main():
    args = parse_args()

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint.get('config', {})

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Build model
    model = build_stage1_model(config.get("model", {})).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")

    # Build dataloader
    print(f"Loading {args.dataset} dataset ({args.split} split)...")
    # TODO: Implement actual dataloader
    dataloader = None

    if dataloader is None:
        print("Error: Dataset loader not implemented yet.")
        print("Please implement dataset loading for evaluation.")
        return

    # Evaluate
    print("\nStarting evaluation...")
    metrics = evaluate_model(
        model, dataloader, device,
        save_outputs=args.save_outputs,
        output_dir=args.output_dir
    )

    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    print("\nGray Shading (S_g):")
    if 's_g_lmse' in metrics:
        print(f"  LMSE: {metrics['s_g_lmse']:.6f} +/- {metrics.get('s_g_lmse_std', 0):.6f}")
        print(f"  RMSE: {metrics['s_g_rmse']:.6f} +/- {metrics.get('s_g_rmse_std', 0):.6f}")
        print(f"  SSIM: {metrics['s_g_ssim']:.4f} +/- {metrics.get('s_g_ssim_std', 0):.4f}")

    print("\nDiffuse Albedo (A_d):")
    if 'a_d_lmse' in metrics:
        print(f"  LMSE: {metrics['a_d_lmse']:.6f} +/- {metrics.get('a_d_lmse_std', 0):.6f}")
        print(f"  RMSE: {metrics['a_d_rmse']:.6f} +/- {metrics.get('a_d_rmse_std', 0):.6f}")
        print(f"  SSIM: {metrics['a_d_ssim']:.4f} +/- {metrics.get('a_d_ssim_std', 0):.4f}")

    print("\nDiffuse Shading (S_d):")
    if 's_d_lmse' in metrics:
        print(f"  LMSE: {metrics['s_d_lmse']:.6f} +/- {metrics.get('s_d_lmse_std', 0):.6f}")
        print(f"  RMSE: {metrics['s_d_rmse']:.6f} +/- {metrics.get('s_d_rmse_std', 0):.6f}")
        print(f"  SSIM: {metrics['s_d_ssim']:.4f} +/- {metrics.get('s_d_ssim_std', 0):.4f}")

    print("\nChroma (C):")
    if 'c_mse' in metrics:
        print(f"  MSE:  {metrics['c_mse']:.6f} +/- {metrics.get('c_mse_std', 0):.6f}")

    print("\n" + "="*60)

    if args.save_outputs:
        print(f"\nOutputs saved to {args.output_dir}")


if __name__ == '__main__':
    main()

