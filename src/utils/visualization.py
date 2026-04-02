"""
Visualization utilities for intrinsic decomposition outputs.
"""

import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def apply_gamma(img, gamma=2.2):
    """Apply gamma correction for visualization."""
    return torch.clamp(img, 0, 1) ** (1/gamma)


def visualize_predictions(rgb, predictions, save_path=None):
    """
    Create visualization grid of all predictions.

    Args:
        rgb: (N, 3, H, W) input RGB
        predictions: Dict with keys ['d_g', 'xi', 'c', 's_c', 'a_d', 'pi']
        save_path: Optional path to save image
    """
    batch_size = rgb.shape[0]

    for i in range(batch_size):
        # Apply gamma for visualization
        rgb_vis = apply_gamma(rgb[i].cpu())
        s_g_vis = apply_gamma(predictions['d_g'][i].cpu()).repeat(3, 1, 1)
        c_vis = torch.clamp(predictions['c'][i].cpu() / 2.0, 0, 1)  # Normalize chroma for vis
        s_c_vis = apply_gamma(predictions['s_c'][i].cpu())
        a_d_vis = apply_gamma(predictions['a_d'][i].cpu())
        s_d_vis = apply_gamma(predictions['pi'][i].cpu())

        # Create grid
        images = [rgb_vis, s_g_vis, c_vis, s_c_vis, a_d_vis, s_d_vis]
        labels = ['Input RGB', 'Gray Shading', 'Chroma', 'Color Shading', 'Albedo', 'Diffuse Shading']

        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, (img, label) in enumerate(zip(images, labels)):
            img_np = img.permute(1, 2, 0).numpy()
            axes[idx].imshow(img_np)
            axes[idx].set_title(label, fontsize=12)
            axes[idx].axis('off')

        plt.tight_layout()

        if save_path:
            base_path = save_path.rsplit('.', 1)[0]
            ext = save_path.rsplit('.', 1)[1] if '.' in save_path else 'png'
            save_file = f"{base_path}_sample_{i}.{ext}"
            plt.savefig(save_file, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_file}")
        else:
            plt.show()

        plt.close()


def create_comparison_grid(rgb, predictions, ground_truths, save_path=None):
    """
    Create comparison grid with predictions and ground truths.

    Args:
        rgb: (N, 3, H, W) input RGB
        predictions: Dict with predicted outputs
        ground_truths: Dict with ground truth outputs
        save_path: Optional path to save image
    """
    batch_size = rgb.shape[0]

    for i in range(batch_size):
        rgb_vis = apply_gamma(rgb[i].cpu())

        # Predictions
        a_d_pred = apply_gamma(predictions['a_d'][i].cpu())
        s_g_pred = apply_gamma(predictions['d_g'][i].cpu()).repeat(3, 1, 1)
        s_d_pred = apply_gamma(predictions['pi'][i].cpu())

        # Ground truths
        a_d_gt = apply_gamma(ground_truths['a_d'][i].cpu())
        s_g_gt = apply_gamma(ground_truths['s_g'][i].cpu()).repeat(3, 1, 1)
        s_d_gt = apply_gamma(ground_truths['pi'][i].cpu())

        # Error maps
        a_d_error = torch.abs(a_d_pred - a_d_gt).mean(dim=0, keepdim=True).repeat(3, 1, 1)
        s_g_error = torch.abs(s_g_pred - s_g_gt).mean(dim=0, keepdim=True).repeat(3, 1, 1)
        s_d_error = torch.abs(s_d_pred - s_d_gt).mean(dim=0, keepdim=True).repeat(3, 1, 1)

        # Create grid layout
        images = [
            rgb_vis,
            a_d_gt, a_d_pred, a_d_error,
            s_g_gt, s_g_pred, s_g_error,
            s_d_gt, s_d_pred, s_d_error
        ]

        labels = [
            'Input RGB',
            'GT Albedo', 'Pred Albedo', 'Albedo Error',
            'GT Gray Shading', 'Pred Gray Shading', 'Shading Error',
            'GT Diffuse Shading', 'Pred Diffuse Shading', 'Diffuse Error'
        ]

        # Create figure
        fig, axes = plt.subplots(4, 3, figsize=(15, 18))
        axes = axes.flatten()

        # First row: input only
        img_np = images[0].permute(1, 2, 0).numpy()
        axes[0].imshow(img_np)
        axes[0].set_title(labels[0], fontsize=12)
        axes[0].axis('off')
        axes[1].axis('off')
        axes[2].axis('off')

        # Remaining rows
        for idx in range(1, len(images)):
            img_np = images[idx].permute(1, 2, 0).numpy()
            axes[idx + 2].imshow(img_np)
            axes[idx + 2].set_title(labels[idx], fontsize=12)
            axes[idx + 2].axis('off')

        plt.tight_layout()

        if save_path:
            base_path = save_path.rsplit('.', 1)[0]
            ext = save_path.rsplit('.', 1)[1] if '.' in save_path else 'png'
            save_file = f"{base_path}_comparison_{i}.{ext}"
            plt.savefig(save_file, dpi=150, bbox_inches='tight')
            print(f"Saved comparison to {save_file}")
        else:
            plt.show()

        plt.close()


def save_tensor_as_image(tensor, path, apply_gamma_correction=True):
    """
    Save a tensor as an image file.

    Args:
        tensor: (C, H, W) or (H, W) tensor
        path: Save path
        apply_gamma_correction: Whether to apply gamma correction
    """
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0).repeat(3, 1, 1)

    if apply_gamma_correction:
        tensor = apply_gamma(tensor)

    vutils.save_image(tensor, path)


if __name__ == '__main__':
    # Test visualization
    print("Testing visualization utilities...")

    # Create dummy data
    rgb = torch.rand(2, 3, 256, 256)
    predictions = {
        'd_g': torch.rand(2, 1, 256, 256),
        'xi': torch.rand(2, 2, 256, 256),
        'c': torch.rand(2, 3, 256, 256),
        's_c': torch.rand(2, 3, 256, 256),
        'a_d': torch.rand(2, 3, 256, 256),
        'pi': torch.rand(2, 3, 256, 256)
    }

    # Test prediction visualization
    visualize_predictions(rgb, predictions, save_path='test_vis.png')

    print("Visualization test completed!")

