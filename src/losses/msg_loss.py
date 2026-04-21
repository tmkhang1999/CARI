"""
Multi-scale gradient (MSG) loss for scale-invariant comparison.
Computes L1 loss on gradients at multiple scales.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleGradientLoss(nn.Module):
    """
    Multi-scale gradient loss summed across M scales.
    Enforces structural similarity at multiple resolutions.
    """
    def __init__(self, num_scales=4):
        super().__init__()
        self.num_scales = num_scales

        # Register Sobel kernels as buffers for efficiency (registered as 1,1,3,3)
        kernel_x = torch.tensor([[-1, 0, 1],
                                 [-2, 0, 2],
                                 [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1, -2, -1],
                                 [ 0,  0,  0],
                                 [ 1,  2,  1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('kernel_x', kernel_x, persistent=False)
        self.register_buffer('kernel_y', kernel_y, persistent=False)

    def compute_gradient(self, x):
        """
        Compute image gradients using Sobel-like filters.
        Args:
            x: (N, C, H, W)
        Returns:
            grad_x, grad_y: (N, C, H, W) horizontal and vertical gradients
        """
        # Apply per-channel gradients in one grouped convolution pass.
        C = x.shape[1]
        weight_x = self.kernel_x.repeat(C, 1, 1, 1).to(x.dtype)
        weight_y = self.kernel_y.repeat(C, 1, 1, 1).to(x.dtype)
        grad_x = F.conv2d(x, weight_x, padding=1, groups=C)
        grad_y = F.conv2d(x, weight_y, padding=1, groups=C)

        return grad_x, grad_y

    def forward(self, pred, target, mask=None):
        """
        Args:
            pred: (N, C, H, W) prediction
            target: (N, C, H, W) ground truth
            mask: (N, 1, H, W) valid pixel mask
        Returns:
            Multi-scale gradient loss (scalar)
        """
        diff = pred - target
        
        if mask is None:
            mask = torch.ones(diff.shape[0], 1, diff.shape[2], diff.shape[3], device=diff.device)
            
        loss = 0.0

        for scale in range(self.num_scales):
            # Scale 0 is original size
            if scale > 0:
                scale_factor = 1.0 / (2 ** scale)
                diff_scaled = F.interpolate(
                    diff, 
                    scale_factor=scale_factor, 
                    mode='bilinear', 
                    align_corners=True, 
                    antialias=True
                )
                mask_scaled = F.interpolate(
                    mask, 
                    scale_factor=scale_factor, 
                    mode='bilinear', 
                    align_corners=True, 
                    antialias=True
                )
                # Ensure mask is strictly binary or near-binary after interpolation
                mask_scaled = torch.floor(mask_scaled + 0.001).clamp(0, 1)
            else:
                diff_scaled = diff
                mask_scaled = mask

            # Erosion of the mask could go here, but omitted to prevent overly shrinking 
            # the valid area on small objects unless strictly required by the use case.

            # Compute gradients of difference
            diff_gx, diff_gy = self.compute_gradient(diff_scaled)
            
            # Gradient Magnitude per channel: torch.sqrt(dx^2 + dy^2 + eps)
            grad_mag = torch.sqrt(diff_gx.pow(2) + diff_gy.pow(2) + 1e-8)
            
            # Mean over channels (C)
            grad_mag_mean = torch.mean(grad_mag, dim=1, keepdim=True)
            
            # Mask sum
            mask_sum = torch.sum(mask_scaled)
            
            if mask_sum > 0:
                # Average per pixel diffs across the masked area
                loss += torch.sum(mask_scaled * grad_mag_mean) / (mask_sum * grad_mag.shape[1])

        return loss / self.num_scales


if __name__ == '__main__':
    # Test
    loss_fn = MultiScaleGradientLoss(num_scales=4)
    pred = torch.randn(2, 3, 256, 256)
    target = torch.randn(2, 3, 256, 256)

    loss = loss_fn(pred, target)
    print(f"MSG Loss: {loss.item()}")

