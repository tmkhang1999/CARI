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

    def compute_gradient(self, x):
        """
        Compute image gradients using Sobel-like filters.
        Args:
            x: (N, C, H, W)
        Returns:
            grad_x, grad_y: (N, C, H, W) horizontal and vertical gradients
        """
        # Sobel kernels
        kernel_x = torch.tensor([[-1, 0, 1],
                                 [-2, 0, 2],
                                 [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)

        kernel_y = torch.tensor([[-1, -2, -1],
                                 [ 0,  0,  0],
                                 [ 1,  2,  1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)

        # Apply per channel
        C = x.shape[1]
        grad_x = []
        grad_y = []

        for c in range(C):
            gx = F.conv2d(x[:, c:c+1, :, :], kernel_x, padding=1)
            gy = F.conv2d(x[:, c:c+1, :, :], kernel_y, padding=1)
            grad_x.append(gx)
            grad_y.append(gy)

        grad_x = torch.cat(grad_x, dim=1)
        grad_y = torch.cat(grad_y, dim=1)

        return grad_x, grad_y

    def forward(self, pred, target):
        """
        Args:
            pred: (N, C, H, W) prediction
            target: (N, C, H, W) ground truth
        Returns:
            Multi-scale gradient loss (scalar)
        """
        loss = 0.0

        for scale in range(self.num_scales):
            # Downsample if not first scale
            if scale > 0:
                factor = 2 ** scale
                pred_scaled = F.avg_pool2d(pred, kernel_size=factor, stride=factor)
                target_scaled = F.avg_pool2d(target, kernel_size=factor, stride=factor)
            else:
                pred_scaled = pred
                target_scaled = target

            # Compute gradients
            pred_gx, pred_gy = self.compute_gradient(pred_scaled)
            target_gx, target_gy = self.compute_gradient(target_scaled)

            # L1 loss on gradients
            loss += F.l1_loss(pred_gx, target_gx) + F.l1_loss(pred_gy, target_gy)

        return loss / self.num_scales


if __name__ == '__main__':
    # Test
    loss_fn = MultiScaleGradientLoss(num_scales=4)
    pred = torch.randn(2, 3, 256, 256)
    target = torch.randn(2, 3, 256, 256)

    loss = loss_fn(pred, target)
    print(f"MSG Loss: {loss.item()}")

