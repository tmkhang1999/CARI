"""
Semantic Total Variation (TV) loss.
Applies TV regularization only on specific semantic classes (e.g., walls, floors, ceilings).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticTVLoss(nn.Module):
    """
    Semantic Total Variation loss for enforcing smoothness on structural surfaces.
    Only applies to pixels belonging to specified semantic classes.
    """
    def __init__(self, target_classes=[1, 2, 22]):
        """
        Args:
            target_classes: List of class IDs to apply TV loss on
                           Default: [1=wall, 2=floor, 22=ceiling] for NYU-40 classes
        """
        super().__init__()
        self.target_classes = target_classes

    def compute_tv(self, x):
        """
        Compute total variation (sum of gradient magnitudes).
        Args:
            x: (N, C, H, W)
        Returns:
            TV value per pixel: (N, 1, H, W)
        """
        # Horizontal gradient
        diff_h = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        diff_h = F.pad(diff_h, (0, 1, 0, 0))  # Pad to match size

        # Vertical gradient
        diff_v = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        diff_v = F.pad(diff_v, (0, 0, 0, 1))  # Pad to match size

        # Sum gradients across channels
        tv = (diff_h + diff_v).mean(dim=1, keepdim=True)  # (N, 1, H, W)

        return tv

    def forward(self, pred, seg_map):
        """
        Args:
            pred: (N, C, H, W) prediction to regularize (e.g., albedo)
            seg_map: (N, H, W) segmentation map with class IDs (long tensor)
        Returns:
            Semantic TV loss (scalar)
        """
        # Create mask for target classes
        mask = torch.zeros_like(seg_map, dtype=torch.float32).unsqueeze(1)  # (N, 1, H, W)

        for class_id in self.target_classes:
            mask = mask + (seg_map.unsqueeze(1) == class_id).float()

        mask = torch.clamp(mask, 0, 1)

        # Compute TV
        tv = self.compute_tv(pred)

        # Apply mask and compute mean
        masked_tv = tv * mask

        # Avoid division by zero
        num_pixels = mask.sum() + 1e-7
        loss = masked_tv.sum() / num_pixels

        return loss


if __name__ == '__main__':
    # Test
    loss_fn = SemanticTVLoss(target_classes=[1, 2, 22])

    pred = torch.rand(2, 3, 256, 256)
    seg_map = torch.randint(0, 40, (2, 256, 256), dtype=torch.long)

    loss = loss_fn(pred, seg_map)
    print(f"Semantic TV Loss: {loss.item()}")

