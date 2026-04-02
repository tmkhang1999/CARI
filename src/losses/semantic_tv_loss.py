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
    def __init__(self, target_classes=[1, 22]):
        """
        Args:
            target_classes: List of class IDs to apply TV loss on
                           Default: [1=wall, 22=ceiling] for NYU-40 classes
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
    loss_fn = SemanticTVLoss(target_classes=[1, 22])

    pred = torch.rand(2, 3, 256, 256)
    seg_map = torch.randint(0, 40, (2, 256, 256), dtype=torch.long)

    loss = loss_fn(pred, seg_map)
    print(f"Semantic TV Loss: {loss.item()}")
    

class NormalGuidedTVLoss(nn.Module):
    """
    Normal-guided Total Variation loss.
    Weights the spatial gradients of the predicted Albedo by the cosine similarity
    of the neighboring surface normals.
    Restores the piecewise-constant prior without needing semantic labels.
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def compute_tv_with_normals(self, x, normals):
        # 1. Horizontal Gradients
        diff_h = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        
        # 2. Horizontal Normal Similarity
        # We compute similarity for the (W-1) pairs
        cos_sim_h = F.cosine_similarity(normals[:, :, :, 1:], normals[:, :, :, :-1], dim=1, eps=self.eps)
        
        # Use a power to make the mask more sensitive to small normal changes (corners)
        weight_h = torch.pow(F.relu(cos_sim_h), 8).unsqueeze(1) 
        
        # 3. Apply Weight THEN Pad (or pad weight with 1.0)
        # This ensures the last column doesn't accidentally lose its TV smoothing
        tv_h = diff_h * weight_h
        tv_h = F.pad(tv_h, (0, 1, 0, 0), mode='replicate')

        # 4. Vertical Gradients
        diff_v = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        cos_sim_v = F.cosine_similarity(normals[:, :, 1:, :], normals[:, :, :-1, :], dim=1, eps=self.eps)
        weight_v = torch.pow(F.relu(cos_sim_v), 8).unsqueeze(1)
        
        tv_v = diff_v * weight_v
        tv_v = F.pad(tv_v, (0, 0, 0, 1), mode='replicate')

        # Return mean across channels (RGB) to get (N, 1, H, W)
        return tv_h.mean(dim=1, keepdim=True) + tv_v.mean(dim=1, keepdim=True)

    def forward(self, pred, normals, valid_mask=None):
        """
        Args:
            pred: (N, C, H, W) prediction
            normals: (N, 3, H, W) normals
            valid_mask: (N, 1, H, W) boolean mask
        """
        tv = self.compute_tv_with_normals(pred, normals)
        if valid_mask is not None:
            mask = valid_mask.float()
            return (tv * mask).sum() / (mask.sum() + self.eps)
        return tv.mean()
