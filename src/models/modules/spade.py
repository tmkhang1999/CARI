"""
Lightweight SPADE normalization module.
Applied in Dec C only, after CCR attention, using raw resized seg map.
(plan Section 2.4)

Input: integer class-ID segmentation map (N, H, W) or (N, 1, H, W).
The seg map is nearest-resized to match the decoder feature spatial size,
then embedded via a small learned conv before producing gamma/beta.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SPADE(nn.Module):
    """
    Standard SPADE (not SPADE ResBlk).
    Accepts integer seg maps, embeds them, and produces per-class affine modulation.
    """

    def __init__(self, num_channels, num_classes=41, hidden_channels=64):
        """
        Args:
            num_channels: number of channels in the decoder feature being normalized.
            num_classes: number of segmentation classes (NYU-40 → 41 including 0).
            hidden_channels: intermediate conv width.
        """
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_channels, affine=False)

        # Learnable class embedding: class ID → num_classes one-hot → hidden
        self.embed = nn.Sequential(
            nn.Conv2d(num_classes, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.gamma = nn.Conv2d(hidden_channels, num_channels, kernel_size=3, padding=1)
        self.beta = nn.Conv2d(hidden_channels, num_channels, kernel_size=3, padding=1)

        self.num_classes = num_classes

    def _prepare_seg(self, seg, target_hw):
        """Convert seg to one-hot float at target spatial size."""
        if seg is None:
            return None

        # Accept (N, H, W) or (N, 1, H, W)
        if seg.dim() == 4:
            seg = seg[:, 0]  # (N, H, W)
        seg = seg.long()

        # Nearest-resize to target spatial size
        if seg.shape[-2:] != target_hw:
            seg = F.interpolate(
                seg.unsqueeze(1).float(), size=target_hw, mode="nearest"
            ).squeeze(1).long()

        # One-hot encode → (N, num_classes, H, W)
        seg = seg.clamp(0, self.num_classes - 1)
        one_hot = F.one_hot(seg, self.num_classes)       # (N, H, W, C)
        one_hot = one_hot.permute(0, 3, 1, 2).float()   # (N, C, H, W)
        return one_hot

    def forward(self, x, seg):
        """
        Args:
            x: (N, C, H, W) decoder features.
            seg: (N, H, W) or (N, 1, H, W) integer class IDs.
        Returns:
            (N, C, H, W) SPADE-normalized features.
        """
        seg_oh = self._prepare_seg(seg, x.shape[-2:])
        if seg_oh is None:
            return x

        h = self.embed(seg_oh)
        gamma = self.gamma(h)
        beta = self.beta(h)
        x_norm = self.norm(x)
        return x_norm * (1.0 + gamma) + beta
