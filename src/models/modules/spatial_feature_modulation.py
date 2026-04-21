"""Spatial Feature Modulation (SFM) with affine conditioning."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialFeatureModulation(nn.Module):
    """
    Affine feature modulation: x' = x * (1 + gamma) + beta.
    Supports both amplification and suppression (gamma in [-1, 1] after tanh).
    """

    def __init__(self, x_channels, prior_channels, hidden_channels=None, gamma_scale=1.0):
        super().__init__()
        self.gamma_scale = float(gamma_scale)

        if hidden_channels is None:
            hidden_channels = min(prior_channels, x_channels // 2)
            hidden_channels = max(hidden_channels, 64)

        self.proj = nn.Sequential(
            nn.Conv2d(prior_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            # Final projection predicts gamma and beta (2 * x_channels)
            nn.Conv2d(hidden_channels, x_channels * 2, kernel_size=3, padding=1),
        )

        # Identity initialization at step 0: gamma=0, beta=0.
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x, prior):
        if prior is None:
            return x
        if prior.shape[-2:] != x.shape[-2:]:
            prior = F.interpolate(prior, size=x.shape[-2:], mode="bilinear", align_corners=False)

        affine = self.proj(prior)
        gamma, beta = affine.chunk(2, dim=1)
        gamma = torch.tanh(gamma) * self.gamma_scale
        return x * (1.0 + gamma) + beta
