"""Residual attention block: x * (1 + sigmoid(proj(prior)))."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualAttention(nn.Module):
    def __init__(self, x_channels, prior_channels):
        super().__init__()
        self.proj = nn.Conv2d(prior_channels, x_channels, kernel_size=1, bias=True)

    def forward(self, x, prior):
        if prior is None:
            return x
        if prior.shape[-2:] != x.shape[-2:]:
            prior = F.interpolate(prior, size=x.shape[-2:], mode="bilinear", align_corners=False)
        gate = torch.sigmoid(self.proj(prior))
        return x * (1.0 + gate)
