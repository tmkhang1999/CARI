"""Spatial Feature Modulation (SFM) with affine conditioning."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialFeatureModulation(nn.Module):
    def __init__(self, x_channels, prior_channels, hidden_channels=None, gamma_scale=1.0, use_seg=False, num_seg_classes=41):
        super().__init__()
        self.gamma_scale = float(gamma_scale)
        self.use_seg = use_seg

        if hidden_channels is None:
            hidden_channels = min(prior_channels, x_channels // 2)
            hidden_channels = max(hidden_channels, 64)

        # If using segmentation, we will embed it and concatenate it with the prior
        input_channels = prior_channels
        if use_seg:
            self.seg_embed = nn.Sequential(
                nn.Conv2d(num_seg_classes, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            input_channels += 32 # Add 32 channels for the seg embedding

        self.proj = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels), #nn.GroupNorm(8, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, x_channels * 2, kernel_size=3, padding=1),
        )

        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x, prior, seg=None):
        if prior is not None and prior.shape[-2:] != x.shape[-2:]:
            prior = F.interpolate(prior, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if self.use_seg and seg is not None:
            # One-hot encode and resize segmentation
            seg = seg.long()
            if seg.dim() == 4: seg = seg[:, 0]
            if seg.shape[-2:] != x.shape[-2:]:
                seg = F.interpolate(seg.unsqueeze(1).float(), size=x.shape[-2:], mode="nearest").squeeze(1).long()
            
            seg_oh = F.one_hot(seg.clamp(0, 40), 41).permute(0, 3, 1, 2).float()
            seg_feat = self.seg_embed(seg_oh)
            
            # FUSE CCR and Segmentation!
            prior = torch.cat([prior, seg_feat], dim=1) if prior is not None else seg_feat

        affine = self.proj(prior)
        gamma, beta = affine.chunk(2, dim=1)
        gamma = torch.tanh(gamma) * self.gamma_scale
        return x * (1.0 + gamma) + beta
