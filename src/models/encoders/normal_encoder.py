"""
Normal encoder with depthwise-separable pyramid blocks.
Architecture per stage: DWConv3x3(stride=2, pad=1) → PWConv1x1 → LayerNorm → GELU
(matches project plan Section 2.1)
"""

import torch
import torch.nn as nn


class _DWBlock(nn.Module):
    """Single depth-wise separable block matching the plan specification."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3, stride=2, padding=1,
            groups=in_channels, bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        # Plan specifies LayerNorm (channel-last), not BatchNorm.
        # GroupNorm(1, C) is equivalent to LayerNorm over (C, H, W) dimensions.
        # self.norm = nn.GroupNorm(1, out_channels)
        # self.act = nn.GELU()
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        return self.act(x)


class NormalEncoder(nn.Module):
    """Produces multi-scale features from normals: [H/2, H/4, H/8, H/16]."""

    def __init__(self, in_channels=3, channels=(64, 128, 256, 512)):
        super().__init__()
        c1, c2, c3, c4 = channels

        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=False),
            # nn.GroupNorm(1, c1),
            # nn.GELU(),
            nn.BatchNorm2d(c1),
            nn.GELU(),
        )
        
        self.block1 = _DWBlock(c1, c1)
        self.block2 = _DWBlock(c1, c2)
        self.block3 = _DWBlock(c2, c3)
        self.block4 = _DWBlock(c3, c4)

    def forward(self, normals):
        x = self.input_proj(normals)
        n1 = self.block1(x)
        n2 = self.block2(n1)
        n3 = self.block3(n2)
        n4 = self.block4(n3)
        return [n1, n2, n3, n4]
