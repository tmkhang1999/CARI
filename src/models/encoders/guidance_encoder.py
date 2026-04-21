"""Guidance encoder mirroring NormalEncoder with configurable input channels."""

import torch.nn as nn


class _DWBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        # self.norm = nn.GroupNorm(1, out_channels)
        # self.act = nn.GELU()
        self.norm = nn.BatchNorm2d(out_channels)  
        self.act = nn.ReLU(inplace=True)         
  

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        return self.act(x)


class GuidanceEncoder(nn.Module):
    """Produces multi-scale guidance features at [H/2, H/4, H/8, H/16]."""

    def __init__(self, in_channels, channels=(64, 128, 256, 512)):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.block1 = _DWBlock(in_channels, c1)
        self.block2 = _DWBlock(c1, c2)
        self.block3 = _DWBlock(c2, c3)
        self.block4 = _DWBlock(c3, c4)

    def forward(self, x):
        g1 = self.block1(x)
        g2 = self.block2(g1)
        g3 = self.block3(g2)
        g4 = self.block4(g3)
        return [g1, g2, g3, g4]
