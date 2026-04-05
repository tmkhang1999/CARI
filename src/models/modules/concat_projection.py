import torch
import torch.nn as nn
import torch.nn.functional as F

class ConcatProjection(nn.Module):
    def __init__(self, x_channels, guide_channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(x_channels + guide_channels, x_channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, x_channels),
            nn.GELU(),
        )
        # Initialize as near-identity
        nn.init.zeros_(self.proj[0].weight)
        with torch.no_grad():
            for i in range(x_channels):
                self.proj[0].weight[i, i, 0, 0] = 1.0

    def forward(self, x, guide):
        if guide.shape[-2:] != x.shape[-2:]:
            guide = F.interpolate(guide, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.proj(torch.cat([x, guide], dim=1))
