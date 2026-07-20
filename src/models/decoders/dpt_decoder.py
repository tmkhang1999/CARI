"""DPT (Dense Prediction Transformer) decoder for the DINOv2 backbone.

Reassembles 4 ViT feature maps (all at the patch grid, ~input/14) into a
{/4, /8, /16, /32} pyramid, then fuses them top-down (RefineNet-style) to a
shared full-resolution trunk feature. A small trainable conv stem on the input
RGB injects the high-frequency detail the /14 tokens lack (sharp albedo edges).

This is the Depth-Anything head, made patch-size agnostic by resampling each
level to an explicit target size (computed from the native input resolution)
rather than assuming a /16 token grid. GroupNorm throughout — BatchNorm would
normalise out the per-image illumination statistics IID depends on.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels: int) -> nn.GroupNorm:
    num_groups = min(32, max(1, num_channels // 8))
    return nn.GroupNorm(num_groups, num_channels)


def _target_size(out_size, stride: int):
    H, W = out_size
    return (max(1, round(H / stride)), max(1, round(W / stride)))


class Reassemble(nn.Module):
    """Project a ViT feature map to `out_ch` and resample to a target size."""
    def __init__(self, in_dim: int, out_ch: int):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor, size) -> torch.Tensor:
        x = self.proj(x)
        if x.shape[-2:] != tuple(size):
            x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        return x


class ResidualConvUnit(nn.Module):
    """RefineNet residual conv unit (pre-activation, GroupNorm)."""
    def __init__(self, ch: int):
        super().__init__()
        self.act = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.norm1 = _gn(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.norm2 = _gn(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.norm1(self.conv1(self.act(x)))
        out = self.norm2(self.conv2(self.act(out)))
        return x + out


class FeatureFusionBlock(nn.Module):
    """Merge the running top-down path with a (finer) skip, then upsample.

    path  : coarser feature already in the fusion stream
    skip  : reassembled pyramid level at the next finer scale (or None at top)
    size  : explicit output size (next finer scale)
    """
    def __init__(self, ch: int):
        super().__init__()
        self.rcu_skip = ResidualConvUnit(ch)
        self.rcu_out = ResidualConvUnit(ch)
        self.out_conv = nn.Conv2d(ch, ch, kernel_size=1)

    def forward(self, path, skip=None, size=None):
        if skip is not None:
            if path.shape[-2:] != skip.shape[-2:]:
                path = F.interpolate(path, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            path = path + self.rcu_skip(skip)
        path = self.rcu_out(path)
        if size is not None:
            path = F.interpolate(path, size=size, mode='bilinear', align_corners=False)
        else:
            path = F.interpolate(path, scale_factor=2, mode='bilinear', align_corners=False)
        return self.out_conv(path)


class DetailStem(nn.Module):
    """Tiny trainable conv stem on the gamma-encoded input → /2 detail feature.

    Restores the high-frequency edges DINOv2's /14 tokens cannot represent.
    """
    def __init__(self, out_ch: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),  # H → H/2
            _gn(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, out_ch, 3, padding=1, bias=False),
            _gn(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        x = (rgb.clamp(0.0, 1.0) + 1e-6).pow(1.0 / 2.2)
        return self.net(x)


class DPTTrunk(nn.Module):
    """4-level DPT reassemble + top-down fusion → shared trunk feature at /2.

    Args:
        in_dim     : ViT embed dim (1024 for DINOv2-L).
        feat_ch    : reassemble projection channels.
        fusion_ch  : fusion-path channels.
        out_ch     : trunk output channels (fed to both heads).
        detail_ch  : DetailStem output channels (concatenated at /2).
    """
    STRIDES = (4, 8, 16, 32)   # shallow→deep ViT layer target strides

    def __init__(self, in_dim: int, feat_ch: int = 256, fusion_ch: int = 128,
                 out_ch: int = 128, detail_ch: int = 48):
        super().__init__()
        self.reassemble = nn.ModuleList([Reassemble(in_dim, feat_ch) for _ in range(4)])
        # "scratch" 3×3 to bring each pyramid level to the fusion width
        self.scratch = nn.ModuleList([
            nn.Conv2d(feat_ch, fusion_ch, 3, padding=1, bias=False) for _ in range(4)
        ])
        self.fuse = nn.ModuleList([FeatureFusionBlock(fusion_ch) for _ in range(4)])

        self.detail = DetailStem(detail_ch)
        self.out_conv = nn.Sequential(
            nn.Conv2d(fusion_ch + detail_ch, out_ch, 3, padding=1, bias=False),
            _gn(out_ch), nn.ReLU(inplace=True),
        )
        self.out_channels = out_ch

    def forward(self, dino_feats, rgb, out_size):
        """dino_feats: 4× (B, in_dim, Hp, Wp); rgb: (B,3,H,W); out_size=(H,W)."""
        s4, s8, s16, s32 = (_target_size(out_size, s) for s in self.STRIDES)

        # Reassemble each ViT layer to its pyramid scale, then to fusion width.
        l0 = self.scratch[0](self.reassemble[0](dino_feats[0], s4))    # /4
        l1 = self.scratch[1](self.reassemble[1](dino_feats[1], s8))    # /8
        l2 = self.scratch[2](self.reassemble[2](dino_feats[2], s16))   # /16
        l3 = self.scratch[3](self.reassemble[3](dino_feats[3], s32))   # /32

        # Top-down fusion: /32 → /16 → /8 → /4 → /2
        size_half = _target_size(out_size, 2)
        p = self.fuse[3](l3, skip=None, size=s16)
        p = self.fuse[2](p, skip=l2, size=s8)
        p = self.fuse[1](p, skip=l1, size=s4)
        p = self.fuse[0](p, skip=l0, size=size_half)

        # Inject high-frequency detail at /2.
        d = self.detail(rgb)
        if d.shape[-2:] != p.shape[-2:]:
            d = F.interpolate(d, size=p.shape[-2:], mode='bilinear', align_corners=False)
        return self.out_conv(torch.cat([p, d], dim=1))   # (B, out_ch, H/2, W/2)
