import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels: int) -> nn.GroupNorm:
    """GroupNorm with 8 channels per group, max 32 groups.

    GroupNorm normalises within each sample independently (no cross-batch
    contamination), preserving per-image illumination statistics that BatchNorm
    would zero out. Critical for IID where global intensity IS the signal.
    """
    num_groups = min(32, max(1, num_channels // 8))
    return nn.GroupNorm(num_groups, num_channels)


class ModernDecoderBlock(nn.Module):
    """Double-conv block with GroupNorm, processing at lower resolution before upsample."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mix = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _gn(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _gn(out_channels),
            nn.ReLU(inplace=True),
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        return self.up(self.mix(x))


class MidScaleCoupling(nn.Module):
    """Injects shading H/8 features into the albedo H/8 decoder path.

    Runs after both decoders have computed their H/8 representations. Zero-init
    output → identity at init. During backprop the albedo loss flows through this
    module into the shading decoder's early-stage parameters, so both decoders
    jointly learn to produce H/8 features that help the other avoid absorbing
    the wrong edge type (material vs. shading).
    """
    def __init__(self, channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, a_h8: torch.Tensor, s_h8: torch.Tensor) -> torch.Tensor:
        return a_h8 + self.proj(s_h8)


class ModernProgressiveDecoder(nn.Module):
    """Four-stage U-Net decoder with optional per-stage ops and mid-decode access.

    Stage ops are indexed by decoder stage:
      ops[0]: applied after dec3 output (H/16), before H/16 skip concat
      ops[1]: applied after dec2 output (H/8),  before H/8  skip concat
      ops[2]: applied after dec1 output (H/4),  before H/4  skip concat
      ops[3]: applied after dec0 output (H/2),  before final upsample

    forward_to_h8 / forward_from_h8 expose the H/8 representation for
    cross-task coupling (MidScaleCoupling). forward() combines both and
    optionally returns the H/8 features alongside the final output.
    """
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        stage_extra_channels=(0, 0, 0),
        activation="sigmoid",
    ):
        super().__init__()
        e3, e2, e1 = stage_extra_channels

        self.dec3 = ModernDecoderBlock(in_channels, 768)
        self.dec2 = ModernDecoderBlock(768 + skip_channels[2] + e3, 384)
        self.dec1 = ModernDecoderBlock(384 + skip_channels[1] + e2, 192)
        self.dec0 = ModernDecoderBlock(192 + skip_channels[0] + e1, 96)

        head_layers = [
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1),
        ]
        if activation == "softplus":
            head_layers.append(nn.Softplus())
        elif activation == "sigmoid":
            head_layers.append(nn.Sigmoid())
        self.head = nn.Sequential(*head_layers)

        # H/8 channel count (used by callers to build MidScaleCoupling)
        self.h8_channels = 384 + skip_channels[1] + e2

    @staticmethod
    def _resize_like(x, ref):
        if x is None:
            return None
        if x.shape[-2:] == ref.shape[-2:]:
            return x
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def _cat(self, x, skip, extra):
        tensors = [x, skip]
        if extra is not None:
            tensors.append(extra)
        return torch.cat(tensors, dim=1)

    @staticmethod
    def _pad_ops(stage_ops, n=4):
        if stage_ops is None:
            return [None] * n
        ops = list(stage_ops)
        while len(ops) < n:
            ops.append(None)
        return ops

    def forward_to_h8(self, z, skip_features, extra_features=None, stage_ops=None):
        """Bottleneck → H/8.  Applies ops[0] (H/16) and ops[1] (H/8 pre-skip).

        Returns H/8 features with skip already concatenated:
            shape (B, 384 + skip_channels[1], H/8, W/8)
        """
        extras = [None, None, None] if extra_features is None else list(extra_features)
        ops = self._pad_ops(stage_ops)

        # H/32 → H/16
        x = self.dec3(z)
        if ops[0] is not None:
            x = ops[0](x)
        x = self._cat(x, skip_features[2], self._resize_like(extras[0], x))

        # H/16 → H/8
        x = self.dec2(x)
        if ops[1] is not None:
            x = ops[1](x)
        x = self._cat(x, skip_features[1], self._resize_like(extras[1], x))

        return x

    def forward_from_h8(self, h8, skip_features, extra_features=None, stage_ops=None):
        """H/8 → output.  Applies ops[2] (H/4 pre-skip) and ops[3] (H/2 post-dec0)."""
        extras = [None, None, None] if extra_features is None else list(extra_features)
        ops = self._pad_ops(stage_ops)

        # H/8 → H/4
        x = self.dec1(h8)
        if ops[2] is not None:
            x = ops[2](x)
        x = self._cat(x, skip_features[0], self._resize_like(extras[2], x))

        # H/4 → H/2
        x = self.dec0(x)
        if ops[3] is not None:
            x = ops[3](x)

        # H/2 → H
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.head(x)

    def forward(self, z, skip_features, extra_features=None, stage_ops=None,
                return_h8: bool = False):
        """Full forward.  If return_h8=True, returns (output, h8_features)."""
        h8 = self.forward_to_h8(z, skip_features, extra_features, stage_ops)
        out = self.forward_from_h8(h8, skip_features, extra_features, stage_ops)
        if return_h8:
            return out, h8
        return out
