"""Shared progressive decoder used by Stage 1 versions 2-4."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import DecoderBlock


class ProgressiveDecoder(nn.Module):
    """
    Four-stage decoder with optional per-stage extra features and stage ops.

    Stages operate at spatial scales:
      s3: H/16, s2: H/8, s1: H/4, then final output at H.
    """
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        stage_extra_channels=(0, 0, 0),
        activation="softplus",
    ):
        super().__init__()
        e3, e2, e1 = stage_extra_channels

        self.dec3 = DecoderBlock(in_channels, 768)
        self.dec2 = DecoderBlock(768 + skip_channels[2] + e3, 384)
        self.dec1 = DecoderBlock(384 + skip_channels[1] + e2, 192)
        self.dec0 = DecoderBlock(192 + skip_channels[0] + e1, 96)

        head_layers = [
            nn.Conv2d(96, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1),
        ]
        if activation == "softplus":
            head_layers.append(nn.Softplus())
        elif activation == "sigmoid":
            head_layers.append(nn.Sigmoid())
        self.head = nn.Sequential(*head_layers)

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

    def forward(self, z, skip_features, extra_features=None, stage_ops=None):
        """
        Args:
            z: (N, C, H/32, W/32) bottleneck input.
            skip_features: image skips [H/4, H/8, H/16, H/32].
            extra_features: list [e3, e2, e1] or None.
            stage_ops: list of callables [op3, op2, op1] or None.
        """
        extras = [None, None, None] if extra_features is None else list(extra_features)
        ops = [None, None, None] if stage_ops is None else list(stage_ops)

        x = self.dec3(z)
        if ops[0] is not None:
            x = ops[0](x)
        x = self._cat(x, skip_features[2], self._resize_like(extras[0], x))

        x = self.dec2(x)
        if ops[1] is not None:
            x = ops[1](x)
        x = self._cat(x, skip_features[1], self._resize_like(extras[1], x))

        x = self.dec1(x)
        if ops[2] is not None:
            x = ops[2](x)
        x = self._cat(x, skip_features[0], self._resize_like(extras[2], x))

        x = self.dec0(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.head(x)

