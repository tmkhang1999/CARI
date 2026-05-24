import torch
import torch.nn as nn
import torch.nn.functional as F

class ModernDecoderBlock(nn.Module):
    """
    Proper feature mixing using a double-convolution residual-style block 
    BEFORE upsampling. Processing channels at the lower spatial resolution saves
    significant VRAM and compute, while the double convolution prevents the 
    catastrophic information collapse of the old `DecoderBlock`.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mix = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        return self.up(self.mix(x))

class ModernProgressiveDecoder(nn.Module):
    """
    Four-stage proper U-Net decoder with optional per-stage extra features and ops.
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
        extras = [None, None, None] if extra_features is None else list(extra_features)
        ops = [None, None, None, None] if stage_ops is None else list(stage_ops)
        
        while len(ops) < 4:
            ops.append(None)

        # H/32 -> H/16
        x = self.dec3(z)
        if ops[0] is not None:
            x = ops[0](x)
        x = self._cat(x, skip_features[2], self._resize_like(extras[0], x))

        # H/16 -> H/8
        x = self.dec2(x)
        if ops[1] is not None:
            x = ops[1](x)
        x = self._cat(x, skip_features[1], self._resize_like(extras[1], x))

        # H/8 -> H/4
        x = self.dec1(x)
        if ops[2] is not None:
            x = ops[2](x)
        x = self._cat(x, skip_features[0], self._resize_like(extras[2], x))

        # H/4 -> H/2
        x = self.dec0(x)
        if ops[3] is not None:
            x = ops[3](x)
            
        # H/2 -> H
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.head(x)