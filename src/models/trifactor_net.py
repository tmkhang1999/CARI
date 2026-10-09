"""TriFactorNet: the next model, I = A * (S_lum * C) + R.

The three trainable decoders have explicit ownership:
  luminance head -> scalar shading intensity S_lum,
  chroma head    -> unit-luminance shading color C,
  albedo head    -> direct diffuse albedo A_d.

The physical image formation is I = A_d * (S_lum * C) + R. Residual R is
signed and analytic; it is not a fourth decoder. This module is isolated from
RGBShadingNet so the ablation checkpoints and behavior remain unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .decoders.dpt_decoder import DPTTrunk
from .encoders.dino_encoder import DINOv2Encoder
from .rgb_shading_net import DecodeHead


def luminance(rgb: torch.Tensor) -> torch.Tensor:
    weights = rgb.new_tensor((0.2126, 0.7152, 0.0722)).view(1, 3, 1, 1)
    return (rgb * weights).sum(dim=1, keepdim=True)


def log_chroma(rgb: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    rgb = rgb.clamp_min(eps)
    return torch.cat(
        (
            torch.log(rgb[:, 0:1] / rgb[:, 1:2]),
            torch.log(rgb[:, 2:3] / rgb[:, 1:2]),
        ),
        dim=1,
    ).clamp(-4.0, 4.0)


def chroma_from_uv(uv: torch.Tensor) -> torch.Tensor:
    """Convert (log R/G, log B/G) to positive shading color with Y(C)=1."""
    red = torch.exp(uv[:, 0:1])
    green = torch.ones_like(red)
    blue = torch.exp(uv[:, 1:2])
    chroma = torch.cat((red, green, blue), dim=1)
    return chroma / luminance(chroma).clamp_min(1e-4)


class TriFactorNet(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.log_shading_limit = float(config.get("log_shading_limit", 6.0))
        self.log_chroma_limit = float(config.get("log_chroma_limit", 2.5))
        self.canonical_max = float(config.get("canonical_max", 4.0))
        self.observable_low = float(config.get("observable_low", 0.01))
        self.observable_high = float(config.get("observable_high", 0.985))

        self.encoder = DINOv2Encoder(
            variant=config.get("dino_variant", "large"),
            pretrained=bool(config.get("dino_pretrained", True)),
        )
        self.trunk = DPTTrunk(
            in_dim=self.encoder.embed_dim,
            feat_ch=int(config.get("dpt_feat_ch", 512)),
            fusion_ch=int(config.get("dpt_fusion_ch", 256)),
            out_ch=int(config.get("dpt_out_ch", 256)),
            detail_ch=int(config.get("detail_ch", 64)),
        )
        trunk_channels = self.trunk.out_channels
        head_mid = int(config.get("head_mid", 128))

        self.luminance_head = DecodeHead(
            trunk_channels, mid=head_mid, out_ch=1, skip_ch=1
        )
        self.chroma_head = DecodeHead(
            trunk_channels, mid=head_mid, out_ch=2, skip_ch=2
        )
        self.chroma_global = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(trunk_channels, max(32, head_mid // 2)),
            nn.ReLU(inplace=True),
            nn.Linear(max(32, head_mid // 2), 2),
        )
        self.albedo_head = DecodeHead(
            trunk_channels, mid=head_mid, out_ch=3, skip_ch=3
        )

    def forward(self, rgb: torch.Tensor, **_: object) -> dict[str, torch.Tensor | tuple[int, int]]:
        _, _, height, width = rgb.shape
        rgb = rgb.clamp(0.0, 1.0)
        dino_features, dino_tokens, patch_hw = self.encoder(rgb)
        features = self.trunk(dino_features, rgb, (height, width))

        image_lum = luminance(rgb).clamp_min(1e-4)
        log_lum = torch.log(image_lum)
        log_lum = log_lum - log_lum.mean(dim=(2, 3), keepdim=True)
        log_s = self.log_shading_limit * torch.tanh(
            self.luminance_head(features, (height, width), log_lum)
            / self.log_shading_limit
        )
        shading_luminance = torch.exp(log_s)

        image_uv = log_chroma(rgb)
        local_uv = self.chroma_head(features, (height, width), image_uv)
        global_uv = self.chroma_global(features).view(-1, 2, 1, 1)
        shading_uv = self.log_chroma_limit * torch.tanh(
            (local_uv + global_uv) / self.log_chroma_limit
        )
        shading_chroma = chroma_from_uv(shading_uv)
        shading_rgb = shading_luminance * shading_chroma

        observable = (
            (rgb.amin(dim=1, keepdim=True) > self.observable_low)
            & (rgb.amax(dim=1, keepdim=True) < self.observable_high)
            & (shading_luminance > self.observable_low)
        ).to(rgb.dtype)
        canonical = (rgb / shading_rgb.clamp_min(1e-4)).clamp(0.0, self.canonical_max)
        canonical = canonical / self.canonical_max
        canonical_skip = canonical * observable
        albedo = torch.sigmoid(
            self.albedo_head(features, (height, width), canonical_skip)
        ).clamp(1e-4, 1.0)

        diffuse = albedo * shading_rgb
        residual = rgb - diffuse
        return {
            "a_d": albedo,
            "shading_luminance": shading_luminance,
            "shading_chroma": shading_chroma,
            "shading_uv": shading_uv,
            "shading_linear": shading_rgb,
            "diffuse": diffuse,
            "residual": residual,
            "rgb_reconstructed": diffuse + residual,
            "observable": observable,
            "canonical": canonical,
            "canonical_linear": (canonical * self.canonical_max).clamp(0.0, 1.0),
            "dino_tokens": dino_tokens,
            "dino_patch_hw": patch_hw,
        }
