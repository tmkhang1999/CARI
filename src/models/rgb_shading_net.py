"""RGBShadingNet (the reported model): frozen DINOv2-L encoder + DPT trunk + albedo and inverse-shading heads.

  I -> DINOv2-L/14 (frozen) -> 4 intermediate maps -> DPT reassemble/fusion + a conv
  detail stem -> shared trunk F. Albedo head -> A in [0,1]^3; shading head -> pi =
  1/(S_d + 1) per channel (three-channel shading); residual R = (I - A*S_d)_+ is analytic.

Colour path (`albedo_rgb_skip`): the gamma-encoded input is concatenated into the albedo
head at full resolution. Without it the frozen-encoder model keeps only about half of the
reference chroma spread; with it (plus the chroma-direction loss on albedo) the colour is
restored. It is also a direct route for illuminant colour into A, which is why the trifactor
model feeds the albedo head the input divided by its predicted shading instead.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders.dino_encoder import DINOv2Encoder
from .decoders.dpt_decoder import DPTTrunk, _gn


class DecodeHead(nn.Module):
    """Shared trunk features -> full-resolution raw output, with an optional image skip.

    Two 3x3 convs on the trunk, bilinear upsample to full resolution, then (if skip_ch > 0)
    concatenation of a small conv encoding of the skip image, one 3x3 refine and a 1x1
    projection. Returns raw logits. Also used by the trifactor heads.
    """
    def __init__(self, in_ch: int, mid: int = 64, out_ch: int = 3, skip_ch: int = 0,
                 skip_feat: int = 32):
        super().__init__()
        self.skip_ch = int(skip_ch)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=1, bias=False), _gn(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1, bias=False), _gn(mid), nn.ReLU(inplace=True),
        )
        refine_in = mid
        if self.skip_ch > 0:
            self.skip_enc = nn.Sequential(
                nn.Conv2d(self.skip_ch, skip_feat, 3, padding=1, bias=False), _gn(skip_feat), nn.ReLU(inplace=True),
                nn.Conv2d(skip_feat, skip_feat, 3, padding=1, bias=False), _gn(skip_feat), nn.ReLU(inplace=True),
            )
            refine_in = mid + skip_feat
        self.refine = nn.Sequential(
            nn.Conv2d(refine_in, mid, 3, padding=1, bias=False), _gn(mid), nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(mid, out_ch, kernel_size=1)

    def forward(self, feat, out_size, skip=None):
        x = self.block(feat)
        x = F.interpolate(x, size=out_size, mode='bilinear', align_corners=False)
        if self.skip_ch > 0 and skip is not None:
            if skip.shape[-2:] != x.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, self.skip_enc(skip)], dim=1)
        x = self.refine(x)
        return self.out(x)


class RGBShadingNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        for retired in ('albedo_chroma_skip', 'shading_lum_skip'):
            if bool(config.get(retired, False)):
                raise ValueError(f'{retired} was an early ablation lever and has been removed; '
                                 'set it to false (only albedo_rgb_skip is supported)')

        self.pi_floor = float(config.get('pi_floor', 5e-3))
        self.use_rgb_skip = bool(config.get('albedo_rgb_skip', False))

        self.encoder = DINOv2Encoder(
            variant=config.get('dino_variant', 'large'),
            pretrained=bool(config.get('dino_pretrained', True)),
        )
        self.trunk = DPTTrunk(
            in_dim=self.encoder.embed_dim,
            feat_ch=int(config.get('dpt_feat_ch', 256)),
            fusion_ch=int(config.get('dpt_fusion_ch', 128)),
            out_ch=int(config.get('dpt_out_ch', 128)),
            detail_ch=int(config.get('detail_ch', 48)),
        )
        trunk_ch = self.trunk.out_channels
        head_mid = int(config.get('head_mid', 64))
        self.albedo_head = DecodeHead(trunk_ch, mid=head_mid, out_ch=3,
                                      skip_ch=3 if self.use_rgb_skip else 0)
        self.shading_head = DecodeHead(trunk_ch, mid=head_mid, out_ch=3, skip_ch=0)

    def forward(self, rgb, **kwargs):
        B, C, H, W = rgb.shape
        dino_feats, dino_tokens, dino_patch_hw = self.encoder(rgb)
        f_trunk = self.trunk(dino_feats, rgb, (H, W))

        alb_skip = (rgb.clamp(0.0, 1.0) + 1e-6).pow(1.0 / 2.2) if self.use_rgb_skip else None
        albedo = torch.sigmoid(
            self.albedo_head(f_trunk, out_size=(H, W), skip=alb_skip)
        ).clamp(1e-4, 1.0)

        shading_pi = torch.sigmoid(
            self.shading_head(f_trunk, out_size=(H, W))
        ).clamp(self.pi_floor, 1.0 - 1e-4)
        shading_linear = (1.0 - shading_pi) / shading_pi

        diffuse = albedo * shading_linear
        residual = (rgb - diffuse).clamp(min=0.0)

        return {
            'a_d': albedo,
            'shading': shading_pi,
            'shading_linear': shading_linear,
            'residual': residual,
            'rgb_reconstructed': diffuse + residual,
            'dino_tokens': dino_tokens,          # for the material-consistency loss
            'dino_patch_hw': dino_patch_hw,
        }
