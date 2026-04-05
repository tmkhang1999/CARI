"""Stage 1 V7: FiLM-DecB + guidance pyramids + CCR/SPADE ordered DecC + normal-guided DecD."""

import torch
import torch.nn as nn

from .encoders.image_encoder import ImageEncoder
from .encoders.normal_encoder import NormalEncoder
from .encoders.guidance_encoder import GuidanceEncoder
from .encoders.ccr_encoder import CCREncoder
from .decoders.progressive_decoder import ProgressiveDecoder
from .modules.residual_attention import ResidualAttention
from .modules.spade import SPADE
from .modules.illuminant_descriptor import IlluminantDescriptor
from .stage1_v5 import compute_ccr


class IntrinsicDecompositionV7(nn.Module):
    def __init__(self, config):
        super().__init__()

        z_channels = config.get('z_channels', 1024)
        freeze_stages = config.get('freeze_stages', [1, 2])
        model_name = config.get('backbone', config.get('model_name', 'convnextv2_base'))
        pretrained = config.get('pretrained', True)
        num_seg_classes = int(config.get('num_seg_classes', 41))

        self.image_encoder = ImageEncoder(
            model_name=model_name,
            freeze_stages=freeze_stages,
            pretrained=pretrained,
        )
        skip_channels = self.image_encoder.feature_channels

        self.normal_encoder = NormalEncoder(in_channels=3, channels=(64, 128, 256, 512))
        self.ccr_encoder = CCREncoder(in_channels=6, channels=(64, 128, 256, 512))
        self.guidance_b = GuidanceEncoder(in_channels=1, channels=(64, 128, 256, 512))
        self.guidance_c = GuidanceEncoder(in_channels=6, channels=(64, 128, 256, 512))
        self.guidance_d = GuidanceEncoder(in_channels=6, channels=(64, 128, 256, 512))
        self.illuminant_descriptor = IlluminantDescriptor(in_dim=4, hidden_dim=128, out_dim=2 * z_channels)

        self.decoder_a = ProgressiveDecoder(
            in_channels=z_channels,
            skip_channels=skip_channels,
            out_channels=1,
            stage_extra_channels=(0, 0, 0),
            activation='sigmoid',
        )
        self.decoder_b = ProgressiveDecoder(
            in_channels=z_channels,
            skip_channels=skip_channels,
            out_channels=2,
            stage_extra_channels=(0, 0, 0),
            activation='sigmoid',
        )
        self.decoder_c = ProgressiveDecoder(
            in_channels=z_channels,
            skip_channels=skip_channels,
            out_channels=3,
            stage_extra_channels=(0, 0, 0),
            activation='sigmoid',
        )
        self.decoder_d = ProgressiveDecoder(
            in_channels=z_channels,
            skip_channels=skip_channels,
            out_channels=3,
            stage_extra_channels=(0, 0, 0),
            activation='sigmoid',
        )

        self.attn_a3 = ResidualAttention(768, 512)
        self.attn_a2 = ResidualAttention(384, 256)
        self.attn_a1 = ResidualAttention(192, 128)

        self.attn_b3 = ResidualAttention(768, 512)
        self.attn_b2 = ResidualAttention(384, 256)
        self.attn_b1 = ResidualAttention(192, 128)

        self.attn_c_ccr3 = ResidualAttention(768, 512)
        self.attn_c_ccr2 = ResidualAttention(384, 256)
        self.attn_c_ccr1 = ResidualAttention(192, 128)
        self.spade_c3 = SPADE(num_channels=768, num_classes=num_seg_classes)
        self.spade_c2 = SPADE(num_channels=384, num_classes=num_seg_classes)
        self.spade_c1 = SPADE(num_channels=192, num_classes=num_seg_classes)
        self.attn_c_g3 = ResidualAttention(768, 512)
        self.attn_c_g2 = ResidualAttention(384, 256)
        self.attn_c_g1 = ResidualAttention(192, 128)

        self.attn_d_n3 = ResidualAttention(768, 512)
        self.attn_d_n2 = ResidualAttention(384, 256)
        self.attn_d_n1 = ResidualAttention(192, 128)
        self.attn_d_g3 = ResidualAttention(768, 512)
        self.attn_d_g2 = ResidualAttention(384, 256)
        self.attn_d_g1 = ResidualAttention(192, 128)

    @staticmethod
    def _to_chroma(xi):
        eps = 1e-7
        c_rg = (1.0 - xi[:, 0:1]) / (xi[:, 0:1] + eps)
        c_bg = (1.0 - xi[:, 1:2]) / (xi[:, 1:2] + eps)
        return torch.cat([c_rg, torch.ones_like(c_rg), c_bg], dim=1)

    @staticmethod
    def _derive_albedo(rgb, shading):
        albedo = (rgb / (shading.clamp(1e-3) + 1e-6)).clamp(0.0, 1.0)
        return torch.nan_to_num(albedo, nan=0.0, posinf=1.0, neginf=0.0)

    def forward(self, rgb, m_diffuse=None, normals=None, seg=None, valid_mask=None, ccr=None, **kwargs):
        z_global, skip_features = self.image_encoder(rgb)

        if normals is None:
            normals = torch.zeros_like(rgb)
        normal_feats = self.normal_encoder(normals)

        if ccr is None:
            ccr = compute_ccr(rgb)
        ccr_feats = self.ccr_encoder(ccr)

        gamma, beta = self.illuminant_descriptor(rgb, valid_mask)
        z_b = z_global * (1.0 + gamma) + beta

        d_g = self.decoder_a(
            z_global,
            skip_features,
            stage_ops=[
                lambda x: self.attn_a3(x, normal_feats[3]),
                lambda x: self.attn_a2(x, normal_feats[2]),
                lambda x: self.attn_a1(x, normal_feats[1]),
            ],
        )
        s_g = 1.0 / (d_g + 1e-6) - 1.0
        s_g_safe = s_g.clamp(0.0, 20.0)

        g_b = self.guidance_b(s_g_safe)
        xi = self.decoder_b(
            z_b,
            skip_features,
            stage_ops=[
                lambda x: self.attn_b3(x, g_b[3]),
                lambda x: self.attn_b2(x, g_b[2]),
                lambda x: self.attn_b1(x, g_b[1]),
            ],
        )

        c = self._to_chroma(xi)
        s_c = s_g * c
        s_c_safe = s_c.detach().clamp(0.0, 20.0)
        a_c = self._derive_albedo(rgb, s_c.detach())

        g_c = self.guidance_c(torch.cat([s_c_safe, a_c], dim=1))
        a_d = self.decoder_c(
            z_global,
            skip_features,
            stage_ops=[
                lambda x: self.attn_c_g3(self.spade_c3(self.attn_c_ccr3(x, ccr_feats[3]), seg), g_c[3]),
                lambda x: self.attn_c_g2(self.spade_c2(self.attn_c_ccr2(x, ccr_feats[2]), seg), g_c[2]),
                lambda x: self.attn_c_g1(self.spade_c1(self.attn_c_ccr1(x, ccr_feats[1]), seg), g_c[1]),
            ],
        )

        g_d = self.guidance_d(torch.cat([s_c_safe, a_d.detach()], dim=1))
        pi = self.decoder_d(
            z_global,
            skip_features,
            stage_ops=[
                lambda x: self.attn_d_g3(self.attn_d_n3(x, normal_feats[3]), g_d[3]),
                lambda x: self.attn_d_g2(self.attn_d_n2(x, normal_feats[2]), g_d[2]),
                lambda x: self.attn_d_g1(self.attn_d_n1(x, normal_feats[1]), g_d[1]),
            ],
        )

        return {
            'd_g': d_g,
            'xi': xi,
            'a_d': a_d,
            'pi': pi,
        }
