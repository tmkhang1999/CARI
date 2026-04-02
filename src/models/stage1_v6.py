"""Stage 1 V6: Version-1 baseline + segmentation SPADE + CCR (Dec C) + normal pyramids (Dec A & D)."""

import torch

from .stage1_v5 import IntrinsicDecompositionV5, compute_ccr
from .encoders.normal_encoder import NormalEncoder
from .modules.residual_attention import ResidualAttention
from .decoders.progressive_decoder import ProgressiveDecoder


class IntrinsicDecompositionV6(IntrinsicDecompositionV5):
    def __init__(self, config):
        super().__init__(config)
        z_channels = config.get('z_channels', 1024)
        skip_channels = self.image_encoder.feature_channels

        self.normal_encoder = NormalEncoder(in_channels=3, channels=(64, 128, 256, 512))
        
        # Replace Decoder A and D with ProgressiveDecoder for normal pyramid injection
        self.decoder_a = ProgressiveDecoder(
            in_channels=z_channels,
            skip_channels=skip_channels,
            out_channels=1,
            stage_extra_channels=(0, 0, 0),
            activation="sigmoid",
        )
        self.decoder_d = ProgressiveDecoder(
            in_channels=z_channels * 3,  # z_global + s_c_adapted + a_d_adapted
            skip_channels=skip_channels,
            out_channels=3,
            stage_extra_channels=(0, 0, 0),
            activation="sigmoid",
        )

        # Pyramids for Normal attention in Dec A and D
        self.attn_a3 = ResidualAttention(768, 512)
        self.attn_a2 = ResidualAttention(384, 256)
        self.attn_a1 = ResidualAttention(192, 128)

        self.attn_d3 = ResidualAttention(768, 512)
        self.attn_d2 = ResidualAttention(384, 256)
        self.attn_d1 = ResidualAttention(192, 128)

    def forward(self, rgb, m_diffuse=None, normals=None, seg=None,
                valid_mask=None, ccr=None, **kwargs):
        z_global, skip_features = self.image_encoder(rgb)
        self._validate_bottleneck_shape(z_global)

        if normals is None:
            normals = torch.zeros_like(rgb)
        normal_feats = self.normal_encoder(normals)

        # Dec A: normal-guided
        d_g = self.decoder_a(
            z_global, 
            skip_features,
            stage_ops=[
                lambda x: self.attn_a3(x, normal_feats[3]),
                lambda x: self.attn_a2(x, normal_feats[2]),
                lambda x: self.attn_a1(x, normal_feats[1]),
            ]
        )
        s_g = 1.0 / (d_g + 1e-6) - 1.0

        s_g_adapted = self.adapter_s_g(s_g)
        z_b = torch.cat([z_global, s_g_adapted], dim=1)
        xi = self.decoder_b(z_b, skip_features)

        eps = 1e-7
        c_rg = (1.0 - xi[:, 0:1]) / (xi[:, 0:1] + eps)
        c_bg = (1.0 - xi[:, 1:2]) / (xi[:, 1:2] + eps)
        c = torch.cat([c_rg, torch.ones_like(c_rg), c_bg], dim=1)
        s_c = s_g * c

        if ccr is None:
            ccr = compute_ccr(rgb)
        ccr_feats = self.ccr_encoder(ccr)

        s_c_adapted = self.adapter_s_c(s_c)
        z_c = torch.cat([z_global, s_c_adapted], dim=1)
        
        # Dec C: CCR attention then SPADE (inherited naturally from V5 style logic)
        a_d = self.decoder_c(
            z_c, 
            skip_features,
            stage_ops=[
                lambda x: self.spade_c3(self.attn_c3(x, ccr_feats[3]), seg),
                lambda x: self.spade_c2(self.attn_c2(x, ccr_feats[2]), seg),
                lambda x: self.spade_c1(self.attn_c1(x, ccr_feats[1]), seg),
            ]
        )

        a_d_adapted = self.adapter_a_d(a_d.detach())
        z_d = torch.cat([z_global, s_c_adapted, a_d_adapted], dim=1)
        
        # Dec D: normal-guided
        pi = self.decoder_d(
            z_d, 
            skip_features,
            stage_ops=[
                lambda x: self.attn_d3(x, normal_feats[3]),
                lambda x: self.attn_d2(x, normal_feats[2]),
                lambda x: self.attn_d1(x, normal_feats[1]),
            ]
        )

        return {
            'd_g': d_g,
            'xi': xi,
            'c': c,
            's_c': s_c,
            'a_d': a_d,
            'pi': pi,
        }
