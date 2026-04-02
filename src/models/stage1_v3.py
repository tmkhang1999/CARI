"""Normal-guided intrinsic decomposition (Stage 1 V3)."""

import torch

from .stage1_v2_5 import IntrinsicDecompositionV2_5
from .encoders.normal_encoder import NormalEncoder
from .modules.residual_attention import ResidualAttention


class IntrinsicDecompositionV3(IntrinsicDecompositionV2_5):
    def __init__(self, config):
        super().__init__(config)

        self.normal_encoder = NormalEncoder(in_channels=3, channels=(64, 128, 256, 512))

        self.attn_a3 = ResidualAttention(768, 512)
        self.attn_a2 = ResidualAttention(384, 256)
        self.attn_a1 = ResidualAttention(192, 128)

        self.attn_d3 = ResidualAttention(768, 512)
        self.attn_d2 = ResidualAttention(384, 256)
        self.attn_d1 = ResidualAttention(192, 128)

    def forward(self, rgb, m_diffuse=None, normals=None, valid_mask=None, **kwargs):
        z_global, skip_features = self.image_encoder(rgb)

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
            ],
        )
        s_g = 1.0 / (d_g + 1e-6) - 1.0

        # Dec B: FiLM from IlluminantDescriptor (inherited from V2.5)
        s_g_pyr = self.shading_adapter(s_g)
        if valid_mask is not None:
            gamma, beta = self.illuminant_desc(rgb, valid_mask)
            z_b = z_global * (1.0 + gamma) + beta
        else:
            z_b = z_global
        xi = self.decoder_b(
            z_b,
            skip_features,
            extra_features=[s_g_pyr[3], s_g_pyr[2], s_g_pyr[1]],
        )

        c = self._to_chroma(xi)
        s_c = s_g * c

        # Dec C: unchanged
        s_c_pyr = self.colorful_adapter(s_c)
        a_d = self.decoder_c(
            z_global,
            skip_features,
            extra_features=[s_c_pyr[3], s_c_pyr[2], s_c_pyr[1]],
        )

        # Dec D: normal-guided
        a_d_pyr = self.albedo_adapter(a_d.detach())
        pi = self.decoder_d(
            z_global,
            skip_features,
            extra_features=[
                torch.cat([s_c_pyr[3], a_d_pyr[3]], dim=1),
                torch.cat([s_c_pyr[2], a_d_pyr[2]], dim=1),
                torch.cat([s_c_pyr[1], a_d_pyr[1]], dim=1),
            ],
            stage_ops=[
                lambda x: self.attn_d3(x, normal_feats[3]),
                lambda x: self.attn_d2(x, normal_feats[2]),
                lambda x: self.attn_d1(x, normal_feats[1]),
            ],
        )

        return {
            "d_g": d_g, "xi": xi, "c": c,
            "s_c": s_c, "a_d": a_d, "pi": pi,
        }
        