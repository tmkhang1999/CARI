"""Stage 1 V4: add CCR guidance and SPADE in Dec C on top of V3."""

import torch
import torch.nn.functional as F

from .stage1_v3 import IntrinsicDecompositionV3
from .encoders.ccr_encoder import CCREncoder
from .modules.residual_attention import ResidualAttention
from .modules.spade import SPADE


def compute_ccr(img, eps=1e-7):
    """
    6-channel illumination-invariant descriptor (matches preprocessor/compute_ccr.py).

    Channels 0-2: clamped log cross-colour ratios (PIE-Net) — reflectance boundary edges.
      Clamped to [-1, 1] to prevent exploding values at near-black pixels.
    Channels 3-5: normalized RGB chromaticity (SIGNet) — illumination-invariant hue.

    Args:
        img: (N, 3, H, W) linear RGB, float32, values > 0
    Returns:
        (N, 6, H, W)
    """
    log_img = torch.log(img + eps)
    r, g, b = log_img[:, 0:1], log_img[:, 1:2], log_img[:, 2:3]

    kernel = torch.tensor(
        [[0, 1, 0], [1, 0, -1], [0, -1, 0]],
        dtype=img.dtype, device=img.device,
    ).view(1, 1, 3, 3)

    def diff(ch):
        return F.conv2d(ch, kernel, padding=1)

    # Clamped log-CCR (PIE-Net)
    m_rg = torch.clamp(diff(r) - diff(g), -1.0, 1.0)
    m_rb = torch.clamp(diff(r) - diff(b), -1.0, 1.0)
    m_gb = torch.clamp(diff(g) - diff(b), -1.0, 1.0)

    # Normalized RGB chromaticity (SIGNet)
    intensity = img[:, 0:1] + img[:, 1:2] + img[:, 2:3] + eps
    norm_rgb = img / intensity  # (N, 3, H, W), in [0, 1]

    return torch.cat([m_rg, m_rb, m_gb, norm_rgb], dim=1)  # (N, 6, H, W)


class IntrinsicDecompositionV4(IntrinsicDecompositionV3):
    def __init__(self, config):
        super().__init__(config)

        num_classes = config.get('num_seg_classes', 41)  # NYU-40 + background

        # 6-channel input: 3 clamped log-CCR (PIE-Net) + 3 norm-RGB chromaticity (SIGNet)
        self.ccr_encoder = CCREncoder(in_channels=6, channels=(64, 128, 256, 512))
        self.attn_c3 = ResidualAttention(768, 512)
        self.attn_c2 = ResidualAttention(384, 256)
        self.attn_c1 = ResidualAttention(192, 128)

        self.spade_c3 = SPADE(768, num_classes=num_classes)
        self.spade_c2 = SPADE(384, num_classes=num_classes)
        self.spade_c1 = SPADE(192, num_classes=num_classes)

    def _c_stage_op(self, x, ccr_feat, seg, attn, spade):
        x = attn(x, ccr_feat)
        return spade(x, seg)

    def forward(self, rgb, m_diffuse=None, normals=None, seg=None, ccr=None, **kwargs):
        z_global, skip_features = self.image_encoder(rgb)

        if normals is None:
            normals = torch.zeros_like(rgb)
        normal_feats = self.normal_encoder(normals)

        s_g = self.decoder_a(
            z_global,
            skip_features,
            stage_ops=[
                lambda x: self.attn_a3(x, normal_feats[3]),
                lambda x: self.attn_a2(x, normal_feats[2]),
                lambda x: self.attn_a1(x, normal_feats[1]),
            ],
        )

        s_g_pyr = self.shading_adapter(s_g)
        xi = self.decoder_b(
            z_global,
            skip_features,
            extra_features=[s_g_pyr[3], s_g_pyr[2], s_g_pyr[1]],
        )

        c = self._to_chroma(xi)
        s_c = s_g * c

        if ccr is None:
            ccr = compute_ccr(rgb)
        ccr_feats = self.ccr_encoder(ccr)

        s_c_pyr = self.colorful_adapter(s_c)
        a_d = self.decoder_c(
            z_global,
            skip_features,
            extra_features=[s_c_pyr[3], s_c_pyr[2], s_c_pyr[1]],
            stage_ops=[
                lambda x: self._c_stage_op(x, ccr_feats[3], seg, self.attn_c3, self.spade_c3),
                lambda x: self._c_stage_op(x, ccr_feats[2], seg, self.attn_c2, self.spade_c2),
                lambda x: self._c_stage_op(x, ccr_feats[1], seg, self.attn_c1, self.spade_c1),
            ],
        )

        a_d_pyr = self.albedo_adapter(a_d)
        s_d = self.decoder_d(
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
            "s_g": s_g,
            "xi": xi,
            "c": c,
            "s_c": s_c,
            "a_d": a_d,
            "s_d": s_d,
        }
