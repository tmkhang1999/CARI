"""Stage 1 V5: Version-1 baseline + segmentation SPADE + CCR guidance pyramids in Dec C."""

import torch
import torch.nn.functional as F

from .stage1_v4 import IntrinsicDecompositionV4
from .encoders.ccr_encoder import CCREncoder
from .modules.residual_attention import ResidualAttention


def compute_ccr(img, eps=1e-7):
    """Build 6-channel CCR descriptor: [log edges(3), normalized rgb(3)]."""
    log_img = torch.log(img + eps)
    r, g, b = log_img[:, 0:1], log_img[:, 1:2], log_img[:, 2:3]

    kernel = torch.tensor(
        [[0, 1, 0], [1, 0, -1], [0, -1, 0]],
        dtype=img.dtype,
        device=img.device,
    ).view(1, 1, 3, 3)

    def diff(ch):
        return F.conv2d(ch, kernel, padding=1)

    m_rg = torch.clamp(diff(r) - diff(g), -1.0, 1.0)
    m_rb = torch.clamp(diff(r) - diff(b), -1.0, 1.0)
    m_gb = torch.clamp(diff(g) - diff(b), -1.0, 1.0)

    intensity = img[:, 0:1] + img[:, 1:2] + img[:, 2:3] + eps
    norm_rgb = img / intensity

    return torch.cat([m_rg, m_rb, m_gb, norm_rgb], dim=1)


class IntrinsicDecompositionV5(IntrinsicDecompositionV4):
    def __init__(self, config):
        super().__init__(config)

        self.ccr_encoder = CCREncoder(in_channels=6, channels=(64, 128, 256, 512))
        
        # Replace bottleneck CCR attention with pyramid
        self.attn_c3 = ResidualAttention(768, 512)
        self.attn_c2 = ResidualAttention(384, 256)
        self.attn_c1 = ResidualAttention(192, 128)

    def _decode_with_guidance(self, z_global, skip_features, seg=None, ccr=None, rgb=None):
        d_g = self.decoder_a(z_global, skip_features)
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
        
        # Apply CCR attention then SPADE at each pyramid level
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
        pi = self.decoder_d(z_d, skip_features)

        return {
            'd_g': d_g,
            'xi': xi,
            'c': c,
            's_c': s_c,
            'a_d': a_d,
            'pi': pi,
        }

    def forward(self, rgb, m_diffuse=None, normals=None, seg=None,
                valid_mask=None, ccr=None, **kwargs):
        z_global, skip_features = self.image_encoder(rgb)
        self._validate_bottleneck_shape(z_global)
        return self._decode_with_guidance(z_global, skip_features, seg=seg, ccr=ccr, rgb=rgb)
