"""Stage 1 V9: Unified SFM-guided decoders with CCR, normals, SPADE, and FiLM."""

import torch
import torch.nn as nn

from .encoders.image_encoder import ImageEncoder
from .encoders.normal_encoder import NormalEncoder
from .encoders.guidance_encoder import GuidanceEncoder
from .encoders.ccr_encoder import CCREncoder
from .decoders.progressive_decoder import ProgressiveDecoder
from .modules.spatial_feature_modulation import SpatialFeatureModulation
from .modules.spade import SPADE
from .modules.illuminant_descriptor import IlluminantDescriptor
from .stage1_v5 import compute_ccr


class IntrinsicDecompositionV9(nn.Module):
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
        self.guidance_b = GuidanceEncoder(in_channels=4, channels=(64, 128, 256, 512))
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

        # Dec A: Normals + CCR via SFM
        self.sfm_a_n3 = SpatialFeatureModulation(768, 512)
        self.sfm_a_ccr3 = SpatialFeatureModulation(768, 512)
        self.sfm_a_n2 = SpatialFeatureModulation(384, 256)
        self.sfm_a_ccr2 = SpatialFeatureModulation(384, 256)
        self.sfm_a_n1 = SpatialFeatureModulation(192, 128)
        self.sfm_a_ccr1 = SpatialFeatureModulation(192, 128)

        # Dec B: FiLM + Guidance_B via SFM
        self.sfm_b3 = SpatialFeatureModulation(768, 512)
        self.sfm_b2 = SpatialFeatureModulation(384, 256)
        self.sfm_b1 = SpatialFeatureModulation(192, 128)

        # Stage-wise FiLM adapters for decoder-B inner features.
        self.film_b3 = nn.Conv2d(2 * z_channels, 2 * 768, kernel_size=1, bias=True)
        self.film_b2 = nn.Conv2d(2 * z_channels, 2 * 384, kernel_size=1, bias=True)
        self.film_b1 = nn.Conv2d(2 * z_channels, 2 * 192, kernel_size=1, bias=True)
        for head in (self.film_b3, self.film_b2, self.film_b1):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

        # Dec C: CCR + SPADE(s2/s1) + Guidance_C via SFM
        self.sfm_c_ccr3 = SpatialFeatureModulation(768, 512)
        self.sfm_c_g3 = SpatialFeatureModulation(768, 512)

        self.sfm_c_ccr2 = SpatialFeatureModulation(384, 256)
        self.spade_c2 = SPADE(num_channels=384, num_classes=num_seg_classes)
        self.sfm_c_g2 = SpatialFeatureModulation(384, 256)

        self.sfm_c_ccr1 = SpatialFeatureModulation(192, 128)
        self.spade_c1 = SPADE(num_channels=192, num_classes=num_seg_classes)
        self.sfm_c_g1 = SpatialFeatureModulation(192, 128)

        # Dec D: Normals + Guidance_D via SFM
        self.sfm_d_n3 = SpatialFeatureModulation(768, 512)
        self.sfm_d_g3 = SpatialFeatureModulation(768, 512)
        self.sfm_d_n2 = SpatialFeatureModulation(384, 256)
        self.sfm_d_g2 = SpatialFeatureModulation(384, 256)
        self.sfm_d_n1 = SpatialFeatureModulation(192, 128)
        self.sfm_d_g1 = SpatialFeatureModulation(192, 128)

    @staticmethod
    def _to_chroma(xi):
        eps = 1e-7
        c_rg = (1.0 - xi[:, 0:1]) / (xi[:, 0:1] + eps)
        c_bg = (1.0 - xi[:, 1:2]) / (xi[:, 1:2] + eps)
        c_rg = torch.clamp(c_rg, 0.0, 20.0)
        c_bg = torch.clamp(c_bg, 0.0, 20.0)
        return torch.cat([c_rg, torch.ones_like(c_rg), c_bg], dim=1)

    @staticmethod
    def _derive_albedo(rgb, shading):
        albedo = (rgb / (shading.clamp(1e-3, 20.0) + 1e-6)).clamp(0.0, 1.0)
        return torch.nan_to_num(albedo, nan=0.0, posinf=1.0, neginf=0.0)

    @staticmethod
    def _apply_stage_film(x, film_head, film_global):
        g_beta = film_head(film_global)
        gamma, beta = g_beta.chunk(2, dim=1)
        return x * (1.0 + gamma) + beta

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
        film_global = torch.cat([gamma, beta], dim=1)

        d_g = self.decoder_a(
            z_global,
            skip_features,
            stage_ops=[
                lambda x: self.sfm_a_ccr3(self.sfm_a_n3(x, normal_feats[3]), ccr_feats[3]),
                lambda x: self.sfm_a_ccr2(self.sfm_a_n2(x, normal_feats[2]), ccr_feats[2]),
                lambda x: self.sfm_a_ccr1(self.sfm_a_n1(x, normal_feats[1]), ccr_feats[1]),
            ],
        )
        s_g = 1.0 / (d_g + 1e-6) - 1.0
        s_g_safe = s_g.clamp(0.0, 20.0)
        a_g = self._derive_albedo(rgb, s_g_safe.detach())

        g_b = self.guidance_b(torch.cat([s_g_safe, a_g], dim=1))
        xi = self.decoder_b(
            z_b,
            skip_features,
            stage_ops=[
                lambda x: self.sfm_b3(self._apply_stage_film(x, self.film_b3, film_global), g_b[3]),
                lambda x: self.sfm_b2(self._apply_stage_film(x, self.film_b2, film_global), g_b[2]),
                lambda x: self.sfm_b1(self._apply_stage_film(x, self.film_b1, film_global), g_b[1]),
            ],
        )

        c = self._to_chroma(xi)
        s_c = s_g * c
        s_c_safe = s_c.detach().clamp(0.0, 20.0)
        a_c = self._derive_albedo(rgb, s_c_safe)

        g_c = self.guidance_c(torch.cat([s_c_safe, a_c], dim=1))
        a_d = self.decoder_c(
            z_global,
            skip_features,
            stage_ops=[
                lambda x: self.sfm_c_g3(self.sfm_c_ccr3(x, ccr_feats[3]), g_c[3]),
                lambda x: self.sfm_c_g2(self.spade_c2(self.sfm_c_ccr2(x, ccr_feats[2]), seg), g_c[2]),
                lambda x: self.sfm_c_g1(self.spade_c1(self.sfm_c_ccr1(x, ccr_feats[1]), seg), g_c[1]),
            ],
        )

        g_d = self.guidance_d(torch.cat([s_c_safe, a_d.detach()], dim=1))
        pi = self.decoder_d(
            z_global,
            skip_features,
            stage_ops=[
                lambda x: self.sfm_d_g3(self.sfm_d_n3(x, normal_feats[3]), g_d[3]),
                lambda x: self.sfm_d_g2(self.sfm_d_n2(x, normal_feats[2]), g_d[2]),
                lambda x: self.sfm_d_g1(self.sfm_d_n1(x, normal_feats[1]), g_d[1]),
            ],
        )

        return {
            'd_g': d_g,
            'xi': xi,
            'a_d': a_d,
            'pi': pi,
        }
