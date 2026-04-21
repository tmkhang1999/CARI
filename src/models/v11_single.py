"""Stage 1 V10: Unified SFM-guided decoders with CCR, normals, SPADE, and FiLM."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders.image_encoder import ImageEncoder
from .encoders.normal_encoder import NormalEncoder
from .encoders.guidance_encoder import GuidanceEncoder
from .encoders.ccr_encoder import CCREncoder
from .decoders.progressive_decoder import ProgressiveDecoder
from .modules.spatial_feature_modulation import SpatialFeatureModulation
from .modules.spade import SPADE
from .modules.illuminant_descriptor import IlluminantDescriptor
from .ccr_utils import compute_ccr


class IntrinsicDecompositionV11Single(nn.Module):
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
        self.sfm_a_ccr0 = SpatialFeatureModulation(96, 64)

        # Dec B: FiLM + Guidance_B via SFM
        self.sfm_b3 = SpatialFeatureModulation(768, 512)
        self.sfm_b2 = SpatialFeatureModulation(384, 256)
        self.sfm_b1 = SpatialFeatureModulation(192, 128)

        # Dec C: CCR + SPADE(s2/s1) + Guidance_C via SFM
        self.sfm_c_ccr3 = SpatialFeatureModulation(768, 512)
        self.spade_c3 = SPADE(num_channels=768, num_classes=num_seg_classes)
        self.sfm_c_g3 = SpatialFeatureModulation(768, 512)

        self.sfm_c_ccr2 = SpatialFeatureModulation(384, 256)
        self.spade_c2 = SPADE(num_channels=384, num_classes=num_seg_classes)
        self.sfm_c_g2 = SpatialFeatureModulation(384, 256)

        self.sfm_c_ccr1 = SpatialFeatureModulation(192, 128)
        self.spade_c1 = SPADE(num_channels=192, num_classes=num_seg_classes)
        self.sfm_c_g1 = SpatialFeatureModulation(192, 128)
        self.sfm_c_ccr0 = SpatialFeatureModulation(96, 64)

        # Direct raw-CCR projections preserve sharper boundary evidence at each stage.
        self.ccr_prior_proj3 = nn.Conv2d(6, 512, kernel_size=1, bias=False)
        self.ccr_prior_proj2 = nn.Conv2d(6, 256, kernel_size=1, bias=False)
        self.ccr_prior_proj1 = nn.Conv2d(6, 128, kernel_size=1, bias=False)
        for proj in (self.ccr_prior_proj3, self.ccr_prior_proj2, self.ccr_prior_proj1):
            nn.init.zeros_(proj.weight)

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
        # Raise clamp to 100.0 to allow saturated lighting
        c_rg = torch.clamp(c_rg, 0.0, 100.0)
        c_bg = torch.clamp(c_bg, 0.0, 100.0)
        return torch.cat([c_rg, torch.ones_like(c_rg), c_bg], dim=1)

    @staticmethod
    def _derive_albedo(rgb, shading):
        albedo = (rgb / (shading.clamp(1e-3, 20.0) + 1e-6)).clamp(0.0, 1.0)
        return torch.nan_to_num(albedo, nan=0.0, posinf=1.0, neginf=0.0)

    def forward(self, rgb, m_diffuse=None, normals=None, seg=None, valid_mask=None, ccr=None, **kwargs):
        z_global, skip_features = self.image_encoder(rgb)

        if normals is None:
            normals = torch.zeros_like(rgb)
        normal_feats = self.normal_encoder(normals)

        if ccr is None:
            ccr = compute_ccr(rgb)
        
        # Normalize per-image to close HDR/sRGB distribution gap
        # ccr_std = ccr.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        # ccr = ccr / ccr_std
        
        ccr_feats = self.ccr_encoder(ccr)

        # Inject raw CCR at decoder stage resolutions to avoid over-smoothed priors.
        ccr_raw3 = self.ccr_prior_proj3(F.interpolate(ccr, size=ccr_feats[3].shape[-2:], mode='bilinear', align_corners=False))
        ccr_raw2 = self.ccr_prior_proj2(F.interpolate(ccr, size=ccr_feats[2].shape[-2:], mode='bilinear', align_corners=False))
        ccr_raw1 = self.ccr_prior_proj1(F.interpolate(ccr, size=ccr_feats[1].shape[-2:], mode='bilinear', align_corners=False))
        ccr_prior3 = ccr_feats[3] + ccr_raw3
        ccr_prior2 = ccr_feats[2] + ccr_raw2
        ccr_prior1 = ccr_feats[1] + ccr_raw1

        gamma, beta = self.illuminant_descriptor(rgb, valid_mask)
        z_b = z_global * (1.0 + gamma) + beta
        # film_global = torch.cat([gamma, beta], dim=1)

        d_g = self.decoder_a(
            z_global,
            skip_features,
            stage_ops=[
                lambda x: self.sfm_a_ccr3(self.sfm_a_n3(x, normal_feats[3]), ccr_prior3),
                lambda x: self.sfm_a_ccr2(self.sfm_a_n2(x, normal_feats[2]), ccr_prior2),
                lambda x: self.sfm_a_ccr1(self.sfm_a_n1(x, normal_feats[1]), ccr_prior1),
                lambda x: self.sfm_a_ccr0(x, ccr_feats[0]),
            ],
        )
        s_g = 1.0 / (d_g + 1e-6) - 1.0
        s_g_safe = s_g.detach().clamp(0.0, 20.0)
        # allow gradient flow to Dec A
        a_g = self._derive_albedo(rgb, s_g_safe)

        g_b = self.guidance_b(torch.cat([s_g_safe, a_g], dim=1))
        xi = self.decoder_b(
            z_b,
            skip_features,
            stage_ops=[
                lambda x: self.sfm_b3(x, g_b[3]),
                lambda x: self.sfm_b2(x, g_b[2]),
                lambda x: self.sfm_b1(x, g_b[1]),
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
                lambda x: self.sfm_c_g3(self.spade_c3(self.sfm_c_ccr3(x, ccr_prior3), seg), g_c[3]),
                lambda x: self.sfm_c_g2(self.spade_c2(self.sfm_c_ccr2(x, ccr_prior2), seg), g_c[2]),
                lambda x: self.sfm_c_g1(self.spade_c1(self.sfm_c_ccr1(x, ccr_prior1), seg), g_c[1]),
                lambda x: self.sfm_c_ccr0(x, ccr_feats[0]),
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
