"""Stage 1 V16: Unified SFM-guided decoders with VGG-style auxiliary encoders.

Architecture:
  - Base: ConvNeXt-V2 (Base)
  - Aux Encoders: PriorEncoder (VGG-style, 2 convs per stage)
  - Dec A: Grayscale inverse shading (Normals + CCR)
  - Dec B: Chroma shading (Guidance B)
  - Dec C: Albedo (Guidance C + CCR + Seg)
  - Dec D: Diffuse shading (Guidance D + Normals)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders.image_encoder import ImageEncoder
from .encoders.prior_encoder import PriorEncoder
from .decoders.modern_decoder import ModernProgressiveDecoder
from .modules.spatial_feature_modulation import SpatialFeatureModulation
from .ccr_utils import compute_ccr
from .iid_utils import rgb_to_iuv, iuv_to_rgb, invert, uninvert, resize_to_base, derive_albedo


class IntrinsicDecompositionV16(nn.Module):
    def __init__(self, config):
        super().__init__()

        z_channels = config.get('z_channels', 1024)
        freeze_stages = config.get('freeze_stages', [1, 2])
        model_name = config.get('backbone', config.get('model_name', 'convnextv2_base'))
        pretrained = config.get('pretrained', True)
        self.base_size = int(config.get('base_size', 384))
        num_seg_classes = int(config.get('num_seg_classes', 41))

        self.image_encoder = ImageEncoder(
            model_name=model_name,
            freeze_stages=freeze_stages,
            pretrained=pretrained,
        )
        skip_channels = self.image_encoder.feature_channels

        # Auxiliary encoders
        self.ccr_encoder = PriorEncoder(in_channels=6, channels=(64, 128, 256, 512), full_res_stage=True)
        self.ccr_edge_head = nn.Conv2d(64, 1, 1) # Full-res edge prediction for exclusion loss
        self.normal_encoder = PriorEncoder(in_channels=3, channels=(64, 128, 256, 512), full_res_stage=False)

        self.guidance_b = PriorEncoder(in_channels=7, channels=(64, 128, 256, 512), full_res_stage=False)
        self.guidance_c = PriorEncoder(in_channels=6, channels=(64, 128, 256, 512), full_res_stage=False)
        self.guidance_d = PriorEncoder(in_channels=6, channels=(64, 128, 256, 512), full_res_stage=False)

        self.decoder_a = ModernProgressiveDecoder(z_channels, skip_channels, 1, (0, 0, 0), 'sigmoid')
        self.decoder_b = ModernProgressiveDecoder(z_channels, skip_channels, 2, (0, 0, 0), 'sigmoid')
        self.decoder_c = ModernProgressiveDecoder(z_channels, skip_channels, 3, (0, 0, 0), 'sigmoid')
        self.decoder_d = ModernProgressiveDecoder(z_channels, skip_channels, 3, (0, 0, 0), 'sigmoid')

        def make_adapter(in_channels, reduction=4):
            hidden = in_channels // reduction
            adapter = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
                nn.GELU(),
                nn.Conv2d(hidden, in_channels, kernel_size=1, bias=True)
            )
            # Initialize final projection with diverse, low-magnitude weights
            # to safely preserve backbone features early in training
            nn.init.kaiming_normal_(adapter[-1].weight)
            adapter[-1].weight.data.mul_(0.1)
            nn.init.zeros_(adapter[-1].bias)
            return adapter

        # Non-linear task-specific bottleneck adapters (SOTA paradigm)
        self.z_adapt_a = make_adapter(z_channels)
        self.z_adapt_b = make_adapter(z_channels)
        self.z_adapt_c = make_adapter(z_channels)
        self.z_adapt_d = make_adapter(z_channels)

        # Dec A SFM: Normals + CCR priors at 3 scales
        self.sfm_a_n3   = SpatialFeatureModulation(768, 512)
        self.sfm_a_ccr3 = SpatialFeatureModulation(768, 512)
        self.sfm_a_n2   = SpatialFeatureModulation(384, 256)
        self.sfm_a_ccr2 = SpatialFeatureModulation(384, 256)
        self.sfm_a_n1   = SpatialFeatureModulation(192, 128)
        self.sfm_a_ccr1 = SpatialFeatureModulation(192, 128)

        # Dec B SFM: Guidance_B at 3 scales
        self.sfm_b3 = SpatialFeatureModulation(768, 512)
        self.sfm_b2 = SpatialFeatureModulation(384, 256)
        self.sfm_b1 = SpatialFeatureModulation(192, 128)

        # Dec C SFM: Guidance_C + CCR priors at 3 scales
        self.sfm_c_g3   = SpatialFeatureModulation(768, 512)
        self.sfm_c_ccr3 = SpatialFeatureModulation(768, 512, use_seg=True, num_seg_classes=num_seg_classes)
        self.sfm_c_g2   = SpatialFeatureModulation(384, 256)
        self.sfm_c_ccr2 = SpatialFeatureModulation(384, 256, use_seg=True, num_seg_classes=num_seg_classes)
        self.sfm_c_g1   = SpatialFeatureModulation(192, 128)
        self.sfm_c_ccr1 = SpatialFeatureModulation(192, 128, use_seg=True, num_seg_classes=num_seg_classes)

        # Dec D SFM: Guidance_D + Normals at 3 scales
        self.sfm_d_n3 = SpatialFeatureModulation(768, 512)
        self.sfm_d_g3 = SpatialFeatureModulation(768, 512)
        self.sfm_d_n2 = SpatialFeatureModulation(384, 256)
        self.sfm_d_g2 = SpatialFeatureModulation(384, 256)
        self.sfm_d_n1 = SpatialFeatureModulation(192, 128)
        self.sfm_d_g1 = SpatialFeatureModulation(192, 128)

    def forward(self, rgb, m_diffuse=None, normals=None, seg=None, valid_mask=None, ccr=None, **kwargs):
        # Defensive seg fallback
        if seg is None:
            seg = torch.zeros((rgb.shape[0], 1, rgb.shape[2], rgb.shape[3]), dtype=torch.long, device=rgb.device)

        # Gamma-encode for ConvNeXtV2 (expects sRGB-like input)
        x_enc = (rgb.clamp(0.0, 1.0) + 1e-6).pow(1.0 / 2.2)
        z_global, skip_features = self.image_encoder(x_enc)

        if normals is None:
            normals = torch.zeros_like(rgb)
        normal_feats = self.normal_encoder(normals)

        if ccr is None:
            ccr = compute_ccr(rgb)
        ccr_feats = self.ccr_encoder(ccr)

        # Auxiliary CCR edge prediction for boundary/exclusion loss supervision
        ccr_edge_pred = self.ccr_edge_head(ccr_feats[0].detach())

        z_a = self.z_adapt_a(z_global)
        z_b = self.z_adapt_b(z_global)
        z_c = self.z_adapt_c(z_global)
        z_d = self.z_adapt_d(z_global)

        # ---- Dec A: Grayscale inverse shading ----
        d_g = self.decoder_a(
            z_a,
            skip_features,
            stage_ops=[
                lambda x: self.sfm_a_ccr3(self.sfm_a_n3(x, normal_feats[3]), ccr_feats[4]),  # H/16 (normal[3], ccr[4])
                lambda x: self.sfm_a_ccr2(self.sfm_a_n2(x, normal_feats[2]), ccr_feats[3]),  # H/8  (normal[2], ccr[3])
                lambda x: self.sfm_a_ccr1(self.sfm_a_n1(x, normal_feats[1]), ccr_feats[2]),  # H/4  (normal[1], ccr[2])
            ],
        )
        s_g = uninvert(d_g)
        a_g = derive_albedo(rgb, s_g.detach())

        # ---- Dec B: Chroma (UV channels of shading) ----
        g_b_in = torch.cat([rgb_to_iuv(rgb), d_g.detach(), rgb_to_iuv(a_g)], dim=1)
        g_b = self.guidance_b(resize_to_base(g_b_in, self.base_size))
        xi = self.decoder_b(
            z_b,
            skip_features,
            stage_ops=[
                lambda x: self.sfm_b3(x, g_b[3]),  # H/16
                lambda x: self.sfm_b2(x, g_b[2]),  # H/8
                lambda x: self.sfm_b1(x, g_b[1]),  # H/4
            ],
        )
        iuv_shd = torch.cat([d_g, xi], dim=1)
        s_c = iuv_to_rgb(iuv_shd)
        s_c_safe = s_c.detach().clamp(min=1e-4)
        a_c = derive_albedo(rgb, s_c_safe)

        # ---- Dec C: Albedo prediction ----
        g_c_in = torch.cat([invert(s_c_safe), a_c], dim=1)
        g_c = self.guidance_c(resize_to_base(g_c_in, self.base_size))
        a_d = self.decoder_c(
            z_c,
            skip_features,
            stage_ops=[
                lambda x: self.sfm_c_ccr3(self.sfm_c_g3(x, g_c[3]), ccr_feats[4], seg),  # H/16 (g_c[3], ccr[4])
                lambda x: self.sfm_c_ccr2(self.sfm_c_g2(x, g_c[2]), ccr_feats[3], seg),  # H/8  (g_c[2], ccr[3])
                lambda x: self.sfm_c_ccr1(self.sfm_c_g1(x, g_c[1]), ccr_feats[2], seg),  # H/4  (g_c[1], ccr[2])
            ],
        )

        # ---- Dec D: Diffuse shading (inverse space) ----
        net_clr_shd = rgb / a_d.detach().clamp(min=1e-3)
        net_clr_shd = torch.nan_to_num(net_clr_shd, nan=0.0, posinf=0.0, neginf=0.0)

        g_d_in = torch.cat([invert(net_clr_shd), a_d.detach()], dim=1)
        g_d = self.guidance_d(resize_to_base(g_d_in, self.base_size))
        pi = self.decoder_d(
            z_d,
            skip_features,
            stage_ops=[
                lambda x: self.sfm_d_n3(self.sfm_d_g3(x, g_d[3]), normal_feats[3]),  # H/16 (g_d[3], normal[3])
                lambda x: self.sfm_d_n2(self.sfm_d_g2(x, g_d[2]), normal_feats[2]),  # H/8  (g_d[2], normal[2])
                lambda x: self.sfm_d_n1(self.sfm_d_g1(x, g_d[1]), normal_feats[1]),  # H/4  (g_d[1], normal[1])
            ],
        )

        return {
            'd_g': d_g,
            'xi': xi,
            'a_d': a_d,
            'pi': pi,
            's_c': s_c.detach(),
            'ccr_edge_pred': ccr_edge_pred,
        }