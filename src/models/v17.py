"""V17: Parallel Joint Factorization with Specular Residual.

Architecture:
  - Base: ConvNeXt-V2 (Base)
  - Bottleneck: Task-specific 1x1 projections from shared z_global
  - Prior Injection: SPADE-style (PhysicalPriorInjector with GroupNorm)
  - Dec A (Albedo): 3-channel, CCR + Segmentation priors, sigmoid activation
  - Dec S (Diffuse Shading): 3-channel, Normal priors, softplus activation
  - Dec R (Specular Residual): 3-channel, Normal priors, linear activation
  - Equation: I = A * S_d + R

Design rationale:
  - Parallel execution prevents cascade error propagation
  - Specular residual absorbs non-Lambertian effects (highlights, glass, metal)
  - Reconstruction loss L1(A*S+R, I) couples the heads via physics, not attention
  - CTAB (Gemini plan) is not used: the reconstruction loss provides stronger
    coupling than attention over 3 tokens, at lower computational cost
  - No frequency-aware skip adapter: MSG p=2 on shading loss already provides
    the smoothness prior; architectural filtering removes useful edge info
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders.image_encoder import ImageEncoder
from .encoders.prior_encoder import PriorEncoder
from .decoders.modern_decoder import ModernProgressiveDecoder
from .modules.prior_injector import PhysicalPriorInjector
from .ccr_utils import compute_ccr


class IntrinsicDecompositionV17(nn.Module):
    def __init__(self, config):
        super().__init__()

        z_channels = config.get('z_channels', 1024)
        freeze_stages = config.get('freeze_stages', [1, 2])
        model_name = config.get(
            'backbone', config.get('model_name', 'convnextv2_base')
        )
        pretrained = config.get('pretrained', True)
        num_seg_classes = int(config.get('num_seg_classes', 41))
        self.max_residual = float(config.get('max_residual', 5.0))

        # Shared backbone
        self.image_encoder = ImageEncoder(
            model_name=model_name,
            freeze_stages=freeze_stages,
            pretrained=pretrained,
        )
        skip_channels = self.image_encoder.feature_channels

        # Prior encoders (shared across decoders)
        self.ccr_encoder = PriorEncoder(
            in_channels=6, channels=(64, 128, 256, 512), full_res_stage=True
        )
        self.normal_encoder = PriorEncoder(
            in_channels=3, channels=(64, 128, 256, 512), full_res_stage=False
        )

        # Task-specific bottleneck projections.
        # Lightweight 1x1 conv gives each decoder its own "view" of z_global.
        # The reconstruction loss provides cross-task coupling.
        self.proj_albedo = nn.Conv2d(z_channels, z_channels, 1)
        self.proj_shading = nn.Conv2d(z_channels, z_channels, 1)
        self.proj_residual = nn.Conv2d(z_channels, z_channels, 1)

        # Initialize projections near identity (small perturbation)
        for proj in [self.proj_albedo, self.proj_shading, self.proj_residual]:
            nn.init.eye_(proj.weight.view(z_channels, z_channels))
            nn.init.zeros_(proj.bias)

        # Albedo decoder: CCR + Seg priors, sigmoid output [0, 1]
        self.decoder_albedo = ModernProgressiveDecoder(
            z_channels, skip_channels, out_channels=3, activation='sigmoid'
        )
        self.spade_albedo3 = PhysicalPriorInjector(
            768, 512, use_seg=True, num_seg_classes=num_seg_classes, normalize=False
        )
        self.spade_albedo2 = PhysicalPriorInjector(
            384, 256, use_seg=True, num_seg_classes=num_seg_classes, normalize=False
        )
        self.spade_albedo1 = PhysicalPriorInjector(
            192, 128, use_seg=True, num_seg_classes=num_seg_classes, normalize=False
        )

        # Shading decoder: Normal priors, softplus output (non-negative, unbounded)
        self.decoder_shading = ModernProgressiveDecoder(
            z_channels, skip_channels, out_channels=3, activation='softplus'
        )
        self.spade_shading3 = PhysicalPriorInjector(768, 512)
        self.spade_shading2 = PhysicalPriorInjector(384, 256)
        self.spade_shading1 = PhysicalPriorInjector(192, 128)

        # Residual decoder: Normal priors, linear output (can be negative)
        self.decoder_residual = ModernProgressiveDecoder(
            z_channels, skip_channels, out_channels=3, activation='linear'
        )
        self.spade_residual3 = PhysicalPriorInjector(768, 512)
        self.spade_residual2 = PhysicalPriorInjector(384, 256)
        self.spade_residual1 = PhysicalPriorInjector(192, 128)

    def forward(self, rgb, normals=None, seg=None, ccr=None,
                valid_mask=None, m_diffuse=None, **kwargs):
        if seg is None:
            seg = torch.zeros(
                (rgb.shape[0], 1, rgb.shape[2], rgb.shape[3]),
                dtype=torch.long, device=rgb.device,
            )

        # Gamma-encode for ConvNeXtV2
        x_enc = (rgb.clamp(0.0, 1.0) + 1e-6).pow(1.0 / 2.2)
        z_global, skip_features = self.image_encoder(x_enc)

        # Prior features
        if normals is None:
            normals = torch.zeros_like(rgb)
        normal_feats = self.normal_encoder(normals)

        if ccr is None:
            with torch.no_grad():
                ccr = compute_ccr(rgb)
        ccr_feats = self.ccr_encoder(ccr)

        # Task-specific projections from shared bottleneck
        z_a = self.proj_albedo(z_global)
        z_s = self.proj_shading(z_global)
        z_r = self.proj_residual(z_global)

        # ---- Albedo (parallel) ----
        albedo = self.decoder_albedo(
            z_a,
            skip_features,
            stage_ops=[
                lambda x: self.spade_albedo3(x, ccr_feats[4], seg),
                lambda x: self.spade_albedo2(x, ccr_feats[3], seg),
                lambda x: self.spade_albedo1(x, ccr_feats[2], seg),
            ],
        ).clamp(1e-4, 1.0)

        # ---- Diffuse Shading (parallel) ----
        shading = self.decoder_shading(
            z_s,
            skip_features,
            stage_ops=[
                lambda x: self.spade_shading3(x, normal_feats[3]),
                lambda x: self.spade_shading2(x, normal_feats[2]),
                lambda x: self.spade_shading1(x, normal_feats[1]),
            ],
        ).clamp(min=1e-4)

        # ---- Specular Residual (parallel) ----
        residual = self.decoder_residual(
            z_r,
            skip_features,
            stage_ops=[
                lambda x: self.spade_residual3(x, normal_feats[3]),
                lambda x: self.spade_residual2(x, normal_feats[2]),
                lambda x: self.spade_residual1(x, normal_feats[1]),
            ],
        ).clamp(-self.max_residual, self.max_residual)

        # Physics-based reconstruction (used for loss, also returned for viz)
        rgb_reconstructed = (albedo * shading) + residual

        return {
            'a_d': albedo,
            'shading': shading,
            'residual': residual,
            'rgb_reconstructed': rgb_reconstructed,
        }
