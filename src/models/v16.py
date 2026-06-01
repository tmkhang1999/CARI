"""Stage 1 V16: Improved cascade with SPADE prior injection.

Changes from original V16:
  - SFM replaced with PhysicalPriorInjector (SPADE-based, GroupNorm before
    modulation). The original SFM lacked normalization, making the affine
    transform dependent on absolute feature magnitude.
  - Guidance encoders B/C/D are RETAINED -- they carry non-redundant
    intermediate cascade information that Dec B/C/D need for disentanglement:
      guidance_b: IUV(rgb) + d_g + IUV(a_g)  -> chroma shading context
      guidance_c: invert(s_c) + a_c           -> colorful shading + albedo hint
      guidance_d: invert(net_shd) + a_d       -> net ratio map + predicted albedo
  - CCR and Normal priors injected via SPADE on top of, not instead of,
    the guidance encoder signals (dual injection per decoder stage)
  - torch.no_grad() on CCR fallback computation in forward pass

Architecture:
  - Base: ConvNeXt-V2 (Base)
  - Prior Encoders: PriorEncoder (CCR + Normals)
  - Guidance Encoders: PriorEncoder B/C/D (cascade intermediate signals)
  - Dec A: Grayscale inverse shading (Normal + CCR priors)
  - Dec B: Chroma shading (Guidance_B priors)
  - Dec C: Albedo (Guidance_C + CCR + Segmentation priors)
  - Dec D: Diffuse shading (Guidance_D + Normal priors)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders.image_encoder import ImageEncoder
from .encoders.prior_encoder import PriorEncoder
from .decoders.modern_decoder import ModernProgressiveDecoder
from .modules.prior_injector import PhysicalPriorInjector
from .ccr_utils import compute_ccr
from .iid_utils import rgb_to_iuv, iuv_to_rgb, invert, uninvert, derive_albedo, resize_to_base


def _make_adapter(channels, reduction=4):
    """Lightweight task-specific bottleneck adapter."""
    hidden = channels // reduction
    adapter = nn.Sequential(
        nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
        nn.GELU(),
        nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
    )
    nn.init.kaiming_normal_(adapter[-1].weight)
    adapter[-1].weight.data.mul_(0.1)
    nn.init.zeros_(adapter[-1].bias)
    return adapter


class IntrinsicDecompositionV16(nn.Module):
    def __init__(self, config):
        super().__init__()

        z_channels = config.get('z_channels', 1024)
        freeze_stages = config.get('freeze_stages', [1, 2])
        model_name = config.get(
            'backbone', config.get('model_name', 'convnextv2_base')
        )
        pretrained = config.get('pretrained', True)
        self.base_size = int(config.get('base_size', 384))
        num_seg_classes = int(config.get('num_seg_classes', 41))

        self.image_encoder = ImageEncoder(
            model_name=model_name,
            freeze_stages=freeze_stages,
            pretrained=pretrained,
        )
        skip_channels = self.image_encoder.feature_channels

        # Auxiliary prior encoders (physics-derived features)
        self.ccr_encoder = PriorEncoder(
            in_channels=6, channels=(64, 128, 256, 512), full_res_stage=True
        )
        self.ccr_edge_head = nn.Conv2d(64, 1, 1)
        self.normal_encoder = PriorEncoder(
            in_channels=3, channels=(64, 128, 256, 512), full_res_stage=False
        )

        # Guidance encoders: carry cascade intermediate state.
        # These are NOT redundant with the prior encoders:
        #   guidance_b encodes iuv(rgb) + d_g + iuv(a_g) (7ch) -- colorful shading context
        #   guidance_c encodes invert(s_c) + a_c (6ch)          -- colorful shading + derived albedo
        #   guidance_d encodes invert(net_shd) + a_d (6ch)      -- net ratio map + albedo prediction
        self.guidance_b = PriorEncoder(
            in_channels=7, channels=(64, 128, 256, 512), full_res_stage=False
        )
        self.guidance_c = PriorEncoder(
            in_channels=6, channels=(64, 128, 256, 512), full_res_stage=False
        )
        self.guidance_d = PriorEncoder(
            in_channels=6, channels=(64, 128, 256, 512), full_res_stage=False
        )

        # Task-specific bottleneck adapters
        self.z_adapt_a = _make_adapter(z_channels)
        self.z_adapt_b = _make_adapter(z_channels)
        self.z_adapt_c = _make_adapter(z_channels)
        self.z_adapt_d = _make_adapter(z_channels)

        # Decoders
        self.decoder_a = ModernProgressiveDecoder(
            z_channels, skip_channels, 1, (0, 0, 0), 'sigmoid'
        )
        self.decoder_b = ModernProgressiveDecoder(
            z_channels, skip_channels, 2, (0, 0, 0), 'sigmoid'
        )
        self.decoder_c = ModernProgressiveDecoder(
            z_channels, skip_channels, 3, (0, 0, 0), 'sigmoid'
        )
        self.decoder_d = ModernProgressiveDecoder(
            z_channels, skip_channels, 3, (0, 0, 0), 'sigmoid'
        )

        # Dec A injectors: Normal priors ONLY.
        # Shading is purely geometric (N . L). CCR carries material/albedo
        # edges -- injecting them here creates a shortcut for the network to
        # copy albedo boundaries into the shading map (leakage).
        self.pi_a_n3 = PhysicalPriorInjector(768, 512)
        self.pi_a_n2 = PhysicalPriorInjector(384, 256)
        self.pi_a_n1 = PhysicalPriorInjector(192, 128)

        # Dec B injectors: guidance_b only (chroma needs cascade context, not normals)
        self.pi_b3 = PhysicalPriorInjector(768, 512)
        self.pi_b2 = PhysicalPriorInjector(384, 256)
        self.pi_b1 = PhysicalPriorInjector(192, 128)

        # Dec C injectors: guidance_c + CCR + Segmentation (dual injection)
        self.pi_c_g3   = PhysicalPriorInjector(768, 512)
        self.pi_c_ccr3 = PhysicalPriorInjector(
            768, 512, use_seg=True, num_seg_classes=num_seg_classes, normalize=False
        )
        self.pi_c_g2   = PhysicalPriorInjector(384, 256)
        self.pi_c_ccr2 = PhysicalPriorInjector(
            384, 256, use_seg=True, num_seg_classes=num_seg_classes, normalize=False
        )
        self.pi_c_g1   = PhysicalPriorInjector(192, 128)
        self.pi_c_ccr1 = PhysicalPriorInjector(
            192, 128, use_seg=True, num_seg_classes=num_seg_classes, normalize=False
        )

        # Dec D injectors: guidance_d + Normal priors (dual injection)
        self.pi_d_n3 = PhysicalPriorInjector(768, 512)
        self.pi_d_g3 = PhysicalPriorInjector(768, 512)
        self.pi_d_n2 = PhysicalPriorInjector(384, 256)
        self.pi_d_g2 = PhysicalPriorInjector(384, 256)
        self.pi_d_n1 = PhysicalPriorInjector(192, 128)
        self.pi_d_g1 = PhysicalPriorInjector(192, 128)

    def forward(self, rgb, m_diffuse=None, normals=None, seg=None,
                valid_mask=None, ccr=None, **kwargs):
        if seg is None:
            seg = torch.zeros(
                (rgb.shape[0], 1, rgb.shape[2], rgb.shape[3]),
                dtype=torch.long, device=rgb.device,
            )

        # Gamma-encode for ConvNeXtV2 (expects sRGB-like input)
        x_enc = (rgb.clamp(0.0, 1.0) + 1e-6).pow(1.0 / 2.2)
        z_global, skip_features = self.image_encoder(x_enc)

        if normals is None:
            normals = torch.zeros_like(rgb)
        normal_feats = self.normal_encoder(normals)

        # CCR: hand-crafted feature, never needs gradients
        if ccr is None:
            with torch.no_grad():
                ccr = compute_ccr(rgb)
        ccr_feats = self.ccr_encoder(ccr)
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
                lambda x: self.pi_a_n3(x, normal_feats[3]),
                lambda x: self.pi_a_n2(x, normal_feats[2]),
                lambda x: self.pi_a_n1(x, normal_feats[1]),
            ],
        )
        s_g = uninvert(d_g)
        a_g = derive_albedo(rgb, s_g.detach())

        # ---- Dec B: Chroma (UV channels of shading) ----
        # guidance_b carries the cascade state: iuv(rgb) + gray_shading + rough_albedo.
        # Dec B only needs to predict the chrominance deviation, not the full signal.
        g_b_in = torch.cat([rgb_to_iuv(rgb), d_g.detach(), rgb_to_iuv(a_g)], dim=1)
        g_b = self.guidance_b(resize_to_base(g_b_in, self.base_size))
        xi = self.decoder_b(
            z_b,
            skip_features,
            stage_ops=[
                lambda x: self.pi_b3(x, g_b[3]),
                lambda x: self.pi_b2(x, g_b[2]),
                lambda x: self.pi_b1(x, g_b[1]),
            ],
        )
        iuv_shd = torch.cat([d_g, xi], dim=1)
        s_c = iuv_to_rgb(iuv_shd)
        s_c_safe = s_c.detach().clamp(min=1e-4)
        a_c = derive_albedo(rgb, s_c_safe)

        # ---- Dec C: Albedo prediction ----
        # guidance_c carries: invert(s_c) + a_c -- tells Dec C what the colorful
        # shading looks like, so albedo prediction doesn't need to re-derive it.
        g_c_in = torch.cat([invert(s_c_safe), a_c], dim=1)
        g_c = self.guidance_c(resize_to_base(g_c_in, self.base_size))
        a_d = self.decoder_c(
            z_c,
            skip_features,
            stage_ops=[
                lambda x: self.pi_c_ccr3(self.pi_c_g3(x, g_c[3]), ccr_feats[4], seg),
                lambda x: self.pi_c_ccr2(self.pi_c_g2(x, g_c[2]), ccr_feats[3], seg),
                lambda x: self.pi_c_ccr1(self.pi_c_g1(x, g_c[1]), ccr_feats[2], seg),
            ],
        )

        # ---- Dec D: Diffuse shading (inverse space) ----
        # guidance_d carries: invert(rgb/a_d) + a_d -- the net ratio map is the
        # closest available estimate of the shading signal, derived from albedo.
        net_clr_shd = rgb / a_d.detach().clamp(min=1e-3)
        net_clr_shd = torch.nan_to_num(net_clr_shd, nan=0.0, posinf=0.0, neginf=0.0)
        g_d_in = torch.cat([invert(net_clr_shd), a_d.detach()], dim=1)
        g_d = self.guidance_d(resize_to_base(g_d_in, self.base_size))
        pi = self.decoder_d(
            z_d,
            skip_features,
            stage_ops=[
                lambda x: self.pi_d_n3(self.pi_d_g3(x, g_d[3]), normal_feats[3]),
                lambda x: self.pi_d_n2(self.pi_d_g2(x, g_d[2]), normal_feats[2]),
                lambda x: self.pi_d_n1(self.pi_d_g1(x, g_d[1]), normal_feats[1]),
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