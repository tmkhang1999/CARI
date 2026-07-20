"""V17 — Depth-Anything-for-IID: frozen DINOv2-L backbone + DPT + physics-typed skips.

Replaces the old ConvNeXt parallel-head / derive-albedo design. The thesis:

  Intrinsic decomposition fails on real photos for two reasons — a domain-shifted
  ImageNet backbone, and an A=I/S division that explodes where shading is small
  (the black/blue blobs at ckpt 4000). Both are fixed by reading from a frozen
  DINOv2 backbone whose patch tokens are illumination-INVARIANT: a shadowed and a
  lit patch of the same material map to similar tokens, so a DPT decoder emits the
  SAME albedo across a shadow boundary. Shadow removal becomes a feature property,
  not a generative act (the Depth-Anything recipe applied to IID).

Physics-typed input skips (OFF by default — kept only as an ablation lever):
  An earlier hypothesis re-injected image-derived color to fight DINOv2's
  color-invariance desaturation: chromaticity c(I)=I/(R+G+B)→albedo, luminance
  L(I)→shading. Dropped as a default (see `albedo_chroma_skip`/`shading_lum_skip`,
  both false) for three reasons:
    (1) the typing was asymmetric — the shading head is 3-ch (COLORED) yet got only a
        1-ch luminance skip, while albedo got the colored chroma skip → color hint fed
        to the wrong head;
    (2) c(I) cancels only WHITE shading, so colored illumination / clipping leaks the
        illuminant straight into albedo (a desaturation FIX that becomes a color-leak);
    (3) the model already models colored illumination — S_d is 3-ch and the recon
        target A·S_d carries illuminant color, so a hand-crafted color skip adds no
        capability, only a white-light failure mode.
  Real-image desaturation is a training-SIGNAL gap (DINOv2 discards absolute color and
  synthetic recon can't correct it on real photos) → addressed by a real-image signal
  (planned Phase-3 MLLM judge), not a 2D input prior. The skip code is retained so the
  ablation "with vs. without typed skips" can be run from config.

Pipeline:
  I → DINOv2-L/14 (frozen) → 4 intermediate maps → DPT reassemble+fusion + a conv
  detail stem → shared trunk F. Albedo head → A; shading head → π=1/(S_d+1)
  (3-ch colored shading). Residual ANALYTIC: R=(I−A·S_d)₊. I = A·S_d + R.

Trained with V17Loss in `diffuse` mode (albedo MSE+MSG+DSSIM, SSI shading, two-sided
diffuse reconstruction A·S_d) on clean synthetic data. No derive_albedo, no chroma_wb,
no CCR-as-feature, no residual decoder, no external geometry/semantic prior.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders.dino_encoder import DINOv2Encoder
from .decoders.dpt_decoder import DPTTrunk, _gn


def image_chromaticity(rgb: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """c(I) = I / (R+G+B). In LINEAR RGB this cancels scalar (white) shading, so a
    shadow boundary vanishes — it carries material color, not shadow-intensity edges."""
    rgb = rgb.clamp(0.0, 1.0)
    s = rgb.sum(dim=1, keepdim=True) + eps
    return rgb / s                                    # (B, 3, H, W), channels ~sum to 1


def image_luminance(rgb: torch.Tensor) -> torch.Tensor:
    """L(I) = 0.299R + 0.587G + 0.114B — the intensity / shading-carrying channel."""
    rgb = rgb.clamp(0.0, 1.0)
    return 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]   # (B,1,H,W)


class DecodeHead(nn.Module):
    """Shared trunk (/2) + optional full-res input-derived skip → full-res raw output.

    Two 3×3 convs at /2 (cheap), bilinear upsample to full res, then — if a typed skip
    is given — concat a small conv encoding of it (chromaticity for albedo, luminance
    for shading) at full res, one 3×3 refine, and a 1×1 projection. Returns raw logits.
    """
    def __init__(self, in_ch: int, mid: int = 64, out_ch: int = 3, skip_ch: int = 0,
                 skip_feat: int = 32):
        super().__init__()
        self.skip_ch = int(skip_ch)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=1, bias=False), _gn(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1, bias=False), _gn(mid), nn.ReLU(inplace=True),
        )
        refine_in = mid
        if self.skip_ch > 0:
            self.skip_enc = nn.Sequential(
                nn.Conv2d(self.skip_ch, skip_feat, 3, padding=1, bias=False), _gn(skip_feat), nn.ReLU(inplace=True),
                nn.Conv2d(skip_feat, skip_feat, 3, padding=1, bias=False), _gn(skip_feat), nn.ReLU(inplace=True),
            )
            refine_in = mid + skip_feat
        self.refine = nn.Sequential(
            nn.Conv2d(refine_in, mid, 3, padding=1, bias=False), _gn(mid), nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(mid, out_ch, kernel_size=1)

    def forward(self, feat, out_size, skip=None):
        x = self.block(feat)
        x = F.interpolate(x, size=out_size, mode='bilinear', align_corners=False)
        if self.skip_ch > 0 and skip is not None:
            if skip.shape[-2:] != x.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, self.skip_enc(skip)], dim=1)
        x = self.refine(x)
        return self.out(x)


class IntrinsicDecompositionV17(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.pi_floor = float(config.get('pi_floor', 5e-3))
        # Physics-typed input skips (default on; gated for ablation)
        self.use_chroma_skip = bool(config.get('albedo_chroma_skip', True))
        self.use_lum_skip = bool(config.get('shading_lum_skip', True))
        # RGB-skip into albedo head only (A2 fix for desaturation):
        # passes gamma-encoded raw RGB directly to the albedo DecodeHead at full resolution.
        # Wider color path than the 64-ch DetailStem → helps high-chroma prediction.
        # MUST be paired with L_inv; without cross-render loss the illuminant color
        # leaks back into albedo (see V19_CARI_design.md §3.1 A2).
        self.use_rgb_skip = bool(config.get('albedo_rgb_skip', False))

        # ── Frozen DINOv2 backbone ──────────────────────────────────────────
        self.encoder = DINOv2Encoder(
            variant=config.get('dino_variant', 'large'),
            pretrained=bool(config.get('dino_pretrained', True)),
        )
        dino_dim = self.encoder.embed_dim

        # ── DPT decoder (reassemble + fusion + detail stem) ─────────────────
        self.trunk = DPTTrunk(
            in_dim=dino_dim,
            feat_ch=int(config.get('dpt_feat_ch', 256)),
            fusion_ch=int(config.get('dpt_fusion_ch', 128)),
            out_ch=int(config.get('dpt_out_ch', 128)),
            detail_ch=int(config.get('detail_ch', 48)),
        )
        trunk_ch = self.trunk.out_channels

        # ── Heads (albedo + π-shading) with physics-typed full-res skips ────
        head_mid = int(config.get('head_mid', 64))
        # albedo_rgb_skip takes precedence over albedo_chroma_skip: if both are set,
        # rgb_skip wins (wider 3-ch raw RGB path instead of normalised chromaticity).
        alb_skip_ch = 0
        if self.use_rgb_skip:
            alb_skip_ch = 3     # raw gamma-encoded RGB — full color capacity
        elif self.use_chroma_skip:
            alb_skip_ch = 3     # normalised chromaticity (legacy)
        self.albedo_head = DecodeHead(
            trunk_ch, mid=head_mid, out_ch=3, skip_ch=alb_skip_ch)
        self.shading_head = DecodeHead(
            trunk_ch, mid=head_mid, out_ch=3, skip_ch=1 if self.use_lum_skip else 0)

    def forward(self, rgb, **kwargs):
        B, C, H, W = rgb.shape

        # Frozen DINOv2 features (illumination-invariant material tokens)
        dino_feats, dino_tokens, dino_patch_hw = self.encoder(rgb)

        # Shared DPT trunk at /2
        f_trunk = self.trunk(dino_feats, rgb, (H, W))

        # Physics-typed input-derived skips (full res)
        lum = image_luminance(rgb) if self.use_lum_skip else None

        # Albedo skip: raw gamma-encoded RGB (rgb_skip) > normalised chromaticity (chroma_skip)
        if self.use_rgb_skip:
            alb_skip = (rgb.clamp(0.0, 1.0) + 1e-6).pow(1.0 / 2.2)  # gamma-encode
        elif self.use_chroma_skip:
            alb_skip = image_chromaticity(rgb)
        else:
            alb_skip = None

        # ── Albedo (direct) ─────────────────────────────────────────────────
        albedo = torch.sigmoid(
            self.albedo_head(f_trunk, out_size=(H, W), skip=alb_skip)
        ).clamp(1e-4, 1.0)

        # ── Diffuse shading (inverse domain π) ──────────────────────────────
        shading_pi = torch.sigmoid(
            self.shading_head(f_trunk, out_size=(H, W), skip=lum)
        ).clamp(self.pi_floor, 1.0 - 1e-4)
        shading_linear = (1.0 - shading_pi) / shading_pi

        # ── Analytic residual ───────────────────────────────────────────────
        diffuse = albedo * shading_linear
        residual = (rgb - diffuse).clamp(min=0.0)

        return {
            'a_d': albedo,
            'shading': shading_pi,
            'shading_linear': shading_linear,
            'residual': residual,
            'rgb_reconstructed': diffuse + residual,
            # DINOv2 tokens for the optional material-consistency loss
            'dino_tokens': dino_tokens,
            'dino_patch_hw': dino_patch_hw,
        }
