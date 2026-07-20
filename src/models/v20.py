"""V20 — shading-first colored-illumination derive (FINAL, post expert review 2026-06-23).

Albedo is DERIVED, not regressed. Two trainable heads on a frozen-DINOv2-L + DPT trunk:

  I → DINOv2-L/14 (frozen) → DPT trunk F → TWO heads (shading-FIRST ordering):
    1. Gray shading g(x)       — achromatic intensity, full-res, luminance-skip (sharp edges)
    2. Dense chroma field c(x)  — LOW-RES (cannot encode texture) colored illuminant, CONDITIONED
                                  on the detached gray shading d_g (shading-first: commit intensity,
                                  then explain residual colour on top → closes the g↔c gauge)
  Compose : S = g·c  ;  a = derive_albedo(I, S) = (I/S) Q99-normed  ;  R = (I − a·S)₊  ;  I ≈ a·S + R

Why this shape (the expert-team verdict that replaced the K-palette MICC):
  • The previous K=6 palette + soft assignment was UNIDENTIFIABLE on white-illuminant data
    (c_k unit-luminance ⇒ L carried only chroma ⇒ on Hypersim the correct chroma is neutral,
    reachable by ANY blend ⇒ no data gradient on the assignment for 30k steps ⇒ it collapsed to a
    green central blob). A dense LOW-RES chroma field keeps the "low-rank illuminant cannot carry
    texture" constancy lever VERBATIM but removes the softmax/assignment over-parameterisation and
    the green-collapse failure mode. (Recover a K-colour palette POST-HOC by clustering c(x) for
    figures.) Constancy is enforced by the LOSS (CARI L_inv), not the head — the head just has to be
    a clean, identifiable, sharp derive.
  • Residual R is ANALYTIC (I − a·S)₊, not a learned head — a free additive R is a garbage-collector
    that absorbs diffuse content as haze (the "residual blob bug" cured once already in V17→V18).
  • derive_albedo runs in FP32 (iid_utils) — the FP16 divide silently zeroed albedo via posinf→0.
  • No albedo_refine: the learned Δ blurred the sharp derive (measured at iter 10k); a ≡ derive.

DINOv2 backbone FROZEN. Data: ¼ Hypersim + MID. CARI L_inv on a; shadow self-sup (Hypersim) on S.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders.dino_encoder import DINOv2Encoder
from .decoders.dpt_decoder import DPTTrunk, _gn
from .iid_utils import invert, uninvert, derive_albedo


class DecodeHead(nn.Module):
    """Shared trunk (/2) + optional full-res skip → raw output at out_size."""
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


class IntrinsicDecompositionV20(nn.Module):
    """Shading-first colored-illumination derive (see module docstring)."""
    def __init__(self, config):
        super().__init__()

        self.pi_floor = float(config.get('pi_floor', 5e-3))
        # The colored illuminant is LOW-FREQUENCY by construction: predict the chroma field at this
        # small resolution then upsample, so it physically cannot encode material texture (the
        # constancy lever) — far smaller than the input, no softmax/assignment.
        self.chroma_base_size = int(config.get('chroma_base_size', 48))
        # Plausible-illuminant clamp on the unit-luminance chroma (1.0 = neutral). A free softplus
        # channel can blow up where the DPT/ViT edge features are unreliable (the image border), and
        # the unit-luminance divide then turns it into a saturated green/magenta FRINGE that the
        # a=I/S derive inherits as its complement. Real illuminants are only mildly-to-moderately
        # chromatic, so we clamp the per-channel chromaticity and re-normalise. Resume-safe (no params).
        self.chroma_sat_min = float(config.get('chroma_sat_min', 0.3))
        self.chroma_sat_max = float(config.get('chroma_sat_max', 3.0))

        # ── Frozen DINOv2 backbone ──────────────────────────────────────────
        self.encoder = DINOv2Encoder(
            variant=config.get('dino_variant', 'large'),   # LARGE for best features (as V17)
            pretrained=bool(config.get('dino_pretrained', True)),
        )
        dino_dim = self.encoder.embed_dim

        # ── DPT decoder (reassemble + fusion + detail stem) ─────────────────
        self.trunk = DPTTrunk(
            in_dim=dino_dim,
            feat_ch=int(config.get('dpt_feat_ch', 512)),
            fusion_ch=int(config.get('dpt_fusion_ch', 256)),
            out_ch=int(config.get('dpt_out_ch', 256)),
            detail_ch=int(config.get('detail_ch', 64)),
        )
        trunk_ch = self.trunk.out_channels
        head_mid = int(config.get('head_mid', 128))

        # ── Two heads ───────────────────────────────────────────────────────
        # 1. Gray shading g — single-res, full-res, luminance skip for sharp shadow edges. (The DPT
        #    trunk already fuses /32→/2 with a raw-RGB detail stem, so a second hand-rolled multi-res
        #    base/detail split is redundant — one head suffices.)
        self.shading_head = DecodeHead(trunk_ch, mid=head_mid, out_ch=1, skip_ch=1)
        # 2. Dense LOW-RES chroma field (the colored illuminant), CONDITIONED on the detached gray
        #    shading d_g (1ch) + a dedicated input-chromaticity cue c(I) (3ch) → skip_ch=4. DINOv2 is
        #    colour-INVARIANT, so the trunk alone starves this colour head of illuminant info (washed
        #    chroma); c(I) gives it a direct colour reference (the V12/V16 CCR philosophy). shading-first
        #    d_g.detach() keeps the g↔c gauge closed; low-res ⇒ no texture leak; feeding colour to the
        #    SHADING factor (albedo is DERIVED) avoids V17's chroma→albedo leak. Output unit-luminance.
        self.chroma_head = DecodeHead(trunk_ch, mid=head_mid, out_ch=3, skip_ch=4)

    def _base_hw(self, H, W, base_size):
        s = float(base_size) / float(max(H, W))
        return max(16, int(round(H * s))), max(16, int(round(W * s)))

    def forward(self, rgb, **kwargs):
        B, C, H, W = rgb.shape
        eps = 1e-6

        # Frozen DINOv2 features → shared DPT trunk
        dino_feats, dino_tokens, dino_patch_hw = self.encoder(rgb)
        f_trunk = self.trunk(dino_feats, rgb, (H, W))

        # ── 1. Gray shading FIRST (achromatic intensity; full-res, luminance-skip → sharp edges) ──
        gamma_lum = ((0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3])
                     .clamp(0.0, 1.0) + 1e-6).pow(1.0 / 2.2)
        d_g = torch.sigmoid(self.shading_head(f_trunk, out_size=(H, W), skip=gamma_lum))
        d_g = d_g.clamp(self.pi_floor, 1.0 - 1e-4)         # π-domain gray shading
        g = uninvert(d_g)                                  # achromatic intensity (linear)

        # ── 2. Dense low-res chroma field, CONDITIONED on d_g (gray, detached) + input chromaticity ──
        # c(I)=I/ΣI is the dedicated illuminant-colour cue (DINOv2 trunk is colour-invariant). Low-res
        # head ⇒ texture cannot survive; d_g.detach() keeps the gauge closed (see __init__).
        chroma_hw = self._base_hw(H, W, self.chroma_base_size)
        in_chroma = rgb / (rgb.sum(dim=1, keepdim=True) + eps)        # (B,3,H,W) input chromaticity cue
        chroma_skip = torch.cat([d_g.detach(), in_chroma], dim=1)     # (B,4,H,W): committed gray | colour
        chroma_low = F.softplus(self.chroma_head(f_trunk, out_size=chroma_hw, skip=chroma_skip)) + 1e-3
        chroma = F.interpolate(chroma_low, size=(H, W), mode='bilinear', align_corners=False)
        lum_c = (0.299 * chroma[:, 0:1] + 0.587 * chroma[:, 1:2] + 0.114 * chroma[:, 2:3])
        chroma = chroma / (lum_c + eps)                    # unit-luminance ⇒ colour only
        # Tame the border fringe: clamp to a plausible illuminant chromaticity, then re-normalise back
        # to unit-luminance (the clamp breaks luminance=1, so divide again). See chroma_sat_* above.
        chroma = chroma.clamp(self.chroma_sat_min, self.chroma_sat_max)
        lum_c = (0.299 * chroma[:, 0:1] + 0.587 * chroma[:, 1:2] + 0.114 * chroma[:, 2:3])
        chroma = chroma / (lum_c + eps)

        # ── Compose colored shading, DERIVE albedo (FP32), ANALYTIC residual ────────────────
        s = (g * chroma).clamp(min=eps)                    # S = g·chroma (colored diffuse shading)
        # anneal: train loop ramps 0→1 (warmup Q99 scale → identity derive); default 1.0 for eval.
        a_d = derive_albedo(rgb, s, anneal=float(kwargs.get('derive_anneal', 1.0)))
        # Analytic R = (I − a·S)₊. At anneal=1 a=clamp(I/S,0,1) ≤ 1 ⇒ a·S ≤ I everywhere,
        # so R→0 on diffuse and is non-zero ONLY at true speculars/lights (where I/S > 1).
        # No recon loss; R is used for viz and eval only.
        residual = (rgb - a_d * s).clamp(min=0.0)
        shading_pi = invert(s).clamp(self.pi_floor, 1.0 - 1e-4)   # π of S → SSI vs pi_star

        return {
            'a_d': a_d,                        # derived albedo — the output; all losses act here
            'shading': shading_pi,             # diffuse shading (π) — SSI vs pi_star
            'shading_linear': s,               # diffuse shading S = g·chroma (linear)
            'residual': residual,              # analytic R=(I−A·S)₊ (viz/eval; ~0 at convergence)
            'chroma_field': chroma,            # (B,3,H,W) dense colored illuminant (viz)
            'd_g': d_g,                        # gray shading (viz)
            'dino_tokens': dino_tokens,
            'dino_patch_hw': dino_patch_hw,
            'rgb_reconstructed': a_d * s + residual,
        }
