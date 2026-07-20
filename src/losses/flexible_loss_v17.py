"""
V17 Heterogeneous Loss for Parallel Intrinsic Decomposition.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .msg_loss import MultiScaleGradientLoss

def material_consistency_loss(albedo, dino_tokens, patch_hw, threshold=0.0,
                              intensity_weight=0.0):
    """DINOv2 neighbour cosine-sim × albedo difference (GT-free, real-photo capable).

    For each pair of adjacent DINOv2 patches, with w = clamp(sim(dino_i,dino_j) − thr, 0):
        chroma term    : w · ||chroma(A_i) − chroma(A_j)||
        intensity term : w · |log L(A_i) − log L(A_j)|     (only if intensity_weight > 0)

    High DINOv2 similarity → same material → penalise albedo divergence. DINOv2's
    illumination-invariant features keep similarity high across a shadow on one material,
    so this flattens albedo WITHIN a material; at material / strong-lighting boundaries
    similarity drops and the penalty vanishes.

    The CHROMA term (always on) removes coloured-cast leak. The INTENSITY term is the
    Retinex within-material flatness prior the F-prior gives on Hypersim, but here gated by
    DINOv2 semantic similarity instead of a chroma gate — so it ALSO fires on real photos
    (IIW/MID), where it is the only signal flattening the within-material shading RAMP that
    WHDR penalises. Pooled to the patch grid (≈/14) → it targets the low-frequency ramp, not
    sub-patch texture (texture is averaged out by the pool → not blurred). log-domain +
    clamp keeps it scale-invariant and stable near dark albedo.

    Args:
        albedo           : (B, 3, H, W) predicted albedo
        dino_tokens      : (B, N, D)    frozen DINOv2 patch tokens (N = H_p*W_p)
        patch_hw         : (H_p, W_p)
        threshold        : cosine-sim floor; pairs below it are downweighted to ~0
        intensity_weight : relative weight of the log-luminance flatness term (0 = chroma only)
    """
    B, D = dino_tokens.shape[0], dino_tokens.shape[-1]
    H_p, W_p = patch_hw

    dino_spatial = dino_tokens.reshape(B, H_p, W_p, D).permute(0, 3, 1, 2)
    dino_norm = F.normalize(dino_spatial, dim=1)
    sim_h = (dino_norm[:, :, :, :-1] * dino_norm[:, :, :, 1:]).sum(dim=1)
    sim_v = (dino_norm[:, :, :-1, :] * dino_norm[:, :, 1:, :]).sum(dim=1)
    w_h = (sim_h - threshold).clamp(min=0.0)
    w_v = (sim_v - threshold).clamp(min=0.0)

    a_patch = F.adaptive_avg_pool2d(albedo, (H_p, W_p))
    a_chroma = F.normalize(a_patch.clamp(1e-6), dim=1)
    diff_h = (a_chroma[:, :, :, :-1] - a_chroma[:, :, :, 1:]).norm(dim=1)
    diff_v = (a_chroma[:, :, :-1, :] - a_chroma[:, :, 1:, :]).norm(dim=1)
    loss = (w_h * diff_h).mean() + (w_v * diff_v).mean()

    if intensity_weight > 0:
        # log-luminance within-material flatness (the WHDR / shading-ramp lever on real photos)
        a_lum = (0.299 * a_patch[:, 0:1] + 0.587 * a_patch[:, 1:2]
                 + 0.114 * a_patch[:, 2:3]).clamp(min=0.02, max=1.0)
        log_lum = torch.log(a_lum)                                   # (B,1,H_p,W_p)
        dl_h = (log_lum[:, :, :, :-1] - log_lum[:, :, :, 1:]).abs().squeeze(1)
        dl_v = (log_lum[:, :, :-1, :] - log_lum[:, :, 1:, :]).abs().squeeze(1)
        loss = loss + intensity_weight * ((w_h * dl_h).mean() + (w_v * dl_v).mean())

    return loss


def _scale_shift_align(pred, target, mask, eps=1e-6):
    """Align pred to target via least-squares scale and shift.

    Solves: argmin_{a,b} || mask * (a*pred + b - target) ||^2
    Returns the aligned prediction.
    """
    B = pred.shape[0]
    p_flat = pred.reshape(B, -1)
    t_flat = target.reshape(B, -1)
    m_flat = mask.expand_as(pred).reshape(B, -1).float()

    sum_m = m_flat.sum(dim=1, keepdim=True).clamp(min=1.0)
    p_mean = (p_flat * m_flat).sum(dim=1, keepdim=True) / sum_m
    t_mean = (t_flat * m_flat).sum(dim=1, keepdim=True) / sum_m

    p_cen = (p_flat - p_mean) * m_flat
    t_cen = (t_flat - t_mean) * m_flat

    covar = (p_cen * t_cen).sum(dim=1, keepdim=True)
    var_p = (p_cen * p_cen).sum(dim=1, keepdim=True)

    a = F.relu(covar / (var_p + eps)) + eps
    b = t_mean - a * p_mean

    shape = [-1] + [1] * (pred.ndim - 1)
    return pred * a.view(shape) + b.view(shape)


class V17Loss(nn.Module):
    """
    Clean, parallel loss design for V17.
    """
    def __init__(self, config):
        super().__init__()

        self.w_a = float(config.get('lambda_a', 1.0))
        self.w_s = float(config.get('lambda_s', 1.0))
        self.w_r = float(config.get('lambda_r', 0.0))
        self.w_recon = float(config.get('lambda_recon', 1.0))

        # ── Absolute shading-scale pin (V20 collapse fix, 2026-06-28) ──────────────
        # The gray-SSI is scale-SHIFT aligned → the absolute scale of g (and hence S) is a FREE
        # GAUGE. With a=clamp(I/S,0,1), once the net drives S down the derived albedo saturates at
        # the clamp ceiling (∂a/∂S→0) and the ONLY absolute anchor (albedo-L1) dies → runaway
        # S→floor, a→white, R→I (measured collapse at 15k when shadow_explain perturbed the gauge).
        # A small NON-aligned MSE on d_g vs π(gray(S_c)) pins absolute scale directly and closes the
        # gauge, independent of which other losses are on. Hypersim-gated. 0 = off. ~0.05–0.1.
        self.lambda_shade_abs = float(config.get('lambda_shade_abs', 0.0))

        # Reconstruction target mode:
        #   'full'    (V17 default) — recon_pred = A·S_d + R, target = I. R is an
        #             independent decoder output, so this is a genuine two-sided recon.
        #   'diffuse' (V18) — R is ANALYTIC, R = (I − A·S_d)₊, so the full-I recon
        #             collapses to recon_pred = max(A·S_d, I) and the loss to the
        #             one-sided relu(A·S_d − I). Supervise the two-sided DIFFUSE product
        #             A·S_d vs A_gt·S_d_gt instead (meaningful on both datasets:
        #             Hypersim → true diffuse render; InteriorVerse → I, since R≈0).
        #             See documents/design/V18_PGID_design.md §5.
        self.recon_mode = str(config.get('recon_mode', 'full'))

        # Per-branch MSG weights. Falls back to global lambda_msg for compat.
        _global_msg = float(config.get('lambda_msg', 0.5))
        self.lambda_msg_a = float(config.get('lambda_msg_albedo', _global_msg))
        self.lambda_msg_s = float(config.get('lambda_msg_shading', _global_msg))
        self.lambda_msg_r = float(config.get('lambda_msg_residual', _global_msg))
        self.lambda_msg_recon = float(config.get('lambda_msg_recon', _global_msg * 0.5))
        # Optional CRefNet-style RGF for albedo MSG: ignore tiny target gradients so
        # pseudo-GT shadow/noise ramps are not copied into albedo. 0.0 = original MSG.
        self.albedo_msg_target_grad_threshold = float(
            config.get('albedo_msg_target_grad_threshold', 0.0)
        )

        self.lambda_dssim = float(config.get('lambda_dssim', 0.2))
        self.lambda_edge = float(config.get('lambda_edge', 0.0))
        # Sparsity prior on the residual. Without it, the residual decoder (a free
        # 3-channel [-max,max] map) absorbs all reconstruction error into arbitrary
        # blobs, so A·S is never forced to explain the image. Applied only inside the
        # valid mask, where R_star is genuinely ~0 (true speculars excepted).
        self.lambda_res_sparse = float(config.get('lambda_res_sparse', 0.05))

        # Gradient decorrelation (Retinex prior). Penalises co-located albedo and
        # shading edges so a material edge cannot also appear in shading and a
        # lighting gradient cannot appear in albedo — the leakage that pure
        # reconstruction coupling between the parallel heads cannot prevent.
        # GT-free, so it regularises every dataset (incl. InteriorVerse/MID where
        # shading is unsupervised). 0 disables it.
        self.lambda_grad_decorr = float(config.get('lambda_grad_decorr', 0.0))

        # F-prior — texture-gated within-material albedo flatness (↓ IIW WHDR shading-ramp
        # leak). Penalise |∇A| ONLY at luminance-only (shadow) edges, PROTECT chroma (texture)
        # edges [PIE-Net chroma gate]. Applied on Hypersim (white-illuminant) samples only —
        # the chroma gate is unreliable on colored shadows (CCR limit), which L_inv handles.
        # See _albedo_flatness + documents/design/V17_shading_first_design.md.
        self.lambda_alb_flat = float(config.get('lambda_alb_flat', 0.0))
        self.flat_chroma_gate = float(config.get('flat_chroma_gate', 0.0))

        # F-prior — LOW-FREQUENCY variant (lever-1 homogeneity, 2026-06). The residual leak
        # MEASURED on real photos (window light-pool on a floor, warm lamp glow on a ceiling) is
        # a SLOW luminance ramp at ~constant chroma. _albedo_flatness_lf low-passes (avg-pool)
        # BOTH albedo and input BEFORE the gradient, so high-freq texture can never enter the
        # penalty (the safe fix for the naive full-res |∇A| that killed texture), and the PIE-Net
        # chroma gate self-backs-off on coloured boundaries → safe on EVERY dataset (loss_mask),
        # which is required because the leak lives on real MID/IIW pixels, not Hypersim. Resume
        # from the trained 50k ckpt → no gauge to collapse (cannot fail like V20).
        self.lambda_alb_flat_lf = float(config.get('lambda_alb_flat_lf', 0.0))
        self.alb_flat_lf_down = int(config.get('alb_flat_lf_down', 8))          # low-pass factor
        self.alb_flat_lf_chroma_gate = float(config.get('alb_flat_lf_chroma_gate', 2.0))

        # Refiner guards (off by default). These only fire for models that emit a_base.
        # delta: keeps the bounded log correction small. hf: preserves high-frequency detail
        # from the frozen base albedo while allowing low-frequency homogeneity correction.
        self.lambda_refiner_delta = float(config.get('lambda_refiner_delta', 0.0))
        self.lambda_refiner_hf = float(config.get('lambda_refiner_hf', 0.0))
        self.refiner_hf_down = int(config.get('refiner_hf_down', 8))

        # ── MICC (multi-illuminant) structured-illuminant regularisers ────────────
        # lambda_normal: predicted normals vs Hypersim GT (geometry-grounds the assignment).
        # lambda_assign_tv: spatial smoothness of the soft light-assignment W (illuminant is
        #   low-frequency). lambda_illum_prior: keep recovered illuminant colours c_k near the
        #   plausible (low-saturation) band so the palette stays physical, not arbitrary.
        self.lambda_normal = float(config.get('lambda_normal', 0.0))
        self.lambda_assign_tv = float(config.get('lambda_assign_tv', 0.0))
        self.lambda_illum_prior = float(config.get('lambda_illum_prior', 0.0))
        self.illum_sat_max = float(config.get('illum_sat_max', 0.35))
        # Anti-collapse: penalise low-entropy global light usage so W can't route all pixels to one k.
        self.lambda_assign_entropy = float(config.get('lambda_assign_entropy', 0.0))

        # ── CARI: cross-render supervision (the V19 contribution) ─────────────────
        # All three operate on paired MID frames (rgb1, rgb2 = same scene, different REAL
        # light), gated by m_invariant and masked by the per-pair HDR-valid mask. They add
        # the constraint single-image recon (A·S≈I) structurally cannot — see
        # documents/design/V19_CARI_design.md §4. The second forward pass (model(rgb2)) is run in
        # the train loop; these methods score it. Weights here are the single source of truth.
        #
        # L_inv: albedo of the same surface must be EQUAL across lightings (kills the free
        #   chroma-split direction → fixes desaturation + shadow-in-albedo). 0 disables.
        self.lambda_alb_invariance = float(config.get('lambda_alb_invariance', 0.0))
        # L_explain: the inter-frame image ratio I_a/I_b must be explained by the shading
        #   ratio S_a/S_b (log domain). Forces the lighting change INTO shading → fixes
        #   texture-in-shading where SSI is absent (InteriorVerse/MID). 0 disables.
        self.lambda_explain = float(config.get('lambda_explain', 0.0))
        # Shading sign/inequality prior (SAIL ℒ_reg): in the π-domain S_d=(1−π)/π≥0 always;
        #   the meaningful constraint is that ALBEDO carries the brightening — penalise
        #   shading that EXCEEDS the image (S_d such that A·S_d > I beyond residual), i.e.
        #   anything brightening must be albedo, not shading. Data-free, all datasets. 0 off.
        self.lambda_shade_sign = float(config.get('lambda_shade_sign', 0.0))

        # ── Shadow-invariance self-supervision (V20) ──────────────────────────────
        # Cast a KNOWN synthetic shadow M (+ tint c) on every image, run a 2nd forward, and
        # require albedo invariant to it (shadow_inv) + shading = S·M·c (shadow_explain, direct
        # KNOWN-target supervision). The mechanism that removes shadow+lighting from albedo on
        # real photos / OOD with no real pair. Scored in the train loop (src/train_v17.py). 0 off.
        self.lambda_shadow_inv = float(config.get('lambda_shadow_inv', 0.0))
        self.lambda_shadow_explain = float(config.get('lambda_shadow_explain', 0.0))
        # Which dataset ROWS the σ branch runs on (front3d-shadow-aug plan,
        # documents/design/FRONT3D_SHADOW_AUG_PLAN.md). Defaults preserve every existing
        # config's behaviour byte-for-byte (Hypersim-only, as before this option existed).
        self.shadow_on_hypersim = bool(config.get('shadow_on_hypersim', True))
        self.shadow_on_front3d = bool(config.get('shadow_on_front3d', False))
        # σ→GT-albedo anchor on front3d rows (anti-washout ingredient): unlike Hypersim's
        # self-supervised shadow_inv, front3d has an EXACT GT albedo under colored light, so
        # this term is L1-to-truth rather than L1-to-self-consistency — invariance alone can be
        # satisfied by flattening (the lever rows' MAW-intensity washout mechanism); L1-to-GT
        # cannot. Scored in train_one_step against targets['A_d_star'] on is_front3d rows only,
        # independent of shadow_on_front3d (that flag only gates the self-sup term above). 0 off.
        self.lambda_shadow_gt_front3d = float(config.get('lambda_shadow_gt_front3d', 0.0))
        # Curriculum: hold the shadow/sun aug OFF until the base decomposition is stable (Phase 2).
        # Early on, shadow_explain's target s_anchor·M·c uses a not-yet-formed shading → noise; and
        # the derive/palette/assignment are still settling. Hypersim GT shading-SSI already teaches
        # shadows→shading in Phase 1, so deferring loses only the OOD-robustness bonus (a Phase-2
        # concern anyway). 0 = on from ssi_warmup (old behaviour). Read in train_one_step.
        self.shadow_start_iter = int(config.get('shadow_start_iter', 0))
        # Derive-anneal ramp (residual-tautology fix): # of steps to anneal the Q99 derive scale
        # from 0.8/q99 (warmup) → identity (a=I/S, clean sparse R) after ssi_warmup. Read in
        # train_one_step → passed as model(..., derive_anneal=...). 0 ⇒ instant (anneal=1 always).
        self.derive_anneal_iters = int(config.get('derive_anneal_iters', 12000))
        # Cap the derive anneal below 1.0 so the derive never sits on the pure clamp(I/S,0,1) rail
        # (where ∂a/∂S=0 in the saturated region → the albedo-L1 anchor on S dies). Keeps a slight
        # Q99 compression so bright pixels stay off the hard ceiling and gradient to S survives.
        # Training-only (eval still uses anneal=1.0 in v20.forward default). 1.0 = no cap (legacy).
        self.derive_anneal_cap = float(config.get('derive_anneal_cap', 1.0))

        # Ordinal shading loss (CD-IID / Ordinal Shading): order-based shading supervision —
        # robust to scale, yields a cleaner shading → cleaner derived albedo. Hypersim-gated. 0 off.
        self.lambda_ordinal = float(config.get('lambda_ordinal', 0.0))

        # IIW WHDR ordinal hinge (CGIntrinsics/CRefNet) — pairwise human reflectance judgments.
        # Used ONLY by the post-baseline joint fine-tune (src/data/iiw_dataset.py); scored in
        # train_one_step on a dedicated IIW batch. 0 = off (the main Hypersim+MID run never uses it).
        self.lambda_ordinal_iiw = float(config.get('lambda_ordinal_iiw', 0.0))
        self.ordinal_margin = float(config.get('ordinal_margin', 0.10))  # WHDR δ (matches eval)

        # Derived-albedo (white-balance) supervision. Pulls the divided albedo
        # a_init=(I-R)/S_wb toward the GT albedo so the illuminant-chroma head learns
        # to remove the light color (the V12 white-balance mechanism). Only fires when
        # the model emits 'a_init' (use_derived_albedo). 0 disables it.
        self.lambda_albedo_init = float(config.get('lambda_albedo_init', 0.0))

        # DINOv2 material-consistency (GT-free). Fires only when the model emits
        # 'dino_tokens' (DINOv2 backbone) and the weight is > 0. Pushes same-material
        # albedo chroma to agree, using DINOv2's illumination-invariant similarity.
        self.lambda_mat_consist = float(config.get('lambda_mat_consist', 0.0))
        self.mat_sim_threshold = float(config.get('mat_sim_threshold', 0.0))
        # Relative weight of the DINOv2-gated log-luminance (intensity) flatness term inside
        # material_consistency_loss. >0 adds within-material intensity flatness on REAL photos
        # (the IIW-WHDR / ARAP si-RMSE shading-ramp lever) — chroma-only when 0. See that fn.
        self.mat_intensity_weight = float(config.get('mat_intensity_weight', 0.0))

        # Chroma-direction albedo loss: L1 between unit-normalised albedo vectors.
        # Makes hue/saturation errors expensive independently of per-channel magnitude —
        # directly prices the trade the optimizer currently makes (desaturate to minimize MSE).
        # Complementary to albedo MSE: MSE anchors absolute scale, chroma-L1 anchors hue.
        # 0 disables. Recommended weight: 0.1-0.3.
        self.lambda_a_chroma = float(config.get('lambda_a_chroma', 0.0))
        self.lambda_chroma_field = float(config.get('lambda_chroma_field', 0.0))
        # Colored-illuminant chroma target on the cross-render PAIR frame (V20 thesis lever,
        # 2026-06-28). chroma_field_gt (compute_targets) is the chroma of S_c on the WB PRIMARY
        # frame ⇒ ≈neutral on white Hypersim / WB MID ⇒ the chroma head is supervised neutral
        # ~100% of the time ⇒ desaturates albedo AND can never learn a colored illuminant (caps the
        # MID cast metric). This term supervises the chroma head on the PAIR frame (rgb2 = un-WB MID
        # or c·rgb Hypersim), whose S_c chroma carries the REAL colored illuminant — the only signal
        # that teaches colored constancy. Scored in train_one_step on pred2. 0 = off. ~0.3–0.4.
        self.lambda_chroma_field_pair = float(config.get('lambda_chroma_field_pair', 0.0))

        # Use L1 (not L2/MSE) for the main albedo data term. L1 is sharper-preserving —
        # it does not blur toward the conditional mean the way MSE does. The strong
        # feed-forward IID methods (IDT, CD-IID, CRefNet) all use L1 for albedo. With the
        # derive cascade albedo is already sharp; L1 keeps the data term from re-blurring it.
        self.albedo_l1 = bool(config.get('albedo_l1', False))

        self.msg_loss = MultiScaleGradientLoss(scales=4)
        
        self._dssim_win_size = 11
        self._dssim_sigma = 1.5
        self._dssim_win = None

    def _masked_mse(self, pred, target, mask):
        diff = F.mse_loss(pred, target, reduction='none')
        return (diff * mask).sum() / (mask.sum() + 1e-7)

    def _masked_l1(self, pred, target, mask):
        diff = F.l1_loss(pred, target, reduction='none')
        return (diff * mask).sum() / (mask.sum() + 1e-7)

    def _grad_decorr(self, a, s, mask, eps=1e-7):
        """Penalise co-located albedo/shading edges: mean(|∇A|·|∇S|) over valid px.

        |∇A| and |∇S| are per-axis mean-abs channel gradients. The product is large
        only where BOTH layers have an edge at the same pixel, so minimising it
        pushes the decomposition to assign each edge to exactly one layer (material
        → albedo, geometry/light → shading). Albedo's own MSE/MSG/DSSIM terms keep
        it from collapsing to a flat (gradient-free) trivial minimum.
        """
        # x-direction
        a_gx = (a[..., 1:] - a[..., :-1]).abs().mean(1, keepdim=True)
        s_gx = (s[..., 1:] - s[..., :-1]).abs().mean(1, keepdim=True)
        m_gx = mask[..., 1:] * mask[..., :-1]
        lx = ((a_gx * s_gx) * m_gx).sum() / (m_gx.sum() + eps)
        # y-direction
        a_gy = (a[:, :, 1:, :] - a[:, :, :-1, :]).abs().mean(1, keepdim=True)
        s_gy = (s[:, :, 1:, :] - s[:, :, :-1, :]).abs().mean(1, keepdim=True)
        m_gy = mask[:, :, 1:, :] * mask[:, :, :-1, :]
        ly = ((a_gy * s_gy) * m_gy).sum() / (m_gy.sum() + eps)
        return lx + ly

    def _albedo_flatness(self, a, rgb, mask, lambda_c=2.0, eps=1e-6):
        """F-prior: texture-gated within-material albedo flatness (↓ IIW WHDR shading leak).

        The defect (measured on IIW flat surfaces): a smooth shading RAMP leaks into albedo, so
        a single material is not homogeneous. WHDR is blind to it but it is the #1 perceptual
        'dirty albedo' issue. We penalise |∇A| ONLY where the INPUT gradient is luminance-only
        (a shadow/shading edge), and PROTECT it where the gradient is chromatic (a material /
        texture edge) — the PIE-Net (CVPR'22) chroma gate. The gate weight is computed from the
        INPUT (no backprop, no instability):
            w_shade = relu(|∇_lum I| − λc·|∇_chroma I|)
        so it fires on the shadow ramp and ~vanishes on tile grout / wood grain (keeping texture
        → not blurry). Albedo's own L1/MSG/DSSIM keep it off the flat (gradient-free) trivial
        minimum. CAVEAT: the chroma gate FAILS on COLORED shadows (CCR limit) → the caller gates
        this to white-illuminant (Hypersim) pixels; MID colored shadows are handled by L_inv."""
        lum = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        chroma = rgb / (rgb.sum(1, keepdim=True) + eps)        # brightness-normalised colour

        def gx(x):
            return (x[..., 1:] - x[..., :-1]).abs()

        def gy(x):
            return (x[:, :, 1:, :] - x[:, :, :-1, :]).abs()

        # x-direction: shading-edge weight from input, albedo gradient from prediction
        w_x = F.relu(gx(lum) - lambda_c * gx(chroma).sum(1, keepdim=True))
        a_gx = gx(a).mean(1, keepdim=True)
        m_x = mask[..., 1:] * mask[..., :-1]
        lx = ((w_x * a_gx) * m_x).sum() / (m_x.sum() + eps)
        # y-direction
        w_y = F.relu(gy(lum) - lambda_c * gy(chroma).sum(1, keepdim=True))
        a_gy = gy(a).mean(1, keepdim=True)
        m_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]
        ly = ((w_y * a_gy) * m_y).sum() / (m_y.sum() + eps)
        return lx + ly

    def _albedo_flatness_lf(self, a, rgb, mask, lambda_c=2.0, down=8, eps=1e-6):
        """LOW-FREQUENCY texture-gated within-material albedo flatness (lever-1 homogeneity).

        Targets the artifact MEASURED on real photos (IIW/MID): a residual LOW-FREQUENCY shading
        ramp — a window light-pool on a floor, a warm lamp glow on a ceiling — leaks into albedo,
        so one material is not homogeneous. We low-pass (avg-pool ×`down`) BOTH the predicted-
        albedo log-luminance and the input BEFORE taking gradients, so:
          • high-frequency texture (grout, grain, fabric) is averaged out by the pool and can
            NEVER enter the penalty — the safe fix for the naive full-res |∇A| that
            [[shading-first-s1-rejected]] showed kills texture;
          • only the slow ramp survives the pool → it is the only thing penalised.
        The penalty is the L1 albedo log-luminance gradient at the coarse scale, weighted by the
        PIE-Net shading-edge gate (detached, from the low-passed INPUT):
            w_shade = relu(|∇lum I| − λc·|∇chroma I|)
        which fires on a luminance-only ramp and ~vanishes where chroma also changes (a material
        edge OR a COLOURED light-pool — the CCR limit, left to L_inv). Because the gate self-
        backs-off on coloured boundaries, this is safe on EVERY dataset (caller passes loss_mask),
        which is required: the leak lives on real MID/IIW photos, not on Hypersim. log-domain →
        scale-invariant; resume-from-trained-ckpt → no free gauge to collapse (cannot fail like
        V20). See documents/design/V17_shading_first_design.md."""
        a_lum = (0.299 * a[:, 0:1] + 0.587 * a[:, 1:2] + 0.114 * a[:, 2:3]).clamp_min(eps)
        log_a_d = F.avg_pool2d(torch.log(a_lum), down)             # (B,1,h,w) low-freq albedo lum
        lum = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        chroma = rgb / (rgb.sum(1, keepdim=True) + eps)            # brightness-normalised colour
        lum_d = F.avg_pool2d(lum, down)
        chroma_d = F.avg_pool2d(chroma, down)
        mask_d = F.avg_pool2d(mask.float(), down)

        def gx(x):
            return x[..., 1:] - x[..., :-1]

        def gy(x):
            return x[:, :, 1:, :] - x[:, :, :-1, :]

        # shading-edge gate (detached, input-derived): luminance change NOT explained by chroma
        w_x = F.relu(gx(lum_d).abs() - lambda_c * gx(chroma_d).abs().sum(1, keepdim=True)).detach()
        w_y = F.relu(gy(lum_d).abs() - lambda_c * gy(chroma_d).abs().sum(1, keepdim=True)).detach()
        a_gx = gx(log_a_d).abs()                                   # penalise the slow albedo ramp
        a_gy = gy(log_a_d).abs()
        m_x = mask_d[..., 1:] * mask_d[..., :-1]
        m_y = mask_d[:, :, 1:, :] * mask_d[:, :, :-1, :]
        lx = ((w_x * a_gx) * m_x).sum() / (m_x.sum() + eps)
        ly = ((w_y * a_gy) * m_y).sum() / (m_y.sum() + eps)
        return lx + ly

    # ── MICC structured-illuminant regularisers ───────────────────────────────
    def _normal_loss(self, n_pred, n_gt, mask, eps=1e-6):
        """Angular loss (1 − cosine) on unit normals, masked to valid GT + Hypersim pixels.
        Grounds the trunk geometry so the light-assignment W follows surface orientation."""
        n_pred = F.normalize(n_pred, dim=1, eps=eps)
        n_gt = F.normalize(n_gt, dim=1, eps=eps)
        valid = (n_gt.norm(dim=1, keepdim=True) > 0.5).float() * mask   # drop zero/invalid GT normals
        cos = (n_pred * n_gt).sum(1, keepdim=True).clamp(-1.0, 1.0)
        return ((1.0 - cos) * valid).sum() / (valid.sum() + 1e-7)

    def _assign_tv(self, w, mask, eps=1e-7):
        """Total-variation smoothness on the soft light-assignment W (B,K,H,W) — the illuminant
        field is low-frequency, so penalise speckled assignment."""
        dx = (w[..., 1:] - w[..., :-1]).abs().mean(1, keepdim=True)
        dy = (w[:, :, 1:, :] - w[:, :, :-1, :]).abs().mean(1, keepdim=True)
        mx = mask[..., 1:] * mask[..., :-1]
        my = mask[:, :, 1:, :] * mask[:, :, :-1, :]
        return ((dx * mx).sum() / (mx.sum() + eps)) + ((dy * my).sum() / (my.sum() + eps))

    def _illum_prior(self, c_k):
        """Keep recovered illuminant colours c_k (B,K,3, unit-luminance) in a plausible band:
        penalise per-channel saturation (max−min) beyond `illum_sat_max`. Real illuminants are
        only mildly chromatic; this stops the palette drifting to arbitrary saturated colours."""
        sat = c_k.max(dim=-1).values - c_k.min(dim=-1).values   # (B,K)
        return F.relu(sat - self.illum_sat_max).mean()

    def _assign_entropy(self, w, mask, eps=1e-7):
        """Anti-collapse on the soft assignment W (B,K,H,W). Penalise the NEGATIVE entropy of the
        spatially-averaged per-light usage p_k = mean_x W_k: a single dominant light gives a near
        one-hot p (entropy→0, penalty max), so minimising this spreads usage across illuminants and
        breaks the 'one light everywhere' collapse. Mild — assign_tv + data terms still let W
        specialise spatially; this only forbids total collapse onto a single (e.g. green) light."""
        wsum = (w * mask).sum(dim=(2, 3))                       # (B,K) masked usage
        p = wsum / (wsum.sum(dim=1, keepdim=True) + eps)        # (B,K) usage distribution
        neg_entropy = (p * (p + eps).log()).sum(dim=1)          # (B,) = −H(p) ∈ [−log K, 0]
        return neg_entropy.mean()

    # ── CARI cross-render losses (V19) ────────────────────────────────────────
    def cari_albedo_invariance(self, a1, a2, mask):
        """L_inv: |A(rgb1) − A(rgb2)| over valid paired pixels. Same surface, different
        light ⇒ same albedo. Masked by loss_mask · m_invariant · pair_valid."""
        return self._masked_l1(a1, a2, mask)

    def cari_chroma_field(self, chroma_pred2, rgb2, a_star, mask, eps=1e-6):
        """Colored-illuminant chroma target on the cross-render PAIR frame (V20 thesis lever).

        The pair frame rgb2 carries the colored illuminant (un-WB MID, or c·rgb Hypersim) that the
        WB primary frame lacks. Its colored diffuse-shading chroma is chroma(S_c2)=chroma(rgb2/A*)
        — the SAME albedo A* as the primary (same scene), so material cancels and what remains is
        the illuminant colour. Supervising the chroma head (on the rgb2 forward) toward it teaches
        the head to emit the real cast → the derive a=I/(g·c) then divides that cast OUT of albedo
        (the structural constancy the neutral WB target can never provide). Unit-vector L1, masked
        by the per-pair HDR-valid CARI mask."""
        s_c2 = rgb2 / a_star.clamp_min(1e-4)
        c_gt = F.normalize(s_c2.clamp_min(eps), dim=1)
        c_pred = F.normalize(chroma_pred2.clamp_min(eps), dim=1)
        return self._masked_l1(c_pred, c_gt, mask)

    def cari_explain(self, rgb1, rgb2, s1, s2, mask, eps=1e-3):
        """L_explain: the photometric change between frames must be explained by the
        shading change. In log domain (robust, scale-free):
            ‖ log(L(I1)/L(I2)) − log(L(S1)/L(S2)) ‖₁
        on luminance (achromatic ratio — colored-light handled by the 3-ch shading itself).
        Material/texture cancels in I1/I2 (verified), so this forces the lighting ratio into
        shading and out of albedo. s1,s2 are linear shading (shading_linear)."""
        def lum(x):
            return (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]).clamp_min(0.0)
        ri = torch.log(lum(rgb1) + eps) - torch.log(lum(rgb2) + eps)
        rs = torch.log(lum(s1) + eps) - torch.log(lum(s2) + eps)
        return self._masked_l1(ri, rs, mask)

    # ── Shadow-invariance self-supervision (V20 — the shadow/lighting removal) ─────
    def shadow_invariance(self, a_clean, a_shadow, mask):
        """L_shadow_inv: albedo must be INVARIANT to a synthetic cast shadow / illuminant tint.
        a_clean is detached by the caller (the anchor) so the gradient pulls the SHADOWED branch's
        albedo back onto it → the shadow/light cannot live in albedo.

        Scale+shift align a_shadow→a_clean per image FIRST (expert review, training-dynamics): the
        per-image Q99 renorm in derive_albedo scales the two branches differently when M is not yet
        absorbed, and a plain L1 could then be cheaply driven down by GLOBAL contrast-collapse
        (desaturation) instead of shadow removal. Aligning removes that gauge freedom — only the
        residual SPATIAL (shadow-shaped) difference is penalised, which is what we want."""
        a_shadow = _scale_shift_align(a_shadow, a_clean, mask)
        return self._masked_l1(a_shadow, a_clean, mask)

    def _ordinal_shading(self, s_pred, s_gt, mask, n_pairs=4096, margin=0.0, eps=1e-6):
        """Ordinal shading loss (Careaga & Aksoy, Ordinal Shading TOG'23 / CD-IID). Absolute
        shading is scale-ambiguous but its ORDER is robust, so penalising pairs whose PREDICTED
        order disagrees with GT order is a strong, outlier-tolerant supervision → a cleaner shading
        → a cleaner DERIVED albedo (shadows go to shading, not albedo). Luminance of linear shading,
        random valid pixel pairs, log-domain difference. Caller gates to Hypersim (GT shading)."""
        B = s_pred.shape[0]
        def lum(x):
            return (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]) if x.shape[1] == 3 else x
        lp = lum(s_pred).reshape(B, -1)
        lg = lum(s_gt).reshape(B, -1)
        m = mask.reshape(B, -1)
        N = lp.shape[1]
        i = torch.randint(0, N, (n_pairs,), device=lp.device)
        j = torch.randint(0, N, (n_pairs,), device=lp.device)
        pair_mask = m[:, i] * m[:, j]                                      # (B, n_pairs) both valid
        dp = torch.log(lp[:, i].clamp_min(eps)) - torch.log(lp[:, j].clamp_min(eps))
        dg = torch.log(lg[:, i].clamp_min(eps)) - torch.log(lg[:, j].clamp_min(eps))
        loss = F.relu(margin - torch.sign(dg) * dp)                        # penalise wrong order
        return (loss * pair_mask).sum() / (pair_mask.sum() + eps)

    def shadow_explain(self, s_clean, s_shadow, M, c, mask, eps=1e-4):
        """L_shadow_explain: the shadowed shading must equal clean-shading × M × c — a KNOWN
        target (we cast M, c ourselves), so this is DIRECT supervision of the (even sharp,
        achromatic) shadow INTO shading — the signal the gray regime otherwise lacks. Log-domain
        (scale-robust). s_* are linear shading (shading_linear); s_clean detached by the caller."""
        target = (s_clean * M * c).clamp_min(0.0)
        return self._masked_l1(torch.log(s_shadow.clamp_min(eps)),
                               torch.log(target.clamp_min(eps)), mask)

    def cari_shade_sign(self, a, s_linear, rgb, mask, eps=1e-4):
        """Shading sign/inequality prior (SAIL ℒ_reg, adapted to A·S physics):
        penalise the part of diffuse shading that makes A·S_d OVERSHOOT the image, i.e.
        relu(A·S_d − I). The analytic residual already clamps R=(I−A·S)₊≥0, so overshoot is
        un-modelled energy that must come from albedo, not shading ⇒ 'anything brightening is
        albedo'. Pushes texture/brightening out of shading. Per-channel, masked."""
        overshoot = F.relu(a * s_linear - rgb)
        return (overshoot * mask).sum() / (mask.sum() + 1e-7)

    def ordinal_hinge_loss(self, albedo, comparisons_list, margin_delta=None, eps=1e-6):
        """IIW WHDR ordinal hinge (CGIntrinsics / CRefNet). Matches the human pairwise
        REFLECTANCE ordering that WHDR scores. On predicted albedo luminance R, log-domain,
        margin ξ = log(1+δ) (δ = WHDR threshold, default 0.10 to match the eval):
            label 0 (equal)     → w·(logR₁ − logR₂)²
            label 1 (p1 darker) → w·relu(logR₁ − logR₂ + ξ)²   (want R₁ < R₂)
            label 2 (p2 darker) → w·relu(logR₂ − logR₁ + ξ)²
        It is a HINGE: zero once the correct ordering holds with margin, so it constrains
        relative ordering only — never absolute albedo magnitude (safe alongside the other
        albedo terms). Points are sampled with bilinear grid_sample (differentiable).

        albedo: (B,3,H,W) in [0,1]. comparisons_list: list of B tensors (K_b,6) =
        [x1,y1,x2,y2,label,weight] with x,y normalised in [0,1]. Returns the weighted-mean
        hinge over all pairs in the batch (0 if there are none)."""
        delta = self.ordinal_margin if margin_delta is None else float(margin_delta)
        xi = math.log(1.0 + delta)
        device = albedo.device
        lum = (0.299 * albedo[:, 0:1] + 0.587 * albedo[:, 1:2]
               + 0.114 * albedo[:, 2:3]).clamp_min(eps)
        log_lum = torch.log(lum)                                   # (B,1,H,W)
        total = log_lum.new_zeros(())
        wsum = log_lum.new_zeros(())
        for b in range(albedo.shape[0]):
            comps = comparisons_list[b]
            if comps is None or comps.numel() == 0:
                continue
            comps = comps.to(device)
            label, w = comps[:, 4], comps[:, 5]

            def _sample(xy):
                # grid_sample grid is (x=width, y=height) in [-1,1]; IIW (x,y) are
                # (col,row) fractions → direct map. align_corners=True matches int-index eval.
                g = (xy * 2.0 - 1.0).view(1, 1, -1, 2)
                s = F.grid_sample(log_lum[b:b + 1], g, mode='bilinear',
                                  align_corners=True, padding_mode='border')
                return s.reshape(-1)

            lr1 = _sample(comps[:, 0:2])
            lr2 = _sample(comps[:, 2:4])
            d = lr1 - lr2
            loss_eq = d * d
            loss_1 = F.relu(d + xi) ** 2
            loss_2 = F.relu(-d + xi) ** 2
            per = torch.where(label == 0, loss_eq,
                              torch.where(label == 1, loss_1, loss_2))
            total = total + (w * per).sum()
            wsum = wsum + w.sum()
        return total / (wsum + eps)

    def _get_dssim_window(self, channels, device, dtype):
        if (self._dssim_win is not None
                and self._dssim_win.shape[0] == channels
                and self._dssim_win.device == device
                and self._dssim_win.dtype == dtype):
            return self._dssim_win

        win_size = self._dssim_win_size
        sigma = self._dssim_sigma
        coords = torch.arange(win_size, dtype=dtype, device=device) - win_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window = g.unsqueeze(1) @ g.unsqueeze(0)
        window = window.unsqueeze(0).unsqueeze(0).expand(channels, 1, -1, -1).contiguous()
        self._dssim_win = window
        return window

    def _compute_dssim(self, pred, target, mask):
        C = pred.shape[1]
        win = self._get_dssim_window(C, pred.device, pred.dtype)
        pad = self._dssim_win_size // 2

        mu_x = F.conv2d(pred, win, padding=pad, groups=C)
        mu_y = F.conv2d(target, win, padding=pad, groups=C)

        mu_x_sq = mu_x * mu_x
        mu_y_sq = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x_sq = (F.conv2d(pred * pred, win, padding=pad, groups=C) - mu_x_sq).clamp_min(0)
        sigma_y_sq = (F.conv2d(target * target, win, padding=pad, groups=C) - mu_y_sq).clamp_min(0)
        sigma_xy = F.conv2d(pred * target, win, padding=pad, groups=C) - mu_xy

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
                   ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))

        dssim_map = (1.0 - ssim_map) / 2.0
        return (dssim_map * mask).sum() / (mask.sum() + 1e-7)



    def forward(self, predictions, targets, loss_mask, m_diffuse, m_residual, rgb,
                use_ssi=True, decorr_scale=1.0):
        """
        predictions: dict with 'a_d', 'shading', 'residual', 'rgb_reconstructed'
        targets: dict with 'A_d_star', 'S_d_star', 'pi_star', 'R_star'
        loss_mask: (B, 1, H, W) float mask of valid pixels
        m_diffuse: (B,) float tensor gating the shading loss per-sample (1.0 = Hypersim only)
        m_residual: (B,) float tensor gating the residual loss per-sample
        rgb: (B, 3, H, W) input image
        use_ssi: bool, if False, falls back to plain MSE for shading (warmup)
        decorr_scale: float in [0,1] ramping the gradient-decorrelation term in after
            the decomposition has formed (it is heavy and would fight early learning).
        """
        zero = torch.tensor(0.0, device=loss_mask.device)
        details = {}

        # Saturated-pixel guard: R_star = rgb - A·S is meaningless where rgb is clipped
        # to [0,1] (A·S is unclipped), which blows R up near windows/lights and made the
        # residual dominate the objective. Exclude those pixels from the R and recon
        # targets; the R sparsity prior still keeps R≈0 there.
        sat_ok = (rgb.amax(dim=1, keepdim=True) < 0.99).float()

        # 1. Albedo (L1 or MSE data term + MSG p=1 + DSSIM)
        a_pred, a_gt = predictions['a_d'], targets['A_d_star']
        l_a_mse = (self._masked_l1(a_pred, a_gt, loss_mask) if self.albedo_l1
                   else self._masked_mse(a_pred, a_gt, loss_mask))
        l_a_msg = (
            self.msg_loss(
                a_pred, a_gt, mask=loss_mask, p=1,
                target_grad_threshold=self.albedo_msg_target_grad_threshold,
            ) if self.lambda_msg_a > 0 else zero
        )
        l_a_dssim = self._compute_dssim(a_pred, a_gt, loss_mask) if self.lambda_dssim > 0 else zero

        # 1c. Chroma-direction L1: normalised albedo vector alignment. Makes hue/saturation
        # errors expensive independently of brightness — prevents desaturation-as-cheap-MSE.
        if self.lambda_a_chroma > 0:
            a_norm = F.normalize(a_pred.clamp(1e-6), dim=1)   # unit chroma vector
            g_norm = F.normalize(a_gt.clamp(1e-6), dim=1)
            l_a_chroma = self._masked_l1(a_norm, g_norm, loss_mask)
        else:
            l_a_chroma = zero
        details['loss_c_chroma'] = l_a_chroma.detach()

        la = (l_a_mse + self.lambda_msg_a * l_a_msg + self.lambda_dssim * l_a_dssim
              + self.lambda_a_chroma * l_a_chroma)
        details.update({'loss_c_l1': l_a_mse.detach(), 'loss_c_msg': l_a_msg.detach(), 'loss_c_dssim': l_a_dssim.detach()})

        # 1b. Derived-albedo white-balance supervision: a_init=(I-R)/S_wb should match GT
        # albedo. This is the gradient that teaches the chroma head to divide the light
        # color out (so the raw albedo is neutral before JointRefine, like V12's a_c).
        if 'a_init' in predictions and self.lambda_albedo_init > 0:
            l_a_init = self._masked_l1(predictions['a_init'], a_gt, loss_mask)
        else:
            l_a_init = zero
        details['loss_albedo_init'] = l_a_init.detach()

        if predictions.get('a_base') is not None:
            a_base = predictions['a_base'].detach().clamp_min(1e-4)
            log_a = torch.log(a_pred.clamp_min(1e-4))
            log_b = torch.log(a_base)
            if self.lambda_refiner_delta > 0:
                l_ref_delta = self._masked_l1(log_a, log_b, loss_mask)
            else:
                l_ref_delta = zero
            if self.lambda_refiner_hf > 0:
                down = max(1, self.refiner_hf_down)
                h = max(1, log_a.shape[-2] // down)
                w = max(1, log_a.shape[-1] // down)
                a_low = F.interpolate(F.interpolate(log_a, size=(h, w), mode='area'),
                                      size=log_a.shape[-2:], mode='bilinear', align_corners=False)
                b_low = F.interpolate(F.interpolate(log_b, size=(h, w), mode='area'),
                                      size=log_b.shape[-2:], mode='bilinear', align_corners=False)
                l_ref_hf = self._masked_l1(log_a - a_low, log_b - b_low, loss_mask)
            else:
                l_ref_hf = zero
        else:
            l_ref_delta = l_ref_hf = zero
        details['loss_refiner_delta'] = l_ref_delta.detach()
        details['loss_refiner_hf'] = l_ref_hf.detach()

        # 2. Shading — inverse domain π=1/(S+1); gated by m_diffuse (Hypersim only). Bounded [0,1].
        # ACHROMATIC SSI on the GRAY factor g vs gray of the COLORFUL shading S_c = I/A_GT (NOT the
        # diffuse shading S_d). This is mandatory for the derive: a=I/S targets A_GT only if S→S_c=I/A_GT
        # (V16 derives D_g_star/xi_star from S_c for the same reason). Targeting S_d (direct-illum only,
        # no indirect/specular) would make a=I/S_d≠A_GT wherever there's bounce light. g is full-res and
        # CAN match GT; chroma is then implicitly anchored (chroma=S/g, with albedo-L1 pinning a → S=I/a
        # → chroma=chroma(S_c)). Falls back to colored-S SSI for models without 'd_g' (V17/V18).
        if predictions.get('d_g') is not None and targets.get('S_c_star') is not None:
            sc = targets['S_c_star']                    # COLORFUL shading = I/A_GT (derive-consistent)
            lum_gt = 0.299 * sc[:, 0:1] + 0.587 * sc[:, 1:2] + 0.114 * sc[:, 2:3]
            s_pred = predictions['d_g']                 # (B,1,H,W) π-domain gray shading
            s_gt = 1.0 / (lum_gt + 1.0)                 # π of gray(S_c) (matches the d_g formula)
        else:
            s_pred, s_gt = predictions['shading'], targets['pi_star']
        m_diff_mask = loss_mask * m_diffuse.view(-1, 1, 1, 1)
        if m_diff_mask.sum() > 0:
            s_aligned = _scale_shift_align(s_pred, s_gt, m_diff_mask) if use_ssi else s_pred
            l_s_mse = self._masked_mse(s_aligned, s_gt, m_diff_mask)
            l_s_msg = self.msg_loss(s_aligned, s_gt, mask=m_diff_mask, p=2) if self.lambda_msg_s > 0 else zero
            ls = l_s_mse + self.lambda_msg_s * l_s_msg
            # Absolute scale pin: NON-aligned MSE on the raw d_g vs π(gray(S_c)) closes the free
            # shading-scale gauge the scale-shift SSI leaves open (anti-collapse). Low weight so the
            # aligned SSI still owns the shape; this only nails the absolute level.
            if self.lambda_shade_abs > 0:
                l_s_abs = self._masked_mse(s_pred, s_gt, m_diff_mask)
                ls = ls + self.lambda_shade_abs * l_s_abs
            else:
                l_s_abs = zero
        else:
            ls = l_s_mse = l_s_msg = l_s_abs = zero
        details.update({'loss_shading_mse': l_s_mse.detach(), 'loss_shading_msg': l_s_msg.detach(),
                        'loss_shading_abs': l_s_abs.detach()})

        # 2b. Ordinal shading (CD-IID): order-based shading supervision (robust → cleaner derived
        # albedo). Uses linear shading vs GT, Hypersim-gated (needs GT shading). 0 disables.
        if (self.lambda_ordinal > 0 and predictions.get('shading_linear') is not None
                and targets.get('S_d_star') is not None and m_diff_mask.sum() > 0):
            l_ordinal = self._ordinal_shading(predictions['shading_linear'], targets['S_d_star'], m_diff_mask)
        else:
            l_ordinal = zero
        details['loss_ordinal'] = l_ordinal.detach()

        # 3. Residual (L1 + MSG p=1), supervised only where rgb is not clipped AND the
        # pixel is loss-valid. loss_mask matters on MID: its specular gate keeps the
        # R*=0 closure from pulling the non-clipped specular sheen into A·S (= into S,
        # since A is pinned by the specular-free pseudo-GT).
        r_pred, r_gt = predictions['residual'], targets['R_star']
        m_res_mask = (m_residual.view(-1, 1, 1, 1) * sat_ok * loss_mask).expand_as(r_pred)

        if m_res_mask.sum() > 0:
            l_r_l1 = self._masked_l1(r_pred, r_gt, m_res_mask)
            l_r_msg = self.msg_loss(r_pred, r_gt, mask=m_res_mask, p=1) if self.lambda_msg_r > 0 else zero
            # Sparsity: push |R|→0 inside the valid mask so the diffuse model (A·S)
            # must reconstruct the image instead of the residual blobbing over it.
            # Masked to loss_mask so it never fights R_star=rgb in invalid regions.
            l_r_sparse = self._masked_l1(r_pred, torch.zeros_like(r_pred), loss_mask) if self.lambda_res_sparse > 0 else zero
            lr = l_r_l1 + self.lambda_msg_r * l_r_msg + self.lambda_res_sparse * l_r_sparse
        else:
            lr, l_r_l1, l_r_msg, l_r_sparse = zero, zero, zero, zero

        details.update({
            'loss_residual_mse': l_r_l1.detach(),
            'loss_residual_msg': l_r_msg.detach(),
            'loss_residual_sparse': l_r_sparse.detach(),
        })

        # 4. Reconstruction (L1 + MSG p=1), excluding clipped pixels (same guard as R).
        # 'diffuse' mode (V18 analytic residual): full-I recon is one-sided because
        # R=(I−A·S_d)₊ makes recon_pred=max(A·S_d,I); supervise the two-sided diffuse
        # product A·S_d vs A_gt·S_d_gt instead.
        if self.recon_mode == 'diffuse':
            # use the physics-consistent derived albedo for recon when present (V20 refinement);
            # falls back to 'a_d' for V17/V19 which have no separate 'a_derived'.
            recon_pred = predictions.get('a_derived', predictions['a_d']) * predictions['shading_linear']
            recon_target = targets['A_d_star'] * targets['S_d_star']
            # HYPERSIM-GATED: this two-sided anchor (a·S vs A*·S_d) is the anti-collapse WALL that
            # pins the a·S product off zero. It is only meaningful where S_d is REAL (Hypersim); on
            # MID S_d_star=S_c ⇒ A*·S_d=I ⇒ the tautology that the gating exists to avoid. m_diffuse
            # selects Hypersim rows.
            recon_mask = sat_ok * loss_mask * m_diffuse.view(-1, 1, 1, 1)
        else:
            recon_pred = predictions['rgb_reconstructed']
            recon_target = rgb
            # Same guard as R: clipped pixels out, and loss_mask in — on MID the recon target
            # is exactly I (A*·(I/A*)), so without the specular gate in loss_mask this term
            # would teach "shading explains the specular sheen" wherever the sheen isn't clipped.
            recon_mask = sat_ok * loss_mask
        l_rec_l1 = self._masked_l1(recon_pred, recon_target, recon_mask)
        l_rec_msg = self.msg_loss(recon_pred, recon_target, mask=recon_mask, p=1) if self.lambda_msg_recon > 0 else zero

        l_recon = l_rec_l1 + self.lambda_msg_recon * l_rec_msg
        details.update({'loss_recon_l1': l_rec_l1.detach(), 'loss_recon_msg': l_rec_msg.detach()})

        # 5. CCR edge auxiliary loss — direct gradient to CCR encoder past SPADE zero-init
        # Active on all datasets (albedo GT always present); no m_diffuse gating needed
        if 'ccr_edge_pred' in predictions and self.lambda_edge > 0:
            edge_gt = targets['ccr_edge_gt']
            edge_pred = torch.sigmoid(predictions['ccr_edge_pred'])
            l_edge = self._masked_mse(edge_pred, edge_gt, loss_mask)
            details['loss_edge'] = l_edge.detach()
        else:
            l_edge = zero
            details['loss_edge'] = zero

        # 6. Gradient decorrelation (Retinex prior) — couples the parallel heads in
        # loss space. GT-free; applied on every dataset inside the valid mask. Ramped
        # in via decorr_scale once the decomposition exists (it is heavily weighted).
        if self.lambda_grad_decorr > 0 and decorr_scale > 0:
            l_decorr = self._grad_decorr(a_pred, s_pred, loss_mask)
        else:
            l_decorr = zero
        details['loss_grad_decorr'] = l_decorr.detach()

        # 6b. F-prior — texture-gated within-material albedo flatness (↓ IIW WHDR shading-ramp
        # leak; the #1 'dirty albedo' look). Gated by m_diff_mask (Hypersim, white-illuminant)
        # so the PIE-Net chroma gate only runs where it is safe; MID colored shadows → L_inv.
        # Ramped via decorr_scale, like the other regional priors (heavy early, fights learning).
        if self.lambda_alb_flat > 0 and decorr_scale > 0 and m_diff_mask.sum() > 0:
            l_alb_flat = self._albedo_flatness(a_pred, rgb, m_diff_mask, lambda_c=self.flat_chroma_gate)
        else:
            l_alb_flat = zero
        details['loss_alb_flat'] = l_alb_flat.detach()

        # 6b-LF. Low-frequency F-prior (lever-1 homogeneity) — shrink the SLOW albedo-luminance
        # ramp (residual light-pool / soft-shadow leak) on EVERY valid pixel (loss_mask, incl.
        # real MID): the low-pass keeps texture safe and the chroma gate keeps coloured edges
        # safe. NO decorr ramp — we fine-tune from a trained ckpt, not from scratch.
        if self.lambda_alb_flat_lf > 0 and loss_mask.sum() > 0:
            l_alb_flat_lf = self._albedo_flatness_lf(
                a_pred, rgb, loss_mask,
                lambda_c=self.alb_flat_lf_chroma_gate, down=self.alb_flat_lf_down)
        else:
            l_alb_flat_lf = zero
        details['loss_alb_flat_lf'] = l_alb_flat_lf.detach()

        # 6c. MICC structured-illuminant terms (only when the model emits the keys).
        # Normal aux: predicted normals vs Hypersim GT (Hypersim-gated via m_diff_mask).
        if (self.lambda_normal > 0 and predictions.get('normals_pred') is not None
                and targets.get('normals') is not None and m_diff_mask.sum() > 0):
            l_normal = self._normal_loss(predictions['normals_pred'], targets['normals'], m_diff_mask)
        else:
            l_normal = zero
        details['loss_normal'] = l_normal.detach()
        # Assignment smoothness (TV) — GT-free, all datasets.
        if self.lambda_assign_tv > 0 and predictions.get('assignment') is not None:
            l_assign_tv = self._assign_tv(predictions['assignment'], loss_mask)
        else:
            l_assign_tv = zero
        details['loss_assign_tv'] = l_assign_tv.detach()
        # Illuminant-colour plausibility prior — GT-free.
        if self.lambda_illum_prior > 0 and predictions.get('illuminants') is not None:
            l_illum_prior = self._illum_prior(predictions['illuminants'])
        else:
            l_illum_prior = zero
        details['loss_illum_prior'] = l_illum_prior.detach()
        # Assignment anti-collapse entropy — GT-free, all datasets.
        if self.lambda_assign_entropy > 0 and predictions.get('assignment') is not None:
            l_assign_entropy = self._assign_entropy(predictions['assignment'], loss_mask)
        else:
            l_assign_entropy = zero
        details['loss_assign_entropy'] = l_assign_entropy.detach()

        # 7. DINOv2 material consistency (GT-free) — only with a DINOv2 backbone
        # (predictions carry 'dino_tokens'). Active on every dataset.
        if (self.lambda_mat_consist > 0
                and predictions.get('dino_tokens') is not None
                and predictions.get('dino_patch_hw') is not None):
            l_mat = material_consistency_loss(
                a_pred, predictions['dino_tokens'], predictions['dino_patch_hw'],
                threshold=self.mat_sim_threshold,
                intensity_weight=self.mat_intensity_weight,
            )
        else:
            l_mat = zero
        details['loss_mat_consist'] = l_mat.detach()

        # 8. CARI shading sign/inequality prior (data-free, all datasets) — penalise
        # diffuse-shading overshoot relu(A·S_d − I). "Anything brightening is albedo."
        if self.lambda_shade_sign > 0:
            l_shade_sign = self.cari_shade_sign(
                predictions['a_d'], predictions['shading_linear'], rgb, loss_mask)
        else:
            l_shade_sign = zero
        details['loss_shade_sign'] = l_shade_sign.detach()

        # 9. Chroma field supervision (V20 explicit color field)
        if self.lambda_chroma_field > 0 and 'chroma_field' in predictions and 'chroma_field_gt' in targets:
            c_pred = F.normalize(predictions['chroma_field'].clamp(1e-6), dim=1)
            c_gt = F.normalize(targets['chroma_field_gt'].clamp(1e-6), dim=1)
            l_chroma_field = self._masked_l1(c_pred, c_gt, loss_mask)
        else:
            l_chroma_field = zero
        details['loss_chroma_field'] = l_chroma_field.detach()

        # Weighting
        la_w = self.w_a * la
        ls_w = self.w_s * ls
        lr_w = self.w_r * lr
        l_recon_w = self.w_recon * l_recon
        l_edge_w = self.lambda_edge * l_edge
        l_decorr_w = self.lambda_grad_decorr * float(decorr_scale) * l_decorr
        l_alb_flat_w = self.lambda_alb_flat * float(decorr_scale) * l_alb_flat
        l_alb_flat_lf_w = self.lambda_alb_flat_lf * l_alb_flat_lf
        l_a_init_w = self.lambda_albedo_init * l_a_init
        l_mat_w = self.lambda_mat_consist * l_mat
        l_shade_sign_w = self.lambda_shade_sign * l_shade_sign
        l_normal_w = self.lambda_normal * l_normal
        l_assign_tv_w = self.lambda_assign_tv * float(decorr_scale) * l_assign_tv
        l_illum_prior_w = self.lambda_illum_prior * l_illum_prior
        l_assign_entropy_w = self.lambda_assign_entropy * float(decorr_scale) * l_assign_entropy
        l_ordinal_w = self.lambda_ordinal * l_ordinal
        l_chroma_field_w = self.lambda_chroma_field * l_chroma_field
        l_ref_delta_w = self.lambda_refiner_delta * l_ref_delta
        l_ref_hf_w = self.lambda_refiner_hf * l_ref_hf

        loss_total = (la_w + ls_w + lr_w + l_recon_w + l_edge_w + l_decorr_w + l_alb_flat_w
                      + l_alb_flat_lf_w
                      + l_a_init_w + l_mat_w + l_shade_sign_w
                      + l_normal_w + l_assign_tv_w + l_illum_prior_w + l_assign_entropy_w
                      + l_ordinal_w + l_chroma_field_w + l_ref_delta_w + l_ref_hf_w)

        out = {
            'loss_total': loss_total,
            'loss_a': la_w,
            'loss_s': ls_w,
            'loss_r': lr_w,
            'loss_recon': l_recon_w,
            'loss_edge': l_edge_w,
            'loss_grad_decorr': l_decorr_w,
            'loss_alb_flat': l_alb_flat_w,
            'loss_alb_flat_lf': l_alb_flat_lf_w,
            'loss_albedo_init': l_a_init_w,
            'loss_mat_consist': l_mat_w,
            'loss_shade_sign': l_shade_sign_w,
            'loss_normal': l_normal_w,
            'loss_assign_tv': l_assign_tv_w,
            'loss_illum_prior': l_illum_prior_w,
            'loss_ordinal': l_ordinal_w,
            'loss_chroma_field': l_chroma_field_w,
            'loss_refiner_delta': l_ref_delta_w,
            'loss_refiner_hf': l_ref_hf_w,
            'loss_c_chroma': self.lambda_a_chroma * l_a_chroma,
        }
        out.update(details)
        return out
