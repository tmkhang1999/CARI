# Why IIW good / ARAP bad, and the directional-shadow-loss survey (2026-06-16)

> **Honesty note:** during this analysis I twice floated a WRONG hypothesis (that the ARAP
> corridor floor clips to white after tonemap). Direct measurement of the SAVED prediction
> refuted it. This doc records the measured truth, not the discarded guesses.

## Measured facts (from the actual saved corridor albedo, not re-derived)

- ARAP corridor floor is **well-exposed** after the p90 training tonemap: displayed-input floor
  p10/p50/p90 = 123/207/238 (8-bit). It is NOT blown white. (My earlier "100% white" was a
  buggy re-tonemap of an already-clipped array — disregard it.)
- **The shadow leaks, and it's real:** input floor luminance std = 64.6; predicted-albedo floor
  std = **64.5**. The cast-shadow band has the SAME contrast in albedo as in the input → the
  model copies the floor shadow straight into albedo on this scene. Verified, not eyeballed.
- IIW (real LDR photos): the model removes floor/wall shadows well and keeps colour
  (12-pair contact sheet: red sofas stay red, blue curtains blue, wood tones held). Strong.

## So why IIW good, ARAP corridor bad? The honest differential

It is **NOT** input clipping (floor is well-exposed). The real differences, in order of likely impact:

1. **Shadow HARDNESS / contrast type.** IIW indoor photos = diffuse ambient light, SOFT shadows.
   ARAP corridor = strong directional light through side windows → HARD, high-frequency cast
   shadow bands with sharp edges on a flat textured floor. The frozen-DINOv2 "shadow=lit patch
   map to similar tokens" invariance is strong for SOFT/object shadows (works on IIW) but a hard
   shadow EDGE looks like a real material/geometry edge to the patch tokens → it survives into
   albedo. This is the genuine limitation, and it is shadow-TYPE specific, not domain-generic.
2. **HDR vs LDR domain.** ARAP is synthetic HDR tonemapped; IIW is native LDR. Even well-exposed,
   the tonemapped HDR floor has a different noise/contrast micro-structure than a real photo.
3. **Synthetic render look.** ARAP's CGI surfaces (uniform tiles, sharp shadow boundaries) differ
   from real-photo statistics the frozen DINOv2 backbone was trained on.

**Conclusion:** the architecture is NOT broken (IIW proves it decomposes shadows + keeps colour).
The corridor failure is a **hard directional cast-shadow on a flat bright floor** — the single
hardest case, where the patch-token shadow-invariance breaks because a hard shadow edge is
locally indistinguishable from a material edge. This is exactly the case CD-IID also admits it
fails on ("hard shadows of the balconies incorrectly included in the initial albedo").

## The directional-shadow-loss survey — does anyone use one on Hypersim?

**Finding: NO published method uses a dedicated "directional/cast-shadow-region-weighted loss"
on Hypersim.** What the field actually does:

- **Ordinal Shading (Careaga & Aksoy, TOG'23) — the SOTA recipe:** predict ONLY shading with a
  **shift-and-scale-invariant ordinal loss**, then derive albedo = I / shading. The shadow is
  handled by being MODELLED in the shading branch (where it belongs), NOT penalised out of albedo.
  No shadow-specific loss; the intrinsic constraint + good shading supervision does it. Hypersim's
  full shading GT makes this work.
- **CD-IID (TOG'24):** multi-stage feed-forward + albedo-SPARSITY prior; still admits hard-shadow
  leak as a limitation (Fig. 12); suggests post-hoc inpainting.
- **IntrinsicAnything (ECCV'25):** uses a generative DIFFUSION prior as a regulariser to steer
  optimisation away from "shadow baked into albedo" — explicitly "less directly expressible as a
  simple loss." I.e. the diffusion camp's answer to hard shadows is a learned prior, not a loss.
- **Shadow-removal literature (general):** scale-invariant GRADIENT loss > L2 for shadows, but
  "even with gradient loss, predictions are smoother, not perfect"; strong cast shadows remain
  "incomplete with artifacts." No loss fully closes hard shadows.

**Net:** the modern, proven mechanism is NOT a shadow loss — it is **shading-side prediction +
intrinsic constraint** (Ordinal Shading). CARI already predicts shading and has a diffuse-recon
constraint, but it predicts ALBEDO directly (not albedo=I/S), so a hard floor shadow has a direct
path into the albedo head that Ordinal Shading's architecture forbids by construction.

## Recommendation (corrected — no architecture swap, no invented shadow loss)

1. **Reframe the corridor as the documented ceiling**, citing CD-IID's identical admission and the
   shadow-type analysis. IIW (the standard real benchmark) is the headline — and it's STRONG.
   ARAP hard-shadow scenes are the honest limit (§8), shared by the feed-forward SOTA.
2. **The ONE principled, field-aligned lever** (if you want to attack it): the Ordinal-Shading
   trick — supervise shading harder / lean on the intrinsic-derived albedo on Hypersim, so the
   shadow is forced into the shading branch. This is NOT a new shadow loss; it is the proven
   shading-first recipe. Optional ablation, AFTER the raw-color-pair result.
3. **Do NOT** add a directional-shadow loss (nobody does; gradient/boundary losses don't close
   hard shadows) and do NOT switch to diffusion (IIW shows the feed-forward model is good; the
   corridor is one hard case, not a systemic failure).

The corridor is a hard-shadow OUTLIER, not evidence the model or architecture is wrong. The IIW
sheet is the real story: CARI does single-image IID well on real photos.

---

## CORRECTION / DEEPER FINDING (2026-06-16): IIW albedo is NOT homogeneous either

User pushed back: "the shadow is washed out on IIW, but the colour is actually not homogeneous."
**Measured, and the user is RIGHT — this is more fundamental than the corridor.** On detected
FLAT single-material regions of IIW saved albedos (input | albedo pngs):

| scene | input range (p90-p10) on flat surface | albedo range | shading removed? |
|---|---|---|---|
| 627 | 118 | 91  | barely (23%) |
| 686 | 176 | 139 | barely (21%) |
| 370 | 115 | **124** | **NONE — albedo varies MORE than input** |
| 593 | 134 | 120 | barely (10%) |
| 502 | 123 | 47  | mostly (62%) — the only good one |
| 815 | 142 | 76  | partial (46%) |

A good albedo should be near-CONSTANT on a flat surface (range→small). CARI's albedo still
varies almost as much as the input → **smooth shading/shadow gradient is LEAKING into albedo
across ordinary real photos**, not just on the hard corridor. Chroma also inhomogeneous (within-
patch R/G std up to 0.54). Confirmed REAL not a display artifact: albedo↔input correlation =
0.73 on scene 370 (monotonic display transforms can't create a gradient that wasn't there);
implied-shading ratio std=0.236 → the model does SOME separation but leaves large residual.

**Why this was missed:** WHDR (27.7%) only checks ORDINAL pairs (is A darker than B?) — it is
BLIND to smooth shading gradients staying in albedo. The albedo "looks plausible" (right colours,
recognisable) so it passed the eyeball test. The flat-region homogeneity test exposes it.

**What this IS:** the F1/F3 failure from LITREVIEW_v12_v17_v18.md — the ~50/50 cast split and
"shadow removal only APPROXIMATELY enforced; nothing pulls residual shadow energy out" — now
MEASURED on IIW, not hypothesised. The A/S split is under-constrained: recon (A·S≈I) + MSG-to-GT
do not force albedo to be flat within a material; a smooth shading ramp satisfies recon perfectly
while sitting in albedo.

**Implication for the thesis (this RAISES the stakes of the contribution, doesn't lower them):**
the residual-shading leak is EXACTLY what cross-render L_inv attacks — same material under
different light must give the SAME albedo, which directly penalises a shading gradient that moves
with the light. So the IIW inhomogeneity is the BEST motivation for CARI yet: it's the measured
defect the mechanism targets. **Re-run this flat-region homogeneity test as a METRIC** (call it
"within-material albedo flatness" / residual-shading index) on 19k vs 40k vs raw-color-pair —
if L_inv reduces the flat-region albedo range, that is a clean, WHDR-independent thesis number
that the standard benchmark cannot show.

**Candidate fixes (ranked, all WITHOUT architecture swap):**
1. The cross-render L_inv (already the plan) — directly penalises shading-that-moves-with-light.
2. Stronger same-material flatness prior on Hypersim/IIW: penalise albedo variance WITHIN a
   material/segment region (CRefNet-style) — directly targets the residual.
3. Ordinal-Shading recipe (predict shading, derive albedo=I/S) — structurally moves shading out
   of the albedo head. Bigger change; optional.
NOT a perceptual loss (doesn't touch the split), NOT diffusion (the split is the issue, not crispness).
