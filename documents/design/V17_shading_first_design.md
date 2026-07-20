# Shading-first / flatness for V17 — THREE-EXPERT VERDICT + corrected design

> **2026-06-16.** Original "shading-first" proposal (S1 + S2) was stress-tested by three
> independent experts (math/physics, recent literature, optimization). **S1 is REJECTED.**
> This doc records why, the corrected design (S2 + a *texture-preserving* flatness prior), and
> the measured answer to "will a flatness prior remove fine floor texture?" (yes if naive; no
> if chroma/frequency-gated — measured below).

## The defect being fixed (measured, real)

On flat single-material regions of real IIW photos, CARI's albedo is NOT homogeneous — a smooth
shading gradient leaks into albedo (albedo↔input corr 0.73; albedo range on flat surfaces ≈
input range). WHDR (27.7%) is BLIND to this (it only checks ordinal pairs). See
`documents/evals/iiw_vs_arap_shadow_analysis.md`.

## THREE-EXPERT VERDICT on the original proposals

### S1 — `L_consist = ‖A_pred − stopgrad(I/S_d)‖₁` on lit pixels — ❌ REJECTED (unanimous)

**Expert 1 (math) — the killer proof.** S1 is algebraically redundant with reconstruction:
```
A_pred − I/S_d = (A_pred·S_d − I)/S_d = r/S_d        (r = recon residual)
⇒ L_consist = ‖r/S_d‖₁   — recon, reweighted by 1/S_d
```
Same zero set, same gradient sign as recon. **The shading-ramp-in-albedo solution `A=I/S_d` makes
`L_consist=0` EXACTLY** — S1 is satisfied by the very leak it should kill. Root cause: the defect
is **regional** (`∇A=0` within a material); S1 is **pointwise** with recon's zero set. Category
mismatch — you cannot remove a spatial-derivative violation with a pointwise term.

**Expert 2 (literature) — nobody does this; proven fixes are elsewhere.** No recent paper reports
an "A=I/S consistency" term beating plain recon. FlowIID framing: separate A+S prediction "often
results in reconstruction inconsistencies" → winners DERIVE albedo so recon holds by construction.
The proven fix for THIS defect = CRefNet's within-material flatness prior (rectified-gradient
filter; explicitly "trades fine albedo detail for consistency"; ~10% WHDR gain). Measure with MAW
**texture metric** (LPIPS on constant-shading rectangles), not WHDR.

**Expert 3 (training dynamics) — symmetric S1 is actively harmful.** (a) It FIGHTS L_inv: on MID
raw-color pairs `I/S_d` carries the colored illuminant, so S1 pushes it BACK into albedo — undoing
the CARI mechanism exactly where the thesis lives. (b) Path of least resistance: the optimizer
satisfies S1 by bending the cheap unsupervised `S_d`, leaving the albedo ramp → won't move the
metric. (c) Unbounded `S_d` gradient near shadows (`∂/∂S_d[I/S_d]=−I/S_d²`) → instability.

> Expert 3's salvage (asymmetric, stopgrad both sides, gradient to A only, Hypersim-only, low
> weight, ramped) is *less harmful* but Expert 1 shows it is still recon-reweighted, so expected
> gain is small. **Not worth the complexity. Drop S1 entirely.**

### S2 — shading-first conditioning (albedo head reads `detach(S_d)`) — ✅ KEEP (low risk)

- Expert 2: this IS the validated paradigm (Ordinal Shading TOG'23, CD-IID TOG'24 condition the
  high-res albedo on the prior shading decomposition). Award-recognized precedent.
- Experts 1 & 3: `detach(S_d)` carries no NEW information beyond trunk `F` (data-processing
  inequality — both derive from the same DINOv2 features), so it can't *fix* the leak alone, but
  it gives the albedo head a clean precomputed feature → mild capacity gain, won't hurt.
- **Verdict:** keep S2 as a capacity enabler, NOT as the fix. It needs a regional term to bite.

## CORRECTED DESIGN — the regional fix the math actually requires

The defect is regional ⇒ the term must act on a SPATIAL DERIVATIVE / regional statistic of `A`.
Two such terms, used together (inter-frame + intra-frame):

1. **(F-prior) Texture-preserving within-material albedo flatness** — penalise `∇A` where the
   gradient is a SHADING edge, NOT where it is a texture/material edge. (Design below.)
2. **L_inv (cross-render, already planned)** — Expert 1 explicitly names this a VALID regional/
   cross-frame coupling: same material under different light → same albedo penalises a shading
   ramp that moves with the light. Inter-frame complement to the intra-frame F-prior.
3. **S2** — conditioning, for capacity.

## "Will the flatness prior remove fine floor texture?" — MEASURED ANSWER

**A NAÏVE `‖∇A‖` prior: YES, it removes texture.** It cannot distinguish a shading ramp
(remove) from tile grout / wood grain (keep) — both are `∇A`. This is the CRefNet tradeoff
("trades fine albedo detail for consistency"). Unacceptable as-is for our defect.

**A GATED prior: NO — it can keep texture. (Design CORRECTED after web-search double-check.)**

The discriminators were checked against the literature. One is validated; one I had over-trusted.

- **✅ Chromaticity gate (VALIDATED — this is the real mechanism).** A shading edge changes
  LUMINANCE only (shadow darkens R,G,B together); a material/texture edge changes CHROMA. This is
  exactly **PIE-Net** (Das et al., CVPR'22): "edges correspond to ILLUMINATION-INVARIANT
  gradients," uses photometric-invariant edge descriptors to classify reflectance-vs-shading edges
  → SOTA, generalises to real images; ablation confirms illumination-invariant descriptors help.
  Also the classic chromaticity prior ("similar chromaticity ⇒ same reflectance"; affinity =
  luminance similarity + colour-angle) and "Fast sparse edge IID guided by chromaticity gradients"
  (IEEE). MEASURED on IIW-370 floor: of strong-luminance edges, **8% are low-chroma (shading)** vs
  **16% high-chroma (texture)** → a chroma-gated penalty fires on the 8% and PROTECTS the 16%.

- **❌ Frequency gate (DROPPED as a texture-protector — literature REFUTES it).** I previously
  proposed "penalise only the low-freq band of ∇A to protect achromatic texture." **Web-search
  correction:** the structure-texture IID literature explicitly rejects the high-freq=reflectance
  / low-freq=shading rule — *"there are cases that high frequency changes are part of the SHADING
  component (e.g. surfaces with bumps and wrinkles)."* So a frequency gate fails in BOTH
  directions: it would leak high-freq SHADING (wrinkles, bump-shadows) into albedo. Frequency is
  NOT a reliable texture protector. Keep it descriptive only, not as the safety mechanism.

- **✅ The robust protector = chroma gate + WITHIN-MATERIAL masking (the literature's actual fix).**
  The proven structure-texture+material-recognition method (PMC5161468, LMSE 0.0129 on MIT)
  protects texture not by frequency but by **material segmentation** (separate texture, decompose
  smooth structure, re-insert texture into the layer material-recognition assigns). We get this
  FREE: restrict the flatness penalty to WITHIN a GT material/segment region (Hypersim
  `render_entity`, MID `materials_mip2.png`), so it never crosses or smooths across a true
  material/texture boundary. Within a segment, the remaining variation is mostly the shading ramp.

**Recommended F-prior (CORRECTED — texture-safe):**
```
w_shade(p) = relu( |∇_lum I(p)| − λ_c·|∇_chroma I(p)| )   # high at luminance-only (shading) edges,
                                                          # ~0 at chroma (texture) edges  [PIE-Net]
L_flat = Σ_p  M_seg(p) · w_shade(p) · |∇A(p)|             # penalise albedo grad ONLY: within a
                                                          # material segment AND at a shading edge
```
- `w_shade` from the INPUT (fixed per-pixel mask, cheap, no backprop instability).
- `M_seg` = within-material mask from GT segmentation (the literature's material-recognition step,
  free from Hypersim/MID GT) — the PRIMARY texture protector, replacing the dropped frequency gate.
- NO low-freq band-pass on ∇A (it would leak bump/wrinkle shading).

### ⚠️ CRITICAL LIMIT — the chroma gate FAILS on COLORED shadows (V12/V16 CCR proof)

The chroma gate is the SAME chromaticity invariant V12/V16 used as CCR, so it inherits CCR's
colored-illuminant failure ([[ccr-colored-illumination-limit]], proven there). Decompose by edge
type:

| Edge | `∇_chroma` | gate says | correct? |
|---|---|---|---|
| WHITE/achromatic shadow (R,G,B darken equally) | ≈0 | penalise ∇A (flatten) | ✅ it IS shading |
| material/texture (grout, tile colour) | high | protect ∇A (keep) | ✅ it IS texture |
| **COLORED shadow** (illuminant colour changes across the boundary, e.g. blue-tinted shadow) | **high** | **protect ∇A** → shadow STAYS in albedo | ❌ WRONG — gate mistakes a colored shadow for a material edge |

**So the chroma gate leaks exactly the COLORED shadow the thesis exists to fix.** This is NOT a
gate-tuning bug — it is the fundamental local-ambiguity limit: at a spatially-varying colored-
illumination boundary, a material edge and an illumination-colour edge are LOCALLY
INDISTINGUISHABLE. No local feature (CCR, chroma-gradient, any pointwise gate) can separate them.
Proven in `ccr-colored-illumination-limit` memory: the CCR kernel cancels a GLOBAL colored cast
but NOT spatially-varying colored shadows. V12/V16 measured this failure directly.

**This SHARPENS the division of labour (and turns the V12/V16 CCR experience into a thesis
argument):**
- **WHITE / achromatic shading leak** (the IIW indoor floor — measured 8% low-chroma shading
  edges) → chroma-gated within-material flatness prior. LOCAL, works.
- **COLORED shading leak** (the headline thesis failure) → ONLY the cross-render `L_inv`
  (INTER-frame) can fix it. It observes the SAME material edge staying fixed while the colored
  shadow MOVES across two real frames — the only signal that disambiguates a colored shadow from a
  material edge. CCR/V12/V16 are the PROOF that no local prior can do this.

So: the flatness prior is a complement for the easy (white) case; **`L_inv` is load-bearing and
irreplaceable for the colored case** — and the chroma gate must be DISABLED (or fall back to
within-material masking only) wherever colored illumination is present (MID raw-color pairs,
colored-aug Hypersim), so it does not wrongly protect a colored shadow. Practically: on
colored-illuminant data, rely on `M_seg` (within-material mask) WITHOUT the chroma term, since
the chroma term is unreliable there; on white-illuminant synthetic data the chroma gate is safe.

**Honest residual risks (literature-confirmed ceilings):**
1. **Achromatic texture at an achromatic shading edge** (grey grout under a shadow line): chroma
   gate can't separate; within-segment masking still protects it UNLESS the shading edge lies
   inside one material. Rare; report it.
2. **High-frequency shading from bumps/wrinkles** (the structure-texture caveat): if a bumpy
   surface is one material, `w_shade` correctly fires (luminance-only) and we'd flatten the
   bump-shading INTO albedo-flatness — which is actually CORRECT (bump shading is shading). The
   risk is the reverse only for chromatic bumpy materials, which is uncommon.
   Net: the chroma+segment gate is the validated design; the frequency gate was the wrong idea.

## Plan
1. Implement the **within-material albedo flatness METRIC** (quantify the leak: 19k vs 40k vs
   raw-color-pair) + adopt **MAW texture metric** as the external version. Measure FIRST.
2. Implement the **texture-preserving F-prior** (`lambda_alb_flat`, chroma+freq gated, input-mask
   weight), config-gated, Hypersim/IV + MID-within-segment.
3. Add **S2** conditioning (capacity).
4. Ablate: V17 / +F-prior / +F-prior+S2 / +L_inv(CARI) on the flatness metric + MAW texture +
   IIW WHDR (must not regress — if texture is lost, WHDR and MAW-texture catch it).
5. **S1 is dropped** — recorded here so it does not resurface.
