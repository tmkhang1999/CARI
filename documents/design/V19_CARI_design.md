# V19 — CARI: Cross-render Albedo-invariant Intrinsics

> **Status:** Design (2026-06-10). A *new* single-image intrinsic-decomposition model whose
> contribution is **scientific, not engineering**: it imports the multi-illumination
> disambiguation signal — the thing GS-2M gets from multi-view 3D — into a **single-image
> feed-forward 2D model at training time**, via a cross-render Siamese objective on
> pixel-aligned multi-light images. No diffusion fine-tuning, no LLM, no image enhancement.
> Fits 24 GB. Backbone: fresh DINOv2-L + DPT transformer (V17 skeleton), but the
> *contribution lives in the data + loss + a light illumination-equivariant module*, which
> together fix V17's measured desaturation **without** a generative prior or an MLLM judge.
>
> Companion to `LITREVIEW_v12_v17_v18.md` (why V12/V17/V18 fail) and `V17_design.md`
> (the backbone skeleton we reuse).

---

## 0. TL;DR — the thesis (one claim, with a number)

**Single-image intrinsic decomposition is ill-posed because one image cannot separate a
material's color from a colored illuminant (the cast splits ~50/50 into albedo and shading
— *measured* on every prior model, including Marigold/V18). We show this ambiguity is
partly resolvable *at inference from a single image* by importing, at *training* time, the
disambiguation signal that multi-view/multi-light capture provides: the same surface under
different lighting has a fixed albedo and a varying shading. We mine this signal — for free,
pixel-aligned — from the Multi-Illumination Dataset (MID), and supervise a single-image
transformer with a cross-render albedo-invariance + shading-explains-the-change objective.
The model learns the illumination invariance a single image alone cannot derive, and we
*measure* the reduction in albedo color-cast and the gain in cross-illumination albedo
constancy on held-out scenes and on IIW/ARAP.**

This is the **GS-2M pattern** ("exploit a free physical signal most methods discard"):
feed-forward, single-image-at-test, using multi-illumination *as supervision* rather than as
a multi-view test-time requirement. The supervision *family* has prior art (§1.1 — BigTime,
CGIntrinsics, SAIL); the contribution is the **physical colorful-diffuse formulation, the
HDR-aware pair validity, the foundation-backbone setting, and the first controlled
quantification** of how much of the colored-light ambiguity this signal recovers — including
transfer to an out-of-domain multi-light benchmark (ARAP `_lightN`, §6) that nobody trains on.

---

## 1. The gap nobody addresses (and why it's real)

Three facts, each independently verifiable in this repo:

1. **The ambiguity is measured, not hypothesized.** `THESIS_design.md §2.2.1`: V18's albedo
   carries ~half the illuminant cast (`kitchen [1.50,1,0.70]`). V17 desaturates because its
   frozen color-invariant backbone discards absolute chroma
   (`LITREVIEW_v12_v17_v18.md §2.2 F1`). V12 leaks colored shadows via `I/S`
   (`§1.2 F4`, [[ccr-colored-illumination-limit]]). **All three fail on the same axis: a
   single image cannot disambiguate material color from light color.**

2. **The modern SOTA line trains i.i.d. on (image, albedo) pairs and therefore inherits
   this ceiling.** Marigold-IID (latent MSE per modality), CD-IID/Colorful (cascade
   regression), CRefNet — none use cross-illumination consistency as a *training* signal;
   they treat each image independently, so they can only learn the *average* split. (An
   older/parallel line DID use multi-illumination supervision — positioned honestly in
   §1.1; the earlier draft's claim that "nobody does this" was an overclaim that would die
   in the related-work chapter.)

3. **The disambiguating signal exists in datasets we already have, and the old docs threw
   it away.** MID ships **985 train scenes × 25 pixel-aligned images under different light
   directions + a shared albedo + per-pixel material segmentation**. The old V17/V18 docs
   *dropped MID* (`V18_PGID_design.md §9`) because they tried to use its (noisy, pseudo-GT)
   albedo as a single-image *target*. **As a cross-render invariance signal, the noisy
   albedo is irrelevant — what matters is that the 25 frames share it.** Recovering a
   discarded dataset by using it as designed is part of the story.

### 1.1 Honest lineage — who has used multi-illumination supervision, and the precise gap

The supervision *family* is not new. The thesis must position against three works explicitly
or it dies in the related-work chapter:

| Work | What it does | What it lacks (= our slot) |
|---|---|---|
| **Li & Snavely, *Watching the World* (CVPR'18, BigTime)** | time-lapse reflectance constancy trains a single-image CNN | grayscale-Retinex era: no colorful 3-ch shading, no residual, no HDR validity; time-lapse lighting variation is weak (SAIL's own observation) |
| **CGIntrinsics (Li & Snavely, ECCV'18)** | synthetic + IIW/SAW ordinal supervision | constancy used as a side prior, not the central mechanism; same grayscale-shading limits |
| **SAIL (arXiv:2505.19751, 2025)** | cross-render albedo invariance **on MID itself**, in SD latent space | **latent-ADDITIVE** `z=z_A+z_E` (not physical `I=A·S`), **albedo-only** (no explicit shading output), LDR (no HDR/specular handling), diffusion backbone; admits "lighting artifacts and shadows remain" |
| **CARI (this thesis)** | cross-render supervision on the **physical multiplicative colorful-diffuse split** `I=A·S₃ch+R` in linear HDR, per-pair HDR validity, frozen-foundation feed-forward backbone | — |

**Honest math note:** `L_inv + L_explain` together are equivalent to **log-domain reflectance
constancy** (from `I=A·S` with shared `A`: `log I_a − log I_b = log S_a − log S_b`) — the
loss *equations* have lineage. The contribution is (i) the physical colorful-diffuse +
analytic-R formulation they supervise, (ii) the HDR-aware per-pair validity that makes them
usable on real flash HDR, (iii) the foundation-backbone setting, and (iv) the first
**controlled quantification** of how much of the colored-light ambiguity this signal
recovers — including transfer to an out-of-domain multi-light benchmark (ARAP `_lightN`,
§6) that no method trains on. GS-2M remains the *pattern* (multi-observation
disambiguation), not the claimed lineage.

---

## 2. The free signal (verified on disk)

`../datasets/MIDIntrinsics/train/<scene>/`:
```
albedo.exr              # one shared albedo for the scene (pseudo-GT; used only weakly)
dir_0_mip2.exr … dir_24_mip2.exr   # 25 PIXEL-ALIGNED images, same scene, 25 light dirs
materials_mip2.png      # per-pixel MATERIAL ID (~17/scene) → same-material masks for FREE
probes/ , meta.json     # chrome/gray light probes → measured illuminant per frame (optional)
```
- **985 train / 30 test scenes.** Pixel `(x,y)` is the **same surface point** in all 25
  frames of a scene → **zero correspondence mining** (the hard part of the GS-2M analogue is
  *given* to us, because MID is captured from a fixed camera).
- `materials_mip2.png` gives **same-material regions without a model** → the consistency
  loss has exact support.

This is the cleanest possible substrate: the multi-view photometric variation GS-2M must
*reconstruct in 3D* is here **pre-aligned in 2D and downloaded**.

### 2.1 MID is HDR and per-frame-clipped — the pair signal needs a validity mask (MEASURED)

Inspected `everett_dining1` (`tests/visualizations/midintrinsics/.../MID_example.png` + value
probe): the `dir_N` EXRs are **genuine HDR** (max 17.7 / 146 / 100 across frames; median ~0.3)
with **negative artifact values** (min −0.06…−0.59 → clamp ≥0). Crucially, **each flash
direction blows out / goes specular in DIFFERENT pixels** (`dir_5`: 0.4% px >1; `dir_12`:
17.5% px >1). The big red/yellow regions in the viz are HDR over-range, not data.

**Consequence for CARI (design correction):** raw frame pairs are *not* usable per-pixel
everywhere. `L_inv (A_a≈A_b)` and `L_explain` are valid **only where BOTH frames are
well-exposed and non-specular** — a strict **per-pair diffuse-validity mask** computed in
**linear HDR, before tonemapping**: (a) both frames' luminance below a clip ceiling, (b)
exclude per-frame specular hotspots (the moving flash highlight is non-diffuse → belongs in
residual, would corrupt invariance), (c) clamp negatives. The existing loader's
display-space `<0.99` mask (post-tonemap, [midintrinsic_dataset.py:176](../src/data/midintrinsic_dataset.py#L176))
is the wrong space for this — CARI needs HDR-linear per-frame validity.

### 2.2 Pair source decision — RAW measured pairs (core) + synth augmentation (auxiliary)

The existing loader *synthesizes* illumination (WB + Dirichlet blend + Lab shift,
[midintrinsic_dataset.py:110-128](../src/data/midintrinsic_dataset.py#L110-L128)) and can emit
a paired second mix (`use_paired`/`extra_rgb`). **Decision:** the *core* cross-render signal
must be **raw measured `dir_a` vs `dir_b`** of the same scene — otherwise the "real
multi-illumination signal" claim (what separates CARI from "Marigold + augmentation") does not
hold; synthetic WB+Lab variation is exactly the augmentation a reviewer dismisses. Keep the
synth path as **auxiliary augmentation only**, enabling the ablation **raw-only vs raw+synth**
that *proves the raw measured signal is what carries the gain*. (Albedo's clean [0,1.1] range
makes it a fine *weak* absolute anchor; its pseudo-GT noise is irrelevant to `L_inv`.)

### 2.3 Chromatic pair augmentation — covering the COLORED-cast axis MID lacks

MID's flashes are **white**, and the loader white-balances every frame by its gray probe
(`_white_balance`), so raw pairs vary in light **direction/intensity** (shadows move —
exactly right for the shadow-leak failure) but carry almost no **illuminant-color**
variation — yet the headline measured failure is the ~50/50 *colored*-cast split. Fix
(cheap, physically exact): **multiply ONE frame of a raw pair by a random illuminant color
`c`**. Since `c·I = A·(c·S) + c·R`, the pair stays physically valid with the **same
albedo**, so `L_inv` now also demands invariance to light *color* on top of real direction
variation, and `L_explain` still holds (`S' = c·S`). Real direction variation × exact
synthetic color variation covers both failure axes. Ablation row: raw-direction-only vs
+chromatic.

---

## 3. Model — CARI

A single-image transformer (V17 skeleton) trained with a Siamese cross-render objective.

```
            ┌───────────────── shared weights ─────────────────┐
   I_a ──▶ DINOv2-L (frozen) ─▶ DPT trunk ─▶ [A_a | π_a]        │   (light dir a)
   I_b ──▶ DINOv2-L (frozen) ─▶ DPT trunk ─▶ [A_b | π_b]        │   (light dir b, SAME scene)
            └──────────────────────────────────────────────────┘
                                   │
        ┌──────────────────────────┼───────────────────────────────────────────┐
        ▼                          ▼                                           ▼
  (L_inv) A_a ≈ A_b          (L_explain) I_a / I_b  ==  S_a / S_b        (L_sup) A_{a,b}≈albedo,
  per-pixel, per-material      at every pixel (light change is ALL shading)   SSI shading (Hypersim)
```

- **Backbone:** reuse `IntrinsicDecompositionV17` *unchanged* (DINOv2-L frozen + DPT +
  `[A | π]` heads, analytic `R`). The contribution is **not** the backbone — it's what
  trains it. This also makes V17 a *clean ablation* (CARI = V17 + cross-render).
### 3.1 Architecture roadmap — what to change, what NOT to change, and in what order

**Principle:** the thesis claim is a *mechanism* proven by a controlled comparison
(CARI = V17 + cross-render, same backbone). Architecture changes therefore enter **only as
ablation rows AFTER the core result (rows 1–3) lands** — a change made before it would
confound the very comparison the thesis rests on. Ranked by expected value/effort:

**(A1) α-gated illumination code — the architectural-contribution candidate.** From Latent
Intrinsics (NeurIPS'24, arXiv:2405.21074): a small MLP predicts a per-image illumination
code `e(I)` from the DINOv2 CLS/global token; the shared trunk features feeding the
**albedo head** pass through `F̃ = F ⊙ (1 + α·tanh(MLP(e)))` with `α ≈ 5e-3` — lighting can
only modulate the albedo stream by a hair, i.e. it is **structurally forbidden** from
writing into albedo, while the shading head consumes `e` freely (equivariant). Trained by
the same cross-render pairs (swapping `e_a↔e_b` must swap the shading, not the albedo).
~10 lines + one MLP; turns "invariance by loss" into "invariance by construction" — the
*architectural* claim if it beats loss-only CARI.

**(A2) RGB color-skip into the albedo head (+ optional Retinex anchor).** Fixes the
measured channel imbalance (≈64 color-bearing detail channels vs ≈256 color-flattened
DINOv2 channels, §4.4): a learned skip from the gamma-encoded input into the albedo head
only, optionally with SAIL's image-anchor loss pulling albedo chroma toward input chroma.
**Only valid WITH `L_inv`** (§4.5) — alone it re-leaks the illuminant; the coupling is part
of the contribution.

**(A3) Normal-conditioned albedo (IDArb-style, arXiv:2412.12083).** A small normal head on
the DPT trunk (normals GT exists in Hypersim/IV); the albedo head consumes predicted
normals. Targets the **intra-frame** "dark paint vs shadow" ambiguity that the
**inter-frame** cross-render losses are mathematically silent on (a convex dark region with
smooth normals is paint, not shadow). Cheap, complementary, orthogonal signal.

**Explicitly REJECTED architecture moves (with reasons, for the defense):**
- **Switching to a diffusion backbone** — DNF-Intrinsic (ICCV'25) argues one-step
  deterministic ≥ multi-step generative *for the disambiguation goal*; a switch would
  destroy the controlled V17 comparison, the 19k warm-start, and the 24 GB/feed-forward
  claims, while inheriting SAIL's optics ("SD fine-tune"). The generative-prior comparison
  is already covered by reporting V18/Marigold as baselines.
- **Swapping DINOv2→DINOv3 (or bigger)** — DINOv3 is trained with the same
  photometric-augmentation invariance that *causes* the color flattening; a swap changes
  capacity, not the mechanism, and breaks the warm start.
- **A from-scratch CNN/hybrid** — the model already IS a hybrid (DetailStem is a CNN branch
  on raw RGB); V12 is the measured evidence that hand-built CNN priors plateau. Widening
  the existing CNN branch (A2) is the principled version of "adapt CNN".

---

## 4. Losses — the core contribution

For a Siamese pair `(I_a, I_b)` of the same MID scene, with material mask `M` and valid mask:

### 4.1 Cross-render albedo invariance `L_inv` (the disambiguator)
```
L_inv = mean_{p}  ‖ A_a(p) − A_b(p) ‖₁         # same surface, different light → SAME albedo
```
Applied per-pixel (pixels are aligned) **and** pooled per-material region (via
`materials_mip2.png`) for a stronger, lower-variance signal. **This is the term that forces
the colored cast OUT of albedo** — because the only thing that differs between `I_a,I_b` is
light, any albedo difference is the model wrongly absorbing illumination. *GT-free w.r.t.
albedo value* (doesn't need MID's noisy albedo).

### 4.2 Shading-explains-the-change `L_explain` (the complement)
```
ratio_I = I_a / (I_b + ε)          # observed photometric change (all due to light)
ratio_S = S_a / (S_b + ε)          # predicted shading change   (S = (1−π)/π)
L_explain = mean_{p}  ‖ log ratio_I − log ratio_S ‖₁    # in log domain, robust
```
Forces the *entire* inter-frame change into shading — the positive counterpart of `L_inv`.
Together they implement "same material under different light → supervise the split" (the
GS-2M idea) **on a single-image model**. Saturation-guarded (mask clipped pixels, reuse
`sat_ok`).

### 4.3 Standard supervised anchors (keep the model honest on absolute scale)
- **Albedo MSE+MSG** on **Hypersim/InteriorVerse** (clean albedo GT) — gives *absolute*
  material color so `L_inv` doesn't collapse to a constant (invariance is necessary, not
  sufficient: a model outputting gray everywhere is perfectly invariant). This pairing is the
  crux — **`L_inv` removes the cast, supervised albedo fixes the absolute level.**
- **SSI shading** (Hypersim, π-domain) and **two-sided diffuse recon** `A·S_d≈A_gt·S_d_gt`
  — reuse `V17Loss` verbatim for these.
- **MID albedo:** used only as a **weak** anchor (low weight) or not at all — its value is
  the cross-render *structure*, not its noisy albedo number. (This is exactly why the old
  docs' reason to drop MID does not apply to CARI.)

### 4.4 Why this fixes V17's desaturation *without* an LLM

**Corrected diagnosis (2026-06-10, after re-reading the code — supersedes the earlier
"information-theoretically destroyed" framing, which was WRONG).** Color is *not* destroyed:
the albedo head reads `f_trunk`, and `f_trunk = out_conv(cat[fusion_feats, DetailStem(rgb)])`
([dpt_decoder.py:150-153](../src/models/decoders/dpt_decoder.py#L150-L153)) — the DetailStem is
a trainable conv on the **raw RGB**, so the input color *is* available to the head. The real
causes of desaturation are:
1. **Channel imbalance:** color enters at one scale through `detail_ch≈64` channels vs
   `fusion_ch≈256` DINOv2-derived channels that are color-*flattened* (DINOv2 is invariant to
   "viewpoint, illumination, **color**" — confirmed: DINOv2 features literature). The optimizer
   leans on the dominant invariant stream → output drifts gray.
2. **No saturation-rewarding signal:** albedo loss is MSE+MSG with `lambda_dssim=0`
   ([v17.yaml:106](../src/configs/v17.yaml#L106)); L2 regresses to the desaturated mean, and
   `mat_consist` operates on L2-normalized chroma (scale-free) so it homogenizes hue without
   restoring saturation.

CARI's two-part fix, each with provenance:
- **(supervised absolute color)** clean-albedo MSE+MSG+**DSSIM** on Hypersim/IV — the
  **CRefNet recipe** (direct GT + multi-scale gradient-*matching to GT* + DSSIM; CRefNet uses
  **no** decorrelation term, `solver/loss.py`). This is the *safe* edge-ownership mechanism: it
  puts each edge where GT says it belongs without assuming albedo/shading edges are exclusive.
- **(disambiguation)** `L_inv`/`L_explain` on MID teach the decoder that chroma differences
  across lighting are *errors* → it stops averaging real material color away, **and** it tells
  the model which RGB color is material (invariant) vs illuminant (varies). Lineage:
  multi-illumination constancy (CGIntrinsics / Li–Snavely; SAIL; the IIW premise).

No GRPO, no VLM, no real albedo labels — the supervision is "these two pixels are the same
material under different light," which MID gives for free. **This achieves the ReasonX goal
(real-image invariance signal) with a dataset instead of an LLM.**

### 4.5 Optional architectural lever — RGB color skip into the albedo head (NOT alone)
Widening the color path (raise `detail_ch`, or a learned RGB skip *into the albedo head only*)
addresses cause (1). **Drawback (real, why it was disabled before):** a raw-RGB skip also feeds
the **illuminant** color → re-introduces the cast leak (the old `albedo_chroma_skip`'s failure;
`c(I)` cancels only white shading). **So this lever is adopted ONLY in conjunction with `L_inv`**
— the cross-render loss is what disambiguates the material vs illuminant color the skip injects.
Ablation: skip-only (expect cast leak) vs skip+`L_inv` (expect clean saturation). This coupling
*is itself part of the contribution* — it shows the architectural color path needs the
cross-render signal to be usable.

### 4.6 Rejected lever — gradient-decorrelation / exclusion loss (evidence + failure mode)
`_grad_decorr = mean(|∇A|·|∇S|)` ([flexible_loss_v17.py:149](../src/losses/flexible_loss_v17.py#L149))
is the **exclusion loss** (Zhang et al., *Single Image Reflection Separation*, CVPR'18:
`tanh(λ_T|∇T|)⊙tanh(λ_R|∇R|)`), minus its normalization. **Rejected as a default for two
evidenced reasons:**
1. **Its core assumption — "an edge belongs to either layer, not both" — is FALSE for IID.** In
   reflection removal, transmission/reflection edges rarely overlap; in IID, albedo and shading
   edges **co-occur constantly** (every object on a surface: the sofa/carpet boundary is *both* a
   material edge and an occlusion-shading edge). The product `|∇A|·|∇S|` is maximal exactly
   there and tries to **erase a genuinely-shared edge** from the weaker layer → can *worsen* the
   sofa-boundary artifact and smear real shading discontinuities.
2. **Without the paper's normalization it is unstable** — "the network would suppress the layer
   with a smaller gradient update rate to close to zero" (Zhang'18). The V18 doc already notes it
   "logged ≈0 in V17 (never bit)."
Modern strongly-supervised IID **drops it**: CRefNet & Marigold-IID use direct GT + recon and
carry no decorrelation term. Kept only as a **cautious ablation row**, never default.

### 4.7 Data-free shading sign/inequality prior (implemented: `lambda_shade_sign`)

`relu(A·S_d − I)`, masked, all datasets, all phases. The analytic residual is one-sided
(`R=(I−A·S_d)₊ ≥ 0`), so diffuse overshoot `A·S_d > I` is un-modelled energy — *"anything
brightening must be albedo, not shading"* (SAIL's `ℒ_reg` adapted to the multiplicative
split). A genuine **half-space constraint** on the free split direction that costs no data
and bites exactly where SSI is absent (InteriorVerse/MID). Implemented in
`V17Loss.cari_shade_sign`, weight `lambda_shade_sign`.

### 4.8 Red-color loss diagnosis and three complementary fixes (2026-06-12)

**Root-cause analysis from training visualisations (checkpoints 20k–30k, red-cloth scene).**
Saturated reds are *not* rare in training data (Hypersim albedo: 8.7% of pixels are
high-saturation reds/oranges; MID: 16.4%) — data scarcity is NOT the cause. The optimizer
uses **three structural escape routes**, each costing near-zero loss while muting the red:

| Escape route | Why it costs zero | Loss that closes it |
|---|---|---|
| Red → shading (pink Shading Pred) | On MID rows, no SSI shading GT; `L_explain` only constrains the *ratio between lightings* — a static pink tint in both S_a and S_b cancels in the ratio. Free on MID. | `L_inv` (forces A_a=A_b, static tint in both SHading not assignable to albedo difference) + RGB-skip (§4.5/A2) for capacity |
| Red → residual (R absorbs undershoot) | On MID rows, `m_residual=0.0` disabled *all* residual supervision including the sparsity term. Analytic R=(I−A·S)₊ absorbs whatever A·S undershoots. Free. | **Fix 3: close residual hatch** — set `m_residual=1.0` on MID + enable `lambda_res_sparse=0.02`. R_star≈0 away from speculars (excluded by HDR mask), so both L1-to-GT and sparsity push R→0. |
| Desaturated albedo (terracotta, not red) | MSE makes desaturating (0.7,0.06,0.08)→(0.5,0.18,0.18) cheap — the chroma direction is nearly free under L2. DetailStem injects color via only ~64 channels vs ~256 DINOv2 channels (§4.4 imbalance). | **Fix 1: A2 RGB-skip** (wider color path); **Fix 2: chroma L1** (prices the hue direction directly) |

**Three fixes implemented (2026-06-12):**

1. **A2 RGB-skip into albedo head** (`albedo_rgb_skip: true` in `v17.yaml`).  
   Passes gamma-encoded raw RGB `(rgb+1e-6)^(1/2.2)` directly to the albedo `DecodeHead`
   as a full-resolution skip — wider color path than the 64-ch DetailStem. Architecture
   change: `IntrinsicDecompositionV17.__init__` checks `albedo_rgb_skip` and sets
   `alb_skip_ch=3`; the existing `DecodeHead` skip machinery handles it.  
   **Config gate:** `albedo_rgb_skip: false` for current ablation-row-2 run (cross-render
   only); `true` for the full-CARI relaunch (row-4). **Requires `L_inv`** — without it the
   illuminant color re-leaks into albedo via this path.

2. **Chroma-direction L1** (`lambda_a_chroma` in `V17Loss`).  
   L1 between unit-normalised albedo vectors: `‖A/‖A‖₁ − A_gt/‖A_gt‖₁‖₁`. This scores the
   *direction* of the albedo color vector, making hue/saturation errors expensive independently
   of per-channel magnitude. Complementary to MSE (MSE anchors scale, chroma-L1 anchors
   direction). Implemented as `loss_c_chroma` in `forward`, TensorBoard tag `1.
   Losses/A_Chroma`. **Config:** `lambda_a_chroma: 0.0` (disabled for ablation-row-2;
   `0.2` for the full-CARI relaunch).

3. **Close the residual undershoot hatch on MID** (MID dataset change + config).  
   Changed `m_residual = 1.0` in `MIDIntrinsicDataset.__getitem__` (was 0.0). Enabled
   `lambda_res_sparse: 0.02` AND `lambda_r: 0.5` (was 0.0). **Bug found & fixed
   (2026-06-12): `lambda_r` multiplies the whole residual block in `V17Loss`
   (`lr_w = w_r·(l1 + msg + sparse)`), so at `lambda_r: 0.0` the m_residual/sparsity
   changes were dead code.** Why the L1-to-GT is exactly the right closure on MID:
   `compute_targets` routes `S_d_star = I/A*` when `M_diffuse=0`, so
   `R_star = I − A*·(I/A*) = 0` **by construction** — the L1-to-GT on MID *is* the
   undershoot penalty `|R−0|`. On Hypersim, `R_star` = true specular/emissive (redundant
   with the diffuse recon, harmless at 0.5). The saturated-pixel guard
   (`sat_ok = rgb_max < 0.99`) still protects against pursuing R_star near clipped
   highlights. Together with the sign-prior (overshoot, §4.7), this makes the MID
   reconstruction constraint effectively **two-sided**: `relu(A·S−I)` from shade_sign +
   `|（I−A·S)₊|` from the residual closure ≈ `|I − A·S|`.

4. **Specular gate on MID (typed-S protection)** — user-caught interaction (2026-06-12).  
   On no-GT datasets the implied shading target is `S* = I/A*`, which enters the loss
   only via recon (`A*·S* = I`) and the residual closure (`R* = 0`). Both push the
   **non-clipped specular sheen into diffuse shading** (A is pinned by the specular-free
   pseudo-GT albedo — the sheen has nowhere else to go), contradicting Hypersim where
   true S_d GT routes speculars into the analytic R. Two changes: (a) the recon and
   residual-L1 masks in `V17Loss` now include `loss_mask` (they previously used only
   `sat_ok`, so the loader's masks never protected them); (b) `MIDIntrinsicDataset`
   folds a per-crop **p97-luminance specular gate** into `loss_mask` (deep shadows
   deliberately kept — shadowed-input→full-albedo supervision is the shadow-removal
   signal; CARI ratios are separately guarded by `pair_valid`). Result: the sheen is
   **unsupervised** on MID, Hypersim owns the spec/diffuse semantics, and the chrome-ball
   highlight lands in R at inference — the physically right outcome. The principle:
   `S*=I/A*` is the correct anchor for the diffuse majority; the specular minority needs
   *exclusion*, not a different target (no per-pixel specular GT exists on MID).

5. **Chromatic pair augmentation** (§2.3 — implemented in `MIDIntrinsicDataset`).  
   `chromatic_aug=True` (config `mid_chromatic_aug`, wired through `get_mixed_loader`):
   with p=0.8 tint the **extra** frame only by per-channel gains `c ~ U[0.6,1.4]`
   mean-normalised. The primary frame stays untinted so its supervised albedo loss
   remains consistent with the white-balanced pseudo-GT. Physically exact
   (`c·I = A·(c·S)+c·R`). This is THE fix for the measured Cast_RMS regression
   (+9.2% in row 2): white-flash pairs put a constant cast in `L_inv`'s null-space;
   tinted pairs make the cast *visible* to the loss.

**Ablation row update:** the current 19k→36k run is **row 2** (cross-render pairs only —
no RGB-skip, no chroma loss, no residual gate, no chromatic aug; its artifacts are the
30k/36k checkpoints). **Row 4 = full CARI** = cross-render + all four fixes (RGB-skip,
chroma L1, residual closure, chromatic pair aug), relaunched from 19k. Rows 1 (19k
baseline) and 2 are already banked; intermediate single-fix rows can be filled in later
from the same 19k warm start if the thesis needs them.

**Thesis metric to add:** albedo error binned by GT-saturation decile (now in
`tests/eval/eval_mid_constancy.py`, metric 4). Report as a 4-bin curve: [q0-25%, q25-50%,
q50-75%, q75-100%]. With the red-cloth strip as the qualitative figure: "where does the
red go" becomes one of the most legible result sections — it demonstrates the degeneracy
(three escape routes), the measurements (constancy eval), and the fixes, all on one scene.

---

## 5. Training

**Status: IMPLEMENTED (2026-06-12).** Everything below is as-built, not aspirational.

- **Resume, don't retrain:** start from `checkpoints/v17/checkpoint_iter_19000.pth`
  (global_step 18999). The 19k synthetic checkpoint already has absolute color/scale
  anchored — exactly the supervised warmup the cross-render losses require before `L_inv`
  engages (gray-collapse guard). Architecture is unchanged → loads with zero shape
  mismatches. Launch: `--resume checkpoints/v17/checkpoint_iter_19000.pth`.
- **Curriculum (`v17.yaml`):** P1 [0,15k) Hypersim (done in ckpt) → P2 [15k,19k)
  +InteriorVerse (done in ckpt) → **P3 [19k,50k) CARI**: sampling
  `hypersim 0.4 / midintrinsic 0.3 / interiorverse 0.3`, `use_mid_paired: true`,
  `mid_pair_mode: raw`. Synthetic stays the majority so absolute color is continuously
  re-anchored (prior-drift guard, expert consensus).
- **Batch = mixed, uniform keys:** MID rows carry `(rgb, rgb2, m_invariant=1, pair_valid)`;
  Hypersim/IV rows carry the supervised anchors; **all** datasets emit `pair_valid` so the
  mixed collate never sees ragged batches. The Siamese second forward runs only when paired
  rows are present (`train_one_step`, `src/train_v17.py`).
- **Weights (as set):** `lambda_a 1.0` (absolute-color anchor, dominant) ·
  `lambda_dssim 0.2` (CRefNet-recipe structure term, was 0) · `lambda_msg_albedo 0.5` ·
  `lambda_alb_invariance 0.5` (`L_inv`) · `lambda_explain 0.25` (`L_explain`) ·
  `lambda_shade_sign 0.05` (§4.7, all phases) · `lambda_grad_decorr 0.0` (rejected, §4.6).
- **Pair sampling:** 2 of the 20 usable light dirs per scene (`skip_list` excludes 5
  hard-flash frames); per-pair HDR validity measured ≈ **0.73 valid fraction**. Optional
  upgrades: hard-pairing by probe illuminant distance; chromatic pair augmentation (§2.3 —
  small loader change, do early in P3).
- **24 GB:** frozen DINOv2-L (`no_grad`) + ~5M trainable. The second forward roughly
  doubles decoder/head activations → expect **bs4–6 at 512** (single-pass was bs8 at
  ~21.8 GB); raise `grad_accum_steps` to keep the effective batch. **No diffusion unroll,
  no VAE.**

---

## 6. Evaluation — the claim is measurable

### 6.0 What the field benchmarks on — and where CARI sits (2026-06-14 lit survey)

Read the actual eval protocols of the recent line (not recalled — verified from the papers).
**The field splits into two camps, and CARI's benchmark plan must straddle them because our
claim — colored-illuminant disambiguation — is precisely the axis the two camps each measure
only half of.**

| Method (year, venue) | Trains on | Evaluates on (zero-shot in **bold**) | Albedo-**color** metric |
|---|---|---|---|
| Ordinal Shading — Careaga&Aksoy (TOG'23) | synthetic | **IIW** (WHDR), **ARAP** (LMSE/RMSE/SSIM), **MIT** | ✗ grayscale shading |
| **CD-IID / Colorful** — Careaga&Aksoy (TOG'24) | synthetic | **MAW** (intensity, **chromaticity**), **ARAP** (LMSE/RMSE/SSIM) | ✅ **chromaticity = 3.37 (their headline)** |
| IntrinsicDiffusion — Luo (SIGGRAPH'24) | InteriorVerse | IV test (in-domain) | PSNR/SSIM/LPIPS |
| Intrinsic Image Diffusion — Kocsis (CVPR'24) | InteriorVerse | IV test (in-domain) | PSNR/SSIM on albedo |
| Marigold-IID (2024) | InteriorVerse+Hypersim | IV/Hypersim test splits (**no alignment**) | PSNR/SSIM/LPIPS |
| RGB↔X — Zhang (SIGGRAPH'24) | InteriorVerse+Hypersim | **MAW** + IV/Hypersim | PSNR/SSIM/LPIPS |
| **ReasonX (2025, the closest peer)** | COCO/Hypersim/IV | **IIW** (WHDR10/20), **MAW** (intensity, **chromaticity**), **ARAP** (LMSE/RMSE/SSIM), **MIT-Multi-Illum (QUALITATIVE only)** | ✅ **MAW chromaticity** |
| LumiX (CVPR'26) | synth+text | preference/alignment (generation) | ✗ not a decomposition benchmark |

**Three load-bearing findings, each changing the plan:**

1. **Our headline metric is already a published, named one — adopt it.** What this doc called
   `Cast_RMS` (albedo `[R/G,1,B/G]` drift) IS the **MAW chromaticity metric**. CD-IID makes
   "chromaticity 3.37" its *headline number*; ReasonX reports it too. So the cast claim has a
   **standard benchmark with leaderboard numbers** — we must report **MAW chromaticity**, not
   only our home-grown Cast_RMS, or the central claim has no comparable.

2. **MID-as-a-QUANTITATIVE-benchmark is genuinely novel — but anchor it with MAW.** ReasonX —
   the most recent, most similar work — uses MIT/MID multi-illumination **qualitatively only**.
   *Nobody has turned multi-illumination into a quantitative albedo-constancy benchmark.* That
   makes our MID-constancy benchmark a real contribution (§7). BUT MID albedo is pseudo-GT, so a
   reviewer discounts the *absolute* numbers. The bridge is **MAW** (real *measured* albedo +
   chromaticity GT): **MID carries the mechanism (constancy across lights); MAW validates the
   same cast claim against real GT.** The two together are far harder to dismiss than either.

3. **ARAP is universal; MAW is the missing real-GT benchmark; IIW is the ranking guardrail.**
   ARAP appears in *every* decomposition-camp paper (Ordinal/CD-IID/ReasonX) and we already run
   it. **MAW is the only benchmark with measured physical albedo + a chromaticity metric, used
   by both our direct rival (CD-IID) and the closest peer (ReasonX) — it is the natural external
   home for the cast claim and is currently MISSING from our suite.**

**DECISION — the four-benchmark suite (why each, in-domain vs out-of-domain):**

| Benchmark | Role | In/Out-domain | Domain | Metrics | Status |
|---|---|---|---|---|---|
| **MID-test constancy** (30 scenes ×25 lights) | THE mechanism (novel) | in-domain (we train on MID) → controlled by row 1/6 (§6.1) | **REAL photos** | C_mat, Cast_RMS, sat-binned MAE | ✅ `eval_mid_constancy.py` |
| **MAW** (888 imgs, 46 scenes, real measured albedo) | the cast claim w/ REAL GT | **out-of-domain** (nobody trains on it) | **REAL photos** | **chromaticity**, intensity (×100) — MAW protocol | ⬜ **NEW — must add** |
| **ARAP** (incl. `_lightN`) | universal guardrail + OOD constancy | **out-of-domain**, EXACT albedo GT | **SYNTHETIC render** | LMSE/RMSE/SSIM (WB protocol) + C_arap constancy — **reported `indoor` AND `all`** | ✅ `eval_arap.py --scene_filter` |
| **IIW** (WHDR) | ranking no-regression | **out-of-domain** | **REAL photos** | WHDR 10%/20% | ✅ `eval_iiw.py` |
| **Beigpour MIID** (5 scenes, 6+9 illum conds, **colors orange→blue**) | **the COLORED-illuminant axis** | **out-of-domain** | **REAL photos, exact GT** | RCI / C_mat + Cast_RMS across light COLORS | ⬜ **NEW — the headline-axis probe** |

**The four AXES of illumination change (the eval backbone, 2026-06-15):** ARAP and MID-direction
vary light by **direction+intensity**; the headline thesis claim is invariance to light **COLOR**,
which NEITHER directly tests (ARAP shading is colorful per-scene but does not give one scene under
multiple light *colors*). **Field survey: colored-illuminant albedo constancy is UNDER-MEASURED
even in 2024 SOTA — IDArb (arXiv:2412.12083) trains under varied light but reports albedo only as
scale-invariant PSNR/SSIM, NOT cross-illuminant consistency.** The one work naming the gap is
Sato et al. *Self-Supervised IID Considering Reflectance Consistency* (arXiv:2111.04506), which
proposes the **Reflectance Consistency Index (RCI)** — the citeable precedent for our C_mat/Cast_RMS.
**Beigpour MIID** (Univ. Siegen; ICCV'15 single-view + BMVC'16 multi-view) is the dedicated
colored-light benchmark: same real scene under illuminant colors orange→blue with per-pixel
reflectance GT → directly measures "does albedo stay fixed as light COLOR changes." Add it (small,
targeted probe) + MAW (real measured chromaticity anchor) to cover the COLOR axis. *That this axis
is barely benchmarked turns our eval into a minor contribution, not just a number.*

**The real-vs-synthetic axis (verified 2026-06-14 — runs OPPOSITE to GT quality):**
ARAP (Bonneel'17) is **synthetic** ("As *Realistic* As Possible" = render quality, not real
capture) → its albedo GT is *exact* precisely because it is rendered. MID & MAW are **real
photographs** → their albedo is pseudo/measured precisely because true albedo of a real scene
can't be rendered. This is why both belong: **MID = real-domain constancy (the hard case, where
real sensor noise / real colored bounce-light make the ambiguity bite); ARAP = exact-GT,
out-of-domain cross-check; MAW = real-domain + real measured color GT.** "Real-domain constancy
(MID) + real-GT color (MAW) + synthetic-exact-GT cross-check (ARAP)" — three complementary axes.

**DECISION — ARAP is split by CONTENT DOMAIN, reported `indoor` AND `all` (2026-06-14):**
CARI trains ONLY on indoor scenes (Hypersim rooms + InteriorVerse rooms + MID real rooms), so
benchmarking on ARAP's outdoor/animals/human scenes (e.g. `alley`, an outdoor alley that scored
si-RMSE 0.62) is an out-of-domain stress test, not the fair comparison. `tests/eval/eval_arap.py
--scene_filter {all,indoor}` reads `tests/testing_data/ARAP_types.json` (corrected 2026-06-14:
fixed `livingroom`/`skyscraper` name bugs; the old `LDR` bucket was a *format* group, not a
domain — re-tagged by content). **`indoor` = Interiors + Objects + architectural-interior LDR
scenes (sponza/sanmiguel/cathedral).** `all` = +Outdoor/Animals/human. **The `indoor` table is
the headline (fair, in-domain); `all` is the honest harder number.** Objects are included as
indoor (tabletop items photographed indoors, present in the training scenes). JSON keys carry a
`__indoor` suffix so both runs coexist.

**Counts differ by METRIC (single-frame scenes have no constancy) — exact, verified 2026-06-14:**

| Split | Standard albedo (si-RMSE/LMSE/SSIM, WB — block 3) | Constancy (C_arap/Cast_RMS — block 2) |
|---|---|---|
| **indoor** (headline) | **30 scenes / 95 cases** | **23 groups / 87 frames** |
| **all** (harder) | **51 scenes / 157 cases** | **37 groups / 142 frames** |
| outdoor (context) | 21 scenes / 62 cases | 14 groups / 55 frames |

Standard-albedo counts every scene with albedo GT (single-frame included). Constancy needs ≥2
lights per scene, so single-frame scenes — incl. the LDR interiors sponza/sanmiguel/cathedral —
are **auto-excluded** by the `len(group)≥2` filter (*constancy of one frame is undefined* — correct,
not a bug). The drop is 51→37 (`all`) and 30→23 (`indoor`): 14 single-frame scenes dataset-wide.
The constancy JSON's `n_groups` already reports the true group count. So sponza/sanmiguel/cathedral
being tagged `indoor` only affects the standard-albedo table (where they belong, having valid
albedo GT); they never enter the constancy claim. **Report the headline as: constancy on 23
indoor groups (87 frames) + 37 all groups (142 frames); standard albedo on 30 indoor / 51 all.**

**DECISION — external SOTA ALSO compared on MID, GT-free metrics only ("Job 2", 2026-06-14):**
Because MID is the *realistic-domain* benchmark, the Marigold/CD-IID comparison runs there too —
but **only on C_mat + Cast_RMS, which are GT-free**. The pseudo-GT columns (LMSE, sat-binned MAE)
are home-field for CARI (we trained on that pseudo-GT) and are **auto-blanked ("—") for external
rows** by `eval_mid_constancy.py` (the `AlbedoPredictor.is_external` flag). The discipline:
*never present an external model's MID LMSE as a fair comparison.* With that one rule, the
realistic-domain SOTA comparison is a highlight, not a liability — CARI-vs-Marigold on MID
constancy/cast complements the same comparison on MAW (real GT) and ARAP (exact GT).

*Dropped from the plan:* SAW (superseded by MAW for the color claim; keep only if a reviewer
asks for shading-boundary recall). MIT-Intrinsic (object-level, dated; ARAP is the modern scene
equivalent everyone reports). We do **not** need to follow any single paper's exact suite —
our path (a *new* multi-illumination constancy benchmark) is the differentiator — but MAW + ARAP
+ IIW is the **intersection** of what CD-IID and ReasonX both report, so omitting them invites
the "non-standard evaluation" rejection.

**DECISION — models to compare against (beyond Marigold-IID appearance/lighting):**

| Model | Why it MUST be in the table | Checkpoint availability |
|---|---|---|
| **CD-IID / Colorful** (Careaga&Aksoy'24) | **THE direct rival** — owns the chromaticity metric, the only prior that explicitly does *colorful* diffuse shading; if CARI beats it on cast-with-real-GT (MAW) that is the result | public, in `documents/references/` repo (`compphoto/Intrinsic`) — already cloned per §4.6 cites |
| **Ordinal Shading** (Careaga&Aksoy'23) | the grayscale-shading predecessor — shows the colored-cast axis CD-IID/CARI add over it; same repo, one flag | public, same repo |
| **IntrinsicDiffusion** (Luo'24) | the diffusion-camp albedo baseline ReasonX & SAIL both cite; generative-prior reference point | public (`JundanLuo/IntrinsicDiffusion`) |
| Marigold-IID (appearance + lighting) | already wired in `eval_arap.py`; the generative SOTA we already report | local checkpoints ✓ |
| ReasonX (2025) | the closest peer (MLLM-judge route to the SAME goal) — cite as related-work contrast; include numbers IF a checkpoint releases | check for release; else cite reported numbers |

**Metric-by-claim mapping (so every table column has a purpose):**
- **Colored-cast claim** → MAW **chromaticity** (real GT, external) + MID Cast_RMS (in-domain) +
  ARAP Cast_RMS (OOD pseudo→true). Three datasets, one claim.
- **Constancy claim** → MID C_mat (in-domain, novel) + ARAP C_arap `_lightN` (OOD, true GT).
- **No-regression** → IIW WHDR + ARAP LMSE/RMSE/SSIM (WB protocol) + MAW intensity (must not
  worsen while chromaticity improves — the CD-IID-style intensity/chromaticity trade-off check).
- **Physicality** → MID cross-render relighting PSNR/SSIM (the defense figure, §6 below).

> **NOTE — MAW metric formulas:** chromaticity = albedo color error over MAW's homogeneous
> measured-albedo regions (per-region mean RGB, scale-aligned, then color distance); intensity =
> scalar-aligned magnitude error ×100. Lift the EXACT formula + region masks from MAW's released
> eval code (`measuredalbedo.github.io`, arXiv:2306.15662) when implementing `eval_maw.py` —
> do not re-derive, so the number is directly comparable to CD-IID's 3.37 and ReasonX's table.

**Primary (the contribution):**
- **Cross-illumination albedo constancy** on **held-out MID test scenes** (30): variance of
  predicted albedo across the 25 light dirs within a material region (lower = better).
  Report CARI vs V17(no cross-render) vs Marigold vs V18. *This number is the thesis.*
- **Albedo color-cast** (the `[R/G,1,B/G]` mid-tone ratio drift vs neutral, the
  `THESIS_design.md §2.2.1` metric) — show CARI's cast < V18's ~50%.
- **Cross-render relighting recomposition (free "wow" + a number):** on held-out MID pairs,
  recompose `A(I_a) · S(I_b)` and score PSNR/SSIM against the real `I_b` (HDR-mask-aware).
  Quantifies that the decomposition is *physical* (albedo transfers across lighting) and
  produces the defense-day figure. No new training — one eval script.
- **OUT-OF-DOMAIN constancy — ARAP `_lightN` (the fairness keystone):** ARAP ships
  multi-light variants of the same scene sharing one albedo GT, and
  `tests/eval/eval_arap.py::get_arap_cases` **already parses** `scene_light0, scene_light1, …`.
  Constancy measured there is (a) out-of-domain for **every** method including CARI, and
  (b) backed by TRUE albedo GT (unlike MID's pseudo-GT). If CARI's constancy gain transfers
  from MID-test to ARAP-light, the mechanism claim survives the in-domain objection (§6.1).

**External real-GT benchmarks (the credibility bridge — §6.0):**
- **MAW** (intensity ×100, **chromaticity**) — real *measured* albedo, the standard home of the
  cast claim (CD-IID's 3.37, ReasonX's table). `tests/eval/eval_maw.py` ⬜ to write; lift the metric
  formula from MAW's released code so the number is directly comparable.
- IIW (WHDR 10/20), ARAP (LMSE/RMSE/SSIM) — `tests/eval/eval_arap.py`/`eval_iiw.py` support the V17
  path. (SAW dropped — superseded by MAW for the color claim; MIT-Intrinsic dropped — ARAP is
  the modern scene-level equivalent every recent paper reports.)

### 6.1 Fairness protocol — "we train on MID; is benchmarking zero-shot SOTA on it unfair?"

Decided protocol (the examiner WILL ask this):
1. **Report external baselines (Marigold-IID, CD-IID/Colorful, IntrinsicDiffusion, Ordinal
   Shading) as released, labeled ZERO-SHOT, with a "trained on MID?" column in every table.**
   This is the field norm and the exact protocol of the closest prior work: SAIL trains on MID
   and reports IntrinsicDiffusion / RGB↔X zero-shot on MID-test; **ReasonX (2025) reports
   Careaga&Aksoy / Kocsis / RGB↔X zero-shot on MAW+ARAP+IIW** (§6.0). **The in-domain MID
   objection is defused by also reporting the SAME cast claim on MAW — real measured albedo GT
   that NOBODY trains on — so the headline (chromaticity) lives on an external real-GT benchmark,
   not only on MID-test.**
2. **Do NOT fine-tune external SOTA on MID.** (a) For diffusion baselines it is
   *ill-defined*: their training formulations have no mechanism that consumes aligned
   pairs, so "fine-tune on MID" degenerates to "use MID pseudo-albedo as a target" — which
   measurably degrades generative priors (the documented reason V18 dropped MID). (b) Any
   pair-consuming variant we invent for them is a new method of OUR design — the comparison
   becomes contestable in the opposite direction. Fine-tuning baselines on data they cannot
   properly consume is not fairness; it is a strawman with extra steps.
3. **The domain-exposure control is ablation row 6:** same backbone, same MID pixels, only
   the *supervision* differs (pseudo-albedo-as-TARGET vs cross-render-as-INVARIANCE). This —
   not the zero-shot table — isolates the mechanism from domain familiarity. Together with
   row 1 (V17, no MID at all) it carries the thesis claim.
4. **ARAP `_lightN` transfer** (above) is the out-of-domain check where no method has any
   advantage.

The headline claim rests on the CONTROLLED comparisons (rows 1/6 + CARI); the zero-shot
SOTA table is context, never the claim.

### 6.2 Optional Phase-4 — the benchmark-competitive variant (not the contribution)

If a leaderboard-adjacent WHDR is wanted: fine-tune with the IIW-train ordinal hinge loss
(the CGIntrinsics/CRefNet recipe; fine-tuned CRefNet sits at ≈10.8% WHDR and **trains on
IIW labels**). Report as a clearly separated "CARI + IIW-ft" row. Without it, CARI claims
*no-regression* on WHDR plus the constancy/cast deltas where the comparison is controlled —
a mechanism claim, not a leaderboard claim.

**Ablations (each isolates one piece):**
| Row | Variant | Tests |
|---|---|---|
| 1 | − cross-render (= plain V17) | the whole contribution |
| 2 | + `L_inv` only | does invariance alone help (or collapse to gray)? |
| 3 | + `L_inv` + `L_explain` | the full split signal |
| 4 | + illumination-equivariant FiLM | the architectural claim |
| 5 | per-pixel vs per-material pooling | the material-mask value |
| 6 | MID albedo as target (old way) vs as invariance (ours) | the **domain-exposure control** (§6.1) — *and* why the old docs were wrong to drop MID |
| 7 | raw measured pairs vs synthesized pairs (`mid_pair_mode`) | the "real measured signal" claim (§2.2) |
| 8 | ± chromatic pair augmentation | the colored-cast axis (§2.3) |
| 9 | ± shading sign prior | the data-free half-space constraint (§4.7) |
| 10 | α-gate / color-skip / normal-conditioning (§3.1) | the architectural levers, each vs loss-only CARI |

---

## 7. Why this is academic, not engineering

- **A falsifiable claim with a number** — "single-image colored-light ambiguity is partly
  resolvable by cross-render supervision; albedo cast ↓ from ~50% to X%, constancy ↑ Y%."
- **Novel use of existing data** — MID-as-invariance-signal for single-image IID is, to our
  knowledge, unpublished; the field uses multi-illumination only for multi-image methods or
  discards it.
- **A method transfer with a name** — GS-2M's multi-view material disambiguation → 2D
  feed-forward single-image. The *transplant* is the idea.
- **Not a fine-tune of someone's model** — clean transformer, our loss, our data framing. The
  SD/diffusion optics you disliked are gone.
- **Benchmark expectation, managed:** the claim is a MECHANISM with a controlled number, not
  a leaderboard. Fine-tuned CRefNet (≈10.8% WHDR) trains on IIW's own labels; CARI never
  sees IIW and claims no-regression there + the constancy/cast/relighting deltas where the
  comparison is controlled (rows 1/6) and out-of-domain (ARAP `_lightN`). §6.2 provides the
  competitive variant if a leaderboard number is wanted.

---

## 8. Risks & honest limits

| Risk | Mitigation |
|---|---|
| `L_inv` collapses albedo to a constant | supervised absolute-albedo anchor (§4.3) is mandatory, ramped first |
| MID is real photos (sensor noise, no perfect albedo) | we never trust MID's albedo *value*; only same-pixel-same-material structure, which is exact |
| MID→Hypersim domain gap | mixed batches + DINOv2 frozen backbone bridges domains (its original V17 rationale) |
| Single image still can't resolve *fully* flat colored scenes | honest ceiling — we **quantify** how much is recoverable, which is itself the result (cf. GS-2M never claims single-view) |
| MID light is directional (point-ish), Hypersim is area | `L_explain` is ratio-based, robust to absolute light type; report per-dataset |
| MID flashes are WHITE + probe-WB → raw pairs carry little illuminant-COLOR variation, yet the headline failure is the colored cast | chromatic pair augmentation (§2.3): physically exact (`c·I = A·(c·S)+c·R`), turns real direction-pairs into color-pairs too |
| In-domain advantage on MID-test ("you trained on MID") | fairness protocol §6.1: zero-shot labeling + row-6 domain-exposure control + ARAP `_lightN` out-of-domain transfer with TRUE albedo GT |

---

## 9. Build plan (concrete, in priority order)

1. ✅ **DONE — go/no-go verification** (`tests/verify_mid_crossrender.py`): albedo-texture
   correlation **0.51 (single frame) → 0.14 (pair ratio)**; `alb_sanity = 0.0000` (perfect
   pixel alignment + truly shared albedo); valid fraction ≈75% after the HDR mask. **GO.**
2. ✅ **DONE — MID raw-pair loader**: `MIDIntrinsicDataset(use_paired=True,
   pair_mode='raw')` + `_hdr_valid_mask` (linear HDR, specular+shadow guards), threaded
   through `prepare_training_tensors` → `pair_valid`; all datasets emit uniform keys.
3. ✅ **DONE — losses**: `cari_albedo_invariance` (`L_inv`), `cari_explain` (`L_explain`),
   `cari_shade_sign` (§4.7); Siamese second-forward hook in `train_one_step`; TB tags.
   **+ Three red-color fixes (§4.8, 2026-06-12):** `albedo_rgb_skip` arch flag (A2),
   `lambda_a_chroma` chroma-direction L1, MID `m_residual=1.0` + `lambda_res_sparse=0.02`.
4. ✅ **DONE — config** (`v17.yaml`): resume-from-19k curriculum (P3 = 19k→50k),
   `lambda_alb_invariance 0.5 / lambda_explain 0.25 / lambda_shade_sign 0.05`.
   **Current ablation-row-2 run** has `albedo_rgb_skip: false, lambda_a_chroma: 0.0`
   (cross-render only). **Full-CARI relaunch (row-4):** flip both to `true` / `0.2`.
5. ✅ **DONE — eval script** (`tests/eval/eval_mid_constancy.py`): 4 metrics across 30 test
   scenes vs 2 checkpoints — C_mat (within-material constancy), R/G+B/G cast ratio, SSI
   LMSE vs GT, saturation-binned MAE. Run and compare 19k vs 30k before the relaunch.
6. ✅ **DONE — 19k vs 30k evaluated** (full 30 scenes, `documents/
   eval_mid_constancy_results.md` + `.json`): **C_mat 0.1573→0.1156 (−26.6%, 28/30
   scenes improved) — L_inv CONFIRMED working.** Cast_RMS +9.2% (expected: white-pair
   null-space → fix = chromatic aug), LMSE +21% (drift from noisy pseudo-GT — the
   correct trade; verify externally on ARAP true-GT). Contactsheets:
   `tests/visualizations/contactsheet_19k_vs_30k/`.
7. ⬜ **Full-CARI relaunch from 19k (row 4)** — config is READY in `v17.yaml`
   (`albedo_rgb_skip: true`, `lambda_a_chroma: 0.2`, `lambda_r: 0.5`,
   `mid_chromatic_aug: true`). Stop the row-2 run (36k is enough), then:
   `python src/train_v17.py --version 17 --config src/configs/v17.yaml \
    --resume checkpoints/checkpoint_v17_iter_19000.pth --skip-optimizer` → 50k.
   Expect `[load] reinitialising 1 shape-mismatched tensor` (albedo_head.refine widens).
8. ⬜ **Quick external de-risk (parallel, cheap)**: `tests/eval/eval_arap.py` on 19k vs 30k —
   ARAP has TRUE albedo GT, so if the LMSE regression were real degradation (not
   pseudo-GT noise) it would show there. Protects the narrative before the long run.
9. ⬜ **Eval suite** (four benchmarks, §6.0): MID constancy/cast/relighting (in-domain,
   novel); **MAW** intensity+chromaticity (`tests/eval/eval_maw.py` — out-of-domain REAL GT, the
   cast claim's external home); ARAP (+`_lightN`, OOD true GT); IIW WHDR — tables per §6.1.
   Compare vs Marigold-IID + **CD-IID/Colorful** (direct rival) + IntrinsicDiffusion + Ordinal.
10. ⬜ **Ablations**: rows 1/2/4 are then banked; fill 3 + 5–10 as time allows;
    optional §6.2 IIW-ft variant.

**One-line:** *CARI = a single-image intrinsic transformer taught illumination invariance by
cross-render supervision mined for free from multi-illumination data — GS-2M's multi-view
material disambiguation, transplanted into a feed-forward 2D model, fixing the colored-light
ambiguity that every single-image method (incl. Marigold/V18) hides.*

---

## 10. Thesis plan (defense-oriented)

**Title.** *Resolving the Albedo–Shading Ambiguity in Single-Image Intrinsic Decomposition
via Cross-Illumination Supervision.*

**Chapter arc (maps 1:1 onto the grading rubric):**
1. **Introduction & measured diagnosis** — the split degeneracy: product-only training
   losses leave the A/S split underdetermined; the SAME failure measured across three
   paradigms (V12 CNN, V17 frozen transformer, V18 one-step diffusion): desaturation,
   shadow leak, ~50/50 colored-cast split. Source material already drafted:
   `LITREVIEW_v12_v17_v18.md`. *(The "failed" versions become the motivation chapter —
   measured, not surveyed. No prior work is wasted.)*
2. **Related work** — the honest-lineage table (§1.1) + the benchmark survey (§6.0): BigTime /
   CGIntrinsics / SAIL / Latent-Intrinsics / Marigold-IID / **CD-IID (the direct chromaticity
   rival)** / IntrinsicDiffusion / Ordinal-Shading / ReasonX, and the precise slot CARI occupies
   (the only one turning multi-illumination into a *quantitative* constancy benchmark, §6.0).
3. **Method** — physical colorful-diffuse split, `L_inv`/`L_explain`/sign-prior, HDR pair
   validity, chromatic augmentation, curriculum; §3.1 architecture levers as extensions.
4. **Experiments** — controlled rows first (row 1 baseline / row 6 domain-control / CARI),
   zero-shot context tables under the §6.1 fairness protocol, relighting figure, ARAP
   `_lightN` transfer, ablations 1–10.
5. **Discussion** — how much of the ambiguity is recoverable (the quantified ceiling),
   where it saturates (flat colored scenes), efficiency vs diffusion baselines
   (one pass, ~5M trainable, 24 GB).
6. **Limitations & future work** — §8 verbatim honesty; §6.2 IIW-ft, §3.1 α-gate, and
   MLLM-judge ordinal supervision as future signal sources.

**Timeline (single 24 GB card):** smoke-test + chromatic augmentation (≈1 day) → P3
training (≈3–5 days) → eval suite + relighting figure (≈2–3 days) → ablation rows 1–3
(≈1 week) → rows 4–10 + writing in parallel. The diagnosis chapter exists; the related-work
table exists; the implementation is done.

**The thesis in one sentence (for the abstract and the defense):** a *named, measured*
degeneracy; a *controlled* mechanism that closes part of it; *honest* lineage, fairness
protocol, and ceiling.
