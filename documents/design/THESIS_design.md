# Thesis Design — Physics-Grounded Intrinsic Decomposition and IID-Based Image Enhancement

> **Status:** Design (2026-06-09). Two-part thesis. **Part 1** = V18, a physics-grounded
> one-step colorful-diffuse intrinsic decomposition (the *method*, built & trained to 56k,
> measured working on 21 real images). **Part 2** = an IID-based **automatic image enhancement**
> pipeline — white balance, color accuracy, lighting adjustment — built on V18's decomposition
> (the *application*, the GS-2M-style "apply the tool to a new matter").
>
> Calibrated against GS-2M (Eurographics 2026): a working baseline + a few well-motivated
> mechanisms + standard benchmarks beats a from-scratch architecture. We do not invent a
> backbone; we contribute the physics coupling (Part 1) and the enhancement formulation (Part 2).

---

## PART 1 — V18: Physics-Grounded One-Step Colorful-Diffuse IID

### 1.1 What it is

V18 fine-tunes a Stable-Diffusion UNet as a **single-step** feed-forward predictor that jointly
outputs colorful albedo `A` and inverse-domain diffuse shading `π=1/(S_d+1)`, with the
non-diffuse residual derived **analytically** as `R=(I−A·S_d)₊`. Warm-started from
Marigold-IID-Lighting via channel surgery (`scripts/marigold_iid_to_v18_init.py`), then trained
**end-to-end with image-space physics losses** — the thing multi-step Marigold-IID structurally
cannot do. (Full architecture/formulation: `documents/design/V18_PGID_design.md`.)

### 1.2 The contribution — and exactly where/why it beats Marigold

Marigold ships **two** IID models, each with a documented weakness our single model fixes:

| | **Marigold-IID Appearance** | **Marigold-IID Lighting** | **V18 (ours)** |
|---|---|---|---|
| Targets | albedo, roughness, metallic | albedo, **shading, residual** | albedo + **π-shading**; R analytic |
| Train data | InteriorVerse (45K) | Hypersim (24K) | Hypersim + InteriorVerse, **one model** |
| Steps | 4-step ε/v diffusion | 4-step ε/v diffusion | **1 step** (≈4–50× faster) |
| Loss | **latent MSE only** | **latent MSE only** | **image-space physics suite** |
| `I=A·S_d+R` enforced? | ✗ | ✗ (R generated yet never tied to `I`) | **✓ exact (analytic R)** |
| Cross-task coupling | ✗ | ✗ | **✓ two-sided diffuse recon `A·S_d≈A_gt·S_d_gt`** |
| Stochastic? | yes | yes | **deterministic** one-step (repeatable) |

**Four concrete "better than Marigold" claims:**

1. **Physics in the loop (core novelty).** Marigold-IID's loss is `mse_loss` *in latent space*;
   it never decodes during training, never enforces reconstruction. Because V18 is one-step, we
   decode `A, S_d` every iteration and apply albedo MSE+MSG+DSSIM, SSI shading, and a **two-sided
   diffuse-reconstruction** loss in *image space*. The decomposition is physically consistent
   (`I=A·S_d+R` by construction), not just per-channel-plausible. *No diffusion-IID does this.*
2. **One unified model vs. two.** Appearance and Lighting are separate checkpoints with disjoint
   targets; you cannot get `(albedo, shading, residual)` + consistency from one pass. V18 does.
   (Directly enables Part 2's shared-backbone enhancement — §2.4.)
3. **Analytic residual = zero wasted capacity + exact recon.** Marigold-Lighting spends a 4-ch
   generative target on R and *still* doesn't guarantee `A·S+R=I`. V18 derives R for free.
4. **Deterministic & fast.** One-step x0 (zero-terminal-SNR + v-pred, the E2E-FT/Lotus recipe):
   repeatable, one pass, no ensembling. Optional multi-step stochastic mode for uncertainty.

**Not claimed:** the SD-prior template and latent-concat interface are borrowed (Marigold/E2E-FT).
The contribution is the physics coupling, analytic-residual unification, and one-step enabling
argument — not the backbone.

### 1.3 Measured status (evidence, not aspiration)

- Trained to **56k iters** (`checkpoints/v18/checkpoint_latest.pth`, step 55999).
- **21 ARAP images** (`tests/v18_arap_contactsheet.py`): albedo colored (chroma-std 0.011–0.126,
  no gray collapse), shadows flatten into shading, highlights → residual, reconstruction holds.
- Weakest on dim **low-key** scenes (Lobby) — achromatic-lightness ambiguity ceiling
  ([[ccr-colored-illumination-limit]]); honest limitation, not a bug.

### 1.4 Part-1 evaluation (to do)

- **Quantitative IID:** ARAP (albedo+shading LMSE / si-RMSE / SSIM), IIW (**WHDR**), SAW.
  *Note:* `tests/eval/eval_arap.py` is hardcoded for the old ConvNeXt models (expects `normals/seg/ccr`,
  reads `d_g`) — a **V18-compatible eval must be written** (RGB-only, `model.sample`,
  reads `a_d`/`shading_linear`).
- **Compare to:** Marigold-IID-Lighting & -Appearance (direct baselines), CD-IID (TOG'24), CRefNet.
  Report speed (1 pass vs 4-step) alongside accuracy.
- **Ablations (Part 1):** physics-loss on/off (the core claim); analytic vs generated R; one-step
  vs multi-step; one unified model vs Appearance+Lighting separately.

---

## PART 2 — IID-Based Automatic Image Enhancement

### 2.1 The idea and why it is the right "new matter"

Albedo/shading is the *physically correct intermediate* for editing illumination. Once an image
is decomposed, classic enhancement tasks become principled edits of the right layer, then
recompose `I' = A'·S_d' + R'`. Editing the *entangled* image (Lightroom curves, global WB) cannot
do this cleanly because albedo and shading are mixed; editing in IID space is local, physical, and
material-preserving by construction.

### 2.2 Prior art — what is taken, and the precise open slot

| Work | What it does | Why it does NOT block us |
|---|---|---|
| Bonneel 2017, *Intrinsic Decompositions for Image Editing* | editing on **classical** IID: logo/shadow removal, texture, wrinkles | classical (weak) decomposition; **not** WB/color-cast/lighting-equalization; not diffusion |
| CCR (Aksoy TOG'24) | colorful-diffuse decomposition; **lists** WB/relight as uses | stops at the maps — **no enhancement pipeline built** |
| **Physically Controllable Relighting** (Aksoy, SIGGRAPH'25) | full 3D relight: mesh + path-tracer + neural renderer | **heavyweight, artistic relighting**; not automatic single-image *correction* |
| ScribbleLight 2024 | indoor relight from **user scribbles** | interactive, not automatic |
| Afifi *Mixed-Illuminant AWB* (CVPR'22) | WB via blending pre-rendered WB versions | image-space blend, **no decomposition**; our direct WB competitor |
| **Retinexformer** (ICCV'23, NTIRE'24/'25/'26 winner) | low-light via Retinex (reflectance × illumination) | a *learned 2-layer* split, not a grounded colorful-diffuse physical model; validates "decompose-then-enhance" |

**Open slot (defensible):** *automatic, single-image **enhancement/correction** (white balance,
color accuracy, lighting equalization) on a **modern diffusion-IID** decomposition, no 3D, no
path-tracer, no user input.* Frame as **automatic correction** — not "editing" (Bonneel) and not
"relighting" (Aksoy). **Window/highlight pull is descoped** from the core (application-side bonus,
not what benchmarks measure); revisit as a v2 generative extension (§2.7).

### 2.2.1 MEASURED reality check — the cast is in BOTH layers (this reshapes the plan)

Direct measurement on V18 (mid-tone WB ratios `[R/G, 1, B/G]`):

| Scene | Input I | **Albedo A** | Shading S_d |
|---|---|---|---|
| Kitchen (strong warm) | `[2.64, 1, 0.40]` | **`[1.50, 1, 0.70]`** | `[1.53, 1, 0.73]` |
| School_corridor | `[1.55, 1, 1.06]` | **`[1.07, 1, 0.90]`** | `[1.18, 1, 0.87]` |
| Bedroom | `[1.14, 1, 0.96]` | **`[1.01, 1, 0.96]`** | `[1.07, 1, 0.96]` |

**The color cast splits ~50/50 between albedo and shading — it does NOT all go to shading.** This
is the [[ccr-colored-illumination-limit]]: colored illumination is single-image-ambiguous, so the
model hedges. **Consequence: white balance is (also) an ALBEDO problem.** A scheme that freezes
albedo and edits only shading *cannot* fully white-balance — it locks the residual cast into
albedo and re-entangles color into shading. (This refuted the initial "freeze albedo, edit shading
for everything" proposal; corrected record kept here.)

### 2.2.2 Per-layer task routing (the corrected core)

Each task is routed to the intrinsic layer that *physically owns the error*:

| Task | Layer edited | Why | Freeze-albedo OK? |
|---|---|---|---|
| **White balance / color accuracy** | **ALBEDO chroma** (+ shading chroma) | illuminant color leaks into both; true fix = restore material color | **NO** — albedo must change |
| **Lighting adjustment** (flatten gradients, equalize walls/ceilings) | **SHADING luminance** | pure illumination geometry; albedo genuinely invariant | **YES** — freeze-albedo idea correct *here* |
| **Exposure / brightness** | **SHADING scale** | global illumination gain | YES |
| *(descoped)* Window/highlight pull | **RESIDUAL** | non-diffuse energy | N/A |

**The freeze-albedo/edit-shading instinct is right for lighting & exposure, wrong for WB.**
A single uniform training scheme across all tasks was the error.

### 2.3 Datasets — training & evaluation

| Purpose | Dataset | Role |
|---|---|---|
| Part-1 IID supervision | **Hypersim** (diffuse `A,S_d` GT), **InteriorVerse** (clean albedo) | albedo + SSI shading + diffuse recon |
| **Synthetic enhancement pairs (primary eval + sup)** | **Hypersim/IV re-rendered** under degraded illumination (color cast / uneven light) from GT shading; GT = clean render | the **quantitative** escape from "looks nicer" — true paired GT |
| **White balance** | **Rendered-WB Set1/Set2, Cube+** (Afifi); **Multi-Illuminant 16K** (2025) | field-standard WB benchmark + targets |
| Color/exposure aesthetic | **MIT-Adobe FiveK** (5K expert-retouched) | aesthetic targets (weak physical GT — user-pref/no-ref only) |
| Low-light / lighting | **LOL-v1/v2** (paired low/normal) | lighting-adjustment + Retinexformer comparison |

### 2.3.1 Benchmark protocol — grounded in the field's current standard

*(Researched 2026-06; the metrics reviewers expect — do not invent our own.)*

- **White balance:** **ΔE2000, Mean Angular Error (MAE), MSE**, reported as **mean + Q1/Q2/Q3
  (quartiles)** on Rendered-WB Set1/Set2 & Cube+. Compare to **Afifi Deep-WB**, **"Revisiting
  Image Fusion for Multi-Illuminant WB" (2025)**, **WB-LUTs**. *Angle:* does WB *in IID space*
  (albedo+shading chroma) beat image-space blending on **mixed-illuminant** scenes — the case
  Afifi explicitly struggles with, where a decomposition should win?
- **Lighting / low-light:** **PSNR, SSIM, LPIPS** (paired, LOL-v1/v2); **NIQE / no-reference**
  for unpaired. Compare to **Retinexformer** (decomposition-based NTIRE winner), Zero-DCE.
- **Synthetic relight-pairs (ours, primary):** **PSNR, ΔE2000** vs the clean GT render — the
  metric Bonneel and most IID-editing work *lacked*. Itself a contribution.
- **Decomposition-matters ablation:** "decompose with **Marigold** + same edit" vs ours — shows
  the gain is from *our* decomposition quality, not just the edit formulation.

### 2.4 Training scheme — corrected (per-layer, one shared model)

**Goal (now properly justified):** *one* model that decomposes **and** enhances. Only possible
because Part-1 is **one unified model** producing `[A | π]` from a shared backbone — Marigold's
two checkpoints structurally cannot.

**Architecture:** an **enhancement head** predicting *edited* latents. **WB edits the albedo
latent `z_A'`; lighting edits the shading latent `z_π'`** — not a single shading-only head.
Recompose `I_enh = A'·S_d' + R'` (R recomputed — §2.6 #1).

**Three-phase training (the two-phase idea, fixed for the WB/albedo finding):**

- **Phase A — decomposition (Part 1).** Train V18 as today. Trustworthy `A, S_d, R`.
- **Phase B1 — albedo WHITE-BALANCE correction (NEW; the fix the original plan missed).** Learn
  `z_A' = WB(z_A)` neutralizing the leaked illuminant cast in albedo. Losses:
  1. **WB target** — Afifi/synthetic pairs: `ΔE2000(A', A_neutral_GT)`.
  2. **Self-sup neutrality** — push *chromaticity* of `A'` toward gray-world / a
     known-neutral-region white point (cleaner on albedo than on `I` — shading removed).
  3. **Structure preservation** — `A'` keeps `A`'s luminance/texture, edits only chroma
     (per-region color transform, not a repaint).
- **Phase B2 — lighting adjustment, albedo (now CORRECTED) FROZEN.** Freeze `A'`; train `z_π'`:
  4. **Lighting target** — synthetic pairs `L(S_d', S_d_clean)`; or LOL paired target.
  5. **Geometry-gated smoothness** — TV/low-freq penalty on `luminance(S_d')` to equalize uneven
     illumination, **gated by albedo/normal edges** so real geometry/material boundaries survive
     (the non-trivial part — not a global curve).
  6. **Albedo-invariance** — *now meaningful*: `L1(A(I_enh), A')` (re-forward enhanced image,
     assert albedo unchanged). Keeps the lighting edit from re-introducing color.
  7. **Recon validity** — `I_enh = A'·S_d' + R'` in-gamut (§2.6 #1–2).

**Why three phases, not two:** WB must fix albedo *before* it is frozen for the lighting stage.
"Freeze albedo first, then shading" would freeze the *cast-contaminated* albedo → WB impossible.

### 2.5 Open questions for Part 2

1. **WB edit: learned latent head vs. parametric color transform?** Learned `z_A'` reuses the SD
   prior; parametric per-region white-point correction is more interpretable. **Likely both** —
   learned head *regularized toward* the parametric WB target.
2. **WB target on real un-paired photos?** Afifi gives paired GT; otherwise self-supervised
   illuminant estimation **on albedo/shading chroma** (cleaner than on `I`).
3. **Aesthetic ambiguity** (candle-lit: neutralize vs. preserve warmth?) — provide a WB
   *strength* control; evaluate physical correction (synthetic/Afifi GT) and aesthetics (FiveK)
   separately.

### 2.6 Pipeline drawbacks NOBODY has addressed (be honest)

1. **The analytic residual is STALE after any edit.** `R=(I−A·S_d)₊` is from the *original* `I`.
   After editing to `A'·S_d'`, naive `A'·S_d'+R` is wrong. **Fix:** recompute `R'` or re-predict
   it. *Neither the original plan nor the first doc draft handled this.*
2. **In-gamut recomposition.** Brightening shading blows highlights; `I_enh` must be
   tone-mapped/clamped to `[0,1]`. Needs an explicit gamut step.
3. **Multiplicative error compounding.** enhancement = decomposition × edit quality. V18 albedo is
   ~50% cast-contaminated (§2.2.1), so edits inherit that error — image-space methods (Afifi)
   avoid this penalty. **Reviewers' #1 attack;** B1 albedo-WB is the mitigation, and "vs
   Marigold+edit" must show our decomposition wins net.
4. **No canonical enhancement target** (aesthetic ambiguity, §2.5.3) — the eval-validity gap.

### 2.7 Genuinely novel angles not yet exploited

- **WB in IID space for MIXED illumination.** Afifi's hard case is multiple illuminants; a
  *spatially-varying* white-point correction on the **albedo chroma** (per-region) is exactly what
  a decomposition enables and image-space blending cannot. **Likely the sharpest, most defensible
  contribution** — a concrete win on a known-hard benchmark.
- **Generative window/highlight fill (v2, descoped).** V18's SD prior + stochastic mode could
  *hallucinate* plausible content in blown windows when editing R — impossible for Marigold (no
  editing) or Afifi (no generation). Park for v2.

---

## 3. Thesis arc (the story)

1. **Method (Part 1):** a physics-grounded one-step IID fixing Marigold-IID's missing
   reconstruction/coupling and unifying its two models — *proven* on IID benchmarks.
2. **Application (Part 2):** that single unified decomposition **unlocks automatic image
   enhancement** — WB via **albedo-chroma** correction (esp. mixed-illuminant, the Afifi-hard
   case) and lighting via **geometry-gated shading** editing — measured on field-standard
   benchmarks (ΔE2000/MAE on Rendered-WB/Cube+; PSNR/SSIM/LPIPS on LOL).

*method → validation → application*, each defensible, neither from-scratch.

---

## 4. Immediate next steps

1. Write the **V18-compatible quantitative eval** (ARAP/IIW/SAW) — gates Part 1.
2. Build the **synthetic enhancement-pair generator** (re-render Hypersim/IV under color-cast /
   uneven light from GT shading) — gates Part 2's primary metric.
3. Prototype **Phase-B1 albedo-WB correction** first (contribution + mitigation for drawback #3),
   then **B2 lighting** — with **stale-R recomputation** (drawback #1) handled.
4. Lock baselines: Marigold-IID (Part 1); **Afifi Deep-WB / Multi-Illuminant-2025 / WB-LUTs**
   (WB), **Retinexformer / Zero-DCE** (lighting), and **"Marigold-decomp + same edit"** ablation.
