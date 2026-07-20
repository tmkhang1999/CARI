# Indoor ARAP results + the two pre-questions (augmentation & white-balance)

> **Date:** 2026-06-15. Supersedes the `all`-only read in `eval_row4_analysis.md` with the
> FAIR indoor split, and answers the two questions raised before re-addressing the 8.

## The indoor ARAP constancy table (the fair headline — 23 groups)

| Row | C_arap ↓ | Cast_RMS ↓ | si-RMSE (guard) | LMSE |
|---|---|---|---|---|
| 19k (row 1, no CARI) | 0.1600 | **0.0350** | 0.2559 | 0.0216 |
| 30k (row 2) | 0.2348 | 0.0693 | 0.3242 | 0.0278 |
| **40k (row 4, full CARI)** | **0.1502** | **0.0777** | **0.2183** | 0.0252 |
| marigold_app | 0.0664 | 0.0319 | 0.1475 | 0.0160 |
| marigold_light | 0.2484 | 1.631 | 0.3973 | 0.1350 |

**Two hard facts the indoor split confirms (it did NOT rescue the `all` story):**
1. **CARI's Cast_RMS got WORSE in-domain too:** 19k 0.035 → 40k **0.078 (2.2× worse).** The cast
   regression is REAL, not an outdoor artifact. Confirmed on the fair indoor set.
2. **C_arap barely moved** (0.160→0.150, −6%) and **Marigold still wins** (0.066, 2.3× better).
   On the axis the thesis claims, a frozen feed-forward CARI trails a 2024 diffusion model AND
   is barely better than its own pre-CARI baseline on indoor constancy.
   *Only* si-RMSE improved (0.256→0.218) — accuracy held, so still no gray collapse.

This sharpens the `eval_row4_analysis.md` problem: **the headline mechanism is underperforming
on its own claimed metric, and the cast fix is actively backfiring.** The two questions below
diagnose WHY — and the answer to Q2 is, in my read, the single most important finding so far.

---

## Q2 (answered first — it's the root cause): YES, copying MID's white-balance fights the thesis

**What the pipeline does:** `MIDIntrinsicDataset._white_balance` ([src/data/midintrinsic_dataset.py#L90](../src/data/midintrinsic_dataset.py#L90))
perfectly white-balances EVERY frame by its gray probe (`r_ratio,1,b_ratio` = probe median),
copied verbatim from `generate_albedo.py:44-51`. **This removes the illuminant color before the
model ever sees it.**

**The measured evidence this erases real signal (scene `kingston_library26`):**

| light dir | probe illuminant R/G | probe illuminant B/G |
|---|---|---|
| dir_0 | 1.071 | 0.998 |
| dir_6 | 1.130 | 0.837 |
| dir_12 | 1.105 | 0.941 |
| dir_18 | 1.243 | 0.782 |

**MID's flashes are NOT white** — the probe chromaticity varies ~15-25% across directions
(B/G 0.78→1.00). This is a GENUINE colored-illuminant signal, measured per-frame, pixel-aligned
— *exactly* the disambiguation signal the thesis is built to exploit. The current pipeline
**throws it away** (WB normalizes every frame to neutral) and then tries to **re-inject it
synthetically** via chromatic aug (`U[0.6,1.4]` tint on one frame).

**Why this is backwards (the core insight):**
- The thesis claim is "single image can't separate material color from *light color*; cross-render
  supervision teaches the invariance." The mechanism NEEDS colored-illuminant variation in the
  pair to learn from.
- WB destroys MID's real colored variation → `L_inv` over a WB'd pair only sees direction/intensity
  change (shadows move), NOT color change. So `L_inv` learns shadow-invariance, not COLOR-invariance.
- The synthetic chromatic aug is then the ONLY color signal — and the indoor numbers show it
  TEACHES THE WRONG DISTRIBUTION: Cast_RMS regressed 0.035→0.078. A `U[0.6,1.4]` per-channel
  multiplicative tint is not how real illuminants vary (real casts are correlated across channels,
  follow a blackbody-ish locus, and co-vary with intensity). The model overfits to synthetic tints
  that don't match ARAP's real colorful shading, and its recovered-albedo color drifts MORE.

**This is not a small bug — it inverts the data strategy.** We white-balanced away the real
signal and substituted a synthetic proxy that measurably hurts. For an *illumination-color-invariance*
thesis, that is the thing to fix before anything else.

### The fix for Q2 (highest-value change in the project right now)
**Use MID's RAW (pre-WB) frames in the cross-render pair**, so `L_inv` sees real colored-illuminant
variation. Concretely, the principled options (in order):
1. **Pair on raw frames, supervise albedo on the WB frame.** Keep WB ONLY for the pseudo-GT-albedo
   anchor (it must match the WB'd `albedo.exr`); feed the *raw* (un-WB'd) pair to `L_inv`/`L_explain`.
   Then `L_inv: A(I_a_raw) ≈ A(I_b_raw)` demands invariance to the REAL color difference between
   dir_a and dir_b. This is physically exact (`A` is the same surface) and is the thesis's actual
   claim. **Drop or shrink the synthetic chromatic aug** — the real signal replaces it.
2. If the pseudo-GT-albedo supervision conflicts with raw frames (it's WB'd), supervise albedo
   only as a weak anchor (lower `lambda_a` on MID) and let raw-pair `L_inv` carry the color
   invariance — the design already says MID albedo is a *weak* anchor (§4.3).
3. Keep a SMALL chromatic aug as a complement (covers casts beyond MID's flash gamut), but it is
   no longer the primary color signal.

---

## Q1: The augmentation isn't "too weak" — it's the WRONG KIND. Two distinct problems:

**(a) High-intensity visualization breakdown is partly a DATA/EVAL artifact, not a model failure.**
ARAP stores albedo at WILDLY inconsistent scales: `whiteroom` GT albedo p99 = **0.005**, `villa`
= 0.006, but `kitchen` = **1.06** — a 200× range across indoor scenes. The model (trained on
Hypersim's [0,1] albedo) outputs ~[0,1] regardless, so against a 0.005-scale GT it looks blown
out no matter what. The "p90 looks better" observation = the p90 INPUT tonemap matches V17's
training preproc (`_tonemap_linear` maps p90→0.8), so the model gets in-distribution input and
its prediction is sharper. **This is an eval-display + input-domain issue, NOT something
augmentation fixes.** (The metrics are scale-invariant so they're already fair; only the *picture*
misleads.)

**(b) The cast regression (the real research problem) is NOT solved by stronger synthetic aug.**
The indoor data shows synthetic chromatic aug is ALREADY HURTING (Cast_RMS 0.035→0.078). Leaning
harder on `U[a,b]` tints would make it worse — wrong distribution, not insufficient strength.

**So, per the three options you raised:**
- **Exposure aug:** WORTH ADDING but for problem (a)/domain-gap, not the cast. ARAP/wild scenes
  span a far wider exposure range than indoor Hypersim. A random-exposure aug (scale input by
  `~LogU[0.25,4]` before tonemap) would close the high-intensity domain gap and likely help
  si-RMSE on bright scenes. Secondary priority.
- **Stronger chromatic aug:** ARGUE AGAINST. The data says it's the problem, not the solution.
- **Another dataset:** This is the strongest "more aug" lever. Two candidates that carry REAL
  colored-light variation (what MID-after-WB lacks):
    - **BigTime (Li & Snavely)** — time-lapse, real outdoor+indoor illumination-color change over
      the day. Already in the §1.1 lineage. Real colored-cast pairs, free reflectance-constancy.
    - **MID RAW (un-WB)** — *we already have it* (Q2). The cheapest "new dataset" is the real
      signal in the data we already use. Do Q2 first; it's free.

---

## ARAP data quality — DIAGNOSED + FIXED (2026-06-15): preprocess, don't skip

Investigating "the viz is bad on high-intensity scenes" found ARAP "indoor" scenes with
near-zero HDR (violin/iron/bread: p99~2e-4) and inconsistent-frame-scale scenes (cream:
light0 p99~2e-4 next to light2 p99~250). **Key finding — these are NOT bad data:**
- violin/iron/bread are RECOVERABLE — dim but valid. After the per-image p99/p90→0.8 tonemap
  the model input already applies, violin has **68% midtone pixels** (a fine input). They were
  never broken — just stored at low exposure that the **constancy code wasn't normalizing
  per-frame** the way training does. → KEEP them (the tonemap recovers them).
- cream is MIXED — light2 fine, but light0 (p99~2e-4, 1e6× below its siblings) is genuinely
  sensor-noise after the tonemap amplifies it. The per-frame tonemap maps siblings to different
  effective brightnesses → "albedo varies across lights" measures an EXPOSURE artifact. → drop
  only the out-of-family FRAME, keep the scene.

**Fix (implemented in `eval_arap.py::_group_frame_validity`):** validity is GROUP-RELATIVE —
drop an HDR frame only if its p99 is >1000× away from the group-median p99 (or truly black);
uniformly-dim scenes (whole group dim → ratios ~1) are KEPT. Verified on CPU (no model needed —
pure numpy): cream drops light0 (3/4 kept), violin/iron/bread keep ALL frames. If dropping
leaves <2 frames the scene falls out (n_groups stays honest).

**High-exposure handling was ALREADY correct for V17:** `_prepare_input_rgb` uses
`percentile=90` for HDR V17 input (matches training; the comment notes p99 pushes strong-light
frames OOD and washes albedo — exactly the user's "p90 is better" observation). Marigold/V18 use
p99 by design (trained on sRGB display, not HDR) — its high-exposure failure is inherent to its
training domain, not a bug to "fix" (forcing p90 on Marigold = a different OOD input). So no
architecture/loss change is needed for high-exposure — it is a per-image-tonemap (preprocess)
matter, already handled, now also covered in constancy mode.

## Revised next steps (Q2 changes the priority order from eval_row4_analysis.md)

1. **DO Q2 FIRST — retrain from 19k with RAW-pair cross-render (drop/shrink synthetic chromatic
   aug).** This is now the highest-value action, ahead of "train to 50k". Rationale: 50k of the
   CURRENT recipe extends a recipe whose color signal is synthetic-and-harmful. Fix the signal,
   THEN scale. Implementation: pair on raw frames in `__getitem__` (the `_load_raw_frame` path
   exists; currently both frames are WB'd via `_white_balance` inside it — add a `wb=False` path
   for the pair while keeping the supervised frame WB'd). New ablation row: **raw-color-pair vs
   WB-pair+synth-aug** — this directly tests the Q2 hypothesis and is itself a thesis result.
2. **Add exposure aug** (problem a / domain gap) — `LogU` input exposure jitter. Cheap, parallel.
3. **Fix the ARAP albedo-scale eval display** — per-scene the GT albedo scale varies 200×;
   normalize the contact-sheet albedo panels by the GT's own scale (already done via
   `_albedo_scale`) but ALSO clamp/report which scenes have degenerate (near-0) GT so the sheet
   isn't misread. (Metrics already scale-invariant — display only.)
4. **THEN train to 50k** on the fixed recipe and re-eval indoor.
5. **Row 6 + MAW** as before — MAW's real-measured chromaticity is the external adjudicator of
   whether the raw-pair fix actually improved cast.

**The reframed thesis bet:** if raw-color-pair `L_inv` (Q2 fix) drops Cast_RMS below baseline on
indoor ARAP — *that* is the result: "the REAL multi-illumination colored signal, used directly,
teaches color-invariance that synthetic augmentation cannot." It also turns the current weakness
(WB-then-synth) into a clean ablation that PROVES the real-signal claim (§2.2 raw-vs-synth, now
with teeth). If even raw-pair can't beat Marigold's constancy, then the honest claim is the
controlled-mechanism one (rows 1/6), and architecture levers (§3.1 A1 α-gate) come next.

---

## The colored-illuminant BENCHMARK problem (user's out-of-box question) — SOLVED

**The gap the user spotted is real and important:** ARAP varies illumination by **intensity +
direction**, NOT by COLOR (its shading is colorful per-scene, but it does not give the SAME scene
under multiple DIFFERENT illuminant *colors*). MID-test, post-WB, has the same limitation. So
neither of our current benchmarks directly measures *colored-illuminant* albedo invariance — the
exact axis the thesis claims. We need a 3rd dataset where one scene is shot under several light
*colors* with a fixed reflectance GT.

**Field-survey verdict (2026-06-15) — almost NOBODY benchmarks this, which is itself a result:**
- **IDArb (arXiv:2412.12083, the 2024 multi-illum SOTA)** trains under varied lighting but
  **evaluates albedo only with scale-invariant PSNR/SSIM — it does NOT report cross-illuminant
  albedo consistency.** Confirmed from the paper. The leading method leaves the axis unmeasured.
- **CD-IID / Marigold / RGB↔X** — none report colored-illuminant constancy.
- **The ONE work that names the exact gap:** Sato et al., *Self-Supervised IID Considering
  Reflectance Consistency* (arXiv:2111.04506): *"most conventional work does not sufficiently
  account for reflectance consistency owing to the use of a white-illuminant model and the lack
  of training images capturing the same objects under various illumination-brightness and -color
  conditions."* They propose a **Reflectance Consistency Index (RCI)** = stability of recovered
  reflectance across illumination conditions. **This is our metric's prior-art name — cite it,
  adopt RCI terminology.** (Confirms our C_mat/Cast_RMS are the right idea, gives them a citation.)

**THE 3rd dataset — Beigpour Multi-Illuminant Intrinsic (MIID), the colored-light benchmark:**
- **Single-view (ICCV'15 / "Comprehensive Multi-Illuminant"):** 5 REAL scenes, 2 objects each,
  **6 single-illuminant + 9 two-illuminant conditions, illuminant colors orange→blue**, full
  per-pixel GT: **reflectance, shading, specularity, illumination**. Real photos, colored shadows.
- **Multi-view (BMVC'16):** same idea, **20 illumination conditions × 5 scenes × 6 cameras**,
  high-res, + depth + 3D point cloud, pixel-wise reflectance/shading GT. "First publicly available
  multi-view real-photo dataset at this complexity with pixel-wise intrinsic GT."
- **Why it's THE fit:** same real scene, multiple DIFFERENT illuminant COLORS, fixed reflectance
  GT → directly measures "does recovered albedo stay constant as the light COLOR changes" — the
  literal thesis claim, on real photos, out-of-domain (nobody trains on it). It is the colored-
  light analogue of what ARAP does for direction/intensity.
- **Host:** University of Siegen CV group (`mi.informatik.uni-siegen.de`); authors Beigpour/Ha.
  Small (5 scenes) → fine as a targeted constancy probe, not a training set. **Action: locate the
  download (Siegen page / email authors) when adding it.**

**The four-axis benchmark story this creates (the thesis's evaluation backbone):**
| Axis of illumination change | Dataset | Domain | What it proves |
|---|---|---|---|
| **direction + intensity** | ARAP `_lightN` | synthetic, exact GT | invariance to where/how-bright the light is |
| **direction + intensity** | MID-test (WB) | real, pseudo-GT | same, in-domain, real photos |
| **COLOR** (the headline axis) | **Beigpour MIID** | **real, exact GT** | **invariance to light COLOR — the literal claim** |
| color (real measured GT, single-light) | MAW (chromaticity) | real, measured | recovered-albedo color correct vs measured albedo |

**So the answer to "how do we benchmark colored-illuminant invariance":** add **Beigpour MIID**
as the dedicated colored-light constancy probe (report RCI / our C_mat+Cast_RMS across its
color conditions), with **MAW** as the real-measured-chromaticity anchor. Together they cover the
COLOR axis that ARAP/MID-direction cannot. And the framing is strengthened by the survey finding:
**colored-illuminant albedo constancy is an under-measured axis even in 2024 SOTA (IDArb doesn't
report it) — turning a benchmark into a small contribution**, with RCI (Sato'21) as the citeable
precedent for the metric.
