# Thesis Review - 2026-07-14

## Scope

This review audits the compiled thesis (`Main.pdf`), the LaTeX sources, the
evaluation implementations, benchmark artifacts under `tests/visualizations`,
the figure builders under `tests/viz`, the training configurations, and the
recorded checkpoint histories. It uses the thesis as a reviewer would: every
claim must have a valid protocol, an identifiable model/checkpoint, evidence
that measures the claimed property, and a figure or table that can be read
without relying on the prose.

No thesis source or benchmark result was changed during this review.

## Overall Verdict

The thesis has a coherent research arc:

1. identify color constancy as an under-evaluated IID failure mode;
2. propose a color-aware architecture and cross-illumination supervision;
3. evaluate constancy and albedo fidelity on complementary datasets;
4. study the effect of the CARI and refinement components;
5. demonstrate editing and transfer applications;
6. report the remaining hard-shadow and ambiguity failures.

The contribution is defensible, but the current PDF is not submission-ready.
The main risk is not prose length. Two numerical interpretations are currently
too strong, the local ARAP comparison does not reproduce the published input
protocol, and several figures do not contain what their captions say. These
must be corrected before beautifying the document or making SOTA claims.

## P0 - Submission Blockers

### 1. Fix and rerun the ARAP standard protocol

`tests/eval/eval_arap.py` labels the `--white_balance` path as white-balanced,
but that path currently calls `_hdr_norm`, which applies one scalar exposure
normalization. It does not remove the colored shading.

The Ordinal Shading ARAP protocol instead derives shading from the input and
ground-truth albedo, desaturates that shading, and recomposes a white-balanced
input. In notation, the required operation is approximately

```text
S = I / max(A*, eps)
I_wb = A* * luminance(S)
```

with the paper's validity mask and clipping rules. This is materially different
from scaling all RGB channels by one number.

The local ARAP `si-RMSE` is also mean-normalized RMSE, not the usual least-squares
scale-aligned RMSE. SSIM alignment and the LMSE window protocol have not yet been
shown to match the published evaluator exactly. Therefore Table 5.5 must not
claim parity with or superiority over published Ordinal Shading numbers until:

- white-balanced inputs are reconstructed using the published procedure;
- RMSE, LMSE, SSIM, masks, resizing, and aggregation are matched exactly;
- a known published baseline or released prediction reproduces its reported
  score within a stated tolerance;
- the full 52-scene/157-image protocol is used and documented.

Until that rerun is complete, retain the raw-input ARAP results only as a local
stress test and remove cross-paper bolding/SOTA language.

Primary reference: [Ordinal Shading](https://arxiv.org/abs/2311.12792).

### 2. Replace the MID fidelity interpretation

The new metric correction correctly identifies the flaw in the old pooled cast
score: pooling material and illumination variation rewards desaturated albedo.
Separating within-material drift from between-material chroma spread is a useful
diagnostic. However, the current `Chroma_fid` interpretation is still too strong.

Current definition:

```text
Chroma_fid = predicted between-material chroma spread
             / pseudo-GT between-material chroma spread
```

A value near 1 means only that the two spreads have similar magnitudes. It does
not establish that each material has the correct hue. A hue permutation or
rotation can score 1. The evaluator already computes the more relevant guard:

```text
Chroma_err = mean material chroma distance(prediction, pseudo-GT)
```

Use `Chroma_err` as the primary MID chroma-calibration guard. Keep a spread
diagnostic only as secondary evidence, preferably as a symmetric error such as
`mean(abs(log(Chroma_fid)))`; do not call it true-hue fidelity.

There is a second aggregation problem. For the selected model, the mean
per-scene `Chroma_fid` is 1.008, but the median is 0.920 and the ratio of the
aggregate spreads is 0.941. One scene reaches 2.520. The phrase "reproduces
chromatic diversity essentially exactly" is therefore not supported by the
reported mean ratio.

Recommended MID table columns:

| Property | Metric | Interpretation |
|---|---|---|
| lightness stability | `C_mat` down | material lightness drift across lights |
| chroma stability | `Cast_rel` down | chroma drift relative to scene chroma separation |
| pseudo-GT chroma calibration | `Chroma_err` down | material hue/chroma error against MID pseudo-GT |
| optional spread diagnostic | `abs(log spread ratio)` down | mismatch in chroma-spread magnitude |

MID albedo is a dataset-derived pseudo-GT, not independently measured albedo.
The table may show in-domain calibration and reject gray collapse, but MAW is the
external measured-color evidence.

### 3. Add uncertainty before ranking small differences

The current tables rank point estimates without confidence intervals. A paired
scene bootstrap over the 30 MID test scenes gives:

| Model | `C_mat`, 95% CI | `Cast_rel`, 95% CI |
|---|---:|---:|
| Ours | 0.130 [0.117, 0.144] | 0.421 [0.357, 0.488] |
| Base CARI | 0.157 [0.142, 0.174] | 0.425 [0.360, 0.493] |
| Marigold-App | 0.191 [0.172, 0.212] | 0.353 [0.308, 0.403] |
| Marigold-Light | 0.543 [0.460, 0.645] | 0.403 [0.340, 0.472] |
| CRefNet | 0.151 [0.138, 0.164] | 0.355 [0.307, 0.405] |
| Ordinal Shading | 0.252 [0.229, 0.280] | 0.549 [0.463, 0.639] |

The paired Ours-minus-Base difference in `Cast_rel` is about -0.004 with a 95%
interval crossing zero. Treat it as unchanged, not as a win. Report paired
bootstrap intervals for ablation deltas and dataset-level intervals for SOTA
tables. Bold a result only when the ranking is meaningful at the stated unit of
analysis.

### 4. Correct experiment provenance and causal wording

The thesis currently mixes three different meanings of "Ours":

- `v17_44@40k`: the complete Table-A CARI/base model;
- `v17_34@60k`: the selected Table-B refinement and current final result;
- generic method descriptions that do not identify a checkpoint.

Define these once at the start of Chapter 5. Every table and figure should list
the exact checkpoint in a note or provenance manifest.

Table A is not a fully matched 2x2 causal experiment. The color-path rows add
parameters and start with a fresh optimizer state; the recorded training history
also indicates a different effective LR trajectory. Moreover, the axis combines
the RGB skip and chroma-loss setting, so it does not isolate the skip alone.
Rename it "color-path bundle" and interpret only the within-architecture CARI
comparisons (Rows 1 to 2 and 3 to 4) causally. Treat cross-axis differences as
descriptive unless all rows are retrained with matched optimizer initialization.

Table B is exploratory, not a controlled single-variable ablation. Checkpoint
history shows different 3D-Front mixtures during training:

- `v17_33`: approximately 0.32/0.24/0.24/0.20 from 40k to 60k;
- `v17_29`: multiple mixtures before settling at 0.30/0.30/0.15/0.25;
- `v17_34`: the standardized 0.30/0.30/0.15/0.25 refinement mix;
- non-3D-Front rows: 0.40/0.30/0.30.

Current YAML files do not retroactively define old checkpoints. Use logs and
checkpoint histories as the provenance source. Replace "controlled negative
result" with "exploratory negative result" and avoid "pre-registered" unless a
timestamped artifact demonstrably predates the results. "Pre-specified in the
experiment config" is safer where supported.

### 5. Rebuild figures whose captions do not match their contents

The exported images and `tests/viz` builders establish the following factual
mismatches:

- Figure 1.1 says three MID scenes; the asset contains one scene in two rows.
- Figure 4.1 says it shows a validity mask; the builder does not render the mask.
- Figure 5.6 says three MID scenes; it reuses the same one-scene asset as Figure 1.1.
- Figure 5.8 says Input/Ours/Marigold-App/GT; the builder emits Input/GT/Ours.
- Figure 5.9 says two multi-light ARAP strips plus variation maps; the builder
  emits three single-light Input/GT/Ours examples.
- Figure 5.10 describes three illuminants, four model rows, and two heatmaps;
  the builder emits one scene under two illuminants without that layout.
- Figure 7.1 claims three failure mechanisms; its asset shows one scene and does
  not demonstrate achromatic ambiguity.

Captions must describe the pixels actually shown. Because these are generated
assets, fix the canonical builder and regenerate rather than editing captions to
preserve an accidental layout.

### 6. Remove administrative submission defects

- Replace the placeholder jury entries in `Main.tex` ("Monsters University" and
  "University of Springfield").
- Populate PDF title and author metadata.
- Disable visible red hyperref rectangles for the final PDF.
- Resolve the overfull Table 5.8 float and other logged overfull boxes.
- Add short optional captions so the List of Figures/Tables does not reproduce
  paragraph-length captions.

## Structure and Length

The high-level chapter order is coherent. The imbalance is in execution:

| Chapter | Approximate source words |
|---|---:|
| 1 Introduction | 1,553 |
| 2 Related Work | 2,196 |
| 3 Method | 2,841 |
| 4 Implementation | 3,449 |
| 5 Experiments | 10,780 |
| 6 Applications | 1,512 |
| 7 Conclusion | 1,555 |

Chapter 5 is about 44% of the body and repeatedly re-explains the gray-collapse
metric problem. Keep one rigorous metric-design subsection, then refer back to
it. Move protocol details, full raw/white-balanced tables, per-scene results,
and old/superseded metrics to an appendix.

The compiled PDF is 106 pages, while the body and bibliography finish around
page 75. The long Lists of Figures and Tables consume roughly 27 pages because
full analytical captions are written into the lists. Use:

```latex
\caption[Short factual list entry]{Two to four sentence main caption.}
```

Captions should identify the data, columns, normalization, and visual encoding.
Interpretation belongs in the body.

Recommended Chapter 5 order:

1. evaluation questions and model/checkpoint definitions;
2. protocols and metric validity;
3. external benchmark comparison;
4. Table-A mechanism ablation;
5. Table-B exploratory refinement study;
6. qualitative results tied directly to the quantitative sections;
7. failure analysis and threats to validity.

Use the same benchmark order in Chapters 2, 4, and 5, or explicitly explain why
the order changes. A useful taxonomy is: measured fidelity (MAW), controlled
multi-light constancy and dense synthetic accuracy (ARAP), in-domain real
multi-light diagnosis (MID), and sparse ordinal reflectance (IIW).

## Metric Review

### MID

`C_mat` measures per-material luminance coefficient of variation across lighting
conditions. It is a constancy diagnostic, not an accuracy score: a constant gray
prediction obtains zero.

`Cast_rel` compares within-material chroma drift with between-material chroma
separation. This is more resistant to gray collapse than the old pooled score,
but it is a discriminability ratio rather than pure hue invariance. Report it
only beside a fidelity/calibration guard.

Implementation improvements:

- compute material statistics on the intersection of valid pixels across all
  light conditions, not condition-dependent masks;
- replace unstable `R/G` and `B/G` ratios for low-green materials with thresholded
  log-chromaticity or normalized `rg` chromaticity;
- report per-scene distributions and paired bootstrap intervals;
- call the reference `pseudo-GT`, consistently;
- describe the within/between quantities as derived diagnostics, not an exact
  square-root decomposition of total variance.

Current aggregate `Chroma_err` values support a more defensible external-method
statement:

| Model | MID `Chroma_err` down |
|---|---:|
| Ours | 0.1213 |
| Base CARI | 0.1287 |
| Ordinal Shading | 0.1476 |
| Marigold-Light | 0.1573 |
| Marigold-App | 0.1959 |
| CRefNet | 0.2007 |

This says the selected model best matches the MID pseudo-GT material chroma in
this comparison. It does not prove globally correct color.

### ARAP

`C_arap` is a per-pixel luminance CoV across lighting conditions. It is also
gamed by constant predictions, so the statement that it is unaffected by
degeneracy is false. Pair ARAP constancy with ARAP's own dense-GT guard rather
than asking the reader to infer fidelity from MID.

Recommended ARAP presentation after protocol correction:

1. all-scene raw-input constancy: `C_arap`, cross-light chroma drift, dense-GT
   `Chroma_err`, and one raw albedo accuracy column;
2. indoor raw-input constancy: the same columns, for the target domain;
3. published white-balanced protocol: LMSE, scale-invariant RMSE, SSIM, compared
   only after exact evaluator parity is established.

Do not add every existing raw/colored/WB score to the main table. Put full metric
matrices in the appendix and keep the main table aligned to one question.

### MAW

The local evaluator correctly calls the bundled MAW scorer. The thesis prose
does not describe that scorer accurately:

- `Delta E` uses robust per-material estimates and aligns each material by a
  scalar before Lab color difference; it measures chromaticity/hue-saturation
  quality after removing material intensity scale, not absolute RGB color.
- the intensity metric fits a global scale; it measures relative material
  intensity, not absolute brightness.

The measurements may be physically calibrated, but the reported metrics remove
scale. Replace "absolute colour and absolute brightness" with the exact protocol.
Do not use an uncited universal `Delta E = 2` just-noticeable-difference claim.

Primary reference and motivation for complementary MAW metrics:
[MAW](https://arxiv.org/abs/2306.15662).

### IIW

WHDR is useful but sparse and lightness-order based; it does not test chroma.
That limitation justifies treating IIW as ancillary, not explaining away every
poor result. The current claim that sharper shadow boundaries in albedo would
improve WHDR is physically confused: shadows should be removed from albedo.
Discuss material-boundary preservation and relative reflectance ordering instead.

## Table Redesign

### External comparison

Use separate tables for separate claims:

- MID: `C_mat`, `Cast_rel`, `Chroma_err`, with confidence intervals.
- MAW: official `Delta E` and intensity metrics, with their scale alignment stated.
- ARAP raw: all and indoor constancy plus a dense-GT fidelity guard.
- ARAP published protocol: LMSE/RMSE/SSIM only after the rerun.
- IIW: WHDR, marked as sparse lightness ordering.
- efficiency: parameters, runtime, resolution, and hardware/protocol footnote.

Do not create a single winner count across these tables. The datasets measure
different properties and use different domains.

### Table A

Part 1 should show mechanism diagnostics:

```text
MID: C_mat, Cast_rel, Chroma_err
ARAP all: C_arap, ARAP Chroma_err
ARAP indoor: C_arap, ARAP Chroma_err
```

Part 2 should show external/accuracy guards:

```text
MAW: Delta E, relative-intensity error
ARAP raw: LMSE, scale-RMSE (local stress test)
ARAP WB: LMSE, scale-RMSE, SSIM (only after protocol correction)
IIW: WHDR (ancillary)
```

Move the superseded pooled MID cast metric to an appendix audit table. The main
text should call the color axis a bundle and disclose optimizer/LR confounding.

### Table B

Use Table B to explain model selection, not to claim a controlled component
ablation. Show MID stability/calibration, ARAP raw stability/fidelity, and MAW
color/intensity. Add confidence intervals and the actual per-run mixture history
in a compact footnote or appendix table. The defensible 3D-Front conclusion is:

> The synthetic pairs transferred on the illuminant-color axis but did not
> improve the hard-shadow structural failure measured on ARAP; the renderer's
> shadow-edge distribution was substantially softer than ARAP's.

That is a useful negative result. It does not require presenting the dataset as
a successful universal invariance source.

## Figure Review

### Canonical generation

All thesis figures should have one canonical builder under `tests/viz`. At
present, multiple scripts can write the same output, including the MAW resolution
asset. This makes the PDF non-reproducible because the last script run wins.

Create a figure manifest containing, for each asset:

```text
figure label
builder script and arguments
output path
checkpoint path and hash
dataset split and sample IDs
lighting IDs
normalization/tone mapping
selection criterion
```

Builders must fail if another builder owns the same output path.

### Sample selection

The current builders often use `sorted(...)[0]`, so the same first MID scene is
reused. That is neither representative sampling nor deliberate case selection.
Do not replace it with best-score cherry-picking. Use model-independent criteria:

- introduction: one or two visually legible scenes with large illuminant change
  and adequate pseudo-GT material diversity;
- MID main result: easy/median/hard quantiles selected by input or GT properties;
- MAW: one median-color case, one chromatic-material case, one relative-intensity
  challenge;
- ARAP: one indoor and one outdoor multi-light group, with GT and shared-scale
  error/variation maps;
- limitations: the worst cases selected by a stated metric, separated from the
  representative main results.

### Per-figure actions

- Figure 1.1: use a clean two-scene summary or correct the caption; do not reuse
  the Chapter 5 asset unchanged.
- Figure 3.2: redraw the architecture as a readable two-row vector diagram. Show
  frozen DINO stages, token reassembly, detail stem/color skip, pyramid fusion,
  the three output heads, analytical shading/residual, and training-only CARI.
- Figure 4.1: add the actual validity mask and visually distinguish supervised
  and raw-pair branches.
- Figure 4.3: correct the curriculum to 0-15k Hypersim, 15-19k Hypersim+IV,
  19-40k CARI, and optional 40-60k refinement.
- Figure 5.3: reduce method count or use zoomed crops; six tiny columns are not
  inspectable in print.
- Figure 5.4: plot `Cast_rel` against `Chroma_err`, not the raw spread ratio. A
  second panel may relate MID stability to MAW `Delta E`, clearly labeled as a
  cross-dataset diagnostic.
- Figure 5.5: show fewer methods or a crop where material colors are visible.
- Figures 5.6, 5.8, 5.9, 5.10: rebuild to match their captions and tables.
- Figure 6.1: replace the seven-panel strip with a 2x3 or 2x4 layout and larger
  crops.
- Figure 6.2: use stronger but defensible edits and zoom regions. Do not call
  clipped content "recovered"; it is model-inferred or hallucinated.
- Figure 7.1: show separate rows for shadow leakage, highlight/specular drift,
  oversaturation, and achromatic ambiguity.

Use shared color scales for comparable heatmaps, a colorblind-safe palette, and
state all normalization in the caption or manifest.

## Method and Training Claims

### Architecture rationale

The architecture is described, but its alternatives are not evaluated. The
reviewer-safe rationale is:

- DINOv2 provides strong general-purpose and dense visual features, but the
  thesis has not established that its tokens are intrinsically illumination
  invariant. Present that as a hypothesis unless token-similarity evidence is
  added. Primary source: [DINOv2](https://arxiv.org/abs/2304.07193).
- DPT-style token reassembly combines global transformer context with multi-scale
  dense refinement and is compatible with a frozen ViT encoder. State that no
  decoder-family ablation was performed. Primary source:
  [DPT](https://openaccess.thecvf.com/content/ICCV2021/html/Ranftl_Vision_Transformers_for_Dense_Prediction_ICCV_2021_paper.html).
- The RGB/detail skip is motivated by possible color loss in frozen features,
  but Table A tests a color-path/loss bundle, not the skip in isolation.

### CARI mechanism

`L_inv` does not by itself force illuminant color into shading. Its interaction
with reconstruction, shading/albedo supervision, and the decomposition heads
creates that pressure. The explanation loss is luminance-based, so the claim
that raw pairs are required specifically because `L_expl` needs illuminant color
is inaccurate. Raw pairs chiefly preserve the RGB variation used by invariance
and reconstruction-related terms.

Any reported correlation change, such as 0.51 to 0.14, needs the sample count,
estimator, script, split, and uncertainty. Otherwise move it to exploratory
analysis or remove it.

The analytical positive residual makes recomposition accuracy partly
tautological. It is a model parameterization, not evidence that the factors are
correct. Zero MID residual targets away from highlights are a modeling
assumption, not a proven physical property.

### Training schedule

The reported base schedule should match the actual checkpoint history:

```text
0-15k: Hypersim
15-19k: Hypersim + InteriorVerse
19-40k: Hypersim + InteriorVerse + MID/CARI
40-60k: optional Table-B refinement at constant 2e-5
```

The base config uses `phase1_iterations: 15000`, `phase2_iterations: 19000`, and
`extend_iterations: 40000`. Remove stale 25k/50k comments and distinguish the
base cosine schedule from the constant-LR refinement.

Hypersim subset size should be presented as a resource-constrained choice, not
as proven sufficient. List the exact scenes/frames and split policy.

## Application Claims

Cross-render transfer recombines albedo from one frame with shading/residual from
another frame predicted by the same model. This is useful internal evidence, but
its PSNR is gameable through compensating factor errors. Call it a decomposition
consistency test, not unbiased proof of physicality.

The statement that residual scaling recovers window blinds entirely invisible
in a clipped input is not physically defensible. Information absent through
clipping cannot be recovered from that image; the network may infer plausible
structure. Use "reveals model-inferred structure" and show the unclipped evidence
if another exposure contains it.

"Preserves material colors and texture exactly" should become "holds the model's
predicted albedo fixed." This distinguishes an editing constraint from a claim
about true scene reflectance.

Primary context for colorful diffuse decomposition and applications:
[CD-IID](https://arxiv.org/abs/2409.13690).

## Citations and Reproducibility

Every dataset paragraph should cite the dataset paper at the first factual
claim. Every borrowed protocol should cite both the paper and, where available,
the official evaluator/repository. The most important primary sources are:

- [MID](https://openaccess.thecvf.com/content_ICCV_2019/html/Murmann_A_Dataset_of_Multi-Illumination_Images_in_the_Wild_ICCV_2019_paper.html)
- [Hypersim](https://openaccess.thecvf.com/content/ICCV2021/html/Roberts_Hypersim_A_Photorealistic_Synthetic_Dataset_for_Holistic_Indoor_Scene_Understanding_ICCV_2021_paper.html)
- [MAW](https://arxiv.org/abs/2306.15662)
- [ARAP](https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.13149)
- [Ordinal Shading](https://arxiv.org/abs/2311.12792)
- [CD-IID](https://arxiv.org/abs/2409.13690)
- [DINOv2](https://arxiv.org/abs/2304.07193)
- [DPT](https://openaccess.thecvf.com/content/ICCV2021/html/Ranftl_Vision_Transformers_for_Dense_Prediction_ICCV_2021_paper.html)

Do not crop a paper's architecture figure when a thesis-specific vector diagram
can explain the actual implementation. If a paper figure or table is reproduced,
verify the publisher/license requirement and state "reproduced" or "adapted"
rather than only adding a citation.

Create one appendix table with:

```text
model name | source | checkpoint | preprocessing | inference resolution
dataset split | sample count | metric implementation | aggregation unit
```

This will answer most reviewer questions about fairness and reproducibility more
efficiently than repeating protocol caveats in every result paragraph.

## Recommended Revision Sequence

1. Freeze a model/checkpoint registry and export exact run histories from logs.
2. Correct the ARAP white-balance and metric protocol; validate parity; rerun.
3. Recompute MID tables with `Chroma_err`, robust spread diagnostics, and paired
   bootstrap intervals.
4. Rebuild Tables 5.2-5.8 around one question per table; move full matrices to
   the appendix.
5. Rewrite claims that currently exceed the metrics, especially SOTA, true hue,
   controlled ablation, physicality, and clipped-content recovery.
6. Make one canonical figure pipeline and manifest; regenerate the mismatched
   figures using predeclared sample-selection criteria.
7. Compress Chapter 5, shorten captions, and rebuild the Lists of Figures/Tables.
8. Redraw the architecture and curriculum figures.
9. Complete a primary-source citation and bibliography audit.
10. Fix front matter, metadata, hyperref styling, float warnings, and compile the
    final PDF twice from a clean auxiliary state.

## Reviewer-Facing Thesis Claim After Correction

A defensible central claim is narrower and stronger than the current wording:

> Cross-illumination supervision improves the stability of predicted albedo on
> aligned real multi-light scenes. A color-preserving prediction path prevents
> that stability objective from collapsing material chroma, while measured MAW
> results provide an external color-fidelity guard. The selected refinement
> improves lightness constancy and MAW color error, but hard cast-shadow leakage
> remains unresolved on ARAP and the synthetic 3D-Front pairs did not close that
> structural domain gap.

This claim is supported by the available evidence once the ARAP comparison and
metric reporting issues above are corrected. It is more credible than claiming
unqualified color truth, universal invariance, or SOTA across incompatible
protocols.

---

# Debate — response from the implementing agent (2026-07-14)

I verified the review's load-bearing claims against the code, the checkpoints and the
result JSONs rather than accepting or rejecting them on reading. **I agree with the review
on every point I was able to check, and on two of them the evidence is stronger than the
review states.** The disagreements below are refinements, not rejections, and I flag them
because acting on the review as literally written would lose something load-bearing.

## Independently confirmed (no debate — these are correct and I was wrong)

* **ARAP `--white_balance` is exposure normalisation, not white balance.** `_hdr_norm`
  multiplies all three channels by a single scalar `target/p`, so `r/g` and `b/g` are
  *unchanged by construction*. The "standard protocol" ARAP table therefore does not feed a
  white-balanced input and cannot be compared to published Ordinal numbers. **P0, agreed.**

* **`Chroma_fid` = 1.008 is a mean-of-ratios artefact.** The ratio of aggregates is
  `Cast_between / GT_between` = 0.4659 / 0.4948 = **0.941**, not 1.008 (Jensen). The phrase
  "reproduces chromatic diversity essentially exactly" rests on the inflated figure and must
  go. **Agreed.**

* **`Chroma_fid` measures spread magnitude, not hue.** A hue rotation preserving spread
  scores 1. Correct in principle. **Agreed** — see the refinement in D1 below.

* **MAW metrics are scale-free.** `chromaticity_deltae` is computed on chromaticity and
  `intensity_si_mse` is explicitly scale-invariant. Ch5 (line ~632) calls them "absolute
  colour and absolute brightness". That is simply wrong. **Agreed.**

* **Figure captions do not match assets.** Confirmed on Fig 1.1: the caption says "three MID
  scenes"; `build_hires_figures.py:186` uses `sorted(os.listdir(MID))[0]` — one scene.
  The same line confirms the review's point about lexicographic-first sample selection
  (every MID figure is `everett_dining1`). **Agreed.**

* **Admin defects.** Jury really does read "Monsters University" / "University of
  Springfield"; PDF Title/Author/Subject are empty; `pdfborder` is commented out in
  `mydiss.sty:475`. **Agreed.**

## Stronger than the review states

* **D0. Table B Row 5 is contaminated, and it is provable from the checkpoint.** The review
  infers mixture drift from training logs. It is recorded in the weights:

  ```
  v17_33 ckpt: {hypersim 0.32, midintrinsic 0.24, interiorverse 0.24, front3d 0.20}  <- OLD
  v17_34 ckpt: {hypersim 0.30, midintrinsic 0.30, interiorverse 0.15, front3d 0.25}
  ```

  `v17_33.yaml` was edited to the standardised mix but the run was never relaunched with
  `checkpoints/v17_33/` cleared, so auto-resume continued the old-mix checkpoint. Rows 4 and
  5 therefore differ in the flatness weight **and** the data mixture. The draft states they
  "differ only in its weight" — false. This specifically invalidates the (5−4) contrast used
  to claim the flatness weight is the operative variable and that full strength is "strictly
  worse". That claim must be withdrawn, not merely relabelled exploratory.

## Debate

### D1. Do not demote the spread diagnostic to "secondary". It is the mechanism evidence.

The review is right that `Chroma_err` is the correct *calibration* metric and that
`Chroma_fid` must not be called hue fidelity. I have adopted that. But `Chroma_err` **cannot
distinguish grey-collapse from hue-rotation** — both inflate it identically. The spread ratio
is the only term that separates them: collapse drives it well below 1, rotation leaves it at
~1.

The thesis's central claim is not merely "rivals have wrong colour" (which `Chroma_err`
shows) but specifically **"rivals achieve apparent invariance by *destroying* colour"** —
a causal, mechanistic claim. That mechanism is established by the spread ratio and by nothing
else in the metric set. So the two metrics answer different questions and the spread term is
load-bearing for the argument, not decorative:

| Failure mode | `Chroma_err` | spread ratio |
|---|---|---|
| grey collapse | high | **≪ 1** |
| hue rotation, spread preserved | high | ≈ 1 |

**Proposal:** `Chroma_err` primary (calibration), spread ratio reported alongside it as the
*collapse detector*, with the collapse-vs-rotation table above stated explicitly. Both, not
one demoted beneath the other.

### D2. The `Chroma_fid` correction does not weaken the headline claim — it strengthens it.

Worth recording, because the review does not say what happens to the thesis once its
correction is applied. Scored on `Chroma_err` (the review's own preferred metric), 30-scene
MID test split:

| | ours | | | | | external | | | |
|---|---|---|---|---|---|---|---|---|---|
| v17_20 | v17_23 | v17_29 | **v17_34** | v17_33 | | Ordinal | M-Light | M-App | CRefNet |
| 0.115 | 0.115 | 0.120 | **0.121** | 0.143 | | 0.148 | 0.157 | 0.196 | 0.201 |

**Every one of our rows beats every external model.** The claim survives the correction on
the correct metric, and more cleanly than it did on `Chroma_fid`. One honest nuance the
thesis must now absorb: v17_34 ("Ours") is *not* the most hue-faithful of our own rows —
v17_20/v17_23 are (0.115 vs 0.121). That is a real cost of the refinement and should be
stated rather than buried.

### D3. Cross-chapter recurrence of the degeneracy argument is not redundancy.

I accept that Chapter 5 is over-long and that there is genuine within-chapter repetition to
cut. I do **not** accept that the appearances of the metric-collapse argument in Ch5, Ch6 and
Ch7 are redundant. They are three *different measurements* of one principle:

* Ch5: the pooled chroma cast is farmable (definitional/metric);
* Ch6: **cross-illuminant recomposition is *also* farmable** — the colour-skip-off ablation
  wins transfer PSNR at 35.06 dB by pushing colour into the shading, where it cancels in the
  product. This is an independent empirical finding, not a restatement;
* Ch7: therefore *no* score computed purely from a model's own outputs escapes this, and only
  measured-albedo anchors do (a scoping limitation on the field).

Collapsing these into one location would delete the Ch6 result, which is the only evidence
that the degeneracy generalises beyond the one metric we happened to fix. Trim within Ch5;
keep the three instances.

### D4. Unverifiable as stated: the per-scene `Chroma_fid` distribution.

I confirm the *direction* of the review's aggregation complaint (0.941 vs 1.008, above), but
the specific figures — median 0.920, max 2.520 — cannot be reproduced from the saved
artefacts: `run_eval` stores only `np.nanmean` of the per-scene values and discards the
per-scene list. These numbers must be regenerated by a rerun that retains per-scene values
(and that rerun should happen anyway, to produce the bootstrap CIs the review asks for in P0-3).
Not a challenge to the conclusion; a note that the evidence for it is not currently on disk.

### D5. Concur on CIs — and it cuts against a claim the review does not name.

Agreed that small differences must not be ranked without intervals. Note this also lands on a
Table-A claim the review leaves alone: "CARI improves `Cast_rel` in **both** architectures"
rests on deltas of ~0.03 and ~0.02, which on the review's own CI widths (~±0.065) are very
unlikely to be significant. The defensible CARI result is on `C_mat` (0.250 → 0.180, a 28%
drop) and `Chroma_err`, not on `Cast_rel`. I would withdraw the `Cast_rel` monotonicity claim
rather than defend it.

## Summary of my position

Adopt the review essentially in full. The four things I would change in how it is actioned:

1. Keep the spread ratio as a co-primary **collapse detector**, not a demoted secondary (D1).
2. Record that the corrected metric **preserves** the headline claim (D2) — and that v17_34 is
   not our most hue-faithful row (an honest cost the review's framing would surface anyway).
3. Trim Ch5 internally but keep the Ch5/Ch6/Ch7 instances of the degeneracy argument (D3).
4. Withdraw the `Cast_rel` monotonicity claim in Table A too, which the review does not
   explicitly flag (D5).

Everything else — the ARAP rerun, `Chroma_err` as primary, dropping "absolute" for MAW,
provenance/checkpoint pinning, exploratory framing for Table B (and **withdrawal** of the 5−4
flatness claim, D0), figure rebuilds, and the admin fixes — I agree with without reservation.

---
# Debate from Claude
# Resolution log (2026-07-15) — reruns complete, tables + text updated

All 13-model reruns finished and validated (every MID JSON = 30 scenes; ARAP-157 = 13/13).

**P0-1 ARAP protocol — DONE.** `_white_balance_gt` replaces the exposure-only `_hdr_norm`
(verified: it drives the implied illuminant to 1.000/1.000; the old path left it unchanged).
`tab:arap_accuracy` rebuilt on the full 157-image protocol (136 evaluable; 21 near-black-GT
cases dropped deterministically for all models). **Cross-paper bolding/SOTA language removed**
— the caption + a footnote state that our harness scores Ordinal at LMSE 0.059 vs its
published 0.021 (2.8x), so no published comparison is drawn. `eval_arap.py` accuracy path also
fixed to route CRefNet/Ordinal through their adapters (was v17+Marigold only).

**P0-2 Chroma_err primary — DONE.** `eval_mid_constancy.py` emits per-material Chroma_err;
`eq:chromaerr` added; `tab:mid` now leads with Chroma_err (colour correctness), keeps
Chroma_fid as the *collapse detector* (debate D1, not demoted). Every one of our rows beats
every external on Chroma_err (0.121 vs 0.148–0.201), all paired-CI significant.

**P0-3 uncertainty — DONE.** `scripts/analyze_mid_bootstrap.py` → 95% scene-bootstrap CIs in
`tab:mid`; paired deltas vs v17_34 computed. Bold only where best AND paired-significant
(Marigold-App/CRefNet tie at Cast_rel 0.355 → neither bolded).

**P0-4 provenance + wording — DONE.** §5.1 now pins "Ours (full model)" = v17_34@60k and
"Ours (base CARI)" = v17_44@40k explicitly, and names the generic third usage. Cast_rel
monotonicity claim WITHDRAWN (deltas ~0.03 sit inside the CIs → reported as direction, not
gain; the real CARI result is C_mat, −28%/−39%). "pre-registered" → "pre-specified in config";
"controlled negative result" → "negative result" (Ch1/Ch7); Table B reframed as exploratory
with the mixture-provenance caveat. MAW "absolute" wording fixed earlier (scale-free).

**Chroma_fid aggregation (D4) — DONE.** Table quotes ratio-of-aggregates (0.941), not the
mean-of-ratios (1.008); §5.2 states this and flags the mean as outlier-inflated.

**Still open (not P0):** figure caption/asset audit (Figs 1.1/4.1/5.6/5.8/5.9/5.10/7.1 —
several already rebuilt: architecture, CARI, datasets, limitations, formulation, chroma-
fidelity, tradeoff, ch6); representative-scene selection rule + provenance manifest for the
remaining qualitative figures; final bib pass on the ~18 unverified entries.

---
# Figure provenance manifest

Every figure in the thesis, mapped to the exact script/function that produces it, its data
source, and (where a model is involved) the checkpoint used. Regenerate a figure by running
its builder; TikZ/pgfplots figures are `\input` from `chapters/fig_*.tex` and rebuild on
compile, but any raster thumbnails they embed come from the listed script.

Paths are relative to the repo root `/home/khang/IR-IID`. All figure assets live under
`documents/thesis/images/<subdir>/`; builders write there directly.

Last updated 2026-07-15.

---

## Method chapter (Ch. 3)

| Figure | Kind | Builder | Data / model |
|---|---|---|---|
| `fig:architecture` | TikZ + thumbnails | `chapters/fig_architecture.tex`; thumbnails `tests/viz/gen_arch_thumbs.py` | thumbnails = v17_29 @ 60k on MID `everett_dining1` (CPU inference) |
| `fig:cari` | TikZ + thumbnails | `chapters/fig_cari.tex`; thumbnails `tests/viz/gen_cari_thumbs.py` | thumbnails = v17_29 @ 60k, two illuminants (warm=dir18, cool=dir12) of `everett_dining1` |
| `fig:inverse_shading` | pgfplots + panel | `chapters/fig_inverse_shading.tex`; panel + histogram data `tests/viz/build_inverse_shading_figure.py` | Hypersim `ai_001_001`; S = I/A from GT; hist data → `chapters/data/shading_hist_*.dat` |

## Method / implementation (Ch. 4)

| Figure | Kind | Builder | Data / model |
|---|---|---|---|
| `fig:mid_preprocessing` | raster | `tests/viz/build_hires_figures.py` → `fig_mid_preprocessing()` | raw vs gray-world-white-balanced MID pair + GT albedo, `everett_dining1`; **no model** (data only) |
| `fig:front3d` | raster | `tests/viz/build_front3d_figure.py` | rescans the rendered 3D-Front-IID corpus (`../datasets/front3d_iid`); stats are re-measured, not asserted |
| `fig:curriculum` | TikZ | `chapters/Chapter4.tex` (inline TikZ, no external asset) | — |

## Experiments (Ch. 5)

| Figure | Kind | Builder | Data / model |
|---|---|---|---|
| `fig:datasets` | raster | `tests/viz/build_datasets_figure.py` | one panel per benchmark (MID/IIW/MAW/ARAP); IIW row overlays real WHDR judgement points |
| `fig:comp_grid` | raster | `tests/viz/build_hires_figures.py` → `fig_comp_grid()` | 3 IIW photos × full roster; per-panel scale-normalised; v17_34 = "Ours" |
| `fig:mid_constancy`, `fig:intro_constancy` | raster (same asset `hires/mid_ours.jpg`) | `tests/viz/build_hires_figures.py` → `fig_mid_ours()` | MID `everett_dining1`, 4 illuminants + GT + CoV map; v17_34 |
| `fig:iiw` | raster | `tests/viz/build_hires_figures.py` → `fig_iiw_ours()` | 4 IIW photos, input/albedo pairs; v17_34 |
| `fig:maw` | raster | `tests/viz/build_hires_figures.py` → `fig_maw_ours()` | 3 MAW scenes, input/measured-albedo/ours; explicitly labelled v17_29 |
| `fig:arap_constancy` | raster | `tests/viz/build_hires_figures.py` → `fig_arap_ours()` | bedroom/classroom/livingroom, dense GT vs ours; explicitly labelled v17_29 |
| `fig:arap_models` | raster | `tests/viz/build_hires_figures.py` → `fig_arap_model_grid()` | `chocofur`, five models across L0/L1/L2 and dense GT; ours explicitly v17_29 |
| `fig:maw_resolution` | raster | `tests/viz/build_hires_figures.py` → `fig_maw_resolution()` | one MAW photo at {512,768,1024,1280}px; v17_34 |
| `fig:ablation` (`hires/ablation_mid.jpg`) | raster | `tests/viz/build_hires_figures.py` → `fig_ablation()` | MID scene, 2 illuminants, Table-A rows v17_41/42/43/44 |
| `fig:shadow_aug` | raster | `tests/viz/build_hires_figures.py` → `fig_shadow_aug()` | composites `tests/visualizations/shadow_aug_out/*.png` (known relight fields) |
| `fig:chroma_fidelity` | raster | `tests/viz/build_chroma_fidelity_figure.py` | MID scene, 2 illuminants; GT vs v17_34 / v17_44 / v17_42 / CRefNet / Marigold-App; Chroma_fid captions from `roster_corrected_cast/` |
| `fig:tradeoff` | pgfplots | `chapters/fig_tradeoff.tex` | Cast_rel vs Chroma_fid scatter; data hardcoded from `roster_corrected_cast/` (print with `scripts/summarize_corrected_cast.py`) |
| `fig:formulation` | raster | `tests/viz/build_formulation_figure.py` | MID `everett_kitchen18`, light 18; grey-shading vs RGB-shading albedo; 2.31-vs-0.22 chroma error |

## Applications (Ch. 6)

| Figure | Kind | Builder | Data / model |
|---|---|---|---|
| `fig:application_checkpoint` | raster | `tests/viz/build_ch6_figures.py` → `fig_checkpoint_compare()` | fixed realistic office sample `test_1.png`; benchmark v17_34 versus explicitly labelled application checkpoint v17_29 |
| `fig:ch6_edits` | raster | `tests/viz/build_ch6_figures.py` → `fig_edits()` | fixed realistic office sample `test_1.png`; residual scaling and percentile-normalised shading lift; v17_29 |

## Conclusion (Ch. 7)

| Figure | Kind | Builder | Data / model |
|---|---|---|---|
| `fig:limitations` | raster | `tests/viz/build_limitations_figure.py` | MID scene; GT vs v17_34 vs v17_33 (flat-4.0 over-saturation) + cross-illuminant instability map |

---

## Notes

* **Thumbnail and application model choice.** `fig:architecture` and `fig:cari` use
  **v17_29** thumbnails (per author instruction). Chapter~6 also uses v17_29, but labels
  it as an application checkpoint and directly compares it with benchmark-selected
  v17_34. Chapter~5 quantitative tables use **v17_34** ("Ours (full model)");
  the MAW and ARAP qualitative sheets use explicitly labelled v17_29 outputs.
  The architecture is identical across v17 variants, so the encoder/decoder/head diagram is
  correct regardless of which checkpoint rendered the thumbnails.
* **Scene selection.** Builders now record their exact samples. The MID fidelity
  figure uses two distinct scenes, MAW and ARAP use three content-diverse scenes,
  the ARAP model grid uses the hard `chocofur` sequence, and Chapter~6 uses the
  supplied office photograph `test_1.png`.
* **CPU-generated thumbnails.** `gen_arch_thumbs.py` / `gen_cari_thumbs.py` run on CPU
  (`CUDA_VISIBLE_DEVICES=""`) so they never contend with training/eval jobs on the GPUs.
* **Regeneration.** Raster builders write straight into `documents/thesis/images/`, so after
  running one, a plain `latexmk` picks up the new asset. TikZ/pgfplots figures need only a
  recompile unless their embedded thumbnails or `.dat` files changed.

---

## Post-review Chapter 6 resolution (2026-07-15)

Chapter~6 is retained, but only as a short qualitative applications chapter. The previous
cross-illuminant transfer figure and PSNR table were removed because compensating albedo and
shading errors cancel in their product; the visibly desaturated colour-path-off model scored
best, so the result could not support physical correctness. The transfer utility remains in
the builder only behind an explicit diagnostic flag and is not part of the thesis.

The replacement figures use the supplied realistic office sample `test_1.png`.
They compare v17_34 with the explicitly labelled v17_29 application checkpoint,
then demonstrate residual scaling and a percentile-normalised low-illumination
gain with v17_29. The residual changes broad scene regions rather than isolating
one highlight, so claims are limited to factor controllability; no semantic
highlight removal, ground-truth recovery, blur metric, or application-level
quantitative improvement is asserted.
