# Thesis results & evidence tracking (NOT part of the compiled report)

This file is the external home for everything that must not appear as a
"pending/preliminary/TODO" note inside the report draft itself. The report is
written as a final version: measured numbers are presented plainly, and cells
whose result is not yet available are left blank ("-") with no in-text analysis.
When a result below becomes available, fill the corresponding table cell and add
its analysis to the running text.

Last audited: 2026-07-15.

**Figure provenance:** every figure -> its generating script/function is listed in
`documents/thesis/FIGURE_PROVENANCE.md` (regenerate any figure from its builder there).

## Current resolution (2026-07-15)

The authoritative MID and ARAP reruns have now been propagated into Chapter~5, including
the full 157-image ARAP harness (136 evaluable images), corrected per-material MID chroma
metrics, uncertainty intervals, and the exploratory treatment of Table~B's unmatched data
mixtures. Chapter~6 has been narrowed to two explicitly qualitative factor edits. It uses
v17\_29 only as a separately labelled application checkpoint and uses fixed realistic
sample `test_2.png`, chosen before inference for its bright windows and darker textured
interior. The gameable cross-illuminant transfer result is no longer in the thesis.

Remaining high-value checks are the bibliography verification pass and the unresolved
caption/asset items listed in the 2026-07-14 review. In particular, do not reinstate the
old pooled chroma-cast metric or the removed transfer PSNR as primary evidence.

---

# R1. FULL THESIS REVIEW (2026-07-13) — CV/colour-science pass

Scope note: Table B (`tab:lever_sweep`), the ablation Table A rows, and the SOTA
benchmark tables were **left untouched** on instruction (results still running).
Everything below concerns the other parts of the document. Changes marked
**[DONE]** are already applied and the document compiles clean (latexmk exit 0,
zero undefined references, worst overfull hbox 6.1pt). Items marked **[TODO]** are
left for you / the next agent.

---

## R1.1 CRITICAL: two fabricated references found and fixed **[DONE]**

This is the most serious finding of the review. Both were verified by web search
against the primary sources.

**(a) `sato21rci` did not exist.** The bib claimed:

> "Reflectance and Chromaticity Inconsistency for Illumination Estimation in HDR
> Photographs", Sato, Okabe & Sato, arXiv:2111.04506, 2021

No paper with that title or author list could be found. **arXiv:2111.04506 is in
fact a completely different paper**: Kinoshita & Kiya, *"Self-Supervised Intrinsic
Image Decomposition Network Considering Reflectance Consistency"* (verified by
fetching the arXiv abstract page directly). This citation was load-bearing: it was
used in Chapter 2 to justify an entire "RCI metric" paragraph, and in Chapter 5 to
motivate our $C_{\text{mat}}$ / $\text{Cast}_{\text{RMS}}$ metrics. **An examiner
who checks one reference will likely check this one**, because it is the only
citation backing our own metric design.

Fix applied: replaced with `kinoshita21consistency`, the real paper at that
identifier. This turned out to be *better* than the fabrication, because the real
paper is genuinely the closest prior work to CARI: it also observes that IID
methods do not constrain reflectance, also adopts a colour-illuminant model, and
also trains with losses across illumination conditions of the same content. The
Chapter 2 passage was rewritten to engage with it honestly, and to state our actual
delta: **their illumination variants are simulated; ours are real photographs of
real rooms under real bounced light.** That is a defensible contribution statement
and it is now backed by a citation that exists.

**(b) `luo20niid` had a fabricated author list.** Bib said "Luo, Jun-Jia and Li,
Zhen and Wang, Jue and Fu, Chi-Wing". The real NIID-Net authors are **Luo, Jundan;
Huang, Zhaoyang; Li, Yijin; Zhou, Xiaowei; Zhang, Guofeng; Bao, Hujun** (verified
against zju3dv.github.io/niid and the ZJU3DV GitHub). Corrected, with volume/pages
added.

**Also removed [DONE]:** three bib entries that were never cited and never reached
the bibliography (`garces12survey`, `mccamy76cct`, `colorchecker`). Note
`colorchecker` was a book miscoded as `@article` with `journal = {John Wiley}`.

**Verified as CORRECT (no action):** `marigoldiid24` (arXiv 2505.09358, Ke et al.),
`reasonx25` (arXiv 2512.04222, Dirik et al., now CVPR 2026), `sato25metrics` (arXiv
2505.19500, Shogo Sato et al.), `yuan25texture` (arXiv 2509.09352).

**[TODO — worth 10 minutes]** I verified the four above plus the two I fixed. The
remaining entries were *not* individually verified against primary sources:
`arap`, `maw`, `bell14iiw`, `careaga23ordinal`, `careaga24cdiid`, `luo23crefnet`,
`das22pienet`, `chen24intrinsicanything`, `kocsis24iid`, `zhu22irisformer`,
`li18bigtime`, `li18cgintrinsics`, `beigpour13miid`, `grosse09mit`,
`barron15sirfs`, `land71retinex`, `oquab24dinov2`, `ranftl21dpt`, `ranftl20midas`,
`roberts21hypersim`, `fu213dfront`, `midintrinsics`. Given that two of the ~28
entries were fabricated, **assume the rest are guilty until checked.** One specific
doubt: `maw` is filed as CVPR 2023, but search results also describe it as IEEE
ICCP 2023 — confirm the venue.

---

## R1.2 CRITICAL: we were describing our own metric incorrectly **[DONE]**

Chapter 5 described $C_{\text{mat}}$ as *"the change in predicted albedo
**chromaticity** across illuminants"*. It is not. Reading the implementation
(`tests/eval/eval_mid_constancy.py:321-357`), $C_{\text{mat}}$ is the within-material
**coefficient of variation of albedo LUMINANCE** across lighting conditions. The
chromaticity metric is the *other* one, $\text{Cast}_{\text{RMS}}$.

This is not cosmetic. The whole argument of the thesis is about **colour** leaking
into albedo, so claiming the headline constancy metric measures chromaticity when it
actually measures lightness would be a direct hit in a viva. It also explains a
result that otherwise looks strange (and is already noted in memory): a
configuration can improve $C_{\text{mat}}$ while *worsening* $\text{Cast}_{\text{RMS}}$,
which is impossible if they measure the same thing and entirely natural if one is
lightness and the other is hue.

Fix applied: both metrics are now defined with explicit equations
(\eqnref{eq:cmat}, \eqnref{eq:castrms}) that match the code, and the text now says
plainly that $C_{\text{mat}}$ is a lightness-stability score and
$\text{Cast}_{\text{RMS}}$ is the hue-stability score, and that they are reported
separately *because* they can move in opposite directions.

### **[TODO — REVIEWER RISK, needs your judgement]** $\text{Cast}_{\text{RMS}}$ may be contaminated

Reading `eval_mid_constancy.py:354-360`: the per-material, per-frame chromaticity
ratios `r = R/G` and `b = B/G` are **pooled across all materials** (`rg_vals.extend(...)`)
and then a single `Var()` is taken over the pooled list. That variance therefore
contains **between-material** variance, not only the across-illuminant drift we
intend to measure. A scene containing a red sofa and a blue wall will score a large
$\text{Cast}_{\text{RMS}}$ even under *perfect* illuminant invariance, simply because
those materials genuinely have different chromaticity.

Consequence: $\text{Cast}_{\text{RMS}}$ as implemented is partly a measure of scene
material diversity. It is still *comparable across models* on a fixed scene set
(every model sees the same materials, so the between-material term is a constant
offset shared by all rows), so the ranking is probably still meaningful and the
existing numbers are not worthless. But the metric as defined in
\eqnref{eq:castrms} does not measure only what we say it measures, and a sharp
examiner may spot it.

Two options, your call:
1. **Cheap and honest:** keep the numbers, and add one sentence to Chapter 5 saying
   the score pools across materials so it carries a constant per-scene offset, and
   is therefore used for *relative* comparison between models on a fixed scene set,
   not as an absolute quantity. Costs nothing, closes the hole.
2. **Correct but costs a rerun:** change the estimator to the within-material CoV of
   chromaticity (mirroring $C_{\text{mat}}$'s structure exactly), i.e. take
   `std_k(r_{m,k})` *inside* each material and then average over materials. This is
   what \eqnref{eq:castrms} currently *implies*. It would require re-running every
   MID evaluation, including the SOTA models.

I did **not** change the estimator, because it would invalidate results that are
currently running. Option 1 is written to be applicable without touching anything.

---

## R1.3 Structure: benchmark order reorganised as requested **[DONE]**

Chapter 5's benchmark sections were reordered **MID → IIW → MAW → ARAP** (was MID →
ARAP → MAW → IIW). Verified in the compiled PDF: pages 50, 53, 56, 57.

I did not just move the blocks; a reshuffle with no argument is just a different flat
list. The chapter now *argues* the order, and each section hands off to the next:

- **Chapter intro** states the order is deliberate and why: benchmarks do not carry
  equal weight and presenting them as a list implies they do.
- **MID** first: the only benchmark that measures the actual thesis property
  (does a real surface's albedo hold still when the light changes). Cannot prove
  correctness (no measured albedo), only stability. Said plainly.
- **IIW** second, retitled *"The Standard Benchmark, and Why It Cannot Settle This
  Question"*. It must be reported (a decade of precedent), and it is used to make the
  argument you asked for: WHDR is written out as an equation (\eqnref{eq:whdr}), and
  the limitation is then read directly *off* the equation — the predicted verdict
  $\hat{r}_i$ depends only on the *ratio* of two intensities, so WHDR is invariant to
  global rescaling and **never looks at colour at all**. A model can tint every
  surface with the lamp's colour and lose nothing. Since illuminant leakage *is* a hue
  error, WHDR is structurally incapable of confirming or refuting our claim. It is
  demoted to a guard.
- **MAW** third, opening with "MAW was built to close the gap IIW just described."
  This is the payoff of putting IIW before it.
- **ARAP** last, and the text now says why: it is synthetic and therefore the easiest
  to satisfy, so it cannot carry a claim about real rooms on its own; but it is the
  only benchmark with dense GT albedo **and** repeated illuminants, so it is the one
  place constancy and accuracy can be measured on the same pixels.

The old standalone "On the reliability of WHDR" paragraph was deleted (now redundant
with the IIW section intro).

---

## R1.4 Metric and formula explanations added **[DONE]**

You asked "explain the formula of metrics and loss for me". Chapter 5 previously
*named* its metrics without defining any of them. Now defined, all matched against
the actual scoring code, not from memory:

| Metric | Eq. | Where | Note |
|---|---|---|---|
| WHDR | `eq:whdr` | IIW §5.3 | with the hue-blindness argument read off the formula |
| $C_{\text{mat}}$ | `eq:cmat` | MID §5.2 | within-material **luminance** CoV (was misdescribed) |
| $\text{Cast}_{\text{RMS}}$ | `eq:castrms` | MID §5.2 | R/G and B/G spread; see caveat R1.2 |
| $\Delta E$ | `eq:deltae` | MAW §5.4 | CIELAB distance + *why* CIELAB (perceptual uniformity; $\Delta E \approx 2$ = JND) |
| SI-MSE | `eq:simse` | MAW §5.4 | scale fitted away first; explains *why* (albedo is scale-ambiguous) and what it catches (washed-out albedo) |

MAW's ground-truth procedure is now described correctly: the authors photograph a
surface twice, once with a **grey card of known reflectance** on it, so the shared
shading cancels in the ratio and the albedo is recovered in absolute terms. (My first
draft said "spectrophotometer" — I web-checked before committing it and it was wrong.
Flagging this because it is exactly the kind of plausible-sounding detail that becomes
a hallucination if unverified.)

**[TODO]** The CARI losses ($\mathcal{L}_{\text{inv}}$, $\mathcal{L}_{\text{expl}}$)
*are* already given as equations in Chapter 3 (`eq:linv`, `eq:lexpl`), and Chapter 4
has the full loss table. Those are fine. But the **remaining loss terms are named in
the Chapter 4 table without formulas** (MSG, DSSIM, residual sparsity, flatness).
If you want the "explain every loss" box fully ticked, add a short appendix with the
formula for each. Medium effort, low risk. I did not add it because it is additive
content rather than a correction, and you asked to prioritise correctness.

---

## R1.5 Factual inconsistencies fixed **[DONE]**

1. **Chapter 1 contributions still asserted the invalidated lever-sweep result**
   ("a manufactured shadow-supervision loss fails a pre-registered no-regression gate
   on all four benchmarks"). This came from the pre-wiring-fix sweep that we already
   established is invalid, and it had already been removed from Chapters 5 and 7 —
   Chapter 1 was missed. Rewritten to claim only the V20 shading-first negative
   result, which is measured and holds.
2. **IIW split size was wrong.** Chapter 2 said "We evaluate on a 200-image test
   split". The harness actually evaluates **1,046 images** (every 5th of 5,230; see
   `protocol.json`). Corrected, and the WHDR aggregation (per-image mean, not pooled)
   is now stated.

---

## R1.6 Figures: findings and TODOs

### **[TODO — HIGH PRIORITY] All qualitative figures are low-resolution**

Measured with `identify`:

| Figure | Current size | Verdict |
|---|---|---|
| `legacy_draft25/datasets.jpg` | **762 × 1032** | far too low; visibly soft in print |
| `legacy_draft25/comp_grid.jpg` | 1820 × 257 | too short; strip is unreadable at page width |
| `tableA_40k/ablation_mid.jpg` | 1518 × 502 | low |
| `legacy_draft25/mid_ours.jpg` | 2000 × 497 | low |
| `front3d/front3d_dataset.jpg` | 2113 × 1250 | acceptable (built at 200 dpi) |

All the `legacy_draft25` figures are built by `tests/viz/build_thesis_figures.py`,
which saves with `quality=94` but composites from **already-downsampled tiles**. The
fix is in the builder (raise the per-tile render size), not in the LaTeX. There is
already a `tests/viz/render_mid_hires.py` that renders high-res per-condition albedo,
which is the right source to composite from. **Regenerating these at high resolution
is the single biggest visual-quality win available and it costs no GPU time**, since
the predictions already exist.

### **[TODO — REQUIRED] `datasets.jpg` now contradicts the chapter order**

`fig:datasets` (Figure 5.2) lays the four benchmarks out as **MID, MAW, ARAP, IIW**.
The chapter now presents them **MID, IIW, MAW, ARAP**. Regenerate with the rows in
the new order (`fig_datasets()` in `build_thesis_figures.py`, around line 170). Low
effort, but it will look sloppy if missed.

### **[TODO] The IIW row of `datasets.jpg` is visually empty**

It shows a single input photograph and nothing else, leaving a large white gap while
the other three rows show input + GT. That is honest (IIW *has* no albedo GT, only
sparse point pairs) but it reads as a rendering bug. Suggest overlaying the actual
IIW judgement points on the photograph — two dots joined by a line, labelled
"lighter / darker" — which would both fill the row and *show* the reader what WHDR is
scored from. This directly supports the new IIW argument in §5.3.

### Figures I judge to be MISSING **[TODO]**

1. **A constancy-vs-accuracy scatter (the "tension" plot).** The thesis repeatedly
   claims a constancy/structure trade-off. It is currently asserted in prose and read
   off tables. One scatter (x = MID $C_{\text{mat}}$, y = ARAP si-RMSE, one point per
   model + our ablation rows) would make the central argument visible in one glance.
   **I can build this in LaTeX/pgfplots directly from the results JSONs once Table B
   lands** — say the word and I will write the builder. This is the highest-value
   missing figure.
2. **A qualitative failure case.** Every qualitative figure currently shows the model
   working. A reviewer will trust the thesis *more*, not less, if one figure shows
   where it breaks (e.g. the washed-out albedo from the flatness lever at weight 4.0,
   which we have measured and which is exactly the MAW-intensity regression). We have
   these renders.
3. **Relighting-transfer figure (Chapter 6).** Already marked `% TODO` in
   `Chapter6.tex:51`. Chapter 6 currently demonstrates *three* applications but shows
   **zero images** — it is entirely equations and prose. This is the weakest chapter
   in the document visually. Eq. `eq:relight_transfer` is a genuinely strong,
   unbiased physicality check (recombine albedo from frame $a$ with shading from frame
   $b$, compare against the real photo $b$) and it deserves a figure. This needs no
   new training, only inference on MID pairs.
4. **Front3D pipeline diagram (draw.io).** `fig:front3d` shows samples + statistics
   but not *how* the corpus is made (GLB → interior camera sampling by ray casting →
   K colour-randomised lightings → emission-pass albedo → validation). A block diagram
   would make the "How" section of §4.2.4 land much faster. **This one is a good
   draw.io candidate** — the others above I can do in code.

### Figure style **[TODO, subjective]**

You asked for the CD-IID / Mesh-thesis "half input, half output in one image" style.
Our current qualitative figures use a strip layout (input | pred | GT | heatmap) with
tiny text labels. The CD-IID style (a single image split by a diagonal or vertical
seam, input on one side, decomposition on the other) is more striking but shows fewer
panels. Recommend keeping the strip layout for the *benchmark* figures (they need to
show 4+ panels including the CoV heatmap, which the split style cannot do) and using
the split style only for the **teaser figure in Chapter 1**, where impact matters more
than completeness. I did not restyle anything, since this is a taste call and the
current layout is information-dense and honest.

---

## R1.7 Reviewer-mindset questions the thesis does NOT currently answer **[TODO]**

Working through "why this, why not that" as an examiner would. These are gaps in
*justification*, not errors:

1. **"Why DINOv2 and not a CNN / other ViT?"** §3.2.1 gives a good *principled*
   answer (frozen, semantic, photometric-augmentation-invariant → robust across shadow
   boundaries) and does **not** claim an empirical bake-off, which is honest. But
   there is no citation supporting "DINOv2 features are illumination-robust". That
   claim is doing real work in the architecture argument. **Find a citation or soften
   it to a hypothesis.**
2. **"Why DPT and not a simple conv decoder?"** §3.2.2 describes *what* DPT does but
   never says why a DPT is the right choice here rather than, say, a U-Net. One
   sentence needed.
3. **"Why is the RGB skip safe?"** §3.2.3 asserts the skip is "only safe in
   combination with CARI". This is the single most interesting architectural claim in
   the thesis and it IS backed by the ablation (rows 3 vs 4). Make the forward
   reference explicit and concrete.
4. **"Why these four benchmarks and not MIT-Intrinsic / MPI-Sintel?"** Now partly
   answered by the new §5.1 ordering argument, but never states why the *classic* IID
   benchmarks are excluded. One sentence (MIT is object-level and grey-illuminant;
   irrelevant to coloured indoor illumination) would close it.
5. **"What are the limits of the Lambertian assumption?"** Ch3 introduces
   $\mathbf{I} = \mathbf{A}\odot\mathbf{S}_d + \mathbf{R}$ and handles non-diffuse
   energy via $\mathbf{R}$, which is good. But nothing says what happens on
   *transmissive* or strongly *interreflecting* surfaces, where the model is simply
   wrong. Belongs in Ch7 Limitations.
6. **Ch7 "Grey shading assumption in the loss" is the strongest self-criticism in
   the document** and it is buried as the fourth paragraph of Limitations. Consider
   promoting it: it is the honest core of why constancy is hard here, and examiners
   reward that.

---

## R1.8 Writing style

Compared against `Mesh Reconstruction via Differentiable Rendering Pipelines.pdf`
(the COSI reference thesis): its register is flowing prose with inline author-year
citations woven into sentences ("As described by Tewari et al. (2020), ..."), long
explanatory paragraphs, and very few bullet lists in the body chapters.

Our draft is closer to a technical report: shorter sentences, more paragraph breaks,
heavier use of bold/paragraph headers. That is not wrong for a master's thesis and it
is *more readable*; I did not attempt a wholesale rewrite, because doing so across
2,500 lines risks introducing exactly the unsupported claims you told me to avoid, for
a purely stylistic gain.

What I did do, in every passage I touched: removed em-dashes in favour of commas or
full stops, cut the fancier register, and wrote in plain declaratives.

**[TODO]** If you want the reference thesis's style applied throughout, do it
chapter-by-chapter as a separate pass with fresh eyes on each. It should not be
bundled with a correctness pass.

---

## R1.9 What I deliberately did NOT touch

- `tab:lever_sweep`, `tab:ablation`, and all four SOTA benchmark tables (your
  instruction: results still running).
- The `Cast_RMS` **estimator** in the eval code (would invalidate in-flight runs; see
  R1.2 for the write-up-only alternative).
- Chapter 4's training-data and schedule sections (already reviewed and corrected in
  the 3D-Front pass earlier today).
- Any figure regeneration (all are listed above as TODOs with the exact builder
  function to change).


## A00g. CRITICAL — Cast_RMS is broken; it inverted the Table-A verdict (2026-07-14)

**The published $\text{Cast}_{\text{RMS}}$ rewards models for destroying albedo colour,
and it bolded the wrong model as best on five columns of Table A.** This was found by
following up the user's observation that v17_42 *looks* visibly desaturated despite
scoring the best cast in the table.

### The defect

`tests/eval/eval_mid_constancy.py` pooled the per-material chroma ratios across ALL
materials (`rg_vals.extend(...)`) and took ONE `np.std` over the pooled list. That
single variance therefore sums two things:

* **within** — per-material chroma drift ACROSS ILLUMINANTS (what we claim to measure);
* **between** — chroma spread ACROSS MATERIALS (scene material diversity).

The R1.2 note already suspected this but concluded the between-material term was "a
constant offset shared by all rows, so the ranking is probably still meaningful."
**That conclusion is wrong.** The between term is *not* model-independent: it shrinks
whenever a model pulls distinct materials toward a common hue. So the metric pays models
for collapsing colour, and the "ranking" it produces is partly a ranking of how much
albedo chroma each model has destroyed.

### Corrected metric (implemented)

`eval_mid_constancy.py` now also emits, per scene, with **GT albedo as the anchor**:

* `Cast_within`, `Cast_between` — the decomposition;
* **`Cast_rel` = within / between** — drift normalised by the model's OWN chroma
  magnitude (a chroma coefficient-of-variation). A model carrying 2x the chroma has 2x
  the room to drift in absolute units, so absolute drift is not scale-fair. **LOWER = better.**
* **`Chroma_fid` = between / GT_between** — fraction of real material chroma retained.
  **1.0 = faithful; <1 = collapsed; >1 = over-saturated.**
* `Chroma_err`, `Sat_ratio` — supporting fidelity diagnostics.

The old pooled `R_cast_rms` is **retained** so the previously published numbers stay
reproducible from the same run and the before/after is auditable. `eval_arap.py` has no
pooling bug (its cast is a per-group variance across illuminants) but its cast is an
*absolute* variance that a flat-grey predictor wins outright, so `Sat_ratio` /
`Chroma_err` guards were added there too.

**Design rule this encodes:** an invariance metric can NEVER be reported alone — a
constant (grey) predictor is perfectly invariant. Constancy must always appear beside a
chroma-fidelity guard, or it can be farmed.

### Corrected Table A (full 30-scene MID test split)

Results: `tests/visualizations/tableA_corrected_cast/`; print with
`python scripts/summarize_corrected_cast.py tests/visualizations/tableA_corrected_cast`.

| Row | CARI | skip | $C_{\text{mat}}$ | POOLED (published) | within | between | **Cast_rel** | **Chroma_fid** | MAW ΔE |
|-----|------|------|-------|--------|--------|---------|----------|------------|--------|
| 1 | off | off | 0.250 | 0.287 | 0.103 | 0.253 | 0.477 | 0.608 | 4.434 |
| 2 | ON  | off | 0.180 | **0.268** ← pooled "best" | 0.088 | 0.240 | 0.447 | 0.552 | 4.631 |
| 3 | off | ON  | 0.259 | 0.519 | 0.168 | 0.458 | 0.444 | **1.006** | **4.115** |
| 4 | ON  | ON  | **0.157** | 0.555 ← pooled "worst" | 0.170 | 0.494 | **0.425** | 1.057 | 4.155 |
| **GT** | | | | | | **0.495** | | **1.000** | |

The `POOLED` column reproduces the currently-printed Table-A values (0.287 / 0.268 /
0.519 / 0.555) exactly, so the decomposition demonstrably explains the published numbers.

**The colour skip lifts material-chroma fidelity from ~55-61% of GT to essentially exact
(1.006 / 1.057).** Corrected invariance is cleanly monotone where pooled never was: CARI
improves `Cast_rel` in BOTH architectures (0.477→0.447, 0.444→0.425) and the skip improves
it in BOTH CARI settings (0.477→0.444, 0.447→0.425). **v17_44 wins $C_{\text{mat}}$ AND
`Cast_rel`** — the pooled metric had it as the worst row in the table. MAW ΔE
independently corroborates the skip on real measured albedo (4.63 → 4.16).

### Corrected roster (same 30 scenes)

Results: `tests/visualizations/roster_corrected_cast/`.

| model | $C_{\text{mat}}$ | POOLED | within | between | **Cast_rel** | **Chroma_fid** |
|---|---|---|---|---|---|---|
| **v17_34 (= "Ours")** | **0.130** | 0.523 | 0.159 | 0.466 | 0.421 | **1.008** |
| v17_20 | 0.161 | 0.531 | 0.160 | 0.473 | 0.415 | 1.017 |
| v17_23 | 0.146 | 0.530 | 0.163 | 0.472 | 0.426 | 1.016 |
| v17_29 | 0.163 | 0.559 | 0.170 | 0.497 | 0.418 | 1.069 |
| v17_33 | 0.149 | 0.586 | 0.177 | 0.522 | 0.421 | 1.121 |
| v17_44 (base CARI) | 0.157 | 0.555 | 0.170 | 0.494 | 0.425 | 1.057 |
| **Marigold-App** | 0.191 | 0.399 | 0.110 | 0.361 | **0.353** | 0.825 |
| CRefNet | 0.151 | 0.265 | 0.075 | 0.239 | 0.355 | 0.575 |
| Marigold-Light | 0.543 | 0.715 | 0.205 | 0.585 | 0.403 | 1.172 |
| Ordinal | 0.252 | 0.543 | 0.214 | 0.457 | 0.549 | 1.036 |
| **GT** | | | | **0.495** | | **1.000** |

**Findings that must change the draft:**

1. **WE DO NOT WIN HUE INVARIANCE — concede it plainly.** Marigold-App (`Cast_rel` 0.353)
   and CRefNet (0.355) are both genuinely more illuminant-invariant than us (0.421).
   **Marigold-App, not CRefNet, is the strongest rival**: it dominates CRefNet on BOTH axes
   (more invariant AND more faithful, 0.825 vs 0.575) and beats us on MAW ΔE (3.775 vs 3.981).
   What we win outright is **chroma fidelity** (1.008 — essentially exact — vs Marigold-App
   0.825 and CRefNet 0.575) and **lightness stability** ($C_{\text{mat}}$ 0.130 vs 0.191 / 0.151).
   Claim: *"the only model simultaneously colour-faithful and competitively invariant, at a
   fraction of the cost"*. NOT *"we win colour constancy"*.
2. **CRefNet is v17_42.** CRefNet's `Chroma_fid` is 0.575; v17_42's is 0.552. Their pooled
   casts are 0.265 vs 0.268. They are behaviourally the same model — and v17_42 is the one
   we rejected on sight for looking desaturated. This is an in-architecture control that
   demonstrates the mechanism, and it is the basis of `fig:chroma_fidelity`.
2. **Table B's constancy column is a NULL RESULT.** The five rows span `Cast_rel`
   0.415-0.426 (2.6%) — indistinguishable. The pooled metric spread them over 0.523-0.586
   (12%) and ranked them confidently; that ranking was chroma fidelity in disguise
   (v17_33's "worst cast" 0.586 is *over-saturation*, fid 1.121, not drift). The levers and
   3D-Front data do NOT measurably move illuminant invariance; they move chroma fidelity and
   lightness stability. Say so, and drop any lever ranking based on cast.

### "Ours" = v17_34 (DECIDED, on the corrected metric)

Best $C_{\text{mat}}$ (0.130), most chroma-faithful in the whole roster (1.008), competitive
`Cast_rel`, plus best MAW ΔE (3.981) and best IIW WHDR (0.264) from the completed
4-benchmark run. Second SOTA-table row: **"Ours (base CARI)" = v17_44@40k**, so the Table-B
refinement gain is visible inside the SOTA tables. **Do NOT add a third "Ours" row** —
v17_33 wins ARAP $C_{\text{arap}}$ and v17_23 wins the white-balanced accuracy columns, and
picking the winner of each table reads as checkpoint-shopping.

### Model budget (MEASURED 2026-07-14, one Quadro RTX 6000, 512x512, median of 5)

| model | total params | trainable | s / img |
|---|---|---|---|
| **Ours (v17)** | 322.9 M | **18.5 M** | **0.148** |
| CRefNet | 66.6 M | 66.6 M | 0.447 (3.0x slower) |
| Marigold-App | 1290.0 M | ~1290 M | 0.805 (5.4x slower) |
| Marigold-Light | 1290.0 M | ~1290 M | 0.982 (6.6x slower) |

Ours = encoder 304.4M **frozen** (DINOv2-L, 0 trainable) + trunk 17.3M + albedo_head 0.6M
+ shading_head 0.6M. **Do NOT claim "low-budget" on TOTAL params — CRefNet is 4.8x SMALLER
than us (66.6M vs 322.9M) and an examiner will catch it.** The true claims are *18.5M
trainable* (3.6x fewer than CRefNet, ~70x fewer than Marigold; the encoder is frozen
off-the-shelf) and *single forward pass, 0.148 s/img* (3-6.6x faster than every baseline).

### **[DONE 2026-07-14]** Draft edits applied

The document compiles clean (latexmk exit 0, **0 undefined references**, 0 BibTeX errors,
100pp). Applied:

1. **§5.2 metric definitions rewritten.** Pooled `Cast_RMS` (\eqnref{eq:castrms}) retained,
   then decomposed by the law of total variance (`eq:castdecomp`); `Cast_rel`
   (`eq:castrel`) and `Chroma_fid` (`eq:chromafid`) defined, with the explicit rule that
   invariance can never be reported alone (a flat-grey predictor attains it perfectly).
2. **All tables filled**: `tab:mid`, `tab:iiw`, `tab:maw`, `tab:arap_constancy`,
   `tab:arap_accuracy`, `tab:ablation` (rebuilt; pooled column now *italic*, never bolded,
   never used for selection), `tab:lever_sweep` (constancy column reported as a NULL result).
3. **New §5.7 Computational Budget** + `tab:budget` (18.5M trainable / 0.148 s per image;
   states plainly that we are NOT the smallest model — CRefNet is 4.8x smaller in total).
4. **Abstract + Ch1 contributions + Ch7 rewritten.** The abstract previously claimed
   "Marigold-App remains stronger on MID $C_{\text{mat}}$" — factually INVERTED (we are
   0.130 vs their 0.191) — and Ch1 claimed "our model wins on three constancy-oriented
   metrics", which is false. Both corrected. The metric correction is now stated as a
   contribution in its own right (Ch1) and as a limitation on the field (Ch7).
5. **LR-confound hedging REMOVED** (user decision 2026-07-14): the fresh optimizer is
   intrinsic to changing the head parameter set — Adam moments are per-parameter and the
   skip introduces parameters that did not exist. Stated once as a design consequence.
6. **Figures built** (all re-runnable):
   - `fig:chroma_fidelity` (`tests/viz/build_chroma_fidelity_figure.py`) — the money figure:
     GT albedo beside ours / v17_42 / CRefNet / Marigold-App with measured `Chroma_fid`.
   - `fig:tradeoff` (`chapters/fig_tradeoff.tex`, pgfplots) — invariance vs fidelity scatter.
     CRefNet lands on top of our colour-skip-off ablation; that coincidence IS the argument.
   - `fig:relight_transfer` + `tab:transfer` + `fig:ch6_edits`
     (`tests/viz/build_ch6_figures.py`) — Ch6 had **zero images**; it now has two figures and
     a quantitative table.
   - `fig:datasets` (`tests/viz/build_datasets_figure.py`) — row order fixed to MID/IIW/MAW/
     ARAP, and the IIW row now OVERLAYS the actual human judgements instead of being blank.
7. **Bibliography**: `maw` venue was **wrong** (filed as CVPR; it is **ICCP 2023**,
   arXiv:2306.15662) — fixed. Two entries were miscoded (`fu213dfront` was an article with a
   conference in `journal`; `kinoshita21consistency` was inproceedings with an arXiv id in
   `booktitle`) — fixed. Verified against primary sources: `maw`, `arap` (Bonneel, CGF 36(2),
   593–609, 2017), `luo23crefnet` (TVCG 2023, DOI 10.1109/TVCG.2023.3337870),
   `beigpour13miid` (TIP 23(1):83–95, 2014). All 28 entries resolve; no orphans.

### NEW RESULT — the degeneracy is not confined to one metric

`tab:transfer` (Ch6): cross-illuminant relighting transfer, PSNR against the real photo,
20 MID scenes. **CARI improves it 31.12 → 34.70 dB on an objective never trained on** — the
strongest physicality evidence in the thesis. BUT the colour-skip-OFF ablation **WINS** the
column at 35.06 dB. Recomposition constrains only the PRODUCT `A ⊙ S`: a model that strips
colour from the albedo deposits it in the shading and reconstructs `I_b` just as well.
**So reconstruction-based checks are farmed by grey-collapse exactly like the chroma-cast
scores.** Only a metric anchored on MEASURED albedo (`Chroma_fid`, MAW ΔE) escapes. This is
now stated in Ch6 and generalised in Ch7.

### **[TODO]** Remaining

1. **`tab:ablation` (Table A):** bolding is WRONG on five columns (it favours v17_42).
### Figure-coverage audit vs CD-IID / Ordinal Shading / the Mesh thesis (2026-07-14)

Counted: Mesh thesis **29** figures, well spread (Ch2: 4, Ch6: 9). Ours was **20**, heavily
concentrated in Ch5 (10), with **Ch2 = 0 and Ch7 = 0**. Both rival papers devote a figure to
their failure cases (Ordinal Fig. 8, low-shading regions; CD-IID Fig. 12, what the pipeline
cannot fix) — we had none, and every qualitative figure showed the model working.

**Closed (2026-07-14):**
- `fig:limitations` (Ch7, `tests/viz/build_limitations_figure.py`) — three MEASURED failures:
  the flatness prior at full strength over-saturates past GT (Chroma_fid 1.121, worst in the
  study); cross-illuminant instability concentrates on the specular sphere and glossy lid
  (a Lambertian split has no representation for a mirror); achromatic ambiguity untouched.
- `fig:formulation` (Ch2, `tests/viz/build_formulation_figure.py`) — why a grayscale-shading
  model MUST leak illuminant colour, shown arithmetically. Granting the grayscale formulation
  the best scalar shading available to it (the luminance of the TRUE shading), its albedo
  still carries mean chroma error **1.09** vs **0.03** for 3-channel shading, and looks almost
  identical to the input. This is the thesis premise, and it now appears before the method.

**Still thin:** Ch6 has 2 figures (Mesh has 9 for applications). Scene diversity is narrow —
our qualitative panels are dominated by indoor rooms and MID tabletops, where CD-IID/Ordinal
show varied in-the-wild subjects (objects, close-ups, outdoor). Consider a wider-variety
comp_grid.

---

1. **Low-resolution legacy figures.** `legacy_draft25/comp_grid.jpg` (1820x257 — a strip too
   short to read at page width), `mid_ours.jpg`, `iiw_ours.jpg`, `maw_ours.jpg`,
   `arap_ours.jpg`, and `tableA_40k/ablation_mid.jpg` are still composited from
   already-downsampled tiles. The fix is in the *builder* (raise the per-tile render size in
   `tests/viz/build_thesis_figures.py`), not the LaTeX. Costs no GPU time — the predictions
   exist. `datasets.jpg` and the four new figures are already high-res.
   These are also the last figures built from the OLD checkpoints; `comp_grid` in particular
   should be regenerated with v17_34 as "Ours".
2. **A qualitative failure case.** Every qualitative figure still shows the model working.
   The obvious candidate is now measured: the flatness prior at full strength (Table-B Row 5
   = v17_33) **over-saturates** the albedo (`Chroma_fid` 1.121, the least faithful row in the
   study) and costs 0.25 ΔE on MAW. Showing that beside Row 4 would make the "levers are not
   free" argument visible.
3. **Table-B qualitative figure** for the five refinement rows (the old `fig:levers` was
   deleted with the cut lever sweep).
4. **Ch4 Front3D pipeline diagram** (draw.io): GLB → interior camera sampling → K
   colour-randomised lightings → emission-pass albedo → validation. `fig:front3d` shows
   samples and statistics but not *how* the corpus is made.
5. **Appendix**: move `within` / `between` / `Chroma_err` / `Sat_ratio` / sat-quartiles there.
   The body tables carry only $C_{\text{mat}}$ / `Cast_rel` / `Chroma_fid`, which is correct,
   but the supporting decomposition should be available to a reader who wants to check it.
6. **Bibliography**: the four highest-risk entries are now verified against primary sources
   (see above), and two type errors plus one wrong venue are fixed. The remaining ~18 entries
   are well-known papers whose metadata was spot-checked but NOT individually confirmed
   against the publisher record. Given that 2 of 28 were fabricated outright, a final pass
   before submission is still advisable.

---

## A00f. Table B rebuilt as a levers-vs-data study; 3D-Front-IID added (2026-07-13)

**Table 5.9 (`tab:lever_sweep`) restructured.** The old 6-row single-lever sweep
(control + flat + sh_inv + flat+sh_inv + sh_inv+sh_explain + ordinal, i.e.
v17_20/21/22/23/24/25) is **cut**. Its numbers were never usable in the final
draft: they came from a 25-scene subset, and they predate the
`mid-pair-wiring-bug` fix, so the two shadow-lever rows were trained by a
configuration that never actually ran `L_inv`/`L_explain`. Reusing them beside
freshly retrained rows would have mixed two lineages, two bases and two
protocols in one controlled-comparison table.

**Replacement (5 rows, all resuming v17_44@40k = Table-A Row 4, +20k -> 60k,
constant LR, same schedule):**

| Row | Config | Levers | Mix h/m/i/f |
|-----|--------|--------|-------------|
| 1 | Control (extra training only) | - | .40/.30/.30/- |
| 2 | + levers | flat 4.0, sh_inv 1.0 | .40/.30/.30/- |
| 3 | + 3D-Front-IID | - | .30/.30/.15/.25 |
| 4 | + 3D-Front + levers | flat **1.5**, sh_inv 1.0 | .30/.30/.15/.25 |
| 5 | + 3D-Front + levers | flat **4.0**, sh_inv 1.0 | .30/.30/.15/.25 |

Configs: v17_20 / v17_23 / v17_29 / v17_34 / v17_33 respectively.

**MIX STANDARDIZED 2026-07-13 (user decision).** All three front3d rows now train
on one identical mix, hypersim .30 / MID .30 / IV .15 / front3d .25 — this is the
single mix the report states, and the earlier per-row churn (0.32/0.24/0.24/0.20,
0.32/0.24/0.14/0.30) is deliberately NOT reported. `v17_33.yaml` was edited from
0.32/0.24/0.24/0.20 to match; **v17_33 must be relaunched with `checkpoints/v17_33/`
cleared first**, or auto-resume continues the old 0.20-mix checkpoint and silently
contaminates the row. Consequence: the study is now a clean factorial with three
single-variable contrasts — (2-1) levers without data, (4-3) and (5-3) levers with
data, (5-4) flatness weight alone. The earlier "Row 5 is descriptive" caveat is
GONE from the draft; do not reintroduce it.

**Cut and NOT to be silently reinstated:** the individual `flat`-alone,
`sh_inv`-alone, `sh_inv+sh_explain` and `ordinal` rows (v17_21/22/24/25). If
time permits later, they would need retraining from v17_44@40k under the post-fix
code to be admissible. The draft does not mention them; the lever *choice*
(flat + sh_inv) inherits from the old sweep and is stated as a design decision,
not as a measured claim.

**Figure removed:** `fig:levers` (`images/legacy_draft25/levers.jpg`) depicted the
cut levers and asserted pass/fail outcomes; it is deleted from Ch5. A qualitative
figure for the 5 new rows should be regenerated once the checkpoints land.
`fig:shadow_aug` is retained (the shadow lever is still in rows 2/4/5).

**3D-Front-IID added as a thesis contribution** (Ch4 sec:front3d, Ch1
contributions, `fig:front3d`). All corpus numbers in the draft are measured from
the rendered dataset, not asserted: 960 rooms, 2,496 accepted views (103 of 2,599
quarantined by the validator), 5,056 cross-illuminant pairs, 6,272 key lights,
median pair rg-chromaticity separation 0.23, enforced minimum 0.055, 512x512 @ 64
spp. Figure builder: `tests/viz/build_front3d_figure.py` (re-runnable; rescans the
corpus). Bib entry `fu213dfront` added.

**Still blank, needs the five runs:** every cell of `tab:lever_sweep`. Also still
blank from A00b: `tab:arap_standard`, `tab:arap_thesis`, `tab:maw`, `tab:iiw`.

---

## A00e. Table A provenance and interpretation correction (2026-07-12)

- Figure 5.8 is regenerated from `v17_41/42/43/44/checkpoint_iter_40000.pth`
  and lives in `tests/visualizations/tableA_40k/` and
  `documents/thesis/images/tableA_40k/`. Its manifest records the scene,
  illuminant conditions, and checkpoint paths. Luminance and chroma-cast CoV
  are computed from raw predictions before display normalization. The selected
  scene is qualitative only and is not used to infer the full-split Cast_RMS
  ranking.
- The old `draft25` asset roots were renamed `legacy_draft25`. They remain only
  for figures that have not yet been regenerated; Chapter 5 Figure 5.8 no
  longer references them.
- Table 5.7 retains its measured values, but the colour-skip axis is not a
  standalone causal comparison: rows 43/44 took the fresh-optimizer fallback
  after the head parameter set changed, while rows 41/42 loaded optimizer state.
  The clean CARI comparisons are 41→42 and 43→44.
- The white-balanced ARAP values were measured by the multi-light constancy
  evaluator (31 groups), so they are internal accuracy guards, not a direct
  all-scene reproduction of a published ARAP table. IIW WHDR is ancillary and
  not used for Table-B or final-model selection.

## A00c--A00d. Superseded Table-A interpretation (2026-07-12)

The earlier Table-A notes in this section were superseded by A00e after the
provenance audit. Do not use their former claims of a fully LR-matched
colour-skip axis, direct published ARAP comparability, or IIW-driven model
selection. The current source, limitations, Figure 5.8 provenance, and
interpretation are recorded in A00e and Chapter 5.

## A00b. Protocol overhaul + full table blank-out (2026-07-12)

Chapter 5 tables and their number-citing prose were reset to blank ("-") ahead
of a full rerun, because several protocol facts changed since the numbers
currently in git history were measured:

- **ARAP split in two.** Standard protocol (white-balanced input, comparable to
  published Ordinal/CD-IID numbers) vs. thesis constancy (raw colored input,
  $C_{\text{arap}}$/$\text{Cast}_{\text{RMS}}$, our own metric). A real bug
  meant `--white_balance` was silently ignored by the constancy eval path
  (`eval_arap_constancy`, not the unused `eval_arap`) until fixed today — any
  pre-fix "standard" numbers were actually raw-colored and mislabeled.
- **Citation targets pinned per benchmark**, see Chapter 5 "Which published
  table each benchmark cites": MAW → CD-IID Table 1; ARAP → **Ordinal Shading
  Table 1, not CD-IID Table 2** (CD-IID modifies the scene set — drops
  redundant illuminations, adds MIST — which we cannot reproduce; Ordinal uses
  the original Bonneel 52-scene set, matching ours); IIW → Ordinal Table 2
  zero-shot (24.9) alongside the existing yuan25texture roster; MID → no
  external citation (training data for us AND CD-IID).
- **Per-benchmark inference resolution**, not a shared 1280: MID/IIW 1280, ARAP
  1500 (matches Ordinal's cap; measured identical to 1280 on the full set),
  MAW 512 (matches CD-IID's protocol; their own scoring code downsamples every
  prediction to 320×240 regardless of inference resolution, confirmed by
  reading `numerical_albedo.py`).
- **Two new locally-run baselines**: CRefNet (`checkpoints/CRefNet/final_real.pt`,
  adapter at `tests/eval/crefnet_adapter.py`) and Ordinal Shading stage-1
  (`tests/eval/ordinal_adapter.py`, weights via `torch.hub` from the CD-IID
  repo's GitHub releases). Both smoke-tested end-to-end on ARAP + MAW.
  Ordinal's LMSE landed at 0.0212 on a 2-scene sample vs. their published 0.021
  on the full 52-scene set — strong evidence the resize/gamma conventions are
  faithful. CD-IID's full 5-stage cascade is NOT integrated (stretch goal,
  Ordinal's stage-1 covers the citation-critical comparison).
- **MAW AMP** (`--amp` on `eval_maw.py`) verified numerically free (<0.1% drift)
  and now default-on for MAW inference.
- **Open item, not yet resolved:** CD-IID's Table 1 footnote claims grayscale-
  shading methods (Ordinal Shading included) have a mathematically *fixed*
  chromaticity error (6.56) since dividing by scalar shading preserves the
  input's own R/G,B/G ratios exactly. A 5-image smoke test measured 3.08 for
  our local Ordinal Shading run — plausibly small-sample noise, but check
  whether `run_ordinal`'s `np.clip(albedo, 0, 1)` (needed since `image/shading`
  can exceed 1) is breaking that exact proportionality once the full run lands.

**Tables blanked**: `tab:mid`, `tab:arap_standard` (new), `tab:arap_thesis`
(renamed from `tab:arap_colored`, merged the old `tab:arap`'s constancy columns
in), `tab:maw`, `tab:iiw`, `tab:ablation` (Table A — training in progress at
time of writing), `tab:lever_sweep` (Table B — not yet started). Number-citing
prose paragraphs in each section were removed or neutralized to structural
claims; qualitative figure references and captions were left untouched (out of
scope for this pass — the underlying images are not being regenerated yet).
Chapter 7 (conclusion) had one stray restatement of the old MID 26% figure,
also neutralized.
Roster to fill once reruns land: Ours, Marigold-App, Marigold-Light, CRefNet,
Ordinal Shading (all five now runnable locally on every benchmark).
Full protocol reference: `documents/evals/BENCHMARK_PROTOCOL.md`.

---

## A00. "Ours (full model)" DECISION + table-fill status (2026-07-04, later)

- **"Ours (full model)" = A4 (full-CARI, v17_44), NOT B_combo/V23.** Rationale:
  A4 wins the two REAL benchmarks the thesis claim rests on — MID C_mat 0.173 and
  MAW (ΔE 3.64 ≈ Marigold 3.625, intensity 0.166 ≪ 0.461/0.54). B_combo (V23 =
  flat+sh_inv) wins synthetic ARAP (C_arap 0.087, si-RMSE 0.229) + WHDR 0.264, but
  loses MID (0.199) and MAW (3.98); it is reported as the structure-oriented
  Table-B alternative, not the headline. Benchmark tables + qualitative figures
  all use A4 as "Ours".
- Benchmark tables now FILLED with A4 draft25 (25-subset) temp numbers; Protocol
  column deleted; CD-IID/Ordinal cited (MAW ΔE/int, ARAP LMSE from CD-IID Tab 1/2);
  MAW/IIW cited rosters below rules.
- ⚠ STILL cross-regime: A4 is 25-subset, Marigold numbers are full-split. For a
  strict head-to-head, re-run Marigold-App/Light on the 25-subset (slow: diffusion)
  OR run A4 on full split. Flagged; numbers are a draft.
- Fig 5.6 tension REPLOTTED on trustworthy axes (MID C_mat vs ARAP si-RMSE), not
  WHDR — WHDR shown unreliable (cite sato25metrics: subjective/relative/hue-blind;
  MAW motivation). Passing levers beat Marigold si-RMSE (0.23/0.25 vs 0.28).
- Figures re-rendered SHARP from hires per-condition albedo (tests/viz/
  render_mid_hires.py → legacy_draft25/hires/); ablation shows 3 conditions + explains
  colour-skip black output; levers sharp + 60k noted.

## A0. legacy_draft25 update status + the protocol-consistency BLOCKER (2026-07-04)

Updated from legacy_draft25 (25-sample subset, matched-step retrained checkpoints):
- **Ablation table (tab:ablation)** — all 4 rows A1–A4 (v17_41–44). CARI (Row1→Row4)
  MID 0.234→0.173 (−26%); on this subset the secondary axes also improve
  (Cast 0.065→0.045, WHDR 0.318→0.287). NOTE: the WHDR/Cast direction REVERSED
  vs the old full-split preliminary (which showed CARI hurting them) — this is
  either a real matched-step effect or 25-sample noise; **needs full-split.**
- **Lever sweep (tab:lever_sweep)** — control + 5 levers (v17_20/21/22/23/24/25).
  Pass: flatness, flatness+shadow-inv (combo = best structure). Fail on all axes:
  explain companion (kill-gate, v17_24) and ordinal shading (v17_25).
- MID + IIW running text reframed to internal-ablation comparisons (removed the
  cross-regime Marigold claims).

**BLOCKER — do not fill the benchmark tables / tension plot until resolved:**
legacy_draft25 = 25-scene subset; the report's Marigold/SOTA numbers are FULL-split.
They are not comparable. On the subset, A4 MID C_mat = 0.173 > Marigold-App
full-split 0.140 — i.e. mixing them would falsely show Marigold winning
constancy (the thesis's core claim). To fix, run Marigold-App/Light (and the
run-local competitors CD-IID, Ordinal Shading) on the SAME 25-sample subset, OR
run our checkpoints on the full 30-scene MID split. THEN fill the benchmark
tables, the tension plot (fig:tension, still on old preliminary coords), and the
ARAP/MAW running-text comparisons coherently.

**v17_25 (ordinal) verdict:** worst MID (0.266) and worst IIW WHDR (0.354) of all
10 — a failed lever, not an "Ours full model" candidate. "Ours (full model)"
remains undetermined (A4 wins constancy; B_combo wins structure; pending v26).

## A. Numbers to fill / replace after the step-matched retraining

The per-benchmark running text currently analyses the **ablation checkpoints**
(no-CARI baseline vs. the CARI-constrained checkpoint), which are measured and
therefore stated as final ablation data. The **"Ours (full model)"** row in each
benchmark table is the retrained model carrying the gate-passing levers; those
cells are blank until that run exists.

| Table | Cell(s) to fill | Currently measured (ablation, keep as ablation data) |
|-------|-----------------|------------------------------------------------------|
| 5.1 MID | Ours (full); CD-IID; Ordinal Shading | CARI ckpt C_mat 0.107; baseline 0.150 |
| 5.2 ARAP indoor | Ours (full); CD-IID; Ordinal Shading | C_arap 0.151–0.166; Cast 0.063–0.071; si-RMSE 0.332; LMSE 0.044 |
| 5.3 ARAP coloured | Ours (full); CD-IID; Ordinal Shading | Cast 0.060; C_arap 0.178 |
| 5.4 MAW | Ours (full); CD-IID; Ordinal Shading; published PIE-Net/CRefNet/ReasonX ΔE + SI-MSE (with protocol footnotes) | CARI ckpt ΔE 4.023, SI-MSE 0.402 |
| 5.5 IIW | Ours (full); Ours-IIW-ft; CD-IID; Ordinal Shading; cited WHDR transcriptions | baseline WHDR 0.290; CARI 0.303 |
| 5.6 Ablation | Rows 2 (cross-render only) and 3 (colour-skip only) | Row 1 = 0.150/0.059/0.290; Row 4 = 0.107/0.071/0.303 |
| 5.8 Lever sweep | Re-run full-protocol under the corrected recipe (main-schedule data mix + LR) | quick-subset numbers currently in table |
| Fig 5.4 (tension plot) | Update the four scatter coordinates | uses ablation coords |

Config rows that produce these numbers: v17_20–v17_24 (unified Table B),
v17_25 (ordinal-shading screen), v17_26 (IIW ordinal-hinge FT).

## B0. Figures — CURRENT STATE (populated from legacy_draft25, 2026-07-04)

All Chapter-1/5 figures now use legacy_draft25 predictions (built by
`tests/viz/build_thesis_figures.py` → `documents/thesis/images/legacy_draft25/`).
"Ours" in every figure = the full-CARI model (legacy_draft25 label A4_fullCARI) as a
temporary stand-in; SOTA panels are N/A placeholders. To REPLACE later:
- Swap "Ours" to the final chosen model once determined (pending v26), rerun the
  builder (change `A4_fullCARI` in the script).
- Fill the SOTA N/A panels (comp_grid.jpg columns, maw_ours.jpg Marigold panel)
  once Marigold/CD-IID/Ordinal predictions on the SAME images exist.
- New reference-style figures added: `datasets.jpg` (benchmark overview, Mesh
  §2.4 style) and `comp_grid.jpg` (multi-method albedo grid, CD-IID Fig 7 style).
- The old-checkpoint crops (images/mid_row_*, arap_bedroom_*, iiw_sheet_top5) and
  `tests/visualizations/maw_sheets/` are now UNREFERENCED and can be deleted.

## B. Figures to regenerate from the retrained checkpoints

- Ch1 teaser strips — regenerate with larger tiles + clean column labels.
- Fig 5.1 (MID strips) — larger tiles, more scenes.
- Fig 5.2 (ARAP bedroom) — 2–3 scenes, annotated wall patches where the
  constancy difference is most visible.
- Fig 5.3 (MAW) — pick 2–3 scenes with more visible cross-model colour
  difference; DSC4366 is an easy scene. Candidates in
  `tests/visualizations/maw_sheets/{v17_50k,marigold_app}/`.
- Fig 5.5 (IIW) — regenerate as a 3-column strip (input | ours | Marigold-App)
  so the shadow-contrast difference is directly visible.
- Fig 5.6 (ablation MID) — add strips for the intermediate ablation cells
  (Rows 2 and 3) once evaluated.

## C. Evidence gaps (claims needing empirical backing)

- **Ch3, DINOv2 illumination-robustness claim** (was `chapters/Chapter3.tex`
  ~line 161): the robustness argument is currently a priors/architecture
  argument, not an in-project measurement. Either add a small probe (feature
  cosine similarity of the frozen encoder across MID illuminant conditions on
  matched scenes) or soften the claim to cite DINOv2's reported invariance only.
- **Ch5, Marigold-Light Cast blow-up on ARAP**: exact cause of the
  out-of-range Cast value not pinned down (suspected: negative / out-of-gamut
  shading channels after the sRGB-to-linear conversion on HDR renderings).
  Verify before treating the parenthesised value as anything but "output-space
  failure, excluded from comparison."

## C1. Inference-resolution robustness (measured 2026-07-04)

Finding that justifies standardizing inference resolution across benchmarks:
- MID native mip2 = 1500px. Benchmarks were previously each run at a *different*
  inference resolution (MAW 512, IIW ~1024–1280, MID native 1500). Fixed MAW to
  aspect-preserving max-1280/min-1024 via eval_maw.py --infer-max-size (default
  now 1280); eval_mid_constancy.py gained --infer-max-size (default None=native).
- MID C_mat is FLAT vs inference resolution (v17_20, 6-scene subset):
  1024→0.2300, 1280→0.2312, 1500→0.2317, 1792→0.2306 (spread 0.7% = noise).
  Cast_RMS 0.334–0.339, LMSE 0.302–0.304 — both flat too.
- MAW visual: sharpness plateaus by ~1280–1792; 2560 adds nothing visible;
  MAW ΔE flat/jittery across 512–2560 (metric downsamples anyway).
- CONCLUSION: constancy does not degrade and sharpness does not improve above
  ~1280 in the 1024–1792 range → adopt 1280 as the single inference protocol
  (or keep native 1500; both sit on the plateau). Raising resolution above this
  buys nothing. Restate the setup section around one inference resolution.

## D0. Methods-overview table (Table 5.x, tab:methods) — cells to source

Our row is exact (18.5M trainable / 323M total, measured from the v17_20
checkpoint: encoder 304.37M frozen + trunk 17.28M + albedo_head 0.64M +
shading_head 0.59M). External cells left "-" pending sourcing from each paper:

- Marigold params ~865/~1300 are the Stable-Diffusion-2 backbone estimate;
  confirm Marigold-IID's exact backbone + trainable count from Ke et al. 2025.
- CD-IID / Ordinal Shading (Careaga): training-data mix and param counts.
- PIE-Net: training data (CGIntrinsics/IIW?) + params + output space.
- CRefNet: confirm full training set (IIW + synthetic?) + params + output.
- ReasonX: architecture family, training data, params, output — paper not yet
  fully catalogued (bib author list also incomplete).

## D. Appendix per-scene tables

- Per-scene MID C_mat breakdown — extract per-scene numbers and populate.
- Per-scene ARAP breakdown — same.

## E. Chapter 6 (applications) — evals and figures to produce

- Relighting-recomposition: run on held-out MID pairs; 4-panel figure
  [input a | input b | a re-lit to b | real b].
- Residual attenuation: 3-panel strips [input | alpha=0.5 | alpha=0] on 3–4
  real photos.
- Shadow-lightening: before/after pair on 2 scenes.
