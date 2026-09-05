# Phase A findings — measurement fixes (no GPU)

Date: 2026-09-05
Scripts: `tests/eval/arap_preprocess.py`, `tests/eval/build_arap_colour_subset.py`,
`tests/eval/mid_colour_stratified.py`

---

## 1. ARAP is three datasets wearing one name — and the evaluator's mask breaks on it

### 1.1 The encodings

GT albedo is stored in three mutually incompatible encodings:

| Encoding | n | albedo max | Note |
|---|---:|---|---|
| `radiance_179` | 40 | ~0.00559 | divided by 179 (Radiance luminous-efficacy constant) |
| `ldr_srgb` | 7 | 1.0 | uint8/255, de-gamma 2.2 |
| `unscaled` | 4 | 0.80–1.40 | `attic`, `bedroom2`, `corridor`, `kitchen` |

`1/179 = 0.00558659`. Rescaling that group by 179 puts its maxima in
**[0.571, 1.000]** — a clean physical albedo range. That is independent
confirmation of the diagnosis, not a fitted correction.

### 1.2 The consequence: the valid mask is meaningless

`eval_arap.py:915` and `:1292` mask with a **fixed absolute** threshold
`alb_lum > 0.004`. Against a GT whose maximum is 0.00559, that keeps only the
brightest sliver; against a GT whose maximum is 1.0, it keeps everything.

| Encoding | median valid fraction (old) | median valid fraction (fixed) |
|---|---:|---:|
| `radiance_179` | **1.5%** | 96.1% |
| `ldr_srgb` | 99.9% | 99.5% |
| `unscaled` | 99.0% | 98.7% |
| **spread across encodings** | **68×** | **1.0×** |

- **6 scenes had a completely EMPTY mask** (albedo max below the threshold):
  `iron katie lobby redhead revolution skin strawberries blenderscene`
- **23/51 scenes kept <5%** of the frame; 32/51 kept <20%.

So the reported ARAP table averages metrics computed over ~1.5% of the frame for
40 scenes with metrics computed over ~99% for 11. **This affects every masked
metric — LMSE, RMSE, si-RMSE, SSIM — not only SSIM.**

The existing `_fixed_ssim` note (eval_arap.py:478) found the *SSIM symptom* of
this and attributed it to zero-padding. The zero-padding is real, but the root
cause is the mask, and the fix must be applied to all four metrics.

**Consequence for the thesis: `tab:arap_accuracy` numbers are not trustworthy and
must be regenerated.** The already-recorded direction (local SSIM 0.80–0.82 vs
published 0.69–0.78) is consistent with the inflation this bug produces.

### 1.3 Two further data hazards

- **Input/albedo scale mismatch**: the input/albedo p99.9 ratio spans **0.16 to
  3.6e5** across scenes; ~24 scenes have input NOT on the albedo's scale. LMSE
  and si-RMSE absorb a per-image scale so they survive, but *any* absolute
  threshold on either image is invalid.
- **`breakfast` ships duplicates**: `.hdr` (720×1280) and `.jpg` (768×768) for
  both base and albedo. `INPUT_EXTS` puts `.hdr` first so `.hdr` wins and
  base/albedo stay self-consistent — but `ARAP_types.json` tags it `ldr: True`,
  so the JPG was probably intended. Flagged, not silently switched.

### 1.4 The fix

`tests/eval/arap_preprocess.py` canonicalises GT albedo to a [0,1] reflectance
range by detecting the encoding and undoing the known 179 constant, then masks
*relative* to canonical white (default 2%). It is **additive** — `eval_arap.py`
is untouched, so previously reported numbers stay reproducible and callers opt in.

Values are deliberately **not clipped** to [0,1]: `kitchen` (1.34) and `bedroom2`
(1.40) legitimately exceed 1, and clipping would silently alter GT.

---

### 1.5 Measured impact on the reported numbers

`v17_34/checkpoint_latest.pth`, full ARAP, identical settings apart from the mask:

| | old mask | canonical mask | change |
|---|---:|---:|---|
| images evaluated | 136 | **157** | **+21 recovered** |
| Albedo LMSE | 0.0347 | 0.0622 | **+79%** |
| Albedo RMSE | 0.1550 | 0.1364 | −12% |
| Albedo si-RMSE | 0.2813 | **0.5209** | **+85%** |
| Albedo SSIM | 0.8152 | 0.4855 | **−40%** |
| Albedo `_fixed_ssim` | 0.6327 | 0.6311 | **−0.3%** |

Two things to read here.

**The 21 recovered images** are scenes the old mask dropped entirely (`valid_mask.sum() < 88`
after an empty or near-empty threshold) — they were silently absent from every
reported ARAP average.

**`_fixed_ssim` barely moves (−0.3%) while plain SSIM moves −40%.** That is the
control: `_fixed_ssim` already rescales by the GT's own maximum, so it was
structurally immune to the encoding problem. The metrics that were *not* scale-robust
move a lot; the one that was does not. This is independent evidence the fix targets
the real defect rather than perturbing everything indiscriminately.

**Which number is "right" is a separate question.** The canonical mask scores ~96–99%
of the frame including dark regions where albedo is genuinely hard, so errors rise;
the old mask scored a bright sliver. Neither is self-evidently the published ARAP
protocol, and we do not know the mask Ordinal Shading et al. used. What is not
defensible is the status quo: a 68× coverage spread across scenes plus 21 silently
dropped images.

Mitigating: every model in the thesis was scored through the *same* broken mask, so
relative rankings may partly survive. But the mask is biased toward bright regions,
which need not be neutral across methods, and no absolute number or per-scene
analysis survives. Regenerate before publishing.

---

## 2. ARAP-Colour: a real external benchmark that actually varies illuminant colour

Measured as median `chroma(I / A_gt)` over the canonical mask, per light variant.
Chromaticity is scale-invariant, so §1.3's scale mismatch cannot bias it — but the
*mask* selecting the pixels is not, which is why §1.4 was a prerequisite.

`documents/evals/arap_colour_subset.json` — **26 pairs across 17 scenes** at
gap ≥ 0.15.

| Corpus | median pair gap | vs MID |
|---|---:|---:|
| **MID** (train pairs *and* Cast_rel benchmark) | **0.021** | 1.0× |
| ARAP, all measurable pairs (n=117) | 0.088 | 4.2× |
| ARAP indoor only (n=74) | 0.070 | 3.3× |
| 3D-Front v1 | 0.151 | 7.2× |
| 3D-Front v2 pilot | 0.243 | 11.6× |

Strongest indoor (in-domain) pairs: `conference` 0.668, `kitchen` 0.572/0.568,
`bread` 0.303, `bedroom2` 0.300, `strawberries` 0.228, `whiteroom` 0.212,
`postit` 0.180, `oldclassroom` 0.168.

> These supersede the exploratory numbers computed earlier in the session
> (e.g. `kitchen` 0.018 → 0.568). The earlier pass used ad-hoc thresholds on
> un-canonicalised albedo and its masks were wrong.

---

## 3. MID stratified by illuminant colour — the mechanism shows up

`documents/evals/mid_colour_stratified.json`. Scenes split into terciles by their
own median illuminant chromaticity gap (range 0.0165–0.0860), then existing
per-scene `Cast_rel` re-aggregated. No inference.

| Method | LOW | MID | HIGH | **HIGH/LOW** | pearson r | shading |
|---|---:|---:|---:|---:|---:|---|
| **Ours (full)** | 0.366 | 0.376 | 0.521 | **1.42** | 0.387 | RGB |
| **Ours (base CARI)** | 0.368 | 0.379 | 0.528 | **1.43** | 0.386 | RGB |
| CRefNet | 0.313 | 0.310 | 0.443 | **1.42** | 0.426 | **grayscale** |
| Ordinal Shading | 0.453 | 0.489 | 0.705 | **1.56** | 0.591 | **grayscale** |
| Marigold-Light | 0.370 | 0.384 | 0.422 | 1.14 | 0.172 | RGB |
| Marigold-App | 0.368 | 0.316 | 0.381 | **1.03** | 0.289 | RGB |
| CD-IID | 0.309 | 0.295 | 0.315 | **1.02** | 0.109 | RGB + chroma stage |
| RGB→X | 0.361 | 0.328 | 0.325 | **0.90** | −0.028 | RGB |

Three readings, all load-bearing:

**(a) CARI did not confer colour invariance.** `Ours (full)` 1.42 vs
`Ours (base CARI)` 1.43 — the CARI axis moves the colour-dependence by nothing.
This is the cleanest available evidence for the transfer diagnosis: CARI saturates
synthetic colour invariance (`front3d val inv_gap 0.0013`) while real-domain
colour-dependence is unchanged.

**(b) The two grayscale-shading models are the two worst.** CRefNet (1.42) and
Ordinal (1.56) — exactly what Chapter 3 predicts: single-channel shading cannot
represent a coloured illuminant, so its hue must land in the albedo. The
mechanism hypothesis is already supported by data on disk.

**(c) Three methods are genuinely colour-invariant**: CD-IID 1.02, RGB→X 0.90,
Marigold-App 1.03. CD-IID is the one with an explicit shading-chroma stage — the
same factorisation V21 proposes.

**Caveat — low power.** MID's entire per-scene colour range is 0.0165–0.0860, so
these terciles compare weak against slightly-less-weak. The effect is large and
consistent, but ARAP-Colour (§2) and the 3D-Front v2 held-out split are where it
should be confirmed at a real colour range.

> Note: the per-scene median gap here (0.021) is smaller than the all-pairs
> pooled figure quoted earlier (0.030) because this excludes MID's `skip_list`
> flash indices `{2,3,20,21,24}` — matching what training and evaluation actually
> consume — and takes a per-scene median rather than pooling all pairs.

---

## 4. What this changes

1. **ARAP accuracy numbers must be regenerated** with the canonical mask before
   any of them appear in a paper. Route through `THESIS_RESULTS_TODO.md`.
2. **ARAP-Colour is now the colour benchmark**, replacing the claim that no real
   colour-varying external test exists.
3. **The mechanism test is already half-confirmed** — grayscale-shading models
   are the colour-dependent ones — which raises the value of the planned
   grayscale-shading ablation of our own model, since that converts a
   cross-method correlation into a controlled result.
4. **CARI's null effect on colour-dependence is now measured, not inferred.**
   Phase B (chroma-stratified sampling + per-direction tinting) is the test of
   whether that is fixable from the data side.

---

## 5. Reproduce

```bash
python tests/eval/build_arap_colour_subset.py --min-gap 0.15
python tests/eval/mid_colour_stratified.py
```

Both are read-only over existing artefacts and write JSON to `documents/evals/`.
