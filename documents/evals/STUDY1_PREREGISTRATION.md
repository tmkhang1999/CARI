# Study 1 preregistration — does the chroma half of L_explain reduce colour-dependence?

**Written 2026-09-08, BEFORE any row of this study finished training.** That is the
point of the document. This project has already been burned once by choosing the
readout after seeing the numbers: the original `Cast_RMS` pooled chroma across
materials, which paid models for *collapsing* colour, and it bolded the visually
broken `v17_42` as the best model. Fixing the endpoint, the guards and the decision
rule in advance is the cheapest available protection against repeating that.

Nothing here may be revised after the first row's metrics are read. If something
turns out to be mis-specified, the revision goes in a dated section at the bottom
with the reason, and the original stays.

---

## 1. The claim under test

`cari_explain` reduces both the image ratio and the shading ratio through `lum()`.
A purely chromatic change between the two frames therefore gives `ri = rs = 0` and
contributes **nothing** to the gradient — colour error is not down-weighted, it is
structurally invisible. `L_inv` does not cover the gap either: a constant albedo
cast lies in its null space (`midintrinsic_dataset.py:50`).

Measured on the shipped model:

| quantity | value |
|---|---|
| chroma change explained by shading | 47.9% |
| unpenalised chroma residual | 0.0741 |
| penalised luminance residual | 0.0757 |

The unpenalised error is the same size as the penalised one. **Hypothesis:** adding
`L_chr_explain` closes that gap and reduces the model's colour-dependence.

## 2. Design

Six configs, three seeds each, 18 runs. All fork from **one** shared base,
`checkpoints/v17_60/checkpoint_iter_19000.pth` (curriculum phases 1–2: Hypersim
only, then +InteriorVerse). No cross-render pair is emitted in those phases, so no
colour-constancy mechanism is active in the shared base — every difference between
rows arises in phase 3.

| config | L_inv | L_explain | L_chr_explain | role |
|---|---|---|---|---|
| v17_61 | 0    | 0    | 0    | control: no CARI |
| v17_62 | 0.5  | 0.25 | 0    | published CARI |
| v17_63 | 0.5  | 0.25 | 0.25 | **proposed loss** |
| v17_64 | 0.5  | 0    | 0.25 | chroma instead of luminance |
| v17_63b| 0.5  | 0.25 | 0.6  | dose |
| v17_63c| 0.5  | 0.25 | 0.1  | dose |

Seeds 42/43/44, passed via `--seed`; the config is otherwise identical across
replicates. Held fixed across every row: data mix (incl. front3d v1), colour path
(`albedo_rgb_skip`, `lambda_a_chroma` 0.2), LR schedule, optimizer path, 21k phase-3
steps. No Table-B refinement levers.

**Seeds are the unit of replication and they matter here.** Until this study,
`train.seed` and `deterministic` were dead settings — never read — so no v17 run was
reproducible. Fixed 2026-09-08; the seed is now recorded in every checkpoint.

## 3. Primary endpoint (ONE, fixed in advance)

**The change in the MID HIGH/LOW `Cast_rel` tercile ratio, `v17_63 − v17_62`,
paired within seed.**

Scenes are split into terciles by their own measured illuminant chromaticity gap;
the ratio asks whether chroma drift *tracks* illuminant colour. Reference values
under the same protocol: ours 1.42, CD-IID 1.02, RGB-X 0.90, Marigold-App 1.03.

**Why not mean `Cast_rel`:** it can be driven down by desaturating, which is exactly
the failure the corrected metrics exist to catch (CRefNet posts good pooled
invariance at Chroma_fid 0.484). A mean that improves while the ratio does not is a
null, and will be reported as one.

## 4. Guards — ALL must hold for a positive result

Every guard is reported independently; a single failure means the result is not a
gain, whatever the primary endpoint did.

| guard | threshold | what it catches |
|---|---|---|
| `Chroma_fid` | must not fall > 0.05 | desaturation dressed as invariance |
| LOW tercile `Cast_rel` | must not rise > 0.01 | a "gain" that is a pivot — weak-colour scenes traded for strong |
| `C_mat` | must not rise > 0.01 | intensity-axis regression; this is the axis we currently lead |
| paired bootstrap 95% CI on the ratio change | must exclude 0 | noise |

Implemented in `tests/eval/study1_readout.py`, written and tested against synthetic
rows with known ground truth *before* any real result existed.

## 5. The noise floor gate

`|v17_62_s42 − v17_62_s43|` on the primary endpoint — two runs of an **identical**
config — is measured **first**, before any treatment row is read.

- If the seed noise floor is **larger than** the effect that CARI itself produced
  in Phase A (1.43 → 1.42, i.e. 0.01), then a single-seed comparison of any two rows
  is uninformative and this is stated plainly in the write-up.
- The primary endpoint is then judged only against the 3-seed paired distribution,
  never against a single pair of runs.

## 6. Secondary endpoints (reported regardless of the primary)

Not used to rescue a null primary. Reported to characterise what the loss did.

1. **Mechanism** — the % of chroma change explained by shading (47.9% baseline),
   re-measured post-training. This is the quantity the loss directly targets, and
   it can move even if the downstream benchmark does not. If the loss falls but this
   does not move, something is wrong with the wiring, not the hypothesis.
2. **MAW** — real measured albedo, chromaticity metric. The external home for the
   colour claim; MID is in-domain and CD-IID's training data.
3. **ARAP** — canonicalised masks (`--canonical_mask`), colour subset and
   hard-shadow subset, both the standard white-balanced and thesis-local tables.
4. **IIW WHDR** — accuracy guard. Must not regress materially.
5. **front3d v2 held-out** (269 views) — out-of-training synthetic colour.
6. **Checkpoint stability** — 36k/38k/40k for the headline rows, so the result is
   not a single-checkpoint artifact.

## 7. Decision rules, fixed in advance

- **Positive** — primary negative, CI excludes 0, all guards hold, at n=3:
  the chroma term reduces colour-dependence. Dose-response (63c/63/63b) is then
  reported as supporting evidence, not as the primary.
- **Null** — primary not separable from 0 at n=3: reported as a null. The
  mechanism statistic (§6.1) then distinguishes *"the loss did not train"* from
  *"the loss trained and it did not help"*. These are different findings and the
  write-up must say which.
- **`(63 − 64)` ≈ 0** — the luminance half contributes nothing beyond `L_chr_explain`
  and `L_inv`, and the honest claim becomes **replace** `L_explain`, not **add** a
  term. This must be reported even though "add" is the more attractive story.
- **Guard failure with a negative primary** — reported as NOT a gain, with the
  failing guard named. Specifically: a `Chroma_fid` drop is reported as
  desaturation, which is the CRefNet failure, not a contribution.

## 8. What would falsify the hypothesis

Stated so it cannot be explained away later:

- `L_chr_explain` trains down (the loss curve falls) **and** the mechanism statistic
  improves, **but** the MID ratio does not move → the mechanism is real but does not
  reach the benchmark. The blind-spot argument survives; the *remedy* does not.
- The ratio improves only because the LOW tercile regressed → the loss traded
  weak-colour scenes for strong ones. Fix is to widen coverage, not shift the mean.
  The same criticism would then apply to the 3D-Front rigs.
- The grayscale-shading control (a 1-channel shading head cannot explain a colour
  change; `cari_chr_explain` raises rather than silently broadcasting) fails to
  behave as predicted → the term is not doing what it is claimed to do.

## 9. Compute

18 runs × ~6h ≈ 108 GPU-hours ≈ 54h wall on 2 GPUs. Order is fixed in advance so a
partial result is still interpretable and the noise floor lands first:

1. `62_s42`, `62_s43` — noise floor
2. `63_s42`, `63_s43` — primary at n=2, paired
3. `62_s44`, `63_s44` — n=3 on both
4. `64` × 3
5. `63b` × 3
6. `61` × 3, `63c` × 3

---

## Revisions

*(none yet — any change after the first metrics are read goes here, dated, with the
reason, and the original text above stays untouched)*
