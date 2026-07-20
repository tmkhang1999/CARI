# Plan: hard-shadow augmentation on 3D-Front rows (v17_35)

2026-07-13. **Status: IMPLEMENTED, S1 SMOKE TEST FAILS — DO NOT LAUNCH.**
Code (dataset tags, loss config, train_v17.py shadow branch, TB logging, v17_35.yaml) is
all in place and syntax/config-verified, but the pre-registered gate in §4/§7 says launch
only if smoke tests pass, and S1 fails by roughly 25–60x with no viable tuning found within
the existing mechanism. See §9 for the full finding. S2/S3 were NOT run (blocked by the gate).
Decision context: Table-B results (`tests/visualizations/all_models_4bench_full_B`)
+ domain-gap measurement (`scripts/measure_shading_domain_stats.py`).

## 0. TL;DR

Reuse the EXISTING synthetic relight machinery (`src/data/shadow_aug.py::
sample_relight_field` — it already generates hard-edged fields: sigmoid sharpness
5–40, half-plane sunbeams AND elliptical occluder blobs, warm/cool tints) on
**front3d rows at training time**. No re-rendering. This targets the measured gap:
front3d matches ARAP on the color axis (illum-chroma gap 0.151 vs 0.100 indoor)
but has **5–8× less hard shadow structure** (hard-edge frac 0.022 vs 0.115 indoor
/ 0.186 outdoor) — and C_arap's failure mode lives exactly in that missing
structure.

Two levers, both revived by σ:
- **Invariance channel** (was saturated: front3d val `inv_gap` = 0.0013): a hard σ
  on a colored-lit front3d frame creates variation the model is NOT yet invariant
  to → gradient returns. Verified expectation, to be confirmed by smoke test S2.
- **Supervision channel** (was plateaued: val `alb_si_rmse` flat 14k steps): the
  σ branch can be anchored to front3d's EXACT GT albedo — supervised, not just
  self-consistent. This is the anti-washout ingredient: invariance alone can be
  satisfied by flattening (the levers' MAW-Int washout mechanism), L1-to-GT
  cannot. Hypersim's sh_inv never had this option (no colored-light GT albedo).

**Do NOT modify v17_34.** It keeps its clean factorial role (gentle-lever row;
(34−29) pure lever contrast). This plan is a NEW row **v17_35 = v17_29 + σ-branch
on front3d rows only** — single variable vs Row 3.

## 1. Why this design (one paragraph of evidence)

Table B: levers win C_arap colored (−14% vs control) but wash out MAW intensity
(+40%); front3d alone wins the cast/chroma axes (Cast colored −11%, best MAW ΔE
4.040, Int clean 0.512) but leaves C_arap nearly flat (−4%). LR exonerated
(constant 2e-5 verified in TB; same LR moved lever metrics). The missing C_arap
signal is hard shadow structure, which front3d's soft area-light renders lack
(measured 5–8×). σ injects exactly that structure into the colored-light domain —
the joint condition (colored illumination × hard shadow) that no current training
source contains and that indoor-ARAP-colored tests.

## 2. Mechanism spec (v1 — implement this)

For batch rows where `is_front3d == 1`, in the existing shadow branch
(`src/train_v17.py` ≈ lines 500–533):

1. σ-invariance (reuse): `L_sh_inv = shadow_invariance(a_anchor, A(I⊙σ), mask)`.
   `shadow_invariance` (`src/losses/flexible_loss_v17.py:472`) already
   scale-shift-aligns before the L1 — keep that; it blocks the desaturation cheat.
2. σ→GT anchor (new, small): `L_sh_gt = shadow_invariance(A_d_star, A(I⊙σ), mask)`
   on front3d rows only (A_d_star = exact GT albedo, in every batch already).
   Reusing `shadow_invariance` for this gives the same alignment convention
   (prediction is Q99-renormed; GT is absolute — alignment removes that gauge).

**v2 (optional, only if v1 smoke-clean AND time):** apply σ to ONE member of the
front3d CARI pair (`rgb2`) → joint color+shadow L_inv. Do NOT touch L_explain for
σ-batches (σ changes the known illuminant decomposition). Keep out of v1.

## 3. Exact code changes

1. **Per-row dataset tag.** Add `out['is_front3d'] = torch.tensor(1.0)` in
   `Front3DDataset.__getitem__` (`src/data/front3d_dataset.py`, after
   `prepare_training_tensors`), and `torch.tensor(0.0)` in every other dataset
   whose samples reach `MixedDataset` batches (hypersim, midintrinsic,
   interiorverse — grep for `prepare_training_tensors` callers). Default collate
   needs the key in ALL of them.
2. **Gate + mask** (`src/train_v17.py` ≈500–533). Current:
   `has_hypersim = (m_diffuse > 0).any()`; `sh_mask *= (m_diffuse > 0)`.
   New config semantics (read in `FlexibleLossV17.__init__` next to
   `lambda_shadow_inv`, `src/losses/flexible_loss_v17.py:219`):
   - `shadow_on_hypersim` (default **true** — preserves all existing configs)
   - `shadow_on_front3d`  (default **false**)
   - `lambda_shadow_gt_front3d` (default **0.0**)
   Row-selection mask = `(m_diffuse > 0) * shadow_on_hypersim
   + (is_front3d > 0) * shadow_on_front3d`; `shadow_active` requires that mask
   nonempty (not `has_hypersim` specifically).
   Add bottom-clip hygiene to `sh_mask`: `* (rgb_s.amax(dim=1, keepdim=True) > 1e-3)`
   (σ-blackened pixels must not be scored; the top clip `< 0.99` already exists).
   Add the GT term for front3d rows when `lambda_shadow_gt_front3d > 0`, logged as
   `losses['loss_shadow_gt_f3d']`, inside the SAME extra forward (no new forward;
   one extra backward term).
3. **Config `src/configs/v17_35.yaml`** = copy of `v17_29.yaml` (same resume
   lineage, `extend_iterations: 60000`, lr 2e-5, mix `0.3/0.3/0.15/0.25`, ALL
   Table-B lever lambdas 0) plus:
   ```yaml
   loss:
     lambda_shadow_inv: 1.0        # σ-invariance strength (same as the lever rows used)
     shadow_on_hypersim: false     # CRITICAL: front3d rows ONLY, else (35−29) is
     shadow_on_front3d: true       #   confounded with the v17_22-style Hypersim lever
     lambda_shadow_gt_front3d: 1.0
     shadow_start_iter: 0
   ```
   Header comment: role = single-variable "(35−29) = σ-on-front3d" row; relaunch
   requires `rm -rf checkpoints/v17_35/`.
4. **TB logging**: the front3d val probe (`_run_front3d_val_probe`,
   `src/train_v17.py:1180`) already logs `inv_gap`; ADD `inv_gap_sigma` there
   (same probe views, σ applied to one branch) so the revived-gradient claim is
   monitored during training, not just at smoke time.

## 4. Smoke tests (pre-registered pass criteria; run BEFORE launching)

- **S1 — correlation recheck (CPU, ~10 min).**
  `python scripts/measure_shading_domain_stats.py --apply-sigma`
  PASS: front3d+σ hard-edge median enters ARAP indoor IQR **[0.07, 0.18]**
  (bare front3d = 0.022). If low: raise `sample_relight_field` kwargs
  (sharpness range 5–40 → e.g. 10–60, `strength_range` up, `n_dir_max` up) —
  expose them via the yaml if tuned.
- **S2 — gradient revival (GPU, ~10 min).** Load `v17_44/checkpoint_iter_40000.pth`,
  16 front3d val views: report mean|A(I) − A(I⊙σ)| vs the clean inv_gap.
  PASS: σ-gap ≥ **5×** clean gap (expect ≳0.01 vs 0.0013). This proves the loss
  has gradient at the exact entry point where the channel was saturated.
- **S3 — training smoke (GPU, ~1h).** 500–1000 steps of v17_35: `loss_shadow_inv`
  and `loss_shadow_gt_f3d` present and nonzero on front3d-containing steps; no
  NaN; step time within ~15% of v17_29's (the branch adds one forward only on
  σ-active steps); VRAM < 24 GB; container RAM stable.

## 5. Launch + eval

```bash
rm -rf checkpoints/v17_35/
bash scripts/train.sh --version 17.35 --cuda <free-gpu> \
    --resume checkpoints/v17_44/checkpoint_iter_40000.pth
# after 60k:
python scripts/eval_all_models_4benchmarks_full.py \
    --ckpt v17_35=checkpoints/v17_35/checkpoint_iter_60000.pth \
    --out tests/visualizations/all_models_4bench_full_B --auto-resume
```
(--auto-resume merges into the existing Table-B result JSONs; rows 20/23/29/33
are skipped.)

## 6. Kill gates & success criteria (pre-registered, vs v17_20 control)

- Gate (same as Table 5.9): MID C_mat/Cast, ARAP C_arap/Cast (colored), MAW ΔE —
  no regression > ~10%.
- Watch explicitly: **MAW Int** — success requires it stays ≈ 0.51 (front3d-clean
  territory), NOT the levers' 0.72. If the σ branch washes out despite the GT
  anchor, that is a reportable negative (the anchor hypothesis failed).
- Primary success: ARAP C_arap colored < 0.111 (beats Row 3); strong success:
  ≤ 0.100 (matches levers) with clean MAW Int → new "Ours" candidate and the
  thesis closes the loop measured-gap → targeted fix → measured gain.

## 7. Timeline / feasibility (answers "can we do it now?")

- Implementation (Sonnet): ~half a day including smoke tests S1–S3.
- Training: ~1.5 days on one RTX 6000 (same as other Table-B rows).
  GPU0 is running the v17_34 relaunch; GPU1 frees when the SOTA eval finishes.
  v17_35 takes the first free GPU. Eval: ~0.5 day.
- **Decision gate: launch only if smoke tests pass AND ≥ 2.5 days remain before
  the results freeze.** Otherwise everything in this doc (with the measured
  numbers already in hand) becomes the future-work paragraph — the plan is
  written so it degrades gracefully into that.

## 9. S1 RESULT (2026-07-13): FAIL, and why — a geometric ceiling, not a tuning gap

`python scripts/measure_shading_domain_stats.py --apply-sigma --n-views 40`:
front3d+σ hard-edge median = **0.0218** (bare front3d = 0.0215 — σ moved it almost NOT AT
ALL) vs the target ARAP indoor IQR [0.07, 0.18]. Root-caused with three isolated experiments
(not composited into images — measuring the raw field `M` alone, so the finding is about the
generator, independent of front3d's own content):

1. **`sample_relight_field`'s DEFAULT settings produce ~zero measurable edge density at all**
   (200 draws, mean edge_frac = 0.0000, 100% of draws below the 0.001 near-zero floor). Not a
   `k=0` frequency issue — even draws WITH ≥1 directional component measure near-zero.
2. **Root cause: `sharp` (hardcoded range 5–40) operates in the [-1,1] NORMALIZED coordinate
   space, not pixel space.** At sharp=40 (the function's max), the sigmoid's ~0.01→0.99
   transition spans Δd ≈ 9.2/40 ≈ 0.23 normalized units ≈ **59 pixels** at 512px resolution —
   a broad, gradual gradient, not a hard boundary. Real cast shadows are 1–3px. Sweeping sharp
   up to 3000 (75x the current max) only reaches edge_frac ≈ 0.003 — still ~25–60x short of
   ARAP's 0.07–0.18 IQR.
3. **Stacking more components does NOT help — it gets WORSE.** `M = M * Mc` compounds
   multiplicatively; past ~12 overlapping components the product saturates into the `m_min`
   clamp over large contiguous areas, which REMOVES local gradient (a clamped/flat region has
   zero edge density) rather than adding it. n_components 24/40/60 measured edge_frac ≈ 0.

**Why this is a geometric ceiling, not a parameter-tuning problem**: `edge_frac` measures
boundary-pixel DENSITY, which for a small number of smooth analytic primitives (half-planes,
ellipses) is bounded by curve-length/area, independent of sharpness. Real ARAP scenes get
their 5–8x higher density from HIGH-FREQUENCY boundary complexity — many small object
silhouettes (furniture legs, window mullions, foliage) — a qualitatively different kind of
structure than "a few sharp smooth curves." Confirmed directly: replacing the multiply-compose
with a non-saturating `min()` composition of 150 SMALL (radius 0.06) blobs reaches edge_frac
≈ 0.052 — closer, but still below the IQR floor (0.07), and 150 dense overlapping circular
blobs is a different, visually-implausible failure mode (reads as noise/confetti lighting, not
plausible illumination) — not something to train on without separately validating it doesn't
teach a degenerate prior. This would also require modifying `sample_relight_field`'s shared
composition rule, affecting every other config that uses it (v17_23/33/34's Hypersim lever) —
out of scope for a single-row fix.

**Conclusion**: reusing the EXISTING relight sampler cannot close the measured shadow-domain
gap through any parameter tuning found. Closing it for real requires either (a) a genuinely
different field generator (dense small-silhouette masks, validated for visual plausibility —
non-trivial redesign) or (b) real rendered hard shadows (the re-render future-work item in §8,
already the higher-confidence path). **Recommendation: do not launch v17_35 as designed.**
Keep the code (harmless, opt-in, every default preserves prior behaviour byte-for-byte,
verified against v17_20/23/29/33/34's resolved configs) as infrastructure; this finding itself
— MEASURED not asserted — becomes the future-work paragraph instead of a trained result.

## 10. Non-interference verification (2026-07-13, requested explicitly)

Caught and fixed one real (if tiny) issue during verification: the bottom-clip mask hygiene
(§3.2) was originally added unconditionally to `sh_mask_spatial`, which is shared by the
Hypersim-gated self-sup path too — meaning if v17_34 were relaunched under this code, its
shadow branch (same mechanism v17_23/33 already trained under) would see a very slightly
different mask than they did. Measured impact: 19 of 1,113,601 top-clip-valid pixels on a
realistic Hypersim-like batch (0.0017%) — negligible, but not zero, and there was no reason to
accept even that when it costs nothing to avoid. **Fixed**: the bottom-clip now only applies
when `shadow_on_front3d` or `lambda_shadow_gt_front3d>0` is active, i.e. only when the NEW
mechanism is in play. Every pre-existing config's Hypersim-gated path is untouched.

Verified two ways, not just by config defaults:
1. **Direct numeric equivalence.** Reconstructed the pre-edit shadow-mask logic verbatim and
   ran it against the current code's logic, called with each config's REAL resolved
   `FlexibleLossV17` criterion (v17_20/23/29/33/34), across batches mimicking each row's actual
   mix (all-Hypersim; mixed Hypersim+MID+IV; mixed batches that also contain front3d rows with
   `shadow_on_front3d=False`, the case that actually matters for v17_29/33/34's real mixes),
   multiple `global_step`/`decorr_scale` combinations including the `shadow_active=False` edge
   case. Result: **byte-identical** (`torch.equal`) for all 5 configs × 3 batch scenarios × 3
   step conditions. `shadow_invariance`/`shadow_explain` method bodies confirmed untouched by
   direct diff (only `__init__` gained new config reads).
2. **Dataset/collate mechanics.** Built samples via the real `prepare_training_tensors` path
   the way each dataset's `__getitem__` actually does, with `is_front3d` added exactly as in
   the shipped dataset files, and ran torch's real `default_collate` over a batch mixing
   hypersim/midintrinsic/interiorverse/front3d-shaped samples (paired and unpaired) — confirmed
   no collate error and correct per-row values.

**Conclusion: v17_20/23/29/33 (already trained) are unaffected by construction — their
checkpoints exist and this code only runs forward. v17_34 (not yet relaunched) will train
under mask logic now PROVEN byte-identical to what v17_23/33 already experienced** — the only
axis it differs from them on remains the intended one (flatness weight 1.0 vs 4.0), not an
accidental side effect of this session's changes.

## 8. Explicitly NOT in this plan

- Changing v17_34 (keeps its factorial role; it answers the washout question).
- Cutting InteriorVerse for mix budget: σ-aug costs ZERO sampling share (it
  augments existing front3d draws). The mix stays `0.3/0.3/0.15/0.25` so (35−29)
  is single-variable. IV's importance is debunked but changing it would confound
  the row for no benefit.
- Re-rendering with hard Blender lights (geometry-consistent shadows): FUTURE
  WORK. σ fields are geometry-unaware (they don't follow occluders); acceptable
  for invariance training (Hypersim's sh_inv accepted the same), but the honest
  future-work item is real rendered hard shadows + entering the curriculum at
  ~20k (where inv_gap still had gradient) instead of the 40k+ fine-tune.
