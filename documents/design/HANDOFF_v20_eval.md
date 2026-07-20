# Handoff: why we run `scripts/eval_row4.sh`, and what V20 must do

*Context dump for a fresh session. Written 2026-06-19. Numbers below are from the
checkpoints that exist **today** (V17 19k/50k + Marigold); verify against the JSONs before
quoting in the thesis, and re-run anything marked ⚠️.*

---

## 1. The one-sentence thesis claim

A feed-forward intrinsic-image model should **not let a colored illuminant leak into the
albedo**. When the light turns warm/cool, the recovered albedo (the material's true color)
must stay constant — the color shift belongs in *shading*, not reflectance. We call the
mechanism **CARI** (cross-render albedo-invariance: an `L_inv` consistency loss across two
renders of the same surface under different light + an `L_explain` reconstruction term).

Everything in `eval_row4.sh` exists to **measure that claim** and to **guard against the
classic failure modes** of achieving "constancy" the cheap way (collapsing albedo toward
gray, or just regressing blurry albedo).

---

## 2. Why we run the four-benchmark suite (`scripts/eval_row4.sh`)

It produces the thesis comparison table: **our V17 checkpoints (19k, 50k) vs Marigold**
(appearance = direct-albedo rival; lighting = context). Four benchmarks, each a different
*kind* of evidence so no single dataset's bias carries the argument:

| # | Benchmark | Domain | Measures | Role |
|---|-----------|--------|----------|------|
| 1 | **MAW1** | real, measured albedo | chromaticity ΔE + intensity SI-MSE | the **external** real-albedo claim; direct rival is **CD-IID (ΔE ≈ 3.37, intensity ×100 ≈ 0.54)** |
| 2 | **MID constancy** | real, multi-illuminant | `C_mat` (within-material invariance), `R_cast_rms` (colored-cast leak), saturation | the **headline in-domain claim** |
| 3 | **ARAP constancy** | synthetic GT | `C_arap`, `Cast_RMS` + si-RMSE/LMSE accuracy guard | **out-of-domain transfer** + anti-gray-collapse guard |
| 4 | **IIW WHDR** | real, ordinal | WHDR on relative reflectance | **no-regression guard** — must not break a standard benchmark to win the others |

Why all four: MID/MAW are *real* but GT-limited; ARAP is *synthetic* with true GT so it can
prove constancy didn't come from gray-collapse (the accuracy guard); IIW is the sanity check
that we didn't overfit the cast story and wreck ordinary reflectance ordering.

**Fairness notes baked into the harness** (each model run as its authors intended):
- Input space differs *on purpose*: V17 gets **linear** RGB (its Hypersim-linear training
  domain); Marigold gets **sRGB** (natural-photo training). This is correct, not a bug.
- Resolution is each model's native: V17 at **512** (`input_size: 512`), Marigold at its
  **768** default. We do **not** force a common size.
- MAW runs all models zero-shot at 512 per the CD-IID protocol; the evaluator is
  scale-invariant (finds optimal θ before scoring), so display normalization doesn't bias it.

---

## 3. Where things stand: the row-4 baseline (today's numbers)

**MAW1** — chromaticity ΔE ↓, intensity ×100 ↓ (CD-IID: 3.37 / 0.54):

| Model | Chroma ΔE | Intensity ×100 |
|-------|-----------|----------------|
| v17_50k | 4.02 | **0.40** (beats CD-IID & all) |
| v17_19k | 4.78 | 0.43 |
| marigold_app | **3.62** ⚠️ | 0.47 |
| marigold_light | 4.05 | 0.42 |

**MID constancy** — `C_mat` ↓, `R_cast_rms` ↓:

| Model | C_mat | R_cast_rms |
|-------|-------|------------|
| marigold_app | **0.070** | **0.038** |
| v17_50k | 0.107 | 0.136 |
| v17_19k | 0.150 | 0.131 |
| marigold_light | 0.284 | 0.126 |

**The honest read (this is the crux of the thesis framing):**
- On **intensity** V17-50k already wins. Good.
- On **chromaticity / cast** (the literal thesis axis), **Marigold-appearance currently
  beats V17** on both MAW ΔE and MID cast. A 1.4B-param diffusion model out-chromas our
  feed-forward model. So the claim is **not** "we beat SOTA on cast" — it is a
  **controlled-mechanism** result: we show CARI *moves* cast/constancy in the right direction
  with a cheap feed-forward model and explicit losses, measured across four benchmarks.
- V17-50k vs 19k: constancy (`C_mat`) improved with training (0.150→0.107) **without**
  gray-collapse — that's the mechanism working. But cast leak (`R_cast_rms`) did **not**
  improve. That gap is exactly what V20 targets.

---

## 4. What V20 is, and what we expect from it

**V20 is the final model** (`src/configs/v20.yaml`, `src/models/v20.py`, train with
`python src/train_v17.py --config src/configs/v20.yaml`). It supersedes the V19/CARI
prototype — same CARI *mechanism*, cleaner implementation. **Status today: trained only to
~6k iters** (`checkpoints/v20/checkpoint_iter_6000.pth`), i.e. still in Phase 1. It is **not
yet ready** to drop into the row-4 table.

**What changed from the V17 row-4 model (the design intended to fix the gaps above):**

1. **Shading-first DERIVE cascade (not albedo regression).** Albedo is *derived*
   `A_rough = I / S_c` from a predicted colorful shading `S_c`, then refined by a
   zero-init residual `Δ` (CD-IID `last_residual`). Deriving inherits the input's
   high-frequency detail arithmetically → fixes the **regression-blur** that hurt albedo
   LMSE. Chroma `ξ` is predicted low-res (256) and upsampled (color is low-frequency →
   regularized smooth); gray shading stays full-res for sharp shadow edges.
2. **Separate diffuse shading `S_d`** for the physical model `I = A·S_d + R` (`S_c ≠ S_d`),
   with `S_c`/`A` detached in `S_d`'s skip.
3. **CARI signal without OpenRooms** (server closed, 0 images downloaded — *memory that says
   "Hypersim+OpenRooms+MID" is stale; OpenRooms weight is 0.0 in every phase*). Cross-render
   `L_inv` now comes from: (a) **Hypersim `c·S` colored-relight pairs** from iter 3000
   (co-anchored by exact Hypersim GT → cannot gray-collapse), and (b) **MID real raw-color
   pairs** (`mid_raw_color_pair: true` — un-white-balanced, because WB erases the real
   colored-illuminant signal the thesis exploits). InteriorVerse stays out (multi-view, not
   multi-light). `lambda_explain: 0.25`.
4. **Curriculum:** Phase 1 [0–15k) Hypersim only (anchor absolute albedo/shading color;
   `c·S` pairs on at 3k); Phase 2 [15k–25k) +MID (0.3); Phase 3 [25k–50k) more MID (0.4) —
   full CARI, Hypersim keeps the absolute-color anchor alive throughout.

**Success criteria for V20 (what we're hoping the eval shows once it's trained to ~50k):**
- **↓ `R_cast_rms` on MID and ↓ ΔE on MAW** vs V17-50k — close the gap to Marigold-appearance
  on the cast/chromaticity axis (the one place V17 currently loses).
- **`C_mat` stays low or improves** while **ARAP si-RMSE/LMSE does NOT rise** (constancy
  must not be gray-collapse — the derive-cascade + Hypersim GT anchor are the safeguards).
- **Albedo LMSE improves** (derive-cascade fixes regression blur).
- **IIW WHDR holds or improves** — no-regression guard.
- Net story stays **controlled-mechanism**, now with a model that is competitive-or-better on
  cast, not just intensity.

---

## 5. How to run / re-run

```bash
# Full four-benchmark table (V17 19k/50k + Marigold). Pick a GPU:
CUDA_ID=1 bash scripts/eval_row4.sh

# Train V20 (final model):
python src/train_v17.py --config src/configs/v20.yaml      # train_v20.py also exists
```

Checkpoints the harness expects (all present today):
`checkpoints/checkpoint_v17_iter_19000.pth`, `checkpoints/v17/checkpoint_iter_50000.pth`,
`checkpoints/marigold-iid-appearance-v1-1`, `checkpoints/marigold-iid-lighting-v1-1`.

Outputs (thesis-table inputs): `documents/evals/results/eval_maw_results.json`,
`documents/evals/results/eval_mid_constancy_results.json`, `documents/evals/results/eval_arap_constancy_results.json`
(use the `__indoor` rows — we train indoor only), `documents/evals/results/eval_iiw_results.json`.

To add V20 to the table once trained, append a checkpoint to each block of
`scripts/eval_row4.sh` (e.g. `v20_50k=checkpoints/v20/checkpoint_iter_50000.pth` with
`model_version 17` — V20 loads through the V17 path) and re-run.

---

## 6. ⚠️ Known caveats / recent fixes (read before trusting old numbers)

- **MAW Marigold-appearance double-gamma (FIXED 2026-06-19).** Marigold's appearance albedo
  is in **sRGB** prediction space; the save path re-applied gamma → double-gamma → corrupted
  its chromaticity ΔE. Fixed in `tests/eval/eval_maw.py`. **The `marigold_app` MAW row in the
  current JSON (ΔE 3.62) predates the fix → re-run it.** V17 rows are unaffected.
- **MAW intensity is now reported ×100** to match CD-IID's table (raw ~0.004 → ×100 ~0.40).
  JSON keeps both `intensity_si_mse` and `intensity_si_mse_x100`.
- **`tests/eval/eval_arap.py:_run_marigold_inference` still has the same albedo-space bug**
  (compares sRGB appearance albedo vs linear GT; pseudo-shading mixes spaces). Marigold's
  ARAP rows should be treated as provisional until that is fixed the same way as MAW.
- **HDR tonemap bug (FIXED).** `infer_wild_v17.py` / `eval_arap.py` HDR path divided by
  `0.8/p99` (a multiplier) instead of `p99/0.8` → amplified HDR to all-white. Fixed via a
  `_hdr_norm` helper. Affects ARAP `.hdr` inputs only.
- **MID white-balance erases the signal.** MID raw-color (un-WB) pairs are required —
  white-balancing removes the real colored-illuminant variation the thesis depends on. V20
  uses `mid_raw_color_pair: true`.
- MAW dataset loads **874 images** (not the paper's rounded "∼850") — correct, no
  double-counting; the public release is slightly larger than the figure in the paper.
