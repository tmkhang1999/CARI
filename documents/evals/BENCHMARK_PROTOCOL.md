# Benchmark Evaluation Protocol

Consolidated 2026-07-11 from the CD-IID + Ordinal Shading papers/supplements and our own
resolution/AMP/white-balance measurements. This is the single source of truth for how each
dataset is processed before scoring. Fairness rule underlying all of it: **any metric compared
against a published number must be produced under that paper's input condition, resolution, and
scene set — otherwise every baseline is re-run locally under our identical condition.**

## Per-dataset processing

| Dataset | Role | Input (white-balance) | Inference size (long side) | AMP | Native res |
|---|---|---|---|---|---|
| **ARAP (standard)** | SOTA positioning | **White-balanced** (desaturate shading × albedo) — matches Ordinal/CD-IID | **1280** (measured identical to 1500; Ordinal's own "1500" is a per-image content-adaptive R_0 ceiling, not a fixed op point — not reproducible either way, and no accuracy cost to skipping it) | off | 640–2880, median 1280 |
| **ARAP (thesis)** | Colored constancy (our claim) | **Raw colored** (illuminant cast preserved) | 1280 | off | same |
| **MAW** | SOTA (real measured albedo) | Raw real photo (linear) | **512** (authors' protocol) | **on** | 5472×3648 / 3039×2014 |
| **MID** | Internal constancy diagnostic only | Raw colored (25 illum/scene) | 1280 (resolution-insensitive) | off | 1500×1000 |
| **IIW** | Guard / secondary | Raw real photo | 1280 (min-floor 1024) | off | ~512×341 (upscaled) |

## Citation target per benchmark (which published table we compare against)

| Benchmark | **Cite** | Their "Ours" numbers | Why THIS table (not the other) |
|---|---|---|---|
| **MAW** | **CD-IID 2024, Table 1** | Int×100 **0.54**, Chroma **3.37** | Direct rival; MAW authors' fixed code+set (~850). Our 512+code+scaling matches. We beat chroma (3.31) |
| **ARAP** | **Ordinal Shading 2023, Table 1** | LMSE **0.021**, si-RMSE **0.252**, SSIM **0.761** | Uses ORIGINAL Bonneel 52-scene set = OURS. **CD-IID Table 2 rejected**: it drops redundant scenes + ADDS MIST (Hao&Funt 2020) → different, non-reproducible dataset. Lead LMSE+SSIM (exact chrislib); si-RMSE approximate |
| **IIW** | **Ordinal Shading 2023, Table 2** | WHDR **24.9** zero-shot | CD-IID doesn't tabulate IIW; standard WHDR + Narihira split. Guard axis only (WHDR exploitable) |
| **MID** | — (Murmann 2019 dataset only) | — | Training data for us AND CD-IID → not a SOTA row |

**KEY: ARAP cites Ordinal, NOT CD-IID.** The two papers report different numbers for the SAME
baseline on "ARAP" (Ordinal-in-Ordinal: 0.021/0.252/0.761; Ordinal-in-CD-IID: 0.035/0.149/0.751)
because CD-IID modified the scene set (−redundant, +MIST) AND uses plain chrislib RMSE vs Ordinal's
si-RMSE. We can only reproduce Ordinal's set, so we cite Ordinal Table 1. CD-IID still appears in the
THESIS (raw-colored constancy) table — but there it is **re-run locally on our set**, since C_arap/
Cast_RMS are our novel metrics with no published values anywhere.

## Per-dataset metrics

| Dataset | Metrics | Compared against | LMSE window |
|---|---|---|---|
| **ARAP (standard)** | Albedo **LMSE**, **si-RMSE**, **SSIM** (+ shading LMSE/si-RMSE/SSIM) | Ordinal Table 1: LMSE 0.021 / si-RMSE 0.252 / SSIM 0.761 (zero-shot) | **20** |
| **ARAP (thesis)** | **C_arap**, **Cast_RMS** (+ our LMSE/si-RMSE/SSIM) | Baselines re-run LOCALLY (raw colored) | 20 |
| **MAW** | **chromaticity ΔE**, **intensity SI-MSE ×100** | CD-IID: ΔE ≈ 3.37, int ×100 ≈ 0.54. Authors' bundled `numerical_albedo.py` + authors' scaling | n/a |
| **MID** | C_mat, R_cast_rms, M_albedo, LMSE | — (training data for us AND CD-IID; NOT a SOTA row) | — |
| **IIW** | WHDR | Narihira test split; zero-shot. WHDR flagged exploitable by authors themselves (Ordinal §8.2) | — |

## Key protocol notes / caveats

- **ARAP white-balance split is mandatory.** The standard benchmark strips illuminant color out
  of the input, so it does NOT test colored constancy. Standard = comparable to published; thesis
  (raw colored) = the novel axis, all baselines re-run locally. Never mix the two conditions in
  one column. See [[arap-protocol-whitebalance]].
- **ARAP metrics = LMSE + si-RMSE + SSIM** (Ordinal §8.1.2). **LMSE is the exact-match anchor**:
  chrislib's `lmse` (window=20, per-window least-squares via `ssq_error`) is what both papers use,
  and we match it byte-for-byte. **SSIM** is standard. **si-RMSE has NO ground truth in chrislib** —
  the shared library provides only `lmse` (scale-invariant) and `rmse_error` (plain, NOT scale-
  invariant); there is no si-RMSE function, so the papers' "scale-invariant RMSE" column is not
  reproducible from chrislib alone. Our `albedo_si_rmse` uses mean-normalization (the training-
  shared `_masked_scale_invariant_rmse`) → 0.29 vs their 0.25 (same ballpark, robust to ARAP's
  tiny-HDR GT albedo scale). ⚠️ Do NOT "fix" this to a least-squares single-scalar RMSE: ARAP GT
  albedo is stored as tiny HDR (alley max 0.0056), so least-squares absorbs the whole scale gap and
  plain rmse_error collapses to ~0.0015 (scale-DEPENDENT garbage). Tried 2026-07-11, reverted.
  **Lead the ARAP comparison on LMSE + SSIM (exact-match); report si-RMSE as supporting only.**
  Use `albedo_si_rmse`, NOT `albedo_rmse` (that is chrislib DIRECT RGB RMSE, a different quantity).
- **ARAP is zero-shot** — we qualify (train on Hypersim/MID/IV/Front3D, none overlap Bonneel/CGI).
  CD-IID also curates the scene set (drops 5+-illum redundancy, adds MIST). ⇒ re-run CD-IID
  locally rather than cite its table if it appears in either ARAP table.
- **MAW 512 is exact protocol match** — the authors' metric code downscales predictions to
  320×240 internally, so inference above 512 is wasted. Measured: intensity 35% better at 512,
  chroma ~tied. AMP numerically free (<0.1% drift).
- **ARAP 1280 vs 1500 identical** (full 52-scene set, all metrics <1%, mixed direction = noise).
  CORRECTED 2026-07-12: "1500" is Ordinal's per-image content-adaptive R_0 CEILING (Miangoleh
  edge-density criterion), not a fixed operating resolution — no fixed number reproduces it
  exactly, so matching it buys no real citation fidelity. Ship 1280 (lower VRAM, same accuracy).
- **MID resolution-insensitive** (~1%); **IIW/ARAP clearly worse at 512** (IIW WHDR 0.351 vs
  0.306) → keep at 1280+.
- **WHDR untrustworthy — citeable.** Ordinal §8.2 + Garces 2022: scaling input to [0.55,1] gives
  25.7 free; +0.5 albedo shift → SOTA-zero-shot. Cite, don't assert on our own authority.

Source memory: [[arap-protocol-whitebalance]], [[cari-benchmark-survey]],
[[maw-eval-marigold-double-gamma]], [[front3d-training-plan]].
