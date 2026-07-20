# Row-4 unified benchmark table (V17 vs Marigold)

All metrics **LOWER = better**. Best per column is **bold**. ARAP uses the `__indoor` split (we train indoor); `—` = not available / not applicable.

## Headline — all four benchmarks

| Model | MAW ΔE | MAW Int×100 | MID C_mat | MID Cast | ARAP C_arap | ARAP Cast | ARAP si-RMSE | ARAP LMSE | IIW WHDR |
|---|---|---|---|---|---|---|---|---|---|
| v17_19k | 4.779 | 0.427 | 0.150 | 0.131 | **0.151** | **0.063** | 0.332 | **0.044** | 0.290 |
| v17_50k | 4.023 | **0.402** | **0.107** | 0.136 | 0.166 | 0.071 | 0.362 | 0.058 | 0.303 |
| marigold_app | **3.625** | 0.461 | 0.140 | **0.085** | 0.170 | 0.125 | **0.283** | 0.045 | **0.130** |
| marigold_light | 4.047 | 0.422 | 0.276 | 0.126 | 0.292 | 10.849 | 0.576 | 0.149 | 0.222 |
| _CD-IID (paper, ~850 imgs)_ | _3.37_ | _0.54_ | — | — | — | — | — | — | — |

## Colored-illuminant axis (ARAP indoor, colored subset — the thesis claim)

| Model | C_arap (colored) | Cast_RMS (colored) |
|---|---|---|
| v17_19k | 0.178 | **0.060** |
| v17_50k | 0.181 | 0.061 |
| marigold_app | **0.174** | 0.071 |
| marigold_light | 0.319 | 8.158 |

> Notes: MAW ΔE/Int over **874** images (public release superset of the paper's ~850 — treat the CD-IID row as approximate). MID `M_albedo`/saturation are pseudo-GT, ours-only (see eval_mid_constancy_results.json). Marigold albedo is linearised to V17's space before scoring (output-space fix).
