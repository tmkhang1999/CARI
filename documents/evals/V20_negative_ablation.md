# V20 (shading-first analytic derive) — Negative Ablation

**Status: REJECTED as the thesis spine (measured 2026-06-29). Ship V17 + CARI.**
**Use this as a documented negative result with a clear mechanism, not a failed deliverable.**

## What V20 was and why we tried it

V20 replaced V17's direct-albedo head with a *shading-first analytic derive*:

- Frozen DINOv2-L + DPT trunk → two heads: a full-res gray shading `g` (π-domain) and a dense
  low-res unit-luminance **chroma field** `c` conditioned on `g.detach()`.
- Compose shading `S = g · c`; **derive** albedo `a = clamp(I / S, 0, 1)` (FP32, Q99 anneal);
  residual `R = (I − a·S)₊` is analytic (no head).
- Constancy was meant to come from CARI `L_inv` (cross-render albedo invariance) on Hypersim
  colored-relight pairs + MID real un-WB pairs.

**Motivation:** V17 leaves visible hard shadows / strong colored illuminants in albedo (see the
old `tests/visualizations/ARAP results/arap_constancy_sheet.jpg`). The hypothesis was that a dense
chroma field + analytic derive would remove colored illuminants and shadows better than V17, raising
ARAP and IIW constancy.

## What happened

1. **Collapse, then fix.** From-scratch V20 collapsed at iter 15k when `shadow_explain` turned on
   (free shading-scale gauge + clamp-gradient-death → S→0, a→white, R→I). A 5-part fix
   (absolute-scale pin on `g`, diffuse-recon wall, cut `shadow_explain`, colored chroma-pair target,
   capped derive anneal) **worked**: V20 then trained stably 15k→40k, no white-collapse.
2. **But a stable V20 still loses to V17 on every constancy axis** — including the colored-illuminant
   axis it was built to win.

## Measured results — V20@40k vs V17@50k (final models)

### MID constancy (real multi-illuminant, in-domain; subset 25 scenes, lower = better)

| Metric | V20@40k | V17@50k | Δ |
|---|---|---|---|
| C_mat (within-material CoV) | 0.275 | **0.177** | **+55% worse** |
| Cast_RMS (colored leak) | 0.146 | **0.119** | +22% worse |
| LMSE (albedo accuracy) | 0.547 | **0.525** | +4% worse |
| SAT bins (saturation MAE) | all higher | **lower** | more desaturated |

**MID C_mat never converged toward V17** (subset trajectory):
`15k 0.323 → 20k 0.401 → 30k 0.279 → 33k 0.278 → 40k 0.275`. Phase 2 (anti-collapse, Hypersim-only)
*hurt* constancy; Phase 3 (MID enters) recovered it partway then flatlined at ~0.28, +55% above V17.

### ARAP cross-illumination constancy (synthetic GT, OOD; indoor, res 768/640, 20 light-groups, lower = better)

| Metric | V20@40k | V17@50k | V20 vs V17 |
|---|---|---|---|
| C_arap (cross-light CoV) | 0.247 | **0.123** | **2.0× worse** |
| Cast_RMS (chroma drift) | 0.128 | **0.045** | **2.8× worse** |
| si-RMSE vs GT (accuracy) | 0.664 | **0.377** | 1.8× worse |
| **COLORED subset** C_arap (n=11) | 0.335 | **0.149** | **2.3× worse** |
| **COLORED subset** Cast_RMS | 0.228 | **0.079** | **2.9× worse** |
| DIRECTION subset C_arap (n=9) | 0.139 | **0.091** | 1.5× worse |

The COLORED subset (light *color* varies) is the literal thesis axis. **V17 is 2.3–2.9× better there.**
V20 does not remove colored illuminants better — it removes them *worse*.

## Why V20 fails (mechanism)

The derive `a = clamp(I/S)` plus the dense low-res chroma field let shading **intensity** leak into
albedo (→ high C_mat) and shading **color** leak into albedo (→ high Cast). V17's direct-albedo head +
CARI `L_inv` pins both far better. The degrees of freedom that had to be stabilized to stop the
collapse (free shading gauge, soft-clamped derive) are the same ones that leak. This confirms both the
pre-registered no-go gate ("beat V17 C_mat by 20k" — failed) and the prior strategy verdict
(the contribution is the CARI **loss**, not the derive **heads**).

## Conclusion

- **Ship V17 + CARI** as the thesis spine (best on both real MID and synthetic ARAP constancy).
- **V20 → negative ablation:** "A shading-first analytic derive underperforms a direct-albedo head +
  CARI on colored-illuminant constancy (real MID and synthetic ARAP), even after the training collapse
  is fixed." Legitimate, defensible, with a mechanism.
- Residual V17 shadow artifacts are a **V17 enhancement** (e.g. an aligned `shadow_inv` invariance loss
  on V17-shape), not a reason to switch architectures.

Raw numbers: `documents/evals/results/v20cmp/` (mid_40k.json, mid_traj.json, arap_indoor.json).
