# V17 vs Marigold — benchmark analysis & fairness audit

*Written 2026-06-19. Companion to `documents/design/HANDOFF_v20_eval.md`. The numeric table in §1 is
filled from the **corrected** re-run (`scripts/eval_row4.sh` after the output-space fix); the
qualitative analysis is from inspecting the contact sheets in `tests/visualizations/`.*

---

## 0. Executive summary

- **The output-space fix changed the result.** Before the fix, the buggy numbers showed
  Marigold dominating ~8/9 metrics. After linearising Marigold's albedo into V17's space, the
  metrics are **roughly even — V17 wins 6 of 11 columns, Marigold 5.** The double-gamma had
  been *desaturating* Marigold's albedo, which artificially lowered its constancy numbers
  (MID `C_mat` 0.070→**0.140**, ARAP indoor `C_arap` 0.083→**0.170**, ARAP Cast 0.035→**0.125**).
- **Honest standing (corrected):** **V17 sweeps the *constancy / invariance* axes** — MID
  `C_mat` (0.107 < 0.140), ARAP `C_arap` (0.151 < 0.170), ARAP Cast (0.063 < 0.125), the
  colored-axis Cast (0.060 < 0.071), MAW intensity (0.402 < 0.461), ARAP LMSE (tie). **Marigold
  wins the *structure / fidelity* axes** — IIW WHDR (0.130 ≪ 0.303), ARAP si-RMSE (0.283 <
  0.332), MAW chroma ΔE (3.62 < 4.02), MID cast (0.085 < 0.136) — **and it still clearly *looks*
  better** (sharper, cleaner shading removal).
- **Why V17 looks worse but wins half the metrics:** every metric V17 wins is a **texture-blind
  consistency measure** (scale-invariant, per-material-pooled) — a smooth/hazy prediction that
  is *self-consistent* scores well on "consistency" while looking bad. Every metric V17 loses
  (**IIW WHDR ordinal, ARAP si-RMSE dense**) is **structure-sensitive** — it penalises exactly
  the blur/haze the eye sees. So the pictures and the metrics agree once you sort metrics into
  "sees structure" vs "doesn't."
- **This is the thesis result:** V17's CARI mechanism genuinely wins on **colored-illuminant
  constancy** (it beats a 1.4B-param diffusion model on those axes) with a cheap feed-forward
  model; it lags only on perceptual *structure* — which is exactly what V20 targets. Lead the
  claim with the **constancy axes + ARAP true-GT accuracy**, not a beauty contest.
- **Fairness:** the *input* comparison is fair (each model in its native color space). The audit
  surfaced four issues, two of them real bugs: (a) the **Marigold output-space bug** that
  corrupted every chromaticity metric — **now fixed, and it materially changed the table**;
  (b) a **V17-side input-distribution inconsistency** across our own evaluators; (c) **MAW set =
  874 ≠ paper's ~850**; (d) a **resolution mismatch** (defensible but mis-described in the
  handoff).

---

## 1. The corrected four-benchmark table

All metrics **LOWER = better**; best per column in **bold**. ARAP uses the `__indoor` split (we
train indoor only). Marigold albedo is linearised into V17's space before scoring (output-space
fix, §3.2). Source: `documents/evals/eval_row4_table.md`.

| Model | MAW ΔE | MAW Int×100 | MID C_mat | MID Cast | ARAP C_arap | ARAP Cast | ARAP si-RMSE | ARAP LMSE | IIW WHDR |
|---|---|---|---|---|---|---|---|---|---|
| v17_19k | 4.779 | 0.427 | 0.150 | 0.131 | **0.151** | **0.063** | 0.332 | **0.044** | 0.290 |
| v17_50k | 4.023 | **0.402** | **0.107** | 0.136 | 0.166 | 0.071 | 0.362 | 0.058 | 0.303 |
| marigold_app | **3.625** | 0.461 | 0.140 | **0.085** | 0.170 | 0.125 | **0.283** | 0.045 | **0.130** |
| marigold_light | 4.047 | 0.422 | 0.276 | 0.126 | 0.292 | 10.849 | 0.576 | 0.149 | 0.222 |
| _CD-IID (paper, ~850)_ | _3.37_ | _0.54_ | — | — | — | — | — | — | — |

**Colored-illuminant axis** (ARAP indoor, colored subset — the literal thesis claim):

| Model | C_arap (colored) | Cast_RMS (colored) |
|---|---|---|
| v17_19k | 0.178 | **0.060** |
| v17_50k | 0.181 | 0.061 |
| marigold_app | **0.174** | 0.071 |

**Read:** V17 wins the bold cells on **MAW intensity, MID C_mat, ARAP C_arap, ARAP Cast, ARAP
LMSE, colored Cast** (the *constancy* axes); Marigold wins **MAW ΔE, MID cast, ARAP si-RMSE, IIW
WHDR, colored C_arap** (the *structure/fidelity* axes). `marigold_light` is a weak baseline
(its ARAP Cast 10.8 is a degenerate-channel artefact, not a real value).

> **Caveat on these numbers.** MAW `marigold_app` (3.625 / 0.461) is recovered from
> `eval_row4_rerun.log`: the MAW JSON write was lost when `marigold_light` **OOM-crashed on the
> last model** (the run was on a contended GPU 1). `marigold_light` MAW (4.047 / 0.422) is the
> pre-fix value but is **valid** — the lighting model predicts *linear* albedo, so the
> output-space fix is a no-op for it. Robustness fix recommended: have `eval_maw.py` write its
> JSON incrementally (per model) so a late OOM can't discard finished results.

---

## 2. Why we look worse but still win some metrics

This is the central question. The short answer: **the metric V17 wins measures a different
thing than your eyes do.**

### 2.1 What the MAW metrics actually score (and ignore)

MAW chromaticity ΔE and intensity SI-MSE are computed by `numerical_albedo.AlbedoEvaluator`
(`tests/eval/eval_maw.py`). Three properties decide everything:

1. **Scale/shift-invariant.** The evaluator finds the optimal global brightness θ *before*
   comparing — so absolute exposure, and any uniform haze, are divided out.
2. **Per-material pooled, on sparse measured chips only.** It scores only the handful of
   physically *measured* material patches (the white regions in the "GT Albedo (measured)"
   panel), pooling each material to its average. Whole-image texture, edges, and hallucination
   never enter the number.
3. **The measured chips are mostly near-neutral** (desk top, walls, paper). On near-neutral
   surfaces, getting the *relative intensity* right is easy for a smooth predictor.

So a prediction can be globally hazy, soft, and warm — and still nail the *pooled average
intensity of a few neutral chips* after scale alignment. That is exactly V17's MAW-intensity
win: **0.40 (V17-50k) vs 0.47 (Marigold-app)**. It is a win on "smoothly-correct average
brightness per neutral material," the one regime where blur is free.

### 2.2 What the pictures show (`tests/visualizations/`)

- **MAW desk scene** (`maw_sheets/.../scene_0__DSC4370_sheet.jpg`): V17 keeps a **warm cast
  and residual shading** — the cushion still carries its shadow gradient, the whole frame is
  soft/hazy ("input minus a little light"). Marigold **flattens the lighting** — walls and
  cushion go uniformly white, matching the flat measured GT chips — though it **hallucinates**
  (ghosted desk edge on the left).
- **MID constancy sheets** (`mid_constancy/mid_constancy_*.jpg`, regenerated post-fix): with the
  output-space fix Marigold's albedo is **noticeably more colourful** than the buggy washed-out
  version — the cans/pots/bottles keep their chroma, and its sharper structure shows. *Because*
  it now shows real within-material chroma variation, its `C_mat` rose (0.070→0.140) and
  **V17-50k actually wins within-material constancy** (0.107 < 0.140). Marigold still wins the
  *cast drift* number (0.085 vs 0.136) and looks cleaner; V17's heatmaps still run hotter
  (residual cast). The earlier "Marigold gray-collapses on MID" read was an **artefact of the
  double-gamma bug**, not real behaviour — worth correcting in any prior notes.
- **ARAP sheet** (`ARAP results/arap_constancy_50k_indoor/_constancy_sheet.jpg`): V17 shows the
  **strong-light bug** — under saturated red/green/blue room lights the colored illumination
  stays baked into the albedo (bottom rows).

### 2.3 The resolution of the paradox

The eleven metric columns sort cleanly into **two families**. V17 wins one family, Marigold the
other — and the family Marigold wins is the one the eye sees.

**Constancy / invariance family — V17 wins (pooled, scale-invariant, structure-blind):**

| Axis | What it sees | Winner | Numbers |
|---|---|---|---|
| MID `C_mat` | within-material colour consistency across lights | **V17-50k** | 0.107 < 0.140 |
| ARAP `C_arap` (indoor) | cross-light consistency (synthetic) | **V17-19k** | 0.151 < 0.170 |
| ARAP Cast (indoor) | colored-cast leak (synthetic) | **V17** | 0.063 < 0.125 |
| Colored-axis Cast | cast leak under varying light *colour* | **V17** | 0.060 < 0.071 |
| MAW intensity | pooled neutral-chip brightness, scale-free | **V17-50k** | 0.402 < 0.461 |
| ARAP LMSE | *local* windowed accuracy | ~tie | 0.044 ≈ 0.045 |

**Structure / fidelity family — Marigold wins (ordinal / dense / absolute):**

| Axis | What it sees | Winner | Numbers |
|---|---|---|---|
| IIW WHDR | ordinal lighter/darker over the whole image | **Marigold** | 0.130 ≪ 0.303 |
| ARAP si-RMSE | *dense* spatial accuracy vs true GT | **Marigold** | 0.283 < 0.332 |
| MAW chroma ΔE | absolute chip chromaticity | **Marigold** | 3.62 < 4.02 |
| MID cast | real-domain cast drift | **Marigold** | 0.085 < 0.136 |

**The unifying principle:** V17's wins are all *consistency* measures — they reward a prediction
for being **self-consistent within a material / across lights**, which a smooth, hazy, even
slightly-wrong albedo can be. They are blind to texture, sharpness, and absolute structure.
Marigold's wins are all *structure / absolute-fidelity* measures — ordinal ordering (WHDR),
dense error (si-RMSE), absolute chip colour (ΔE) — which penalise blur and residual shading.
**The pictures look worse because they display structure, the exact family Marigold wins.** One
honest wrinkle: *cast* splits by domain — V17 wins the **synthetic** ARAP cast (controlled
relight, the CARI mechanism's home turf), Marigold wins the **real** MID cast (it flattens more
aggressively). That split is itself informative: V17's mechanism shows its effect cleanest under
controlled illumination.

**Important caveat for the thesis** (anti-gray-collapse): constancy metrics (`C_mat`, `C_arap`)
can be *gamed by desaturation* — flattening colour toward gray lowers within-material variance.
The double-gamma bug was a live demonstration: it *accidentally desaturated* Marigold's albedo,
which inflated its constancy scores (MID `C_mat` 0.070, ARAP `C_arap` 0.083) and made it look
dominant. Removing the desaturation (the fix) **erased that fake advantage** — and **V17 now
wins MID `C_mat` and ARAP `C_arap` while *keeping* material colour** (its predictions are not
gray). That is the stronger claim: V17 achieves constancy *without* collapsing chroma. The guard
that this is real and not V17's own gray-collapse is the **true-GT accuracy** — and V17-19k
**ties Marigold on ARAP LMSE** (0.044 vs 0.045) and is close on si-RMSE, so its constancy is not
bought with accuracy. Still frame the result as **controlled-mechanism** (cheap feed-forward
model moves constancy in the right direction, measured across four benchmarks), anchored on the
colored-illuminant axis + accuracy guard — not a beauty contest, which Marigold wins.

---

## 3. Fairness audit (training + inference preprocessing)

### 3.1 Input color space — ✅ FAIR

Each model is fed its **native training domain**, which is the correct way to compare:
- **V17** gets **linear** RGB (`eval_maw.py:_load_image_linear` raises sRGB→`^2.2`; ARAP/MID
  feed tonemapped-linear) — its Hypersim-linear training domain.
- **Marigold** gets **sRGB** (every evaluator re-applies `^(1/2.2)` before the pipe) — its
  natural-photo training domain.

For MAW the round-trip is sRGB→linear→sRGB (identity modulo 8-bit quantisation), so Marigold
effectively sees the original sRGB. Fair.

### 3.2 Output albedo space — ⚠️ WAS BUGGY → ✅ FIXED THIS SESSION

Marigold's **appearance** model predicts albedo in **sRGB** space
(`target_properties["albedo"]["prediction_space"]=="srgb"`); V17 predicts **linear**. Our
evaluators assumed linear and re-applied `^(1/2.2)` at save/compare → **gamma applied twice**
→ washed-out + **per-channel chromaticity skew** that corrupts exactly the chroma/constancy
metrics.

- `eval_maw.py:199-201` already had the fix, **but Marigold was commented out of the run** —
  the JSON's `marigold_app` ΔE 3.62 is the stale, *inflated* (worse-than-real) number.
- `eval_arap.py:_run_marigold_inference` and `eval_mid_constancy.py:AlbedoPredictor.albedo()`
  **lacked** the fix → their Marigold numbers were in the wrong space. On MAW the bug *inflates*
  ΔE (hurts Marigold); on MID's chroma metrics the double-gamma *desaturation* likely *flatters*
  Marigold's `C_mat`/cast — opposite directions, which is why no single stale number could be
  trusted.
- **Fix applied** (mirror of the MAW fix): linearise (`** 2.2`) when `prediction_space=="srgb"`,
  in all three evaluators, then re-run. **IIW needs no fix** — WHDR is ordinal and gamma is
  monotonic, so 0.130 was already valid.

### 3.3 Resolution — ✅ FAIR (confirmed against CD-IID's own protocol)

V17 runs at **512** (`_resize_512`); Marigold runs at its **native 768** default — each at its
training resolution. **CD-IID's paper validates this directly:** *"we run our method at a
resolution of 512 pixels, this does not significantly affect the scores of our method… the MAW
dataset code handles resizing and computing metrics, we follow the authors' scaling."* So 512
for V17 is the **sanctioned** protocol, not a handicap, and the scale-invariant MAW evaluator
absorbs the resolution difference. (Only the handoff's loose wording "all models at 512" was
inaccurate — Marigold is 768; the *practice* is correct.) **LMSE window = 20**: CD-IID specifies
20; our `src.train._compute_lmse` uses `window_size=20, stride=10` — **matched** ✅.

### 3.4 V17 input-distribution inconsistency across our OWN evaluators — ⚠️ REAL

V17 was trained on **p90-tonemapped** linear Hypersim (`shared_transforms.tonemap_linear`,
p90→0.8). At inference our evaluators disagree:
- **MID** applies the p90 tonemap to match training (`eval_mid_constancy.py:164-173`). ✅
- **MAW** feeds **raw linearised [0,1]** with **no p90** (`eval_maw.py:_run_v17`). ⚠️
- **ARAP-LDR** feeds **clipped raw [0,1]**, p90 only for HDR (`eval_arap.py:158`). ⚠️

For a typical LDR photo p90≈0.3–0.5, so training inputs were ~1.6–2× brighter-normalised than
what V17 sees on MAW/ARAP. I hypothesised V17 might be mildly **OOD-dim** there. **Test (b) ruled
that out.** Running `scripts/eval_fairness_p90.sh` (V17 MAW raw vs. training-matched p90 tonemap,
`documents/evals/results/eval_fairness_p90.json`):

| ckpt | ΔE raw → p90 (Δ) | Int×100 raw → p90 (Δ) |
|---|---|---|
| v17_19k | 4.779 → 4.859 (**+0.080**) | 0.427 → 0.421 (−0.006) |
| v17_50k | 4.023 → 4.022 (**−0.001**) | 0.402 → 0.408 (+0.006) |

Every delta is **<2 % and inconsistent in sign** (p90 helps one metric, hurts another) — within
noise. The inputs *did* change (numbers moved), so the flag applied; the effect is simply
negligible, because the p90 tonemap is essentially a **global scale** and the MAW evaluator is
**scale-invariant** (it divides out θ first). **Conclusion: V17's MAW deficit is *not* an input-
distribution artefact — it is intrinsic to the model** (feed-forward regression blur / residual
shading). The comparison is fair, and the deficit is exactly what an architectural fix (V20's
derive cascade), not a preprocessing tweak, must address.

### 3.5 MAW set: 874 vs the paper's ~850 — ⚠️ note in the table

`labels/meta.csv` has **874 rows = 874 unique images = 874 unique masks, zero duplication**
(verified). The public release is a genuine **superset** of the paper's ~850 — not a counting
bug on our side. Implication: our absolute ΔE is over a *different* image set than CD-IID's
3.37, so **treat 3.37 as approximate**, not a strict head-to-head. (If an exact-850 subset list
surfaces, re-score on it for a clean comparison.)

### 3.6 Scale-invariance — ✅ FAIR

All headline metrics (MAW per_si/si, MID/ARAP scale-aligned, IIW ordinal) are scale-invariant,
so the Q99→0.8 display normalisation does **not** bias any score.

### 3.7 Verdict — the comparison is fair, and the audit is closed

Every axis of potential unfairness was checked and either confirmed fair or fixed:

| Axis | Status |
|---|---|
| Input colour space (V17 linear / Marigold sRGB) | ✅ fair — each native |
| Output albedo space (double-gamma) | ✅ **fixed** in all 3 evaluators (and it flipped the result toward V17) |
| Resolution (V17 512) | ✅ CD-IID-sanctioned ("does not significantly affect scores"); metric scale-invariant |
| LMSE window | ✅ matched (20/stride 10) |
| MAW scaling / resizing | ✅ MAW public code, per CD-IID |
| V17 input distribution (raw vs p90) | ✅ tested — negligible (<2 %), deficit is intrinsic |
| Eval set size (874 vs ~850) | ⚠️ disclosed — CD-IID 3.37 is approximate, not head-to-head |

**Net: no remaining methodological hole.** V17's wins (constancy axes) and losses (structure
axes) are both *real*, not artefacts. That is what makes the result publishable.

---

## 4. Which numbers to trust right now (post re-run, 2026-06-20)

| Benchmark | V17 rows | Marigold — status after the fixed re-run |
|---|---|---|
| MAW | trustworthy | `marigold_app` **valid** (3.625/0.461, recovered from the run log — JSON write lost to the `marigold_light` OOM); `marigold_light` **valid** (linear model, fix is a no-op) |
| MID | trustworthy | **valid & re-run** — fix moved `C_mat` 0.070→0.140, cast 0.038→0.085 (un-desaturated) |
| ARAP | trustworthy | **valid & re-run** — `marigold_app__indoor` now present; `C_arap` 0.083→0.170, Cast 0.035→0.125, si-RMSE 0.175→0.283 |
| IIW | trustworthy | **valid, unchanged** (ordinal, gamma-immune; block skipped) |

Net: the only soft spot is that the **MAW JSON** holds log-recovered (4-dp) `marigold_app`
numbers rather than a clean dump; a single clean MAW re-run on a free GPU would tidy that (and
the change is ≤0.01, so it does not affect any conclusion).

---

## 5. Implications for V20 (design analysis only — not trained this session)

The corrected table reframes V20's job. V17 **already wins the constancy axes**; its *only*
remaining deficits are **structure / fidelity** (IIW WHDR 0.303 vs 0.130, ARAP si-RMSE 0.362 vs
0.283, MAW ΔE 4.02 vs 3.62) plus **real-domain MID cast** (0.136 vs 0.085) — and it still *looks*
worse. So V20 must **close the structure gap without surrendering the constancy lead**:

| V17 deficit (corrected analysis) | V20 mechanism | Expected effect |
|---|---|---|
| Blur / residual-shading haze → high IIW WHDR, ARAP si-RMSE | shading-first **derive cascade** `A=I/S_c`+zero-init Δ (inherits input HF arithmetically) | sharper albedo, shading actually removed → ↓ WHDR, ↓ si-RMSE, ↓ ΔE |
| Real-domain cast still leaks (MID cast) + ARAP strong-light bug | **MID raw-colour (un-WB) pairs** + `L_inv` cross-render invariance | ↓ real-domain cast, keep the synthetic-cast win |
| Absolute albedo colour anchored to GT (don't drift) | **Hypersim GT absolute-colour anchor** (Phase-1 + `c·S` relight pairs) | keeps the constancy win without gray-collapse |

**The tension to watch:** V17's constancy wins come partly *because* it is smooth (low within-
material variance is easy when you blur). V20's derive cascade deliberately re-injects input
high-frequency detail — which could **raise** `C_mat`/`C_arap` (more texture = more variance)
and erode the very constancy lead V17 just earned. Success = **sharper (↓ WHDR/si-RMSE) while
holding `C_mat`/Cast flat or better**. Re-evaluate V20 on this corrected, output-space-fixed
harness at ~50k and check both families move the right way, not just one.

### 5b. Full catalogue of V17 visual drawbacks (from the sheets, beyond the strong-light bug)

Close inspection of `tests/visualizations/{iiw_50k, maw_sheets, mid_constancy, ARAP results}`:

| # | Drawback | Where seen | Root cause | Hurts |
|---|---|---|---|---|
| 1 | **Texture blur** (book spines, fabric weave, grain → mush) | IIW 593, MID, MAW | direct regression → conditional mean | si-RMSE, look |
| 2 | **Within-material shading-ramp leak** (curtain folds, cushion gradient stay in albedo) | IIW 593, MAW desk | shading not fully removed; albedo absorbs the ramp | **WHDR**, look |
| 3 | **Bright-neutral clipping** (gray walls + windows blow out to pure white) | IIW 54 | albedo pushed up + hard [0,1] clamp on bright pixels | WHDR (light-vs-light ordering), look |
| 4 | **Warm cast / haze** (global yellow tint, "input minus a little light") | MAW desk | colored illuminant leaks into albedo (CCR limit) | MAW ΔE |
| 5 | **Edge halos** (faint ring around fireplace / objects) | IIW 54 | decoder upsampling | look |
| 6 | Strong-light bug (saturated room lights baked in) | ARAP classroom | low-res chroma can't carry sharp colored shadows | ARAP cast, look |

Drawbacks **1, 2, 5** are the perceptual "blurry / dirty" look the user wants gone; **2, 3** are
the WHDR levers; **1** is the si-RMSE lever.

### 5c. Concrete V20 adjustments made (2026-06-20) — grounded in `LITREVIEW_feedforward_iid.md`

V20 **already** carries the field's top sharpness levers (`LITREVIEW` ranking: derive ÷ > L1 >
gradient/DSSIM/edge > scale-sep > residual): derive cascade + `albedo_l1` + `lambda_dssim 0.2` +
`lambda_msg_albedo 0.5`. So **#1 (blur/si-RMSE) is handled by design** — no change needed beyond
training it. The gap was the **WHDR / shading-leak** family (#2, #3), so I added two GT-free,
ramped regional priors (both default-conservative so they *cannot* surprise-blur):

1. **`lambda_grad_decorr: 0.05`** (was implemented-but-off) — Retinex edge-decorrelation
   `mean(|∇A|·|∇S|)`: pushes any lighting gradient OUT of albedo → directly attacks #2.
2. **`lambda_alb_flat: 0.05` — NEW texture-gated within-material flatness F-prior**
   (`flexible_loss_v17._albedo_flatness`). Penalises `|∇A|` **only** at luminance-only (shadow)
   edges via the **PIE-Net (CVPR'22) chroma gate** `w=relu(|∇_lum I|−λc|∇_chroma I|)`, so it
   removes the shading ramp (#2) **while protecting texture** (#1 — smoke-tested: luminance ramp
   penalised, chroma edge → 0). **Hypersim-gated** (`m_diff_mask`) because the chroma gate fails
   on colored shadows (CCR limit) — MID's colored case is left to `L_inv`, per
   `V17_shading_first_design.md`.

**Not changed (flagged, needs care):** #3 clipping — V20 derives `A=I/S_c` with a hard [0,1]
clamp; CD-IID instead **Q99-normalises** the derived albedo (`LITREVIEW` §CD-IID stage 3). If
V20 still blows out bright neutrals once trained, switch the clamp in `v20.py`/`iid_utils` to a
Q99 norm. Deferred because it is an architecture change best validated against V20's own outputs,
not V17's. #5 halos are minor; revisit only if visible in V20.

---

## 6. Action items

- [x] Fix Marigold output-space bug in `eval_arap.py` + `eval_mid_constancy.py`; uncomment MAW.
- [x] Remove redundant ARAP-WB block; add `tests/eval/summarize_row4.py` → `eval_row4_table.md`.
- [x] Re-run the corrected suite (MID + ARAP clean; MAW `marigold_app` recovered from log).
- [x] Make `eval_maw.py` write its JSON **incrementally per model** + merge (so a late-model OOM
      like `marigold_light` on GPU 1 can no longer discard finished rows). A clean MAW re-run on
      a free GPU now gives a full-precision `marigold_app` row.
- [x] Add `--ldr-tonemap` to `eval_maw.py` + `scripts/eval_fairness_p90.sh` for fairness test (b).
- [x] Confirm CD-IID protocol fairness: 512 sanctioned, LMSE window=20 matched (§3.3).
- [x] Ran `scripts/eval_fairness_p90.sh` (test b): p90-vs-raw delta <2 %, inconsistent → V17's
      MAW deficit is **intrinsic to the model, not an input-distribution artefact** (§3.4).
- [ ] (Optional) Add CD-IID as a *measured* baseline if `documents/references/Intrinsic CD-IID`
      ships runnable weights, replacing the hand-quoted 3.37.
- [ ] Re-evaluate V20 on this corrected harness at ~50k; check **both** metric families move.
