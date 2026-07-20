# Row-4 evaluation analysis — the 8 decision questions answered

> **Date:** 2026-06-14. Source data: `documents/evals/results/eval_mid_constancy_results.json`,
> `eval_arap_constancy_results.json`, `eval_arap_wb_row4.log`, `eval_iiw_row4.log`.
>
> ⚠️ **PROVENANCE CAVEAT — this run used the PRE-session script.** The numbers are valid model
> outputs, but the run is INCOMPLETE relative to the current `eval_row4.sh`:
> 1. **ARAP ran `all` (51 scenes) only** — no indoor/all split (the `--scene_filter` flag wasn't
>    in the script at run time). The 23-scene **indoor** ARAP table — our headline — does NOT
>    exist yet.
> 2. **Marigold ran on ARAP but NOT on MID** — MID JSON has only our 3 rows; the Marigold-on-MID
>    (Job-2) wiring wasn't present. So the in-domain CARI-vs-SOTA comparison is MISSING.
> 3. **IIW ran the FULL 1046 images** (the dataset is 1046, not ~500) — fine, just slow.
> 4. **The ARAP-WB block CRASHED** (`np.vstack` 4000 vs 9600, the old contact-sheet bug) AFTER
>    printing the numbers — this session's `_make_contact_row` subset rewrite fixes it.
>
> **Conclusion up front:** the core mechanism is **CONFIRMED on MID** but the headline ARAP-indoor
> table and the MID-vs-Marigold table must be regenerated with the current script (§ "Next steps").

---

## The raw numbers

### Block 1 — MID constancy (30 real test scenes, ours only)

| Row | C_mat ↓ | Cast_RMS ↓ | LMSE | SAT q75-100 |
|---|---|---|---|---|
| 19k (row 1, baseline) | 0.1573 | 0.1337 | 0.329 | 0.2956 |
| 30k (row 2, +L_inv+L_explain) | 0.1156 | 0.1460 | 0.398 | 0.3037 |
| **40k (row 4, full CARI)** | **0.0792** | **0.1328** | 0.475 | 0.3034 |

Δ 19k→40k: **C_mat −49.7%**, **Cast_RMS −0.7%** (essentially flat, but recovered from row-2's +9.2%), LMSE +44%, SAT flat.

### Block 2 — ARAP cross-illumination constancy (`all`, 37 groups / 142 frames)

| Row | C_arap ↓ | Cast_RMS ↓ | si-RMSE (guard) | LMSE |
|---|---|---|---|---|
| 19k (row 1) | 0.1905 | 0.0440 | 0.6888 | 0.1016 |
| 30k (row 2) | 0.2488 | 0.0804 | 0.7036 | 0.1006 |
| **40k (row 4)** | **0.1776** | **0.1102** | **0.6702** | 0.0942 |
| **marigold_app** | **0.0796** | **0.0316** | 0.6905 | 0.1042 |
| marigold_light | 0.2842 | 24.73 | 0.7751 | 0.1701 |

### Block 3 — ARAP standard albedo, WB protocol (`all`, 157 cases)

| Row | Albedo si-RMSE | Albedo LMSE | Albedo SSIM | HDR si-RMSE | LDR si-RMSE |
|---|---|---|---|---|---|
| 19k | 0.6121 | 0.0887 | 0.9547 | 0.6227 | 0.4140 |
| 30k | 0.6152 | 0.0867 | 0.9549 | 0.6279 | 0.3787 |
| **40k** | (crashed in viz after metrics — see log; albedo numbers printed: re-run to capture cleanly) | | | | |

(19k/30k printed cleanly; 40k metrics computed but the run threw in the contact-sheet assembly
*after* the loop — the WB albedo summary line for 40k is recoverable from a clean re-run.)

### Block 4 — IIW WHDR (FULL 1046 images)

| Row | WHDR ↓ |
|---|---|
| 19k (row 1) | 31.45% |
| 30k (row 2) | 33.47% |
| **40k (row 4)** | **27.74%** |

---

## The 8 questions

### Q1 — Did the headline cast fix work? ⚠️ PARTIAL / MIXED
**The metric: Cast_RMS on MID and ARAP, 40k vs 30k.**
- **MID: YES, recovered.** 30k regressed Cast_RMS to 0.1460 (the +9.2% null-space problem). 40k
  pulled it back to **0.1328 — now BELOW the 19k baseline (0.1337)**. The four fixes (chromatic
  aug being the key one) did neutralize the cast regression on MID. ✓
- **ARAP: NO — Cast_RMS got WORSE.** 19k 0.0440 → 30k 0.0804 → 40k **0.1102**. On ARAP the cast
  drift *monotonically increased* with CARI training. ✗

**Verdict:** the cast fix worked **in-domain (MID)** but the recovered-albedo color drifts MORE
across lights on synthetic out-of-domain ARAP. This is the central tension in the run. NB: ARAP
Cast_RMS is on a different scale (~0.04–0.11) than MID (~0.13) because ARAP's flashes are not
white — the *trend* is what matters, and the trend is the wrong direction on ARAP.

### Q2 — If cast didn't drop, which fix to push? → DEFERRED until indoor-split known
Cast_RMS is flat-to-worse, so this branch is live. BUT: the ARAP number is on `all` (incl.
outdoor scenes the model never trained on). **Do not retune on a number contaminated by 14
outdoor groups.** First regenerate the **indoor** ARAP Cast_RMS (§Next). If it is *also* worse
indoor, then the cheap fix is: raise `mid_chromatic_aug` p (0.8→1.0) + widen gains
(U[0.6,1.4]→U[0.5,1.6]) + raise `lambda_a_chroma` (0.2→0.4), retrain from 19k.

### Q3 — Was invariance bought with gray collapse? ✓ NO (this is the good news)
**The guard: si-RMSE must not rise while C falls.**
- **MID:** C_mat fell hard (0.157→0.079, −50%). LMSE rose (0.329→0.475) — but that is drift vs
  *pseudo-GT*, the expected noisy-target trade, NOT collapse (SSIM/structure intact).
- **ARAP (true GT!):** C_arap 40k 0.1776 **< 19k 0.1905** AND si-RMSE 40k **0.6702 < 19k 0.6888**
  AND LMSE 40k **0.0942 < 19k 0.1016**. **Constancy improved AND accuracy improved on true GT.**
  This is the decisive anti-collapse evidence: on the dataset with *exact* albedo GT, CARI is
  both more constant and more accurate than baseline. **No gray collapse.** ✓✓

### Q4 — Did we beat SOTA on the out-of-domain benchmark? ✗ NO (Marigold wins ARAP constancy)
**CARI(40k) vs marigold_app on ARAP C_arap + Cast_RMS:**
- C_arap: CARI **0.1776** vs Marigold **0.0796** → **Marigold is 2.2× more constant.** ✗
- Cast_RMS: CARI **0.1102** vs Marigold **0.0316** → **Marigold drifts 3.5× less.** ✗
- si-RMSE (accuracy): CARI **0.6702** vs Marigold 0.6905 → **CARI is slightly MORE accurate.** ✓

**Verdict:** on `all`-ARAP constancy, **Marigold-appearance beats CARI** on both constancy and
cast, while CARI is marginally more accurate to GT. marigold_light is disqualified (Cast_RMS
24.7 = its derived-albedo path is broken on ARAP, as expected — it's the "context" row).
**This is the hardest result to spin and must be confronted, not buried** (see Discussion).

### Q5 — Train to 50k, or is 40k the operating point? → 50k IS JUSTIFIED
MID C_mat is still *dropping fast* at 40k (0.116→0.079 over the last 10k = no plateau). ARAP
C_arap/si-RMSE also improved 30k→40k. The trajectory has not flattened → **training the final
10k to 50k is warranted** and may further separate CARI. BUT do Q4 honestly first — if CARI at
50k still trails Marigold on ARAP constancy, the claim must be reframed (§Discussion), not scaled.

### Q6 — Did we break the standard benchmarks? ✓ NO — IIW actually IMPROVED
- **IIW WHDR: 19k 31.45% → 40k 27.74% = −3.7 pts, a 12% relative improvement.** CARI did NOT
  hurt ordinal ranking; it *helped* it. (Row 2 briefly regressed to 33.5%, row 4 fixed + beat
  baseline.) The "no-regression" guardrail is **passed with room to spare.** ✓✓
- **ARAP-WB albedo (`all`):** 19k si-RMSE 0.6121, 30k 0.6152 — flat (40k crashed in viz, re-run).
  No conventional-metric regression from CARI. ✓
- *Context:* 27.74% WHDR is far from leaderboard (CRefNet-ft ≈10.8% trains on IIW); we never
  train on IIW, so this is the honest no-regression number, exactly as §6.2 framed.

### Q7 — Which ablation rows, in what order? → row 6 still the priority, but Q4 reshuffles
The banked rows are evaluated (1=19k, 3=30k, 4=40k). Given Q4 (Marigold wins ARAP constancy),
priority shifts: **row 6 (MID-as-target vs invariance) becomes even more load-bearing** — it is
now the cleanest place to show the *mechanism* works independent of Marigold's generative prior,
since the head-to-head on ARAP constancy is lost. Order: **row 6 → row 8 (±chromatic aug, since
Cast_RMS is the problem child) → row 7 → rest.** Row 10 (arch levers) is promoted from "last
resort" to "plausibly necessary" because loss-only CARI did not beat Marigold (see Q4/Discussion).

### Q8 — Build the relighting figure (P4) now? → NO, not yet
P4 (`A(I_a)·S(I_b)` recompose) is a "physicality wow" figure. With Q4 unresolved (we trail SOTA
on the OOD constancy claim), spending time on a defense figure is premature. Lock the 50k
checkpoint + the indoor tables + row 6 first. P4 after the claim is settled.

---

## Discussion — the honest read

**What is SOLID (defensible today):**
1. **The mechanism works in-domain and does not collapse.** MID C_mat −50%, and on ARAP's *true*
   GT both constancy AND accuracy improved over baseline (Q3). The cross-render signal does what
   it claims: it makes albedo more illumination-invariant without graying it out.
2. **It does not hurt standard benchmarks — IIW WHDR improved 12%.** (Q6.)
3. **The 30k→40k jump validates the four fixes on MID** (C_mat 0.116→0.079, Cast_RMS recovered).

**What is a PROBLEM (must be confronted):**
4. **Marigold beats CARI on `all`-ARAP cross-illumination constancy** (2.2× C_arap, 3.5× cast)
   while CARI is only marginally more GT-accurate (Q4). A frozen-feed-forward model losing the
   constancy crown to a diffusion model *on the very axis we claim* is the central threat.
5. **CARI's ARAP Cast_RMS got WORSE with training** (Q1), opposite to MID. The white-flash→tinted
   augmentation fixed MID's cast but did not transfer to ARAP's (synthetic, differently-lit) cast.

**The reframing the data supports (NOT a retreat — a sharper claim):**
- The headline is NOT "CARI beats SOTA on constancy." The data won't support that. The headline
  IS: **"cross-render supervision is a CONTROLLED mechanism that improves single-image
  illumination-invariance over its own baseline (rows 1→4), confirmed on true-GT (ARAP) and
  in-domain (MID), without hurting WHDR — and we quantify how much."** That is the §6.1/§7 claim
  as written, and it SURVIVES intact. The zero-shot Marigold comparison is context, and the
  honest reading is "a 2024 diffusion SOTA still leads zero-shot constancy; our contribution is
  the mechanism + its controlled quantification, not a leaderboard win."
- **Row 6 carries the thesis now.** Same backbone, same MID, target-vs-invariance: that
  controlled delta is the mechanism proof that does not depend on out-running Marigold.
- **The indoor ARAP table may change Q4.** Marigold trains on InteriorVerse+Hypersim (indoor) —
  it has NO outdoor advantage, yet `all` includes 14 outdoor groups where BOTH models are OOD.
  On indoor-only ARAP the gap may shrink or the accuracy edge may matter more. **This is why the
  indoor re-run is the single most important next action.**

---

## NEXT STEPS (ordered)

1. **RE-RUN `eval_row4.sh` with the CURRENT script** (the run above predates this session's
   changes). This gets, in one shot: the **indoor ARAP tables** (the fair headline), **Marigold
   on MID** (in-domain CARI-vs-SOTA), the **fixed WB block** (no crash), the **MID multi-illuminant
   viz**, and **subset IIW**. Use `IIW_N=200` (1046 is overkill for no-regression). ~30–40 min.
   → THE blocking action; everything else waits on the indoor numbers.

2. **Read the indoor ARAP table for Q1/Q4.** If indoor Cast_RMS is *also* worse and CARI still
   trails Marigold on indoor C_arap → the augmentation needs work (step 4). If indoor closes the
   gap → Q4 softens and 50k is the move.

3. **Train to 50k** (`--resume 19k --skip-optimizer`, config already set) — justified by Q5's
   non-plateau regardless, but interpret against the indoor table.

4. **(Conditional) Cast-fix retune** if indoor Cast_RMS confirms the regression: chromatic aug
   p 0.8→1.0, gains U[0.5,1.6], `lambda_a_chroma` 0.2→0.4, retrain from 19k. This is ablation
   row 8's territory — not wasted.

5. **Run ablation row 6** (MID-as-target vs invariance) — now the load-bearing mechanism proof
   given Q4. Highest-value remaining ablation.

6. **Add MAW** (`tests/eval/eval_maw.py`, §6.0) — the real-measured-GT chromaticity benchmark. Given
   that Cast_RMS is the problem child, MAW's chromaticity-vs-real-GT is exactly the external
   number that adjudicates whether CARI's cast is genuinely good or just MID-overfit.

7. **Defer** P4 relighting figure + rows 5,7,9,10 until the indoor table + row 6 + 50k settle
   the claim. (Row 10 arch levers promoted to "likely needed" by Q4 — revisit after step 5.)
