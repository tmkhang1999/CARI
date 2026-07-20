# Thesis ablation design — FINAL (2026-07-07)

Internal planning doc (pending language allowed here, never in the draft).
Companion: `THESIS_RESULTS_TODO.md` (evidence debt), `documents/references/sota_pipeline_improvement_panel.md` (§10 dataset direction).

## 0. One family tree

Every table row descends from one lineage — no cross-lineage comparisons anywhere:

```
v17_44@50k  (= v17_old@50k, relocated; "the base model")
   ├── Table 1 siblings: v17_41 / v17_42 / v17_43 (re-runs at the same 50k horizon)
   └── Table 2 rows (50k → 60k/70k): v17_20 ctrl · 21 flat · 22 sh_inv · 23 combo · 24 +explain · 25 ordinal
          └── combo = v17_23@70k  ("the fine-tuned model")
                 └── Table 3 ladder (70k → 80k): L0 · L1 · L2 · L3  →  **Ours**
```

Training story for the draft = a **three-stage curriculum**: (S1) supervised
decomposition pre-training [Table 1] → (S2) prior/lever fine-tuning [Table 2]
→ (S3) invariance stage: cross-render objectives + synthetic relight pairs
[Table 3]. This is the honest description of Ours' history (stages S1–S2
trained without the cross-render objectives; S3 switches them on) and standard
fine-tuning practice — no bug narrative needed in the draft.

## 1. Principles

- **Step-matched rows, one variable per row**, deltas vs the in-table control.
- **Noise floor**: Table 1 contains two independent runs per condition; cite
  |1a − 1b| per metric as the run-to-run spread. Any delta in ANY table smaller
  than this floor is not a claim.
- **Metric columns identical across tables** — constancy: MID C_mat, ARAP
  C_arap, ARAP Cast_RMS, MAW ΔE · guards: ARAP si-RMSE, IIW WHDR.
- **Kill gates**: constancy axes must not regress vs the row's control; guards
  within ~10%.
- **No codenames in the draft** — use the thesis labels below; the ckpt column
  here is internal bookkeeping only.

## 2. Table 1 — Architecture (all @50k, step-matched, same data)

| Thesis label | ckpt | role |
|---|---|---|
| Direct-albedo baseline | `v17_41@50k` | axis arm A |
| Direct-albedo baseline (independent run) | `v17_42@50k` | **noise floor** vs 41 |
| + wide color path & chroma objective | `v17_43@50k` | axis arm B |
| **The base model** (as shipped; carried into Tables 2–3) | `v17_44@50k` | reference row only |

Claim: the color-path axis is **41 ↔ 43 ONLY** (same July code, same data —
both trained with IV 0.3 and MID from step 25k; single variable = color path).
Noise floor = 41 ↔ 42. **Never use 43 ↔ 44 as a replicate or an axis**: the
config diff (in-ckpt) proves v17_44 trained under a different DATA regime —
(i) MID from step 20k (30k steps of it, vs 43's 25k), and (ii) **zero
InteriorVerse** (its IV root `../datasets/InteriorVerse` never existed and the
June-era loader only constructed hypersim+mid, renormalizing the phase-3 mix
to ≈ hyp 0.57 / mid 0.43) — plus fresh-Adam resume. All nine extra July-era
loss keys in 43's log average exactly 0.0, so the objectives match; the 43↔44
gap is data exposure, not losses. 44 sits in the table as lineage/reference.
**Do NOT present this table as testing the cross-render objectives** — that
axis lives in Table 3.

## 3. Table 2 — Fine-tuning levers (base@50k → 60k/70k, deltas vs ctrl)

| Thesis label | ckpt | status |
|---|---|---|
| Continuation control (no lever) | `v17_20` | ✅ banked |
| + low-frequency albedo flatness | `v17_21` | ✅ banked |
| + shadow invariance | `v17_22` | ✅ banked |
| + both — **the fine-tuned model** | `v17_23@70k` | ✅ banked |
| + shadow inv. + explain (kill-gate demonstration) | `v17_24` | ✅ optional row |
| + ordinal shading | `v17_25` | ✅ optional row |

Claim: single-lever attribution; the combo passes the gates and is carried
forward. All rows (incl. ctrl and the base they resume) share the identical
training condition, so attribution is unconfounded.

## 4. Table 3 — Invariance stage (combo@70k → 80k, deltas vs L0) — THE NEW TABLE

| Thesis label | row | ckpt | status |
|---|---|---|---|
| Continuation control (+10k) | L0 | `v17_23/prefix_nopairs_iter_80000.pth` | ✅ done (rename!) |
| + cross-render objectives (L_inv, L_explain on real raw pairs) | L1 | `v17_23` relaunch @80k | ⏳ run now, GPU 0 |
| + cross-render + rectified-gradient MSG (θ=0.2) | L2 | `v17_27@80k` | ⏳ slack slot; cut first |
| + cross-render + synthetic relight pairs (3D-FRONT, 0.25) | L3 | `v17_28@80k` | ⏳ after render + `--revalidate` |

Claims: **L1−L0 = the cross-render objective's effect (the thesis mechanism
evidence)** · L3−L1 = synthetic pair-data effect · L2−L1 = RGF garnish.
**Ours = kill-gate winner of {L1, L2, L3}; fallback = the fine-tuned model
(combo@70k).** If L1−L0 lands below the Table-1 noise floor, the constancy
mechanism is attributed to the data path (raw pairs + 3D-FRONT), carried by
L3−L1 — the narrative survives either outcome.
Optional L4 (front3d + RGF combined): only if L2 AND L3 both pass gates and
≥1.5 days remain.

## 5. Table 4 — SOTA comparison

Ours (single highlighted row) vs Marigold-IID, CD-IID, … on the corrected
evaluators (post output-space fix). Optionally include "ours w/o stages S2–S3"
(= base@50k) as a reference row. Only Ours carries the SOTA claim; ablation
checkpoints never appear here.

`v17_26` (IIW ordinal fine-tune) = a separate "leaderboard variant" paragraph
if ever trained — not a row in any table above.

## 6. Remaining run/eval checklist

1. `mv checkpoints/v17_23/checkpoint_iter_80000.pth checkpoints/v17_23/prefix_nopairs_iter_80000.pth`
2. Launch **L1** on GPU 0 (`train.sh --version 17.23 --cuda 0 --resume checkpoints/checkpoint_iter_70000.pth`).
   After ~1h: TB `1. Losses/CARI_L_inv` nonzero (pairs firing) + note steps/h.
3. **L0 eval suite can start immediately** on GPU 1 alongside the blender
   render (inference ≈ few GB; fits next to blender on 24 GB). Never two
   *trainers* + render at once — that's what killed the Jul-7 18:45 launch.
4. Render done (~30h) → `render_3dfront_dataset.py --out ~/datasets/front3d_iid --revalidate` → launch **L3** on GPU 1.
5. L1 done → launch **L2** on GPU 0. Eval each row as it lands.
6. Pick **Ours** by the gates → run its Table-4 eval → freeze the draft numbers.
