# Report benchmark & visualization plan (draft-generation stage)

Working plan for filling Chapter 5 (and the Ch2 dataset section + Ch6 applications)
with real numbers and figures. Not part of the compiled report. Drafted 2026-07-04.

Modeled on **CD-IID** (Careaga & Aksoy 2024 — MAW/ARAP tables, multi-method
qualitative grid, params comparison, cited-vs-run separation) and the **Mesh
Reconstruction / GS-2M thesis** (validation-datasets figure, experiments split by
task, per-scene qualitative grids, limitations chapter).

---

## Checkpoint → report-row mapping (what we actually have)

**Table A — CARI 2×2 ablation (matched-step; report all as "50k"):**
| Row | Config | Cross-render | Colour-skip | Checkpoint |
|-----|--------|--------------|-------------|------------|
| 1 | v17_41 | off | off | iter_40000 |
| 2 | v17_42 | on  | off | iter_40000 |
| 3 | v17_43 | off | on  | iter_50000 |
| 4 | v17_44 | on  | on  | iter_50000 (full CARI) |

**Table B — fine-tuning lever sweep (all resume base@50k, +10k → 60k):**
| Row | Config | Levers | Checkpoint |
|-----|--------|--------|------------|
| ctrl | v17_20 | none | iter_60000 |
| flat | v17_21 | alb_flat_lf 4.0 | iter_60000 |
| sh_inv | v17_22 | shadow_inv 1.0 | iter_60000 |
| **combo** | **v17_23** | **flat 4.0 + sh_inv 1.0** | **iter_60000** |
| sh_inv+explain | v17_24 | inv 1.0 + explain 0.5 (kill-gate demo) | iter_60000 |
| ordinal | v17_25 | ordinal 0.5 | **LEAVE EMPTY** (per user) |
| iiw-ft / full-combo | v17_26 | — | **LEAVE EMPTY** (no ckpt) |

**"Ours (full model)"** = **UNDETERMINED** — do not lock it yet. It will be
chosen from the Table-B lever winners once v17_25 (ordinal) and v17_26 finish.
Leave the "Ours (full model)" cells EMPTY in all 4 benchmark tables for now.
v17_23 (combo) is reported as a Table-B *row*, not promoted to the headline.

✅ Base lineage RESOLVED (user, 2026-07-04): base@50k full CARI is now stored
at **checkpoints/v17_44** — this is both Table-A Row 4 AND the base that every
Table-B row (v17_20–24) resumes from. So the ablation and the lever sweep share
one lineage; the matched-step framing is honest.

---

## PART 1 — Quantitative content (tables)

Keep the existing table skeletons; fill from the 25-sample draft run, then the
full split. Each benchmark table gets a **run-local block** and, below a
`\midrule`/horizontal rule (CD-IID style), a **cited block**.

### T5.1 Methods overview — DONE (params/training-data/output). Add cited-method params later.

### T5.2 MID — C_mat (our GT-free constancy metric)
- Run-local: Ours (full = v17_23), Marigold-App, Marigold-Light, CD-IID†, Ordinal Shading†.
- Cited: none (C_mat is our own metric; no paper reports it → cited block omitted).

### T5.3 / T5.4 ARAP — C_arap, Cast_RMS, si-RMSE, LMSE (indoor + coloured subset)
- Run-local: Ours (v17_23), Marigold-App, Marigold-Light, CD-IID†, Ordinal Shading†.
- Cited: **caution** — CD-IID Table 2 reports LMSE/RMSE/SSIM on a *different* ARAP
  protocol (augmented with MIST, ~50 scenes, they removed duplicates). Only LMSE
  overlaps our metric set. Recommendation: keep ARAP **run-local only**, or cite
  CD-IID's LMSE column in a clearly-footnoted separate mini-row. Do NOT mix
  C_arap/Cast (our metrics) with their table.

### T5.5 MAW — chromaticity ΔE, intensity SI-MSE ×100  ← our metrics == CD-IID Table 1
This is the strongest cited table because the metric is identical.
- **Run-local block:** Ours (v17_23), Marigold-App, Marigold-Light, CD-IID†, Ordinal Shading†.
- **Cited block (below rule; from CD-IID 2024 Table 1 / MAW authors):**
  | Method | Intensity ×100 ↓ | Chromaticity ↓ |
  |--------|------------------|----------------|
  | Bell et al. 2014 | 3.11 | 6.61 |
  | CGIntrinsics (Li & Snavely 2018) | 2.71 | 5.15 |
  | NIID-Net (Luo et al. 2020) | 1.24 | 4.73 |
  | Zhu et al. 2022 | 1.44 | 4.94 |
  | Kocsis et al. 2024 | 1.13 | 5.35 |
  | Chen et al. 2024 | 0.98 | 4.12 |
  | CD-IID (Careaga & Aksoy 2024) | 0.54 | 3.37 |
  Footnote: "numbers reproduced from Careaga & Aksoy (2024), Table 1; the first
  block is computed by the MAW dataset authors. Grayscale-shading methods have a
  fixed chromaticity (6.56)."

### T5.6 IIW — WHDR
- Run-local: Ours (v17_23), Ours-IIW-ft (empty, v17_26 not trained), Marigold-App, Marigold-Light, CD-IID†, Ordinal Shading†.
- Cited (below rule; transcribe WHDR from each paper — TODO, only fill confident):
  CGIntrinsics, PIE-Net, CRefNet† (IIW-trained), Ordinal Shading, CD-IID.
  → exact WHDR values to transcribe live in THESIS_RESULTS_TODO.md (do not guess).

### T5.7 Ablation (CARI 2×2) — v17_41–44
Fills Rows 1–4 (finally including Rows 2 & 3) with MID C_mat, ARAP Cast_RMS, IIW WHDR.

### T5.8 Lever sweep — v17_20 (ctrl) vs 21 (flat) vs 22 (sh_inv) vs 24 (kill-gate)
Plus v17_23 (combo) as the adopted config. Ordinal row (v17_25) left empty.

† = weights are public/vendored → user will run locally later; leave the cell
"-" now with a footnote, do NOT cite a paper number for these (per user rule:
public weights → run it ourselves, don't cite).

---

## PART 2 — SOTA comparison roster (decided 2026-07-04)

**Vendored competitor code already in `documents/references/`:** chrislib
(Careaga: Ordinal Shading + CD-IID), CRefNet, PIE-Net, IntrinsicAnything
(= Chen et al. 2024), diffusion-perception (GenPercept, Lotus, diffusion-e2e-ft),
GS-2M (the reference thesis), Marigold.

**Weight-availability check (beyond Marigold ✓ and CD-IID ✓):**
| Method | Weights | Runnable in our pipeline? | Decision |
|--------|---------|---------------------------|----------|
| CD-IID (Careaga '24) | chrislib auto-download | yes (colour, our metrics fit) | **run-local** |
| Ordinal Shading (Careaga '23) | chrislib auto-download | yes | **run-local** |
| Marigold-App / Light | ✓ have | yes | run-local (done) |
| CRefNet (Luo '23) | public (drive) | yes, but grayscale + IIW-trained | **cite** IIW WHDR |
| PIE-Net (Das '22) | public but Torch7 `.t7` | high effort | **cite** IIW WHDR |
| IntrinsicAnything / Chen '24 | HF | object/single-view diffusion, slow, ill-suited to scenes | **cite** MAW |
| Kocsis '24, NIID-Net, IRISformer/Zhu '22, CGIntrinsics, Bell '14 | mixed | not worth integrating | **cite** MAW/IIW |

**Roster decision:**
- **Run-local (produce numbers + visualisations):** Ours (TBD), Marigold-App,
  Marigold-Light, CD-IID, Ordinal Shading. → 2 diffusion + 2 Careaga + ours =
  covers diffusion / feed-forward-CNN paradigms; CD-IID is the direct colour rival.
- **Cite-only (below a horizontal rule, footnoted):**
  - MAW: Bell '14, CGIntrinsics '18, NIID-Net '20, IRISformer/Zhu '22,
    Kocsis '24, IntrinsicAnything/Chen '24 (numbers from CD-IID Tab 1 — have them).
  - IIW WHDR: CGIntrinsics, PIE-Net, CRefNet† (transcribe from their papers — TODO).
- **Rule:** cite only where metric AND protocol match (MAW ✓ our eval == CD-IID
  Tab 1; IIW WHDR ✓ with subset caveat; ARAP ✗ different protocol → run-local only).
- **SAW** (Kovacs '17): a shading-quality benchmark (AP%), not a model. Our suite
  already covers perceptual structure via IIW; adding SAW = optional 5th axis,
  out of scope unless a shading-quality axis is wanted. Mention in related work.
- Bib entries to add: kocsis24, chen24intrinsicanything, zhu22irisformer
  (li18cgintrinsics ✓, luo20niid ✓, das22pienet ✓, luo23crefnet ✓).

**Next integration step (separate from the 25-sample draft):** wire CD-IID +
Ordinal Shading (chrislib) into an eval adapter so eval_{mid,maw,arap,iiw}.py can
run them like Marigold; then they move from "-" to real numbers + appear in the
multi-method comparison grid.

---

## PART 3 — Qualitative content (figures)

Money figures first (comparison = best, per CD-IID Fig 7):

1. **Multi-method albedo grid (NEW, CD-IID Fig 7 style)** — one/two in-the-wild
   images; rows = Input / Ours (v17_23) / Marigold-App / Marigold-Light / [CD-IID
   later]. Optional second block for shading. → needs an assembler script
   (`tests/viz/make_comparison_sheet.py`).
2. **Benchmark-datasets figure (NEW, Mesh §2.4 style)** — 1–2 example scenes per
   dataset (MID stack, ARAP group, MAW masked-GT, IIW pairs), captioned with what
   GT each provides. Static assembly.
3. **MID cross-illuminant constancy** (existing sheet, regen from v17_23) — strip
   + CoV heatmap; the headline constancy figure.
4. **ARAP constancy** (existing bedroom sheet, regen from v17_23).
5. **MAW** (existing 3-panel, regen at 1280 from v17_23 + Marigold).
6. **IIW** (existing, regen as Input | Ours | Marigold-App 3-col).
7. **Ablation qualitative** (CARI 2×2 corners, regen from v17_41 & v17_44).
8. **Architecture diagram** — existing TikZ Fig 3.1 (keep).
9. **Inverse-shading figure** — existing cropped Fig 3.1 from Ordinal Shading (keep).
10. **Failure cases (NEW, Mesh Ch6 style)** — emissive surfaces (TV screens),
    hard window shadows; supports the limitations section.
11. **Applications (Ch6)** — relighting recomposition + highlight removal; needs
    the recompose script (already tracked in THESIS_RESULTS_TODO.md §E).

---

## PART 4 — Code changes + run command

**Viz code changes needed:**
- eval_maw.py: vis_dir mkdir + --infer-max-size + --scene-filter — DONE.
- eval_mid_constancy.py: --infer-max-size — DONE.
- NEW `tests/viz/make_comparison_sheet.py`: assemble the multi-method grid (fig 1)
  from per-method albedo PNGs the evals already dump (--pred-dir / --save_dir).
- All four evals already emit per-method sheets; the draft run just requests them.

**Run command:** `scripts/eval_draft_25.sh` — 25 samples, all 4 benchmarks, the
right checkpoints, staged so the headline table (Ours + Marigold) lands first.
Inference resolution unified at 1280.
