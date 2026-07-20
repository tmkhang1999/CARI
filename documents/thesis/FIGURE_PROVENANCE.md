# Figure provenance manifest

Every figure in the thesis, mapped to the exact script/function that produces it, its data
source, and (where a model is involved) the checkpoint used. Regenerate a figure by running
its builder; TikZ/pgfplots figures are `\input` from `chapters/fig_*.tex` and rebuild on
compile, but any raster thumbnails they embed come from the listed script.

Paths are relative to the repo root `/home/khang/IR-IID`. All figure assets live under
`documents/thesis/images/<subdir>/`; builders write there directly.

Last updated 2026-07-15.

---

## Method chapter (Ch. 3)

| Figure | Kind | Builder | Data / model |
|---|---|---|---|
| `fig:architecture` | TikZ + thumbnails | `chapters/fig_architecture.tex`; thumbnails `tests/viz/gen_arch_thumbs.py` | thumbnails = v17_29 @ 60k on MID `everett_dining1` (CPU inference) |
| `fig:cari` | TikZ + thumbnails | `chapters/fig_cari.tex`; thumbnails `tests/viz/gen_cari_thumbs.py` | thumbnails = v17_29 @ 60k, two illuminants (warm=dir18, cool=dir12) of `everett_dining1` |
| `fig:inverse_shading` | pgfplots + panel | `chapters/fig_inverse_shading.tex`; panel + histogram data `tests/viz/build_inverse_shading_figure.py` | Hypersim `ai_001_001`; S = I/A from GT; hist data → `chapters/data/shading_hist_*.dat` |

## Method / implementation (Ch. 4)

| Figure | Kind | Builder | Data / model |
|---|---|---|---|
| `fig:mid_preprocessing` | raster | `tests/viz/build_hires_figures.py` → `fig_mid_preprocessing()` | raw vs gray-world-white-balanced MID pair + GT albedo, `everett_dining1`; **no model** (data only) |
| `fig:front3d` | raster | `tests/viz/build_front3d_figure.py` | rescans the rendered 3D-Front-IID corpus (`../datasets/front3d_iid`); stats are re-measured, not asserted |
| `fig:curriculum` | TikZ | `chapters/Chapter4.tex` (inline TikZ, no external asset) | — |

## Experiments (Ch. 5)

| Figure | Kind | Builder | Data / model |
|---|---|---|---|
| `fig:datasets` | raster | `tests/viz/build_datasets_figure.py` | one panel per benchmark (MID/IIW/MAW/ARAP); IIW row overlays real WHDR judgement points |
| `fig:comp_grid` | raster | `tests/viz/build_hires_figures.py` → `fig_comp_grid()` | 3 IIW photos × full roster; per-panel scale-normalised; v17_34 = "Ours" |
| `fig:mid_constancy`, `fig:intro_constancy` | raster (same asset `hires/mid_ours.jpg`) | `tests/viz/build_hires_figures.py` → `fig_mid_ours()` | MID `everett_dining1`, 4 illuminants + GT + CoV map; v17_34 |
| `fig:iiw` | raster | `tests/viz/build_hires_figures.py` → `fig_iiw_ours()` | 4 IIW photos, input/albedo pairs; v17_34 |
| `fig:maw` | raster | `tests/viz/build_hires_figures.py` → `fig_maw_ours()` | scene 28 kitchen, scene 29 dark office, scene 33 living area; input/measured albedo/ours; explicitly v17_29 |
| `fig:arap_constancy` | raster | `tests/viz/build_hires_figures.py` → `fig_arap_ours()` | bedroom/classroom/livingroom, dense GT vs ours; explicitly v17_29 |
| `fig:maw_resolution` | raster | `tests/viz/build_hires_figures.py` → `fig_maw_resolution()` | one MAW photo at {512,768,1024,1280}px; v17_34 |
| `fig:ablation` (`hires/ablation_mid.jpg`) | raster | `tests/viz/build_hires_figures.py` → `fig_ablation()` | MID scene, 2 illuminants, Table-A rows v17_41/42/43/44 |
| `fig:shadow_aug` | raster | `tests/viz/build_hires_figures.py` → `fig_shadow_aug()` | composites `tests/visualizations/shadow_aug_out/*.png` (known relight fields) |
| `fig:chroma_fidelity` | raster | `tests/viz/build_chroma_fidelity_figure.py` | MID `everett_dining1` L0 and `everett_kitchen18` L18; pseudo-GT vs v17_34 / v17_42 / CRefNet / Marigold-App; Chroma_err and aggregate spread ratios from `rerun_mid_perscene/` |
| `fig:tradeoff` | pgfplots | `chapters/fig_tradeoff.tex` | Cast_rel vs Chroma_err scatter; values from `tests/visualizations/rerun_mid_perscene/` |
| `fig:formulation` | raster | `tests/viz/build_formulation_figure.py` | MID `everett_kitchen18`, light 18; grey-shading vs RGB-shading albedo; 2.31-vs-0.22 chroma-error demonstration |

## Applications (Ch. 6)

| Figure | Kind | Builder | Data / model |
|---|---|---|---|
| `fig:arap_models` | raster | `tests/viz/build_hires_figures.py` → `fig_arap_model_grid()` | ARAP `chocofur`; inputs and five local methods across L0/L1/L2 plus repeated dense GT; ours explicitly v17_29 |
| `fig:decomposition` | raster | `tests/viz/build_ch6_figures.py` → `fig_decomposition()` | fixed realistic indoor photograph; input, diffuse albedo $\mathbf{A}_d$, diffuse shading $\mathbf{S}_d$, analytic residual $\mathbf{R}$; v17_29 |

The builder retains `fig_relight()` and `score_transfer()` only as optional
diagnostic utilities behind `--score-transfer`; their asset and table are no longer
included in the thesis because compensating factor errors make the score degenerate.

## Conclusion (Ch. 7)

| Figure | Kind | Builder | Data / model |
|---|---|---|---|
| `fig:limitations` | raster | `tests/viz/build_limitations_figure.py` | MID scene; GT vs v17_34 vs v17_33 (flat-4.0 over-saturation) + cross-illuminant instability map |

---

## Notes

* **Thumbnail model choice.** `fig:architecture` and `fig:cari` use **v17_29** thumbnails
  (per author instruction); all Ch.5/6/7 result figures use **v17_34** ("Ours (full model)").
  The architecture is identical across v17 variants, so the encoder/decoder/head diagram is
  correct regardless of which checkpoint rendered the thumbnails.
* **Scene selection.** Most builders default to the first lexicographic MID scene
  (specified MID scenes) or a supplied real photo (`test_1.png` for Ch.6). A
  model-independent representative-scene rule is a remaining TODO from the external review;
  when adopted it should be recorded here.
* **CPU-generated thumbnails.** `gen_arch_thumbs.py` / `gen_cari_thumbs.py` run on CPU
  (`CUDA_VISIBLE_DEVICES=""`) so they never contend with training/eval jobs on the GPUs.
* **Regeneration.** Raster builders write straight into `documents/thesis/images/`, so after
  running one, a plain `latexmk` picks up the new asset. TikZ/pgfplots figures need only a
  recompile unless their embedded thumbnails or `.dat` files changed.

## README figures

The top-level README reuses thesis figures. The two TikZ ones are vector-only inside
`Main.pdf`, so they are rasterised by `tests/viz/render_readme_figures.sh` (pdflatex +
pdftoppm, 200 dpi) into `documents/thesis/images/readme/`:

| Asset | Source |
|---|---|
| `images/readme/architecture.png` | `chapters/fig_architecture.tex` |
| `images/readme/cari.png` | `chapters/fig_cari.tex` |

Re-run that script if either TikZ figure changes. The remaining README images are used
directly from `documents/thesis/images/` and are produced by the builders listed above.
