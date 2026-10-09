# Figure provenance

Every figure in the report, the script that produces it, and the model it shows. Builders write
into `documents/thesis/images/<subdir>/`; TikZ/pgfplots figures rebuild on compile.

The reported model is **ciai at 40k** (`src/configs/ciai.yaml`, `checkpoints/ciai/checkpoint_iter_40000.pth`). All
builders default to it. Figures marked **regenerate** were last rendered with an earlier
checkpoint of the same model trained 20k further steps (old runs v17_29 or v17_34 from the removed
refinement study, now under `checkpoints/history/`); re-run their builder before publishing.

| Figure | Builder | Model | Status |
|---|---|---|---|
| `fig:intro_layers`, `fig:architecture` thumbnails | `tests/viz/gen_arch_thumbs.py` | ciai | regenerate |
| `fig:ciai` thumbnails | `tests/viz/gen_ciai_thumbs.py` | ciai | regenerate |
| `fig:inverse_shading` | `tests/viz/build_inverse_shading_figure.py` | data only | current |
| `fig:formulation` | `tests/viz/build_formulation_figure.py` | data only | current |
| `fig:wb_lanes`, `fig:dataset_roles` | `tests/viz/build_dataset_roles_figure.py` | data only | current |
| `fig:mid_preprocessing` | `build_hires_figures.py fig_mid_preprocessing` | data only | current |
| `fig:datasets` | `tests/viz/build_datasets_figure.py` | data only | current |
| `fig:comp_grid` | `build_hires_figures.py fig_comp_grid` | roster | regenerate (Ours) |
| `fig:mid_constancy`, `fig:intro_constancy` | `build_hires_figures.py fig_mid_ours` | ciai | regenerate |
| `fig:iiw` | `build_hires_figures.py fig_iiw_ours` | ciai | regenerate |
| `fig:maw` | `build_hires_figures.py fig_maw_ours` | ciai | regenerate |
| `fig:maw_resolution` | `build_hires_figures.py fig_maw_resolution` | ciai | regenerate |
| `fig:arap_constancy`, `fig:arap_models` | `build_hires_figures.py fig_arap_ours`, `fig_arap_model_grid` | roster | regenerate (also after the ARAP mask fix) |
| `fig:limit_reflective` | `tests/viz/build_presentation_figures.py` (crop of the page teaser) | v17_29 | regenerate |
| `fig:ablation` | `build_hires_figures.py fig_ablation` | ablation rows (ablation_none, ablation_ciai, ablation_colour, ciai) | current |
| `fig:tradeoff` | `chapters/fig_tradeoff.tex` | values from `documents/results/mid_per_scene.json` | current |
| Chapter 6 edits | `tests/viz/build_ch6_figures.py` | ciai | regenerate |

Project page figures come from `tests/viz/build_web_figures.py` (teaser, ambiguity, pairs,
mechanism, qualitative, portfolio thumbnail) and `tests/viz/build_compare_widget_assets.py`
(compare widget). `tests/viz/build_presentation_figures.py` adds the metric diagrams, the
baseline comparison under two lights and the reflective-surface crop; it reuses only deck images
without our model's predictions, because the deck's "Ours" panels came from the over-smoothed
v17_34 checkpoint. The teaser, qualitative matrix and compare widget show the v17_29 checkpoint and
need the same regeneration. The portfolio card thumbnail is cut from the teaser, so it follows it.

Regeneration needs the MID test split (`../datasets/MIDIntrinsics/test`), the MAW, IIW and ARAP
test data under `tests/testing_data/`, the baseline checkpoints under `checkpoints/`, and a
complete ciai checkpoint. The copy currently in `checkpoints/ciai/` is truncated (26 MB of
about 1.4 GB) and does not load.

## Next-stage pipeline figure

`docs/static/img/ciai-pipeline.jpg` (also `documents/thesis/images/readme/`) is built by
`tests/viz/build_next_stage_figure.py` with no model: two lightings of one 3D-Front-IID view, with the
ground-truth targets S = I / A, S_lum and C drawn as the three outputs. The photographs are cropped from
`documents/thesis/images/front3d/front3d_dataset.jpg`; pass `--front3d-root` (the full-resolution corpus)
to regenerate them sharply.
