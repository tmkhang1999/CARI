"""CD-IID adapter — Colorful Diffuse Intrinsic Image Decomposition (Careaga & Aksoy, TOG 2024).

This is the direct rival of this thesis: CD-IID is the one published method that
explicitly models *coloured* illumination rather than assuming white light, and it
leads the MAW chromaticity score (dE 3.37) that our own colour claim is measured
against. Chapter 5 excludes it because only its stage-1 (Ordinal Shading) network
was wired into our harness; this adapter closes that gap by running the full
five-stage cascade.

It ships in the same `intrinsic` pip package as ordinal_adapter.py (see that file
for the install note); the difference is `load_models('v2')` — the 2024 release,
whose stage_0..stage_4 checkpoints are fetched via torch.hub from the repo's
GitHub releases (~1.5 GB total) — instead of the V1 `paper_weights` at stage=1.

Pipeline contract, read off intrinsic/pipeline.py:run_pipeline:
  - Input is sRGB when `linear=False`; the function gamma-decodes internally and
    keeps the decoded copy as `results['lin_img']`.
  - `hr_alb` is the stage-3 high-resolution albedo, and it lives in the LINEAR
    domain: the very next line computes `hr_shd = lin_img / hr_alb`. That matches
    the linear-albedo convention our metrics and every other adapter use, so no
    gamma correction is applied here. (Getting this wrong is exactly the
    double-gamma bug that once inflated Marigold's constancy scores.)
  - `dif_shd` is the stage-4 diffuse shading, already un-inverted by the pipeline.

CAVEAT for whoever reads the numbers: CD-IID's training set includes MIDIntrinsics
(Chapter 5, tab:methods). Scores on our MID benchmark are therefore *not* zero-shot
for this method and must be labelled the same way CRefNet's IIW WHDR is. ARAP and
MAW remain held out for it.
"""
from __future__ import annotations

import numpy as np

_load_models = None
_run_pipeline = None


def _ensure_imported():
    global _load_models, _run_pipeline
    if _load_models is not None:
        return
    from intrinsic.pipeline import load_models, run_pipeline
    _load_models, _run_pipeline = load_models, run_pipeline


def load_cdiid(device, variant: str = 'v2'):
    """Download (first call only, via torch.hub) and build the full CD-IID cascade."""
    _ensure_imported()
    if variant != 'v2':
        raise ValueError(f"Unknown CD-IID variant {variant!r}; only 'v2' is released")
    models = _load_models('v2', stage=4, device=str(device))
    print('  Loaded CD-IID (v2) — full 5-stage colourful-diffuse cascade')
    return models


def run_cdiid(models, rgb_display_linear: np.ndarray, max_size: int, device):
    """Run the full colourful-diffuse pipeline.

    Args:
        rgb_display_linear: display-space LINEAR [0,1] HWC float array — the same
            convention as run_ordinal / run_crefnet. Gamma-encoded here because
            run_pipeline expects sRGB and decodes it again internally.
        max_size: long-side cap in pixels, passed through as `resize_conf`.

    Returns:
        (albedo_hwc, shading_hwc): float32 HWC [0,1] LINEAR arrays at the input's
        original resolution.
    """
    import cv2

    _ensure_imported()
    h, w = rgb_display_linear.shape[:2]
    img = np.power(np.clip(rgb_display_linear, 0.0, 1.0), 0.45454545454545453).astype(np.float32)
    results = _run_pipeline(
        models, img,
        stage=4, resize_conf=int(max_size), base_size=384,
        linear=False, device=str(device),
    )
    albedo = np.clip(np.nan_to_num(results['hr_alb'], nan=0.0, posinf=1.0), 0.0, 1.0).astype(np.float32)
    shading = np.nan_to_num(results['dif_shd'], nan=0.0, posinf=1.0).astype(np.float32)
    if shading.ndim == 2:
        shading = np.repeat(shading[..., None], 3, axis=-1)

    # run_pipeline() has no `maintain_size` flag (unlike run_gray_pipeline, which
    # ordinal_adapter uses), so its outputs come back at the internally resized
    # resolution. Every other adapter returns predictions at the input's own size,
    # and the metrics index them against full-resolution masks, so restore that here.
    if albedo.shape[:2] != (h, w):
        albedo = cv2.resize(albedo, (w, h), interpolation=cv2.INTER_LINEAR)
        shading = cv2.resize(shading, (w, h), interpolation=cv2.INTER_LINEAR)
    return albedo.astype(np.float32), shading.astype(np.float32)
