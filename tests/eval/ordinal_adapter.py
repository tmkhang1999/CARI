"""Ordinal Shading adapter — shared by eval_arap.py and eval_maw.py.

Ordinal Shading (Careaga & Aksoy 2023, "Intrinsic Decomposition via Ordinal
Shading") is stage 1 of the CD-IID pipeline (documents/references/Intrinsic
CD-IID): a grayscale ordinal-shading network + a second network that converts
the ordinal estimate into real (inverse) shading, from which albedo is derived
by division. It ships as the `intrinsic` pip package (already installed in
this env, along with its `altered_midas` and `chrislib` dependencies — see
that repo's setup.py) with weights fetched live via torch.hub from GitHub
releases (verified reachable 2026-07-12; ~484MB combined ord+iid checkpoint).

We call the paper's own `run_gray_pipeline` directly rather than re-deriving
the ordinal-to-albedo math, so this adapter is a thin, faithful wrapper:
  - Input: sRGB [0,1] HWC numpy array (the function gamma-decodes internally).
  - `resize_conf=<int>` reproduces our "long-side cap" protocol exactly
    (documents/references/Intrinsic CD-IID/intrinsic/pipeline.py:176-179:
    "rescale image so that the large side of the image matches the specified
    integer value").
  - Output albedo (`gry_alb`) is the LINEAR implied albedo I/shading, matching
    the same linear-albedo convention our other adapters/metrics use.
"""
from __future__ import annotations

import numpy as np
import torch

_load_models = None
_run_gray_pipeline = None
_uninvert = None


def _ensure_ordinal_imported():
    global _load_models, _run_gray_pipeline, _uninvert
    if _load_models is not None:
        return
    from intrinsic.pipeline import load_models, run_gray_pipeline
    from chrislib.general import uninvert
    _load_models, _run_gray_pipeline, _uninvert = load_models, run_gray_pipeline, uninvert


# Weight names accepted by intrinsic.pipeline.load_models (V1 releases only — stage=1
# forces the paper's original ordinal + grayscale-albedo networks, no colour/diffuse stages).
_WEIGHTS = {
    'ordinal': 'paper_weights',
    'ordinal-rendered-only': 'rendered_only',
}


def load_ordinal(device, variant: str = 'ordinal'):
    """Download (first call only, via torch.hub) and build the stage-1 Ordinal Shading models."""
    _ensure_ordinal_imported()
    if variant not in _WEIGHTS:
        raise ValueError(f"Unknown Ordinal Shading variant {variant!r}; expected one of {list(_WEIGHTS)}")
    models = _load_models(_WEIGHTS[variant], stage=1, device=str(device))
    print(f'  Loaded Ordinal Shading ({variant}) — stage 1 (ord_model + iid_model)')
    return models


def run_ordinal(models, rgb_display_linear: np.ndarray, max_size: int, device):
    """Run the stage-1 grayscale pipeline.

    Args:
        rgb_display_linear: display-space LINEAR [0,1] HWC float array — same
            convention as crefnet_adapter.run_crefnet's input. This function does
            the sRGB gamma encoding itself (run_gray_pipeline expects sRGB, then
            gamma-decodes internally via **2.2 when linear=False).
        max_size: long-side cap in pixels (passed straight through as an int
            `resize_conf`, matching our shared per-dataset resolution protocol).

    Returns:
        (albedo_hwc, shading_hwc): float32 HWC arrays in [0,1], LINEAR space, at the
        input's original resolution (`maintain_size=True`).
    """
    _ensure_ordinal_imported()
    img = np.power(np.clip(rgb_display_linear, 0.0, 1.0), 0.45454545454545453).astype(np.float32)
    results = _run_gray_pipeline(
        models, img,
        resize_conf=int(max_size), base_size=384,
        maintain_size=True, linear=False, device=str(device),
    )
    albedo = np.clip(np.nan_to_num(results['gry_alb'], nan=0.0, posinf=1.0), 0.0, 1.0).astype(np.float32)
    shading = _uninvert(results['gry_shd']).astype(np.float32)
    if shading.ndim == 2:
        shading = np.repeat(shading[..., None], 3, axis=-1)
    return albedo, shading
