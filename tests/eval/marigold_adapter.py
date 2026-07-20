"""Marigold-IID v1-1 albedo extraction — shared by eval_mid/eval_maw/eval_arap/eval_iiw.

The Marigold-IID v1-1 pipeline (documents/references/marigold) returns
``MarigoldIIDOutput`` objects whose per-target ``out['albedo']`` is an ``IIDEntry``
holding the prediction in ``.array`` (CHW, [0,1]). The colour-space metadata lives on
the PIPELINE (``pipe.target_properties[name]``), NOT on the entry — so the older
``out['albedo'].target_properties`` access raises ``AttributeError: 'IIDEntry' object
has no attribute 'target_properties'`` and ``np.array(out['albedo'])`` yields a 0-d
object array. This helper reads the correct locations and returns LINEAR albedo HWC,
matching the crefnet_adapter / ordinal_adapter output convention.

Verified against documents/references/marigold/marigold/marigold_iid_pipeline.py
(IIDEntry.array is [3,H,W] in the model's native prediction space; fill_entry's
per-space handling — 'srgb' left as-is, 'linear' optionally up_to_scale-normalised):
  - appearance: target_properties['albedo'] = {'prediction_space': 'srgb'}
  - lighting  : target_properties['albedo'] = {'prediction_space': 'linear', 'up_to_scale': False}
"""
from __future__ import annotations

import numpy as np


def marigold_albedo_hwc_linear(pipe, out, target: str = 'albedo') -> np.ndarray:
    """Return the target modality as LINEAR [0,1] HWC float32 from a v1-1 pipeline output."""
    entry = out[target]
    arr = entry.array if getattr(entry, 'array', None) is not None else np.asarray(entry)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 3:       # CHW -> HWC
        hwc = np.transpose(arr, (1, 2, 0))
    else:
        hwc = arr
    props = (getattr(pipe, 'target_properties', None) or {}).get(target, {}) or {}
    space = props.get('prediction_space', 'srgb')
    hwc = np.clip(hwc, 0.0, 1.0)
    if space == 'srgb':
        return (hwc ** 2.2).astype(np.float32)    # sRGB -> linear
    if space == 'linear':
        if props.get('up_to_scale', False):
            hwc = hwc / max(float(np.nanmax(hwc)), 1e-6)
        return hwc.astype(np.float32)
    return hwc.astype(np.float32)                 # 'stack'/unknown: pass through
