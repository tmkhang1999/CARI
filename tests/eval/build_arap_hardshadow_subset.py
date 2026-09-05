"""Measure hard-shadow density per ARAP scene, using the SAME definition as the V21 gate.

WHY
---
The thesis's first named limitation is that hard shadows survive into the albedo.
V21 attributes this to missing supervision: 3D-Front v1 renders a hard-edge fraction
of 0.022 against ARAP indoor 0.115-0.186, so the training data simply contains almost
no sharp cast shadows.

That comparison deserves to be reproducible from our own code rather than quoted, and
we need a *pre-registered* hard-shadow subset of ARAP to test any future claim on.

DEFINITION (identical to scripts/validate_3dfront_v2.py:91)
    hard_edge_fraction = mean( |grad(log S_lum)| > 0.3 )   over valid pixels

ARAP ships no shading GT, so shading is derived as S = I / A_gt. Deriving it this way
is exact under the diffuse model and needs no extra data. The gradient of log S is
invariant to any per-image scale on I or A, so ARAP's documented input/albedo scale
mismatch (ratio spans 0.16-3.6e5) cannot bias this statistic -- but the MASK does
depend on the albedo encoding, which is why canonicalisation is a prerequisite.

Usage:
    python tests/eval/build_arap_hardshadow_subset.py
"""

from __future__ import annotations

import os
import sys
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arap_preprocess import (  # noqa: E402
    list_scenes, light_stems, load_arap_gt, load_arap_input,
    valid_mask, DEFAULT_MASK_FRAC,
)
from build_arap_colour_subset import INDOOR  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DIR = os.path.join(ROOT, 'tests', 'testing_data', 'ARAP_dataset')

# Matches validate_3dfront_v2.py:91 exactly so ARAP and 3D-Front numbers are comparable.
GRAD_THRESHOLD = 0.3

_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)


def hard_edge_fraction(inp: np.ndarray, albedo_canon: np.ndarray,
                       mask: np.ndarray) -> dict | None:
    """Fraction of valid pixels whose |grad(log shading luminance)| exceeds 0.3."""
    m = mask & (albedo_canon > 1e-6).all(axis=-1) & (inp > 0).all(axis=-1)
    if int(m.sum()) < 1000:
        return None

    shading = inp.astype(np.float64) / np.maximum(albedo_canon.astype(np.float64), 1e-6)
    s_lum = shading @ _LUMA
    s_lum = np.where(np.isfinite(s_lum) & (s_lum > 0), s_lum, np.nan)
    if np.isnan(s_lum).all():
        return None

    log_s = np.log(np.where(np.isnan(s_lum), 1.0, s_lum))
    gy, gx = np.gradient(log_s)
    grad = np.hypot(gx, gy)

    v = grad[m & np.isfinite(grad) & ~np.isnan(s_lum)]
    if v.size < 1000:
        return None
    return {
        'hard_edge_fraction': float((v > GRAD_THRESHOLD).mean()),
        'p95_grad_log_shading': float(np.percentile(v, 95)),
        'n_valid': int(v.size),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-dir', default=DEFAULT_DIR)
    ap.add_argument('--mask-frac', type=float, default=DEFAULT_MASK_FRAC)
    ap.add_argument('--out', default=os.path.join(ROOT, 'documents', 'evals',
                                                  'arap_hardshadow_subset.json'))
    args = ap.parse_args()

    per = {}
    for scene in list_scenes(args.dataset_dir):
        try:
            gt = load_arap_gt(args.dataset_dir, scene)
        except (OSError, FileNotFoundError):
            continue
        m = valid_mask(gt.rgb, args.mask_frac)

        # Base frame plus every light variant; report the base (or the median variant)
        # so a scene gets one number regardless of how many lights it ships.
        stems = [scene] + light_stems(args.dataset_dir, scene)
        vals = {}
        for stem in stems:
            try:
                img = load_arap_input(args.dataset_dir, stem)
            except (OSError, FileNotFoundError):
                continue
            if img.rgb.shape != gt.rgb.shape:
                continue
            r = hard_edge_fraction(img.rgb, gt.rgb, m)
            if r is not None:
                vals[stem] = r
        if not vals:
            continue
        hef = [v['hard_edge_fraction'] for v in vals.values()]
        per[scene] = {
            'domain': 'indoor' if scene in INDOOR else 'outdoor',
            'encoding': gt.encoding,
            'valid_frac': round(float(m.mean()), 4),
            'n_frames': len(vals),
            'hard_edge_fraction_median': round(float(np.median(hef)), 4),
            'hard_edge_fraction_max': round(float(np.max(hef)), 4),
            'per_frame': {k: round(v['hard_edge_fraction'], 4) for k, v in vals.items()},
        }

    print(f'{"scene":14s} {"domain":8s} {"valid":>7s} {"frames":>7s} '
          f'{"hard-edge frac":>15s}')
    print('-' * 58)
    for s in sorted(per, key=lambda k: -per[k]['hard_edge_fraction_median']):
        r = per[s]
        print(f'{s:14s} {r["domain"]:8s} {r["valid_frac"]*100:6.1f}% '
              f'{r["n_frames"]:7d} {r["hard_edge_fraction_median"]:15.4f}')

    ind = [r['hard_edge_fraction_median'] for r in per.values() if r['domain'] == 'indoor']
    out = [r['hard_edge_fraction_median'] for r in per.values() if r['domain'] == 'outdoor']
    print(f'\nARAP indoor  n={len(ind):2d}  median {np.median(ind):.4f}  '
          f'p25 {np.percentile(ind, 25):.4f}  p75 {np.percentile(ind, 75):.4f}')
    print(f'ARAP outdoor n={len(out):2d}  median {np.median(out):.4f}  '
          f'p25 {np.percentile(out, 25):.4f}  p75 {np.percentile(out, 75):.4f}')
    print('\nreference (same definition, scripts/validate_3dfront_v2.py):')
    print('  3D-Front v1 (in V17 training)  0.022')
    print('  3D-Front v2 pilot              0.0596   (gate 0.070 -> FAIL)')

    payload = {
        '_generated': '2026-09-05',
        '_script': 'tests/eval/build_arap_hardshadow_subset.py',
        '_definition': f'mean(|grad(log S_lum)| > {GRAD_THRESHOLD}) over canonical-mask '
                       f'valid pixels, S = I / A_gt; identical to '
                       f'scripts/validate_3dfront_v2.py:91',
        '_reference': {'front3d_v1': 0.022, 'front3d_v2_pilot': 0.0596,
                       'front3d_v2_gate': 0.070},
        'mask_frac': args.mask_frac,
        'per_scene': per,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(payload, f, indent=1)
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
