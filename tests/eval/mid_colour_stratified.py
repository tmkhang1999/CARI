"""Stratify MID constancy results by how much the illuminant COLOUR actually changes.

WHY
---
Cast_rel and Chroma_fid are reported on MID, but MID's illuminants are 25 white
flashes bounced around the room -- the code has known this all along
(midintrinsic_dataset.py:48: "MID's 25 flashes are all WHITE and probe-WB'd").
What colour variation survives is bounce off coloured surfaces, and it is small.

Measured here from the gray probes (the direct measurement of the light):

    median pairwise chromaticity gap  0.030
    fraction of pairs >= 0.08          14%
    per-scene medians span            0.019 - 0.115

So a scene-averaged Cast_rel is an average over a corpus that barely exercises the
axis it scores. This script splits the existing per-scene results by per-scene
illuminant colour range, which answers a question the aggregate cannot:

    does a method's chroma drift actually TRACK illuminant colour change?

A method whose Cast_rel is flat across strata is colour-invariant; one whose
Cast_rel climbs is not. That distinction is invisible in the reported mean.

Reads only artefacts already on disk -- no GPU, no model inference.

Usage:
    python tests/eval/mid_colour_stratified.py
"""

from __future__ import annotations

import os
import json
import argparse
from itertools import combinations

import numpy as np

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import cv2  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PER_SCENE = os.path.join(ROOT, 'documents', 'thesis', 'data', 'mid_per_scene.json')
MID_TEST = '/home/khang/datasets/MIDIntrinsics/test'
CACHE = os.path.join(ROOT, 'documents', 'evals', 'mid_illuminant_gaps.json')

# MID skips these flash indices (hard flash / saturated), matching
# midintrinsic_dataset.py:65 `skip_list`. Using the same set keeps the measured
# illuminant statistics consistent with what training and evaluation actually see.
SKIP = {2, 3, 20, 21, 24}
VALID_IDX = [i for i in range(25) if i not in SKIP]


def probe_chromaticity(scene_dir: str, idx: int) -> np.ndarray | None:
    """Mean chromaticity of the gray probe's central disc = the illuminant colour.

    The gray probe is a matte neutral sphere, so its measured chromaticity IS the
    incident light's chromaticity, with no albedo to divide out. Sampling only the
    central disc avoids the sphere's grazing-angle rim and its dark surround.
    """
    p = os.path.join(scene_dir, 'probes', f'dir_{idx}_gray256.exr')
    if not os.path.exists(p):
        return None
    im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    rgb = im[:, :, :3][:, :, ::-1].astype(np.float64)
    h, w, _ = rgb.shape
    yy, xx = np.mgrid[0:h, 0:w]
    disc = ((yy - h / 2) ** 2 + (xx - w / 2) ** 2) < (min(h, w) * 0.28) ** 2
    v = rgb[disc]
    v = v[np.isfinite(v).all(axis=1)]
    v = v[v.sum(axis=1) > 1e-6]
    if len(v) < 50:
        return None
    m = v.mean(axis=0)
    return m / m.sum()


def measure_scene_gaps(scenes, mid_root=MID_TEST, cache=CACHE):
    if os.path.exists(cache):
        with open(cache) as f:
            d = json.load(f)
        if all(s in d.get('per_scene', {}) for s in scenes):
            return d['per_scene']

    per = {}
    for s in scenes:
        sd = os.path.join(mid_root, s)
        ch = {i: probe_chromaticity(sd, i) for i in VALID_IDX}
        ch = {i: c for i, c in ch.items() if c is not None}
        gaps = [float(np.hypot(ch[i][0] - ch[j][0], ch[i][2] - ch[j][2]))
                for i, j in combinations(sorted(ch), 2)]
        per[s] = {
            'n_illum': len(ch),
            'median_gap': float(np.median(gaps)) if gaps else None,
            'p90_gap': float(np.percentile(gaps, 90)) if gaps else None,
            'max_gap': float(max(gaps)) if gaps else None,
            'frac_ge_008': float(np.mean(np.array(gaps) >= 0.08)) if gaps else None,
        }
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, 'w') as f:
        json.dump({'_note': 'per-scene MID illuminant chromaticity gaps from gray probes',
                   '_skip_indices': sorted(SKIP), 'per_scene': per}, f, indent=1)
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-scene', default=PER_SCENE)
    ap.add_argument('--metric', default='Cast_rel')
    ap.add_argument('--out', default=os.path.join(ROOT, 'documents', 'evals',
                                                  'mid_colour_stratified.json'))
    args = ap.parse_args()

    with open(args.per_scene) as f:
        d = json.load(f)
    scenes, methods = d['scenes'], d['methods']

    per = measure_scene_gaps(scenes)
    gaps = np.array([per[s]['median_gap'] for s in scenes], dtype=float)

    print(f'MID test: {len(scenes)} scenes')
    print(f'per-scene median illuminant gap: min {gaps.min():.4f}  '
          f'median {np.median(gaps):.4f}  max {gaps.max():.4f}')
    print('reference: 3D-Front v1 = 0.151, 3D-Front v2 pilot = 0.243, '
          'ARAP indoor = 0.070\n')

    # Terciles by per-scene colour range.
    order = np.argsort(gaps)
    n = len(scenes)
    t = [order[:n // 3], order[n // 3:2 * n // 3], order[2 * n // 3:]]
    names = ['LOW colour', 'MID colour', 'HIGH colour']
    for lbl, idx in zip(names, t):
        print(f'  {lbl:12s} n={len(idx):2d}  gap {gaps[idx].min():.4f}-{gaps[idx].max():.4f}')

    m = args.metric
    print(f'\n=== {m} by illuminant-colour tercile ===')
    print(f'{"method":18s} {"LOW":>9s} {"MID":>9s} {"HIGH":>9s} {"HIGH/LOW":>9s} '
          f'{"pearson r":>10s}')
    print('-' * 70)
    results = {}
    for name, v in methods.items():
        if m not in v:
            continue
        y = np.array(v[m], dtype=float)
        means = [float(np.mean(y[i])) for i in t]
        ratio = means[2] / means[0] if means[0] else float('nan')
        r = float(np.corrcoef(y, gaps)[0, 1])
        results[name] = {'tercile_means': means, 'high_over_low': ratio, 'pearson_r': r}
        print(f'{name:18s} {means[0]:9.3f} {means[1]:9.3f} {means[2]:9.3f} '
              f'{ratio:9.2f} {r:10.3f}')

    print('\nA method whose score is FLAT across terciles (ratio ~1.0, r ~0) is')
    print('colour-invariant: its residual drift is not driven by illuminant colour.')
    print('A rising ratio means the method degrades as the light changes colour.')

    payload = {
        '_generated': '2026-09-05',
        '_script': 'tests/eval/mid_colour_stratified.py',
        '_metric': m,
        '_note': ('Per-scene MID illuminant chromaticity gap measured from gray '
                  'probes; scenes split into terciles by that gap; existing '
                  'per-scene metrics re-aggregated within each tercile. No model '
                  'inference -- reads documents/thesis/data/mid_per_scene.json.'),
        'scene_gaps': {s: per[s] for s in scenes},
        'terciles': {lbl: [scenes[i] for i in idx] for lbl, idx in zip(names, t)},
        'results': results,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(payload, f, indent=1)
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
