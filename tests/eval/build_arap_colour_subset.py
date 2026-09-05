"""Build a pre-registered ARAP-Colour subset: scene pairs whose ILLUMINANT COLOUR changes.

MOTIVATION
----------
MID -- the corpus supplying CARI's real training pairs AND the corpus on which
Cast_rel / Chroma_fid are reported -- barely varies illuminant colour. Measured from
its gray probes over 30 test scenes / 9000 illuminant pairs (2026-09-05):

    median pairwise chromaticity gap   0.030
    fraction of pairs >= 0.08          14%          (0.08 = V21's stated minimum
    fraction of pairs >= 0.15           4%           meaningful colour separation)

So the colour axis is scored on a benchmark that hardly exercises it. ARAP's
multi-light scenes DO contain genuine illuminant-colour change and are external,
real, and already on disk -- but ARAP has never been used this way.

This script measures the illuminant chromaticity of every ARAP light variant and
emits a frozen subset of pairs above a chosen gap, to be registered BEFORE any model
is run on it.

CORRECTNESS
-----------
All loading goes through arap_preprocess, which fixes the three-encoding GT problem
documented there. Chromaticity itself is scale-invariant, so the input/albedo
encoding mismatch cannot bias these numbers -- but the *mask* selecting which pixels
are measured is not scale-invariant, which is exactly why the canonical mask matters
here. Under the old absolute mask 6 scenes had an empty mask and 23/51 kept <5% of
the frame; those scenes would have been silently dropped or measured on a tiny,
brightness-biased sliver.

Usage:
    python tests/eval/build_arap_colour_subset.py --min-gap 0.15
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arap_preprocess import (  # noqa: E402
    list_scenes, light_stems, load_arap_gt, load_arap_input,
    valid_mask, illuminant_chromaticity, chroma_gap, DEFAULT_MASK_FRAC,
)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DIR = os.path.join(ROOT, 'tests', 'testing_data', 'ARAP_dataset')

# Content taxonomy: indoor scenes are in-domain for a model trained on
# Hypersim + InteriorVerse + MID. Mirrors ARAP_types.json's own domain split.
INDOOR = {
    'attic', 'workshop', 'breakfast', 'lobby', 'classroom', 'classroom2', 'kitchen',
    'staircase', 'corridor', 'conference', 'livingroom', 'villa', 'whiteroom',
    'bedroom', 'bamboo', 'bedroom2', 'chocofur', 'restroom', 'oldclassroom',
    'blenderscene', 'bread', 'iron', 'strawberries', 'camera', 'postit', 'violin',
    'cream', 'sponza', 'sanmiguel', 'cathedral',
}


def measure(dataset_dir: str, mask_frac: float):
    """Per-scene illuminant chromaticity for every light variant."""
    out = {}
    for scene in list_scenes(dataset_dir):
        stems = light_stems(dataset_dir, scene)
        if len(stems) < 2:
            continue
        try:
            gt = load_arap_gt(dataset_dir, scene)
        except (OSError, FileNotFoundError) as e:
            out[scene] = {'error': f'gt: {e}'}
            continue

        m = valid_mask(gt.rgb, mask_frac)
        chroma, skipped = {}, {}
        for stem in stems:
            try:
                img = load_arap_input(dataset_dir, stem)
            except (OSError, FileNotFoundError) as e:
                skipped[stem] = f'load: {e}'
                continue
            if img.rgb.shape != gt.rgb.shape:
                # Never silently resize GT to match: a resample would blur the
                # albedo edges the mask depends on. Record and skip instead.
                skipped[stem] = f'shape {img.rgb.shape[:2]} != gt {gt.rgb.shape[:2]}'
                continue
            c = illuminant_chromaticity(img.rgb, gt.rgb, m)
            if c is None:
                skipped[stem] = 'insufficient valid pixels'
                continue
            chroma[stem] = c

        rec = {
            'encoding': gt.encoding,
            'gt_canon_max': round(float(gt.rgb.max()), 5),
            'valid_frac': round(float(m.mean()), 5),
            'n_lights_found': len(stems),
            'n_lights_measured': len(chroma),
            'domain': 'indoor' if scene in INDOOR else 'outdoor',
            'chroma': {k: [round(float(x), 6) for x in v] for k, v in chroma.items()},
            'skipped': skipped,
        }
        if len(chroma) >= 2:
            pairs = []
            for a, b in combinations(sorted(chroma), 2):
                pairs.append({'a': a, 'b': b,
                              'gap': round(chroma_gap(chroma[a], chroma[b]), 5)})
            pairs.sort(key=lambda p: -p['gap'])
            rec['pairs'] = pairs
            rec['max_gap'] = pairs[0]['gap']
            rec['median_gap'] = round(float(np.median([p['gap'] for p in pairs])), 5)
        out[scene] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-dir', default=DEFAULT_DIR)
    ap.add_argument('--min-gap', type=float, default=0.15,
                    help='minimum illuminant chromaticity gap for subset membership')
    ap.add_argument('--mask-frac', type=float, default=DEFAULT_MASK_FRAC)
    ap.add_argument('--out', default=os.path.join(ROOT, 'documents', 'evals',
                                                  'arap_colour_subset.json'))
    args = ap.parse_args()

    meas = measure(args.dataset_dir, args.mask_frac)

    print(f'{"scene":14s} {"dom":8s} {"enc":13s} {"valid":>7s} {"lights":>7s} '
          f'{"median":>8s} {"max gap":>8s}')
    print('-' * 74)
    for s in sorted(meas):
        r = meas[s]
        if 'error' in r or 'pairs' not in r:
            print(f'{s:14s} {"-":8s} {"-":13s} {"-":>7s} {"-":>7s}   '
                  f'{r.get("error", "too few measurable lights")}')
            continue
        print(f'{s:14s} {r["domain"]:8s} {r["encoding"]:13s} '
              f'{r["valid_frac"]*100:6.1f}% {r["n_lights_measured"]:2d}/'
              f'{r["n_lights_found"]:<4d} {r["median_gap"]:8.4f} {r["max_gap"]:8.4f}')

    subset = []
    for s in sorted(meas):
        for p in meas[s].get('pairs', []):
            if p['gap'] >= args.min_gap:
                subset.append({'scene': s, 'domain': meas[s]['domain'],
                               'a': p['a'], 'b': p['b'], 'gap': p['gap']})
    subset.sort(key=lambda r: -r['gap'])

    allgaps = np.array([p['gap'] for r in meas.values() for p in r.get('pairs', [])])
    indoor = np.array([p['gap'] for r in meas.values() if r.get('domain') == 'indoor'
                       for p in r.get('pairs', [])])

    print(f'\nAll measurable pairs: n={len(allgaps)}  median={np.median(allgaps):.4f}  '
          f'p90={np.percentile(allgaps, 90):.4f}  max={allgaps.max():.4f}')
    print(f'Indoor only:          n={len(indoor)}  median={np.median(indoor):.4f}  '
          f'p90={np.percentile(indoor, 90):.4f}')
    print(f'\nARAP-Colour subset (gap >= {args.min_gap}): {len(subset)} pairs across '
          f'{len({r["scene"] for r in subset})} scenes')
    for r in subset:
        print(f'   {r["gap"]:.4f}  {r["scene"]:14s} {r["domain"]:8s} {r["a"]} vs {r["b"]}')

    payload = {
        '_generated': '2026-09-05',
        '_script': 'tests/eval/build_arap_colour_subset.py',
        '_note': ('Illuminant chromaticity from median chroma(I/A_gt) over the '
                  'canonical-albedo valid mask. GT albedo canonicalised by '
                  'arap_preprocess (three on-disk encodings unified). Gap is '
                  'Euclidean distance in the (r,b) chromaticity plane -- the same '
                  'statistic used for MID gray probes and 3D-Front, so numbers are '
                  'directly comparable across corpora.'),
        '_reference_points': {
            'MID_median_pair_gap': 0.030,
            'MID_frac_pairs_ge_0.08': 0.14,
            'front3d_v1_median_gap': 0.151,
            'front3d_v2_pilot_median_gap': 0.2425,
        },
        'min_gap': args.min_gap,
        'mask_frac': args.mask_frac,
        'subset': subset,
        'per_scene': meas,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(payload, f, indent=1)
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
