"""Seed-paired analysis for Study 1: contrast two configs, replicate over seeds.

WHY THIS EXISTS SEPARATELY FROM study1_readout.py
-------------------------------------------------
study1_readout.py assumes ONE run per config and identifies rows by config tag, so
two seeds of the same config collide ("two files map to row 17_62"). That was the
right shape for a single-seed study and is the wrong shape here.

The design this implements is the one the study actually uses, and it is the answer
to "why not just fix the seed?": the seed IS fixed within each contrast. For every
seed s we compute

    delta_s = metric(63_s) - metric(62_s)

with identical data order, crops and MID pair draws inside the pair. What replication
adds is the ability to tell that delta apart from trajectory divergence -- the loss
term perturbs the weights at step 1 and that perturbation amplifies chaotically over
21k steps, so a single delta_s contains BOTH the loss effect and the divergence. The
spread of delta_s across seeds is what separates them.

MEASURED NOISE FLOOR (2026-09-09, v17_62 s42 vs s43, identical config):
    HIGH/LOW Cast_rel ratio   0.014
    mean Cast_rel             0.004
    C_mat                     0.018  (10.7% relative)
    Chroma_fid                0.013
A delta smaller than these is not an effect, at any n.

CHECKPOINTS ARE NOT NEEDED, ONLY METRICS. Storage on this machine cannot hold the
study's checkpoints (~26 GB against ~3 GB of home quota and a ~16 GB session cap), so
rows are evaluated then discarded. This script therefore joins runs from SEPARATE
eval JSONs by scene, which is what makes evaluate-then-discard viable.

Usage:
    python tests/eval/study1_paired.py --contrast 17_62 17_63 \
        documents/evals/study1_round1_noisefloor.json \
        documents/evals/study1_headline_s42.json \
        documents/evals/study1_63_s43.json
"""

from __future__ import annotations

import os
import re
import sys
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mid_colour_stratified import measure_scene_gaps       # noqa: E402
from phase_b_readout import load_run, terciles, analyse    # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LABEL_RE = re.compile(r'v?(?P<cfg>1?7?_?\d+)_s(?P<seed>\d+)')

# Guard thresholds, fixed in documents/evals/STUDY1_PREREGISTRATION.md before results.
GUARDS = {
    'Chroma_fid': (-0.05, 'fell below -0.05 -- DESATURATION, not constancy'),
    'low_tercile': (0.01, 'rose above +0.01 -- gain is a pivot, not an improvement'),
    'C_mat': (0.01, 'rose above +0.01 -- intensity axis regressed'),
}
MIN_MEANINGFUL = 0.035   # n=3 minimum detectable difference at sd 0.014


def parse_label(label: str):
    """'v17_63_s42' -> ('17_63', 42). Raises rather than guessing, because a
    mis-parsed label would silently pair the wrong runs."""
    m = LABEL_RE.search(label)
    if not m:
        raise ValueError(f'cannot parse config/seed from label {label!r}; '
                         f'expected something like v17_63_s42')
    cfg = m.group('cfg')
    if not cfg.startswith('17_'):
        cfg = '17_' + cfg.lstrip('_')
    return cfg, int(m.group('seed'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+', help='eval_mid_constancy --save-json outputs')
    ap.add_argument('--contrast', nargs=2, metavar=('BASE', 'TREAT'), default=['17_62', '17_63'])
    ap.add_argument('--out', default=os.path.join(ROOT, 'documents', 'evals', 'study1_paired.json'))
    args = ap.parse_args()

    runs = {}
    for p in args.runs:
        for r in load_run(p):
            cfg, seed = parse_label(r['label'])
            if (cfg, seed) in runs:
                # The same checkpoint can legitimately appear in two eval JSONs (a run
                # re-evaluated alongside a later one). That is a free determinism check:
                # identical weights on identical scenes must give identical metrics, so
                # a disagreement means the EVALUATOR is noisy and every delta in this
                # study is suspect. Verify rather than silently keeping one.
                prev = runs[(cfg, seed)]
                d = max(float(np.nanmax(np.abs(prev[k] - r[k])))
                        for k in ('Cast_rel', 'C_mat', 'Chroma_fid'))
                if d > 1e-6:
                    raise ValueError(
                        f'{cfg}_s{seed} appears twice with DIFFERENT metrics '
                        f'(max per-scene diff {d:.2e}). The evaluator is not '
                        f'deterministic; deltas of order 0.01 cannot be trusted.')
                print(f'note: {cfg}_s{seed} evaluated twice, metrics identical '
                      f'(max diff {d:.1e}) -- evaluator is deterministic')
                continue
            runs[(cfg, seed)] = r

    # All runs must cover the same scenes in the same order, or the pairing is wrong.
    ref = next(iter(runs.values()))['scenes']
    for (cfg, seed), r in runs.items():
        if r['scenes'] != ref:
            raise ValueError(f'{cfg}_s{seed} has a different scene list; cannot pair')

    gaps = np.array([measure_scene_gaps(ref)[s]['median_gap'] for s in ref], dtype=float)
    stats = {k: analyse(r, gaps) for k, r in runs.items()}

    print(f'MID test: {len(ref)} scenes, illuminant gap {gaps.min():.4f}-{gaps.max():.4f}')
    print(f'\n{"run":<16}{"ratio":>8}{"Cast_rel":>10}{"C_mat":>8}{"Chr_fid":>9}{"LOW":>8}{"HIGH":>8}')
    print('-' * 67)
    for (cfg, seed) in sorted(stats):
        a = stats[(cfg, seed)]
        print(f'{cfg}_s{seed:<9}{a["high_over_low"]:>8.3f}{a["Cast_rel_mean"]:>10.4f}'
              f'{a["C_mat_mean"]:>8.3f}{a["Chroma_fid_mean"]:>9.3f}'
              f'{a["tercile_means"][0]:>8.3f}{a["tercile_means"][2]:>8.3f}')

    base, treat = args.contrast
    seeds = sorted({s for (c, s) in runs if c == base} & {s for (c, s) in runs if c == treat})
    if not seeds:
        print(f'\nno seed has BOTH {base} and {treat}; nothing to contrast')
        return

    print(f'\n{"="*67}\nPAIRED CONTRAST  {treat} - {base}   (seed fixed within each pair)\n{"="*67}')
    keys = [('ratio', 'high_over_low'), ('Cast_rel', 'Cast_rel_mean'),
            ('C_mat', 'C_mat_mean'), ('Chroma_fid', 'Chroma_fid_mean')]
    deltas = {name: [] for name, _ in keys}
    deltas['low_tercile'], deltas['high_tercile'] = [], []
    for s in seeds:
        a, b = stats[(base, s)], stats[(treat, s)]
        for name, k in keys:
            deltas[name].append(b[k] - a[k])
        deltas['low_tercile'].append(b['tercile_means'][0] - a['tercile_means'][0])
        deltas['high_tercile'].append(b['tercile_means'][2] - a['tercile_means'][2])

    hdr = ''.join(f'{"s"+str(s):>10}' for s in seeds)
    print(f'{"metric":<14}{hdr}{"mean":>10}{"spread":>9}')
    print('-' * (14 + 10 * len(seeds) + 19))
    for name in ['ratio', 'Cast_rel', 'C_mat', 'Chroma_fid', 'low_tercile', 'high_tercile']:
        v = np.array(deltas[name])
        sp = f'{v.max()-v.min():.4f}' if len(v) > 1 else 'n/a'
        print(f'{name:<14}' + ''.join(f'{x:>10.4f}' for x in v) + f'{v.mean():>10.4f}{sp:>9}')

    n = len(seeds)
    mean_ratio = float(np.mean(deltas['ratio']))
    print(f'\nn = {n} seed pair(s)')

    failures = []
    for gname, (thr, msg) in GUARDS.items():
        m = float(np.mean(deltas[gname]))
        if (thr < 0 and m < thr) or (thr > 0 and m > thr):
            failures.append(f'{gname} {m:+.4f} {msg}')

    if mean_ratio >= 0:
        verdict = 'NULL on the colour axis (ratio did not fall)'
    elif abs(mean_ratio) < MIN_MEANINGFUL:
        verdict = (f'BELOW THRESHOLD: |{mean_ratio:.4f}| < {MIN_MEANINGFUL} minimum '
                   f'meaningful effect -- reported as a null, not a trend')
    elif failures:
        verdict = 'NOT A GAIN'
    else:
        verdict = 'REDUCES COLOUR-DEPENDENCE'

    print(f'\nVERDICT: {verdict}')
    for f in failures:
        print(f'   ! {f}')
    if n < 3:
        print(f'   NOTE: n={n}. The preregistered threshold assumes n=3; with fewer seeds a')
        print('         null is provisional. A guard failure of several times the noise')
        print('         floor is still informative at n=1-2 -- magnitude, not count, is what')
        print('         makes it readable.')

    payload = {'_script': 'tests/eval/study1_paired.py', 'contrast': [base, treat],
               'seeds': seeds, 'deltas': {k: list(map(float, v)) for k, v in deltas.items()},
               'verdict': verdict, 'guard_failures': failures,
               'runs': {f'{c}_s{s}': v for (c, s), v in stats.items()}}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(payload, f, indent=1)
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
