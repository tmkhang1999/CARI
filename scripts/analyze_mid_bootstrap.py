#!/usr/bin/env python3
"""Paired scene-bootstrap CIs for the MID metrics (P0 #3 from THESIS_REVIEW_2026-07-14).

The review's objection is that the tables rank point estimates with no uncertainty, so
differences well inside the noise were being bolded as wins. This computes, over the 30 MID
test scenes:

  * per-model 95% CIs (percentile bootstrap, resampling SCENES);
  * PAIRED deltas vs a reference model, resampling the same scene indices for both — which is
    the correct test, since the two models see identical scenes and their errors are
    correlated. An unpaired test would be needlessly conservative here.

It also reports, for Chroma_fid, both the mean-of-ratios (what the thesis printed: 1.008) and
the ratio-of-aggregates (0.941). They differ by Jensen; the latter is the honest statement of
"how much chroma spread is retained", and the former should not be quoted.

Usage: python scripts/analyze_mid_bootstrap.py tests/visualizations/rerun_mid_perscene
"""
import glob
import json
import sys

import numpy as np

REF = 'v17_34'          # "Ours"
N_BOOT = 10000
RNG = np.random.default_rng(0)
# lower-is-better for all of these; Chroma_fid is a ratio whose target is 1.0
METRICS = ['C_mat', 'Cast_rel', 'Chroma_err']


def ci(v, lo=2.5, hi=97.5):
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else 'tests/visualizations/rerun_mid_perscene'
    models = {}
    for p in sorted(glob.glob(f'{d}/mid_*.json')):
        for r in json.load(open(p)).get('results', []):
            if 'per_scene' in r:
                models[r['label']] = r
    if not models:
        sys.exit(f'no per-scene results in {d} (rerun eval_mid_constancy with the '
                 'per-scene patch)')

    # align scenes across models
    names = [list(m['per_scene']['scene']) for m in models.values()]
    common = sorted(set(names[0]).intersection(*map(set, names[1:]))) if len(names) > 1 \
        else names[0]
    n = len(common)
    print(f'{len(models)} models, {n} common scenes, {N_BOOT} bootstrap resamples\n')

    def vec(m, key):
        ps = models[m]['per_scene']
        idx = {s: i for i, s in enumerate(ps['scene'])}
        return np.array([ps[key][idx[s]] for s in common], dtype=float)

    boot_idx = RNG.integers(0, n, size=(N_BOOT, n))

    # ── per-model CIs ────────────────────────────────────────────────────────
    print('=' * 92)
    print('PER-MODEL 95% CI  (percentile bootstrap over scenes)')
    print('=' * 92)
    hdr = f"{'model':16}" + ''.join(f'{m:>25}' for m in METRICS)
    print(hdr)
    print('-' * 92)
    order = [REF] + [m for m in models if m != REF]
    for m in order:
        if m not in models:
            continue
        row = f'{m:16}'
        for k in METRICS:
            v = vec(m, k)
            bs = np.nanmean(v[boot_idx], axis=1)
            lo, hi = ci(bs)
            row += f'{np.nanmean(v):8.3f} [{lo:.3f},{hi:.3f}]'.rjust(25)
        print(row)

    # ── paired deltas vs REF ─────────────────────────────────────────────────
    print()
    print('=' * 92)
    print(f'PAIRED DELTA vs {REF}   (negative = {REF} better; CI crossing 0 = NOT a win)')
    print('=' * 92)
    for k in METRICS:
        print(f'\n  {k}:')
        ref = vec(REF, k)
        for m in order[1:]:
            other = vec(m, k)
            dv = ref - other                      # ours - theirs
            bs = np.nanmean(dv[boot_idx], axis=1)
            lo, hi = ci(bs)
            sig = 'ns (crosses 0)' if lo <= 0 <= hi else \
                  ('OURS BETTER' if hi < 0 else 'ours worse')
            print(f'    {REF} - {m:16} {np.nanmean(dv):+7.3f}  [{lo:+.3f},{hi:+.3f}]   {sig}')

    # ── Chroma_fid: mean-of-ratios vs ratio-of-aggregates ────────────────────
    print()
    print('=' * 92)
    print('Chroma_fid AGGREGATION  (the review is right: these are not the same number)')
    print('=' * 92)
    print(f"{'model':16}{'mean-of-ratios':>18}{'median':>10}{'max':>9}"
          f"{'ratio-of-aggregates':>22}")
    print('-' * 92)
    for m in order:
        if m not in models:
            continue
        v = vec(m, 'Chroma_fid')
        roa = models[m].get('Chroma_fid_ratio_of_aggregates', float('nan'))
        print(f'{m:16}{np.nanmean(v):18.3f}{np.nanmedian(v):10.3f}{np.nanmax(v):9.3f}'
              f'{roa:22.3f}')
    print('\n  The thesis quoted the mean-of-ratios. Quote the ratio-of-aggregates instead,')
    print('  and report Chroma_err (above) as the primary hue-calibration guard.')


if __name__ == '__main__':
    main()
