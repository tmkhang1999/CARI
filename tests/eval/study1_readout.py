"""Study 1 readout: does the chroma half of L_explain reduce colour-dependence?

WHAT THIS ADDS OVER phase_b_readout.py
--------------------------------------
That script answers one A/B (runs[0] = control, runs[1] = treatment) and hardcodes
that pairing. Study 1 is four rows and the question is NOT "did row 3 beat row 2" --
it is which PART of the explain constraint carries the effect. That needs three
contrasts read together:

    (62 - 61)  does CIAI do anything at all, on the rebuilt base
    (63 - 62)  what the chroma half ADDS to the published loss    <- the headline
    (64 - 62)  whether chroma can SUBSTITUTE for luminance
    (63 - 64)  whether the luminance half still contributes once chroma is present

The last one is the one most likely to be skipped and most likely to matter. If
(63 - 64) is ~0 then the luminance half is doing nothing that L_chr_explain and
L_inv do not already cover, and the honest claim is "replace L_explain", not "add a
term". Reporting only (63 - 62) would hide that.

WHY THE READ IS THE TERCILE RATIO, NOT MEAN Cast_rel
----------------------------------------------------
Unchanged from phase_b_readout.py, and it is the whole reason these metrics were
rebuilt: mean Cast_rel can be driven down by DESATURATING, which is exactly the
failure the corrected metric exists to catch (CRefNet posts good pooled invariance
at Chroma_fid 0.484). So every contrast here is gated on Chroma_fid, and a mean
that improves while the ratio does not is reported as a null, not a win.

The colour axis and the intensity axis are reported SEPARATELY and both must be
read. The model is already best-of-8 on intensity (HIGH/LOW 0.76) and worst-tier on
colour (1.42). A colour gain bought by regressing the weak-colour scenes is not a
gain -- hence the LOW-tercile guard, which is what would catch it.

Usage:
    for v in 61 62 63 64; do
      python tests/eval/eval_mid_constancy.py \
        --ckpts checkpoints/v17_$v/checkpoint_iter_40000.pth \
        --mid-root ../datasets/MIDIntrinsics --split test \
        --save-json documents/results/study1_v17_$v.json
    done
    python tests/eval/study1_readout.py documents/results/study1_v17_6*.json
"""

from __future__ import annotations

import os
import sys
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mid_colour_stratified import measure_scene_gaps            # noqa: E402
from phase_b_readout import load_run, terciles, analyse, REFERENCE  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# What each row is, keyed by the version tag that appears in its checkpoint path.
# Kept here so the printed table is self-describing and a mislabelled json cannot
# quietly turn into a wrong conclusion.
ROWS = {
    '17_61': ('no CIAI',            'L_inv 0    L_expl 0     L_chr 0'),
    '17_62': ('CIAI (published)',   'L_inv 0.5  L_expl 0.25  L_chr 0'),
    '17_63': ('CIAI + chroma',      'L_inv 0.5  L_expl 0.25  L_chr 0.25'),
    '17_64': ('chroma, no lum',     'L_inv 0.5  L_expl 0     L_chr 0.25'),
}

# The contrasts to print, in the order they should be read.
CONTRASTS = [
    ('17_61', '17_62', 'does CIAI do anything on the rebuilt base?'),
    ('17_62', '17_63', 'what the chroma half ADDS  [HEADLINE]'),
    ('17_62', '17_64', 'can chroma SUBSTITUTE for luminance?'),
    ('17_64', '17_63', 'does the luminance half still contribute?'),
]


def row_key(label: str, path: str) -> str:
    """Identify which study row a json belongs to. Checked against ROWS so an
    unrecognised file is reported rather than silently ordered by argv position --
    the failure mode that would swap control and treatment."""
    hay = f'{label} {path}'
    hits = [k for k in ROWS if k in hay.replace('.', '_')]
    if len(hits) != 1:
        raise ValueError(
            f'cannot identify the study row for {path!r} (label {label!r}); '
            f'matched {hits or "nothing"}. Expected exactly one of {sorted(ROWS)} '
            f'in the label or filename.')
    return hits[0]


def paired_bootstrap(cr_a, cr_b, gaps, n_boot=20000, seed=0):
    """95% CI on the HIGH/LOW ratio CHANGE (b - a). Paired: the same scenes appear
    in both rows, so resampling scenes independently per row would understate the
    precision. Terciles are recomputed inside each resample, because the split is
    itself a function of the resampled scenes."""
    rng = np.random.RandomState(seed)
    n = len(gaps)
    boot = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        o = np.argsort(gaps[idx])
        lo, hi = o[:n // 3], o[2 * n // 3:]
        al, ah = cr_a[idx][lo].mean(), cr_a[idx][hi].mean()
        bl, bh = cr_b[idx][lo].mean(), cr_b[idx][hi].mean()
        if al > 0 and bl > 0:
            boot.append((bh / bl) - (ah / al))
    boot = np.array(boot)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def contrast(a_key, b_key, why, by_key, raw_by_key, gaps):
    """One b - a comparison. Returns a dict and prints the reasoned verdict."""
    a, b = by_key[a_key], by_key[b_key]
    d_ratio = b['high_over_low'] - a['high_over_low']
    d_cast = b['Cast_rel_mean'] - a['Cast_rel_mean']
    d_fid = b['Chroma_fid_mean'] - a['Chroma_fid_mean']
    d_cmat = b['C_mat_mean'] - a['C_mat_mean']
    d_low = b['tercile_means'][0] - a['tercile_means'][0]
    d_high = b['tercile_means'][2] - a['tercile_means'][2]
    lo_ci, hi_ci = paired_bootstrap(raw_by_key[a_key]['Cast_rel'],
                                    raw_by_key[b_key]['Cast_rel'], gaps)

    print(f'\n{ROWS[b_key][0]}  -  {ROWS[a_key][0]}     ({why})')
    print(f'   HIGH/LOW ratio  {d_ratio:+.3f}   95% CI [{lo_ci:+.3f}, {hi_ci:+.3f}]'
          f'   (want NEGATIVE)')
    print(f'   mean Cast_rel   {d_cast:+.4f}')
    print(f'   Chroma_fid      {d_fid:+.4f}   (want ~0; a drop means colour was drained)')
    print(f'   C_mat           {d_cmat:+.4f}   (intensity axis -- must not regress)')
    print(f'   LOW tercile     {d_low:+.4f}   (want <= 0: weak-colour scenes must not regress)')
    print(f'   HIGH tercile    {d_high:+.4f}   (want < 0)')

    # EVERY failing guard is reported, not just the first. An if/elif chain here
    # masked a Chroma_fid collapse of -0.43 behind a LOW-tercile regression during
    # testing -- i.e. it hid the more serious finding (the model had drained colour,
    # the CRefNet failure at 0.484) behind the less serious one. These failures are
    # independent and a row can exhibit several at once.
    reasons = []
    if d_low > 0.01:
        reasons.append(f'LOW tercile regressed {d_low:+.4f} (gain is a pivot, not an improvement)')
    if d_fid < -0.05:
        reasons.append(f'Chroma_fid fell {d_fid:+.3f} -- DESATURATION, not constancy')
    if d_cmat > 0.01:
        reasons.append(f'C_mat rose {d_cmat:+.4f} -- intensity axis regressed')
    if d_ratio < 0 and hi_ci >= 0:
        reasons.append('ratio change not separable from zero')

    if d_ratio >= 0:
        verdict = 'NULL on the colour axis'
    elif not reasons:
        verdict = 'REDUCES COLOUR-DEPENDENCE'
    else:
        verdict = 'NOT A GAIN'
    print(f'   -> {verdict}')
    for r in reasons:
        print(f'      ! {r}')

    if d_cast < 0 and d_ratio >= 0:
        print('   NOTE: mean Cast_rel fell while the ratio did not. Not a colour-constancy')
        print('         gain -- check Chroma_fid for desaturation.')

    return {'a': a_key, 'b': b_key, 'why': why, 'd_ratio': d_ratio,
            'ci': [lo_ci, hi_ci], 'd_cast': d_cast, 'd_fid': d_fid, 'd_cmat': d_cmat,
            'd_low': d_low, 'd_high': d_high, 'verdict': verdict, 'failures': reasons}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+', help='eval_mid_constancy --save-json outputs')
    ap.add_argument('--out', default=os.path.join(ROOT, 'documents', 'results',
                                                  'study1_readout.json'))
    args = ap.parse_args()

    raw_by_key, paths = {}, {}
    for p in args.runs:
        for r in load_run(p):
            k = row_key(r['label'], p)
            if k in raw_by_key:
                raise ValueError(f'two files map to row {k}: {paths[k]} and {p}')
            raw_by_key[k], paths[k] = r, p

    scenes = next(iter(raw_by_key.values()))['scenes']
    for k, r in raw_by_key.items():
        if r['scenes'] != scenes:
            raise ValueError(f'scene order differs in {k}; comparisons must be paired')

    per = measure_scene_gaps(scenes)
    gaps = np.array([per[s]['median_gap'] for s in scenes], dtype=float)
    lo, hi = terciles(gaps)[0], terciles(gaps)[2]

    print(f'MID test: {len(scenes)} scenes, per-scene illuminant gap '
          f'{gaps.min():.4f}-{gaps.max():.4f} (median {np.median(gaps):.4f})')
    print(f'  LOW  n={len(lo)}  {gaps[lo].min():.4f}-{gaps[lo].max():.4f}')
    print(f'  HIGH n={len(hi)}  {gaps[hi].min():.4f}-{gaps[hi].max():.4f}')

    by_key = {k: analyse(r, gaps) for k, r in raw_by_key.items()}

    print(f'\n{"row":<20}{"weights":<30}{"Cast_rel":>9}{"LOW":>7}{"HIGH":>7}'
          f'{"HIGH/LOW":>10}{"r":>7}{"C_mat":>7}{"Chr_fid":>9}')
    print('-' * 106)
    for k in sorted(by_key):
        a, (name, w) = by_key[k], ROWS[k]
        print(f'{name:<20}{w:<30}{a["Cast_rel_mean"]:9.3f}{a["tercile_means"][0]:7.3f}'
              f'{a["tercile_means"][2]:7.3f}{a["high_over_low"]:10.2f}{a["pearson_r"]:7.3f}'
              f'{a["C_mat_mean"]:7.3f}{a["Chroma_fid_mean"]:9.3f}')

    print('\nPhase A reference HIGH/LOW ratios (external methods, same protocol):')
    for k, v in REFERENCE.items():
        print(f'   {k:26s} {v:.2f}')

    print('\n' + '=' * 106)
    print('CONTRASTS')
    print('=' * 106)
    out = []
    for a_key, b_key, why in CONTRASTS:
        if a_key in by_key and b_key in by_key:
            out.append(contrast(a_key, b_key, why, by_key, raw_by_key, gaps))
        else:
            missing = [k for k in (a_key, b_key) if k not in by_key]
            print(f'\n({ROWS[b_key][0]} - {ROWS[a_key][0]}) SKIPPED: missing {missing}')

    payload = {'_script': 'tests/eval/study1_readout.py', 'scenes': scenes,
               'gaps': gaps.tolist(), 'rows': by_key, 'contrasts': out,
               'reference': REFERENCE}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(payload, f, indent=1)
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
