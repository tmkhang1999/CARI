"""Phase B readout: did concentrating the illuminant-colour signal move the colour axis?

WHAT THIS ANSWERS, AND WHY NOT THE OBVIOUS THING
------------------------------------------------
The obvious readout -- "did mean Cast_rel go down?" -- cannot answer the question and
can actively mislead:

  1. MID's whole illuminant-colour range is 0.0165-0.0860 (per-scene medians), so a
     scene-averaged Cast_rel mostly reflects direction/intensity change, which is the
     axis we already win. Phase A showed CARI moves the mean while leaving the
     colour-dependence untouched.
  2. A LOWER mean Cast_rel is achievable by desaturating. That is the exact failure
     the corrected metric exists to catch (CRefNet: good pooled invariance,
     Chroma_fid 0.484). So Cast_rel must never be read without Chroma_fid.

The diagnostic that does answer it is the HIGH/LOW tercile ratio: split scenes by
their own measured illuminant chromaticity gap and ask whether the method's chroma
drift TRACKS illuminant colour. Reference values from Phase A:

    Ours (full)        1.42     <- degrades as the light changes colour
    Ours (base CARI)   1.43     <- CARI moved this by nothing
    CD-IID             1.02     <- genuinely colour-invariant
    RGB-X              0.90
    Marigold-App       1.03
    CRefNet            1.42     (grayscale shading -- structurally cannot)
    Ordinal Shading    1.56     (grayscale shading)

SUCCESS = the treatment arm's ratio falls toward 1.0 while Chroma_fid stays near 1.
A lower mean Cast_rel with an unchanged ratio is NOT success; it is most likely
desaturation or a uniform shift, and should be reported as a null.

Usage:
    # after both arms finish
    python tests/eval/eval_mid_constancy.py --ckpts checkpoints/v17_50/checkpoint_iter_68000.pth \
        --mid-root ../datasets/MIDIntrinsics --split test --save-json documents/evals/phaseB_v17_50.json
    python tests/eval/eval_mid_constancy.py --ckpts checkpoints/v17_51/checkpoint_iter_68000.pth \
        --mid-root ../datasets/MIDIntrinsics --split test --save-json documents/evals/phaseB_v17_51.json
    python tests/eval/phase_b_readout.py documents/evals/phaseB_v17_50.json documents/evals/phaseB_v17_51.json
"""

from __future__ import annotations

import os
import sys
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mid_colour_stratified import measure_scene_gaps  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Phase A reference ratios, for context in the printed table.
REFERENCE = {
    'Ours (full) [v17_34]': 1.42,
    'Ours (base CARI)': 1.43,
    'CD-IID': 1.02,
    'RGB-X': 0.90,
    'Marigold-App': 1.03,
    'CRefNet (gray shading)': 1.42,
    'Ordinal (gray shading)': 1.56,
}


def load_run(path: str):
    """Pull per-scene arrays out of an eval_mid_constancy --save-json dump."""
    with open(path) as f:
        d = json.load(f)
    out = []
    for r in d.get('results', []):
        ps = r.get('per_scene')
        if not ps or 'scene' not in ps:
            continue
        out.append({
            'label': r.get('label', os.path.basename(path)),
            'scenes': list(ps['scene']),
            'Cast_rel': np.array(ps['Cast_rel'], dtype=float),
            'C_mat': np.array(ps['C_mat'], dtype=float),
            'Chroma_fid': np.array(ps['Chroma_fid'], dtype=float),
            'Chroma_err': np.array(ps['Chroma_err'], dtype=float),
        })
    if not out:
        raise ValueError(f'no per-scene results in {path}')
    return out


def terciles(gaps: np.ndarray):
    order = np.argsort(gaps)
    n = len(gaps)
    return order[:n // 3], order[n // 3:2 * n // 3], order[2 * n // 3:]


def analyse(run, gaps):
    lo, mid, hi = terciles(gaps)
    cr = run['Cast_rel']
    return {
        'label': run['label'],
        'Cast_rel_mean': float(cr.mean()),
        'tercile_means': [float(cr[i].mean()) for i in (lo, mid, hi)],
        'high_over_low': float(cr[hi].mean() / cr[lo].mean()) if cr[lo].mean() else float('nan'),
        'pearson_r': float(np.corrcoef(cr, gaps)[0, 1]),
        'C_mat_mean': float(run['C_mat'].mean()),
        'Chroma_fid_mean': float(run['Chroma_fid'].mean()),
        'Chroma_err_mean': float(run['Chroma_err'].mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+', help='eval_mid_constancy --save-json outputs')
    ap.add_argument('--out', default=os.path.join(ROOT, 'documents', 'evals',
                                                  'phase_b_readout.json'))
    args = ap.parse_args()

    runs = []
    for p in args.runs:
        runs.extend(load_run(p))

    scenes = runs[0]['scenes']
    for r in runs:
        if r['scenes'] != scenes:
            raise ValueError(f"scene order differs in {r['label']}; comparisons must be paired")

    per = measure_scene_gaps(scenes)
    gaps = np.array([per[s]['median_gap'] for s in scenes], dtype=float)
    lo, mid, hi = terciles(gaps)

    print(f'MID test: {len(scenes)} scenes, per-scene illuminant gap '
          f'{gaps.min():.4f}-{gaps.max():.4f} (median {np.median(gaps):.4f})')
    print(f'  LOW  n={len(lo)}  {gaps[lo].min():.4f}-{gaps[lo].max():.4f}')
    print(f'  HIGH n={len(hi)}  {gaps[hi].min():.4f}-{gaps[hi].max():.4f}')
    print()

    res = [analyse(r, gaps) for r in runs]

    print(f'{"run":26s} {"Cast_rel":>9s} {"LOW":>7s} {"HIGH":>7s} {"HIGH/LOW":>9s} '
          f'{"r":>7s} {"C_mat":>7s} {"Chr_fid":>8s}')
    print('-' * 86)
    for a in res:
        print(f'{a["label"][:26]:26s} {a["Cast_rel_mean"]:9.3f} {a["tercile_means"][0]:7.3f} '
              f'{a["tercile_means"][2]:7.3f} {a["high_over_low"]:9.2f} {a["pearson_r"]:7.3f} '
              f'{a["C_mat_mean"]:7.3f} {a["Chroma_fid_mean"]:8.3f}')

    print('\nPhase A reference HIGH/LOW ratios:')
    for k, v in REFERENCE.items():
        print(f'   {k:26s} {v:.2f}')

    if len(res) >= 2:
        ctrl, treat = res[0], res[1]
        d_ratio = treat['high_over_low'] - ctrl['high_over_low']
        d_cast = treat['Cast_rel_mean'] - ctrl['Cast_rel_mean']
        d_fid = treat['Chroma_fid_mean'] - ctrl['Chroma_fid_mean']
        print(f'\nTREATMENT - CONTROL')
        print(f'   HIGH/LOW ratio  {d_ratio:+.3f}   (want NEGATIVE: drift stops tracking colour)')
        print(f'   mean Cast_rel   {d_cast:+.4f}')
        print(f'   Chroma_fid      {d_fid:+.4f}   (want ~0: a drop means colour was drained)')

        # Paired bootstrap on the ratio -- the scenes are the same in both arms, so the
        # comparison is paired and marginal intervals would understate significance.
        rng = np.random.RandomState(0)
        cr_c, cr_t = runs[0]['Cast_rel'], runs[1]['Cast_rel']
        boot = []
        for _ in range(20000):
            idx = rng.randint(0, len(scenes), len(scenes))
            g = gaps[idx]
            o = np.argsort(g)
            l, h = o[:len(g) // 3], o[2 * len(g) // 3:]
            cl, ch = cr_c[idx][l].mean(), cr_c[idx][h].mean()
            tl, th = cr_t[idx][l].mean(), cr_t[idx][h].mean()
            if cl > 0 and tl > 0:
                boot.append((th / tl) - (ch / cl))
        boot = np.array(boot)
        lo_ci, hi_ci = np.percentile(boot, [2.5, 97.5])
        print(f'   paired bootstrap 95% CI on the ratio change: [{lo_ci:+.3f}, {hi_ci:+.3f}]')
        print(f'   -> {"SIGNIFICANT" if hi_ci < 0 else "not separable from zero"}')

        verdict = ('TREATMENT REDUCES COLOUR-DEPENDENCE'
                   if (d_ratio < 0 and hi_ci < 0 and d_fid > -0.05)
                   else 'NULL on the colour axis')
        print(f'\nVERDICT: {verdict}')
        if d_cast < 0 and d_ratio >= 0:
            print('   NOTE: mean Cast_rel fell while the ratio did not. That is NOT a')
            print('   colour-constancy gain -- check Chroma_fid for desaturation.')

    payload = {'_script': 'tests/eval/phase_b_readout.py', 'scenes': scenes,
               'gaps': gaps.tolist(), 'runs': res, 'reference': REFERENCE}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(payload, f, indent=1)
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
