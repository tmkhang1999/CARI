#!/usr/bin/env python3
"""Paired bootstrap for the MID comparisons in Chapter 5.

Chapter 5 asserts two significance results about the external baselines:

  * "Marigold-App and CRefNet both attain Cast_rel = 0.355 against our 0.421, and
    the paired difference is significant in each case"
  * "on Chroma_err ... every external method is significantly worse"

Neither interval was reported anywhere: the first forward-references
\\secref{sec:ablation}, but that section reports only the Row 1-2 and Row 3-4
contrasts of our own ablation. This script computes the missing intervals and
emits them both as a readable table and as the LaTeX body of tab:mid_paired.

Why paired: every method is scored on the same 30 held-out scenes, so the
per-scene difference cancels scene difficulty. Overlap of the marginal
per-method intervals in tab:mid is not a test of the difference -- with
between-scene spread this large it would hide real effects.

Input is documents/thesis/data/mid_per_scene.json, a tracked extract of the
evaluator's per-scene dumps whose scene means reproduce tab:mid exactly.

Usage:
  python tests/eval/paired_bootstrap_mid.py            # table to stdout
  python tests/eval/paired_bootstrap_mid.py --latex    # LaTeX rows
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PER_SCENE = ROOT / 'documents/thesis/data/mid_per_scene.json'

OURS = 'Ours (full)'
BASELINES = ['CD-IID', 'RGB-X', 'CRefNet', 'Marigold-App',
             'Marigold-Light', 'Ordinal Shading']
# All three are lower-is-better, so a negative difference means our model wins.
METRICS = [('C_mat', r'$C_{\text{mat}}$'),
           ('Chroma_err', r'$\text{Chroma}_{\text{err}}$'),
           ('Cast_rel', r'$\text{Cast}_{\text{rel}}$')]

# LaTeX names, so the table matches the citations used elsewhere in Chapter 5.
TEX_NAME = {
    'CD-IID': r'CD-IID \mycite{careaga24cdiid}',
    'RGB-X': r'RGB$\rightarrow$X \mycite{zeng24rgbx}',
    'CRefNet': r'CRefNet \mycite{luo23crefnet}',
    'Marigold-App': r'Marigold-App \mycite{marigoldiid24}',
    'Marigold-Light': r'Marigold-Light \mycite{marigoldiid24}',
    'Ordinal Shading': r'Ordinal Shading \mycite{careaga23ordinal}',
}


def paired_ci(a, b, idx):
    """Mean paired difference a-b with a percentile-bootstrap 95% interval."""
    d = np.asarray(a, float) - np.asarray(b, float)
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def compute(n_boot=20000, seed=0):
    payload = json.loads(PER_SCENE.read_text())
    methods = payload['methods']
    n = len(payload['scenes'])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    out = {}
    for key, _ in METRICS:
        out[key] = {b: paired_ci(methods[OURS][key], methods[b][key], idx)
                    for b in BASELINES}
    return out, n, n_boot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--latex', action='store_true', help='emit LaTeX table rows')
    ap.add_argument('--n-boot', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    res, n, n_boot = compute(args.n_boot, args.seed)

    if not args.latex:
        print(f'Paired bootstrap over the same {n} MID scenes, {n_boot:,} resamples')
        print(f'delta = {OURS} - baseline; negative = our model better '
              f'(all metrics lower-is-better)\n')
        for key, _ in METRICS:
            print(f'{key}:')
            for b in BASELINES:
                m, lo, hi = res[key][b]
                sig = 'significant' if (lo > 0) == (hi > 0) else 'n.s.'
                print(f'   vs {b:<17s} {m:+.4f}  [{lo:+.4f}, {hi:+.4f}]  {sig}')
            print()
        return

    for b in BASELINES:
        cells = []
        for key, _ in METRICS:
            m, lo, hi = res[key][b]
            sig = (lo > 0) == (hi > 0)
            body = f'{m:+.3f}\\,{{\\scriptsize[{lo:+.3f},{hi:+.3f}]}}'
            cells.append(f'\\textbf{{{body}}}' if sig else body)
        print(f'    {TEX_NAME[b]:<44s} & ' + ' & '.join(cells) + r' \\')


if __name__ == '__main__':
    main()
