#!/usr/bin/env python3
"""Merge the per-model corrected-cast MID JSONs and print the Table-A before/after.

The published Cast_RMS pools chroma ratios across ALL materials, so it sums the
across-illuminant drift we claim to measure with the between-material chroma spread of
the scene. That second term shrinks when a model collapses material colour, so the
pooled metric pays models for washing out chroma. This prints the pooled column beside
the decomposed one so the reversal is auditable.

Usage: python scripts/summarize_corrected_cast.py <dir-with-mid_*.json>
"""
import json
import sys
from pathlib import Path

ROWS = {  # label -> (row, CARI, colour skip)
    'v17_41': ('1', 'off', 'off'),
    'v17_42': ('2', 'ON ', 'off'),
    'v17_43': ('3', 'off', 'ON '),
    'v17_44': ('4', 'ON ', 'ON '),
}
# MAW dE, measured in the completed 4-benchmark run (all_models_4bench_full_A)
MAW = {'v17_41': 4.434, 'v17_42': 4.631, 'v17_43': 4.115, 'v17_44': 4.155}


def main():
    d = Path(sys.argv[1] if len(sys.argv) > 1
             else 'tests/visualizations/tableA_corrected_cast')
    recs = {}
    for p in sorted(d.glob('mid_*.json')):
        blob = json.loads(p.read_text())
        for r in blob.get('results', []):
            recs[r['label']] = r
    if not recs:
        sys.exit(f'no mid_*.json found in {d}')

    def g(r, k):
        v = r.get(k)
        return float('nan') if v is None else float(v)

    print()
    print('=' * 108)
    print('TABLE A — pooled (published) vs corrected cast     [GT anchor: Chroma_fid 1.0 = faithful]')
    print('=' * 108)
    print(f'{"Row":>3} {"CARI":>4} {"skip":>4} | {"C_mat":>7} {"POOLED":>7} | '
          f'{"within":>7} {"between":>7} {"Cast_rel":>8} {"Chroma_fid":>10} {"Sat_ratio":>9} | {"MAW dE":>7}')
    print('-' * 108)
    for lab in ('v17_41', 'v17_42', 'v17_43', 'v17_44'):
        if lab not in recs:
            continue
        r = recs[lab]
        row, cari, skip = ROWS[lab]
        print(f'{row:>3} {cari:>4} {skip:>4} | '
              f'{g(r, "C_mat"):7.3f} {g(r, "R_cast_rms"):7.3f} | '
              f'{g(r, "Cast_within"):7.3f} {g(r, "Cast_between"):7.3f} '
              f'{g(r, "Cast_rel"):8.3f} {g(r, "Chroma_fid"):10.3f} {g(r, "Sat_ratio"):9.3f} | '
              f'{MAW.get(lab, float("nan")):7.3f}')
    print('-' * 108)

    gtb = next((g(r, 'GT_between') for r in recs.values()
                if g(r, 'GT_between') == g(r, 'GT_between')), float('nan'))
    print(f'GT_between = {gtb:.3f}   (the between-material chroma spread a faithful model should reproduce)')
    print()
    print('POOLED     = published Cast_RMS = within + between, conflated  -> rewards chroma collapse')
    print('Cast_rel   = within / between = scale-fair invariance          [LOWER = better]')
    print('Chroma_fid = between / GT_between = material chroma retained   [1.0 = faithful, <1 = collapsed]')
    print()

    best = lambda k: min(recs, key=lambda l: g(recs[l], k))  # noqa: E731
    # Chroma_fid is a fidelity RATIO, so the target is 1.0 — not the maximum.
    # Over-saturating (fid > 1) is as much an error as collapsing (fid < 1).
    faithful = min(recs, key=lambda l: abs(g(recs[l], 'Chroma_fid') - 1.0))
    print(f'best POOLED (published verdict) : {best("R_cast_rms")}')
    print(f'best Cast_rel (corrected)       : {best("Cast_rel")}')
    print(f'best C_mat                      : {best("C_mat")}')
    print(f'most faithful |Chroma_fid - 1|  : {faithful}')
    print(f'best MAW dE (external)          : {min(MAW, key=MAW.get)}')


if __name__ == '__main__':
    main()
