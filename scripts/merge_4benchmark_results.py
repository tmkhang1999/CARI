#!/usr/bin/env python3
"""Merge two or more completed 4-benchmark result sets (MID/MAW/ARAP/IIW JSONs) into
one combined results directory + summary.md, without re-running any evaluator.

Designed to combine the outputs of eval_all_models_4benchmarks_full.py (our v17
checkpoints, e.g. Table A's v17_41-44) with eval_sota_4benchmarks_full.py (the four
external SOTA methods), which both write identically-shaped JSONs under
<out>/results/eval_{mid_constancy,maw,arap_constancy,iiw}_results.json:
  - MID:  {'split': ..., 'results': [ {label, ...}, ... ]}   -> lists are concatenated
  - MAW/ARAP/IIW: {label: {...}, ...}                        -> dicts are merged

A later source overwrites an earlier source's entry for the same label (last-wins),
so duplicate labels across sources are a silent overwrite, not an error — pass
sources in the order you want to win.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = "/home/khang/miniconda3/envs/IR/bin/python"

_FILES = {
    'mid': 'eval_mid_constancy_results.json',
    'maw': 'eval_maw_results.json',
    'arap': 'eval_arap_constancy_results.json',
    'iiw': 'eval_iiw_results.json',
}


def resolve_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + '\n')


def merge(sources: list[Path], out_results: Path) -> dict[str, Path]:
    merged_mid = {'split': None, 'results': []}
    merged_maw: dict = {}
    merged_arap: dict = {}
    merged_iiw: dict = {}

    seen_mid_labels: dict[str, int] = {}  # label -> index in merged_mid['results']

    for src in sources:
        results_dir = src / 'results'
        mid = load(results_dir / _FILES['mid'])
        if merged_mid['split'] is None:
            merged_mid['split'] = mid.get('split')
        elif mid.get('split') and mid.get('split') != merged_mid['split']:
            print(f"[warn] MID split mismatch: {merged_mid['split']!r} vs "
                  f"{mid.get('split')!r} from {src}; keeping the first")
        for row in mid.get('results', []):
            label = row.get('label')
            if label in seen_mid_labels:
                print(f"[warn] duplicate MID label {label!r} in {src}; overwriting")
                merged_mid['results'][seen_mid_labels[label]] = row
            else:
                seen_mid_labels[label] = len(merged_mid['results'])
                merged_mid['results'].append(row)

        for dst, name in ((merged_maw, 'maw'), (merged_arap, 'arap'), (merged_iiw, 'iiw')):
            data = load(results_dir / _FILES[name])
            overlap = set(dst) & set(data)
            if overlap:
                print(f"[warn] duplicate {name} label(s) {sorted(overlap)} in {src}; overwriting")
            dst.update(data)

    write_json(out_results / _FILES['mid'], merged_mid)
    write_json(out_results / _FILES['maw'], merged_maw)
    write_json(out_results / _FILES['arap'], merged_arap)
    write_json(out_results / _FILES['iiw'], merged_iiw)
    return {k: out_results / v for k, v in _FILES.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', action='append', required=True,
                     help='Directory containing a results/ subdir (e.g. '
                          'tests/visualizations/all_models_4bench_full). Repeatable; '
                          'later sources overwrite earlier ones on label collision.')
    ap.add_argument('--out', required=True, help='Output directory for the merged results/ + summary.md.')
    args = ap.parse_args()

    sources = [resolve_path(s) for s in args.source]
    missing = [s for s in sources if not (s / 'results').is_dir()]
    if missing:
        raise SystemExit('Missing results/ dir under: ' + ', '.join(str(m) for m in missing))

    out = resolve_path(args.out)
    results = out / 'results'
    logs = out / 'logs'
    results.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    print('Merging (later wins on label collision):')
    for s in sources:
        print(f'  {s}')
    paths = merge(sources, results)

    for name, p in paths.items():
        data = load(p)
        n = len(data.get('results', [])) if name == 'mid' else len(data)
        print(f'  {name}: {n} entries -> {p}')

    summary_out = out / 'summary.md'
    summary_cmd = [
        PY, str(ROOT / 'tests/eval/summarize_row4.py'),
        '--maw', str(paths['maw']), '--mid', str(paths['mid']),
        '--arap', str(paths['arap']), '--iiw', str(paths['iiw']),
        '--out', str(summary_out),
    ]
    print('+ ' + ' '.join(summary_cmd))
    log_path = logs / 'summary.log'
    with log_path.open('w') as logf:
        proc = subprocess.run(summary_cmd, cwd=ROOT, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
        logf.write(proc.stdout)
        print(proc.stdout)
        if proc.returncode != 0:
            print(f'[warn] summary generation failed (merged JSONs are still complete); see {log_path}')
        else:
            print(f'Wrote summary: {summary_out}')


if __name__ == '__main__':
    main()
