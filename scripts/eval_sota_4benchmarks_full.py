#!/usr/bin/env python3
"""Full four-model SOTA evaluation on MID, MAW, ARAP, and IIW.

Runs the four external methods reported in the thesis — Marigold-App,
Marigold-Light, CRefNet, and Ordinal Shading — locally under our exact pipeline
(same output-space normalisation, splits, and per-benchmark inference
resolution), producing the same JSON/summary artifacts as
``eval_all_models_4benchmarks_full.py`` does for our v17 checkpoints.

Each method is loaded through the shared adapters that eval_maw/eval_arap
already use (Marigold pipeline, ``crefnet_adapter``, ``ordinal_adapter``);
``eval_mid_constancy.py`` and ``eval_iiw.py`` were extended to call those same
adapters so all four methods run on all four benchmarks. The benchmark jobs are
independent per method and assigned round-robin to the requested GPUs.

Model spec format is ``LABEL=PATH=KIND`` where KIND is one of
``marigold-appearance | marigold-lighting | crefnet | crefnet-e | ordinal |
ordinal-rendered-only``. For Ordinal Shading the weights are fetched by the
adapter via torch.hub, so PATH is an ignored placeholder.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = Path(os.environ.get('IR_PYTHON', '/home/khang/miniconda3/envs/IR/bin/python'))

_MARIGOLD_VERSIONS = {'marigold-appearance', 'marigold-lighting'}
_CREFNET_VERSIONS = {'crefnet', 'crefnet-e'}
_ORDINAL_VERSIONS = {'ordinal', 'ordinal-rendered-only'}
_KNOWN_KINDS = _MARIGOLD_VERSIONS | _CREFNET_VERSIONS | _ORDINAL_VERSIONS

# Ordinal fetches weights via torch.hub; its --checkpoint arg is an ignored placeholder.
_ORDINAL_PLACEHOLDER = 'ordinal-hub-weights'

DEFAULT_CKPTS = [
    'Marigold-App=checkpoints/marigold-iid-appearance-v1-1=marigold-appearance',
    'Marigold-Light=checkpoints/marigold-iid-lighting-v1-1=marigold-lighting',
    'CRefNet=checkpoints/CRefNet/final_real.pt=crefnet',
    f'Ordinal={_ORDINAL_PLACEHOLDER}=ordinal',
]
DEFAULT_MID_ROOT = Path('/home/khang/datasets/MIDIntrinsics')
DEFAULT_MAW_ROOT = ROOT / 'tests/testing_data/MAW'
DEFAULT_ARAP_ROOT = ROOT / 'tests/testing_data/ARAP_dataset'
DEFAULT_IIW_ROOT = ROOT / 'tests/testing_data/iiw-dataset/data'


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def parse_spec(spec: str) -> dict:
    parts = spec.split('=')
    if len(parts) < 3:
        raise ValueError(f'SOTA checkpoint must be LABEL=PATH=KIND, got {spec!r}')
    label, raw_path, kind = parts[0], parts[1], parts[2]
    kind = kind.lower()
    if kind not in _KNOWN_KINDS:
        raise ValueError(f'Unknown model KIND {kind!r}; expected one of {sorted(_KNOWN_KINDS)}')
    is_ordinal = kind in _ORDINAL_VERSIONS
    # Ordinal's path is a placeholder; keep it as-is (not resolved to a file).
    path = Path(raw_path) if is_ordinal else resolve_path(raw_path)
    return {'label': label, 'path': path, 'kind': kind, 'is_ordinal': is_ordinal}


def safe_name(text: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', text)


def run_logged(cmd: list[str | Path], log_path: Path, gpu: int) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault('HF_HUB_OFFLINE', '1')
    env.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
    env.setdefault('PYTHONUNBUFFERED', '1')
    print(f'[gpu {gpu}] + ' + ' '.join(map(str, cmd)), flush=True)
    with log_path.open('w') as log:
        proc = subprocess.Popen(
            [str(x) for x in cmd], cwd=ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(f'[{log_path.parent.name}] {line}', end='', flush=True)
            log.write(line)
        if proc.wait() != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)


def run_logged_with_retry(cmd: list[str | Path], log_path: Path, gpu: int, args: argparse.Namespace) -> None:
    """Retry a failed child process while preserving completed JSON stages."""
    for attempt in range(args.max_retries + 1):
        try:
            run_logged(cmd, log_path, gpu)
            return
        except subprocess.CalledProcessError:
            if attempt >= args.max_retries:
                raise
            delay = args.retry_delay_sec
            print(f'[gpu {gpu}] stage failed; retrying in {delay}s '
                  f'({attempt + 1}/{args.max_retries})', flush=True)
            time.sleep(delay)


def read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def stage_has_label(path: Path, label: str, *, mid: bool = False) -> bool:
    data = read_json(path)
    if data is None:
        return False
    if mid:
        return any(row.get('label') == label for row in data.get('results', []) if isinstance(row, dict))
    return label in data


def read_whdr(log_path: Path) -> float:
    matches = re.findall(r'Dataset WHDR:\s*([0-9.]+)', log_path.read_text(errors='ignore'))
    if not matches:
        raise RuntimeError(f'WHDR not found in {log_path}')
    return float(matches[-1])


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + '\n')


def run_model(spec: dict, gpu: int, args: argparse.Namespace, work_root: Path) -> dict:
    label, ckpt, kind = spec['label'], spec['path'], spec['kind']
    model_root = work_root / safe_name(label)
    result_dir = model_root / 'results'
    logs_dir = model_root / 'logs'
    result_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # MID — LABEL=PATH=KIND; MID's constancy metrics are GT-free, so external models score fine.
    mid_json = result_dir / 'mid.json'
    if args.auto_resume and stage_has_label(mid_json, label, mid=True):
        print(f'[{label}] resume: MID already complete', flush=True)
    else:
        run_logged_with_retry([
            PY, '-u', 'tests/eval/eval_mid_constancy.py',
            '--ckpts', f'{label}={ckpt}={kind}', '--mid-root', args.mid_root, '--split', 'test',
            '--device', f'cuda:{gpu}', '--infer-max-size', str(args.infer_max_size),
            '--save-json', mid_json,
        ], logs_dir / 'mid.log', gpu, args)

    # MAW — LABEL=PATH=KIND (TYPE dispatch already native to eval_maw).
    maw_json = result_dir / 'maw.json'
    if args.auto_resume and stage_has_label(maw_json, label):
        print(f'[{label}] resume: MAW already complete', flush=True)
    else:
        maw_cmd = [
            PY, '-u', 'tests/eval/eval_maw.py',
            '--ckpts', f'{label}={ckpt}={kind}', '--dataset', 'maw1', '--maw-root', args.maw_root,
            '--device', 'cuda', '--cuda-index', str(gpu),
            '--infer-max-size', str(args.maw_infer_size), '--save-json', maw_json,
        ]
        if args.maw_amp:
            maw_cmd.append('--amp')
        run_logged_with_retry(maw_cmd, logs_dir / 'maw.log', gpu, args)

    # ARAP — raw-colored = thesis constancy axis (C_arap/Cast_RMS); white-balanced = standard
    # SOTA protocol (LMSE/si-RMSE/SSIM comparable to Ordinal/CD-IID). External kind passed via
    # --model_version. '__wb' marks the white-balanced (standard) rows.
    arap_json = result_dir / 'arap.json'
    arap_runs = [
        ('all', False, label),
        ('indoor', False, f'{label}__indoor'),
    ]
    if args.arap_whitebalance:
        arap_runs.append(('all', True, f'{label}__wb'))
    for split, white_balance, result_label in arap_runs:
        if args.auto_resume and stage_has_label(arap_json, result_label):
            print(f'[{label}] resume: ARAP {result_label} already complete', flush=True)
            continue
        arap_cmd = [
            PY, '-u', 'tests/eval/eval_arap.py', '--constancy',
            '--checkpoint', ckpt, '--model_version', kind,
            '--dataset_dir', args.arap_root, '--device', 'cuda', '--cuda_index', str(gpu),
            '--max_size', str(args.arap_infer_size), '--scene_filter', split,
            '--save_json', arap_json, '--label', result_label,
        ]
        if white_balance:
            arap_cmd.append('--white_balance')
        run_logged_with_retry(arap_cmd, logs_dir / f'arap_{result_label}.log', gpu, args)

    # IIW — external kind via --model_version; eval_iiw bypasses _build_model for external models.
    iiw_json = result_dir / 'iiw.json'
    if args.auto_resume and stage_has_label(iiw_json, label):
        print(f'[{label}] resume: IIW already complete', flush=True)
    else:
        iiw_log = logs_dir / 'iiw.log'
        run_logged_with_retry([
            PY, '-u', 'tests/eval/eval_iiw.py', '--checkpoint', ckpt, '--model_version', kind,
            '--dataset_dir', args.iiw_root, '--device', 'cuda', '--cuda_index', str(gpu),
            '--max_size', str(args.infer_max_size),
        ], iiw_log, gpu, args)
        write_json(iiw_json, {label: {'label': label, 'WHDR': read_whdr(iiw_log)}})

    return {'label': label, 'root': model_root, 'gpu': gpu}


def merge_jsons(model_runs: list[dict], out_results: Path) -> None:
    merged_maw, merged_arap, merged_iiw = {}, {}, {}
    merged_mid = {'split': 'test', 'results': []}
    for item in model_runs:
        results = Path(item['root']) / 'results'
        for dst, name in ((merged_maw, 'maw.json'), (merged_arap, 'arap.json'), (merged_iiw, 'iiw.json')):
            path = results / name
            if not path.exists():
                raise FileNotFoundError(path)
            dst.update(json.loads(path.read_text()))
        mid = json.loads((results / 'mid.json').read_text())
        merged_mid['results'].extend(mid.get('results', []))
    write_json(out_results / 'eval_maw_results.json', merged_maw)
    write_json(out_results / 'eval_arap_constancy_results.json', merged_arap)
    write_json(out_results / 'eval_iiw_results.json', merged_iiw)
    write_json(out_results / 'eval_mid_constancy_results.json', merged_mid)


def protocol_manifest(args: argparse.Namespace, specs: list[dict]) -> dict:
    with (Path(args.maw_root) / 'labels' / 'meta.csv').open() as f:
        maw_rows = sum(1 for r in csv.reader(f, delimiter='\t') if len(r) >= 6)
    iiw_ids = sorted(int(p.stem) for p in Path(args.iiw_root).glob('*.png') if p.stem.isdigit())
    mid_scenes = [p for p in (Path(args.mid_root) / 'test').iterdir() if (p / 'albedo.exr').exists()]
    arap_splits = ['all', 'indoor']
    if args.arap_whitebalance:
        arap_splits.append('all__wb')
    return {
        'scope': 'full SOTA benchmark (Marigold-App/Light, CRefNet, Ordinal); no persisted predictions',
        'protocol_doc': 'documents/evals/BENCHMARK_PROTOCOL.md',
        'models': [{'label': s['label'], 'path': str(s['path']), 'kind': s['kind']} for s in specs],
        'infer_size': {'mid': args.infer_max_size, 'iiw': args.infer_max_size,
                       'arap': args.arap_infer_size, 'maw': args.maw_infer_size},
        'maw_amp': args.maw_amp,
        'mid': {'split': 'test', 'scenes': len(mid_scenes),
                'role': 'GT-free constancy diagnostic (training data for some methods; not a zero-shot SOTA row)'},
        'maw': {'images': maw_rows},
        'iiw': {'all_images': len(iiw_ids), 'test_every_5th_starting_at_0': len(iiw_ids[0::5])},
        'arap': {'splits': arap_splits,
                 'input': 'raw-colored (thesis) + __wb white-balanced (standard, comparable to published)',
                 'metric': 'LMSE + si-RMSE + SSIM (published-comparable); C_arap + Cast_RMS (thesis)'},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Full SOTA (Marigold-App/Light, CRefNet, Ordinal) evaluation on MID, MAW, ARAP, IIW.')
    parser.add_argument('--ckpt', action='append', metavar='LABEL=PATH=KIND',
                        help='Repeatable; defaults to the four thesis SOTA methods.')
    parser.add_argument('--cuda-indices', default='0', help='Comma-separated physical GPU indices; default is one GPU (0).')
    parser.add_argument('--out', default='tests/visualizations/sota_4bench_full')
    parser.add_argument('--mid-root', default=str(DEFAULT_MID_ROOT))
    parser.add_argument('--maw-root', default=str(DEFAULT_MAW_ROOT))
    parser.add_argument('--arap-root', default=str(DEFAULT_ARAP_ROOT))
    parser.add_argument('--iiw-root', default=str(DEFAULT_IIW_ROOT))
    parser.add_argument('--auto-resume', action='store_true',
                        help='Reuse completed per-model benchmark stages and retry failed child evaluators.')
    parser.add_argument('--max-retries', type=int, default=3,
                        help='Retries per failed child evaluator when --auto-resume is enabled.')
    parser.add_argument('--retry-delay-sec', type=int, default=60,
                        help='Delay between child-evaluator retries.')
    # Per-dataset inference resolution — see documents/evals/BENCHMARK_PROTOCOL.md.
    parser.add_argument('--infer-max-size', type=int, default=1280,
                        help='Long-side cap for MID and IIW (default 1280, the comparable protocol).')
    parser.add_argument('--arap-infer-size', type=int, default=1280,
                        help='Long-side cap for ARAP (default 1280; measured identical to 1500).')
    parser.add_argument('--maw-infer-size', type=int, default=512,
                        help='Long-side cap for MAW (default 512, the MAW authors\' eval protocol).')
    parser.add_argument('--maw-amp', dest='maw_amp', action='store_true', default=True,
                        help='Run MAW inference under fp16 autocast (default on; numerically <0.1%% drift).')
    parser.add_argument('--no-maw-amp', dest='maw_amp', action='store_false',
                        help='Disable MAW fp16 autocast.')
    parser.add_argument('--no-arap-whitebalance', dest='arap_whitebalance', action='store_false', default=True,
                        help='Skip the standard-protocol white-balanced ARAP run (keep only the raw-colored '
                             'thesis run). By default BOTH are produced.')
    args = parser.parse_args()

    args.mid_root = str(resolve_path(args.mid_root))
    args.maw_root = str(resolve_path(args.maw_root))
    args.arap_root = str(resolve_path(args.arap_root))
    args.iiw_root = str(resolve_path(args.iiw_root))
    gpu_ids = [int(x.strip()) for x in args.cuda_indices.split(',') if x.strip()]
    if not gpu_ids:
        raise SystemExit('--cuda-indices must contain at least one GPU index')
    if args.max_retries < 0 or args.retry_delay_sec < 0:
        raise SystemExit('--max-retries and --retry-delay-sec must be non-negative')
    if not args.auto_resume:
        args.max_retries = 0

    specs = [parse_spec(x) for x in (args.ckpt or DEFAULT_CKPTS)]
    # Ordinal's path is a torch.hub placeholder; only the file-backed methods must exist on disk.
    missing = [s for s in specs if not s['is_ordinal'] and not s['path'].exists()]
    if missing:
        raise SystemExit('Missing checkpoint(s): ' + ', '.join(f"{s['label']}={s['path']}" for s in missing))

    out = resolve_path(args.out)
    results = out / 'results'
    work = out / 'work'
    logs = out / 'logs'
    results.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    write_json(out / 'protocol.json', protocol_manifest(args, specs))

    print('Full SOTA benchmark: Marigold-App/Light, CRefNet, Ordinal Shading on MID, MAW, ARAP, IIW.')
    if args.auto_resume:
        print(f'Auto-resume enabled: completed JSON stages are reused; max retries={args.max_retries}.')
    for i, spec in enumerate(specs):
        print(f"  {spec['label']} [{spec['kind']}]: GPU {gpu_ids[i % len(gpu_ids)]}  {spec['path']}")

    runs = []
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        future_map = {
            executor.submit(run_model, spec, gpu_ids[i % len(gpu_ids)], args, work): spec
            for i, spec in enumerate(specs)
        }
        for future in concurrent.futures.as_completed(future_map):
            spec = future_map[future]
            try:
                runs.append(future.result())
            except Exception as exc:
                failures.append(f'{spec["label"]}: {exc}')
    if failures:
        raise SystemExit('Failed benchmark job(s):\n  ' + '\n  '.join(failures))

    merge_jsons(runs, results)
    # Summary is presentation-only; the merged JSONs above are the authoritative results.
    summary_cmd = [
        PY, 'tests/eval/summarize_row4.py', '--maw', results / 'eval_maw_results.json',
        '--mid', results / 'eval_mid_constancy_results.json',
        '--arap', results / 'eval_arap_constancy_results.json',
        '--iiw', results / 'eval_iiw_results.json', '--out', out / 'summary.md',
    ]
    try:
        run_logged(summary_cmd, logs / 'summary.log', gpu_ids[0])
    except subprocess.CalledProcessError:
        print('[warn] summary generation failed (JSON results are still complete); see logs/summary.log', flush=True)
    print(f'Completed full SOTA benchmark: {results}')


if __name__ == '__main__':
    main()
