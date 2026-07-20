#!/usr/bin/env python3
"""Measure the two shading-domain statistics that explain front3d's Table-B behaviour,
with an IDENTICAL scale-invariant estimator on every dataset:

  1. illum-chroma gap : max pairwise rg-chromaticity distance between the illuminant
                        estimates (channel-mean of I / channel-mean of A over valid
                        pixels) of a group's lighting variants. The COLOR axis.
  2. hard-edge frac   : fraction of valid pixels with |grad log S| > 0.3 at 512px,
                        S = lum(I)/lum(A). The SHADOW-STRUCTURE axis.

Measured 2026-07-13 (n=37 ARAP groups / 40 front3d views):
  color axis   front3d 0.151 >= ARAP indoor 0.100  -> covered (Cast gains transfer)
  shadow axis  front3d 0.022 vs ARAP indoor 0.115 (5.3x) / outdoor 0.186 (8.5x) -> the gap

--apply-sigma runs the S1 smoke test for the shadow-augmentation plan
(documents/design/FRONT3D_SHADOW_AUG_PLAN.md): it applies sample_relight_field to each
front3d variant before computing the stats. PASS = front3d hard-edge frac (median)
enters ARAP indoor's IQR [0.07, 0.18].

CAUTION: ARAP albedo .hdr files are stored UP-TO-SCALE (P99 ~ 0.005 for ~32 of the
scenes). All validity thresholds here are RELATIVE for exactly that reason; both
statistics are scale-invariant (rg normalises; grad log S shifts by a constant).

Usage:
  python scripts/measure_shading_domain_stats.py                      # both datasets
  python scripts/measure_shading_domain_stats.py --apply-sigma        # S1 smoke test
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import re
from pathlib import Path

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import cv2  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load_linear(p: Path, target=None):
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = img[:, :, :3][:, :, ::-1].astype(np.float32)  # BGR -> RGB
    if p.suffix.lower() in ('.jpg', '.jpeg', '.png'):
        img = (img / 255.0) ** 2.2
    if target is not None:
        img = cv2.resize(img, (target[1], target[0]), interpolation=cv2.INTER_AREA)
    else:
        H, W = img.shape[:2]
        s = 512.0 / max(H, W)
        img = cv2.resize(img, (int(round(W * s)), int(round(H * s))), interpolation=cv2.INTER_AREA)
    return np.clip(img, 0, None)


def lum(x):
    return 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]


def rg(v):
    s = float(v.sum()) + 1e-8
    return (v[0] / s, v[1] / s)


def analyze(alb, variants, edge_thresh=0.3):
    """Returns (chroma gap, mean hard-edge frac, mean P95 |grad log S|) or a reject reason str."""
    al = lum(alb)
    p99 = np.percentile(al[np.isfinite(al)], 99)
    valid = np.isfinite(al) & (al > 0.02 * p99)  # RELATIVE: ARAP albedos are up-to-scale
    if valid.mean() < 0.2:
        return 'low_albedo_coverage'
    chromas, edge_fracs, p95s = [], [], []
    for I in variants:
        Il = lum(I)
        ip99 = np.percentile(Il[np.isfinite(Il)], 99)
        ok = valid & np.isfinite(Il) & (Il > 1e-4 * max(ip99, 1e-8))
        if ok.mean() < 0.15:
            continue
        i_mean = I[ok].mean(axis=0)
        a_mean = alb[ok].mean(axis=0)
        chromas.append(rg(i_mean / np.maximum(a_mean, 1e-8)))
        S = Il / np.maximum(al, 1e-8)
        s = np.log(np.clip(S, 1e-8, None))
        gy, gx = np.gradient(s)
        gv = np.hypot(gx, gy)[ok]
        edge_fracs.append(float((gv > edge_thresh).mean()))
        p95s.append(float(np.percentile(gv, 95)))
    if len(chromas) < 2:
        return 'lt2_valid_variants'
    gap = max(math.dist(a, b) for a, b in itertools.combinations(chromas, 2))
    return gap, float(np.mean(edge_fracs)), float(np.mean(p95s))


def collect_arap(arap_root: Path, types_json: Path):
    types = json.load(open(types_json))
    name2dom = {}
    for cat in types.get('categories', []):
        dom = cat.get('domain', '').lower()
        for smp in cat.get('samples', []):
            name2dom[smp.get('name', '')] = dom if dom in ('indoor', 'outdoor') else 'outdoor'

    stems: dict[str, list[Path]] = {}
    for p in arap_root.iterdir():
        if '_albedo' in p.stem or p.suffix.lower() not in ('.hdr', '.jpg', '.jpeg', '.png'):
            continue
        stems.setdefault(re.sub(r'_light\d+$', '', p.stem), []).append(p)

    out = {'indoor': [], 'outdoor': []}
    for base, files in sorted(stems.items()):
        if not any(re.search(r'_light\d+$', f.stem) for f in files):
            continue
        albs = list(arap_root.glob(f'{base}_albedo.*'))
        if not albs:
            continue
        alb = load_linear(albs[0])
        if alb is None:
            continue
        variants = [v for v in (load_linear(f, target=alb.shape[:2])
                                for f in sorted(files, key=lambda q: q.stem)) if v is not None]
        r = analyze(alb, variants)
        if not isinstance(r, str):
            out[name2dom.get(base, 'outdoor')].append((base, *r))
    return out


def collect_front3d(f3root: Path, n_views: int, apply_sigma: bool, seed: int = 0):
    sigma_fn = None
    if apply_sigma:
        import sys
        import torch
        sys.path.insert(0, str(ROOT / 'src'))
        from data.shadow_aug import sample_relight_field

        def sigma_fn(img):
            t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
            M, c = sample_relight_field(1, t.shape[2], t.shape[3], t.device)
            return (t * M * c).squeeze(0).numpy().transpose(1, 2, 0)

    views = sorted(f3root.rglob('meta.json'))
    random.Random(seed).shuffle(views)
    rows = []
    for m in views:
        if len(rows) >= n_views:
            break
        d = m.parent
        alb = load_linear(d / 'albedo.exr')
        if alb is None:
            continue
        variants = [v for v in (load_linear(p) for p in sorted(d.glob('rgb_L*.exr')))
                    if v is not None and v.shape == alb.shape]
        if sigma_fn is not None:
            variants = [sigma_fn(v) for v in variants]
        r = analyze(alb, variants)
        if not isinstance(r, str):
            rows.append((str(d.relative_to(f3root)), *r))
    return rows


def summ(name, rows):
    if not rows:
        print(f'{name}: EMPTY')
        return
    gaps = np.array([r[1] for r in rows])
    ef = np.array([r[2] for r in rows])
    p95 = np.array([r[3] for r in rows])
    print(f'{name} (n={len(rows)}):')
    print(f'  illum-chroma gap:  median={np.median(gaps):.4f}  IQR=[{np.percentile(gaps, 25):.4f},{np.percentile(gaps, 75):.4f}]')
    print(f'  hard-edge frac:    median={np.median(ef):.4f}  IQR=[{np.percentile(ef, 25):.4f},{np.percentile(ef, 75):.4f}]')
    print(f'  P95 |grad log S|:  median={np.median(p95):.4f}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arap-root', default=str(ROOT / 'tests/testing_data/ARAP_dataset'))
    ap.add_argument('--arap-types', default=str(ROOT / 'tests/testing_data/ARAP_types.json'))
    ap.add_argument('--front3d-root', default=os.path.expanduser('~/datasets/front3d_iid'))
    ap.add_argument('--n-views', type=int, default=40)
    ap.add_argument('--apply-sigma', action='store_true',
                    help='S1 smoke test: apply sample_relight_field to front3d variants first')
    ap.add_argument('--skip-arap', action='store_true')
    args = ap.parse_args()

    if not args.skip_arap:
        arap = collect_arap(Path(args.arap_root), Path(args.arap_types))
        summ('ARAP indoor', arap['indoor'])
        summ('ARAP outdoor', arap['outdoor'])
    f3d = collect_front3d(Path(args.front3d_root), args.n_views, args.apply_sigma)
    summ('front3d' + (' + sigma' if args.apply_sigma else ''), f3d)
    if args.apply_sigma and f3d:
        med = float(np.median([r[2] for r in f3d]))
        ok = 0.07 <= med
        print(f"\nS1 smoke test: front3d+sigma hard-edge median = {med:.4f} "
              f"(target: enter ARAP indoor IQR [0.07, 0.18]) -> {'PASS' if ok else 'FAIL — raise sharpness/strength in sample_relight_field kwargs'}")


if __name__ == '__main__':
    main()
