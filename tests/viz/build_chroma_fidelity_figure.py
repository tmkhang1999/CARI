#!/usr/bin/env python3
"""Qualitative figure for the chroma-fidelity result (Ch5, tab:mid / tab:ablation).

Shows the claim the corrected metric makes numerically: the methods that score best on a
pooled/absolute chroma-cast metric do so by DESATURATING the albedo, not by being more
illuminant-invariant. One MID test scene under two illuminants, columns:

    Input | GT albedo | Ours (v17_34) | v17_42 (skip off) | CRefNet | Marigold-App

v17_42 and CRefNet should read as visibly washed out beside the GT; ours should not.
Each column is captioned with pseudo-GT Chroma_err and the aggregate chroma-spread
ratio. Chroma_err is the calibration guard; the ratio diagnoses collapse or expansion.

Usage:  python tests/viz/build_chroma_fidelity_figure.py --scene <name> --out <dir>
"""
import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = '/home/khang/IR-IID'
sys.path.insert(0, os.path.join(ROOT, 'tests/eval'))
os.chdir(os.path.join(ROOT, 'tests/eval'))

from eval_mid_constancy import AlbedoPredictor, _raw_frame, _tonemap_frame  # noqa: E402

MID = '/home/khang/datasets/MIDIntrinsics/test'

# label, path, version, Chroma_err, ratio of aggregate predicted/pseudo-GT spread
MODELS = [
    ('Ours (full model)',  f'{ROOT}/checkpoints/v17_34/checkpoint_iter_60000.pth', '17', 0.121, 0.941),
    ('Ours, colour path OFF',   f'{ROOT}/checkpoints/v17_42/checkpoint_iter_40000.pth', '17', 0.201, 0.484),
    ('CRefNet',                 f'{ROOT}/checkpoints/CRefNet/final_real.pt', 'crefnet', 0.201, 0.484),
    ('Marigold-App',            f'{ROOT}/checkpoints/marigold-iid-appearance-v1-1',
     'marigold-appearance', 0.195, 0.728),
]

FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONTB = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def _fnt(sz, bold=False):
    try:
        return ImageFont.truetype(FONTB if bold else FONT, sz)
    except Exception:
        return ImageFont.load_default()


def _norm_albedo(a, pct=99.5):
    """Scale-normalise for display only; albedo is defined up to a global scale."""
    s = np.percentile(a[a > 1e-6], pct) if (a > 1e-6).any() else 1.0
    return np.clip(a / (s + 1e-8), 0, 1)


def _srgb(x):
    return np.clip(x, 0, 1) ** (1 / 2.2)


def _panel(arr, w, h):
    im = Image.fromarray((_srgb(arr) * 255).astype(np.uint8))
    return im.resize((w, h), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', nargs=2,
                    default=['everett_dining1', 'everett_kitchen18'])
    ap.add_argument('--lights', type=int, nargs=2, default=[0, 18],
                    help='One illuminant index for each scene.')
    ap.add_argument('--panel-w', type=int, default=430)
    ap.add_argument('--infer', type=int, default=1280)
    ap.add_argument('--out', default=f'{ROOT}/tests/visualizations/chroma_fidelity')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cases = list(zip(args.scenes, args.lights))
    print('cases:', cases)
    albedos = []
    inputs = []
    for scene, light in cases:
        sp = os.path.join(MID, scene)
        alb = cv2.imread(os.path.join(sp, 'albedo.exr'),
                         cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        albedos.append(_tonemap_frame(alb[..., ::-1].copy().astype(np.float32)))
        inputs.append(_tonemap_frame(_raw_frame(sp, light)))
    H, W = albedos[0].shape[:2]
    ar = H / W
    PW = args.panel_w
    PH = int(PW * ar)

    rows = []          # one row per scene
    col_labels = ['Input', 'GT albedo'] + [m[0] for m in MODELS]

    preds = {}
    for label, path, ver, _err, _spread in MODELS:
        p = AlbedoPredictor(path, ver, 'cuda', infer_max_size=args.infer)
        preds[label] = [p.albedo(x) for x in inputs]
        del p
        import torch
        torch.cuda.empty_cache()
        print(f'  predicted: {label}')

    for r, _ in enumerate(args.lights):
        panels = [_panel(inputs[r], PW, PH), _panel(_norm_albedo(albedos[r]), PW, PH)]
        for label, _p, _v, _err, _spread in MODELS:
            panels.append(_panel(_norm_albedo(preds[label][r]), PW, PH))
        rows.append(panels)

    ncol = len(col_labels)
    gap, head, foot = 6, 62, 92
    Wtot = ncol * PW + (ncol - 1) * gap
    Htot = head + len(rows) * (PH + gap) + foot
    canvas = Image.new('RGB', (Wtot, Htot), (255, 255, 255))
    d = ImageDraw.Draw(canvas)

    for c, lab in enumerate(col_labels):
        x = c * (PW + gap)
        bold = lab.startswith('Ours (') or lab == 'GT albedo'
        d.text((x + PW // 2, head // 2), lab, fill=(20, 20, 20), anchor='mm',
               font=_fnt(28, bold=bold))

    for r, panels in enumerate(rows):
        y = head + r * (PH + gap)
        for c, pil in enumerate(panels):
            x = c * (PW + gap)
            canvas.paste(pil, (x, y))
            if col_labels[c] == 'GT albedo':
                d.rectangle([x, y, x + PW - 1, y + PH - 1], outline=(200, 30, 30), width=3)

    # Pseudo-GT calibration and aggregate spread caption strip
    yf = head + len(rows) * (PH + gap)
    d.text((PW + gap + PW // 2, yf + 14), 'pseudo-GT reference', fill=(200, 30, 30),
           anchor='mm', font=_fnt(23, bold=True))
    for c, (label, _p, _v, err, spread) in enumerate(MODELS):
        x = (c + 2) * (PW + gap) + PW // 2
        collapsed = spread < 0.9
        col = (200, 30, 30) if collapsed else (25, 110, 40)
        d.text((x, yf + 12), f'Chroma_err = {err:.3f}', fill=col, anchor='mm',
               font=_fnt(22, bold=True))
        d.text((x, yf + 39), f'GT spread retained = {spread:.3f}',
               fill=col, anchor='mm', font=_fnt(22, bold=collapsed))

    out = os.path.join(args.out, 'chroma_fidelity.jpg')
    canvas.save(out, quality=94)
    print(f'wrote {out}  ({canvas.size[0]}x{canvas.size[1]})')

    # copy into the thesis image tree
    dst = f'{ROOT}/documents/thesis/images/chroma_fidelity'
    os.makedirs(dst, exist_ok=True)
    canvas.save(os.path.join(dst, 'chroma_fidelity.jpg'), quality=94)
    print(f'wrote {dst}/chroma_fidelity.jpg')

    with open(os.path.join(args.out, 'manifest.txt'), 'w') as f:
        f.write(f'cases: {cases}\ninfer: {args.infer}\n')
        for label, path, ver, err, spread in MODELS:
            f.write(f'{label}\t{path}\t{ver}\tChroma_err={err}\tspread_ratio={spread}\n')


if __name__ == '__main__':
    main()
