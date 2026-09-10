#!/usr/bin/env python3
"""Ours vs. all four external baselines on IIW (Intrinsic Images in the Wild).
One real photograph per scene, no dense ground-truth albedo (IIW ships only
pairwise WHDR judgements), so no GT column -- just Input + method columns.
One figure per scene, reusing 2 of the 3 recurring IIW ids already used in
the deck (Slide 42): 100010, 100038.

Usage: python build_iiw_method_matrix.py <image_id>
"""
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = '/home/khang/IR-IID'
sys.path.insert(0, os.path.join(ROOT, 'tests/eval'))
os.chdir(os.path.join(ROOT, 'tests/eval'))
from eval_mid_constancy import AlbedoPredictor  # noqa: E402

IIW = f'{ROOT}/tests/testing_data/iiw-dataset/data'
OUT = f'{ROOT}/presentation/assets/generated/all_models_light_matrix'
os.makedirs(OUT, exist_ok=True)

CK = f'{ROOT}/checkpoints'
METHODS = [
    ('Ours',            f'{CK}/v17_29/checkpoint_iter_60000.pth', '17'),
    ('CRefNet',         f'{CK}/CRefNet/final_real.pt', 'crefnet'),
    ('Marigold-App',    f'{CK}/marigold-iid-appearance-v1-1', 'marigold-appearance'),
    ('Marigold-Light',  f'{CK}/marigold-iid-lighting-v1-1', 'marigold-lighting'),
    ('Ordinal Shading', '', 'ordinal'),
]

FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONTB = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def fnt(sz, bold=False):
    return ImageFont.truetype(FONTB if bold else FONT, sz)


def norm(a, pct=99.0):
    v = a[a > 1e-6]
    s = float(np.percentile(v, pct)) if v.size else 1.0
    return np.clip(a / (s + 1e-8), 0, 1)


def srgb(x):
    return np.clip(x, 0, 1) ** (1 / 2.2)


def load_srgb(path):
    b = cv2.imread(path, cv2.IMREAD_COLOR)
    return np.clip(b[..., ::-1].astype(np.float32) / 255.0, 0, 1)


def to_panel(arr_srgb01, w, h):
    return Image.fromarray((np.clip(arr_srgb01, 0, 1) * 255).astype(np.uint8)).resize((w, h), Image.LANCZOS)


def main(image_id):
    ip = f'{IIW}/{image_id}.png'
    inp_srgb = load_srgb(ip)
    inp_lin = inp_srgb ** 2.2

    PW = 480
    ph = int(PW * inp_srgb.shape[0] / inp_srgb.shape[1])

    preds = {}
    for label, ckpt, ver in METHODS:
        p = AlbedoPredictor(ckpt, ver, 'cuda', infer_max_size=1024)
        alb = p.albedo(inp_lin)
        del p
        torch.cuda.empty_cache()
        preds[label] = srgb(norm(alb))
        print(f'  [{image_id}] predicted: {label}')

    col_labels = ['Input'] + [m[0] for m in METHODS]
    panels = [inp_srgb] + [preds[m[0]] for m in METHODS]

    ncol = len(col_labels)
    gap, head = 8, 50
    W = ncol * PW + (ncol - 1) * gap
    H = head + ph
    canvas = Image.new('RGB', (W, H), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    for c, lab in enumerate(col_labels):
        x = c * (PW + gap)
        bold = lab == 'Ours'
        d.text((x + PW // 2, head // 2), lab, fill=(23, 32, 51) if not bold else (22, 119, 255),
               anchor='mm', font=fnt(19, True))
        canvas.paste(to_panel(panels[c], PW, ph), (x, head))

    out = f'{OUT}/all_models_iiw_{image_id}.jpg'
    canvas.save(out, quality=93)
    print(f'wrote {out}  {canvas.size}')


if __name__ == '__main__':
    image_id = sys.argv[1] if len(sys.argv) > 1 else '100010'
    main(image_id)
