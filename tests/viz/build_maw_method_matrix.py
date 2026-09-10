#!/usr/bin/env python3
"""Ours vs. all four external baselines on MAW (Measured Albedo in the Wild).
One real photograph per scene (MAW has no multi-light variants like MID), so
columns are methods, no light rows. One figure per scene, matching the
recurring scenes already used in the deck (Slide 38): scene_0 "Living area"
and scene_28 "Kitchen".

Usage: python build_maw_method_matrix.py <scene_dir> <image_stem>
       python build_maw_method_matrix.py scene_0 _DSC4366
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

MAW = f'{ROOT}/tests/testing_data/MAW'
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


def main(scene_dir, stem):
    ip = f'{MAW}/images_png/{scene_dir}/{stem}.png'
    gp1 = f'{MAW}/labels/new_masks/{scene_dir}/{stem}_albedo.png'
    gp2 = f'{MAW}/labels/new_masks2/{scene_dir}/{stem}_albedo.png'
    gp = gp1 if os.path.exists(gp1) else gp2

    inp_srgb = load_srgb(ip)
    inp_lin = inp_srgb ** 2.2
    gt_srgb = load_srgb(gp) if os.path.exists(gp) else None

    PW = 480
    ph = int(PW * inp_srgb.shape[0] / inp_srgb.shape[1])

    preds = {}
    for label, ckpt, ver in METHODS:
        p = AlbedoPredictor(ckpt, ver, 'cuda', infer_max_size=1280)
        alb = p.albedo(inp_lin)
        del p
        torch.cuda.empty_cache()
        preds[label] = srgb(norm(alb))
        print(f'  [{scene_dir}/{stem}] predicted: {label}')

    col_labels = ['Input'] + (['Measured mask'] if gt_srgb is not None else []) + [m[0] for m in METHODS]
    panels = [inp_srgb] + ([gt_srgb] if gt_srgb is not None else []) + [preds[m[0]] for m in METHODS]

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
        if lab == 'Measured mask':
            d.rectangle([x, head, x + PW - 1, head + ph - 1], outline=(200, 30, 30), width=3)

    out = f'{OUT}/all_models_maw_{scene_dir}.jpg'
    canvas.save(out, quality=93)
    print(f'wrote {out}  {canvas.size}')


if __name__ == '__main__':
    scene_dir = sys.argv[1] if len(sys.argv) > 1 else 'scene_0'
    stem = sys.argv[2] if len(sys.argv) > 2 else '_DSC4366'
    main(scene_dir, stem)
