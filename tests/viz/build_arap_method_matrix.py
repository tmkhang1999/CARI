#!/usr/bin/env python3
"""Multi-light method comparison matrix on ARAP (real illuminant variants).

Lights as ROWS (light0, light1, light2), methods as COLUMNS
(Ours, CRefNet, Marigold-App, Marigold-Light, Ordinal Shading), plus a bottom
VARIATION row (per-pixel luminance coefficient-of-variation across the
available lights, TURBO colormap) so cross-light stability is visible per
method, not just inferred from the aggregate table.

Usage: CUDA_VISIBLE_DEVICES=1 python build_arap_method_matrix.py <scene>
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
sys.path.insert(0, os.path.join(ROOT, 'tests/viz'))
from eval_mid_constancy import AlbedoPredictor  # noqa: E402
from build_hires_figures import load_hdr  # noqa: E402

ARAP = f'{ROOT}/tests/testing_data/ARAP_dataset'
OUT = f'{ROOT}/presentation/assets/generated/rev2_matrices'
os.makedirs(OUT, exist_ok=True)

CK = f'{ROOT}/checkpoints'
METHODS = [
    ('Ours',           f'{CK}/v17_29/checkpoint_iter_60000.pth', '17'),
    ('CRefNet',        f'{CK}/CRefNet/final_real.pt', 'crefnet'),
    ('Marigold-App',   f'{CK}/marigold-iid-appearance-v1-1', 'marigold-appearance'),
    ('Marigold-Light', f'{CK}/marigold-iid-lighting-v1-1', 'marigold-lighting'),
    ('Ordinal Shading', f'{ROOT}/ordinal-hub-weights', 'ordinal'),
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


def to_panel(arr, w, h):
    return Image.fromarray((srgb(arr) * 255).astype(np.uint8)).resize((w, h), Image.LANCZOS)


def variation_heatmap(preds_norm_list, w, h):
    st = np.stack(preds_norm_list, 0)
    lum = 0.2126 * st[..., 0] + 0.7152 * st[..., 1] + 0.0722 * st[..., 2]
    cov = lum.std(0) / (lum.mean(0) + 1e-6)
    cov_u8 = (np.clip(cov / 0.25, 0, 1) * 255).astype(np.uint8)
    cov_color = cv2.applyColorMap(cov_u8, cv2.COLORMAP_TURBO)[..., ::-1].astype(np.float32) / 255.0
    return Image.fromarray((cov_color * 255).astype(np.uint8)).resize((w, h), Image.LANCZOS)


def main(scene, n_lights=3):
    lights = list(range(n_lights))
    ins = [load_hdr(f'{ARAP}/{scene}_light{i}.hdr') for i in lights]
    ins = [norm(x) for x in ins]
    PW = 480
    ph = int(PW * ins[0].shape[0] / ins[0].shape[1])

    col_preds = {}
    for label, ckpt, ver in METHODS:
        p = AlbedoPredictor(ckpt, ver, 'cuda', infer_max_size=1280)
        preds = [p.albedo(x) for x in ins]
        del p
        torch.cuda.empty_cache()
        col_preds[label] = [norm(x) for x in preds]
        print(f'  predicted: {label}')

    ncol = len(METHODS) + 1
    nrow = len(lights) + 1
    gap, head, rowlab_w = 8, 50, 130
    W = rowlab_w + ncol * PW + (ncol - 1) * gap
    H = head + nrow * (ph + gap) - gap
    canvas = Image.new('RGB', (W, H), (255, 255, 255))
    d = ImageDraw.Draw(canvas)

    col_labels = ['Input'] + [m[0] for m in METHODS]
    for c, lab in enumerate(col_labels):
        x = rowlab_w + c * (PW + gap)
        bold = lab == 'Ours'
        d.text((x + PW // 2, head // 2), lab, fill=(23, 32, 51) if not bold else (22, 119, 255),
               anchor='mm', font=fnt(19, True))

    for r, light in enumerate(lights):
        y = head + r * (ph + gap)
        d.text((rowlab_w // 2, y + ph // 2), f'light {light}', fill=(23, 32, 51), anchor='mm', font=fnt(20, True))
        canvas.paste(to_panel(ins[r], PW, ph), (rowlab_w, y))
        for c, (label, _ckpt, _ver) in enumerate(METHODS):
            x = rowlab_w + (c + 1) * (PW + gap)
            canvas.paste(to_panel(col_preds[label][r], PW, ph), (x, y))

    yv = head + len(lights) * (ph + gap)
    d.text((rowlab_w // 2, yv + ph // 2), f'variation\n({n_lights} lights)', fill=(196, 61, 61), anchor='mm', font=fnt(16, True))
    for c, (label, _ckpt, _ver) in enumerate(METHODS):
        x = rowlab_w + (c + 1) * (PW + gap)
        heat = variation_heatmap(col_preds[label], PW, ph)
        canvas.paste(heat, (x, yv))

    out = f'{OUT}/arap_matrix_{scene}.jpg'
    canvas.save(out, quality=93)
    print(f'wrote {out}  {canvas.size}')


if __name__ == '__main__':
    scene = sys.argv[1] if len(sys.argv) > 1 else 'kitchen'
    n_lights = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    main(scene, n_lights=n_lights)
