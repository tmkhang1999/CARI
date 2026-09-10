#!/usr/bin/env python3
"""Slide 33 (Results: held-out scenes) rebuilt WITH inputs, per feedback #8:
Input A | Input B | Reference | Ours A | Ours B, per scene, two scenes.
v17_29 qualitative checkpoint.
"""
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
OUT = f'{ROOT}/presentation/assets/generated/rev2_matrices'
CK = f'{ROOT}/checkpoints/v17_29/checkpoint_iter_60000.pth'

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


def main():
    scenes = [('everett_kitchen5', [0, 18]), ('everett_dining2', [0, 18])]
    p = AlbedoPredictor(CK, '17', 'cuda', infer_max_size=1280)

    PW, gap, head, rowlab_w = 380, 8, 46, 170
    ncol = 5
    rows_data = []
    for scene, lights in scenes:
        sp = os.path.join(MID, scene)
        ins = [_tonemap_frame(_raw_frame(sp, l)) for l in lights]
        alb = cv2.imread(os.path.join(sp, 'albedo.exr'), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        gt = norm(_tonemap_frame(alb[..., ::-1].copy().astype(np.float32)))
        preds = [norm(p.albedo(x)) for x in ins]
        rows_data.append((scene, [ins[0], ins[1], gt, preds[0], preds[1]]))
        print('  predicted:', scene)
    del p
    import torch
    torch.cuda.empty_cache()

    ph = int(PW * rows_data[0][1][0].shape[0] / rows_data[0][1][0].shape[1])
    col_labels = ['Input A', 'Input B', 'Reference', 'Ours A', 'Ours B']
    W = rowlab_w + ncol * PW + (ncol - 1) * gap
    H = head + len(rows_data) * (ph + gap) - gap
    canvas = Image.new('RGB', (W, H), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    for c, lab in enumerate(col_labels):
        x = rowlab_w + c * (PW + gap)
        d.text((x + PW // 2, head // 2), lab, fill=(23, 32, 51), anchor='mm', font=fnt(20, True))
    for r, (scene, imgs) in enumerate(rows_data):
        y = head + r * (ph + gap)
        d.text((rowlab_w // 2, y + ph // 2), scene.replace('everett_', ''), fill=(23, 32, 51),
               anchor='mm', font=fnt(17, True))
        for c, arr in enumerate(imgs):
            x = rowlab_w + c * (PW + gap)
            panel = to_panel(arr, PW, ph)
            canvas.paste(panel, (x, y))
            if c == 2:  # reference: thin blue border
                dd = ImageDraw.Draw(canvas)
                dd.rectangle([x, y, x + PW - 1, y + ph - 1], outline=(22, 119, 255), width=3)
    out = f'{OUT}/slide33_with_inputs.jpg'
    canvas.save(out, quality=93)
    print('wrote', out, canvas.size)


if __name__ == '__main__':
    main()
