#!/usr/bin/env python3
"""Figure-5.11-style multi-light matched ablation matrix for Slides 30-31.

Lights as ROWS (0, 6, 12, 18), configs as COLUMNS (Input / GT albedo / Row1
neither / Row2 CARI only / Row3 colour path only / Row4 full CARI).

One medium blue box is auto-placed on the most colourful, textured object
cluster (scored on the light-0 input, so it lands on real material rather
than a flat wall, the table, or the neutral calibration spheres). The same
box coordinates are drawn on every column and every row, so the audience can
track one object group across lights and across configurations.

Only the horizontal CARI contrasts (col1->col2, col3->col4) are causal
(FACT_CHECK S11.1/A06.2); the vertical colour-path difference is descriptive
only (S11.5) -- both facts are annotated on the slide, not baked into images.

Usage: CUDA_VISIBLE_DEVICES=0 python build_mid_ablation_matrix.py <scene>
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
from eval_mid_constancy import AlbedoPredictor, _raw_frame, _tonemap_frame  # noqa: E402

MID = '/home/khang/datasets/MIDIntrinsics/test'
OUT = f'{ROOT}/presentation/assets/generated/rev2_matrices'
os.makedirs(OUT, exist_ok=True)

CK = f'{ROOT}/checkpoints'
CONFIGS = [
    ('Row 1: neither',      f'{CK}/v17_41/checkpoint_iter_40000.pth'),
    ('Row 2: CARI only',    f'{CK}/v17_42/checkpoint_iter_40000.pth'),
    ('Row 3: colour path only', f'{CK}/v17_43/checkpoint_iter_40000.pth'),
    ('Row 4: full CARI',    f'{CK}/v17_44/checkpoint_iter_40000.pth'),
]
LIGHTS = [0, 6, 12, 18]
BOX = (66, 133, 244)  # blue

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


def find_object_box(ref_input_norm, win_frac=0.34, stride_frac=0.04):
    """Single medium box over the most colourful, textured object cluster.

    Scored on the light-0 input (not on a CoV map): windows are ranked by how
    much saturated, high-frequency material they contain, so the box lands on
    the actual coloured objects rather than a flat wall, the table, or the
    neutral calibration spheres."""
    H, W = ref_input_norm.shape[:2]
    win = int(min(H, W) * win_frac)
    stride = max(1, int(min(H, W) * stride_frac))

    gray = cv2.cvtColor((ref_input_norm * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    texture = np.sqrt(gx ** 2 + gy ** 2)
    texture /= (texture.max() + 1e-6)
    mx, mn = ref_input_norm.max(-1), ref_input_norm.min(-1)
    saturation = (mx - mn) / (mx + 1e-6)

    # Colourful AND detailed. Saturation (not raw chroma) keeps bright warm-but-
    # washed surfaces down, and squaring it further favours strongly coloured
    # material: without both, the search settles on a lit wooden worktop or a
    # door frame instead of the actual coloured objects.
    score_map = (saturation ** 2) * texture

    margin = int(min(H, W) * 0.04)
    best, best_xy = -1.0, (max(0, (W - win) // 2), max(0, (H - win) // 2))
    for y in range(margin, max(margin + 1, H - win - margin), stride):
        for x in range(margin, max(margin + 1, W - win - margin), stride):
            s = float(score_map[y:y + win, x:x + win].mean())
            if s > best:
                best, best_xy = s, (x, y)

    return (best_xy[0], best_xy[1], win, win)


def draw_box(d, box, scale_x, scale_y, offset_x, offset_y, label=None):
    x, y, w, h = box
    x0, y0 = offset_x + x * scale_x, offset_y + y * scale_y
    x1, y1 = offset_x + (x + w) * scale_x, offset_y + (y + h) * scale_y
    d.rectangle([x0, y0, x1, y1], outline=BOX, width=4)
    if label:
        tw = d.textlength(label, font=fnt(15, True))
        d.rectangle([x0, y0 - 22, x0 + tw + 10, y0], fill=BOX)
        d.text((x0 + 5, y0 - 20), label, fill=(255, 255, 255), font=fnt(15, True))


def main(scene):
    sp = os.path.join(MID, scene)
    ins = [_tonemap_frame(_raw_frame(sp, l)) for l in LIGHTS]
    gt = _tonemap_frame(cv2.imread(os.path.join(sp, 'albedo.exr'),
                                    cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[..., ::-1].copy().astype(np.float32))
    PW = 520
    ph = int(PW * ins[0].shape[0] / ins[0].shape[1])

    col_preds = {}
    for label, ckpt in CONFIGS:
        p = AlbedoPredictor(ckpt, '17', 'cuda', infer_max_size=1280)
        preds = [p.albedo(x) for x in ins]
        del p
        torch.cuda.empty_cache()
        col_preds[label] = [norm(x) for x in preds]
        print(f'  predicted: {label}')

    ref = norm(ins[0])
    box = find_object_box(ref)
    src_h, src_w = ref.shape[:2]
    scale_x, scale_y = PW / src_w, ph / src_h

    ncol = len(CONFIGS) + 2  # +1 Input, +1 GT albedo
    nrow = len(LIGHTS)
    gap, head, rowlab_w = 8, 50, 130
    W = rowlab_w + ncol * PW + (ncol - 1) * gap
    H = head + nrow * (ph + gap) - gap
    canvas = Image.new('RGB', (W, H), (255, 255, 255))
    d = ImageDraw.Draw(canvas)

    col_labels = ['Input', 'GT albedo'] + [c[0] for c in CONFIGS]
    for c, lab in enumerate(col_labels):
        x = rowlab_w + c * (PW + gap)
        d.text((x + PW // 2, head // 2), lab, fill=(23, 32, 51), anchor='mm', font=fnt(20, True))

    gt_panel = to_panel(norm(gt), PW, ph)
    for r, light in enumerate(LIGHTS):
        y = head + r * (ph + gap)
        d.text((rowlab_w // 2, y + ph // 2), f'light {light}', fill=(23, 32, 51), anchor='mm', font=fnt(20, True))

        x_in = rowlab_w
        canvas.paste(to_panel(ins[r], PW, ph), (x_in, y))
        draw_box(d, box, scale_x, scale_y, x_in, y)

        x_gt = rowlab_w + (PW + gap)
        canvas.paste(gt_panel, (x_gt, y))
        draw_box(d, box, scale_x, scale_y, x_gt, y)

        for c, (label, _ckpt) in enumerate(CONFIGS):
            x = rowlab_w + (c + 2) * (PW + gap)
            canvas.paste(to_panel(col_preds[label][r], PW, ph), (x, y))
            draw_box(d, box, scale_x, scale_y, x, y)

    out = f'{OUT}/mid_ablation_{scene}.jpg'
    canvas.save(out, quality=93)
    print(f'wrote {out}  {canvas.size}')


if __name__ == '__main__':
    scene = sys.argv[1] if len(sys.argv) > 1 else 'everett_dining1'
    if len(sys.argv) > 2:
        LIGHTS = [int(x) for x in sys.argv[2:]]
    main(scene)
