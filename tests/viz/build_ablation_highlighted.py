#!/usr/bin/env python3
"""CARI 2x2 ablation figures with a highlight box around a colourful reference
object in every panel, so a viewer can see at a glance that CARI-off /
colour-path-off configurations desaturate that object while full CARI keeps
its colour. Same TABLE_A checkpoints and protocol as fig_ablation() in
build_hires_figures.py; output to presentation assets only.
"""
import os
import sys

import torch
from PIL import Image, ImageDraw

sys.path.insert(0, '/home/khang/IR-IID/tests/eval')
os.chdir('/home/khang/IR-IID/tests/eval')
sys.path.insert(0, '/home/khang/IR-IID/tests/viz')
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')

from eval_mid_constancy import AlbedoPredictor, _raw_frame, _tonemap_frame  # noqa: E402
from build_hires_figures import norm, scale_of, load_hdr, panel, fnt, INK, RED, MUTE, TABLE_A  # noqa: E402

MID = '/home/khang/datasets/MIDIntrinsics/test'
OUT_DIR = '/home/khang/IR-IID/presentation/assets/generated/mid_diversity'
CARI_BLUE = (22, 119, 255)

# (scene, lights, highlight box as fraction of frame (x0,y0,x1,y1), draw box)
SCENES = [
    ('everett_dining1', [0, 6, 12, 18], (0.34, 0.03, 0.56, 0.62), False),
    ('everett_kitchen5', [0, 6, 12, 18], (0.60, 0.10, 0.92, 0.60), True),
]


def build(scene, lights, box, draw_box, out_name):
    print(f'{out_name}: {scene} lights {lights}')
    sp = os.path.join(MID, scene)
    inps = [_tonemap_frame(_raw_frame(sp, l)) for l in lights]
    gt = _tonemap_frame(load_hdr(f'{sp}/albedo.exr'))
    preds = {lab: [] for lab, _ in TABLE_A}
    for label, path in TABLE_A:
        p = AlbedoPredictor(path, '17', 'cuda', infer_max_size=1280)
        for inp in inps:
            preds[label].append(p.albedo(inp))
        del p
        torch.cuda.empty_cache()

    PW, gap, head, rowlabel_w = 300, 10, 36, 90
    ar = inps[0].shape[0] / inps[0].shape[1]
    PH = int(PW * ar)
    col_names = ['Input', 'GT albedo'] + [lab for lab, _ in TABLE_A]
    ncols = len(col_names)
    W = rowlabel_w + ncols * PW + (ncols - 1) * gap
    H = head + len(lights) * (PH + gap) - gap
    canvas = Image.new('RGB', (W, H), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    for i, label in enumerate(col_names):
        x = rowlabel_w + i * (PW + gap)
        d.text((x + PW // 2, head // 2), label, anchor='mm',
               font=fnt(17, bold=(i == ncols - 1)), fill=(RED if i == 1 else INK))
    x0f, y0f, x1f, y1f = box
    for r, light in enumerate(lights):
        y = head + r * (PH + gap)
        d.text((rowlabel_w - 10, y + PH // 2), f'light {light}', anchor='rm',
               font=fnt(16, bold=True), fill=MUTE)
        cols = [norm(inps[r], scale_of(inps[r])), norm(gt, scale_of(gt))] + \
               [norm(preds[lab][r], scale_of(preds[lab][r])) for lab, _ in TABLE_A]
        for c, arr in enumerate(cols):
            x = rowlabel_w + c * (PW + gap)
            canvas.paste(panel(arr, PW, PH, gt=(c == 1)), (x, y))
            if draw_box:
                bx0, by0, bx1, by1 = x + x0f * PW, y + y0f * PH, x + x1f * PW, y + y1f * PH
                d.rectangle([bx0, by0, bx1, by1], outline=CARI_BLUE, width=3)
    out = f'{OUT_DIR}/{out_name}.jpg'
    canvas.save(out, quality=95, subsampling=0)
    print(f'  wrote {out}  {canvas.size}')


if __name__ == '__main__':
    for scene, lights, box, draw_box in SCENES:
        build(scene, lights, box, draw_box, f'ablation_highlighted_{scene}')
