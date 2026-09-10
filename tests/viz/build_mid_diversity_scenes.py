#!/usr/bin/env python3
"""Additional MID qualitative scenes for presentation diversity (Section 13.1).

mid_ours.jpg / ablation_mid.jpg only ever show 'everett_dining1' (the
alphabetically-first MID test scene per fig_mid_ours()'s sc = sorted(...)[0]).
This regenerates the identical fig_mid_ours() protocol on two additional,
genuinely different held-out MID test scenes for qualitative diversity:
  - everett_dining2 (wood desk + colourful board-game box)
  - everett_kitchen5 (colourful bottles/labels, different material profile)

Uses the approved qualitative full-CARI model. Quantitative slides retain their
separately verified selected-evaluation provenance. Display follows the same
tonemap/scale-normalisation protocol as tests/viz/build_hires_figures.py.
"""
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, '/home/khang/IR-IID/tests/eval')
os.chdir('/home/khang/IR-IID/tests/eval')
sys.path.insert(0, '/home/khang/IR-IID/tests/viz')
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')

from eval_mid_constancy import AlbedoPredictor, _raw_frame, _tonemap_frame  # noqa: E402
from build_hires_figures import norm, scale_of, load_hdr, panel, fnt, INK, MUTE, RED, ROSTER  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

MID = '/home/khang/datasets/MIDIntrinsics/test'
OUT_DIR = '/home/khang/IR-IID/presentation/assets/generated/mid_diversity'
os.makedirs(OUT_DIR, exist_ok=True)


def grid_to_presentation(rows, col_labels, out_name, PW, ar, row_labels=None, gt_cols=()):
    """Local copy of build_hires_figures.grid(), redirected to presentation assets
    only -- does not touch documents/thesis/images/hires (the submitted thesis's
    own asset directory)."""
    PH = int(PW * ar)
    gap = max(5, PW // 60)
    header_font = int(PW / 17)
    row_font = int(PW / 19)
    head = 34
    if row_labels:
        rf = fnt(row_font, bold=True)
        probe = ImageDraw.Draw(Image.new('RGB', (8, 8)))
        lw = max(probe.textbbox((0, 0), t, font=rf)[2] for t in row_labels) + 26
    else:
        lw = 0
    ncol = len(col_labels)
    W = lw + ncol * PW + (ncol - 1) * gap
    H = head + len(rows) * (PH + gap) - gap
    cv = Image.new('RGB', (W, H), (255, 255, 255))
    d = ImageDraw.Draw(cv)
    for c, lab in enumerate(col_labels):
        x = lw + c * (PW + gap) + PW // 2
        d.multiline_text((x, head // 2), lab, anchor='mm', align='center', spacing=4,
                          font=fnt(header_font, bold=False), fill=(RED if c in gt_cols else INK))
    for r, cells in enumerate(rows):
        y = head + r * (PH + gap)
        if row_labels:
            d.text((lw - 14, y + PH // 2), row_labels[r], anchor='rm',
                   font=fnt(row_font, bold=True), fill=MUTE)
        for c, a in enumerate(cells):
            cv.paste(panel(a, PW, PH, gt=(c in gt_cols)), (lw + c * (PW + gap), y))
    out_path = f'{OUT_DIR}/{out_name}.jpg'
    cv.save(out_path, quality=95, subsampling=0)
    print(f'  wrote {out_path}  {cv.size[0]}x{cv.size[1]}')


def build_scene(scene_name, out_name):
    print(f'{out_name}: {scene_name}')
    sp = os.path.join(MID, scene_name)
    lights = [0, 6, 12, 18]
    ins = [_tonemap_frame(_raw_frame(sp, i)) for i in lights]
    label, ckpt, arch = ('Ours', '/home/khang/IR-IID/checkpoints/v17_29/checkpoint_iter_60000.pth', '17')
    p = AlbedoPredictor(ckpt, arch, 'cuda', infer_max_size=1280)
    albs = [p.albedo(x) for x in ins]
    del p
    torch.cuda.empty_cache()
    gt = _tonemap_frame(load_hdr(f'{sp}/albedo.exr'))

    st = np.stack([norm(a, scale_of(a)) for a in albs], 0)
    lum = 0.2126 * st[..., 0] + 0.7152 * st[..., 1] + 0.0722 * st[..., 2]
    cov = lum.std(0) / (lum.mean(0) + 1e-6)
    cov_v = cv2.applyColorMap((np.clip(cov / 0.25, 0, 1) * 255).astype(np.uint8),
                               cv2.COLORMAP_TURBO)[..., ::-1].astype(np.float32) / 255.0
    cov_v = cov_v ** 2.2

    rows = [[norm(x, scale_of(x)) for x in ins] + [norm(gt, scale_of(gt))],
            [norm(a, scale_of(a)) for a in albs] + [cov_v]]
    ar = ins[0].shape[0] / ins[0].shape[1]
    grid_to_presentation(rows, [f'light {i}' for i in lights] + ['GT albedo  /  variation'],
                          out_name, 440, ar,
                          row_labels=['Input (the light moves)', 'Our albedo (it should not)'],
                          gt_cols=(4,))


if __name__ == '__main__':
    build_scene('everett_dining2', 'mid_diversity_scene2')
    build_scene('everett_kitchen5', 'mid_diversity_scene3')
    print('done')
