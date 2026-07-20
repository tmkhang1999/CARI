#!/usr/bin/env python3
"""fig:limitations — where the model fails.

Every other qualitative figure in the thesis shows the model working. Both of the works we
compare against (Ordinal Shading Fig. 8; CD-IID Fig. 12) devote a figure to their failure
cases, and a reviewer trusts a thesis MORE, not less, for showing them. All three failures
below are ones we have measured, not invented:

  (a) OVER-SATURATION FROM THE FLATNESS PRIOR. Table-B Row 5 (v17_33, flat weight 4.0)
      is associated with worse pseudo-GT calibration and an aggregate spread ratio of
      1.055 versus 0.941 for the selected model. The logged mixtures differ, so this
      figure documents a configuration-level failure rather than a flatness-only cause.

  (b) NON-DIFFUSE SURFACES. The specular sphere in MID is where cross-illuminant albedo
      drift concentrates (it is the hot region in the CoV map of fig:mid_ours and in the
      cross-illuminant instability). A Lambertian albedo/shading split has no
      representation for a mirror.

  (c) ACHROMATIC TEXTURE/SHADING CONFUSION. CARI constrains COLOUR differences across
      illuminants; it says nothing about a dark material under bright light versus a light
      material in shadow. This ambiguity is untouched by our contribution (Ch. 7).
"""
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
ROOT = '/home/khang/IR-IID'
sys.path.insert(0, os.path.join(ROOT, 'tests/eval'))
os.chdir(os.path.join(ROOT, 'tests/eval'))

from eval_mid_constancy import AlbedoPredictor, _raw_frame, _tonemap_frame  # noqa: E402

MID = '/home/khang/datasets/MIDIntrinsics/test'
OUT = f'{ROOT}/tests/visualizations/limitations'
DST = f'{ROOT}/documents/thesis/images/limitations'
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONTB = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
RED, INK, MUTE = (196, 30, 36), (25, 25, 28), (95, 95, 102)

OURS = f'{ROOT}/checkpoints/v17_34/checkpoint_iter_60000.pth'
FLAT4 = f'{ROOT}/checkpoints/v17_33/checkpoint_iter_60000.pth'


def fnt(s, b=False):
    try:
        return ImageFont.truetype(FONTB if b else FONT, s)
    except Exception:
        return ImageFont.load_default()


def srgb(x):
    return np.clip(x, 0, 1) ** (1 / 2.2)


def nrm(a, pct=99.0):
    v = a[a > 1e-6]
    s = float(np.percentile(v, pct)) if v.size else 1.0
    return np.clip(a / (s + 1e-8), 0, 1)


def pil(a, w, h):
    return Image.fromarray((srgb(a) * 255).astype(np.uint8)).resize((w, h), Image.LANCZOS)


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(DST, exist_ok=True)
    sc = sorted(os.listdir(MID))[0]
    sp = os.path.join(MID, sc)
    lights = [0, 12]
    ins = [_tonemap_frame(_raw_frame(sp, i)) for i in lights]
    gt = _tonemap_frame(cv2.imread(f'{sp}/albedo.exr',
                                   cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH
                                   )[..., ::-1].astype(np.float32))

    preds = {}
    for lab, ck in (('ours', OURS), ('flat4', FLAT4)):
        p = AlbedoPredictor(ck, '17', 'cuda', infer_max_size=1280)
        preds[lab] = [p.albedo(x) for x in ins]
        del p
        torch.cuda.empty_cache()

    # (b) cross-illuminant instability, per pixel — concentrates on the specular sphere
    st = np.stack([nrm(a) for a in preds['ours']], 0)
    lum = 0.2126 * st[..., 0] + 0.7152 * st[..., 1] + 0.0722 * st[..., 2]
    cov = lum.std(0) / (lum.mean(0) + 1e-6)
    cov_v = cv2.applyColorMap((np.clip(cov / 0.25, 0, 1) * 255).astype(np.uint8),
                              cv2.COLORMAP_TURBO)[..., ::-1].astype(np.float32) / 255.0
    cov_v = cov_v ** 2.2

    cells = [
        ('Input', nrm(ins[0]), None),
        ('Ground-truth albedo', nrm(gt), 'gt'),
        ('Ours  (flat 1.5)', nrm(preds['ours'][0]), 'ok'),
        ('Flatness prior at full strength', nrm(preds['flat4'][0]), 'bad'),
        ('Cross-illuminant instability', cov_v, 'bad'),
    ]
    notes = [
        '',
        'Chroma$_{fid}$ = 1.000',
        'spread ratio = 0.941',
        'spread ratio = 1.055; Chroma_err = 0.143',
        'concentrated on the specular sphere',
    ]

    PW = 620
    H, W = ins[0].shape[:2]
    PH = int(PW * H / W)
    gap, head, foot = 8, 46, 54
    Wt = len(cells) * PW + (len(cells) - 1) * gap
    canvas = Image.new('RGB', (Wt, head + PH + foot), (255, 255, 255))
    d = ImageDraw.Draw(canvas)

    for i, ((lab, arr, kind), note) in enumerate(zip(cells, notes)):
        x = i * (PW + gap)
        canvas.paste(pil(arr, PW, PH), (x, head))
        col = RED if kind in ('gt', 'bad') else INK
        d.text((x + PW // 2, head // 2), lab, anchor='mm',
               font=fnt(24, b=(kind in ('ok', 'gt', 'bad'))), fill=col)
        if kind in ('gt', 'bad'):
            d.rectangle([x, head, x + PW - 1, head + PH - 1], outline=RED, width=5)
        if note:
            nc = RED if kind == 'bad' else ((25, 110, 40) if kind == 'ok' else RED)
            d.text((x + PW // 2, head + PH + foot // 2), note.replace('$_{fid}$', '_fid'),
                   anchor='mm', font=fnt(21, b=True), fill=nc)

    for p in (f'{OUT}/limitations.jpg', f'{DST}/limitations.jpg'):
        canvas.save(p, quality=95, subsampling=0)
        print('wrote', p, canvas.size)


if __name__ == '__main__':
    main()
