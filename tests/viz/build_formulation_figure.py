#!/usr/bin/env python3
"""fig:formulation (Ch2) — why a grayscale-shading model must leak illuminant colour.

Chapter 2 currently argues the thesis premise entirely in prose. The rival works both make
this point with a figure (CD-IID Figs. 2 and 7), and it is the single idea the whole thesis
rests on, so it should be visible before the method chapter rather than after.

The argument is arithmetic, not empirical, and the figure shows it as such. Under a coloured
illuminant, I = A * S. If the model is only allowed a SCALAR shading s, then the only place
the illuminant's hue can go is the albedo:

    A_grey = I / s        <- inherits the lamp's colour, by construction
    A_ours = I / S_rgb    <- the 3-channel shading can absorb it

We show both, computed from the SAME real MID frame and its ground-truth albedo, so the two
albedos differ only in the shading model they were divided by. The grey-shading albedo is
the best any grayscale method could possibly do on this frame -- it is fitted against the
ground truth -- and it is still visibly tinted.
"""
import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
ROOT = '/home/khang/IR-IID'
sys.path.insert(0, os.path.join(ROOT, 'tests/eval'))
os.chdir(os.path.join(ROOT, 'tests/eval'))

from eval_mid_constancy import _raw_frame, _tonemap_frame  # noqa: E402

MID = '/home/khang/datasets/MIDIntrinsics/test'
OUT = f'{ROOT}/tests/visualizations/formulation'
DST = f'{ROOT}/documents/thesis/images/formulation'
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONTB = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
RED, GRN, INK = (196, 30, 36), (25, 110, 40), (25, 25, 28)


def fnt(s, b=False):
    try:
        return ImageFont.truetype(FONTB if b else FONT, s)
    except Exception:
        return ImageFont.load_default()


def srgb(x):
    return np.clip(x, 0, 1) ** (1 / 2.2)


def nrm(a, pct=99.0):
    v = a[np.isfinite(a) & (a > 1e-6)]
    s = float(np.percentile(v, pct)) if v.size else 1.0
    return np.clip(np.nan_to_num(a) / (s + 1e-8), 0, 1)


def gray(a):
    return 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]


def load_case(sc, light):
    sp = os.path.join(MID, sc)
    I = _tonemap_frame(_raw_frame(sp, light))                 # coloured-illuminant frame
    A = _tonemap_frame(cv2.imread(f'{sp}/albedo.exr',
                                  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH
                                  )[..., ::-1].astype(np.float32))

    valid = (gray(A) > 0.02) & (gray(I) > 0.01)

    # The TRUE shading implied by the data is 3-channel and carries the lamp's hue.
    S_rgb = np.zeros_like(I)
    np.divide(I, np.clip(A, 1e-3, None), out=S_rgb, where=valid[..., None])

    # A grayscale model is constrained to a SCALAR shading. Give it the most favourable one
    # possible: the per-pixel luminance of the true shading. Even so, dividing by it cannot
    # remove the illuminant's hue, because a scalar has no hue to remove.
    s_grey = gray(S_rgb)
    A_grey = np.zeros_like(I)
    np.divide(I, np.clip(s_grey, 1e-3, None)[..., None], out=A_grey, where=valid[..., None])

    # chroma error of each albedo against ground truth (r/g, b/g), display as a heat map
    def chroma_err(X):
        g = np.clip(X[..., 1], 1e-3, None)
        gg = np.clip(A[..., 1], 1e-3, None)
        e = np.hypot(X[..., 0] / g - A[..., 0] / gg, X[..., 2] / g - A[..., 2] / gg)
        e = np.where(valid, e, 0)
        v = cv2.applyColorMap((np.clip(e / 0.6, 0, 1) * 255).astype(np.uint8),
                              cv2.COLORMAP_INFERNO)[..., ::-1].astype(np.float32) / 255.0
        return v ** 2.2, float(e[valid].mean())

    err_grey, m_grey = chroma_err(A_grey)
    _err_rgb, m_rgb = chroma_err(I / np.clip(S_rgb, 1e-3, None))  # exact by construction
    return I, A, S_rgb, A_grey, err_grey, m_grey, m_rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default='everett_kitchen18')
    ap.add_argument('--light', type=int, default=18)
    ap.add_argument('--scan', action='store_true',
                    help='Print high-error candidate scene/light pairs without writing a figure.')
    args = ap.parse_args()

    if args.scan:
        scores = []
        for sc in sorted(d for d in os.listdir(MID) if os.path.isdir(os.path.join(MID, d))):
            sp = os.path.join(MID, sc)
            lights = sorted({int(f.split('_')[1]) for f in os.listdir(sp)
                             if f.startswith('dir_') and f.endswith('_mip2.exr')})
            # Four evenly distributed illuminants are enough to find a clear teaching case.
            picks = lights if len(lights) <= 4 else [lights[i] for i in
                    np.linspace(0, len(lights) - 1, 4, dtype=int)]
            for light in picks:
                try:
                    *_, m_grey, m_rgb = load_case(sc, light)
                    scores.append((m_grey - m_rgb, m_grey, sc, light))
                except Exception as exc:
                    print(f'[skip] {sc} light {light}: {exc}')
        for gap, err, sc, light in sorted(scores, reverse=True)[:20]:
            print(f'{sc:24s} light={light:2d}  grey_chroma={err:.3f}  gap={gap:.3f}')
        return

    os.makedirs(OUT, exist_ok=True)
    os.makedirs(DST, exist_ok=True)
    I, A, S_rgb, A_grey, err_grey, m_grey, m_rgb = load_case(args.scene, args.light)

    print(f'scene={args.scene} light={args.light}: mean albedo chroma error — '
          f'grey shading: {m_grey:.3f}   colour shading: {m_rgb:.3f}')

    cells = [
        ('Input  $I$  (coloured light)', nrm(I), None, ''),
        ('True albedo  $A$', nrm(A), 'gt', 'what the paint actually is'),
        ('Colour shading  $S_{RGB}$', nrm(S_rgb), None, 'the lamp’s hue lives here'),
        ('Grey-shading albedo  $I/s$', nrm(A_grey), 'bad',
         f'chroma error {m_grey:.2f}: tinted'),
    ]

    PW = 760
    H, W = I.shape[:2]
    PH = int(PW * H / W)
    gap, head, foot = 10, 112, 98
    Wt = len(cells) * PW + (len(cells) - 1) * gap
    canvas = Image.new('RGB', (Wt, head + PH + foot), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    for i, (lab, arr, kind, note) in enumerate(cells):
        x = i * (PW + gap)
        im = Image.fromarray((srgb(arr) * 255).astype(np.uint8)).resize((PW, PH),
                                                                        Image.LANCZOS)
        canvas.paste(im, (x, head))
        col = RED if kind in ('gt', 'bad') else INK
        d.text((x + PW // 2, head // 2), lab.replace('$', '').replace('_{RGB}', ' RGB')
               .replace('_{fid}', ''), anchor='mm', font=fnt(52, b=kind is not None), fill=col)
        if kind in ('gt', 'bad'):
            d.rectangle([x, head, x + PW - 1, head + PH - 1], outline=RED, width=5)
        if note:
            d.text((x + PW // 2, head + PH + foot // 2), note, anchor='mm',
                   font=fnt(42, b=True), fill=(RED if kind == 'bad' else INK))
    for p in (f'{OUT}/formulation.jpg', f'{DST}/formulation.jpg'):
        canvas.save(p, quality=95, subsampling=0)
        print('wrote', p, canvas.size)


if __name__ == '__main__':
    main()
