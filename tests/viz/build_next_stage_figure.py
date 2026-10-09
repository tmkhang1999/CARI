#!/usr/bin/env python3
"""Pipeline figure for the planned trifactor model: three outputs and three CIAI losses.

Two pixel-aligned photographs of one 3D-Front-IID view under different illuminants go through
one shared model. Each pass outputs an albedo A, a grey shading S_lum and a unit-luminance
shading chroma C (I = A * (S_lum * C) + R). The two passes are tied by three losses:

    L_inv        A1 = A2
    L_lum_expl   luminance ratio of S_lum matches that of the images
    L_chr_expl   chroma ratio of C matches that of the images (only for pairs whose measured
                 illuminant gap reaches 0.08)

The model is NOT trained yet, so the output panels are the ground-truth targets, computed with
the renderer's albedo as S = I / A, S_lum = luminance(S), C = S / S_lum. No model prediction
appears in the figure. The image carries no explanatory text of its own: that the panels are
targets and the model is untrained is stated in the page text and the README caption.

Source of the photographs: --front3d-root (full-resolution corpus: rgb_L*.png, albedo.png of one
view), otherwise crops of documents/thesis/images/front3d/front3d_dataset.jpg (a small,
JPEG-compressed preview; fine for the diagram).

Run:  python tests/viz/build_next_stage_figure.py --out docs/static/img/ciai-pipeline.jpg
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
FONT = font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans'))
FONT_BOLD = font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans', weight='bold'))

FRONT3D_VIEW = 'fc20b1fa-5b64-4821-bbf6-075a7d35741b/LivingRoom-24701/view_01'
# (x0, y0, x1, y1) of the living-room row in front3d_dataset.jpg: illuminant 1, illuminant 3, albedo.
# The key-light swatch sits in the bottom-left, so only the top 86% of each panel is used.
FIGURE_PANELS = {'L0': (49, 906, 306, 1163), 'L2': (683, 906, 941, 1163), 'albedo': (1002, 906, 1259, 1163)}

INK, BLUE, RED, MUTE = (17, 24, 39), (29, 78, 216), (153, 27, 27), (92, 100, 112)
GAP_THRESHOLD = 0.08   # matches chr_explain_min_gap in src/configs/trifactor.yaml


def srgb_to_linear(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def lum(x):
    return 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]


def to_srgb(x):
    return np.clip(x, 0.0, 1.0) ** (1 / 2.2)


def load_front3d(root: Path | None):
    """Linear (I1, I2, A) of one 3D-Front-IID view, all HxWx3 float arrays of equal size."""
    if root is not None:
        view = root / FRONT3D_VIEW
        ims = [np.asarray(Image.open(view / n).convert('RGB'), dtype=np.float64) / 255.0
               for n in ('rgb_L0.png', 'rgb_L2.png', 'albedo.png')]
    else:
        sheet = Image.open(ROOT / 'documents/thesis/images/front3d/front3d_dataset.jpg').convert('RGB')
        ims = []
        for key in ('L0', 'L2', 'albedo'):
            x0, y0, x1, y1 = FIGURE_PANELS[key]
            h = y1 - y0
            ims.append(np.asarray(sheet.crop((x0 + 3, y0 + 3, x1 - 3, y0 + round(h * 0.86))),
                                  dtype=np.float64) / 255.0)
    h, w = min(m.shape[0] for m in ims), min(m.shape[1] for m in ims)   # crops can differ by a pixel
    return tuple(srgb_to_linear(m[:h, :w]) for m in ims)


def factors(i, a):
    """S = I / A, S_lum = luminance(S), C = S / S_lum (unit luminance); plus a validity mask."""
    s = i / np.maximum(a, 1e-3)
    s_lum = np.maximum(lum(s), 1e-6)
    valid = (lum(a) > 0.04) & (i.max(axis=-1) < 0.98) & (lum(i) > 0.012)
    return s_lum, s / s_lum[..., None], valid


def tile(img, size):
    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize(size, Image.LANCZOS)


def output_tiles(i, a, size):
    """Display tiles for albedo, grey shading and shading chroma (masked pixels light grey)."""
    s_lum, c, valid = factors(i, a)
    v = valid[..., None]
    grey = s_lum / (np.percentile(s_lum[valid], 99) + 1e-9)
    disp_grey = np.where(v, to_srgb(np.repeat(grey[..., None], 3, axis=-1)), 0.88)
    disp_c = np.where(v, to_srgb(c * 0.4), 0.88)           # neutral chroma -> mid grey
    return [tile(to_srgb(a), size), tile(disp_grey, size), tile(disp_c, size)]


def input_tile(i, size):
    return tile(to_srgb(i / (np.percentile(lum(i), 99) + 1e-9) * 0.8), size)


def arrow(d, x0, y0, x1, y1, color=MUTE, both=False, w=3):
    d.line((x0, y0, x1, y1), fill=color, width=w)
    ends = [(x0, y0, x1, y1)] + ([(x1, y1, x0, y0)] if both else [])
    for ax, ay, bx, by in ends:
        v = np.array([bx - ax, by - ay], dtype=float)
        v /= np.linalg.norm(v) + 1e-9
        n = np.array([-v[1], v[0]])
        tip = np.array([bx, by])
        d.polygon([tuple(tip), tuple(tip - 14 * v + 7 * n), tuple(tip - 14 * v - 7 * n)], fill=color)


def border(d, x, y, size, color=(203, 213, 225)):
    d.rectangle((x - 1, y - 1, x + size[0], y + size[1]), outline=color, width=2)


def build(out: Path, front3d_root: Path | None, tile_w: int = 260):
    """Same visual language as the CIAI mechanism figure (build_web_figures.build_mechanism)."""
    i1, i2, a = load_front3d(front3d_root)
    size = (tile_w, round(tile_w * i1.shape[0] / i1.shape[1]))
    th = size[1]
    f_big, f_mid, f_sm = (ImageFont.truetype(FONT_BOLD, 20), ImageFont.truetype(FONT, 19),
                          ImageFont.truetype(FONT, 15))
    f_bold = ImageFont.truetype(FONT_BOLD, 20)

    pad, row_gap, col_gap, model_w, link = 30, 112, 70, 190, 70
    top = 50
    x_in = pad
    x_model = x_in + tile_w + link
    x_out = x_model + model_w + link
    cols = [x_out + k * (tile_w + col_gap) for k in range(3)]
    y1, y2 = top, top + th + row_gap
    W = cols[2] + tile_w + 190
    H = y2 + th + 50
    canvas = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(canvas)

    canvas.paste(input_tile(i1, size), (x_in, y1))
    canvas.paste(input_tile(i2, size), (x_in, y2))
    for x in [x_in] + cols:
        for y in (y1, y2):
            border(d, x, y, size)
    d.text((x_in + tile_w // 2, y1 - 22), 'Photo, illuminant 1', fill=INK, font=f_bold, anchor='mm')
    d.text((x_in + tile_w // 2, y2 + th + 22), 'Photo, illuminant 2', fill=INK, font=f_bold, anchor='mm')

    # shared model
    my0, my1 = y1 + th // 3, y2 + 2 * th // 3
    d.rounded_rectangle((x_model, my0, x_model + model_w, my1), radius=14, fill=(234, 243, 255),
                        outline=BLUE, width=3)
    cy = (my0 + my1) // 2
    d.text((x_model + model_w // 2, cy - 44), 'Shared model', fill=INK, font=f_big, anchor='mm')
    for k, line in enumerate(('same weights', 'two passes', 'three heads')):
        d.text((x_model + model_w // 2, cy + k * 30), line, fill=(71, 85, 105), font=f_mid, anchor='mm')
    for y in (y1, y2):
        arrow(d, x_in + tile_w + 10, y + th // 2, x_model - 8, y + th // 2, color=(71, 85, 105))
        arrow(d, x_model + model_w + 8, y + th // 2, cols[0] - 10, y + th // 2, color=(71, 85, 105))

    # outputs: labels above the first pass and below the second, as in the mechanism figure
    names = [('Albedo A', BLUE), ('Grey shading S_lum', INK), ('Shading chroma C', INK)]
    for k, (name, color) in enumerate(names):
        cx = cols[k] + tile_w // 2
        d.text((cx, y1 - 22), name + '1', fill=color, font=f_bold, anchor='mm')
        d.text((cx, y2 + th + 22), name + '2', fill=color, font=f_bold, anchor='mm')
    for y, i in ((y1, i1), (y2, i2)):
        for k, t in enumerate(output_tiles(i, a, size)):
            canvas.paste(t, (cols[k], y))
            border(d, cols[k], y, size)

    # the three losses, centred on each column between the two passes
    mid = (y1 + th + y2) // 2
    labels = [
        ('L_inv:', ['A1 = A2']),
        ('L_lum_expl:', ['lum(S_lum1 / S_lum2)', '= lum(I1 / I2)']),
        ('L_chr_expl:', ['chroma(C1 / C2)', '= chroma(I1 / I2)', 'only if pair gap \u2265 %.2f' % GAP_THRESHOLD]),
    ]
    for k, (name, lines) in enumerate(labels):
        cx = cols[k] + tile_w // 2
        arrow(d, cx, y1 + th + 8, cx, y2 - 8, color=RED, both=True)
        n = len(lines) + 1
        y0 = mid - (n * 21) // 2 + 10
        d.text((cx + 16, y0), name, fill=RED, font=f_bold, anchor='lm')
        for j, ln in enumerate(lines):
            d.text((cx + 16, y0 + 24 + 21 * j), ln, fill=RED, font=f_sm, anchor='lm')

    canvas.save(out, quality=92, optimize=True)
    return canvas.size


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=ROOT / 'docs/static/img/ciai-pipeline.jpg')
    ap.add_argument('--front3d-root', type=Path, default=None,
                    help='full-resolution 3D-Front-IID corpus; default crops the thesis figure')
    args = ap.parse_args()
    print(build(args.out, args.front3d_root))
