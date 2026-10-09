#!/usr/bin/env python3
"""Build the figure set for the CIAI project page (tmkhang1999.github.io/CIAI/, served from docs/).

Every image on the public page must trace to a file this repository produced. Two
sources qualify, and nothing else is allowed in here:

  1. `presentation/assets/from_submission/` - the figures of the defence deck, exported
     from `Khang_submission.pptx`. Each one was hash-matched back to a repo file or to a
     crop of `documents/thesis/Main.pdf`; see MANIFEST.json in that directory.
  2. `documents/thesis/images/` - figures compiled into the thesis itself.

Outputs (written to --out, default documents/thesis/images/web/):
  ciai-teaser.jpg       one MID scene under four flash directions, with the recovered albedo
  ciai-ambiguity.jpg    why one-channel shading forces the light's hue into the albedo
  ciai-pairs.jpg        one MID pair; the deck's "cool"/"warm" header is repainted as
                        "flash direction A/B" (see relabel_pairs)
  ciai-mechanism.jpg    the CIAI training-time constraint, composed from the thesis thumbnails
  ciai-qualitative.jpg  held-out scene, all seven methods, three flash directions
                        (from tests/viz/build_all_models_light_matrix.py)
  ciai-thumb.jpg        card thumbnail for the portfolio (232x142 aspect); copy it by hand

The compare-widget images under docs/static/img/compare/ come from
tests/viz/build_compare_widget_assets.py.

Run:
  python tests/viz/build_web_figures.py --out docs/static/img
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[2]
DECK = ROOT / 'presentation/assets/from_submission'
THESIS = ROOT / 'documents/thesis/images'

# DejaVu ships with matplotlib, so resolving it through font_manager works on Linux, macOS
# and Windows alike; a hard-coded /usr/share path only existed on the Linux build box.
from matplotlib import font_manager
FONT = font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans'))
FONT_BOLD = font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans', weight='bold'))

# Light palette. Academic project pages are white-backed (verified against the
# IntrinsicImageDiffusion, ColorfulShading, RGB->X and Nerfies pages, all
# rgb(255,255,255)), every other figure in this set is already on white, and
# GitHub renders the README on white too -- so dark plots were the odd ones out.
BG, FG, GRID = '#ffffff', '#1f2937', '#e5e7eb'
SKY, SLATE, AMBER = '#1d4ed8', '#6b7280', '#b45309'

# Chapter 5, tab:mid - held-out MIDIntrinsics split, 30 scenes. Brackets are the same
# 95% percentile-bootstrap intervals (over scenes) the thesis table reports; Chroma_fid
# carries no interval there, so none is fabricated here either.
#   name: (Cast_rel, [lo, hi], Chroma_err, [lo, hi], Chroma_fid)

# Straight copies, resized for the web. (source, output name, max width)
COPIES = [
    (THESIS / 'formulation/formulation.jpg', 'ciai-ambiguity.jpg', 1500),
    (ROOT / 'presentation/assets/generated/all_models_light_matrix/'
     'all_models_everett_lobby3.jpg',                     'ciai-qualitative.jpg', 1900),
]

# The three limitation crops, in the order they are discussed on the page.


def _resize(src: Path, dst: Path, max_width: int, quality: int = 88):
    im = Image.open(src).convert('RGB')
    if im.width > max_width:
        im = im.resize((max_width, round(im.height * max_width / im.width)), Image.LANCZOS)
    im.save(dst, quality=quality, optimize=True)
    return im.size


def build_teaser(out: Path, max_width: int = 1700):
    """Hero image: input vs. recovered albedo across four lamp settings.

    The source grid (from build_mid_diversity_scenes.py) has a fifth column  - 
    GT albedo over a Turbo colormap of cross-light variation - that carries no
    colorbar, scale, or clip threshold, and the page never explains it. Rather
    than caption a diagnostic we can't calibrate for a reader, drop that column:
    the self-consistency claim ("same four albedo panels") stands on its own
    without it. Column geometry is exact, from grid_to_presentation() in that
    script: lw=363, PW=440, gap=7, so the 5th column starts at 363 + 4*447 = 2151.
    """
    im = Image.open(DECK / 'slide34_image80.png').convert('RGB')
    im = im.crop((0, 0, 2144, im.height))
    if im.width > max_width:
        im = im.resize((max_width, round(im.height * max_width / im.width)), Image.LANCZOS)
    im.save(out / 'ciai-teaser.jpg', quality=88, optimize=True)
    return im.size


def relabel_pairs(im: Image.Image, header_h: int = 44) -> Image.Image:
    """Repaint the pair figure's white header band with accurate labels.

    The deck labelled the two MID frames "cool" and "warm". MID bounces one white flash in
    25 directions; the frames differ mainly in light direction and intensity, with only a
    small bounce-colour change (median chromaticity gap 0.030). The header is plain white
    above the photographs (rows 0..44 at 1400 px width), so it is safe to overwrite.
    """
    im = im.copy()
    draw = ImageDraw.Draw(im)
    scale = im.width / 1400
    h = round(header_h * scale)
    draw.rectangle((0, 0, im.width, h), fill=(255, 255, 255))
    font = ImageFont.truetype(FONT_BOLD, max(12, round(22 * scale)))
    for cx, label in ((0.248, 'flash direction A'), (0.752, 'flash direction B')):
        draw.text((round(cx * im.width), h // 2), label, fill=(17, 24, 39), font=font, anchor='mm')
    return im


def build_pairs(out: Path, src: Path | None = None, max_width: int = 1400):
    """MID pair figure with corrected labels. `src` defaults to the deck export."""
    src = src or DECK / 'slide07_image27.jpg'
    im = Image.open(src).convert('RGB')
    if im.width > max_width:
        im = im.resize((max_width, round(im.height * max_width / im.width)), Image.LANCZOS)
    relabel_pairs(im).save(out / 'ciai-pairs.jpg', quality=88, optimize=True)
    return im.size


def build_mechanism(out: Path, tile_w: int = 300):
    """Training-time diagram: one MID pair, shared model, two albedos tied by L_inv and two
    shadings tied by L_expl. Thumbnails are our model's own outputs, so the two albedos
    agree only approximately -- in luminance more than in colour, which is the honest picture.
    """
    names = ['I1', 'I2', 'A1', 'A2', 'S1', 'S2']
    tiles = {}
    for n in names:
        path = THESIS / 'arch' / f'ciai_{n}.png'
        if not path.exists():
            raise SystemExit(f'missing thumbnail: {path}')
        t = Image.open(path).convert('RGB')
        tiles[n] = t.resize((tile_w, round(t.height * tile_w / t.width)), Image.LANCZOS)
    th = tiles['I1'].height

    pad, gap_y, model_w, col_gap = 30, 70, 190, 70
    top = 50
    # extra right margin so the L_expl annotation beside the last column is not clipped
    W = pad * 2 + tile_w * 3 + model_w + col_gap * 3 + 240
    H = top + th * 2 + gap_y + 60
    c = Image.new('RGB', (W, H), (255, 255, 255))
    d = ImageDraw.Draw(c)
    f = ImageFont.truetype(FONT, 19)
    fb = ImageFont.truetype(FONT_BOLD, 20)
    ink, blue, red = (17, 24, 39), (29, 78, 216), (153, 27, 27)

    x_in = pad
    x_model = x_in + tile_w + col_gap
    x_alb = x_model + model_w + col_gap
    x_sh = x_alb + tile_w + col_gap
    y1, y2 = top, top + th + gap_y

    c.paste(tiles['I1'], (x_in, y1)); c.paste(tiles['I2'], (x_in, y2))
    c.paste(tiles['A1'], (x_alb, y1)); c.paste(tiles['A2'], (x_alb, y2))
    c.paste(tiles['S1'], (x_sh, y1)); c.paste(tiles['S2'], (x_sh, y2))
    for x in (x_in, x_alb, x_sh):
        for y in (y1, y2):
            d.rectangle((x - 1, y - 1, x + tile_w, y + th), outline=(203, 213, 225), width=2)

    d.text((x_in + tile_w // 2, y1 - 22), 'Photo, flash direction A', fill=ink, font=fb, anchor='mm')
    d.text((x_in + tile_w // 2, y2 + th + 22), 'Photo, flash direction B', fill=ink, font=fb, anchor='mm')
    d.text((x_alb + tile_w // 2, y1 - 22), 'Albedo A1', fill=blue, font=fb, anchor='mm')
    d.text((x_alb + tile_w // 2, y2 + th + 22), 'Albedo A2', fill=blue, font=fb, anchor='mm')
    d.text((x_sh + tile_w // 2, y1 - 22), 'Shading S1', fill=ink, font=fb, anchor='mm')
    d.text((x_sh + tile_w // 2, y2 + th + 22), 'Shading S2', fill=ink, font=fb, anchor='mm')

    my0, my1 = y1 + th // 3, y2 + 2 * th // 3
    d.rounded_rectangle((x_model, my0, x_model + model_w, my1), radius=14,
                        fill=(234, 243, 255), outline=blue, width=3)
    cy = (my0 + my1) // 2
    d.text((x_model + model_w // 2, cy - 30), 'Shared model', fill=ink, font=fb, anchor='mm')
    d.text((x_model + model_w // 2, cy + 4), 'same weights', fill=(71, 85, 105), font=f, anchor='mm')
    d.text((x_model + model_w // 2, cy + 32), 'two passes', fill=(71, 85, 105), font=f, anchor='mm')

    def arrow(x0, y0, x1, y1_, color=(71, 85, 105), both=False):
        d.line((x0, y0, x1, y1_), fill=color, width=3)
        for (ax, ay, bx, by) in ([(x0, y0, x1, y1_)] + ([(x1, y1_, x0, y0)] if both else [])):
            v = np.array([bx - ax, by - ay], dtype=float)
            v /= np.linalg.norm(v) + 1e-9
            n = np.array([-v[1], v[0]])
            tip = np.array([bx, by])
            p1 = tip - 14 * v + 7 * n
            p2 = tip - 14 * v - 7 * n
            d.polygon([tuple(tip), tuple(p1), tuple(p2)], fill=color)

    arrow(x_in + tile_w + 8, y1 + th // 2, x_model - 8, y1 + th // 2)
    arrow(x_in + tile_w + 8, y2 + th // 2, x_model - 8, y2 + th // 2)
    arrow(x_model + model_w + 8, y1 + th // 2, x_alb - 8, y1 + th // 2)
    arrow(x_model + model_w + 8, y2 + th // 2, x_alb - 8, y2 + th // 2)

    mid_y = (y1 + th + y2) // 2
    arrow(x_alb + tile_w // 2, y1 + th + 8, x_alb + tile_w // 2, y2 - 8, color=red, both=True)
    d.text((x_alb + tile_w // 2 + 14, mid_y), 'L_inv: A1 = A2', fill=red, font=fb, anchor='lm')
    arrow(x_sh + tile_w // 2, y1 + th + 8, x_sh + tile_w // 2, y2 - 8, color=red, both=True)
    d.text((x_sh + tile_w // 2 + 14, mid_y - 12), 'L_expl:', fill=red, font=fb, anchor='lm')
    d.text((x_sh + tile_w // 2 + 14, mid_y + 14), 'lum(S1/S2) = lum(I1/I2)', fill=red, font=f, anchor='lm')

    c.save(out / 'ciai-mechanism.jpg', quality=90, optimize=True)
    return c.size






def build_thumb(out: Path, width: int = 700, scene: str = 'everett_lobby3',
                box=(0.0, 0.05, 1.0, 0.75)):
    """Card thumbnail for the portfolio (.research-thumb is 232x142, ~1.63:1).

    One MID test scene, split diagonally: the input photograph on the right and the albedo the
    model recovers from that same photograph on the left. Both halves are the same crop of the
    same frame, so the split shows the decomposition itself and makes no claim about illuminant
    colour. Source images are the compare-widget outputs (docs/static/img/compare), so no model
    inference is needed; `box` is the crop as fractions (x0, y0, x1, y1), chosen to avoid the
    probe spheres.
    """
    cdir = ROOT / 'docs/static/img/compare'
    inp = Image.open(cdir / f'{scene}_input.jpg').convert('RGB')
    alb = Image.open(cdir / f'{scene}_ours.jpg').convert('RGB')
    if inp.size != alb.size:
        alb = alb.resize(inp.size, Image.LANCZOS)
    W, H = inp.size
    x0, y0, x1, y1 = round(box[0] * W), round(box[1] * H), round(box[2] * W), round(box[3] * H)
    card_ratio = 232 / 142
    cw, ch = x1 - x0, y1 - y0
    if cw / ch > card_ratio:                       # too wide: trim sides
        nw = round(ch * card_ratio); x0 += (cw - nw) // 2; x1 = x0 + nw
    else:                                          # too tall: trim top and bottom
        nh = round(cw / card_ratio); y0 += (ch - nh) // 2; y1 = y0 + nh
    size = (width, round(width / card_ratio))
    top = inp.crop((x0, y0, x1, y1)).resize(size, Image.LANCZOS)
    bottom = alb.crop((x0, y0, x1, y1)).resize(size, Image.LANCZOS)

    # Albedo on the upper-left triangle, input on the lower-right, split by a white diagonal.
    mask = Image.new('L', size, 0)
    md = ImageDraw.Draw(mask)
    w, h = size
    md.polygon([(0, 0), (round(w * 0.62), 0), (round(w * 0.38), h), (0, h)], fill=255)
    im = Image.composite(bottom, top, mask)
    d = ImageDraw.Draw(im)
    d.line([(round(w * 0.62), 0), (round(w * 0.38), h)], fill=(255, 255, 255), width=4)
    font = ImageFont.truetype(FONT_BOLD, 26)
    for (x, y, label, anchor) in ((18, 16, 'Recovered albedo', 'la'),
                                  (w - 18, h - 16, 'Input photograph', 'rd')):
        box_ = d.textbbox((x, y), label, font=font, anchor=anchor)
        d.rounded_rectangle((box_[0] - 10, box_[1] - 7, box_[2] + 10, box_[3] + 7),
                            radius=8, fill=(17, 24, 39))
        d.text((x, y), label, fill=(255, 255, 255), font=font, anchor=anchor)
    im.save(out / 'ciai-thumb.jpg', quality=90, optimize=True)
    return im.size




def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=THESIS / 'web',
                    help='output directory (point at docs/static/img to publish the page)')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    missing = [p for p, _, _ in COPIES if not p.exists()]
    missing += [p for p in (DECK / 'slide34_image80.png', DECK / 'slide07_image27.jpg')
                if not p.exists()]
    if missing:
        raise SystemExit('missing source figures:\n  ' + '\n  '.join(str(p) for p in missing))

    print(f'  {"ciai-teaser.jpg":26s} {build_teaser(args.out)}')
    for src, name, width in COPIES:
        print(f'  {name:26s} {_resize(src, args.out / name, width)}')
    print(f'  {"ciai-pairs.jpg":26s} {build_pairs(args.out)}')
    print(f'  {"ciai-mechanism.jpg":26s} {build_mechanism(args.out)}')
    print(f'  {"ciai-thumb.jpg":26s} {build_thumb(args.out)}')
    print(f'\nwrote {len(COPIES) + 4} figures to {args.out}')


if __name__ == '__main__':
    main()
