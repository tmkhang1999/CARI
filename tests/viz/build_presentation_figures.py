#!/usr/bin/env python3
"""Figures reused from the defence presentation, plus one crop of the project-page teaser.

Only slide images that contain no prediction of our own model are taken from the deck: the
deck's "Ours" panels came from later checkpoints (one of them is visibly over-smoothed), so they
are not reused. The one figure that shows our model is cropped from the page teaser, which
was rendered with a later checkpoint of the same model.

Outputs:
  docs/static/img/ciai-metrics.jpg            the three MID metric diagrams (slides 24-25), one row
  docs/static/img/ciai-baselines.jpg          CRefNet and Marigold-App under two lights (slide 6)
  docs/static/img/ciai-limit-reflective.jpg   mirror sphere: input and our albedo, four lights
  documents/thesis/images/limits/reflective.jpg             same crop as the page

Run:
  python tests/viz/build_presentation_figures.py --pptx Khang_presentation.pptx
"""
from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / 'docs/static/img'
THESIS = ROOT / 'documents/thesis/images'
FONT = font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans'))
FONT_BOLD = font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans', weight='bold'))
INK, MUTE = (17, 24, 39), (92, 100, 112)

# Deck media names (ppt/media/) and what they show. None contains a prediction of our model.
METRIC_CARDS = [
    ('image68.png', 'C_mat', 'lightness stability'),
    ('image69.png', 'Cast_rel', 'hue stability'),
    ('image70.png', 'Chroma_err', 'colour accuracy'),
]
# The deck's fourth card (image71, chroma spread) is left out on purpose: the spread ratio is
# not one of the reported metrics.
BASELINES = 'image26.jpg'

# Teaser layout (docs/static/img/ciai-teaser.jpg, 1700 x 497): four panels per row.
TEASER_COLS = [(287, 637), (642, 992), (996, 1346), (1351, 1700)]
TEASER_ROWS = [(26, 260), (264, 497)]
SPHERE = (0.0, 0.62, 0.30, 1.0)   # mirror sphere, as fractions of a panel (x0, y0, x1, y1)


def read_media(pptx: Path, name: str) -> Image.Image:
    with zipfile.ZipFile(pptx) as z:
        try:
            data = z.read(f'ppt/media/{name}')
        except KeyError:
            raise SystemExit(f'{name} is not in {pptx}; the deck may have been re-saved')
    return Image.open(io.BytesIO(data)).convert('RGB')


def save(im: Image.Image, dst: Path, max_width: int | None = None, quality: int = 90):
    if max_width and im.width > max_width:
        im = im.resize((max_width, round(im.height * max_width / im.width)), Image.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, quality=quality, optimize=True)
    print(f'  {dst.relative_to(ROOT)}  {im.size}')


def build_metrics(pptx: Path):
    cards = [read_media(pptx, n) for n, _, _ in METRIC_CARDS]
    w = 760
    cards = [c.resize((w, round(c.height * w / c.width)), Image.LANCZOS) for c in cards]
    head, gap = 54, 30
    cell_h = head + max(c.height for c in cards)
    n = len(cards)
    canvas = Image.new('RGB', (n * w + (n - 1) * gap, cell_h), 'white')
    d = ImageDraw.Draw(canvas)
    fb, fr = ImageFont.truetype(FONT_BOLD, 26), ImageFont.truetype(FONT, 22)
    for k, (c, (_, name, role)) in enumerate(zip(cards, METRIC_CARDS)):
        x, y = k * (w + gap), 0
        d.text((x + 12, y + head // 2), name, fill=INK, font=fb, anchor='lm')
        d.text((x + 12 + d.textlength(name, font=fb) + 14, y + head // 2), role,
               fill=MUTE, font=fr, anchor='lm')
        canvas.paste(c, (x, y + head))
    save(canvas, WEB / 'ciai-metrics.jpg', max_width=1600)


def build_baselines(pptx: Path):
    save(read_media(pptx, BASELINES), WEB / 'ciai-baselines.jpg', max_width=1100)


def build_reflective():
    src = WEB / 'ciai-teaser.jpg'
    if not src.exists():
        raise SystemExit(f'missing {src}; build it with tests/viz/build_web_figures.py')
    teaser = Image.open(src).convert('RGB')
    crops = []
    for y0, y1 in TEASER_ROWS:
        row = []
        for x0, x1 in TEASER_COLS:
            w, h = x1 - x0, y1 - y0
            fx0, fy0, fx1, fy1 = SPHERE
            row.append(teaser.crop((x0 + round(fx0 * w), y0 + round(fy0 * h),
                                    x0 + round(fx1 * w), y0 + round(fy1 * h))))
        crops.append(row)
    scale = 2   # the crops are small; upsample so the page shows them at a readable size
    cw, ch = crops[0][0].width * scale, crops[0][0].height * scale
    lab, head, gap = 150, 36, 6
    canvas = Image.new('RGB', (lab + 4 * cw + 3 * gap, head + 2 * ch + gap), 'white')
    d = ImageDraw.Draw(canvas)
    fb, fr = ImageFont.truetype(FONT_BOLD, 18), ImageFont.truetype(FONT, 17)
    for j, name in enumerate(('light 0', 'light 6', 'light 12', 'light 18')):
        d.text((lab + j * (cw + gap) + cw // 2, head // 2), name, fill=INK, font=fr, anchor='mm')
    for i, (name, row) in enumerate(zip(('Input', 'Our albedo'), crops)):
        y = head + i * (ch + gap)
        d.text((lab - 12, y + ch // 2), name, fill=INK, font=fb, anchor='rm')
        for j, c in enumerate(row):
            canvas.paste(c.resize((cw, ch), Image.LANCZOS), (lab + j * (cw + gap), y))
    save(canvas, WEB / 'ciai-limit-reflective.jpg')
    save(canvas, THESIS / 'limits/reflective.jpg')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pptx', type=Path, default=ROOT / 'Khang_presentation.pptx',
                    help='the defence deck (not tracked in git)')
    args = ap.parse_args()
    if not args.pptx.exists():
        raise SystemExit(f'deck not found: {args.pptx}')
    build_metrics(args.pptx)
    build_baselines(args.pptx)
    build_reflective()


if __name__ == '__main__':
    main()
