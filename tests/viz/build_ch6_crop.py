#!/usr/bin/env python3
"""Build Figure 6.1 by cropping the saved diagnostic sheet.

No model is loaded and no inference is run. The pixel boxes correspond to the
four image bodies in outputs/wild_inference.png at its native 2700x1800 size.
"""
import os

from PIL import Image, ImageDraw, ImageFont


ROOT = '/home/khang/IR-IID'
SRC = f'{ROOT}/outputs/wild_inference.png'
OUT = f'{ROOT}/tests/visualizations/ch6/decomposition.jpg'
DST = f'{ROOT}/documents/thesis/images/ch6/decomposition.jpg'
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONTB = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'

PANELS = [
    ('Input  I', (63, 82, 447, 624)),
    ('Ours: albedo  A_d', (1143, 667, 1527, 1209)),
    ('Ours: diffuse shading  S_d', (1683, 667, 2067, 1209)),
    ('Ours: residual  R', (2223, 1252, 2607, 1794)),
]


def font(size, bold=False):
    try:
        return ImageFont.truetype(FONTB if bold else FONT, size)
    except Exception:
        return ImageFont.load_default()


def main():
    src = Image.open(SRC).convert('RGB')
    panel_w = 620
    native_w = PANELS[0][1][2] - PANELS[0][1][0]
    native_h = PANELS[0][1][3] - PANELS[0][1][1]
    panel_h = round(panel_w * native_h / native_w)
    gap, head = 10, 88
    canvas = Image.new('RGB', (4 * panel_w + 3 * gap, head + panel_h), 'white')
    draw = ImageDraw.Draw(canvas)

    for i, (label, box) in enumerate(PANELS):
        x = i * (panel_w + gap)
        crop = src.crop(box).resize((panel_w, panel_h), Image.Resampling.LANCZOS)
        canvas.paste(crop, (x, head))
        draw.text((x + panel_w // 2, head // 2), label, anchor='mm',
                  fill=(25, 25, 25), font=font(40, bold=i > 0))

    for path in (OUT, DST):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        canvas.save(path, quality=95, subsampling=0)
        print('wrote', path, canvas.size)


if __name__ == '__main__':
    main()
