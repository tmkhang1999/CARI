#!/usr/bin/env python3
"""Compose the ARAP hard-shadow figure from the full-resolution bedroom panels.

The inset is deliberately centered on the window-shadow bands across the bed,
rather than on a material-texture detail.
"""
import numpy as np
from PIL import Image, ImageDraw
import sys
sys.path.insert(0, '/home/khang/IR-IID/tests/viz')
from build_hires_figures import fnt, INK, RED, MUTE  # noqa: E402

OUT_DIR = '/home/khang/IR-IID/presentation/assets/generated/mid_diversity'
# Source panels are 800x600. This square crop centers the clearly visible
# window-shadow bands across the duvet, with no padding outside the image.
CROP = (190, 180, 610, 600)

names = ['input', 'gt', 'ours']
labels = ['Input', 'Dense GT albedo', 'Ours']
srcs = [Image.open(f'{OUT_DIR}/arap_bedroom_{n}_full.png') for n in names]

PW = 560
PH = PW  # square source
gap = 16
head = 44
zoom_head = 40

W = 3 * PW + 2 * gap
H = head + PH + 24 + zoom_head + PW

canvas = Image.new('RGB', (W, H), (255, 255, 255))
d = ImageDraw.Draw(canvas)

CARI_BLUE = (22, 119, 255)

for i, (src, label) in enumerate(zip(srcs, labels)):
    x = i * (PW + gap)
    d.text((x + PW // 2, head // 2), label, anchor='mm',
           font=fnt(26, bold=(i == 2)), fill=(RED if i == 1 else INK))
    thumb = src.resize((PW, PH), Image.LANCZOS)
    canvas.paste(thumb, (x, head))
    # Highlight the cast-shadow bands, scaled from the full-resolution panel.
    sx0, sy0, sx1, sy1 = CROP
    scale = PW / src.width
    bx0, by0, bx1, by1 = sx0 * scale, sy0 * scale, sx1 * scale, sy1 * scale
    d.rectangle([x + bx0, head + by0, x + bx1, head + by1], outline=CARI_BLUE, width=4)

zy = head + PH + 24
d.text((W // 2, zy + zoom_head // 2), 'Magnified: window-shadow bands across the bed, which should be explained by shading, not material colour',
       anchor='mm', font=fnt(20, bold=True), fill=MUTE)

zpw = PW
for i, src in enumerate(srcs):
    x = i * (PW + gap)
    crop = src.crop(CROP).resize((zpw, zpw), Image.LANCZOS)
    canvas.paste(crop, (x, zy + zoom_head))
    d.rectangle([x, zy + zoom_head, x + zpw - 1, zy + zoom_head + zpw - 1], outline=CARI_BLUE, width=3)

out = f'{OUT_DIR}/arap_hardshadow_final.jpg'
canvas.save(out, quality=95, subsampling=0)
print('wrote', out, canvas.size)
