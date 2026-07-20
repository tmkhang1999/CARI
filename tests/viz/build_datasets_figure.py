#!/usr/bin/env python3
"""Figure 5.2 (fig:datasets): the four benchmarks, in the order the chapter presents them.

Two fixes over the previous version:
  * Row order is MID, IIW, MAW, ARAP -- matching Chapter 5. The old figure ran
    MID, MAW, ARAP, IIW, which contradicted the text.
  * The IIW row is no longer a single lone photograph with a large white gap beside it.
    IIW genuinely has no albedo ground truth -- only sparse pairwise human lightness
    judgements -- so instead of leaving the row visually empty we OVERLAY those actual
    judgements: each comparison is drawn as two dots joined by a line, with the point the
    annotators called darker marked. That both fills the row and shows the reader exactly
    what WHDR is scored from, which is the argument of Sect. 5.3.
"""
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = '/home/khang/IR-IID'
OUT = f'{ROOT}/tests/visualizations/datasets'
DST = f'{ROOT}/documents/thesis/images/datasets'
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONTB = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
RED = (200, 30, 30)
CW, CH = 600, 452


def fnt(sz, bold=False):
    try:
        return ImageFont.truetype(FONTB if bold else FONT, sz)
    except Exception:
        return ImageFont.load_default()


def tone(x, pct=99.0):
    v = x[x > 1e-6]
    s = float(np.percentile(v, pct)) if v.size else 1.0
    return np.clip(x / (s + 1e-8), 0, 1) ** (1 / 2.2)


def load_hdr(p):
    a = cv2.imread(p, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    return tone(a[..., ::-1].astype(np.float32))


def load_png(p):
    return cv2.imread(p, cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32) / 255.0


def pil(a):
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))


def fit(im, w=CW, h=CH):
    return im.resize((w, h), Image.LANCZOS)


def panel(img, cap, gt=False):
    p = fit(img)
    if gt:
        d = ImageDraw.Draw(p)
        d.rectangle([0, 0, p.size[0] - 1, p.size[1] - 1], outline=RED, width=4)
    lab = Image.new('RGB', (CW, 40), (255, 255, 255))
    d = ImageDraw.Draw(lab)
    f = fnt(22)
    w = d.textbbox((0, 0), cap, font=f)[2]
    d.text(((CW - w) // 2, 4), cap, font=f, fill=(RED if gt else (70, 70, 70)))
    out = Image.new('RGB', (CW, p.size[1] + 41), (255, 255, 255))
    out.paste(p, (0, 0))
    out.paste(lab, (0, p.size[1] + 1))
    return out


def iiw_judgements(img_id='1094', max_pairs=14):
    """Draw IIW's actual sparse human judgements on the photograph: what WHDR scores."""
    base = f'{ROOT}/tests/testing_data/iiw-dataset/data'
    img = load_png(f'{base}/{img_id}.png')
    meta = json.load(open(f'{base}/{img_id}.json'))
    pts = {p['id']: p for p in meta['intrinsic_points']}
    H, W = img.shape[:2]
    im = pil(img)
    d = ImageDraw.Draw(im)
    r = max(3, int(0.006 * max(H, W)))
    n = 0
    for c in meta['intrinsic_comparisons']:
        if n >= max_pairs or c.get('darker') not in ('1', '2'):
            continue
        p1, p2 = pts.get(c['point1']), pts.get(c['point2'])
        if not p1 or not p2:
            continue
        a = (p1['x'] * W, p1['y'] * H)
        b = (p2['x'] * W, p2['y'] * H)
        d.line([a, b], fill=(255, 215, 0), width=max(2, r // 2))
        dark, light = (a, b) if c['darker'] == '1' else (b, a)
        d.ellipse([light[0] - r, light[1] - r, light[0] + r, light[1] + r],
                  fill=(255, 255, 255), outline=(20, 20, 20), width=2)
        d.ellipse([dark[0] - r, dark[1] - r, dark[0] + r, dark[1] + r],
                  fill=(25, 25, 25), outline=(255, 255, 255), width=2)
        n += 1
    return np.asarray(im).astype(np.float32) / 255.0


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(DST, exist_ok=True)
    rows = []

    # ---- 1. MID (chapter order: first) ----
    mid = '/home/khang/datasets/MIDIntrinsics/test/everett_kitchen12'
    rows.append(('MID — real, multi-illuminant capture of one scene; no measured albedo '
                 '(GT-free constancy diagnostic)',
                 [panel(pil(load_hdr(f'{mid}/dir_0_mip2.exr')), 'input, light A'),
                  panel(pil(load_hdr(f'{mid}/dir_18_mip2.exr')), 'input, light B'),
                  panel(pil(load_hdr(f'{mid}/albedo.exr')), 'pseudo-GT albedo', gt=True)]))

    # ---- 2. IIW (chapter order: second) — show the judgements, not a blank row ----
    rows.append(('IIW — real photographs; NO albedo ground truth, only sparse pairwise '
                 'human lightness judgements (this is all WHDR scores)',
                 [panel(pil(load_png(f'{ROOT}/tests/testing_data/iiw-dataset/data/1094.png')),
                        'input photograph'),
                  panel(pil(iiw_judgements('1094')),
                        'the annotation: darker (●) vs lighter (○)', gt=True)]))

    # ---- 3. MAW ----
    b = f'{ROOT}/tests/testing_data/MAW'
    rows.append(('MAW — real photographs; physically measured albedo on masked patches',
                 [panel(pil(load_png(f'{b}/images_png/scene_28/_DSC2770.png')),
                        'input photograph'),
                  panel(pil(load_png(f'{b}/labels/new_masks/scene_28/_DSC2770_albedo.png')),
                        'measured albedo (masked)', gt=True)]))

    # ---- 4. ARAP ----
    arap = f'{ROOT}/tests/testing_data/ARAP_dataset'
    rows.append(('ARAP — synthetic renders; dense ground-truth albedo',
                 [panel(pil(load_hdr(f'{arap}/bedroom.hdr')), 'rendered input'),
                  panel(pil(load_hdr(f'{arap}/bedroom_albedo.hdr')), 'GT albedo', gt=True)]))

    # Canvas must fit the WIDEST of (panel strip, title text) or the titles get clipped.
    tf = fnt(30, bold=True)
    probe = ImageDraw.Draw(Image.new('RGB', (10, 10)))
    W = max(
        max(sum(p.size[0] for p in c) + 8 * (len(c) - 1) for _, c in rows),
        max(probe.textbbox((0, 0), t, font=tf)[2] for t, _ in rows) + 24,
    )
    built = []
    for title, cells in rows:
        bar = Image.new('RGB', (W, 48), (244, 244, 247))
        d = ImageDraw.Draw(bar)
        d.text((14, 10), title, font=tf, fill=(20, 20, 20))
        strip = Image.new('RGB', (W, cells[0].size[1]), (255, 255, 255))
        x = 0
        for c in cells:
            strip.paste(c, (x, 0))
            x += c.size[0] + 8
        blk = Image.new('RGB', (W, 48 + 4 + strip.size[1]), (255, 255, 255))
        blk.paste(bar, (0, 0))
        blk.paste(strip, (0, 52))
        built.append(blk)

    Ht = sum(b.size[1] for b in built) + 18 * (len(built) - 1)
    canvas = Image.new('RGB', (W, Ht), (255, 255, 255))
    y = 0
    for b in built:
        canvas.paste(b, (0, y))
        y += b.size[1] + 18
    for p in (f'{OUT}/datasets.jpg', f'{DST}/datasets.jpg'):
        canvas.save(p, quality=95)
        print('wrote', p, canvas.size)


if __name__ == '__main__':
    main()
