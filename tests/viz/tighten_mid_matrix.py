#!/usr/bin/env python3
"""Post-process the MID ablation matrix composites: crop each of the 25 cells
to the colourful-object cluster (drop the two neutral calibration spheres and
most of the bare table), so the colour-constancy signal isn't diluted by
achromatic pixels. Pure post-processing on the already-rendered composite --
no re-inference, since the underlying predictions don't change.
"""
from PIL import Image, ImageDraw, ImageFont

FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONTB = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'

ROWLAB_W, GAP, HEAD, PW, PH = 130, 8, 50, 520, 346
NCOL, NROW = 5, 5
CROP = (90, 0, 470, 270)  # x0,y0,x1,y1 relative to each cell


def fnt(sz, bold=False):
    return ImageFont.truetype(FONTB if bold else FONT, sz)


def process(scene):
    src_path = f'/home/khang/IR-IID/presentation/assets/generated/rev2_matrices/mid_ablation_{scene}.jpg'
    src = Image.open(src_path)
    cw, ch = CROP[2] - CROP[0], CROP[3] - CROP[1]

    col_labels = ['Input', 'Row 1: neither', 'Row 2: CARI only', 'Row 3: colour path only', 'Row 4: full CARI']
    row_labels = ['light 0', 'light 6', 'light 12', 'light 18']

    head2, rowlab2 = 56, 150
    W = rowlab2 + NCOL * cw + (NCOL - 1) * GAP
    H = head2 + NROW * ch + (NROW - 1) * GAP
    out = Image.new('RGB', (W, H), (255, 255, 255))
    d = ImageDraw.Draw(out)

    for c in range(NCOL):
        x_src = ROWLAB_W + c * (PW + GAP)
        d.text((rowlab2 + c * (cw + GAP) + cw // 2, head2 // 2), col_labels[c],
               fill=(23, 32, 51), anchor='mm', font=fnt(20, True))
        for r in range(NROW):
            y_src = HEAD + r * (PH + GAP)
            cell = src.crop((x_src + CROP[0], y_src + CROP[1], x_src + CROP[2], y_src + CROP[3]))
            x_dst = rowlab2 + c * (cw + GAP)
            y_dst = head2 + r * (ch + GAP)
            out.paste(cell, (x_dst, y_dst))
    for r in range(4):
        y_dst = head2 + r * (ch + GAP)
        d.text((rowlab2 // 2, y_dst + ch // 2), row_labels[r], fill=(23, 32, 51), anchor='mm', font=fnt(19, True))
    y_dst = head2 + 4 * (ch + GAP)
    d.text((rowlab2 // 2, y_dst + ch // 2), 'variation\n(4 lights)', fill=(196, 61, 61), anchor='mm', font=fnt(16, True))

    out_path = f'/home/khang/IR-IID/presentation/assets/generated/rev2_matrices/mid_ablation_{scene}_tight.jpg'
    out.save(out_path, quality=93)
    print('wrote', out_path, out.size)


if __name__ == '__main__':
    import sys
    scenes = sys.argv[1:] or ['everett_dining1', 'everett_kitchen5']
    for s in scenes:
        process(s)
