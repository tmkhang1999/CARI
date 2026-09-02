#!/usr/bin/env python3
"""Three-panel overview for Chapter 4's Training Data intro (sec:data): one
representative frame per dataset, labelled with the role Sec 4.2's own prose
already assigns it -- Hypersim and InteriorVerse supply supervised albedo,
MID supplies the real cross-render pairs CARI trains on.

Source frames (presentation/assets/generated/rev2_matrices/dataset_*.jpg) are
curated representative renders/photographs, one per dataset, not derived data;
this script's job is only the composite layout, which is where the "traces to
a script" guarantee actually applies here.

Run:
  python tests/viz/build_dataset_roles_figure.py
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'presentation/assets/generated/rev2_matrices'
OUT = ROOT / 'documents/thesis/images/data/dataset_roles.jpg'

FONTB = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

PANELS = [
    ('dataset_hypersim.jpg', 'Hypersim', 'supervised albedo + shading'),
    ('dataset_interiorverse.jpg', 'InteriorVerse', 'supervised albedo, added diversity'),
    ('dataset_mid.jpg', 'MID', 'real cross-render pairs -- the CARI signal'),
]


def main():
    PW, PH, gap, head = 520, 400, 14, 46
    ims = []
    for f, _, _ in PANELS:
        im = Image.open(SRC / f).convert('RGB')
        s = max(PW / im.width, PH / im.height)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
        x0 = (im.width - PW) // 2
        y0 = (im.height - PH) // 2
        ims.append(im.crop((x0, y0, x0 + PW, y0 + PH)))

    W = len(PANELS) * PW + (len(PANELS) - 1) * gap
    H = head + PH + 34
    canvas = Image.new('RGB', (W, H), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    fb, fr = ImageFont.truetype(FONTB, 24), ImageFont.truetype(FONT, 18)

    x = 0
    for im, (_, name, role) in zip(ims, PANELS):
        d.text((x + PW // 2, head // 2), name, fill=(23, 32, 51), anchor='mm', font=fb)
        canvas.paste(im, (x, head))
        d.rectangle([x, head, x + PW - 1, head + PH - 1], outline=(217, 224, 232), width=2)
        d.text((x + PW // 2, head + PH + 17), role, fill=(71, 84, 103), anchor='mm', font=fr)
        x += PW + gap

    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT, quality=95)
    print(f'wrote {OUT}  {canvas.size}')


if __name__ == '__main__':
    main()
