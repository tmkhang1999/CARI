#!/usr/bin/env python3
"""Ours vs. every external baseline we run locally, several real lighting samples
of one MID scene per figure. Rows = lights, columns = methods.

Originally built because no deck figure put all the external methods in one
qualitative panel — only CRefNet + Marigold-App appeared visually (Slides 7, 32)
while Marigold-Light and Ordinal Shading lived only in the tables. CD-IID and
RGB->X were added 2026-09-02 for the same reason and a sharper one: CD-IID beats
us on both MID colour metrics, so a page that reports that and shows no picture
of its albedo is hiding the comparison a reader most wants to make.

Usage:  python tests/viz/build_all_models_light_matrix.py --scene everett_dining1 --lights 0 9 18
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = '/home/khang/IR-IID'
sys.path.insert(0, os.path.join(ROOT, 'tests/eval'))
os.chdir(os.path.join(ROOT, 'tests/eval'))

from eval_mid_constancy import AlbedoPredictor, _raw_frame, _tonemap_frame  # noqa: E402

MID = '/home/khang/datasets/MIDIntrinsics/test'

# label, ckpt_path, version -- "Ours" pinned to the deck's qualitative-gallery
# checkpoint convention (v17_29/checkpoint_iter_60000.pth).
MODELS = [
    ('Ours', f'{ROOT}/checkpoints/v17_29/checkpoint_iter_60000.pth', '17'),
    ('CRefNet', f'{ROOT}/checkpoints/CRefNet/final_real.pt', 'crefnet'),
    ('Marigold-App', f'{ROOT}/checkpoints/marigold-iid-appearance-v1-1', 'marigold-appearance'),
    ('Marigold-Light', f'{ROOT}/checkpoints/marigold-iid-lighting-v1-1', 'marigold-lighting'),
    ('Ordinal Shading', '', 'ordinal'),
    ('CD-IID', '', 'cdiid'),
    ('RGB->X', '', 'rgbx'),
]

FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONTB = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def _fnt(sz, bold=False):
    try:
        return ImageFont.truetype(FONTB if bold else FONT, sz)
    except Exception:
        return ImageFont.load_default()


def _norm_albedo(a, pct=99.5):
    s = np.percentile(a[a > 1e-6], pct) if (a > 1e-6).any() else 1.0
    return np.clip(a / (s + 1e-8), 0, 1)


def _srgb(x):
    return np.clip(x, 0, 1) ** (1 / 2.2)


def _panel(arr, w, h):
    im = Image.fromarray((_srgb(arr) * 255).astype(np.uint8))
    return im.resize((w, h), Image.LANCZOS)


# Fixed camera, so one set of boxes (in the panel's own w x h coordinate frame)
# lands correctly on every column and every light row. Hand-placed on the
# everett_lobby3 scene by inspection (see scratchpad boxes_row0b.jpg during
# development) -- not auto-detected, because the point is two specific
# objects: a non-diffuse calibration sphere and a large flat dark garment,
# not "the most colourful region" find_object_box() in the sibling ablation
# script would pick.
TRACK_BOXES = {
    'everett_lobby3': [
        ((150, 190, 195, 235), (200, 30, 30), 'mirror sphere'),
        ((245, 8, 352, 232), (30, 90, 220), 'dark jacket'),
    ],
}


def _draw_track_boxes(d, scene, x0, y0, pw, ph):
    for (bx1, by1, bx2, by2), colour, _label in TRACK_BOXES.get(scene, []):
        sx, sy = pw / 360, ph / 240  # boxes were placed at panel-w=360
        d.rectangle([x0 + bx1 * sx, y0 + by1 * sy, x0 + bx2 * sx, y0 + by2 * sy],
                    outline=colour, width=max(2, pw // 120))


def build_crop_comparison(scene, gt, preds, lights, out_dir, max_cell=200):
    """Second figure: each tracked region cropped from the FULL-RESOLUTION source
    (not re-cropped from the small, box-annotated matrix panels) and shown as a
    method x light grid, at a size where the thing the box was drawn to show is
    actually legible -- the box in the main matrix is ~45px on a page that
    displays it around 0.5x, well below where the sphere's reflection detail or
    the jacket's hue survive.

    Cell size is capped on its LONG side (max_cell), not fixed on width: the
    sphere box is square and the jacket box is portrait (ar~2), and a fixed
    width blew the jacket cells out to 355px tall, stacking two sections to an
    unusable 668x3860 image. Capping instead gives both sections nearly the
    same cell height, so they sit SIDE BY SIDE (not stacked) and the whole
    figure comes out close to landscape.
    """
    boxes = TRACK_BOXES.get(scene, [])
    if not boxes:
        return None
    H, W = gt.shape[:2]
    methods = [m[0] for m in MODELS]
    lab_px, hdr_px = 20, 24

    # Row-label column sized to the actual text, not a guessed constant --
    # 'Marigold-Light' / 'Ordinal Shading' were clipped at a fixed 150px.
    probe = ImageDraw.Draw(Image.new('RGB', (8, 8)))
    row_lab_w = max(probe.textbbox((0, 0), m, font=_fnt(lab_px, bold=True))[2]
                    for m in methods) + 18

    sections = []
    for (bx1, by1, bx2, by2), colour, label in boxes:
        sx, sy = W / 360, H / 240
        fx1, fy1, fx2, fy2 = bx1 * sx, by1 * sy, bx2 * sx, by2 * sy
        ar = (fy2 - fy1) / (fx2 - fx1)
        if ar >= 1:
            ch, cw = max_cell, max(1, int(max_cell / ar))
        else:
            cw, ch = max_cell, max(1, int(max_cell * ar))

        ncol = len(lights)
        sec_w = row_lab_w + ncol * cw + (ncol - 1) * 4
        sec_h = hdr_px + 34 + len(methods) * (ch + 4) + 12
        sec = Image.new('RGB', (sec_w, sec_h), (255, 255, 255))
        sd = ImageDraw.Draw(sec)
        sd.text((6, 6), label, fill=colour, font=_fnt(hdr_px, bold=True))
        for c, light in enumerate(lights):
            x = row_lab_w + c * (cw + 4)
            sd.text((x + cw // 2, hdr_px + 20), f'light {light}', fill=(20, 20, 20),
                    anchor='mm', font=_fnt(lab_px, bold=True))
        y0 = hdr_px + 34
        for r, name in enumerate(methods):
            y = y0 + r * (ch + 4)
            sd.text((6, y + ch // 2), name, fill=(20, 20, 20), anchor='lm',
                    font=_fnt(lab_px, bold=(name == 'Ours')))
            for c, light in enumerate(lights):
                arr = _norm_albedo(preds[name][c])
                crop = arr[int(fy1):int(fy2), int(fx1):int(fx2)]
                im = _panel(crop, cw, ch)
                x = row_lab_w + c * (cw + 4)
                sec.paste(im, (x, y))
                sd.rectangle([x, y, x + cw - 1, y + ch - 1], outline=colour, width=3)
        sections.append(sec)

    gap = 30
    W_all = sum(s.width for s in sections) + gap * (len(sections) - 1)
    H_all = max(s.height for s in sections)
    canvas = Image.new('RGB', (W_all, H_all), (255, 255, 255))
    x = 0
    for sec in sections:
        canvas.paste(sec, (x, 0))
        x += sec.width + gap

    out = os.path.join(out_dir, f'all_models_{scene}_crops.jpg')
    canvas.save(out, quality=94)
    print(f'wrote {out}  ({canvas.size[0]}x{canvas.size[1]})')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', required=True)
    ap.add_argument('--lights', type=int, nargs='+', required=True)
    ap.add_argument('--panel-w', type=int, default=360)
    ap.add_argument('--infer', type=int, default=1280)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default=f'{ROOT}/presentation/assets/generated/all_models_light_matrix')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    sp = os.path.join(MID, args.scene)
    gt = _tonemap_frame(np.asarray(
        __import__('cv2').imread(os.path.join(sp, 'albedo.exr'),
                                  __import__('cv2').IMREAD_ANYCOLOR | __import__('cv2').IMREAD_ANYDEPTH)
    )[..., ::-1].copy().astype(np.float32))
    inputs = [_tonemap_frame(_raw_frame(sp, i)) for i in args.lights]
    H, W = gt.shape[:2]
    ar = H / W
    PW = args.panel_w
    PH = int(PW * ar)

    col_labels = ['Input', 'GT albedo'] + [m[0] for m in MODELS]
    preds = {}
    for label, path, ver in MODELS:
        p = AlbedoPredictor(path, ver, args.device, infer_max_size=args.infer)
        preds[label] = [p.albedo(x) for x in inputs]
        del p
        import torch
        torch.cuda.empty_cache()
        print(f'  [{args.scene}] predicted: {label}')

    rows = []
    for r, light in enumerate(args.lights):
        panels = [_panel(inputs[r], PW, PH), _panel(_norm_albedo(gt), PW, PH)]
        for label, _p, _v in MODELS:
            panels.append(_panel(_norm_albedo(preds[label][r]), PW, PH))
        rows.append((light, panels))

    ncol = len(col_labels)
    # Label sizes are proportional to the panel, not fixed: with 9 columns this
    # canvas is ~3400px wide and gets displayed at ~950-1200px on the web page,
    # so a fixed 24px label renders at 7px. Tie them to PW so the text keeps a
    # constant apparent size however many methods the roster grows to.
    lab_px = max(24, PW // 8)
    row_px = max(22, PW // 9)
    gap, head, rowlab_w, foot = 6, int(lab_px * 2.4), int(row_px * 5.0), 20
    Wtot = rowlab_w + ncol * PW + (ncol - 1) * gap
    Htot = head + len(rows) * (PH + gap) + foot
    canvas = Image.new('RGB', (Wtot, Htot), (255, 255, 255))
    d = ImageDraw.Draw(canvas)

    for c, lab in enumerate(col_labels):
        x = rowlab_w + c * (PW + gap)
        bold = lab in ('Ours', 'GT albedo')
        d.text((x + PW // 2, head // 2), lab, fill=(20, 20, 20), anchor='mm',
               font=_fnt(lab_px, bold=bold))

    for r, (light, panels) in enumerate(rows):
        y = head + r * (PH + gap)
        d.text((rowlab_w // 2, y + PH // 2), f'light {light}', fill=(20, 20, 20),
               anchor='mm', font=_fnt(row_px, bold=True))
        for c, pil in enumerate(panels):
            x = rowlab_w + c * (PW + gap)
            canvas.paste(pil, (x, y))
            if col_labels[c] == 'GT albedo':
                d.rectangle([x, y, x + PW - 1, y + PH - 1], outline=(200, 30, 30), width=3)
            _draw_track_boxes(d, args.scene, x, y, PW, PH)

    out = os.path.join(args.out, f'all_models_{args.scene}.jpg')
    canvas.save(out, quality=94)
    print(f'wrote {out}  ({canvas.size[0]}x{canvas.size[1]})')

    with open(os.path.join(args.out, f'all_models_{args.scene}_manifest.txt'), 'w') as f:
        f.write(f'scene: {args.scene}\nlights: {args.lights}\ninfer: {args.infer}\n')
        for label, path, ver in MODELS:
            f.write(f'{label}\t{path}\t{ver}\n')

    build_crop_comparison(args.scene, gt, preds, args.lights, args.out)


if __name__ == '__main__':
    main()
