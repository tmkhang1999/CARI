#!/usr/bin/env python3
"""High-resolution rebuild of the qualitative thesis figures.

The legacy figures (images/legacy_draft25/*) were composited from already-downsampled
tiles, so they are soft in print -- and iiw_ours.jpg was 860x30, effectively broken. They
were also built from OLD checkpoints, so "Ours" in them is not the model the thesis now
reports. This builder regenerates them from the current roster, at print resolution.

Design rules (kept consistent across every figure):
  * Panels are rendered at PW px wide; a 6-panel strip lands ~3.6k px, comfortably over
    300 dpi at full text width.
  * Ground truth always carries a red border; "Ours" is always bold in the header.
  * Albedo is scale-normalised for display PER PANEL, by that panel's own 99th percentile.
    This is required, not a convenience: albedo is defined only up to a global scale and
    every method picks its own, so a scale shared across methods simply renders whichever
    method predicts the dimmest absolute albedo as "dark" -- a display artefact, not a
    result. Per-panel scaling is also safe for the argument these figures make, because
    saturation and hue are invariant to a global scale: chroma collapse still shows.
  * Beyond that global scale, no colour correction of any kind: differences in saturation
    are differences in the predictions.

Figures:  comp_grid | mid_ours | iiw_ours | maw_ours | arap_ours |
          arap_model_grid | ablation_mid
Usage:    python tests/viz/build_hires_figures.py --figures all
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')

ROOT = '/home/khang/IR-IID'
sys.path.insert(0, os.path.join(ROOT, 'tests/eval'))
os.chdir(os.path.join(ROOT, 'tests/eval'))

from eval_mid_constancy import AlbedoPredictor, _raw_frame, _tonemap_frame  # noqa: E402

OUT = f'{ROOT}/tests/visualizations/hires'
DST = f'{ROOT}/documents/thesis/images/hires'
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONTB = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
RED, INK, MUTE = (196, 30, 36), (25, 25, 28), (95, 95, 102)

MID = '/home/khang/datasets/MIDIntrinsics/test'
IIW = f'{ROOT}/tests/testing_data/iiw-dataset/data'
MAW = f'{ROOT}/tests/testing_data/MAW'
ARAP = f'{ROOT}/tests/testing_data/ARAP_dataset'

CK = f'{ROOT}/checkpoints'
ROSTER = [
    ('Ours',           f'{CK}/v17_34/checkpoint_iter_60000.pth', '17'),
    ('Marigold-App',   f'{CK}/marigold-iid-appearance-v1-1', 'marigold-appearance'),
    ('Marigold-Light', f'{CK}/marigold-iid-lighting-v1-1', 'marigold-lighting'),
    ('CRefNet',        f'{CK}/CRefNet/final_real.pt', 'crefnet'),
    ('Ordinal',        f'{ROOT}/ordinal-hub-weights', 'ordinal'),
]
QUALITATIVE_CKPT = f'{CK}/v17_29/checkpoint_iter_60000.pth'
TABLE_A = [
    ('Row 1: no CARI, no skip',  f'{CK}/v17_41/checkpoint_iter_40000.pth'),
    ('Row 2: CARI, no skip',     f'{CK}/v17_42/checkpoint_iter_40000.pth'),
    ('Row 3: no CARI, skip',     f'{CK}/v17_43/checkpoint_iter_40000.pth'),
    ('Row 4: full CARI (base)',  f'{CK}/v17_44/checkpoint_iter_40000.pth'),
]


def fnt(sz, bold=False):
    try:
        return ImageFont.truetype(FONTB if bold else FONT, sz)
    except Exception:
        return ImageFont.load_default()


def srgb(x):
    return np.clip(x, 0, 1) ** (1 / 2.2)


def norm(a, s):
    return np.clip(a / (s + 1e-8), 0, 1)


def scale_of(a, pct=99.0):
    v = a[a > 1e-6]
    return float(np.percentile(v, pct)) if v.size else 1.0


def load_linear_png(p):
    b = cv2.imread(p, cv2.IMREAD_COLOR)
    return np.clip(b[..., ::-1].astype(np.float32) / 255.0, 0, 1) ** 2.2


def load_hdr(p):
    a = cv2.imread(p, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    return a[..., ::-1].astype(np.float32)


def panel(arr, PW, PH, gt=False):
    im = Image.fromarray((srgb(arr) * 255).astype(np.uint8)).resize((PW, PH), Image.LANCZOS)
    if gt:
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, PW - 1, PH - 1], outline=RED, width=max(3, PW // 90))
    return im


def grid(rows, col_labels, out_name, PW, ar, row_labels=None, gt_cols=(), bold_cols=(),
         title=None, header_font=None, row_font=None):
    """rows: list of list of HWC arrays (already display-ready)."""
    PH = int(PW * ar)
    gap = max(5, PW // 60)
    header_font = header_font or int(PW / 17)
    row_font = row_font or int(PW / 19)
    max_header_lines = max((str(label).count('\n') + 1 for label in col_labels), default=1)
    head = max(34, max_header_lines * (header_font + 4) + 14)
    # Gutter must fit the longest row label, or it gets clipped at the left edge.
    if row_labels:
        rf = fnt(row_font, bold=True)
        probe = ImageDraw.Draw(Image.new('RGB', (8, 8)))
        lw = max(probe.textbbox((0, 0), t, font=rf)[2] for t in row_labels) + 26
    else:
        lw = 0
    tH = 0 if not title else head + 8
    ncol = len(col_labels)
    W = lw + ncol * PW + (ncol - 1) * gap
    H = tH + head + len(rows) * (PH + gap) - gap
    cv = Image.new('RGB', (W, H), (255, 255, 255))
    d = ImageDraw.Draw(cv)

    if title:
        d.text((6, 6), title, font=fnt(int(PW / 15), bold=True), fill=INK)

    for c, lab in enumerate(col_labels):
        x = lw + c * (PW + gap) + PW // 2
        b = c in bold_cols
        d.multiline_text((x, tH + head // 2), lab, anchor='mm', align='center', spacing=4,
                         font=fnt(header_font, bold=b),
                         fill=(RED if c in gt_cols else INK))

    for r, cells in enumerate(rows):
        y = tH + head + r * (PH + gap)
        if row_labels:
            d.text((lw - 14, y + PH // 2), row_labels[r], anchor='rm',
                   font=fnt(row_font, bold=True), fill=MUTE)
        for c, a in enumerate(cells):
            cv.paste(panel(a, PW, PH, gt=(c in gt_cols)), (lw + c * (PW + gap), y))

    os.makedirs(OUT, exist_ok=True)
    os.makedirs(DST, exist_ok=True)
    for p in (f'{OUT}/{out_name}.jpg', f'{DST}/{out_name}.jpg'):
        cv.save(p, quality=95, subsampling=0)
    print(f'  wrote {out_name}.jpg  {cv.size[0]}x{cv.size[1]}')


def predictors(spec, infer=1280):
    for label, path, ver in spec:
        yield label, AlbedoPredictor(path, ver, 'cuda', infer_max_size=infer)


# ─────────────────────────────────────────────────────────────────────────────
def fig_comp_grid(PW=520):
    """SOTA qualitative comparison on an in-the-wild (IIW) photograph."""
    print('comp_grid: SOTA albedo comparison on IIW')
    imgs = ['1094', '100010', '100091']
    inputs = [load_linear_png(f'{IIW}/{i}.png') for i in imgs]
    preds = {}
    for label, p in predictors(ROSTER):
        preds[label] = [p.albedo(x) for x in inputs]
        del p
        torch.cuda.empty_cache()
    rows = []
    for r in range(len(imgs)):
        # per-method scale: albedo is defined up to a global scale and each method sets its
        # own, so a shared scale would just render the dimmest method as "dark".
        rows.append([norm(inputs[r], scale_of(inputs[r]))] +
                    [norm(preds[lab][r], scale_of(preds[lab][r])) for lab, _, _ in ROSTER])
    ar = inputs[0].shape[0] / inputs[0].shape[1]
    grid(rows, ['Input photograph'] + [l for l, _, _ in ROSTER], 'comp_grid', PW, ar,
         bold_cols=(1,))


def fig_mid_ours(PW=440):
    """Albedo held constant while the illuminant moves -- the thesis claim, qualitatively.

    Carries the GT albedo and a per-pixel coefficient-of-variation map across the four
    conditions: red marks pixels whose predicted albedo moved when only the light moved,
    i.e. exactly the error CARI exists to remove.
    """
    print('mid_ours: albedo stability across illuminants (+ GT and CoV map)')
    sc = sorted(os.listdir(MID))[0]
    sp = os.path.join(MID, sc)
    lights = [0, 6, 12, 18]
    ins = [_tonemap_frame(_raw_frame(sp, i)) for i in lights]
    _, p = next(iter(predictors([ROSTER[0]])))
    albs = [p.albedo(x) for x in ins]
    del p
    torch.cuda.empty_cache()
    gt = _tonemap_frame(load_hdr(f'{sp}/albedo.exr'))

    st = np.stack([norm(a, scale_of(a)) for a in albs], 0)          # (N,H,W,3)
    lum = 0.2126 * st[..., 0] + 0.7152 * st[..., 1] + 0.0722 * st[..., 2]
    cov = lum.std(0) / (lum.mean(0) + 1e-6)
    cov_v = cv2.applyColorMap((np.clip(cov / 0.25, 0, 1) * 255).astype(np.uint8),
                              cv2.COLORMAP_TURBO)[..., ::-1].astype(np.float32) / 255.0
    cov_v = cov_v ** 2.2   # undo the srgb() the panel writer applies

    rows = [[norm(x, scale_of(x)) for x in ins] + [norm(gt, scale_of(gt))],
            [norm(a, scale_of(a)) for a in albs] + [cov_v]]
    ar = ins[0].shape[0] / ins[0].shape[1]
    grid(rows, [f'light {i}' for i in lights] + ['GT albedo  /  variation'], 'mid_ours',
         PW, ar, row_labels=['Input (the light moves)', 'Our albedo (it should not)'],
         gt_cols=(4,))


def fig_iiw_ours(PW=640):
    """The old iiw_ours.jpg was 860x30 -- unusable. Rebuilt as input/albedo pairs."""
    print('iiw_ours: our albedo on in-the-wild photographs')
    ids = ['1094', '100010', '100091', '100038']
    ins = [load_linear_png(f'{IIW}/{i}.png') for i in ids]
    _, p = next(iter(predictors([ROSTER[0]])))
    albs = [p.albedo(x) for x in ins]
    del p
    torch.cuda.empty_cache()
    rows = [[ins[i], norm(albs[i], scale_of(albs[i]))] for i in range(len(ids))]
    # 4 rows x 2 cols is tall; lay out as 2 rows x 4 cols (input,albedo) pairs
    flat = [[ins[0], norm(albs[0], scale_of(albs[0])), ins[1], norm(albs[1], scale_of(albs[1]))],
            [ins[2], norm(albs[2], scale_of(albs[2])), ins[3], norm(albs[3], scale_of(albs[3]))]]
    ar = np.mean([x.shape[0] / x.shape[1] for x in ins])
    grid(flat, ['Input', 'Our albedo', 'Input', 'Our albedo'], 'iiw_ours', PW, ar,
         bold_cols=(1, 3))
    del rows


def fig_maw_ours(PW=820):
    print('maw_ours: selected qualitative albedo vs physically measured albedo')
    # Three measured examples; the living-area annotation is stored in new_masks2.
    picks = [
        ('Kitchen', 'scene_28', '_DSC2770', 'new_masks'),
        ('Dark office', 'scene_29', '_DSC2744', 'new_masks'),
        ('Living area', 'scene_0', '_DSC4366', 'new_masks2'),
    ]
    rows = []
    row_labels = []
    for label, scene, name, mask_dir in picks:
        ip = f'{MAW}/images_png/{scene}/{name}.png'
        gp = f'{MAW}/labels/{mask_dir}/{scene}/{name}_albedo.png'
        if not os.path.exists(ip) or not os.path.exists(gp):
            continue
        image = load_linear_png(ip)
        gt = load_linear_png(gp)
        rows.append((image, gt))
        row_labels.append(label)
    if not rows:
        print('  [skip] no MAW pairs resolved')
        return
    p = AlbedoPredictor(QUALITATIVE_CKPT, '17', 'cuda', infer_max_size=1792)
    albs = [p.albedo(x) for x, _ in rows]
    del p
    torch.cuda.empty_cache()
    ar = np.mean([x.shape[0] / x.shape[1] for x, _ in rows])
    PH = int(PW * ar)
    gap, head = 10, 60
    rf = fnt(28, bold=True)
    probe = ImageDraw.Draw(Image.new('RGB', (8, 8)))
    gutter = max(probe.textbbox((0, 0), x, font=rf)[2] for x in row_labels) + 28
    canvas = Image.new('RGB', (gutter + 3 * PW + 2 * gap,
                               head + len(rows) * (PH + gap) - gap), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    headers = ['Input photograph', 'Measured albedo', 'Ours']
    for c, label in enumerate(headers):
        x = gutter + c * (PW + gap)
        d.text((x + PW // 2, head // 2), label, anchor='mm',
               font=fnt(30, bold=c == 2), fill=RED if c == 1 else INK)
    for r, ((image, gt), albedo) in enumerate(zip(rows, albs)):
        y = head + r * (PH + gap)
        d.text((gutter - 16, y + PH // 2), row_labels[r], anchor='rm', font=rf, fill=MUTE)
        x0 = gutter
        canvas.paste(panel(image, PW, PH), (x0, y))
        canvas.paste(panel(gt, PW, PH, gt=True), (x0 + PW + gap, y))
        canvas.paste(panel(norm(albedo, scale_of(albedo)), PW, PH),
                     (x0 + 2 * (PW + gap), y))
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(DST, exist_ok=True)
    for path in (f'{OUT}/maw_ours.jpg', f'{DST}/maw_ours.jpg'):
        canvas.save(path, quality=95, subsampling=0)
    print(f'  wrote maw_ours.jpg  {canvas.size[0]}x{canvas.size[1]}')


def fig_arap_ours(PW=820):
    print('arap_ours: selected qualitative albedo vs dense GT on synthetic renders')
    scenes = ['bedroom', 'classroom', 'livingroom']
    ins, gts = [], []
    for s in scenes:
        ip, gp = f'{ARAP}/{s}.hdr', f'{ARAP}/{s}_albedo.hdr'
        if not (os.path.exists(ip) and os.path.exists(gp)):
            continue
        a = load_hdr(ip)
        ins.append(norm(a, scale_of(a)))
        g = load_hdr(gp)
        gts.append(norm(g, scale_of(g)))
    if not ins:
        print('  [skip] no ARAP scenes resolved')
        return
    p = AlbedoPredictor(QUALITATIVE_CKPT, '17', 'cuda', infer_max_size=1792)
    albs = [p.albedo(x) for x in ins]
    del p
    torch.cuda.empty_cache()
    rows = [[ins[i], gts[i], norm(albs[i], scale_of(albs[i]))] for i in range(len(ins))]
    ar = np.mean([x.shape[0] / x.shape[1] for x in ins])
    grid(rows, ['Rendered input', 'Ground-truth albedo', 'Ours'],
         'arap_ours', PW, ar, gt_cols=(1,), bold_cols=(2,))


def fig_arap_model_grid(PW=460):
    """Model-by-illuminant comparison with a single, non-duplicated GT reference."""
    print('arap_model_grid: methods x illuminants on camera')
    scene = 'camera'
    lights = [0, 1]
    ins = [norm(load_hdr(f'{ARAP}/{scene}_light{i}.hdr'),
                scale_of(load_hdr(f'{ARAP}/{scene}_light{i}.hdr'))) for i in lights]
    gt_raw = load_hdr(f'{ARAP}/{scene}_albedo.hdr')
    gt = norm(gt_raw, scale_of(gt_raw))

    roster = [('Ours', QUALITATIVE_CKPT, '17')] + ROSTER[1:]
    pred_rows = []
    row_labels = []
    for label, predictor in predictors(roster, infer=1792):
        albs = [predictor.albedo(x) for x in ins]
        pred_rows.append([norm(a, scale_of(a)) for a in albs])
        row_labels.append(label)
        del predictor
        torch.cuda.empty_cache()

    ar = ins[0].shape[0] / ins[0].shape[1]
    PH = int(PW * ar)
    gap, card_gap, row_gap = 10, 28, 22
    ref_head, card_head, section_gap = 58, 72, 36
    ref_w = 3 * PW + 2 * gap
    card_w = 2 * PW + gap
    ncols = 2
    nrows = (len(pred_rows) + ncols - 1) // ncols
    grid_w = ncols * card_w + (ncols - 1) * card_gap
    W = max(ref_w, grid_w)
    H = (ref_head + PH + section_gap +
         nrows * (card_head + PH) + (nrows - 1) * row_gap)
    canvas = Image.new('RGB', (W, H), (255, 255, 255))
    d = ImageDraw.Draw(canvas)

    # One reference strip makes the inputs and GT available without repeating GT per model.
    ref_x = (W - ref_w) // 2
    for c, (label, arr) in enumerate(zip(['Input L0', 'Input L1', 'Dense GT albedo'], ins + [gt])):
        x = ref_x + c * (PW + gap)
        d.text((x + PW // 2, ref_head // 2), label, anchor='mm',
               font=fnt(32, bold=c == 2),
               fill=RED if c == 2 else INK)
        canvas.paste(panel(arr, PW, PH, gt=c == 2), (x, ref_head))

    grid_x = (W - grid_w) // 2
    y0 = ref_head + PH + section_gap
    for i, (label, row) in enumerate(zip(row_labels, pred_rows)):
        rr, cc = divmod(i, ncols)
        if rr == nrows - 1 and len(pred_rows) % ncols == 1:
            x0 = (W - card_w) // 2
        else:
            x0 = grid_x + cc * (card_w + card_gap)
        y = y0 + rr * (card_head + PH + row_gap)
        d.text((x0 + card_w // 2, y + 22), label, anchor='mm',
               font=fnt(31, bold=i == 0), fill=INK if i == 0 else MUTE)
        for c, light_label in enumerate(['L0', 'L1']):
            x = x0 + c * (PW + gap)
            d.text((x + PW // 2, y + 52), light_label, anchor='mm',
                   font=fnt(24, bold=True), fill=INK)
            canvas.paste(panel(row[c], PW, PH), (x, y + card_head))

    os.makedirs(OUT, exist_ok=True)
    os.makedirs(DST, exist_ok=True)
    for p in (f'{OUT}/arap_model_grid.jpg', f'{DST}/arap_model_grid.jpg'):
        canvas.save(p, quality=95, subsampling=0)
    print(f'  wrote arap_model_grid.jpg  {W}x{H}')


def fig_ablation(PW=440):
    """Table-A rows on one MID scene under two illuminants + the GT anchor."""
    print('ablation_mid: the four Table-A configurations')
    sc = sorted(os.listdir(MID))[0]
    sp = os.path.join(MID, sc)
    lights = [0, 12]
    ins = [_tonemap_frame(_raw_frame(sp, i)) for i in lights]
    gt = _tonemap_frame(load_hdr(f'{sp}/albedo.exr'))
    preds = {}
    for label, path in TABLE_A:
        p = AlbedoPredictor(path, '17', 'cuda', infer_max_size=1280)
        preds[label] = [p.albedo(x) for x in ins]
        del p
        torch.cuda.empty_cache()
    rows = []
    for r in range(len(lights)):
        rows.append([norm(ins[r], scale_of(ins[r])), norm(gt, scale_of(gt))] +
                    [norm(preds[lab][r], scale_of(preds[lab][r])) for lab, _ in TABLE_A])
    ar = ins[0].shape[0] / ins[0].shape[1]
    grid(rows, ['Input', 'GT albedo'] + [l for l, _ in TABLE_A], 'ablation_mid', PW, ar,
         row_labels=[f'light {i}' for i in lights], gt_cols=(1,), bold_cols=(5,),
         header_font=30, row_font=28)


def fig_shadow_aug(PW=520):
    """The synthetic relighting used by the shadow-invariance lever.

    The sources are 1024x768; the legacy figure downsampled them to 210px tiles, which is
    why it printed soft. Nothing is recomputed here -- only recomposited at full size.
    """
    print('shadow_aug: synthetic relighting for the shadow lever')
    src = f'{ROOT}/tests/visualizations/shadow_aug_out'
    spec = [('original.png', 'Original Hypersim\nimage'),
            ('shadow_field.png', 'Shadow field\n(known)'),
            ('shadow.png', 'Shadowed\nobservation'),
            ('sun_field.png', 'Sun field\n(known)'),
            ('sun.png', 'Sunlit\nobservation')]
    arrs, labs = [], []
    for name, lab in spec:
        p = f'{src}/{name}'
        if not os.path.exists(p):
            print(f'  [skip] missing {name}')
            return
        arrs.append(load_linear_png(p))
        labs.append(lab)
    ar = arrs[0].shape[0] / arrs[0].shape[1]
    grid([arrs], labs, 'shadow_aug', PW, ar, header_font=42)


def fig_maw_resolution(PW=470):
    """Inference resolution sweep on one MAW photograph, against the measured albedo."""
    print('maw_resolution: inference-resolution sweep')
    import csv
    with open(f'{MAW}/labels/meta.csv') as f:
        rows_csv = [r for r in csv.reader(f, delimiter='\t') if len(r) >= 5]
    scene, name = rows_csv[0][2], rows_csv[0][3]
    ip = f'{MAW}/images_png/{scene}/{name}.png'
    gp = f'{MAW}/labels/new_masks/{scene}/{name}_albedo.png'
    img = load_linear_png(ip)
    gt = load_linear_png(gp)
    sizes = [512, 768, 1024, 1280]
    outs = []
    for s in sizes:
        p = AlbedoPredictor(ROSTER[0][1], '17', 'cuda', infer_max_size=s)
        a = p.albedo(img)
        outs.append(norm(a, scale_of(a)))
        del p
        torch.cuda.empty_cache()
    cells = [norm(img, scale_of(img))] + outs + [norm(gt, scale_of(gt))]
    labs = ['Input photograph'] + [f'{s} px' for s in sizes] + ['Measured albedo']
    ar = img.shape[0] / img.shape[1]
    grid([cells], labs, 'maw_resolution', PW, ar, gt_cols=(len(cells) - 1,))


def fig_mid_preprocessing(PW=470):
    """What white balance destroys: the raw pair carries the illuminant colour, the
    white-balanced pair does not. This is the figure behind the raw-pair contribution."""
    print('mid_preprocessing: raw vs white-balanced MID pairs')
    sp = '/home/khang/datasets/MIDIntrinsics/train/kingston_library26'
    if not os.path.isdir(sp):
        sp = os.path.join(MID, sorted(os.listdir(MID))[0])
    a = _tonemap_frame(_raw_frame(sp, 0))
    b = _tonemap_frame(_raw_frame(sp, 18))

    def gray_world(x):
        m = x.reshape(-1, 3).mean(0) + 1e-6
        return np.clip(x * (m.mean() / m), 0, 1)

    gt = _tonemap_frame(load_hdr(f'{sp}/albedo.exr'))
    rows = [[norm(a, scale_of(a)), norm(b, scale_of(b)), norm(gt, scale_of(gt))],
            [gray_world(a), gray_world(b), norm(gt, scale_of(gt))]]
    ar = a.shape[0] / a.shape[1]
    grid(rows, ['Frame a', 'Frame b', 'Pseudo-GT albedo'], 'mid_preprocessing', PW, ar,
         row_labels=['RAW pair (what CARI trains on)',
                     'White-balanced (illuminant erased)'],
         gt_cols=(2,))


FIGS = {
    'comp_grid': fig_comp_grid, 'mid_ours': fig_mid_ours, 'iiw_ours': fig_iiw_ours,
    'maw_ours': fig_maw_ours, 'arap_ours': fig_arap_ours,
    'arap_model_grid': fig_arap_model_grid, 'ablation_mid': fig_ablation,
    'shadow_aug': fig_shadow_aug, 'maw_resolution': fig_maw_resolution,
    'mid_preprocessing': fig_mid_preprocessing,
}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--figures', nargs='+', default=['all'])
    a = ap.parse_args()
    want = list(FIGS) if a.figures == ['all'] else a.figures
    for k in want:
        try:
            FIGS[k]()
        except Exception as e:
            import traceback
            print(f'  [FAIL] {k}: {e}')
            traceback.print_exc()
