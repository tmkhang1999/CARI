#!/usr/bin/env python3
"""Bespoke composite assets for CARI_SUPER_DETAILED_REBUILD_SPEC.md Slides
6, 7, 8, 28, 30, 31, 35 — generated once, loading each checkpoint a single
time and reusing it across every slide that needs it.

Checkpoint conventions (per user rule, confirmed against FACT_CHECK.md):
  - qualitative "Ours" in any photo gallery -> v17_29/checkpoint_iter_60000.pth
  - "without paired CARI training" vs "with CARI paired training" (Slides 6, 28)
    -> the clean Row3->Row4 matched contrast: v17_43 (skip on, CARI off) vs
       v17_44 (skip on, CARI on) -- FACT_CHECK S11.1/A06.2.

Opening scene = everett_dining1, lights 0 (cool) / 18 (warm) -- same scene
already used for S01_title_mid.jpg, chroma_fidelity.jpg row 1, and
ablation_highlighted -- kept consistent across the deck.
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = '/home/khang/IR-IID'
sys.path.insert(0, os.path.join(ROOT, 'tests/eval'))
os.chdir(os.path.join(ROOT, 'tests/eval'))
from eval_mid_constancy import AlbedoPredictor, _raw_frame, _tonemap_frame  # noqa: E402

MID = '/home/khang/datasets/MIDIntrinsics/test'
MAW = f'{ROOT}/tests/testing_data/MAW'
OUT = f'{ROOT}/presentation/assets/generated/mid_diversity'
CK = f'{ROOT}/checkpoints'

FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONTB = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
INK = (23, 32, 51)
MUTE = (71, 84, 103)


def fnt(sz, bold=False):
    return ImageFont.truetype(FONTB if bold else FONT, sz)


def norm(a, pct=99.0):
    v = a[a > 1e-6]
    s = float(np.percentile(v, pct)) if v.size else 1.0
    return np.clip(a / (s + 1e-8), 0, 1)


def srgb(x):
    return np.clip(x, 0, 1) ** (1 / 2.2)


def to_panel(arr, w, h):
    return Image.fromarray((srgb(arr) * 255).astype(np.uint8)).resize((w, h), Image.LANCZOS)


def grid(rows, col_labels, row_labels, out_path, PW=560, gap=8, head=46, rowlab_w=0):
    PH = int(PW * rows[0][0].height / rows[0][0].width) if hasattr(rows[0][0], 'height') else None
    ncol = len(rows[0])
    ph = rows[0][0].height
    W = rowlab_w + ncol * PW + (ncol - 1) * gap
    H = head + len(rows) * (ph + gap) - gap
    canvas = Image.new('RGB', (W, H), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    for c, lab in enumerate(col_labels):
        x = rowlab_w + c * (PW + gap)
        d.text((x + PW // 2, head // 2), lab, fill=INK, anchor='mm', font=fnt(21, True))
    for r, row in enumerate(rows):
        y = head + r * (ph + gap)
        if row_labels:
            d.text((rowlab_w // 2, y + ph // 2), row_labels[r], fill=INK, anchor='mm', font=fnt(18, True))
        for c, im in enumerate(row):
            x = rowlab_w + c * (PW + gap)
            canvas.paste(im, (x, y))
    canvas.save(out_path, quality=94)
    print('wrote', out_path, canvas.size)


def get_predictor(path, ver):
    return AlbedoPredictor(path, ver, 'cuda', infer_max_size=1280)


def free(p):
    del p
    import torch
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
def scene_inputs(scene, lights):
    sp = os.path.join(MID, scene)
    return [_tonemap_frame(_raw_frame(sp, l)) for l in lights]


def slide6_mechanism():
    """3 rows x 2 cols: observed / without-paired (v17_43) / with-CARI (v17_44); cool/warm."""
    scene, lights = 'everett_dining1', [0, 18]
    ins = scene_inputs(scene, lights)
    PW = 560
    ph = int(PW * ins[0].shape[0] / ins[0].shape[1])

    def preds(ckpt, ver):
        p = get_predictor(ckpt, ver)
        out = [norm(p.albedo(x)) for x in ins]
        free(p)
        return out

    off = preds(f'{CK}/v17_43/checkpoint_iter_40000.pth', '17')
    on = preds(f'{CK}/v17_44/checkpoint_iter_40000.pth', '17')
    rows = [
        [to_panel(ins[0], PW, ph), to_panel(ins[1], PW, ph)],
        [to_panel(off[0], PW, ph), to_panel(off[1], PW, ph)],
        [to_panel(on[0], PW, ph), to_panel(on[1], PW, ph)],
    ]
    grid(rows, ['cool', 'warm'], ['observed', 'without paired training', 'with CARI paired training'],
         f'{OUT}/slide6_mechanism.jpg', PW=PW, rowlab_w=210)


def slide28_generality():
    """2 held-out scenes x 2x2 grid (without-paired / with-CARI, cool/warm)."""
    scenes = [('everett_kitchen5', [0, 18]), ('everett_dining2', [0, 18])]
    p_off = get_predictor(f'{CK}/v17_43/checkpoint_iter_40000.pth', '17')
    p_on = get_predictor(f'{CK}/v17_44/checkpoint_iter_40000.pth', '17')
    for scene, lights in scenes:
        ins = scene_inputs(scene, lights)
        PW = 460
        ph = int(PW * ins[0].shape[0] / ins[0].shape[1])
        off = [norm(p_off.albedo(x)) for x in ins]
        on = [norm(p_on.albedo(x)) for x in ins]
        rows = [
            [to_panel(off[0], PW, ph), to_panel(off[1], PW, ph)],
            [to_panel(on[0], PW, ph), to_panel(on[1], PW, ph)],
        ]
        grid(rows, ['cool', 'warm'], ['without paired training', 'with CARI paired training'],
             f'{OUT}/slide28_{scene}.jpg', PW=PW, rowlab_w=250)
    free(p_off)
    free(p_on)


def slide30_callback():
    """Opening scene: observed / Ours(v17_29) / CRefNet / Marigold-App, cool+warm."""
    scene, lights = 'everett_dining1', [0, 18]
    ins = scene_inputs(scene, lights)
    PW = 560
    ph = int(PW * ins[0].shape[0] / ins[0].shape[1])
    rows_data = [[to_panel(ins[0], PW, ph), to_panel(ins[1], PW, ph)]]
    for label, ckpt, ver in [
        ('Ours', f'{CK}/v17_29/checkpoint_iter_60000.pth', '17'),
        ('CRefNet', f'{CK}/CRefNet/final_real.pt', 'crefnet'),
        ('Marigold-App', f'{CK}/marigold-iid-appearance-v1-1', 'marigold-appearance'),
    ]:
        p = get_predictor(ckpt, ver)
        out = [norm(p.albedo(x)) for x in ins]
        free(p)
        rows_data.append([to_panel(out[0], PW, ph), to_panel(out[1], PW, ph)])
    grid(rows_data, ['cool', 'warm'], ['Observed', 'Ours', 'CRefNet — transformer baseline', 'Marigold-App — diffusion baseline'],
         f'{OUT}/slide30_callback.jpg', PW=PW, rowlab_w=330)


def slide7_external():
    """Distinct scene: 2 lights x [observed, CRefNet, Marigold-App]."""
    scene, lights = 'everett_kitchen18', [0, 18]
    sp = os.path.join(MID, scene)
    if not os.path.isdir(sp):
        scene = 'everett_dining2'
        sp = os.path.join(MID, scene)
    ins = scene_inputs(scene, lights)
    PW = 560
    ph = int(PW * ins[0].shape[0] / ins[0].shape[1])
    rows_data = [[to_panel(ins[0], PW, ph), to_panel(ins[1], PW, ph)]]
    for label, ckpt, ver in [
        ('CRefNet', f'{CK}/CRefNet/final_real.pt', 'crefnet'),
        ('Marigold-App', f'{CK}/marigold-iid-appearance-v1-1', 'marigold-appearance'),
    ]:
        p = get_predictor(ckpt, ver)
        out = [norm(p.albedo(x)) for x in ins]
        free(p)
        rows_data.append([to_panel(out[0], PW, ph), to_panel(out[1], PW, ph)])
    grid(rows_data, ['lighting condition 1', 'lighting condition 2'], ['Observed', 'CRefNet', 'Marigold-App'],
         f'{OUT}/slide7_external.jpg', PW=PW, rowlab_w=260)
    print('slide7 scene used:', scene)


def slide31_generality2():
    """2 held-out scenes x [reference albedo, Ours@light A, Ours@light B]."""
    scenes = [('everett_kitchen5', [0, 18]), ('everett_dining2', [0, 18])]
    p = get_predictor(f'{CK}/v17_29/checkpoint_iter_60000.pth', '17')
    rows = []
    PW = 460
    for scene, lights in scenes:
        sp = os.path.join(MID, scene)
        alb = cv2.imread(os.path.join(sp, 'albedo.exr'), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        gt = norm(_tonemap_frame(alb[..., ::-1].copy().astype(np.float32)))
        ins = scene_inputs(scene, lights)
        preds = [norm(p.albedo(x)) for x in ins]
        ph = int(PW * gt.shape[0] / gt.shape[1])
        rows.append([to_panel(gt, PW, ph), to_panel(preds[0], PW, ph), to_panel(preds[1], PW, ph)])
    free(p)
    grid(rows, ['reference material colour', 'Ours · light A', 'Ours · light B'],
         ['held-out scene 1', 'held-out scene 2'], f'{OUT}/slide31_generality.jpg', PW=PW, rowlab_w=220)


def slide35_maw_scene2():
    """Second MAW scene (Kitchen, scene_28) matching S13_maw_example's layout."""
    im_path = f'{MAW}/images_png/scene_28/_DSC2770.png'
    mask_path = f'{MAW}/labels/new_masks/scene_28/_DSC2770_albedo.png'
    if not (os.path.exists(im_path) and os.path.exists(mask_path)):
        print('MAW scene_28 files not found, skipping slide35 second scene')
        return
    inp = cv2.imread(im_path, cv2.IMREAD_COLOR)
    inp = np.clip(inp[..., ::-1].astype(np.float32) / 255.0, 0, 1) ** 2.2
    mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
    mask = np.clip(mask[..., ::-1].astype(np.float32) / 255.0, 0, 1) ** 2.2
    p = get_predictor(f'{CK}/v17_29/checkpoint_iter_60000.pth', '17')
    pred = norm(p.albedo(inp))
    free(p)
    PW = 560
    ph = int(PW * inp.shape[0] / inp.shape[1])
    row = [to_panel(inp, PW, ph), to_panel(mask, PW, ph), to_panel(pred, PW, ph)]
    grid([row], ['Input', 'Measured region / reference', 'Ours'], ['Kitchen'],
         f'{OUT}/slide35_maw_kitchen.jpg', PW=PW, rowlab_w=140)


def slide8_correspondence():
    """2 full observations + 2 magnified crops of the same material region."""
    scene, lights = 'everett_dining1', [0, 18]
    ins = scene_inputs(scene, lights)
    PW = 700
    ph = int(PW * ins[0].shape[0] / ins[0].shape[1])
    full = [to_panel(x, PW, ph) for x in ins]
    # crop coordinates in the full-res tonemap frame (fraction of image), same box both lights
    h, w = ins[0].shape[:2]
    x0, y0, x1, y1 = int(0.30 * w), int(0.35 * h), int(0.55 * w), int(0.62 * h)
    CW = 560
    ch = int(CW * (y1 - y0) / (x1 - x0))
    crops = [to_panel(x[y0:y1, x0:x1], CW, ch) for x in ins]

    gap, head, mid_gap, foot = 10, 46, 26, 6
    W = 2 * PW + gap
    H = head + ph + mid_gap + ch + foot
    canvas = Image.new('RGB', (W, H), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    for i, lab in enumerate(['cool', 'warm']):
        x = i * (PW + gap)
        d.text((x + PW // 2, head // 2), lab, fill=INK, anchor='mm', font=fnt(22, True))
        canvas.paste(full[i], (x, head))
        d.rectangle([x + x0 * PW // w, head + y0 * ph // h, x + x1 * PW // w, head + y1 * ph // h],
                    outline=(196, 61, 61), width=3)
        cx = x + (PW - CW) // 2
        canvas.paste(crops[i], (cx, head + ph + mid_gap))
    canvas.save(f'{OUT}/slide8_correspondence.jpg', quality=94)
    print('wrote', f'{OUT}/slide8_correspondence.jpg', canvas.size)


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    fns = {
        'slide6': slide6_mechanism,
        'slide7': slide7_external,
        'slide8': slide8_correspondence,
        'slide28': slide28_generality,
        'slide30': slide30_callback,
        'slide31': slide31_generality2,
        'slide35': slide35_maw_scene2,
    }
    if which == 'all':
        for name, fn in fns.items():
            print(f'--- {name} ---')
            fn()
    else:
        fns[which]()
