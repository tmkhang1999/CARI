#!/usr/bin/env python3
"""Chapter 6 qualitative figures and an optional transfer diagnostic.

Two qualitative editing demonstrations plus an optional transfer diagnostic.
The editing figures use v17_29 explicitly: it is the pre-existing Table-B row
without the flatness lever, not the benchmark-selected v17_34 checkpoint.

  Optional: cross-illuminant relighting transfer
         I_hat(a->b) = A(I_a) * S_d(I_b) + R(I_b)  ~=  I_b
     Nothing is trained on this recomposition, but it is only a factor-consistency
     diagnostic and not a physicality test.

     MEASURED CAVEAT (2026-07-14): it is NOT degeneracy-proof, contrary to what one might
     hope. The colour-skip-OFF ablation, whose albedo is visibly drained of colour, scores
     the BEST transfer PSNR of any configuration (35.06 dB). Recomposition only constrains
     the PRODUCT A*S: a model that strips colour out of the albedo can push it into the
     shading and still reconstruct I_b exactly. The check therefore measures invariance,
     not fidelity, and is farmed by grey-collapse just like the chroma-cast scores. Only a
     metric anchored on an external reference (MID Chroma_err, MAW dE) escapes this.

  1. Highlight / glare suppression (eq:glare): A*S_d + alpha*R
  2. Factor-space illumination edit (eq:illum_edit): A*f(S_d) + R

Usage:
  python tests/viz/build_ch6_figures.py                # thesis figures only
  python tests/viz/build_ch6_figures.py --score-only   # scores only (all models)
  python tests/viz/build_ch6_figures.py --score-transfer  # figures + diagnostic
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = '/home/khang/IR-IID'
sys.path.insert(0, os.path.join(ROOT, 'tests/eval'))
os.chdir(os.path.join(ROOT, 'tests/eval'))

from eval_mid_constancy import load_v17, _raw_frame, _tonemap_frame, _hdr_valid  # noqa: E402

MID = '/home/khang/datasets/MIDIntrinsics/test'
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONTB = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'

MODELS = [
    ('Ours (v17-34)',        f'{ROOT}/checkpoints/v17_34/checkpoint_iter_60000.pth'),
    ('Ours base CARI',       f'{ROOT}/checkpoints/v17_44/checkpoint_iter_40000.pth'),
    ('Ours, colour skip OFF', f'{ROOT}/checkpoints/v17_42/checkpoint_iter_40000.pth'),
    ('No CARI',              f'{ROOT}/checkpoints/v17_41/checkpoint_iter_40000.pth'),
]

BENCHMARK_CKPT = MODELS[0][1]
APPLICATION_CKPT = f'{ROOT}/checkpoints/v17_29/checkpoint_iter_60000.pth'


def _fnt(sz, bold=False):
    try:
        return ImageFont.truetype(FONTB if bold else FONT, sz)
    except Exception:
        return ImageFont.load_default()


@torch.no_grad()
def decompose(model, rgb, dev='cuda', cap=1024):
    """rgb: (H,W,3) linear [0,1] -> dict of (H,W,3) numpy layers."""
    H, W = rgb.shape[:2]
    s = cap / max(H, W)
    inp = cv2.resize(rgb, (int(round(W * s)), int(round(H * s))),
                     interpolation=cv2.INTER_LINEAR) if s < 1 else rgb
    h, w = inp.shape[:2]
    ph, pw = (14 - h % 14) % 14, (14 - w % 14) % 14
    t = torch.from_numpy(inp).permute(2, 0, 1)[None].float().to(dev)
    if ph or pw:
        t = torch.nn.functional.pad(t, (0, pw, 0, ph), mode='replicate')
    o = model(t)

    def g(k):
        a = o[k].squeeze(0).permute(1, 2, 0).cpu().numpy()[:h, :w]
        if a.ndim == 2:
            a = a[..., None]
        if a.shape[-1] == 1:
            a = np.repeat(a, 3, axis=-1)
        return cv2.resize(a, (W, H), interpolation=cv2.INTER_LINEAR)
    # The shading head predicts inverse-domain pi. Convert it explicitly before
    # diffuse reconstruction: S_d = uninvert(pi) = (1 - pi) / pi.
    shading_pi = g('shading')
    shading_linear = (1.0 - shading_pi) / np.clip(shading_pi, 1e-4, None)
    return {
        'a': g('a_d'),
        's_pi': shading_pi,
        's': shading_linear,
        'r': g('residual'),
    }


def transfer(dec_a, dec_b):
    """Optional diagnostic: albedo from a, light from b."""
    return np.clip(dec_a['a'] * dec_b['s'] + dec_b['r'], 0, 1)


def _srgb(x):
    return np.clip(x, 0, 1) ** (1 / 2.2)


def _norm(a, pct=99.0):
    """Display-only scale normalisation. Albedo is defined up to a global scale, and
    shading_linear = (1-pi)/pi is UNBOUNDED, so neither can simply be clipped to [0,1]
    without going black or blowing out to white."""
    v = a[a > 1e-6]
    s = float(np.percentile(v, pct)) if v.size else 1.0
    return np.clip(a / (s + 1e-8), 0, 1)


def _pil(a, w, h, norm=False):
    x = _norm(a) if norm else a
    return Image.fromarray((_srgb(x) * 255).astype(np.uint8)).resize((w, h), Image.LANCZOS)


def _shadow_lift(x, gamma, percentile=95.0, max_gain=2.5):
    """Lift low luminance without assuming the factor is bounded by one."""
    lum = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
    positive = lum[lum > 1e-6]
    scale = float(np.percentile(positive, percentile)) if positive.size else 1.0
    relative = np.clip(lum / (scale + 1e-8), 0.02, 1.0)
    gain = np.minimum(relative ** (gamma - 1.0), max_gain)
    return x * gain[..., None]


def score_transfer(models, n_scenes=20, cap=1024, dev='cuda'):
    """PSNR of the recomposition against the REAL photograph I_b, over MID test pairs."""
    scenes = sorted(d for d in os.listdir(MID) if os.path.isdir(os.path.join(MID, d)))[:n_scenes]
    out = {}
    for label, ckpt in models:
        m = load_v17(ckpt, dev)
        psnrs = []
        for sc in scenes:
            sp = os.path.join(MID, sc)
            idxs = sorted({int(f.split('_')[1]) for f in os.listdir(sp)
                           if f.startswith('dir_') and f.endswith('_mip2.exr')})
            if len(idxs) < 2:
                continue
            # a fixed, well-separated pair per scene
            ia, ib = idxs[0], idxs[len(idxs) // 2]
            try:
                ra, rb = _raw_frame(sp, ia), _raw_frame(sp, ib)
            except Exception:
                continue
            ta, tb = _tonemap_frame(ra), _tonemap_frame(rb)
            vm = _hdr_valid(rb, lo_pct=20, hi_pct=99.5) > 0.5
            if vm.sum() < 100:
                continue
            da, db = decompose(m, ta, dev, cap), decompose(m, tb, dev, cap)
            rec = transfer(da, db)
            mse = float(np.mean((rec[vm] - tb[vm]) ** 2))
            psnrs.append(10 * np.log10(1.0 / max(mse, 1e-10)))
        out[label] = float(np.mean(psnrs)) if psnrs else float('nan')
        print(f'  {label:22} transfer PSNR = {out[label]:.2f} dB   (n={len(psnrs)})', flush=True)
        del m
        torch.cuda.empty_cache()
    return out


def fig_relight(model, scene, ia, ib, outdir, cap=1024, PW=470):
    sp = os.path.join(MID, scene)
    ta, tb = _tonemap_frame(_raw_frame(sp, ia)), _tonemap_frame(_raw_frame(sp, ib))
    da, db = decompose(model, ta, cap=cap), decompose(model, tb, cap=cap)
    rec = transfer(da, db)
    err = np.abs(rec - tb).mean(-1)
    err_v = cv2.applyColorMap((np.clip(err / 0.15, 0, 1) * 255).astype(np.uint8),
                              cv2.COLORMAP_INFERNO)[..., ::-1] / 255.0

    H, W = ta.shape[:2]
    PH = int(PW * H / W)
    # (label, array, normalise-for-display?)
    cols = [('Frame a (albedo source)', ta, False),
            ('A(I_a)  albedo from a', da['a'], True),
            ('Frame b (light source)', tb, False),
            ('S_d(I_b)  shading from b', db['s'], True),
            ('Recomposed  A(I_a)·S_d(I_b)+R(I_b)', rec, False),
            ('Real photo  I_b', tb, False),
            ('|error|', err_v, False)]
    gap, head = 5, 40
    Wt = len(cols) * PW + (len(cols) - 1) * gap
    canvas = Image.new('RGB', (Wt, head + PH), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    for c, (lab, arr, nrm) in enumerate(cols):
        x = c * (PW + gap)
        canvas.paste(_pil(arr, PW, PH, norm=nrm), (x, head))
        bold = 'Recomposed' in lab or 'Real photo' in lab
        d.text((x + PW // 2, head // 2), lab, fill=(20, 20, 20), anchor='mm',
               font=_fnt(17, bold=bold))
        if bold:
            d.rectangle([x, head, x + PW - 1, head + PH - 1],
                        outline=(30, 110, 45) if 'Real' in lab else (200, 30, 30), width=3)
    p = os.path.join(outdir, 'relight_transfer.jpg')
    canvas.save(p, quality=94)
    print('wrote', p)


def _load_real(path):
    """A real photograph (sRGB png) -> linear [0,1], the space the model expects."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    rgb = bgr[..., ::-1].astype(np.float32) / 255.0
    return np.clip(rgb, 0, 1) ** 2.2


def _focus_crop(a):
    """Central MAW crop used after full-image inference for every displayed factor."""
    h, w = a.shape[:2]
    x0, x1 = int(round(0.17 * w)), int(round(0.88 * w))
    y0, y1 = int(round(0.05 * h)), int(round(0.80 * h))
    return a[y0:y1, x0:x1]


def fig_edits(model, img_path, outdir, cap=1024, PW=640):
    """Glare suppression plus a percentile-normalised shading edit.

    Run on a REAL photograph rather than a MID lab capture: both edits act on the
    non-diffuse residual and the shading, and real interiors are where actual specular
    highlights, lamp glare and clipped windows occur.
    """
    t_full = _load_real(img_path)
    dec = decompose(model, t_full, cap=cap)
    t = _focus_crop(t_full)
    A, S = _focus_crop(dec['a']), _focus_crop(dec['s'])
    R = np.clip(t - A * S, 0, 1)

    # Residual isolates the non-diffuse energy (speculars, lamp glare, clipped windows);
    # boost it purely so the reader can see what the alpha sweep is removing.
    glare = [('Input photograph', t), ('Residual R  (non-diffuse)', np.clip(R * 3, 0, 1))]
    for a in (0.5, 0.0):
        glare.append((f'alpha = {a:.1f}', np.clip(A * S + a * R, 0, 1)))

    # Percentile-normalised gamma gain lifts only low predicted illumination. Directly
    # applying S**gamma would be wrong because this model's shading is unbounded.
    edits = [('Input photograph', t)]
    for gmm in (0.7, 0.5):
        lifted = _shadow_lift(S, gmm)
        edits.append((f'Ours: shading lift, gamma = {gmm}',
                      np.clip(A * lifted + R, 0, 1)))
    edits.append(('Naive: same lift on the image',
                  np.clip(_shadow_lift(t, 0.5), 0, 1)))

    H, W = t.shape[:2]
    PH = int(PW * H / W)
    gap, head, rowgap = 5, 100, 35

    def strip(items, title, y0, canvas, d):
        d.text((6, y0 + 4), title, fill=(20, 20, 20), font=_fnt(36, bold=True))
        for c, (lab, arr) in enumerate(items):
            x = c * (PW + gap)
            canvas.paste(_pil(arr, PW, PH), (x, y0 + head))
            d.text((x + PW // 2, y0 + head - 20), lab, fill=(40, 40, 40), anchor='mm',
                   font=_fnt(26))
        return y0 + head + PH + rowgap

    ncol = max(len(glare), len(edits))
    Wt = ncol * PW + (ncol - 1) * gap
    Ht = 2 * (head + PH + rowgap) + 10
    canvas = Image.new('RGB', (Wt, Ht), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    y = strip(glare, 'Residual-space suppression:  A ⊙ S_d + α·R', 6, canvas, d)
    strip(edits, 'Factor-space illumination edit:  A ⊙ f(S_d) + R', y, canvas, d)
    p = os.path.join(outdir, 'ch6_edits.jpg')
    canvas.save(p, quality=94)
    print('wrote', p)


def fig_decomposition(model, img_path, outdir, cap=1024, PW=640):
    """One model's factors on a real photograph, with residual at true display scale."""
    t_full = _load_real(img_path)
    dec = decompose(model, t_full, cap=cap)
    t = _focus_crop(t_full)
    A, S = _focus_crop(dec['a']), _focus_crop(dec['s'])
    R = np.clip(t - A * S, 0, 1)
    cols = [
        ('Input  I', t, False),
        ('Ours: albedo  A_d', A, True),
        ('Ours: diffuse shading  S_d', S, True),
        ('Ours: residual  R', R, False),
    ]
    H, W = t.shape[:2]
    PH = int(PW * H / W)
    gap, head = 8, 82
    canvas = Image.new('RGB', (4 * PW + 3 * gap, head + PH), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    for c, (lab, arr, normalise) in enumerate(cols):
        x = c * (PW + gap)
        canvas.paste(_pil(arr, PW, PH, norm=normalise), (x, head))
        d.text((x + PW // 2, head // 2), lab, fill=(25, 25, 25), anchor='mm',
               font=_fnt(40, bold=c in (1, 2, 3)))
    p = os.path.join(outdir, 'decomposition.jpg')
    canvas.save(p, quality=94)
    print('wrote', p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default=None)
    ap.add_argument('--score-only', action='store_true')
    ap.add_argument('--score-transfer', action='store_true',
                    help='also recompute the optional transfer diagnostic')
    ap.add_argument('--n-scenes', type=int, default=20)
    ap.add_argument('--real',
                    default=f'{ROOT}/tests/testing_data/MAW/images_png/scene_0/_DSC4366.png',
                    help='photograph for the decomposition and editing figures')
    ap.add_argument('--out', default=f'{ROOT}/tests/visualizations/ch6')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dst = f'{ROOT}/documents/thesis/images/ch6'
    os.makedirs(dst, exist_ok=True)

    if args.score_only or args.score_transfer:
        print('Relighting-transfer PSNR vs the REAL photograph I_b '
              f'(MID test, {args.n_scenes} scenes):')
        scores = score_transfer(MODELS, n_scenes=args.n_scenes)
        with open(os.path.join(args.out, 'transfer_psnr.json'), 'w') as f:
            json.dump(scores, f, indent=1)

    if args.score_only:
        return

    scenes = sorted(d for d in os.listdir(MID) if os.path.isdir(os.path.join(MID, d)))
    scene = args.scene or scenes[0]
    sp = os.path.join(MID, scene)
    idxs = sorted({int(f.split('_')[1]) for f in os.listdir(sp)
                   if f.startswith('dir_') and f.endswith('_mip2.exr')})
    ia, ib = idxs[0], idxs[len(idxs) // 2]
    print(f'figures on scene={scene}  pair=({ia},{ib})')

    application_model = load_v17(APPLICATION_CKPT, 'cuda')
    fig_edits(application_model, args.real, args.out, cap=1792)
    if args.score_transfer:
        benchmark_model = load_v17(BENCHMARK_CKPT, 'cuda')
        fig_relight(benchmark_model, scene, ia, ib, args.out)
        del benchmark_model
        torch.cuda.empty_cache()
    for f in ('ch6_edits.jpg',):
        Image.open(os.path.join(args.out, f)).save(os.path.join(dst, f), quality=94)
    if args.score_transfer:
        Image.open(os.path.join(args.out, 'relight_transfer.jpg')).save(
            os.path.join(dst, 'relight_transfer.jpg'), quality=94)
    print('copied into', dst)


if __name__ == '__main__':
    main()
