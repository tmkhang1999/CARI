#!/usr/bin/env python3
"""fig:inverse_shading — why shading is regressed in the inverse domain.

Replaces a low-resolution panel reproduced from the Ordinal Shading paper with one built
from our own Hypersim data. Emits:
  * images/inverse_shading/panels.jpg  — Input | Regular S | Log S | Inverse S (high-res)
  * chapters/data/shading_hist_{regular,log,inverse}.dat — histogram data for pgfplots,
    so the three distributions are drawn as VECTOR charts in the document's own fonts
    rather than as a screenshot of somebody else's matplotlib.

Shading is derived from Hypersim ground truth as S = color / diffuse_reflectance, which is
the same quantity the model is trained against (Sect. 3.x).
"""
import os

import cv2
import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = '/home/khang/IR-IID'
HS = '/home/khang/datasets/hypersim'
IMG_DST = f'{ROOT}/documents/thesis/images/inverse_shading'
DAT_DST = f'{ROOT}/documents/thesis/chapters/data'
FONTB = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def h5(p, key=None):
    with h5py.File(p, 'r') as f:
        k = key or list(f.keys())[0]
        return np.asarray(f[k]).astype(np.float32)


def srgb(x):
    return np.clip(x, 0, 1) ** (1 / 2.2)


def norm(a, pct=99.0):
    v = a[np.isfinite(a) & (a > 1e-6)]
    s = float(np.percentile(v, pct)) if v.size else 1.0
    return np.clip(a / (s + 1e-8), 0, 1)


def gray(a):
    return 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]


def main():
    os.makedirs(IMG_DST, exist_ok=True)
    os.makedirs(DAT_DST, exist_ok=True)

    scene, cam, frame = 'ai_001_001', 'scene_cam_00_final_hdf5', '0000'
    d = f'{HS}/{scene}/images/{cam}'
    fr = sorted({f.split('.')[1] for f in os.listdir(d) if f.endswith('.hdf5')})[0]
    rgb = h5(f'{d}/frame.{fr}.color.hdf5')
    alb = h5(f'{d}/frame.{fr}.diffuse_reflectance.hdf5')
    print(f'scene {scene} frame {fr}  rgb{rgb.shape}')

    valid = np.isfinite(rgb).all(-1) & np.isfinite(alb).all(-1) & (gray(alb) > 0.02)
    S = np.zeros_like(rgb)
    np.divide(rgb, np.clip(alb, 1e-3, None), out=S, where=valid[..., None])
    s = gray(S)
    s = np.clip(np.nan_to_num(s, nan=0, posinf=0), 0, None)

    reps = {
        'regular': (s, 'Regular shading  $S$'),
        'log': (np.log(s + 1e-3), 'Log shading  $\\log S$'),
        'inverse': (1.0 / (s + 1.0), 'Inverse shading  $1/(S{+}1)$'),
    }

    # ── histogram data for pgfplots (vector) ──────────────────────────────────
    v = s[valid]
    ranges = {'regular': (0, np.percentile(v, 99.5)),
              'log': (np.percentile(np.log(v + 1e-3), 0.5), np.percentile(np.log(v + 1e-3), 99.5)),
              'inverse': (0.0, 1.0)}
    for k, (arr, _) in reps.items():
        x = arr[valid]
        lo, hi = ranges[k]
        cnt, edges = np.histogram(np.clip(x, lo, hi), bins=60, range=(lo, hi))
        cnt = cnt / cnt.max()
        with open(f'{DAT_DST}/shading_hist_{k}.dat', 'w') as f:
            f.write('x y\n')
            for i in range(len(cnt)):
                f.write(f'{0.5 * (edges[i] + edges[i + 1]):.5f} {cnt[i]:.5f}\n')
        print(f'  wrote shading_hist_{k}.dat  range=[{lo:.3f},{hi:.3f}]')

    # ── image panels (raster, high-res) ───────────────────────────────────────
    PW = 700
    H, W = rgb.shape[:2]
    PH = int(PW * H / W)
    panels = [('Input', srgb(norm(rgb)))]
    for k in ('regular', 'log', 'inverse'):
        a, _ = reps[k]
        x = a.copy()
        x[~valid] = np.nan
        lo, hi = np.nanpercentile(x, 1), np.nanpercentile(x, 99)
        g = np.clip((np.nan_to_num(x, nan=lo) - lo) / (hi - lo + 1e-8), 0, 1)
        panels.append((k, np.repeat(g[..., None], 3, -1)))

    labels = ['Input', 'Regular  S', 'Log  log S', 'Inverse  1/(S+1)']
    gap, head = 6, 42
    canvas = Image.new('RGB', (4 * PW + 3 * gap, head + PH), (255, 255, 255))
    dr = ImageDraw.Draw(canvas)
    try:
        f = ImageFont.truetype(FONTB, 26)
    except Exception:
        f = ImageFont.load_default()
    for i, ((_, arr), lab) in enumerate(zip(panels, labels)):
        x = i * (PW + gap)
        im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
        canvas.paste(im.resize((PW, PH), Image.LANCZOS), (x, head))
        dr.text((x + PW // 2, head // 2), lab, anchor='mm', font=f, fill=(25, 25, 28))
    p = f'{IMG_DST}/panels.jpg'
    canvas.save(p, quality=95, subsampling=0)
    print(f'wrote {p}  {canvas.size[0]}x{canvas.size[1]}')


if __name__ == '__main__':
    main()
