#!/usr/bin/env python3
"""Genuine Hypersim ground-truth 4-panel figure (Photograph / Material colour /
RGB shading / Non-diffuse residual) for Overture Slide 3 -- exact synthetic
GT, not a model prediction. Fixes the bug where S02_iid_layers.png was
actually a real MID photo run through our own trained model (v17_44),
used to introduce A/S_d/R before the audience has any reason to trust "Ours".
"""
import glob
import os
import sys

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, '/home/khang/IR-IID')
from src.data.shared_transforms import compute_tonemap_scale  # noqa: E402

ROOT = '/home/khang/datasets/hypersim'
OUT_DIR = '/home/khang/IR-IID/presentation/assets/generated/mid_diversity'
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def load_hdf5(path):
    with h5py.File(path, 'r') as f:
        return np.asarray(f['dataset'], dtype=np.float32)


def srgb(x):
    return np.clip(x, 0, 1) ** (1 / 2.2)


def panel(arr, w, h):
    im = Image.fromarray((srgb(arr) * 255).astype(np.uint8))
    return im.resize((w, h), Image.LANCZOS)


def main(scene='ai_001_002', cam='scene_cam_00', frame='0060'):
    base = f'{ROOT}/{scene}/images/{cam}_final_hdf5/frame.{frame}'
    color = load_hdf5(f'{base}.color.hdf5')
    alb = load_hdf5(f'{base}.diffuse_reflectance.hdf5')
    illum = load_hdf5(f'{base}.diffuse_illumination.hdf5')

    scale = compute_tonemap_scale(color, percentile=90.0)
    rgb_tm = np.clip(color * scale, 0, 1)
    diffuse = alb * illum
    diffuse_tm = np.clip(diffuse * scale, 0, 1)
    residual = np.clip(color - diffuse, 0, None) * scale * 3.0  # x3 visibility, matches project convention
    residual = np.clip(residual, 0, 1)
    alb_disp = np.clip(alb / max(np.percentile(alb, 99), 1e-4), 0, 1)

    PW, PH = 620, int(620 * color.shape[0] / color.shape[1])
    gap, head = 14, 44
    labels = ['Photograph', 'Material colour  A', 'RGB shading  S_d', 'Non-diffuse residual  R']
    imgs = [rgb_tm, alb_disp, diffuse_tm / max(diffuse_tm.max(), 1e-4) * 0 + alb_disp * 0, residual]
    # shading display: illum scaled for visibility (illum can be >>1 near bright sources)
    shading_disp = np.clip(illum / max(np.percentile(illum, 97), 1e-4), 0, 1)
    imgs[2] = shading_disp

    W = 4 * PW + 3 * gap
    H = head + PH
    canvas = Image.new('RGB', (W, H), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    fnt = ImageFont.truetype(FONT, 20)
    for i, (lab, arr) in enumerate(zip(labels, imgs)):
        x = i * (PW + gap)
        d.text((x + PW // 2, head // 2), lab, anchor='mm', font=fnt, fill=(23, 32, 51))
        canvas.paste(panel(arr, PW, PH), (x, head))
    out = f'{OUT_DIR}/hypersim_gt_panels.jpg'
    canvas.save(out, quality=95)
    print(f'wrote {out}  {canvas.size}  scene={scene} frame={frame}')

    # Presentation Slide 4: exact Hypersim buffers arranged as the diffuse
    # image-formation identity. Residual is amplified only for visibility.
    labels_r = ['Observed  I', 'Diffuse reconstruction  A x S_d', 'Residual  R  (3x display)']
    imgs_r = [rgb_tm, diffuse_tm, residual]
    W_r = 3 * PW + 2 * gap
    canvas_r = Image.new('RGB', (W_r, H), (255, 255, 255))
    d_r = ImageDraw.Draw(canvas_r)
    for i, (lab, arr) in enumerate(zip(labels_r, imgs_r)):
        x = i * (PW + gap)
        d_r.text((x + PW // 2, head // 2), lab, anchor='mm', font=fnt, fill=(23, 32, 51))
        canvas_r.paste(panel(arr, PW, PH), (x, head))
    out_r = f'{OUT_DIR}/hypersim_residual_panels.jpg'
    canvas_r.save(out_r, quality=95)
    print(f'wrote {out_r}  {canvas_r.size}  scene={scene} frame={frame}')


if __name__ == '__main__':
    main()
