#!/usr/bin/env python3
"""Individual per-(scene, method) albedo crops for the project page's interactive
comparison widget (Ours vs. a chosen baseline, cycle through scenes) -- modelled
on GS-2M's viewer (https://ndming.github.io/publications/gs2m/), which flips
between "Ours" and a dropdown-selected baseline with prev/next scene arrows.

GS-2M's widget renders live 3D meshes; ours has nothing to render live (the
deliverable is 2D albedo, not geometry), so this produces static images instead
and a small vanilla-JS swapper does the "flip" -- same interaction, no WebGL.

Reuses the same model roster and predictor as build_all_models_light_matrix.py,
one light per scene rather than that script's 3-light matrix, since the widget
shows one view at a time.

Usage:
  CUDA_VISIBLE_DEVICES=0 python tests/viz/build_compare_widget_assets.py
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image

ROOT = '/home/khang/IR-IID'
sys.path.insert(0, os.path.join(ROOT, 'tests/eval'))
os.chdir(os.path.join(ROOT, 'tests/eval'))

from eval_mid_constancy import AlbedoPredictor, _raw_frame, _tonemap_frame  # noqa: E402

MID = '/home/khang/datasets/MIDIntrinsics/test'
OUT = f'{ROOT}/presentation/assets/generated/compare_widget'
os.makedirs(OUT, exist_ok=True)

# (scene, light) -- light 12 is the session's standing "warm/colour-varying"
# reference index (used throughout tests/viz/build_mid_diversity_scenes.py);
# kept fixed across scenes so the widget isn't also confounding lamp colour.
SCENES = [
    ('everett_kitchen17', 12),
    ('everett_lobby3', 12),
    ('everett_dining2', 12),
    ('everett_kitchen5', 12),
]

MODELS = [
    ('Ours', f'{ROOT}/checkpoints/v17_29/checkpoint_iter_60000.pth', '17'),
    ('CD-IID', '', 'cdiid'),
    ('CRefNet', f'{ROOT}/checkpoints/CRefNet/final_real.pt', 'crefnet'),
    ('Marigold-App', f'{ROOT}/checkpoints/marigold-iid-appearance-v1-1', 'marigold-appearance'),
    ('Marigold-Light', f'{ROOT}/checkpoints/marigold-iid-lighting-v1-1', 'marigold-lighting'),
    ('Ordinal Shading', '', 'ordinal'),
    ('RGB-X', '', 'rgbx'),
]

WEB_W = 900


def _norm(a, pct=99.5):
    v = a[a > 1e-6]
    s = float(np.percentile(v, pct)) if v.size else 1.0
    return np.clip(a / (s + 1e-8), 0, 1)


def _save(arr, path, gamma=True):
    x = np.clip(arr, 0, 1) ** (1 / 2.2) if gamma else np.clip(arr, 0, 1)
    im = Image.fromarray((x * 255).astype(np.uint8))
    if im.width > WEB_W:
        im = im.resize((WEB_W, round(im.height * WEB_W / im.width)), Image.LANCZOS)
    im.save(path, quality=90, optimize=True)


def main(device='cuda'):
    # Load ONE model at a time, predict it across every scene, delete, move on --
    # mirrors build_all_models_light_matrix.py's proven pattern. An earlier version
    # of this script kept all 7 predictors resident simultaneously and every
    # prediction came back washed out to near-white (p99.5 normalisation still ran,
    # but something about 7 models' simultaneous CUDA/global state -- RGB-X's fp16
    # diffusers pipeline is the prime suspect -- corrupted the others' outputs).
    # A single-model diagnostic on the same scene/light gave the expected
    # albedo stats (p99.5=0.029, matching the light-0 baseline), which pinned the
    # bug to co-residency, not the model or the normalisation.
    inputs, gts = {}, {}
    for scene, light in SCENES:
        sp = os.path.join(MID, scene)
        inputs[scene] = _tonemap_frame(_raw_frame(sp, light))
        gts[scene] = _tonemap_frame(cv2.imread(os.path.join(sp, 'albedo.exr'),
                     cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[..., ::-1].copy().astype(np.float32))
        _save(inputs[scene], f'{OUT}/{scene}_input.jpg')
        _save(_norm(gts[scene]), f'{OUT}/{scene}_gt.jpg')

    manifest = {s: {'scene': s, 'light': l} for s, l in SCENES}
    for label, path, ver in MODELS:
        import torch
        p = AlbedoPredictor(path, ver, device, infer_max_size=1280)
        key = label.lower().replace(' ', '-')
        for scene, _light in SCENES:
            alb = p.albedo(inputs[scene])
            _save(_norm(alb), f'{OUT}/{scene}_{key}.jpg')
            manifest[scene][label] = f'{scene}_{key}.jpg'
        del p
        torch.cuda.empty_cache()
        print(f'  {label}: done (all scenes)')

    import json
    with open(f'{OUT}/manifest.json', 'w') as f:
        json.dump({'models': [m[0] for m in MODELS],
                   'scenes': [manifest[s] for s, _ in SCENES]}, f, indent=1)
    print(f'wrote {OUT}/manifest.json')


if __name__ == '__main__':
    main()
