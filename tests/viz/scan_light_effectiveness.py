#!/usr/bin/env python3
"""Scan all 25 MID lights for one scene: measure how much CARI (Row 4) reduces
chroma drift versus no-CARI (Row 1), inside the same object box the ablation
matrix auto-selects. Used to pick a small, well-spaced light subset where the
effect is genuinely visible, instead of guessing by eye.

Usage: CUDA_VISIBLE_DEVICES=0 python scan_light_effectiveness.py <scene>
"""
import os
import sys

import cv2
import numpy as np
import torch

ROOT = '/home/khang/IR-IID'
sys.path.insert(0, os.path.join(ROOT, 'tests/eval'))
os.chdir(os.path.join(ROOT, 'tests/eval'))
from eval_mid_constancy import AlbedoPredictor, _raw_frame, _tonemap_frame  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, 'tests/viz'))
from build_mid_ablation_matrix import find_object_box, norm  # noqa: E402

MID = '/home/khang/datasets/MIDIntrinsics/test'
CK = f'{ROOT}/checkpoints'
ROW1 = f'{CK}/v17_41/checkpoint_iter_40000.pth'
ROW4 = f'{CK}/v17_44/checkpoint_iter_40000.pth'


def box_chroma(arr, box):
    x, y, w, h = box
    patch = arr[y:y + h, x:x + w]
    r = float(patch[..., 0].mean())
    g = float(patch[..., 1].mean()) + 1e-6
    b = float(patch[..., 2].mean())
    return np.array([r / g, b / g])


def main(scene):
    sp = os.path.join(MID, scene)
    lights = list(range(25))
    ins = [_tonemap_frame(_raw_frame(sp, l)) for l in lights]
    ref = norm(ins[0])
    box = find_object_box(ref)
    print(f'box (on light-0 input): {box}')

    gt = _tonemap_frame(cv2.imread(os.path.join(sp, 'albedo.exr'),
                                    cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[..., ::-1].copy().astype(np.float32))
    gt_chroma = box_chroma(norm(gt), box)
    print(f'GT chroma (r/g, b/g): {gt_chroma}')

    preds = {}
    for label, ckpt in [('Row1', ROW1), ('Row4', ROW4)]:
        p = AlbedoPredictor(ckpt, '17', 'cuda', infer_max_size=1280)
        preds[label] = [p.albedo(x) for x in ins]
        del p
        torch.cuda.empty_cache()
        print(f'  predicted: {label}')

    rows = []
    for i, light in enumerate(lights):
        c1 = box_chroma(norm(preds['Row1'][i]), box)
        c4 = box_chroma(norm(preds['Row4'][i]), box)
        drift1 = float(np.linalg.norm(c1 - gt_chroma))
        drift4 = float(np.linalg.norm(c4 - gt_chroma))
        illum_chroma = box_chroma(norm(ins[i]), box)
        illum_drift = float(np.linalg.norm(illum_chroma - gt_chroma))
        rows.append((light, drift1, drift4, drift1 - drift4, illum_drift))

    print(f'\n{"light":>5} {"drift1(no-CARI)":>16} {"drift4(CARI)":>13} {"benefit":>9} {"illum_drift":>12}')
    for light, d1, d4, benefit, idrift in rows:
        print(f'{light:>5} {d1:>16.4f} {d4:>13.4f} {benefit:>9.4f} {idrift:>12.4f}')

    ranked = sorted(rows, key=lambda r: -r[3])
    print('\nTop 10 by benefit (drift1 - drift4, higher = CARI helps more here):')
    for light, d1, d4, benefit, idrift in ranked[:10]:
        print(f'  light {light:2d}  benefit={benefit:.4f}  drift1={d1:.4f}  drift4={d4:.4f}  illum_drift={idrift:.4f}')


if __name__ == '__main__':
    scene = sys.argv[1] if len(sys.argv) > 1 else 'everett_dining1'
    main(scene)
