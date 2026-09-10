#!/usr/bin/env python3
"""Multi-method comparison on two additional held-out MID scenes.

Each sheet uses one illuminant per scene (light 12, distinct from the
cross-light grids) and includes the complete local comparison roster. The
Ordinal Shading adapter fetches its published weights through ``torch.hub``;
its placeholder path is intentionally not opened as a local checkpoint.
"""
import os
import sys

import torch

sys.path.insert(0, '/home/khang/IR-IID/tests/eval')
os.chdir('/home/khang/IR-IID/tests/eval')
sys.path.insert(0, '/home/khang/IR-IID/tests/viz')
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')

from eval_mid_constancy import AlbedoPredictor, _raw_frame, _tonemap_frame  # noqa: E402
from build_hires_figures import norm, scale_of, load_hdr  # noqa: E402
from build_mid_diversity_scenes import grid_to_presentation  # noqa: E402

MID = '/home/khang/datasets/MIDIntrinsics/test'
CK = '/home/khang/IR-IID/checkpoints'
COMPARISON_ROSTER = [
    ('CRefNet', f'{CK}/CRefNet/final_real.pt', 'crefnet'),
    ('Marigold-Light', f'{CK}/marigold-iid-lighting-v1-1', 'marigold-lighting'),
    ('Marigold-App', f'{CK}/marigold-iid-appearance-v1-1', 'marigold-appearance'),
    ('Ordinal Shading', f'{CK}/ordinal-hub-weights', 'ordinal'),
    ('Ours', f'{CK}/v17_29/checkpoint_iter_60000.pth', '17'),
]
SCENES = ['everett_dining2', 'everett_kitchen5']
LIGHT = 12


def main():
    inputs = []
    for sc in SCENES:
        sp = os.path.join(MID, sc)
        inputs.append(_tonemap_frame(_raw_frame(sp, LIGHT)))
    gts = [_tonemap_frame(load_hdr(os.path.join(MID, sc, 'albedo.exr'))) for sc in SCENES]

    preds = {}
    for label, path, arch in COMPARISON_ROSTER:
        print(f'running {label}...')
        p = AlbedoPredictor(path, arch, 'cuda', infer_max_size=1280)
        preds[label] = [p.albedo(x) for x in inputs]
        del p
        torch.cuda.empty_cache()

    for i, sc in enumerate(SCENES):
        row = [norm(gts[i], scale_of(gts[i]))] + \
              [norm(preds[lab][i], scale_of(preds[lab][i])) for lab, _, _ in COMPARISON_ROSTER]
        ar = inputs[i].shape[0] / inputs[i].shape[1]
        grid_to_presentation([row], ['Pseudo-GT'] + [l for l, _, _ in COMPARISON_ROSTER],
                              f'mid_comparison_{sc}', 520, ar, gt_cols=(0,))


if __name__ == '__main__':
    main()
