#!/usr/bin/env python3
"""Second scene for the CARI 2x2 ablation qualitative slide (Section 13.2).

ablation_mid.jpg only ever shows 'everett_dining1'. This regenerates the
identical fig_ablation() protocol (TABLE_A: v17_41/42/43/44 @ 40k, lights
0 and 12) on 'everett_kitchen5' -- colourful bottle labels, a scene where
hue leakage should be more visually obvious than the muted dining scene.
Output goes to presentation assets only, not documents/thesis/images/hires.
"""
import os
import sys

import torch

sys.path.insert(0, '/home/khang/IR-IID/tests/eval')
os.chdir('/home/khang/IR-IID/tests/eval')
sys.path.insert(0, '/home/khang/IR-IID/tests/viz')
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')

from eval_mid_constancy import AlbedoPredictor, _raw_frame, _tonemap_frame  # noqa: E402
from build_hires_figures import norm, scale_of, load_hdr, TABLE_A  # noqa: E402
from build_mid_diversity_scenes import grid_to_presentation  # noqa: E402

MID = '/home/khang/datasets/MIDIntrinsics/test'


def main():
    sc = 'everett_kitchen5'
    print(f'ablation_diversity_scene2: {sc}')
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
    grid_to_presentation(rows, ['Input', 'GT albedo'] + [l for l, _ in TABLE_A],
                          'ablation_diversity_scene2', 440, ar,
                          row_labels=[f'light {i}' for i in lights], gt_cols=(1,))


if __name__ == '__main__':
    main()
