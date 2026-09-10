#!/usr/bin/env python3
"""Purpose-built ARAP hard-shadow figure: full-resolution Input/GT/Ours on the
bedroom scene, plus an actual magnified inset of the window-shadow bands on the
bed. This avoids using the visually ambiguous camera-material texture.

Same checkpoint (v17_29/checkpoint_iter_60000.pth) as the other ARAP
qualitative figures (arap_ours.jpg, arap_model_grid.jpg) for consistency --
this is the thesis's own disclosed convention for ARAP/MAW/IIW qualitative
illustration, distinct from the v17_34 checkpoint whose numbers are reported.
"""
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, '/home/khang/IR-IID/tests/eval')
os.chdir('/home/khang/IR-IID/tests/eval')
sys.path.insert(0, '/home/khang/IR-IID/tests/viz')
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')

from eval_mid_constancy import AlbedoPredictor  # noqa: E402
from build_hires_figures import norm, scale_of, load_hdr, panel, fnt, INK, RED  # noqa: E402
from PIL import ImageDraw  # noqa: E402

ARAP = '/home/khang/IR-IID/tests/testing_data/ARAP_dataset'
CK = '/home/khang/IR-IID/checkpoints/v17_29/checkpoint_iter_60000.pth'
OUT_DIR = '/home/khang/IR-IID/presentation/assets/generated/mid_diversity'

scene = 'bedroom'
in_raw = load_hdr(f'{ARAP}/{scene}.hdr')
gt_raw = load_hdr(f'{ARAP}/{scene}_albedo.hdr')
inp = norm(in_raw, scale_of(in_raw))
gt = norm(gt_raw, scale_of(gt_raw))

p = AlbedoPredictor(CK, '17', 'cuda', infer_max_size=1792)
ours_raw = p.albedo(inp)
del p
torch.cuda.empty_cache()
ours = norm(ours_raw, scale_of(ours_raw))

PW = 900
ar = inp.shape[0] / inp.shape[1]
PH = int(PW * ar)
full_row = [panel(inp, PW, PH), panel(gt, PW, PH, gt=True), panel(ours, PW, PH)]

# Save full-resolution panels so the cast-shadow crop is selected from source pixels.
Image.fromarray((inp.clip(0, 1) ** (1 / 2.2) * 255).astype('uint8')).save(
    f'{OUT_DIR}/arap_bedroom_input_full.png')
Image.fromarray((gt.clip(0, 1) ** (1 / 2.2) * 255).astype('uint8')).save(
    f'{OUT_DIR}/arap_bedroom_gt_full.png')
Image.fromarray((ours.clip(0, 1) ** (1 / 2.2) * 255).astype('uint8')).save(
    f'{OUT_DIR}/arap_bedroom_ours_full.png')
print('wrote full-resolution panels:', inp.shape, gt.shape, ours.shape)
