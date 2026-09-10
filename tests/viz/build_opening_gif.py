#!/usr/bin/env python3
"""Same-scene light-sweep GIF for Slide 2, replacing the mismatched
everett_kitchen18 GIF (which was shown beside everett_dining1 static panels --
a genuine same-scene violation caught in review).

Uses everett_dining1 (the opening/callback scene used everywhere else in the
deck) across all 25 real lamp settings, same crop/tonemap convention as
opening_cool.jpg / opening_warm.jpg, no baked labels or annotation boxes.
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, '/home/khang/IR-IID/tests/eval')
os.chdir('/home/khang/IR-IID/tests/eval')
from eval_mid_constancy import _raw_frame, _tonemap_frame  # noqa: E402

MID = '/home/khang/datasets/MIDIntrinsics/test'
OUT = '/home/khang/IR-IID/presentation/assets/generated/mid_diversity/opening_light_sweep.gif'
SCENE = 'everett_dining1'
LIGHTS = list(range(25))


def srgb(x):
    return np.clip(x, 0, 1) ** (1 / 2.2)


def main():
    sp = os.path.join(MID, SCENE)
    frames = []
    for light in LIGHTS:
        arr = _tonemap_frame(_raw_frame(sp, light))
        im = Image.fromarray((srgb(arr) * 255).astype(np.uint8))
        im = im.resize((750, 500), Image.LANCZOS)
        frames.append(im)
    frames[0].save(
        OUT, save_all=True, append_images=frames[1:],
        duration=250, loop=0, optimize=False,
    )
    print('wrote', OUT, frames[0].size, f'{len(frames)} frames')


if __name__ == '__main__':
    main()
