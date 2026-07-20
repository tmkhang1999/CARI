#!/usr/bin/env python3
"""Thumbnails for fig:cari: two illuminant conditions of one MID scene + their predicted
albedos and shadings. CPU (no GPU contention). -> documents/thesis/images/arch/cari_*.png

The point the figure makes visually: I1 and I2 differ in illuminant cast, the two predicted
albedos A1/A2 look IDENTICAL (L_inv), and the shadings S1/S2 differ (they absorb the cast).
"""
import os, sys
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import cv2, numpy as np, torch
sys.path.insert(0, '/home/khang/IR-IID/tests/eval')
os.chdir('/home/khang/IR-IID/tests/eval')
from eval_mid_constancy import load_v17, _raw_frame, _tonemap_frame

DST = '/home/khang/IR-IID/documents/thesis/images/arch'
os.makedirs(DST, exist_ok=True)
MID = '/home/khang/datasets/MIDIntrinsics/test'

def srgb(x): return np.clip(x, 0, 1) ** (1/2.2)
def norm(a, p=99.0):
    v = a[a > 1e-6]; s = float(np.percentile(v, p)) if v.size else 1.0
    return np.clip(a/(s+1e-8), 0, 1)
def save(a, name, gamma=True, sz=300):
    im = ((srgb(a) if gamma else np.clip(a,0,1))*255).astype(np.uint8)
    im = cv2.resize(im, (sz, int(sz*a.shape[0]/a.shape[1])), interpolation=cv2.INTER_AREA)
    cv2.imwrite(f'{DST}/{name}.png', im[..., ::-1]); print('wrote', name)

m = load_v17('/home/khang/IR-IID/checkpoints/v17_29/checkpoint_iter_60000.pth', 'cpu')
sp = os.path.join(MID, 'everett_dining1')

# pick two illuminants with the largest chroma difference among a few candidates
cands = [0, 6, 12, 18, 3, 9]
def cast(fr):
    x = _raw_frame(sp, fr); g = x[...,1].mean()+1e-6
    return x[...,0].mean()/g, x[...,2].mean()/g
casts = {c: cast(c) for c in cands}
# warmest (high r/g) and coolest (high b/g)
warm = max(cands, key=lambda c: casts[c][0]-casts[c][1])
cool = max(cands, key=lambda c: casts[c][1]-casts[c][0])
print(f'warm={warm} {casts[warm]}  cool={cool} {casts[cool]}')

for tag, fr in (('1', warm), ('2', cool)):
    I = _tonemap_frame(_raw_frame(sp, fr))
    H, W = I.shape[:2]; s = 336/max(H,W)
    Is = cv2.resize(I, (int(W*s)//14*14, int(H*s)//14*14), interpolation=cv2.INTER_AREA)
    with torch.no_grad():
        o = m(torch.from_numpy(Is).permute(2,0,1)[None].float())
    save(Is, f'cari_I{tag}')
    save(norm(o['a_d'].squeeze(0).permute(1,2,0).numpy()), f'cari_A{tag}')
    save(norm(o['shading_linear'].squeeze(0).permute(1,2,0).numpy()), f'cari_S{tag}')
print('done')
