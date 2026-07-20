#!/usr/bin/env python3
"""Generate the 4 layer thumbnails for the architecture figure: I, A, S_d, R.
Runs on CPU to avoid contending with the Marigold jobs on the GPUs (one small image
through DINOv2-L on CPU is ~20 s). Outputs go to documents/thesis/images/arch/.
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

def save(a, name, gamma=True, sz=340):
    im = (srgb(a) if gamma else np.clip(a,0,1)) * 255
    im = cv2.resize(im.astype(np.uint8), (sz, int(sz*a.shape[0]/a.shape[1])), interpolation=cv2.INTER_AREA)
    cv2.imwrite(f'{DST}/{name}.png', im[..., ::-1])
    print('wrote', name)

dev = 'cpu'
m = load_v17('/home/khang/IR-IID/checkpoints/v17_29/checkpoint_iter_60000.pth', dev)
sc = 'everett_dining1'
sp = os.path.join(MID, sc)
I = _tonemap_frame(_raw_frame(sp, 0))
# downsize input for a fast CPU forward (arch figure only needs a legible thumbnail)
H, W = I.shape[:2]
s = 392 / max(H, W)
Ismall = cv2.resize(I, (int(W*s)//14*14, int(H*s)//14*14), interpolation=cv2.INTER_AREA)
t = torch.from_numpy(Ismall).permute(2,0,1)[None].float()
with torch.no_grad():
    o = m(t)
A  = o['a_d'].squeeze(0).permute(1,2,0).numpy()
Sd = o['shading_linear'].squeeze(0).permute(1,2,0).numpy()
R  = o['residual'].squeeze(0).permute(1,2,0).numpy()

save(Ismall,      'input')
save(norm(A),     'albedo')
save(norm(Sd),    'shading')
save(norm(R*3.0), 'residual')
print('done')
