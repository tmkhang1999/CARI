#!/usr/bin/env python
"""test_mid_preprocess_ab2.py — decompose the cross-frame color delta into
GLOBAL CAST (what L_inv-for-color wants) vs LOCAL/shadow scatter (what it doesn't).

Follow-up to test_mid_preprocess_ab.py, which found the RAW pair's cross-frame chroma
delta is WEAKER and LESS coherent than the synthetic tint. The decisive question for the
thesis: of the raw frame-to-frame difference, how much is a real COLORED ILLUMINANT (a
global multiplicative cast c — the thesis axis) vs just SHADOWS MOVING (local, which is
the SAME axis ARAP/MID-direction already covers and which WB does NOT erase)?

Decompose the per-pixel log-color-shift lr(p) = log(I_b) - log(I_a) on the valid overlap:
  global_cast      = ‖mean_p lr(p)‖              (the achromatic+chromatic GLOBAL shift)
  global_cast_CHROMA = ‖mean_p lr(p) − mean_chan(mean_p lr(p))‖   (color-only part of cast)
  local_residual   = mean_p ‖lr(p) − mean_p lr(p)‖   (spatial scatter = moving shadows)

If raw's global_cast_CHROMA >> wb's → un-WB really does add a colored-illuminant axis.
If raw's global_cast_CHROMA ≈ wb's and local_residual dominates → the raw signal is mostly
moving shadows (NOT the color axis), and synthetic tint is the only real colored-cast source.
"""
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import sys, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.midintrinsic_dataset import MIDIntrinsicDataset
from src.data.shared_transforms import compute_tonemap_scale, tonemap_linear


def decomp(p_tm, e_tm, mask):
    eps = 1e-4
    lr = np.log(e_tm[mask] + eps) - np.log(p_tm[mask] + eps)   # (K,3)
    cast = lr.mean(0)                                          # (3,) global shift
    cast_achroma = cast.mean()                                # luminance part
    cast_chroma = cast - cast_achroma                         # color-only part of the GLOBAL cast
    local = lr - cast[None, :]                                # per-pixel residual after removing global
    return dict(
        global_cast=float(np.linalg.norm(cast)),
        global_cast_chroma=float(np.linalg.norm(cast_chroma)),   # << the colored-illuminant axis
        local_residual=float(np.mean(np.linalg.norm(local, axis=-1))),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/khang/datasets/MIDIntrinsics")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_scenes", type=int, default=6)
    ap.add_argument("--n_pairs", type=int, default=8)
    args = ap.parse_args()

    ds = MIDIntrinsicDataset(root_dir=args.root, split=args.split, use_paired=True, pair_mode='raw')
    accB, accW, accA = [], [], []
    rng = np.random.RandomState(0)
    for s in ds.scenes[:args.n_scenes]:
        sp = os.path.join(ds.root_dir, s)
        for _ in range(args.n_pairs):
            a, b = rng.choice(ds.valid_indices, size=2, replace=False)
            primary = np.clip(np.nan_to_num(ds._white_balance(sp, int(a))), 0, None).astype(np.float32)
            e_wb = np.clip(np.nan_to_num(ds._load_raw_frame(sp, int(b), wb=True)), 0, None)
            e_raw = np.clip(np.nan_to_num(ds._load_raw_frame(sp, int(b), wb=False)), 0, None)
            c = rng.uniform(0.6, 1.4, size=3).astype(np.float32); c /= c.mean()
            e_synth = e_wb * c.reshape(1, 1, 3)
            scale = compute_tonemap_scale(primary)
            p_tm = tonemap_linear(primary, scale=scale)
            vp = ds._hdr_valid_mask(primary)
            for acc, extra in [(accW, e_wb), (accB, e_raw), (accA, e_synth)]:
                e_tm = tonemap_linear(extra, scale=scale)
                m = (vp * ds._hdr_valid_mask(extra)).astype(bool)
                if m.sum() >= 200:
                    acc.append(decomp(p_tm, e_tm, m))

    def mac(acc, k): return float(np.mean([r[k] for r in acc]))
    print("\n" + "=" * 76)
    print(f"{'PATH':22s} {'global_cast':>12s} {'cast_CHROMA':>12s} {'local_resid':>12s}")
    print(f"{'':22s} {'(any shift)':>12s} {'(COLOR axis)':>12s} {'(shadows)':>12s}")
    print("-" * 76)
    for name, acc in [("WB only (current B base)", accW), ("RAW pair (path B)", accB),
                      ("WB + synth tint (path A)", accA)]:
        print(f"{name:22s} {mac(acc,'global_cast'):12.4f} "
              f"{mac(acc,'global_cast_chroma'):12.4f} {mac(acc,'local_residual'):12.4f}")
    print("=" * 76)
    print("""
DECISIVE READ — cast_CHROMA is the thesis axis (colored illuminant invariance):
  • If RAW cast_CHROMA >> WB cast_CHROMA  → un-WB adds a REAL colored-light axis (memo right).
  • If RAW cast_CHROMA ≈ WB cast_CHROMA   → un-WB does NOT add color; raw delta is just
    moving shadows (local_resid), which WB already keeps. Then synth tint is the ONLY
    source of a colored-cast signal and path A (or raw+synth) is better for the claim.
""")


if __name__ == "__main__":
    main()
