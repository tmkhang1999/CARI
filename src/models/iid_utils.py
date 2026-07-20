"""Shared intrinsics helpers for IUV processing and inverse shading."""

import math
import torch
import torch.nn.functional as F


def round_32(x):
    if torch.is_tensor(x):
        return 32 * torch.ceil(x / 32.0)
    return 32 * math.ceil(float(x) / 32.0)


def invert(x):
    return 1.0 / (x + 1.0)


def uninvert(x, eps=1e-4):
    return (1.0 / x.clamp(min=eps)) - 1.0


def rgb_to_iuv(rgb, eps=1e-4):
    r = rgb[:, 0:1]
    g = rgb[:, 1:2]
    b = rgb[:, 2:3]

    l = (0.299 * r) + (0.587 * g) + (0.114 * b)
    i = invert(l)
    u = invert(r / (g + eps))
    v = invert(b / (g + eps))
    return torch.cat([i, u, v], dim=1)


def iuv_to_rgb(iuv, eps=1e-4):
    # Force FP32: the uninvert -> divide chain overflows in FP16
    # because uninvert(x) = 1/x - 1 produces large values near x=0,
    # and the subsequent division by a sum of small terms can exceed FP16 max.
    orig_dtype = iuv.dtype
    iuv = iuv.float()

    l = uninvert(iuv[:, 0:1], eps=eps)
    u = uninvert(iuv[:, 1:2], eps=eps)
    v = uninvert(iuv[:, 2:3], eps=eps)

    denom = (u * 0.299) + (v * 0.114) + 0.587
    g = l / denom.clamp(min=eps)
    r = g * u
    b = g * v
    return torch.cat([r, g, b], dim=1).to(orig_dtype)


def resize_to_base(x, base_size):
    if x is None or base_size is None or base_size <= 0:
        return x
    h, w = x.shape[-2:]
    scale = float(base_size) / float(max(h, w))
    new_h = int(round_32(h * scale))
    new_w = int(round_32(w * scale))
    if new_h == h and new_w == w:
        return x
    try:
        return F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False, antialias=True)
    except TypeError:
        return F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)

# def derive_albedo(rgb, shading):
#     """
#     Derives albedo using I / S. 
#     Applies Q99 normalization to safely scale the HDR albedo into the [0, 1] range 
#     without causing a massive gradient ceiling ceiling early in training.
#     """
#     eps = 1e-5
#     raw = rgb / shading.clamp(min=eps)
#     raw = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    
#     b = raw.shape[0]
#     # Compute single Q99 scalar per image across all channels to preserve color ratios
#     q99 = torch.quantile(raw.float().reshape(b, -1), 0.99, dim=1).view(b, 1, 1, 1)
    
#     # Scale such that the 99th percentile hits 0.75 (safe headroom)
#     return (raw * (0.75 / q99.clamp(min=eps))).clamp(0.0, 1.0)

def derive_albedo(rgb, shading, anneal=0.0, q_target=0.8):
    """Derive albedo a = I/S, scale-normalised, clamped to [0,1].

    ANNEALED Q99 (residual-tautology fix, 2026-06-23). The scale applied is target/q99 where
    target = (1-anneal)*q_target + anneal*q99, i.e. the scale anneals from q_target/q99 (warmup)
    toward 1.0 (the identity derive a=I/S) as anneal→1:
      • anneal=0 (warmup): scale = 0.8/q99 ≪ 1 keeps the clamp OFF its rail early, so I/S stays
        small and gradient flows to S (a pure clamp(I/S) would saturate on random early S → dead).
      • anneal=1 (steady/eval): scale = 1 ⇒ a·S = I on diffuse ⇒ analytic R=(I−a·S)₊ → 0 except
        where I/S>1 (true speculars) → R becomes SPARSE, not a ~20% haze; and the 0.8 target no
        longer fights albedo-L1 (which wants a to match GT albedo, whose q99 varies per scene).
    Default anneal=0.0 preserves the old 0.8/q99 behaviour for legacy callers (v16). V20 controls it
    explicitly: v20.forward defaults derive_anneal=1.0 (eval = clean identity derive), and the train
    loop passes a ramp 0→1 (train_v17.train_one_step). Output stays clamped [0,1] regardless.

    FP32 forced: under autocast the (I/S) divide overflows in dim regions (S<<1 → I/S>65504 → inf),
    then nan_to_num(posinf->0) would SILENTLY ZERO albedo. eps=1e-5 (1e-6 underflows in FP16).
    """
    orig_dtype = rgb.dtype
    rgb = rgb.float()
    shading = shading.float()
    eps = 1e-5

    albedo = rgb / shading.clamp(min=eps)
    albedo = torch.nan_to_num(albedo, nan=0.0, posinf=0.0, neginf=0.0)

    B = albedo.shape[0]
    q99 = torch.quantile(
        albedo.float().reshape(B, -1), 0.99, dim=1
    ).view(B, 1, 1, 1).clamp(min=eps)
    a = float(max(0.0, min(1.0, anneal)))
    target = (1.0 - a) * q_target + a * q99          # → q99 (scale→1, identity) as anneal→1
    albedo = (albedo * (target / q99)).clamp(0.0, 1.0)
    return albedo.to(orig_dtype)
