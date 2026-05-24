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
    l = uninvert(iuv[:, 0:1], eps=eps)
    u = uninvert(iuv[:, 1:2], eps=eps)
    v = uninvert(iuv[:, 2:3], eps=eps)

    g = l / ((u * 0.299) + (v * 0.114) + 0.587)
    r = g * u
    b = g * v
    return torch.cat([r, g, b], dim=1)


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
#         # In FP16, 1e-6 can sometimes underflow to 0.0.
#         # 1e-5 is much safer for FP16 math.
#         eps = 1e-5 
        
#         # Remove upper bound to preserve specular highlights and HDR
#         albedo = rgb / shading.clamp(min=eps)
#         albedo = torch.nan_to_num(albedo, nan=0.0, posinf=0.0, neginf=0.0)
#         return albedo.clamp(0.0, 1.0)

def derive_albedo(rgb, shading):
    # In FP16, 1e-6 can sometimes underflow to 0.0.
    # 1e-5 is much safer for FP16 math.
    eps = 1e-5 
    
    # Remove upper bound to preserve specular highlights and HDR
    albedo = rgb / shading.clamp(min=eps)
    albedo = torch.nan_to_num(albedo, nan=0.0, posinf=0.0, neginf=0.0)
    return albedo.clamp(0.0, 1.0)
