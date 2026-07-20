"""CRefNet adapter — shared by eval_arap.py and eval_maw.py.

CRefNet (Luo et al. 2023, "Learning Consistent Reflectance Estimation With a
Decoder-Sharing Transformer") ships as a standalone repo + checkpoint, not a
pip package. This module wires its released code (documents/references/CRefNet)
and downloaded weights (checkpoints/CRefNet/{final_real,crefnet-e}.pt) into our
eval pipeline, mirroring the existing Marigold adapter pattern.

Verified against documents/references/CRefNet/infer.py + modeling/crefnet_swin_v2.py:
  - Input: sRGB [0,1] CHW tensor (NOT linear, no ImageNet mean/std normalization).
  - Output ("IID" mode): pred_R is LINEAR reflectance (rgI color_rep — the decoder's
    linear chromaticity*intensity reconstruction is used directly, no output gamma).
    pred_S is shading (grayscale, repeated to 3ch — s_chs=1 in both released configs).
  - Resize: aspect-preserving long-side cap; the cap must be an exact multiple of
    min_input_size = 2**(len(enc_ch_mult)-1) * patch_size * window_size = 4*1*14 = 56
    for BOTH variants (min_input_size depends only on the encoder, not swin depth).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
CREFNET_ROOT = ROOT_DIR / 'documents/references/CRefNet'
MIN_INPUT_SIZE = 56  # fixed by the encoder architecture, independent of variant

_CRefNet = None


def _ensure_crefnet_imported():
    global _CRefNet
    if _CRefNet is not None:
        return
    sys.path.insert(0, str(CREFNET_ROOT))
    from modeling.crefnet_swin_v2 import CRefNet as _CRefNetCls
    _CRefNet = _CRefNetCls


def round_to_multiple(x: int, m: int = MIN_INPUT_SIZE) -> int:
    """Floor x to the nearest positive multiple of m (CRefNet's resize contract)."""
    return max(m, (int(x) // m) * m)


# Variant configs from documents/references/CRefNet/modeling/get_models.py.
_VARIANT_CFG = {
    'crefnet':   dict(num_swin_groups=8, depth_per_swin_group=4),
    'crefnet-e': dict(num_swin_groups=4, depth_per_swin_group=4),
}


def load_crefnet(checkpoint_path, device, variant: str = 'crefnet'):
    """Build CRefNet (or the -e efficient variant) and load released weights."""
    _ensure_crefnet_imported()
    if variant not in _VARIANT_CFG:
        raise ValueError(f"Unknown CRefNet variant {variant!r}; expected one of {list(_VARIANT_CFG)}")
    cfg = _VARIANT_CFG[variant]
    model = _CRefNet(
        3, 3, 1, 448,  # in_chans, out_r_chans, out_s_chans, input_img_size (default only)
        num_swin_groups=cfg['num_swin_groups'],
        depth_per_swin_group=cfg['depth_per_swin_group'],
        color_rep='rgI',
        use_checkpoint=False,  # gradient-checkpointing flag only; irrelevant at eval time
    )
    # weights_only=False: CRefNet's checkpoint stores training metadata (other_info) that
    # includes numpy scalars, which PyTorch 2.6+'s default safe-unpickler rejects. Safe here
    # since this is our own download from the paper's official release link.
    ckpt = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    model.load_state_dict(state_dict, strict=True)  # raises on any missing/unexpected key
    model = model.to(device).eval()
    print(f'  Loaded CRefNet ({variant}) from {checkpoint_path}')
    return model


@torch.no_grad()
def run_crefnet(model, rgb_display_srgb_linear: np.ndarray, max_size: int, device):
    """Run CRefNet in IID mode.

    Args:
        rgb_display_srgb_linear: display-space LINEAR [0,1] HWC float array — i.e. the
            same tonemapped-but-not-yet-gamma-encoded image eval_arap's
            `_external_display_linear` / eval_maw's HDR-norm already produce for Marigold.
            This function applies the sRGB gamma itself (CRefNet expects sRGB input).
        max_size: desired long-side cap; rounded down to a multiple of 56 internally.

    Returns:
        (albedo_hwc, shading_hwc): float32 HWC arrays in [0,1], LINEAR space, resized
        back to the input's original resolution (CRefNet's own output_original_size=True).
    """
    H, W = rgb_display_srgb_linear.shape[:2]
    rgb_srgb = np.power(np.clip(rgb_display_srgb_linear, 0.0, 1.0), 0.45454545454545453)
    t = torch.from_numpy(rgb_srgb.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)

    max_dim = round_to_multiple(max_size)
    out = model(t, mode='IID', resize_input=True,
                min_img_size=None, max_img_size=max_dim,
                output_original_size=True)

    albedo = out['pred_R'].clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy()
    shading = out['pred_S'].clamp(min=0.0).squeeze(0).permute(1, 2, 0).cpu().numpy()
    if albedo.shape[:2] != (H, W):
        import cv2
        albedo = cv2.resize(albedo, (W, H), interpolation=cv2.INTER_LINEAR)
        shading = cv2.resize(shading, (W, H), interpolation=cv2.INTER_LINEAR)
    return albedo.astype(np.float32), shading.astype(np.float32)
