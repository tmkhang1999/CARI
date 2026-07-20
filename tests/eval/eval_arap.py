"""ARAP cross-illumination evaluation.

Two modes:
  eval_arap          -- per-image LMSE / si-RMSE / SSIM + contact sheets
  eval_arap_constancy -- cross-light CoV (C_arap), Cast_RMS, coloured-illuminant breakdown

Usage:
  # constancy mode (primary for thesis)
  python tests/eval/eval_arap.py --constancy --checkpoint checkpoints/v20/... --label 40k_v20

  # full eval with contact sheets
  python tests/eval/eval_arap.py --checkpoint checkpoints/v20/...
"""
import argparse
import collections
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / 'src'))
sys.path.insert(0, str(ROOT_DIR / 'preprocessor'))
sys.path.insert(0, str(ROOT_DIR / 'tests' / 'infer'))   # infer_wild moved here

INTRINSIC_HDR_PATH = ROOT_DIR / 'documents/references/IntrinsicHDR/intrinsic_decomposition'
sys.path.insert(0, str(INTRINSIC_HDR_PATH))

MARIGOLD_PATH = ROOT_DIR / 'documents/references/marigold'

from infer_wild import (  # noqa: E402
    load_image, get_normals_metric3d, get_segmentation_nyu40,
    resolve_device, _infer_model_version, _build_model,
)
from src.models.ccr_utils import compute_ccr
from src.models.iid_utils import uninvert, iuv_to_rgb
from src.models import IntrinsicDecompositionV17, IntrinsicDecompositionV17Refiner, IntrinsicDecompositionV20
from src.models.v18_pgid import V18PGID
from src.train import (  # noqa: E402
    _compute_lmse, _masked_scale_invariant_rmse,
    _compute_ssim_bounded, _compute_shading_ssim,
)
from documents.references.chrislib.chrislib.metrics import rmse_error
from crefnet_adapter import load_crefnet, run_crefnet
from ordinal_adapter import load_ordinal, run_ordinal

try:
    from common.general import guided_filter
except ImportError:
    guided_filter = None

MarigoldIIDPipeline = None
PILImage = None


def _ensure_marigold_imported():
    global MarigoldIIDPipeline, PILImage
    if MarigoldIIDPipeline is not None and PILImage is not None:
        return
    sys.path.insert(0, str(MARIGOLD_PATH))
    from marigold import MarigoldIIDPipeline as _MarigoldIIDPipeline
    from PIL import Image as _PILImage
    MarigoldIIDPipeline = _MarigoldIIDPipeline
    PILImage = _PILImage

# ── Dataset constants ──────────────────────────────────────────────────────────
_ARAP_TYPES_JSON = ROOT_DIR / 'tests/testing_data/ARAP_types.json'
INPUT_EXTS = ('.exr', '.hdr', '.png', '.jpg', '.jpeg')
_DERIVED_KEYWORDS = ('_albedo', '_shading', '_normal', '_depth')


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def _load_marigold_pipeline(checkpoint, device):
    _ensure_marigold_imported()
    pipe = MarigoldIIDPipeline.from_pretrained(checkpoint, torch_dtype=torch.float32).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _load_model_versioned(checkpoint, version, device):
    """Load V17, V18, V20, or auto-detect from checkpoint config."""
    state_dict = torch.load(checkpoint, map_location='cpu')
    config = state_dict.get('config', {})
    model_state = state_dict.get('model_state_dict', state_dict.get('model', {}))
    model_cfg = config.get('model', {})

    # Determine version
    inferred = _infer_model_version(config, checkpoint, version)

    if inferred == '20' or inferred == 20:
        model = IntrinsicDecompositionV20(model_cfg).to(device)
    elif inferred in ('18', 18):
        model_config = dict(model_cfg)
        model_config.setdefault('sd_pretrained', 'Manojb/stable-diffusion-2-1-base')
        model_config.setdefault('cross_attn_dim', 1024)
        model_config.setdefault('null_seq_len', 77)
        model_config.setdefault('num_seg_classes', 41)
        model_config.setdefault('input_size', 384)
        model = V18PGID(model_config).to(device)
    elif str(inferred) in ('17.27', '17_27') or abs(float(model_cfg.get('version', 0.0)) - 17.27) < 1e-6:
        model = IntrinsicDecompositionV17Refiner(model_cfg).to(device)
    elif inferred in ('17', 17):
        model = IntrinsicDecompositionV17(model_cfg).to(device)
    elif inferred == 'auto' or model_cfg.get('backbone') == 'convnextv2_base':
        model = _build_model(model_cfg, inferred, device)
    else:
        model = IntrinsicDecompositionV17(model_cfg).to(device)

    own = model.state_dict()
    filtered = {k: v for k, v in model_state.items()
                if k in own and v.shape == own[k].shape}
    model.load_state_dict(filtered, strict=False)
    return model.eval()


def _is_external_version(version):
    return version in ('marigold-appearance', 'marigold-lighting')


def _is_crefnet_version(version):
    return version in ('crefnet', 'crefnet-e')


def _is_ordinal_version(version):
    return version in ('ordinal', 'ordinal-rendered-only')


# ──────────────────────────────────────────────────────────────────────────────
# Input preparation
# ──────────────────────────────────────────────────────────────────────────────

def _white_balance_gt(rgb, albedo_gt, eps=1e-3, percentile=99.0, target=0.8):
    """TRUE white balance for the standard (Ordinal Shading) ARAP protocol.

    THE BUG THIS REPLACES (found 2026-07-14): the `--white_balance` path previously called
    `_hdr_norm`, which multiplies all three channels by ONE scalar. That is EXPOSURE
    normalisation. It leaves r/g and b/g *identically unchanged*, so it removes no coloured
    illuminant whatsoever — despite the call site claiming it "desaturates the illuminant".
    Every published-comparison ARAP number produced through that path was therefore computed
    on a raw coloured input and is NOT comparable to Ordinal Shading's reported scores.

    The published protocol instead reconstructs a white-lit image: derive the (coloured)
    shading implied by the ground-truth albedo, strip its chroma by taking its luminance, and
    recompose. In notation:

        S     = I / max(A*, eps)          # coloured shading implied by the GT albedo
        I_wb  = A* * luminance(S)         # recompose under an achromatic illuminant

    The result is the same scene lit by a grey light of the same intensity, which is what the
    standard protocol scores against. Pixels where A* is degenerate carry no recoverable
    shading and are left to the caller's validity mask.
    """
    rgb = np.clip(np.nan_to_num(rgb, nan=0.0), 0.0, None)
    a = np.clip(np.nan_to_num(albedo_gt, nan=0.0), 0.0, None)
    # SCALE-RELATIVE floor. ARAP GT albedo is mixed-scale: some .hdr files peak near 0.006
    # while .jpg files run to 255. A fixed absolute eps (e.g. 1e-3) would clamp ~16% of the
    # full range on the dim HDR files and destroy S = I/A. Anchor it to the image's own scale.
    a_hi = float(np.percentile(a[a > 0], 99.0)) if (a > 0).any() else 1.0
    floor = max(eps * a_hi, 1e-12)
    A = np.clip(a, floor, None)
    S = rgb / A                                        # (H,W,3) coloured shading
    s_lum = (0.2126 * S[..., 0] + 0.7152 * S[..., 1] + 0.0722 * S[..., 2])[..., None]
    wb = a * s_lum                                     # recompose under a grey illuminant
    # Match the exposure convention of the rest of the pipeline (percentile -> target).
    lum = 0.3 * wb[..., 0] + 0.59 * wb[..., 1] + 0.11 * wb[..., 2]
    pos = lum[lum > 1e-4]
    if pos.size:
        p = float(np.percentile(pos, percentile)) + 1e-6
        wb = wb * (target / p)
    return np.clip(np.nan_to_num(wb, nan=0.0), 0.0, 1.0)


def _hdr_norm(rgb, percentile=99.0, target=0.8):
    img = np.nan_to_num(rgb, nan=0.0)
    img = np.clip(img, 0.0, None)
    lum = 0.3 * img[..., 0] + 0.59 * img[..., 1] + 0.11 * img[..., 2]
    lum_pos = lum[lum > 0.0001]
    if lum_pos.size == 0:
        return img
    p = float(np.percentile(lum_pos, percentile)) + 1e-6
    return np.clip(img * (target / p), 0.0, 1.0)


def _external_display_linear(rgb_linear, is_hdr):
    """Display-space image for Marigold (input pre-processing)."""
    img = np.nan_to_num(np.clip(rgb_linear, 0.0, None))
    if is_hdr:
        p80 = float(np.percentile(img, 80.0)) + 1e-6
        x = img / p80 * 0.8
        x = x / (1.0 + x) * 2.0
        return np.clip(x, 0.0, 1.0)
    return np.clip(img, 0.0, 1.0)


def _prepare_input_rgb(rgb_linear, is_hdr, version):
    """Return tonemapped [0,1] RGB for model input."""
    if _is_external_version(version):
        return _external_display_linear(rgb_linear, is_hdr)
    if is_hdr:
        return _hdr_norm(rgb_linear, percentile=90.0, target=0.8)
    return np.power(np.clip(rgb_linear, 0.0, 1.0), 0.45454545454545453)


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────

def _run_marigold_inference(pipe, rgb_linear, is_hdr, max_size=None):
    """
    Returns (albedo_hwc, shading_hwc) float32 [0,1] in tonemapped/display space.
    For appearance model (no shading), pseudo shading = I/A in display space.

    BUG FIX (2026-07-13): `pipe()` was called with no `processing_res`, so
    Marigold ignored --max_size and processed at its own internal default
    (768) while CRefNet/Ordinal/our-model below all honour it. match_input_res
    defaults to True in the pipeline, so the returned array still matches the
    input's native resolution.
    """
    from marigold_adapter import marigold_albedo_hwc_linear
    rgb_display = _external_display_linear(rgb_linear, is_hdr)
    rgb_srgb = np.power(np.clip(rgb_display, 0.0, 1.0), 0.45454545454545453)
    img_uint8 = (np.clip(rgb_srgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    pil_img = PILImage.fromarray(img_uint8)
    kwargs = {'processing_res': int(max_size)} if max_size is not None else {}
    pipe_out = pipe(pil_img, **kwargs)

    # Albedo — v1-1: colour space lives on pipe.target_properties, not the entry.
    albedo_hwc = marigold_albedo_hwc_linear(pipe, pipe_out, target='albedo')

    # Shading (or pseudo-shading from appearance). target_names is on the output object.
    if 'shading' in getattr(pipe_out, 'target_names', []):
        shading_hwc = marigold_albedo_hwc_linear(pipe, pipe_out, target='shading')
    else:
        eps = 1e-6
        shading_hwc = np.maximum(rgb_display, 0.0) / (albedo_hwc + eps)

    return albedo_hwc, shading_hwc


def _run_own_model_inference(model, version, rgb_linear, is_hdr, img_path,
                              device, max_size=1280, min_size=1024):
    """Returns (pred_ad, pred_sd) HWC float32 at original resolution."""
    H_orig, W_orig = rgb_linear.shape[:2]

    # Resize to fit max/min constraints
    md = max(H_orig, W_orig)
    if md > max_size:
        proc_scale = max_size / float(md)
    elif md < min_size:
        proc_scale = min_size / float(md)
    else:
        proc_scale = 1.0

    H = int(round(H_orig * proc_scale))
    W = int(round(W_orig * proc_scale))
    if (H, W) != (H_orig, W_orig):
        rgb_res = cv2.resize(rgb_linear, (W, H), interpolation=cv2.INTER_LINEAR)
    else:
        rgb_res = rgb_linear

    rgb_in = _prepare_input_rgb(rgb_res, is_hdr, version)
    t_rgb = torch.from_numpy(rgb_in).permute(2, 0, 1).unsqueeze(0).float().to(device)

    version_int = int(version) if str(version).isdigit() else 0

    # V18: needs normals + segmentation + CCR
    if version_int == 18:
        t_normals = torch.from_numpy(
            get_normals_metric3d(rgb_in, device).astype(np.float32)
        ).permute(2, 0, 1).unsqueeze(0).to(device)
        seg_nyu40 = get_segmentation_nyu40(str(img_path))
        if seg_nyu40.shape != (H, W):
            seg_nyu40 = cv2.resize(seg_nyu40.astype(np.int32), (W, H),
                                   interpolation=cv2.INTER_NEAREST)
        t_seg = torch.from_numpy(seg_nyu40).long().unsqueeze(0).unsqueeze(0).to(device)
        t_masks = torch.ones((1, 1, H, W), dtype=torch.bool).to(device)
        t_ccr = compute_ccr(t_rgb)

        stride = 32
        pad_h = (stride - (H % stride)) % stride
        pad_w = (stride - (W % stride)) % stride
        if pad_h > 0 or pad_w > 0:
            pad_spec = (0, pad_w, 0, pad_h)
            t_rgb = torch.nn.functional.pad(t_rgb, pad_spec, mode='replicate')
            t_normals = torch.nn.functional.pad(t_normals, pad_spec, mode='replicate')
            t_ccr = torch.nn.functional.pad(t_ccr, pad_spec, mode='replicate')
            t_seg = torch.nn.functional.pad(t_seg.float(), pad_spec, mode='replicate').long()
            t_masks = torch.nn.functional.pad(t_masks.float(), pad_spec, mode='replicate').bool()

        with torch.no_grad():
            preds = model(rgb=t_rgb, normals=t_normals, seg=t_seg,
                          valid_mask=t_masks, ccr=t_ccr)
    else:
        # V17, V20: RGB only; pad to stride-14 multiple
        stride = 8 if version_int == 17 else 8
        ph = (32 - (H % 32)) % 32
        pw = (32 - (W % 32)) % 32
        if ph > 0 or pw > 0:
            t_rgb = torch.nn.functional.pad(t_rgb, (0, pw, 0, ph), mode='replicate')

        with torch.no_grad():
            preds = model(t_rgb)

    def to_hwc(t):
        arr = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        return arr

    t_ad = preds.get('a_d', preds.get('albedo', t_rgb[:, :3]))
    t_sd = preds.get('shading_linear', preds.get('pi', t_rgb[:, :1]))

    pred_ad = to_hwc(t_ad.clamp(0.0, 1.0))[:H, :W]
    pred_sd = to_hwc(t_sd.clamp(0.0, None))[:H, :W]

    # Uninvert shading (π → S) ONLY if we fell back to the π-domain field. Both V17 and
    # V20 expose 'shading_linear' (already-linear S); V20 additionally exposes 'd_g' (π-domain
    # gray), so gating on d_g alone would double-invert the already-linear V20 shading.
    if preds.get('shading_linear') is None and (preds.get('d_g') is not None or 'pi' in preds):
        try:
            pred_sd = uninvert(t_sd[:, :, :H, :W]).squeeze(0).permute(1, 2, 0).cpu().numpy()
        except Exception:
            pass

    # Back to original resolution
    rgb_tm_orig = _hdr_norm(rgb_linear, 99.0) if is_hdr else np.clip(rgb_linear, 0.0, 1.0)

    if (H, W) != (H_orig, W_orig):
        pred_ad = cv2.resize(pred_ad, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
        pred_sd = cv2.resize(pred_sd, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)

    return pred_ad, pred_sd, rgb_tm_orig


# ──────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ──────────────────────────────────────────────────────────────────────────────

def _is_derived(stem):
    return any(kw in stem for kw in _DERIVED_KEYWORDS)


def get_arap_cases(dataset_dir):
    """Returns sorted list of (base_name, input_stem, ext) tuples."""
    base_exts = {}
    for fname in os.listdir(dataset_dir):
        stem, ext = os.path.splitext(fname.lower())
        if ext not in INPUT_EXTS or _is_derived(stem):
            continue
        base = re.sub(r'_light\d+$', '', stem)
        base_exts.setdefault(base, set()).add((stem, ext))

    cases = []
    for base, light_inputs in sorted(base_exts.items()):
        for stem, ext in sorted(light_inputs):
            has_albedo = any(
                os.path.exists(os.path.join(dataset_dir, f'{stem}_albedo{e}'))
                for e in INPUT_EXTS
            )
            # Include if it matches a light pattern or has GT
            if re.match(r'^' + re.escape(base) + r'_light\d+$', stem, re.IGNORECASE) or has_albedo:
                cases.append((base, stem, ext.lstrip('.')))

    return sorted(set(cases))


def _load_scene_domains():
    """Load ARAP_types.json → {scene_name: 'indoor'|'outdoor'}."""
    try:
        with open(_ARAP_TYPES_JSON) as f:
            data = json.load(f)
    except Exception:
        return {}
    name2dom = {}
    for cat in data.get('categories', []):
        dom = cat.get('domain', '').lower()
        if dom not in ('indoor', 'outdoor'):
            dom = 'outdoor'
        for s in cat.get('samples', []):
            name2dom[s.get('name', '')] = dom
    return name2dom


def filter_cases_by_domain(cases, scene_filter):
    """Keep only cases whose base scene is in the requested domain.
    scene_filter: 'all' (no filter) | 'indoor' | 'outdoor'."""
    if scene_filter == 'all':
        return cases
    name2dom = _load_scene_domains()
    if not name2dom:
        print(f'  WARN: ARAP_types.json missing/empty — --scene_filter {scene_filter} ignored.')
        return cases
    kept = [c for c in cases if name2dom.get(c[0], '').lower() == scene_filter.lower()]
    n_scenes = len(set(c[0] for c in kept))
    print(f"  scene_filter='{scene_filter}': {len(kept)} cases / {n_scenes} scenes "
          f"(of {len(cases)} total cases)")
    return kept


def _find_file(dir_path, stem, exts):
    for ext in exts:
        p = os.path.join(dir_path, f'{stem}{ext}')
        if os.path.exists(p):
            return p
    return None


def load_gt(dataset_dir, base_name, input_stem, ext):
    """Load GT albedo (and shading if available). Returns (albedo, shading) or (albedo, None)."""
    def _resolve(stem):
        same = _find_file(dataset_dir, f'{stem}_albedo', INPUT_EXTS)
        if same:
            return same
        other = _find_file(dataset_dir, f'{base_name}_albedo', INPUT_EXTS)
        if other:
            print(f"  WARN: GT '{stem}_albedo' has no '{ext}' file; using '{base_name}_albedo' "
                  f"(input is '{input_stem}' — GT/input dynamic range may differ).")
            return other
        return None

    albedo_path = _resolve(input_stem) or _resolve(base_name)
    if albedo_path is None:
        return None, None

    albedo = load_image(albedo_path)[0]
    shading_path = _find_file(dataset_dir, f'{input_stem}_shading',
                              tuple(f'.{e}' for e in INPUT_EXTS))
    shading = load_image(shading_path)[0] if shading_path else None
    return albedo, shading


def align_scale(pred, gt, mask):
    """Least-squares scalar c s.t. c*pred ≈ gt over mask."""
    p_flat = pred.reshape(-1)[mask.reshape(-1) > 0]
    t_flat = gt.reshape(-1)[mask.reshape(-1) > 0]
    c = float(np.sum(p_flat * t_flat)) / (float(np.sum(p_flat * p_flat)) + 1e-6)
    return pred * max(c, 1e-6)


def _cd_iid_rmse(pred_t, gt_t, mask_t):
    """Direct RGB RMSE matching CD-IID's bundled chrislib metric."""
    pred = pred_t.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()
    gt = gt_t.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()
    mask = mask_t.detach().squeeze().bool().cpu().numpy()
    return float(rmse_error(pred, gt, mask))


# NOTE (2026-07-11): a least-squares single-scalar si-RMSE was tried here to match the Grosse/
# Ordinal convention exactly, but ARAP GT albedo is stored as TINY-valued HDR (alley: max 0.0056),
# so least-squares absorbs the whole pred↔GT scale gap and chrislib's plain rmse_error then reports
# RMSE in those tiny absolute units → collapses to ~0.0015, scale-DEPENDENT on the arbitrary HDR
# range. The mean-normalized _masked_scale_invariant_rmse is invariant to that GT scale and already
# lands in the published ballpark (~0.29 vs their 0.25), so it is the robust choice here.
def compute_metrics(pred_t, gt_t, mask_t, metric_type='albedo'):
    lmse = _compute_lmse(pred_t, gt_t, mask_t).item()
    rmse = _cd_iid_rmse(pred_t, gt_t, mask_t)
    si_rmse = _masked_scale_invariant_rmse(pred_t, gt_t, mask_t).item()
    if metric_type == 'albedo':
        ssim = _compute_ssim_bounded(pred_t, gt_t, mask_t)
    else:
        ssim = _compute_shading_ssim(pred_t, gt_t, mask_t)
    return lmse, rmse, si_rmse, ssim


def compute_diffuse_recon(pred_ad, pred_sd, albedo_gt, shading_gt):
    pred_diffuse = pred_ad * pred_sd
    gt_diffuse = albedo_gt * shading_gt

    def _tm(img):
        return _hdr_norm(img, 99.0)

    return float(np.mean(np.abs(_tm(pred_diffuse) - _tm(gt_diffuse))))


# ──────────────────────────────────────────────────────────────────────────────
# Visualization helpers
# ──────────────────────────────────────────────────────────────────────────────

def _tonemap_shading(img, scale=None):
    """Tonemap shading for display. If `scale` is given (a shared 90th-pct), use it so
    GT and pred shading share one exposure; else self-normalize (legacy single-image)."""
    t = np.nan_to_num(img, nan=0.0)
    t = np.clip(t, 0.0, None)
    if scale is None:
        scale = float(np.percentile(t, 90.0)) + 1e-6
        scale = min(scale, 1.5)
    return np.clip(t / scale, 0.0, 1.0)


def _albedo_scale(gt_albedo):
    img = np.nan_to_num(gt_albedo, nan=0.0)
    img = np.clip(img, 0.0, None)
    valid = img[img > 0.0]
    if valid.size < 100:
        return 1.0
    return float(np.percentile(valid, 99.9)) + 1e-6


def _show_albedo(img, scale):
    """Display an albedo map under a SHARED scale (from _albedo_scale), then gamma."""
    return np.power(np.clip(np.nan_to_num(img, 0.0) / scale, 0.0, 1.0), 0.45454545454545453)


def _normalize_albedo(img):
    img = np.nan_to_num(img, nan=0.0)
    img = np.clip(img, 0.0, None)
    valid = img[img > 0.01]
    if valid.size < 100:
        return np.power(np.clip(img, 0.0, 1.0), 0.45454545454545453)
    p99 = float(np.percentile(valid, 99.0)) + 1e-6
    return np.power(np.clip(img / p99, 0.0, 1.0), 0.45454545454545453)


def _gamma(img):
    return np.power(np.clip(np.nan_to_num(img, nan=0.0), 0.0, 1.0), 0.45454545454545453)


def _display_input(img):
    """Reinhard tonemap for display (fixes ARAP brightness bug with outdoor HDR)."""
    img = np.nan_to_num(img, nan=0.0)
    img = np.clip(img, 0.0, None)
    p80 = float(np.percentile(img, 80.0)) + 1e-6
    x = img / p80 * 0.8
    x = x / (1.0 + x) * 2.0
    return np.power(np.clip(x, 0.0, 1.0), 0.45454545454545453)


def _cov_heat(cov_pix, mask):
    """Auto-ranged CoV heatmap. Real albedo CoV is ~0.02–0.1, so clip(0,1) made the map
    almost uniformly blue. Scale to the masked 95th percentile so flicker is visible,
    and leave masked-out pixels black."""
    vals = cov_pix[mask > 0]
    if vals.size < 50:
        hi = 0.1
    else:
        hi = float(np.percentile(vals, 95.0)) + 1e-6
    heat = cv2.applyColorMap(
        (np.clip(cov_pix / hi, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat_u8 = heat.copy()
    heat_u8[mask == 0] = 0
    return heat_u8


def _to_u8_bgr(img_rgb_01):
    return cv2.cvtColor(
        (np.clip(img_rgb_01, 0, 255.0) / 255.0 if img_rgb_01.max() > 2.0 else
         np.clip(img_rgb_01 * 255.0, 0, 255)).astype(np.uint8),
        cv2.COLOR_RGB2BGR)


def _annotate(img, text, pos=(3, 22)):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)


def _scale_align_np(pred, gt, mask):
    """Least-squares single scalar mapping pred→gt over masked pixels (same as the
    metric's align_scale, in numpy). So the displayed pred matches the si-RMSE column."""
    m = (mask > 0).reshape(-1)
    if pred.ndim == 2:
        pred = pred[..., np.newaxis]
    if gt.ndim == 2:
        gt = gt[..., np.newaxis]
    p = pred.reshape(-1, pred.shape[-1])[m]
    g = gt.reshape(-1, gt.shape[-1])[m]
    c = float(np.sum(p * g)) / (float(np.sum(p * p)) + 1e-6)
    return pred * max(c, 1e-6)


def _make_contact_row(input_vis, a_gt, a_pred, s_gt, s_pred,
                      a_metrics, s_metrics, recon, infer_ms, name, valid_mask):
    """Build a contact row: [Input | GT Albedo | Pred Albedo | GT Shading | Pred Shading]."""
    al, ar, as_ = a_metrics
    sl, sr, ss = s_metrics

    a_scale = _albedo_scale(a_gt)
    a_pred_al = _scale_align_np(a_pred, a_gt, valid_mask)
    s_scale = max(float(np.percentile(np.nan_to_num(s_gt, 0.0), 90.0)), 1e-6)
    s_pred_al = _scale_align_np(s_pred, s_gt, valid_mask) if s_gt is not None else s_pred

    row_imgs = [
        _display_input(input_vis),       # input
        _show_albedo(a_gt, a_scale),     # GT albedo
        _show_albedo(a_pred_al, a_scale), # pred albedo aligned
    ]
    if s_gt is not None:
        row_imgs.append(_tonemap_shading(s_gt, scale=s_scale))
        row_imgs.append(_tonemap_shading(s_pred_al, scale=s_scale))

    H = max(im.shape[0] for im in row_imgs)
    resized = []
    for im in row_imgs:
        h, w = im.shape[:2]
        scale = H / max(h, 1)
        nw = max(1, int(round(w * scale)))
        if im.ndim == 2 or im.shape[-1] == 1:
            im = np.repeat(im[..., np.newaxis] if im.ndim == 2 else im, 3, axis=-1)
        resized.append(cv2.resize(_to_u8_bgr(im), (nw, H)))

    row = np.concatenate(resized, axis=1)
    # Info strip
    strip_h = 22
    strip = np.zeros((strip_h, row.shape[1], 3), dtype=np.uint8)
    txt = (f'{name}   Albedo LMSE/si-RMSE/SSIM: {al:.3f}/{ar:.3f}/{as_:.3f}'
           f'   Shading: {sl:.3f}/{sr:.3f}/{ss:.3f}'
           f'   Recon L1: {recon:.4f}   Infer: {infer_ms:.0f}ms')
    cv2.putText(strip, txt, (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
    return np.vstack([row, strip])


# ──────────────────────────────────────────────────────────────────────────────
# Constancy helpers
# ──────────────────────────────────────────────────────────────────────────────

def _group_cases(cases):
    """Group (base, stem, ext) into {base: [(stem, ext), ...]} keeping multi-light scenes."""
    groups = collections.defaultdict(list)
    for base, stem, ext in cases:
        groups[base].append((stem, ext))
    return {k: sorted(v) for k, v in groups.items() if len(v) >= 2}


def _frame_p99(rgb_linear):
    """99th-pct luminance of a linear frame (the exposure level the p99 tonemap keys on)."""
    a = np.nan_to_num(rgb_linear, nan=0.0)
    a = np.clip(a, 0.0, None)
    lum = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    return float(np.percentile(lum, 99.0))


def _group_frame_validity(p99s, is_hdr_flags, max_exposure_ratio):
    """Return bool array: True = frame is within exposure family of the group median."""
    valid = np.ones(len(p99s), dtype=bool)
    hdr = np.asarray(is_hdr_flags, dtype=np.float64)
    if not np.any(hdr > 0):
        return valid
    med = float(np.median([p for p, h in zip(p99s, is_hdr_flags) if h]))
    if med < 1e-12:
        return valid
    for i, (p, h) in enumerate(zip(p99s, is_hdr_flags)):
        if not h:
            continue
        ratio = float(p) / med if med > 1e-7 else 1.0
        if ratio > max_exposure_ratio or ratio < (1.0 / max_exposure_ratio):
            valid[i] = False
    return valid


def _illuminant_color_spread(frames, degenerate_ratio=0.15):
    """
    Returns (rg_vals, bg_vals, spread, is_degenerate) for a group of albedo predictions.
    spread: std of R/G + std of B/G across frames (the Cast axis).
    """
    rgs, bgs = [], []
    degenerate = []
    for rgb in frames:
        x = np.nan_to_num(rgb, nan=0.0)
        x = np.clip(x, 0.0, None)
        lum = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
        lo, hi = float(np.percentile(lum, 40)), float(np.percentile(lum, 90))
        m = (lum >= lo) & (lum <= hi)
        px = x[m]
        if px.shape[0] < 50:
            degenerate.append(True)
            continue
        rg = float(px[:, 0].mean()) / (float(px[:, 1].mean()) + 1e-6)
        bg = float(px[:, 2].mean()) / (float(px[:, 1].mean()) + 1e-6)
        rgs.append(rg)
        bgs.append(bg)
        degenerate.append(False)

    if len(rgs) < 2:
        return rgs, bgs, 0.0, True

    spread = float(np.std(rgs) + np.std(bgs))
    is_degen = float(sum(degenerate)) / max(len(degenerate), 1) > degenerate_ratio
    return rgs, bgs, spread, is_degen


# ──────────────────────────────────────────────────────────────────────────────
# Constancy evaluation (thesis primary)
# ──────────────────────────────────────────────────────────────────────────────

def eval_arap_constancy(args):
    """Cross-illumination constancy: C_arap, Cast_RMS, coloured/direction/degenerate breakdown."""
    version_arg = getattr(args, 'model_version', 'auto')
    version_int = None

    print('\n=============================================')
    print(f'ARAP CROSS-ILLUMINATION CONSTANCY: {args.label or os.path.basename(args.checkpoint)}')
    print(f'  ({version_arg})')

    device = resolve_device(args.device, getattr(args, 'cuda_index', None))

    if _is_external_version(version_arg):
        model = _load_marigold_pipeline(args.checkpoint, device)
        version_for_input = version_arg
    elif _is_crefnet_version(version_arg):
        model = load_crefnet(args.checkpoint, device, variant=version_arg)
        version_for_input = version_arg
    elif _is_ordinal_version(version_arg):
        model = load_ordinal(device, variant=version_arg)
        version_for_input = version_arg
    else:
        model = _load_model_versioned(args.checkpoint, version_arg, device)
        version_for_input = version_arg

    cases = get_arap_cases(args.dataset_dir)
    cases = filter_cases_by_domain(cases, getattr(args, 'scene_filter', 'all'))
    groups = _group_cases(cases)

    _lim = getattr(args, 'limit_groups', 0)
    if _lim and _lim > 0:
        groups = dict(sorted(groups.items())[:_lim])
        print(f'  --limit_groups {_lim}: evaluating first {len(groups)} scene-groups only')

    label = args.label or os.path.basename(args.checkpoint)

    n_frames_total = sum(len(v) for v in groups.values())
    _input_desc = 'WHITE-BALANCED input (standard protocol)' if getattr(args, 'white_balance', False) \
        else 'RAW colored input'
    print(f'  {len(groups)} multi-light groups ({n_frames_total} frames), {_input_desc}')
    print('=============================================')

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    per_group_C = []
    per_group_cast = []
    per_group_spread = []
    per_group_degen = []
    per_group_base = []
    per_group_sat_ratio = []    # predicted albedo saturation / GT albedo saturation
    per_group_chroma_err = []   # |mean predicted chroma - GT chroma|
    all_a_rmse = []
    all_a_si_rmse = []
    all_a_lmse = []
    all_a_ssim = []
    per_group_rows = []

    def _predict_albedo(rgb_linear, is_hdr, img_path):
        """Return predicted albedo (H,W,3) at input resolution for either backbone."""
        if _is_external_version(version_for_input):
            ad, _ = _run_marigold_inference(model, rgb_linear, is_hdr,
                                             max_size=getattr(args, 'max_size', 1280))
        elif _is_crefnet_version(version_for_input):
            rgb_display = _external_display_linear(rgb_linear, is_hdr)
            ad, _ = run_crefnet(model, rgb_display, getattr(args, 'max_size', 1280), device)
        elif _is_ordinal_version(version_for_input):
            rgb_display = _external_display_linear(rgb_linear, is_hdr)
            ad, _ = run_ordinal(model, rgb_display, getattr(args, 'max_size', 1280), device)
        else:
            ad, _, _ = _run_own_model_inference(
                model, version_for_input, rgb_linear, is_hdr, img_path,
                device, getattr(args, 'max_size', 1280), getattr(args, 'min_size', 1024))
        return ad

    for base, stems_exts in tqdm(sorted(groups.items()), desc='ARAP constancy'):
        albedo_gt, _ = load_gt(args.dataset_dir, base, stems_exts[0][0], stems_exts[0][1])
        if albedo_gt is None:
            continue

        loaded = []
        for stem, ext in stems_exts:
            img_path = _find_file(args.dataset_dir, stem, [f'.{ext}'] + list(INPUT_EXTS))
            if img_path is None:
                continue
            rgb_linear, is_hdr = load_image(img_path)
            # Standard-protocol white balance (Ordinal/CD-IID): reconstruct the scene under an
            # achromatic illuminant via I_wb = A* * luminance(I / A*). This ACTUALLY removes the
            # coloured illuminant; the previous `_hdr_norm` call here did not (it is a single
            # scalar multiply and leaves chromaticity untouched — see _white_balance_gt).
            # Off by default (thesis = raw coloured input, which exercises constancy).
            # NOTE: applied to LDR frames too. The old `and is_hdr` guard silently skipped every
            # .jpg/.png scene, so those were scored on raw input even with --white_balance set.
            if getattr(args, 'white_balance', False):
                # Frames in a group need not share the GT albedo's resolution (e.g. a 768x768
                # frame against a 720x1280 albedo), so align before the per-pixel division.
                _a = albedo_gt
                if _a.shape[:2] != rgb_linear.shape[:2]:
                    _a = cv2.resize(_a, (rgb_linear.shape[1], rgb_linear.shape[0]),
                                    interpolation=cv2.INTER_LINEAR)
                rgb_linear = _white_balance_gt(rgb_linear, _a)
                is_hdr = False
            loaded.append((stem, rgb_linear, is_hdr, img_path))

        if len(loaded) < 2:
            continue

        # Exposure family filtering
        p99s = [_frame_p99(x[1]) for x in loaded]
        hdr_flags = [x[2] for x in loaded]
        frame_valid = _group_frame_validity(p99s, hdr_flags, max_exposure_ratio=4.6e-05 * 1e5 or 4.0)
        n_drop = int(np.sum(~frame_valid))
        if n_drop > 0:
            print(f'  [ {base} ] dropped {n_drop}/{len(loaded)} frame(s) '
                  f'(exposure out of family): {[loaded[i][0] for i in range(len(loaded)) if not frame_valid[i]]}')

        valid_frames = [x for x, v in zip(loaded, frame_valid) if v]
        if len(valid_frames) < 2:
            continue

        # Predict albedo for all valid frames
        color_spread = _illuminant_color_spread([x[1] for x in valid_frames])
        rg_vals, bg_vals, spread, is_degen = color_spread

        preds = []
        inputs_tm = []
        ref_hw = albedo_gt.shape[:2]

        for stem, rgb_linear, is_hdr, img_path in valid_frames:
            pred_ad = _predict_albedo(rgb_linear, is_hdr, img_path)
            if pred_ad.shape[:2] != ref_hw:
                pred_ad = cv2.resize(pred_ad, (ref_hw[1], ref_hw[0]), interpolation=cv2.INTER_LINEAR)
            preds.append(pred_ad)
            if is_hdr:
                disp = _hdr_norm(rgb_linear, 99.0)
            else:
                disp = np.clip(rgb_linear, 0.0, 1.0)
            if disp.shape[:2] != ref_hw:
                disp = cv2.resize(disp, (ref_hw[1], ref_hw[0]), interpolation=cv2.INTER_LINEAR)
            inputs_tm.append(disp)

        if len(preds) < 2:
            continue

        # ── C_arap: cross-light CoV of albedo luminance ────────────────
        H, W = ref_hw
        stack = np.stack(preds, axis=0).astype(np.float32)  # (N, H, W, 3)
        gmean = stack.mean(axis=0)  # (H, W, 3)

        # Luminance CoV
        lum_stack = (0.2126 * stack[..., 0] + 0.7152 * stack[..., 1] +
                     0.0722 * stack[..., 2])  # (N, H, W)
        mu = lum_stack.mean(axis=0)
        sd = lum_stack.std(axis=0)

        # Valid mask: exclude specular/shadow via GT albedo lum
        alb_lum = (0.299 * albedo_gt[..., 0] + 0.587 * albedo_gt[..., 1] +
                   0.114 * albedo_gt[..., 2])
        mask = (alb_lum > 0.004)

        if mask.sum() < 100:
            continue

        cov_pix = sd / (mu + 1e-6)
        C_g = float(cov_pix[mask].mean())
        per_group_C.append(C_g)

        # ── Cast: chroma drift ─────────────────────────────────────────
        cov_dir = float(np.std(rg_vals) + np.std(bg_vals)) if len(rg_vals) >= 2 else 0.0
        rg = [float(preds[k][mask, 0].mean() / (preds[k][mask, 1].mean() + 1e-6))
              for k in range(len(preds))]
        bg = [float(preds[k][mask, 2].mean() / (preds[k][mask, 1].mean() + 1e-6))
              for k in range(len(preds))]
        r_over_g = np.array(rg)
        b_over_g = np.array(bg)
        cast = float(np.sqrt(np.var(r_over_g) + np.var(b_over_g)))
        per_group_cast.append(cast)
        per_group_spread.append(spread)

        # ── Degeneracy guard: cast is an ABSOLUTE chroma variance, so a model that
        # predicts flat grey albedo scores a perfect 0. Cast is therefore only
        # meaningful read against chroma fidelity vs the GT albedo (1.0 = faithful).
        # ARAP's GT albedo is mixed-scale (.hdr files peak near 0.005, .jpg near 255)
        # while predictions are ~[0,1], so saturation must be computed scale-free:
        # rescale each set of pixels by its own 99th percentile before the ratio, or a
        # fixed epsilon silently dominates the denominator of the dim HDR albedos.
        def _sat(px):
            s = float(np.percentile(px, 99.0))
            if not np.isfinite(s) or s <= 0:
                return float('nan')
            q = px / s
            mx = q.max(axis=-1)
            return float(np.mean((mx - q.min(axis=-1)) / (mx + 1e-3)))

        s_gt = _sat(albedo_gt[mask])
        s_pr = float(np.mean([_sat(preds[k][mask]) for k in range(len(preds))]))
        per_group_sat_ratio.append(
            s_pr / s_gt if np.isfinite(s_gt) and s_gt > 1e-3 else float('nan'))

        g_gt = albedo_gt[mask][:, 1].mean() + 1e-6
        gt_rg = float(albedo_gt[mask][:, 0].mean() / g_gt)
        gt_bg = float(albedo_gt[mask][:, 2].mean() / g_gt)
        per_group_chroma_err.append(
            float(np.hypot(float(r_over_g.mean()) - gt_rg, float(b_over_g.mean()) - gt_bg)))
        per_group_degen.append(is_degen)
        per_group_base.append(base)

        # ── GT accuracy ────────────────────────────────────────────────
        t_mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        a_gt_t = torch.from_numpy(albedo_gt.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        for p in preds:
            p_t = torch.from_numpy(p.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
            try:
                l = _compute_lmse(p_t, a_gt_t, t_mask).item()
                r = _cd_iid_rmse(p_t, a_gt_t, t_mask)
                si_r = _masked_scale_invariant_rmse(p_t, a_gt_t, t_mask).item()
                s = _compute_ssim_bounded(p_t, a_gt_t, t_mask)
                all_a_rmse.append(r)
                all_a_si_rmse.append(si_r)
                all_a_lmse.append(l)
                all_a_ssim.append(float(s))
            except Exception:
                pass

        # ── Visualization ──────────────────────────────────────────────
        if args.save_dir:
            disp = inputs_tm[0] if inputs_tm else gmean
            preds_al = [_scale_align_np(p, albedo_gt, mask.astype(np.float32)) for p in preds]
            a_scale = _albedo_scale(albedo_gt)
            tiles = []
            for k, (pa, pv) in enumerate(zip(preds_al, inputs_tm)):
                pred_al = _show_albedo(pa, a_scale)
                tile_pred = _to_u8_bgr(pred_al)
                tile_in = _to_u8_bgr(_display_input(pv))
                Hh = 200
                tile_pred = cv2.resize(tile_pred, (int(W * Hh / max(H, 1)), Hh))
                tile_in = cv2.resize(tile_in, (int(W * Hh / max(H, 1)), Hh))
                tiles += [tile_in, tile_pred]

            # GT
            tile_gt = _to_u8_bgr(_show_albedo(albedo_gt, a_scale))
            tile_gt = cv2.resize(tile_gt, (int(W * 200 / max(H, 1)), 200))
            _annotate(tile_gt, 'GT Albedo')
            tiles.insert(0, tile_gt)

            # CoV heatmap
            heat_u8 = _cov_heat(cov_pix, mask.astype(np.uint8))
            heat_hi = float(np.percentile(cov_pix[mask], 95.0)) if mask.sum() > 0 else 0.1
            heat_tile = cv2.resize(heat_u8, (int(W * 200 / max(H, 1)), 200))
            _annotate(heat_tile, f'CoV {C_g:.3f} (max {heat_hi:.2f})')
            tiles.append(heat_tile)

            Hh = max(t.shape[0] for t in tiles)
            row = np.concatenate([cv2.resize(t, (t.shape[1], Hh)) for t in tiles], axis=1)
            strip = np.zeros((22, row.shape[1], 3), dtype=np.uint8)
            txt = (f'C_arap: {C_g:.4f}   Cast_RMS: {cast:.4f}'
                   f'   Albedo RMSE: {np.mean(all_a_rmse[-len(preds):]):.4f}'
                   f'   LMSE: {np.mean(all_a_lmse[-len(preds):]):.4f}'
                   f'   n_lights: {len(preds)}')
            cv2.putText(strip, txt, (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
            per_group_rows.append(np.vstack([row, strip]))

            safe = base.lstrip('_').replace('/', '_')
            cv2.imwrite(os.path.join(args.save_dir, f'{safe}.png'), row)

    # ── Sheet ───────────────────────────────────────────────────────────
    if args.save_dir and per_group_rows:
        # Scale each row to the same width so the sheet has no black dead space.
        wmax = max(r.shape[1] for r in per_group_rows)
        scaled = []
        for r in per_group_rows[:getattr(args, 'max_vis', 12)]:
            if r.shape[1] != wmax:
                r = cv2.resize(r, (wmax, r.shape[0]), interpolation=cv2.INTER_LINEAR)
            scaled.append(r)
        sheet = np.vstack(scaled)
        sheet_path = os.path.join(args.save_dir, f'{label}_constancy_sheet.jpg')
        cv2.imwrite(sheet_path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])

    # ── Summary ─────────────────────────────────────────────────────────
    C_arap = float(np.nanmean(per_group_C)) if per_group_C else float('nan')
    Cast_RMS = float(np.nanmean(per_group_cast)) if per_group_cast else float('nan')
    a_rmse = float(np.nanmean(all_a_rmse)) if all_a_rmse else float('nan')
    a_si_rmse = float(np.nanmean(all_a_si_rmse)) if all_a_si_rmse else float('nan')
    a_lmse = float(np.nanmean(all_a_lmse)) if all_a_lmse else float('nan')
    a_ssim = float(np.nanmean(all_a_ssim)) if all_a_ssim else float('nan')

    Sat_ratio = float(np.nanmean(per_group_sat_ratio)) if per_group_sat_ratio else float('nan')
    Chroma_err = float(np.nanmean(per_group_chroma_err)) if per_group_chroma_err else float('nan')

    print(f'\n--- ARAP constancy: [{label}] ---')
    print(f'Groups evaluated : {len(per_group_C)}')
    print(f'C_arap (cross-light CoV)   : {C_arap:.4f}   [LOWER = more invariant]')
    print(f'Cast_RMS (chroma drift)    : {Cast_RMS:.4f}   [LOWER = less cast leak]')
    print(f'  Sat_ratio (pred/GT sat)  : {Sat_ratio:.4f}   [DEGENERACY GUARD — 1.0 = faithful, '
          f'<1 = colour collapsed; Cast_RMS is trivially won by a flat grey albedo, so it is '
          f'only meaningful read together with this]')
    print(f'  Chroma_err vs GT         : {Chroma_err:.4f}   [LOWER = truer albedo hue]')
    print(f'Albedo RMSE vs GT          : {a_rmse:.4f}   [CD-IID-compatible, lower is better]')
    print(f'Albedo si-RMSE vs GT       : {a_si_rmse:.4f}   [scale-normalized diagnostic]')
    print(f'Albedo LMSE vs GT          : {a_lmse:.4f}')

    # Colored vs direction breakdown
    thr = getattr(args, 'color_spread_thr', 0.15)
    print(f'  ── colored-illuminant breakdown (spread thr={thr}) ──')
    colored_sel = [not d and s >= thr for d, s in zip(per_group_degen, per_group_spread)]
    direction_sel = [not d and s < thr for d, s in zip(per_group_degen, per_group_spread)]
    degen_sel = per_group_degen

    def _bin(sel):
        if not any(sel):
            return 0, float('nan'), float('nan')
        n = int(sum(sel))
        return (n,
                float(np.nanmean([c for c, s in zip(per_group_C, sel) if s])),
                float(np.nanmean([c for c, s in zip(per_group_cast, sel) if s])))

    c_n, c_carap, c_cast = _bin(colored_sel)
    d_n, d_carap, d_cast = _bin(direction_sel)
    b_n, b_carap, b_cast = _bin(degen_sel)

    print(f'  COLORED   (light color varies) n={c_n}: C_arap={c_carap:.4f}  '
          f'Cast_RMS={c_cast:.4f}   ← thesis axis')
    print(f'  DIRECTION (intensity/dir only) n={d_n}: C_arap={d_carap:.4f}  Cast_RMS={d_cast:.4f}')
    print(f'  DEGENERATE (unobservable chan) n={b_n}: C_arap={b_carap:.4f}   (reported, excluded from above)')
    if per_group_degen:
        degen_scenes = [b for b, d in zip(per_group_base, degen_sel) if d]
        print(f'             scenes: {degen_scenes[:5]}')

    summary = {
        'label': label,
        'n_groups': len(per_group_C),
        'C_arap': C_arap,
        'Cast_RMS': Cast_RMS,
        'Sat_ratio': Sat_ratio,      # degeneracy guard: read Cast_RMS only alongside this
        'Chroma_err': Chroma_err,
        'albedo_rmse': a_rmse,
        'albedo_si_rmse': a_si_rmse,
        'albedo_lmse': a_lmse,
        'albedo_ssim': a_ssim,
        'colored': {'n': c_n, 'C_arap': c_carap, 'Cast_RMS': c_cast},
        'direction': {'n': d_n, 'C_arap': d_carap, 'Cast_RMS': d_cast},
        'degenerate': {'n': b_n, 'C_arap': b_carap, 'degenerate_scenes': per_group_base},
        'all': per_group_C,
        'per_group': [
            {
                'scene': base,
                'C_arap': float(c),
                'Cast_RMS': float(cast),
                'illuminant_spread': float(spread),
                'degenerate': bool(degen),
            }
            for base, c, cast, spread, degen in zip(
                per_group_base, per_group_C, per_group_cast,
                per_group_spread, per_group_degen)
        ],
    }

    save_json = getattr(args, 'save_json', None)
    if save_json:
        json_key = label.replace(' ', '_')
        json_path = str(ROOT_DIR / save_json) if not os.path.isabs(str(save_json)) else str(save_json)
        data = {}
        if os.path.exists(json_path):
            try:
                with open(json_path) as f:
                    data = json.load(f)
            except Exception:
                pass
        data[json_key] = summary
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f'→ appended to {json_path}  (key: {json_key}')

    return summary


# ──────────────────────────────────────────────────────────────────────────────
# Full eval (per-image metrics + contact sheets)
# ──────────────────────────────────────────────────────────────────────────────

def eval_arap(args):
    device = resolve_device(args.device, getattr(args, 'cuda_index', None))

    print('\n=============================================')
    mode = 'WHITE-BALANCED input (Ordinal Shading protocol — illuminant cast removed)' \
           if getattr(args, 'white_balance', False) else \
           'RAW colored input (default — exercises cross-illumination constancy)'
    print(f'Evaluating: {args.checkpoint}')
    print(f'Input mode: {mode}')
    print('=============================================')

    version_arg = getattr(args, 'model_version', 'auto')
    is_marigold = _is_external_version(version_arg)

    is_crefnet = _is_crefnet_version(version_arg)
    is_ordinal = _is_ordinal_version(version_arg)
    if is_marigold:
        if version_arg == 'marigold-appearance':
            print('Note: appearance model — pseudo shading I/A will be used.')
        pipe = _load_marigold_pipeline(args.checkpoint, device)
        print(f'Loaded Marigold {version_arg} from {args.checkpoint}')
        model = pipe
        inferred_version = version_arg
        version_int = None
    elif is_crefnet:
        # The accuracy path previously fell through to _load_model_versioned for every
        # non-Marigold model, which broke CRefNet (torch>=2.6 weights_only pickle error) and
        # Ordinal (tries to open the hub name as a file). Route both through the same adapters
        # the constancy path already uses. (Fix 2026-07-15.)
        model = load_crefnet(args.checkpoint, device, variant=version_arg)
        inferred_version = version_arg
        version_int = None
        print(f'Loaded CRefNet ({version_arg})')
    elif is_ordinal:
        model = load_ordinal(device, variant=version_arg)
        inferred_version = version_arg
        version_int = None
        print(f'Loaded Ordinal ({version_arg})')
    else:
        model = _load_model_versioned(args.checkpoint, version_arg, device)
        inferred_version = version_arg
        version_int = int(version_arg) if str(version_arg).isdigit() else 17
        print(f'Loaded V{inferred_version} model from {args.checkpoint}')

    dataset_dir = args.dataset_dir
    cases = get_arap_cases(dataset_dir)
    cases = filter_cases_by_domain(cases, getattr(args, 'scene_filter', 'all'))
    _lim = getattr(args, 'limit_groups', 0)
    if _lim and _lim > 0:
        bases_keep = sorted({c[0] for c in cases})[:_lim]
        cases = [c for c in cases if c[0] in bases_keep]
        print(f'  --limit_groups {_lim}: evaluating first {len(bases_keep)} scene-groups only')
    print(f'Found {len(cases)} evaluable cases in {dataset_dir}')

    save_dir = getattr(args, 'save_dir', None)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        print(f'Contact sheets will be saved to: {save_dir}')

    all_a_lmse, all_a_rmse, all_a_si_rmse, all_a_ssim = [], [], [], []
    all_s_lmse, all_s_rmse, all_s_si_rmse, all_s_ssim = [], [], [], []
    all_recon, all_infer_ms = [], []
    hdr_a_rmse, ldr_a_rmse = [], []
    contact_rows = []

    max_size = getattr(args, 'max_size', 1280)
    min_size = getattr(args, 'min_size', 1024)

    for base_name, input_stem, ext in tqdm(cases, desc='Evaluating ARAP'):
        case_label = input_stem.lstrip('_')
        img_path = _find_file(dataset_dir, input_stem, [f'.{ext}'] + list(INPUT_EXTS))
        if img_path is None:
            print(f'  skip: no input found for {input_stem}')
            continue

        albedo_gt, shading_gt = load_gt(dataset_dir, base_name, input_stem, ext)
        if albedo_gt is None:
            print(f'  skip: albedo GT missing for {input_stem}')
            continue

        rgb_linear, is_hdr = load_image(img_path)
        H_orig, W_orig = rgb_linear.shape[:2]

        # Standard-protocol white balance for the Ordinal Shading comparison. See
        # _white_balance_gt: the previous `_hdr_norm` call was exposure normalisation and
        # removed NO illuminant colour. Requires albedo_gt at native resolution, so it runs
        # before the resize below.
        if getattr(args, 'white_balance', False):
            _a_gt = albedo_gt
            if _a_gt.shape[:2] != (H_orig, W_orig):
                _a_gt = cv2.resize(_a_gt, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
            rgb_linear = _white_balance_gt(rgb_linear, _a_gt)
            is_hdr = False

        if albedo_gt.shape[:2] != (H_orig, W_orig):
            albedo_gt = cv2.resize(albedo_gt, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
        if shading_gt is not None and shading_gt.shape[:2] != (H_orig, W_orig):
            shading_gt = cv2.resize(shading_gt, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)

        # Shading luminance
        if shading_gt is not None:
            alb_lum = (0.299 * albedo_gt[..., 0] + 0.587 * albedo_gt[..., 1] +
                       0.114 * albedo_gt[..., 2])
            shd_lum = (0.299 * shading_gt[..., 0] + 0.587 * shading_gt[..., 1] +
                       0.114 * shading_gt[..., 2]) if shading_gt.ndim == 3 else shading_gt

        # Inference
        t0 = time.perf_counter()
        if is_marigold:
            model_input = rgb_linear.copy()
            input_is_hdr = is_hdr
            pred_ad, pred_sd = _run_marigold_inference(pipe, model_input, input_is_hdr,
                                                        max_size=max_size)
            rgb_tm_orig = _external_display_linear(rgb_linear, is_hdr)
        elif is_crefnet or is_ordinal:
            # Same adapter path the constancy branch uses: both take display-linear input
            # and return albedo only (no shading), so pred_sd stays None.
            rgb_disp = _external_display_linear(rgb_linear, is_hdr)
            runner = run_crefnet if is_crefnet else run_ordinal
            pred_ad, _ = runner(model, rgb_disp, max_size, device)
            pred_sd = None
            rgb_tm_orig = rgb_disp
        else:
            pred_ad, pred_sd, rgb_tm_orig = _run_own_model_inference(
                model, inferred_version, rgb_linear, is_hdr, img_path, device, max_size, min_size)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        infer_ms = (time.perf_counter() - t0) * 1000.0

        # Pad pred to GT size if needed
        if pred_ad.shape[:2] != (H_orig, W_orig):
            pred_ad = cv2.resize(pred_ad, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
        if pred_sd is not None and pred_sd.shape[:2] != (H_orig, W_orig):
            pred_sd = cv2.resize(pred_sd, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)

        # Ensure 3-channel
        def _ensure3(arr):
            if arr.ndim == 2 or arr.shape[-1] == 1:
                return np.repeat(arr[..., np.newaxis] if arr.ndim == 2 else arr, 3, axis=-1)
            return arr

        pred_ad = _ensure3(pred_ad)
        albedo_gt = _ensure3(albedo_gt)
        if shading_gt is not None:
            shading_gt = _ensure3(shading_gt)
        if pred_sd is not None:
            pred_sd = _ensure3(pred_sd)

        # Valid mask: mid-range luminance
        alb_lum = (0.2126 * albedo_gt[..., 0] + 0.7152 * albedo_gt[..., 1] +
                   0.0722 * albedo_gt[..., 2])
        valid_mask = (alb_lum > 0.004).astype(np.float32)
        if valid_mask.sum() < 88:
            continue

        # Tensor conversion
        def _t(arr):
            return torch.from_numpy(arr.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)

        t_mask = torch.from_numpy(valid_mask).unsqueeze(0).unsqueeze(0).to(device)
        a_gt_t = _t(albedo_gt)
        a_pred_t = _t(pred_ad)

        try:
            a_lmse, a_rmse, a_si_rmse, a_ssim = compute_metrics(a_pred_t, a_gt_t, t_mask, 'albedo')
        except Exception:
            a_lmse, a_rmse, a_si_rmse, a_ssim = (float('nan'),) * 4

        all_a_lmse.append(a_lmse)
        all_a_rmse.append(a_rmse)
        all_a_si_rmse.append(a_si_rmse)
        all_a_ssim.append(a_ssim)
        (hdr_a_rmse if is_hdr else ldr_a_rmse).append(a_rmse)

        s_lmse = s_rmse = s_si_rmse = s_ssim = float('nan')
        recon = float('nan')
        if shading_gt is not None and pred_sd is not None:
            s_gt_t = _t(shading_gt)
            s_pred_t = _t(pred_sd)
            try:
                s_lmse, s_rmse, s_si_rmse, s_ssim = compute_metrics(s_pred_t, s_gt_t, t_mask, 'shading')
                recon = compute_diffuse_recon(pred_ad, pred_sd, albedo_gt, shading_gt)
            except Exception:
                pass

        all_s_lmse.append(s_lmse)
        all_s_rmse.append(s_rmse)
        all_s_si_rmse.append(s_si_rmse)
        all_s_ssim.append(s_ssim)
        all_recon.append(recon if not np.isnan(recon) else 0.0)
        all_infer_ms.append(infer_ms)

        # Optional guided filter post-processing
        if getattr(args, 'guided_filter_iters', 0) > 0 and guided_filter is not None:
            for _ in range(args.guided_filter_iters):
                pred_ad = guided_filter(rgb_tm_orig, pred_ad,
                                        r=getattr(args, 'gf_r', 8),
                                        eps=getattr(args, 'gf_eps', 4.6e-05))

        # Contact sheets
        if save_dir and len(contact_rows) < getattr(args, 'max_vis', 12):
            try:
                contact_row = _make_contact_row(
                    rgb_tm_orig, albedo_gt, pred_ad,
                    shading_gt if shading_gt is not None else np.zeros_like(albedo_gt),
                    pred_sd if pred_sd is not None else np.zeros_like(pred_ad),
                    (a_lmse, a_si_rmse, a_ssim), (s_lmse, s_si_rmse, s_ssim),
                    recon if not np.isnan(recon) else 0.0, infer_ms,
                    case_label, valid_mask)
                contact_rows.append(contact_row)
                save_path = os.path.join(save_dir, f'{case_label}_contact_sheet.jpg')
                cv2.imwrite(save_path, contact_row)
            except Exception:
                pass

    # Sheet
    N = len(all_a_lmse)
    if save_dir and contact_rows:
        wmax = max(r.shape[1] for r in contact_rows)
        padded = [cv2.copyMakeBorder(r, 0, 0, 0, wmax - r.shape[1], cv2.BORDER_CONSTANT)
                  for r in contact_rows]
        sheet = np.vstack(padded)
        sheet_path = os.path.join(save_dir, f'{Path(args.checkpoint).name}_arap_sheet.jpg')
        cv2.imwrite(sheet_path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f'Contact sheet saved → {sheet_path}')

    if N == 0:
        print('No images evaluated.')
        return

    print(f'\n--- Results for {args.checkpoint}  ({N} images) ---')
    print(f'Albedo  - LMSE: {np.mean(all_a_lmse):.4f}  RMSE: {np.mean(all_a_rmse):.4f}  '
          f'si-RMSE: {np.mean(all_a_si_rmse):.4f}  SSIM: {np.mean(all_a_ssim):.4f}')
    print(f'Shading - LMSE: {np.nanmean(all_s_lmse):.4f}  RMSE: {np.nanmean(all_s_rmse):.4f}  '
          f'si-RMSE: {np.nanmean(all_s_si_rmse):.4f}  SSIM: {np.nanmean(all_s_ssim):.4f}')
    print(f'Diffuse recon L1: {np.mean(all_recon):.4f}'
          f'  (by input: HDR albedo RMSE {np.nanmean(hdr_a_rmse):.4f} '
          f'[{len(hdr_a_rmse)}]  |  LDR albedo RMSE {np.nanmean(ldr_a_rmse):.4f} '
          f'[{len(ldr_a_rmse)}])')

    times = [t for t in all_infer_ms if t < 1000]
    if times:
        print(f'Inference time  : {np.mean(times):.0f} ms/img  (median {np.median(times):.0f} ms,'
              f'  total {sum(all_infer_ms)/1000.0:.1f} s)')


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default='tests/testing_data/ARAP_dataset')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--model_version', default='auto',
                        help='auto | 17 | 18 | 20 | marigold-lighting | marigold-appearance')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--cuda_index', type=int, default=None)
    parser.add_argument('--max_size', type=int, default=1280)
    parser.add_argument('--min_size', type=int, default=1024)
    parser.add_argument('--save_dir', default=None)
    parser.add_argument('--save_cov', default=None)
    parser.add_argument('--scene_filter', default='all',
                        help='all | indoor | outdoor')
    parser.add_argument('--color_spread_thr', type=float, default=0.15)
    parser.add_argument('--max_vis', type=int, default=12,
                        help='Save contact rows for at most this many cases/groups '
                             '(ALL cases are still scored — this only caps visualization '
                             'output to keep sheets readable and fast). Default 12.')
    parser.add_argument('--white_balance', action='store_true')
    parser.add_argument('--constancy', action='store_true')
    parser.add_argument('--label', default=None,
                        help='Row label in the constancy JSON/printout (e.g. 40k_r4, marigold-app). '
                             'Defaults to the checkpoint basename.')
    parser.add_argument('--save_json', default=None,
                        help="Constancy mode: append this run's summary to a shared JSON keyed "
                             "by --label, so multiple checkpoints accumulate into one table.")
    parser.add_argument('--guided_filter_iters', type=int, default=0)
    parser.add_argument('--gf_r', type=int, default=8)
    parser.add_argument('--gf_eps', type=float, default=4.6e-05)
    parser.add_argument('--limit_groups', type=int, default=0,
                        help='Quick-test cap: evaluate only the first N scene-groups '
                             '(0 = no limit). Applies to both constancy and accuracy modes.')
    args = parser.parse_args()

    # Resolve dataset_dir relative to ROOT_DIR if needed
    if not os.path.isabs(args.dataset_dir):
        args.dataset_dir = str(ROOT_DIR / args.dataset_dir)

    if args.constancy:
        eval_arap_constancy(args)
    else:
        eval_arap(args)
