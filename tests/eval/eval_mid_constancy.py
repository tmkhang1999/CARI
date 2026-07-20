"""Cross-illumination constancy evaluation on MIDIntrinsics.

Metrics (lower = more invariant):
  C_mat       : within-material CoV of albedo luminance across lighting conditions
  R_cast_rg   : std of R/G ratio across lightings (per-material, scene-averaged)
  R_cast_bg   : std of B/G ratio across lightings (per-material, scene-averaged)
  Cast_RMS    : sqrt(R_cast_rg^2 + R_cast_bg^2) — combined chroma cast
  M_albedo    : scale-invariant log-MSE vs GT albedo (in-domain models only)
  sat_bin_mae : per-saturation-quartile albedo MAE (in-domain only)
"""
import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

from src.models import IntrinsicDecompositionV17, IntrinsicDecompositionV17Refiner, IntrinsicDecompositionV20
from src.data.hypersim_dataset import _compute_tonemap_scale, _tonemap_linear

_MARIGOLD_PATH = ROOT / 'documents/references/marigold'
_MARIGOLD_VERSIONS = {'marigold-appearance', 'marigold-lighting'}
# External SOTA models wired via the shared adapters (same ones eval_maw/eval_arap use).
# Their contract: display-linear [0,1] HWC in -> linear albedo [0,1] HWC out, i.e. exactly
# what AlbedoPredictor.albedo already receives as rgb_tm and returns.
_CREFNET_VERSIONS = {'crefnet', 'crefnet-e'}
_ORDINAL_VERSIONS = {'ordinal', 'ordinal-rendered-only'}
_EXTERNAL_ADAPTER_VERSIONS = _CREFNET_VERSIONS | _ORDINAL_VERSIONS

MarigoldIIDPipeline = None
PILImage = None
_load_crefnet = _run_crefnet = None
_load_ordinal = _run_ordinal = None


def _ensure_marigold_imported():
    global MarigoldIIDPipeline, PILImage
    if MarigoldIIDPipeline is not None and PILImage is not None:
        return
    sys.path.insert(0, str(_MARIGOLD_PATH))
    from marigold import MarigoldIIDPipeline as _MarigoldIIDPipeline
    from PIL import Image as _PILImage
    MarigoldIIDPipeline = _MarigoldIIDPipeline
    PILImage = _PILImage


def _ensure_crefnet_imported():
    global _load_crefnet, _run_crefnet
    if _load_crefnet is not None:
        return
    from crefnet_adapter import load_crefnet, run_crefnet  # tests/eval is on sys.path (script dir)
    _load_crefnet, _run_crefnet = load_crefnet, run_crefnet


def _ensure_ordinal_imported():
    global _load_ordinal, _run_ordinal
    if _load_ordinal is not None:
        return
    from ordinal_adapter import load_ordinal, run_ordinal
    _load_ordinal, _run_ordinal = load_ordinal, run_ordinal


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_v17(ckpt_path, device):
    """Load a V17 or V20 checkpoint with shape-filtered state dict."""
    state = torch.load(ckpt_path, map_location='cpu')
    cfg = state.get('config', {})
    model_cfg = cfg.get('model', {})
    version_value = float(model_cfg.get('version', 17))
    version = int(version_value)

    if version == 20:
        model = IntrinsicDecompositionV20(model_cfg).to(device)
    elif abs(version_value - 17.27) < 1e-6:
        model = IntrinsicDecompositionV17Refiner(model_cfg).to(device)
    else:
        model = IntrinsicDecompositionV17(model_cfg).to(device)

    ms = state.get('model_state_dict', state.get('model', {}))
    own = model.state_dict()
    filtered = {k: v for k, v in ms.items() if k in own and v.shape == own[k].shape}
    missing = [k for k in own if k not in filtered]
    if missing:
        print(f'  [warn] {len(missing)} keys not loaded (shape mismatch / new keys): ...')
    model.load_state_dict(filtered, strict=False)
    step = state.get('global_step', state.get('iteration', '?'))
    print(f'  Loaded {Path(ckpt_path).name}')
    print(f'  step={step}  ({len(filtered)}/{len(own)} keys)')
    return model.eval()


def _load_marigold(ckpt_path, device):
    _ensure_marigold_imported()
    pipe = MarigoldIIDPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float32).to(device)
    pipe.set_progress_bar_config(disable=True)
    print(f'  Loaded Marigold from {ckpt_path}')
    return pipe


# ──────────────────────────────────────────────────────────────────────────────
# Predictor wrapper
# ──────────────────────────────────────────────────────────────────────────────

class AlbedoPredictor:
    """Unified interface for V17, V20, Marigold, CRefNet, and Ordinal Shading models."""

    def __init__(self, ckpt_path: str, version, device, infer_max_size=None):
        self.version = version
        self.device = device
        # Long-side cap for our-model inference. None = run at the frame's native
        # resolution (MID mip2 = 1500px), which is the default this eval has always used.
        self.infer_max_size = infer_max_size
        self.is_marigold = version in _MARIGOLD_VERSIONS
        self.is_crefnet = version in _CREFNET_VERSIONS
        self.is_ordinal = version in _ORDINAL_VERSIONS
        # is_external gates the GT-based M_albedo metric path (all three externals are
        # out-of-our-training-space: constancy metrics still apply, absolute-scale ones don't).
        self.is_external = self.is_marigold or self.is_crefnet or self.is_ordinal
        self.pipe = self.model = None
        if self.is_marigold:
            self.pipe = _load_marigold(ckpt_path, device)
        elif self.is_crefnet:
            _ensure_crefnet_imported()
            self.model = _load_crefnet(ckpt_path, device, variant=version)
        elif self.is_ordinal:
            _ensure_ordinal_imported()
            self.model = _load_ordinal(device, variant=version)
        else:
            self.model = load_v17(ckpt_path, device)
            self.pipe = None

    def albedo(self, rgb_tm: np.ndarray) -> np.ndarray:
        """rgb_tm: (H,W,3) tonemapped [0,1] LINEAR. Returns (H,W,3) albedo [0,1]."""
        if self.is_marigold:
            # Encode linear → sRGB for Marigold input
            # BUG FIX (2026-07-13): processing_res was never passed, so Marigold
            # ignored --infer-max-size and processed at its own internal default
            # (768) regardless of our requested cap, unlike CRefNet/Ordinal/v17
            # below, which all honour self.infer_max_size. match_input_res=True
            # (the pipeline default) still returns at native (H,W).
            from marigold_adapter import marigold_albedo_hwc_linear
            srgb = np.power(np.clip(rgb_tm, 0.0, 1.0), 0.45454545454545453)
            pil = PILImage.fromarray((srgb * 255).astype(np.uint8), 'RGB')
            max_size = int(self.infer_max_size) if self.infer_max_size is not None else 1280
            out = self.pipe(pil, processing_res=max_size, match_input_res=True)
            return marigold_albedo_hwc_linear(self.pipe, out)
        elif self.is_crefnet or self.is_ordinal:
            # Both adapters take display-linear [0,1] HWC (== rgb_tm) and return linear albedo.
            # They apply their own sRGB-gamma + long-side cap internally; default 1280 when the
            # save-vis predictor constructs us without an explicit cap.
            max_size = int(self.infer_max_size) if self.infer_max_size is not None else 1280
            runner = _run_crefnet if self.is_crefnet else _run_ordinal
            alb, _ = runner(self.model, np.clip(rgb_tm, 0.0, 1.0).astype(np.float32), max_size, self.device)
            return alb.astype(np.float32)
        else:
            H, W = rgb_tm.shape[:2]
            rgb_in = rgb_tm
            if self.infer_max_size is not None:
                long_side = max(H, W)
                scale = float(self.infer_max_size) / long_side
                if abs(scale - 1.0) > 1e-4:
                    nh, nw = int(round(H * scale)), int(round(W * scale))
                    rgb_in = cv2.resize(rgb_tm, (nw, nh), interpolation=cv2.INTER_LINEAR)
            h, w = rgb_in.shape[:2]
            ph, pw = (14 - h % 14) % 14, (14 - w % 14) % 14
            t = torch.from_numpy(rgb_in).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
            if ph or pw:
                t = torch.nn.functional.pad(t, (0, pw, 0, ph), mode='replicate')
            with torch.no_grad():
                out = self.model(t)
            a = out['a_d'].squeeze(0).permute(1, 2, 0).cpu().numpy()[:h, :w]
            if a.shape[:2] != (H, W):
                a = cv2.resize(a, (W, H), interpolation=cv2.INTER_LINEAR)
            return a


# ──────────────────────────────────────────────────────────────────────────────
# Frame utilities
# ──────────────────────────────────────────────────────────────────────────────

def _raw_frame(scene_path, idx):
    """Load one raw EXR frame WITHOUT white-balancing, returned as RGB float32.
    The model was trained with mid_raw_color_pair=true (raw pairs), so eval must also
    use raw frames.  WB erases the colored-illuminant signal the model was trained to see.
    Returns RGB (matches the training dataset loader which also flips BGR→RGB)."""
    img_p = os.path.join(scene_path, f'dir_{idx}_mip2.exr')
    img = cv2.imread(img_p, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise FileNotFoundError(f'Missing EXR: {img_p}')
    return img[:, :, ::-1].astype(np.float32)  # BGR→RGB


def _wb_frame(scene_path, idx):
    """Load and white-balance one raw EXR frame using its gray probe.
    DEPRECATED for V17/V20 eval — use _raw_frame instead."""
    img_p = os.path.join(scene_path, f'dir_{idx}_mip2.exr')
    prb_p = os.path.join(scene_path, 'probes', f'dir_{idx}_gray256.exr')

    img = cv2.imread(img_p, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32)
    prb = cv2.imread(prb_p, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32)

    if len(prb.shape) == 3:
        prb_msk = np.any(prb > 0.01, axis=-1)
    else:
        prb_msk = prb > 0.01
    prb_msk = cv2.erode(prb_msk.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)

    if prb_msk.any():
        valid_px = prb[prb_msk]
    else:
        valid_px = prb.reshape(-1, prb.shape[-1])
    med = np.array(np.median(valid_px, axis=0))
    wb = med.reshape(1, 1, -1)

    return (img / (wb + 1e-6)).astype(np.float32)


def _tonemap_frame(rgb_linear):
    return _tonemap_linear(rgb_linear.astype(np.float32), percentile=90.0)


def _hdr_valid(rgb, lo_pct, hi_pct):
    """(H,W) float32 {0,1} — exclude specular tail + deep shadow in linear HDR."""
    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    lum = np.clip(lum, 0.0, None)
    pos = lum > 0.005
    lo = float(np.percentile(lum[pos], lo_pct)) if pos.any() else 0.0
    hi = float(np.percentile(lum, hi_pct))
    return ((lum >= lo) & (lum <= hi)).astype(np.float32)


def _run_inference(model, rgb_tm, device):
    """Run V17/V20 on a tonemapped [0,1] HWC RGB.  Returns dict of CHW tensors on CPU."""
    t = torch.from_numpy(rgb_tm).permute(2, 0, 1).unsqueeze(0).float().to(device)
    with torch.no_grad():
        out = model(t)
    return {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in out.items()}


def _lmse(pred, gt, mask):
    """Scale-invariant log-MSE: regress log(pred)*s ≈ log(gt), return residual MSE."""
    eps = 0.0001
    p = np.log(np.clip(pred, eps, None))
    g = np.log(np.clip(gt, eps, None))
    m = mask.astype(bool)
    if m.sum() < 10:
        return float('nan')
    p_m = p[m]
    g_m = g[m]
    s = float(np.dot(p_m, g_m)) / (float(np.dot(p_m, p_m)) + 1e-8)
    return float(np.mean((s * p_m - g_m) ** 2))


# ──────────────────────────────────────────────────────────────────────────────
# Per-scene evaluation
# ──────────────────────────────────────────────────────────────────────────────

def eval_scene(scene_path, predictor, device, skip_list=()):
    """Evaluate one MID scene. Returns metrics dict or None if insufficient data."""
    alb_raw = cv2.imread(os.path.join(scene_path, 'albedo.exr'),
                         cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if alb_raw is None:
        return None
    alb_raw = alb_raw[..., ::-1].copy().astype(np.float32)  # BGR→RGB
    alb_tm = _tonemap_frame(alb_raw)

    seg_path = os.path.join(scene_path, 'materials_mip2.png')
    seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
    if seg is None:
        return None

    valid_indices = sorted([
        int(f.split('_')[1])
        for f in os.listdir(scene_path)
        if f.startswith('dir_') and f.endswith('_mip2.exr')
        and int(f.split('_')[1]) not in (skip_list or [])
    ])

    if len(valid_indices) < 2:
        return None

    H, W = alb_raw.shape[:2]
    albedo_preds = []
    valid_masks = []

    for idx in valid_indices:
        try:
            rgb_hdr = _raw_frame(scene_path, idx)
        except Exception:
            continue
        valid_mask = _hdr_valid(rgb_hdr, lo_pct=20, hi_pct=99.5)
        rgb_tm = _tonemap_frame(rgb_hdr)
        a_pred = predictor.albedo(rgb_tm)
        if a_pred.shape[:2] != (H, W):
            a_pred = cv2.resize(a_pred, (W, H), interpolation=cv2.INTER_LINEAR)
        if valid_mask.shape[:2] != (H, W):
            valid_mask = cv2.resize(valid_mask, (W, H), interpolation=cv2.INTER_LINEAR)
        albedo_preds.append(a_pred)
        valid_masks.append(valid_mask)

    n = len(albedo_preds)
    if n < 2:
        return None

    albedo_stack = np.stack(albedo_preds, axis=0)   # (N, H, W, 3)
    valid_stack = np.stack(valid_masks, axis=0)       # (N, H, W)

    # ── C_mat: within-material CoV of luminance across lightings ──────
    C_mat_vals = []
    rg_vals = []
    bg_vals = []
    mat_ids = np.unique(seg)

    # Corrected cast accumulators. R_cast_rms (below) pools rg/bg over ALL materials
    # and takes one std, so it sums the across-illuminant drift we mean to measure with
    # the between-material chroma spread of the scene. That second term shrinks when a
    # model collapses distinct materials toward a common hue, so the pooled metric pays
    # models for destroying colour (v17_42 scores best on it while being visibly
    # colour-collapsed). Decompose instead, and anchor fidelity on the GT albedo.
    within_rms = []          # per-material chroma drift ACROSS ILLUMINANTS  <- the claim
    mat_rg, mat_bg = [], []  # per-material illuminant-averaged chroma       -> between
    gt_rg, gt_bg = [], []    # per-material GT albedo chroma                 -> anchor
    chroma_err = []          # per-material |pred chroma - GT chroma|
    sat_pred_vals, sat_gt_vals = [], []
    vm_any = (valid_stack.mean(axis=0) > 0.5)

    def _sat(px):
        mx = px.max(axis=-1)
        return float(np.mean((mx - px.min(axis=-1)) / (mx + 1e-6)))

    for m in mat_ids:
        if m == 0:
            continue
        m_px = (seg == m)
        if m_px.sum() < 25:
            continue

        per_frame_means = []
        per_frame_rg = []
        per_frame_bg = []
        for k in range(n):
            vm = (valid_stack[k] > 0.5) & m_px
            if vm.sum() < 25:
                continue
            a_k = albedo_stack[k][vm]   # (P, 3)
            lum_k = 0.299 * a_k[:, 0] + 0.587 * a_k[:, 1] + 0.114 * a_k[:, 2]
            g = a_k[:, 1].mean() + 1e-8
            per_frame_means.append(float(lum_k.mean()))
            per_frame_rg.append(float(a_k[:, 0].mean() / g))
            per_frame_bg.append(float(a_k[:, 2].mean() / g))
            sat_pred_vals.append(_sat(a_k))

        if len(per_frame_means) < 2:
            continue
        means = np.array(per_frame_means)
        denom = float(np.mean(means)) + 1e-8
        cov = float(np.std(means)) / denom
        C_mat_vals.append(cov)
        rg_vals.extend(per_frame_rg)
        bg_vals.extend(per_frame_bg)

        within_rms.append(float(np.sqrt(np.var(per_frame_rg) + np.var(per_frame_bg))))
        mr, mb = float(np.mean(per_frame_rg)), float(np.mean(per_frame_bg))
        mat_rg.append(mr)
        mat_bg.append(mb)

        gvm = vm_any & m_px
        if gvm.sum() >= 25:
            g_px = alb_tm[gvm]
            gg = g_px[:, 1].mean() + 1e-8
            grg = float(g_px[:, 0].mean() / gg)
            gbg = float(g_px[:, 2].mean() / gg)
            gt_rg.append(grg)
            gt_bg.append(gbg)
            chroma_err.append(float(np.hypot(mr - grg, mb - gbg)))
            sat_gt_vals.append(_sat(g_px))

    C_mat = float(np.mean(C_mat_vals)) if C_mat_vals else float('nan')
    R_cast_rg = float(np.std(rg_vals)) if len(rg_vals) >= 2 else float('nan')
    R_cast_bg = float(np.std(bg_vals)) if len(bg_vals) >= 2 else float('nan')
    R_cast_rms = float(np.sqrt(np.var(rg_vals) + np.var(bg_vals))) if len(rg_vals) >= 2 else float('nan')

    # ── Corrected constancy + the fidelity guard it must be read with ──────────
    # Cast_within alone is still trivially won by a constant (grey) predictor, so it is
    # NOT reported on its own: Chroma_fid / Sat_ratio (both 1.0 = faithful to GT) expose
    # that degenerate direction, and Cast_rel divides the drift by the model's own chroma
    # magnitude so a more colourful model is not charged for having more room to drift.
    _nan = float('nan')
    Cast_within = float(np.mean(within_rms)) if within_rms else _nan
    Cast_between = (float(np.sqrt(np.var(mat_rg) + np.var(mat_bg)))
                    if len(mat_rg) >= 2 else _nan)
    GT_between = (float(np.sqrt(np.var(gt_rg) + np.var(gt_bg)))
                  if len(gt_rg) >= 2 else _nan)
    Cast_rel = (Cast_within / Cast_between
                if Cast_between and Cast_between > 1e-6 else _nan)
    Chroma_err = float(np.mean(chroma_err)) if chroma_err else _nan
    Chroma_fid = (Cast_between / GT_between
                  if GT_between and GT_between > 1e-6 else _nan)
    Sat_pred = float(np.mean(sat_pred_vals)) if sat_pred_vals else _nan
    Sat_gt = float(np.mean(sat_gt_vals)) if sat_gt_vals else _nan
    Sat_ratio = (Sat_pred / Sat_gt if Sat_gt and Sat_gt > 1e-6 else _nan)

    # ── M_albedo: scale-invariant log-MSE vs GT ──────────────────────
    lmse_vals = []
    vm_all = (valid_stack.mean(axis=0) > 0.5)
    mean_pred = albedo_stack.mean(axis=0)   # (H, W, 3)
    for c in range(3):
        lmse_vals.append(_lmse(mean_pred[..., c], alb_tm[..., c], vm_all))
    M_albedo = float(np.nanmean(lmse_vals))

    valid_frac = float(vm_all.mean())

    # ── sat_bin_mae: albedo MAE per saturation quartile ───────────────
    gt_alb = alb_tm
    max_rgb = gt_alb.max(axis=-1)
    gt_sat = (max_rgb - gt_alb.min(axis=-1)) / (max_rgb + 1e-6)

    sat_flat = gt_sat[vm_all].flatten()
    if sat_flat.size > 50:
        edges = [np.percentile(sat_flat, q) for q in [0, 25, 50, 75, 100]]
        bin_maes = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = vm_all & (gt_sat >= lo) & (gt_sat <= hi)
            if sel.sum() > 0:
                bin_maes.append(float(np.abs(mean_pred[sel] - gt_alb[sel]).mean()))
            else:
                bin_maes.append(float('nan'))
        sat_bin_mae = bin_maes
    else:
        sat_bin_mae = [float('nan')] * 4

    return {
        'C_mat': C_mat,
        'R_cast_rg': R_cast_rg,
        'R_cast_bg': R_cast_bg,
        'R_cast_rms': R_cast_rms,
        'M_albedo': M_albedo,
        'valid_frac': valid_frac,
        'sat_bin_mae': sat_bin_mae,
        # corrected cast (see decomposition note above); R_cast_rms kept so the
        # previously published pooled numbers stay reproducible from the same run.
        'Cast_within': Cast_within,
        'Cast_between': Cast_between,
        'Cast_rel': Cast_rel,
        'GT_between': GT_between,
        'Chroma_err': Chroma_err,
        'Chroma_fid': Chroma_fid,
        'Sat_pred': Sat_pred,
        'Sat_gt': Sat_gt,
        'Sat_ratio': Sat_ratio,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Visualization helpers
# ──────────────────────────────────────────────────────────────────────────────

def _gamma_u8(img):
    g = np.power(np.clip(np.nan_to_num(img, 0.0), 0.0, 1.0), 0.45454545454545453)
    return cv2.cvtColor((g * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def _albedo_u8(img, scale):
    a = np.power(np.clip(np.nan_to_num(img * scale, 0.0), 0.0, 1.0), 0.45454545454545453)
    return cv2.cvtColor((a * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def _label(img, text):
    cv2.putText(img, text, (3, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3)
    cv2.putText(img, text, (3, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)


def make_mid_constancy_row(scene_path, predictor, device, light_idxs=None, tile_w=200,
                           display_mode='shared'):
    """Render visualization row. Returns (row_img or None, cov, cast_rms)."""
    alb_raw = cv2.imread(os.path.join(scene_path, 'albedo.exr'),
                         cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if alb_raw is None:
        return None, float('nan'), float('nan')
    alb_raw = alb_raw[..., ::-1].copy().astype(np.float32)
    alb_tm = _tonemap_frame(alb_raw)

    H, W = alb_raw.shape[:2]
    all_idxs = sorted([
        int(f.split('_')[1])
        for f in os.listdir(scene_path)
        if f.startswith('dir_') and f.endswith('_mip2.exr')
    ])
    if light_idxs is None:
        step = max(1, len(all_idxs) // 4)
        light_idxs = all_idxs[::step][:4]

    rgb_hdrs, albedos, used_idxs = [], [], []
    for idx in light_idxs:
        try:
            rgb_hdr = _raw_frame(scene_path, idx)
        except Exception:
            continue
        rgb_hdrs.append(rgb_hdr)
        rgb_tm = _tonemap_frame(rgb_hdr)           # model input: p90→0.8 (training-matched)
        a = predictor.albedo(rgb_tm)
        if a.shape[:2] != (H, W):
            a = cv2.resize(a, (W, H), interpolation=cv2.INTER_LINEAR)
        albedos.append(a)
        used_idxs.append(idx)

    if len(albedos) < 2:
        return None, float('nan'), float('nan')

    if display_mode == 'current':
        inputs = [np.clip(_tonemap_frame(rgb_hdr) * 1.5, 0.0, 1.0) for rgb_hdr in rgb_hdrs]
    else:
        # Shared exposure makes the raw cast easier to compare across L0/L6/L12/L18.
        stacked = np.concatenate(rgb_hdrs, axis=0)
        shared_scale = _compute_tonemap_scale(stacked, percentile=95.0, target_brightness=0.72)
        inputs = [_tonemap_linear(rgb_hdr, percentile=95.0, scale=shared_scale) for rgb_hdr in rgb_hdrs]

    stack = np.stack(albedos, axis=0)   # (N, H, W, 3)
    valid_gt = (alb_tm.sum(axis=-1) > 1e-6)

    # GT albedo scale (for the GT tile display only)
    a_scale_val = float(np.percentile(alb_tm[valid_gt], 99.9)) if valid_gt.any() else 1.0
    a_scale = 1.0 / (a_scale_val + 1e-6)

    # Predicted albedo scale: normalize by the stack's own p99 so predictions always look bright
    # regardless of absolute magnitude (model may predict different scale than GT EXR).
    pred_vals = stack[stack > 0.001]
    pred_scale = 1.0 / (float(np.percentile(pred_vals, 99.0)) + 1e-6) if pred_vals.size > 0 else 1.0

    # Luminance CoV
    alb_lum = 0.2126 * stack[..., 0] + 0.7152 * stack[..., 1] + 0.0722 * stack[..., 2]
    mu = alb_lum.mean(axis=0) + 1e-6
    sd = alb_lum.std(axis=0)
    cov = float(np.mean((sd / mu)[valid_gt])) if valid_gt.any() else float('nan')

    # CoV heatmap
    hi = float(np.abs(np.percentile((sd / mu)[valid_gt], 95.0))) if valid_gt.any() else 0.004
    C_vis = np.clip((sd / mu) / (hi + 1e-6), 0.0, 1.0)
    heat = cv2.applyColorMap((C_vis * 255).clip(0, 255).astype(np.uint8), cv2.COLORMAP_JET)

    def _fit(im):
        h = int(im.shape[0])
        return cv2.resize(im, (tile_w, int(round(tile_w * h / max(im.shape[1], 1)))))

    def _align(pred):
        p = pred.reshape(-1)
        g = alb_tm.reshape(-1)
        c = float(np.sum(p * g)) / (float(np.sum(p * p)) + 1e-6)
        return pred * c

    Hh = int(round(tile_w * H / max(W, 1)))
    tiles = []

    # Layout: [in L0 | in L1 | in L2 | in L3] | GT Albedo | [A L0 | A L1 | A L2 | A L3] | CoV
    for k, idx in enumerate(used_idxs):
        t = _gamma_u8(inputs[k])
        t = cv2.resize(t, (tile_w, Hh))
        _label(t, f'in L{idx}')
        tiles.append(t)

    gt_tile = _albedo_u8(alb_tm, a_scale)
    gt_tile = cv2.resize(gt_tile, (tile_w, Hh))
    _label(gt_tile, 'GT Albedo')
    tiles.append(gt_tile)

    for k, idx in enumerate(used_idxs):
        a_tile = _albedo_u8(albedos[k], pred_scale)
        a_tile = cv2.resize(a_tile, (tile_w, Hh))
        _label(a_tile, f'A L{idx}')
        tiles.append(a_tile)

    heat_tile = cv2.resize(heat, (tile_w, Hh))
    _label(heat_tile, f'CoV {cov:.3f}')
    tiles.append(heat_tile)

    row = np.concatenate(tiles, axis=1)

    rg = [albedos[k][..., 0].mean() / (albedos[k][..., 1].mean() + 1e-6) for k in range(len(albedos))]
    bg = [albedos[k][..., 2].mean() / (albedos[k][..., 1].mean() + 1e-6) for k in range(len(albedos))]
    cast_rms = float(np.sqrt(np.var(rg) + np.var(bg)))

    strip = np.zeros((22, row.shape[1], 3), dtype=np.uint8)
    mode_note = 'shared exposure' if display_mode != 'current' else 'current per-frame display'
    txt = (f'CoV(C_mat proxy): {cov:.4f}   Cast_RMS: {cast_rms:.4f}'
           f'   n_lights: {len(albedos)}   scene: {Path(scene_path).name}   {mode_note}')
    cv2.putText(strip, txt, (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
    return np.vstack([row, strip]), cov, cast_rms


def save_mid_constancy_sheet(scenes, predictor, device, out_path, max_scenes=40, display_mode='shared'):
    """Render the multi-illuminant constancy sheet for the first `max_scenes` scenes."""
    modes = ('current', 'shared') if display_mode == 'both' else (display_mode,)

    def _mode_out_path(path, mode):
        path = Path(path)
        if mode == 'shared':
            return str(path)
        return str(path.with_name(f'{path.stem}_{mode}{path.suffix}'))

    for mode in modes:
        rows = []
        for sp in scenes[:max_scenes]:
            try:
                r, _, _ = make_mid_constancy_row(sp, predictor, device, display_mode=mode)
            except Exception as e:
                print(f'  [vis-skip] {Path(sp).name}: {e}')
                r = None
            if r is not None:
                rows.append(r)

        if not rows:
            print(f'  [vis] no rows rendered for mode={mode}')
            continue

        wmax = max(r.shape[1] for r in rows)
        padded = [cv2.copyMakeBorder(r, 0, 0, 0, wmax - r.shape[1], cv2.BORDER_CONSTANT) for r in rows]
        sheet = np.vstack(padded)
        out_file = _mode_out_path(out_path, mode)
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        cv2.imwrite(out_file, sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f'  MID constancy sheet ({len(rows)} scenes, mode={mode}) → {out_file}')


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────────────────────────────────────

def run_eval(ckpt_path, label, scenes, device, version, infer_max_size=None):
    print('\n' + '=' * 60)
    print(f'  Model: {label}')
    print(f'  (version={version})')
    print(f'  Path: {ckpt_path}')
    if infer_max_size is not None:
        print(f'  (inference long-side = {infer_max_size}px)')

    predictor = AlbedoPredictor(ckpt_path, version, device, infer_max_size=infer_max_size)

    if predictor.is_external:
        print('  [external/zero-shot] → constancy + cast ONLY; LMSE/sat-bins BLANKED '
              '(MID albedo is pseudo-GT, in-domain for CARI — §6.0).')

    all_cmat, all_rg, all_bg, all_rms, all_lmse, all_sat_bins, all_valid = [], [], [], [], [], [], []
    _EXTRA = ('Cast_within', 'Cast_between', 'Cast_rel', 'GT_between',
              'Chroma_err', 'Chroma_fid', 'Sat_pred', 'Sat_gt', 'Sat_ratio')
    extra = {k: [] for k in _EXTRA}

    for i, scene_path in enumerate(scenes):
        sname = Path(scene_path).name
        try:
            res = eval_scene(scene_path, predictor, device, skip_list=())
        except Exception as e:
            print(f'  [FAIL] {sname}: {e}')
            continue
        if res is None:
            print(f'  [SKIP] {sname}: insufficient frames')
            continue

        all_cmat.append(res['C_mat'])
        all_rg.append(res['R_cast_rg'])
        all_bg.append(res['R_cast_bg'])
        all_rms.append(res['R_cast_rms'])
        all_lmse.append(res['M_albedo'])
        all_sat_bins.append(res['sat_bin_mae'])
        all_valid.append(res['valid_frac'])
        for k in _EXTRA:
            extra[k].append(res[k])

        b = res['R_cast_rms']
        v = res['C_mat']
        print(f'  {sname:30s}  C_mat={v:.4f}  Cast_RMS={b:.4f}  LMSE={res["M_albedo"]:.4f}  '
              f'within={res["Cast_within"]:.4f}  rel={res["Cast_rel"]:.4f}  '
              f'fid={res["Chroma_fid"]:.4f}')

    sat_bins = (list(np.nanmean(np.array(all_sat_bins), axis=0))
                if all_sat_bins else [float('nan')] * 4)

    agg = {
        'label': label,
        'path': str(ckpt_path),
        'version': str(version),
        'is_external': predictor.is_external,
        'C_mat': float(np.nanmean(all_cmat)) if all_cmat else float('nan'),
        'R_cast_rg': float(np.nanmean(all_rg)) if all_rg else float('nan'),
        'R_cast_bg': float(np.nanmean(all_bg)) if all_bg else float('nan'),
        'R_cast_rms': float(np.nanmean(all_rms)) if all_rms else float('nan'),
        'M_albedo': float(np.nanmean(all_lmse)) if all_lmse else float('nan'),
        'valid_frac': float(np.nanmean(all_valid)) if all_valid else float('nan'),
        'sat_bin_mae': sat_bins,
    }
    for k in _EXTRA:
        agg[k] = float(np.nanmean(extra[k])) if extra[k] else float('nan')

    # ── PER-SCENE RETENTION (added 2026-07-14) ────────────────────────────────────────────
    # Previously only np.nanmean survived, which (a) hid that the reported Chroma_fid of
    # 1.008 is a MEAN OF RATIOS — the ratio of aggregates is 0.941, and by Jensen the mean is
    # inflated by outlier scenes — and (b) made paired bootstrap CIs impossible, so small
    # deltas were being ranked as wins without any test. Keep the per-scene vectors so both
    # can be computed downstream. `scenes` is aligned index-for-index with every list here.
    agg['per_scene'] = {
        'scene': [Path(s).name for s in scenes][:len(all_cmat)],
        'C_mat': [float(x) for x in all_cmat],
        'R_cast_rms': [float(x) for x in all_rms],
        **{k: [float(x) for x in extra[k]] for k in _EXTRA},
    }
    # Ratio-of-aggregates spread: the correct way to state "how much chroma spread is
    # retained", as opposed to averaging per-scene ratios.
    _cb = np.nanmean(extra['Cast_between']) if extra['Cast_between'] else float('nan')
    _gb = np.nanmean(extra['GT_between']) if extra['GT_between'] else float('nan')
    agg['Chroma_fid_ratio_of_aggregates'] = float(_cb / _gb) if _gb else float('nan')
    return agg


# ──────────────────────────────────────────────────────────────────────────────
# Table printing
# ──────────────────────────────────────────────────────────────────────────────

def print_table(results):
    def _c(v, w, p=4):
        if not isinstance(v, float) or np.isnan(v):
            return f'{"—":>{w}}'
        return f'{v:>{w}.{p}f}'

    print('\n' + '=' * 90)
    print('CROSS-ILLUMINATION CONSTANCY RESULTS  (lower C_mat/LMSE/cast_rms = better)')
    print('=' * 90)
    print(f'{"Model":>30s}  {"C_mat":>7s}  {"R/G":>6s}  {"B/G":>6s}  '
          f'{"Cast_RMS":>9s}  {"LMSE":>7s}  '
          f'{"SAT q0-25":>8s} {"q25-50":>8s} {"q50-75":>8s} {"q75-100":>8s}')
    print('-' * 90)

    for r in results:
        tag = ' *' if r.get('is_external') else ''
        sat = r.get('sat_bin_mae') or [float('nan')] * 4
        print(f'{r.get("label", "?"):>28s}{tag}  '
              f'{_c(r.get("C_mat"), 7)}  '
              f'{_c(r.get("R_cast_rg"), 6, 3)}  '
              f'{_c(r.get("R_cast_bg"), 6, 3)}  '
              f'{_c(r.get("R_cast_rms"), 9)}  '
              f'{_c(r.get("M_albedo"), 7)}  '
              f'{_c(sat[0] if len(sat) > 0 else float("nan"), 8, 5)}  '
              f'{_c(sat[1] if len(sat) > 1 else float("nan"), 8, 5)}  '
              f'{_c(sat[2] if len(sat) > 2 else float("nan"), 8, 5)}  '
              f'{_c(sat[3] if len(sat) > 3 else float("nan"), 8, 5)}')

    if any(r.get('is_external') for r in results):
        print('\n  * external/zero-shot — LMSE & SAT bins omitted (MID albedo is pseudo-GT, '
              'in-domain for CARI). Compare these rows on C_mat & Cast_RMS only (§6.0).')

    # ── Corrected cast: the pooled Cast_RMS above conflates across-illuminant drift with
    # between-material chroma spread, and so rewards models that collapse material colour.
    # Read Cast_rel (drift ÷ the model's own chroma magnitude) TOGETHER WITH Chroma_fid /
    # Sat_ratio: invariance alone is trivially won by a flat grey predictor.
    print('\n' + '=' * 100)
    print('CORRECTED CAST DECOMPOSITION   (Chroma_fid & Sat_ratio: 1.000 = faithful to GT albedo)')
    print('=' * 100)
    print(f'{"Model":>30s}  {"within":>7s}  {"between":>8s}  {"Cast_rel":>9s}  '
          f'{"GT_betw":>8s}  {"Chroma_fid":>10s}  {"Chroma_err":>10s}  {"Sat_ratio":>9s}')
    print('-' * 100)
    for r in results:
        print(f'{r.get("label", "?"):>30s}  '
              f'{_c(r.get("Cast_within"), 7)}  '
              f'{_c(r.get("Cast_between"), 8)}  '
              f'{_c(r.get("Cast_rel"), 9)}  '
              f'{_c(r.get("GT_between"), 8)}  '
              f'{_c(r.get("Chroma_fid"), 10)}  '
              f'{_c(r.get("Chroma_err"), 10)}  '
              f'{_c(r.get("Sat_ratio"), 9)}')
    print('-' * 100)
    print('  within   = per-material chroma drift ACROSS ILLUMINANTS (absolute; scales with chroma)')
    print('  Cast_rel = within / between — scale-normalised invariance   [LOWER = better]')
    print('  Chroma_fid = between / GT_between — material chroma retained [1.0 = faithful, <1 = collapsed]')

    if len(results) >= 2:
        ours = results[0]
        base = results[-1]
        print(f'\n  Δ ({ours.get("label", "ours")} − {base.get("label", "base")}'
              ', negative = improved):')
        for key in ['C_mat', 'R_cast_rg', 'R_cast_bg', 'R_cast_rms', 'M_albedo']:
            o = ours.get(key, float('nan'))
            b = base.get(key, float('nan'))
            if not (isinstance(o, float) and isinstance(b, float)):
                continue
            if np.isnan(o) or np.isnan(b):
                continue
            delta = o - b
            pct = delta / (abs(b) + 1e-9) * 100.0
            sign = '✓ improved' if delta < 0 else '✗ worsened'
            print(f'    {key:>12s}: {delta:+.4f}  ({pct:+.1f}%, {sign})')

        bin_labels = ['SAT q0-25', 'q25-50', 'q50-75', 'q75-100']
        o_sat = ours.get('sat_bin_mae', [])
        b_sat = base.get('sat_bin_mae', [])
        if o_sat and b_sat:
            print('    SAT-binned MAE Δ:')
            for lab, db, dr in zip(bin_labels, b_sat, o_sat):
                if isinstance(db, float) and isinstance(dr, float):
                    if not (np.isnan(db) or np.isnan(dr)):
                        d = dr - db
                        sign = '✓' if d < 0 else '✗'
                        print(f'      {lab:>18s}: {d:+.5f} {sign}')


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpts', nargs='+',
                        help='N-way: space-separated "label=path" pairs (first = baseline).')
    parser.add_argument('--ckpt19k', default='checkpoints/checkpoint_v17_iter_19000.pth')
    parser.add_argument('--ckpt30k', default='checkpoints/v17_row2/checkpoint_iter_30000.pth')
    parser.add_argument('--mid-root', default='/home/khang/datasets/MIDIntrinsics')
    parser.add_argument('--split', default='test')
    parser.add_argument('--max-scenes', type=int, default=None)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save-json',
                        default='documents/evals/results/eval_mid_constancy_results.json',
                        help='Path to save results JSON (relative to repo root)')
    parser.add_argument('--save-vis',
                        help='Directory to save the multi-illuminant constancy sheet.')
    parser.add_argument('--vis-scenes', type=int, default=8)
    parser.add_argument('--vis-mode', choices=['current', 'shared', 'both'], default='shared',
                        help='Visual style for MID constancy sheets: per-frame boost, shared exposure, or both.')
    parser.add_argument('--infer-max-size', type=int, default=None,
                        help='Long-side cap for our-model inference. Default None = '
                             'native mip2 resolution (1500px). Set e.g. 1024/1280/1792 '
                             'to sweep. No effect on Marigold.')
    parser.add_argument('--include-marigold', action='store_true',
                        help='Also append local Marigold Appearance/Lighting baselines. '
                             'By default this happens only for the built-in comparison, '
                             'not when --ckpts is supplied.')
    args = parser.parse_args()

    mid_split = os.path.join(args.mid_root, args.split)
    if not os.path.isdir(mid_split):
        raise FileNotFoundError(f'MID {args.split} dir not found: {mid_split}')

    scenes = sorted([
        os.path.join(mid_split, s)
        for s in os.listdir(mid_split)
        if os.path.isdir(os.path.join(mid_split, s))
        and os.path.isfile(os.path.join(mid_split, s, 'albedo.exr'))
    ])
    scenes = scenes[:args.max_scenes] if args.max_scenes else scenes
    print(f'Evaluating {len(scenes)} scenes from {mid_split}')

    def _abs(p):
        return p if os.path.isabs(str(p)) else str(ROOT / p)

    def _infer_version(label, path):
        """Pick model family from an explicit suffix, else from the label/path. Marigold
        checkpoints are directories named marigold-iid-{appearance,lighting}-*."""
        low = label.lower()
        if ' ' in low:
            low = low.split(' ')[0]
        if 'appearance' in low:
            return 'marigold-appearance'
        if 'lighting' in low:
            return 'marigold-lighting'
        if 'crefnet' in low or 'crefnet' in str(path).lower():
            return 'crefnet'
        if 'ordinal' in low:
            return 'ordinal'
        return '17'

    ckpt_specs = []
    if args.ckpts:
        for spec in args.ckpts:
            # LABEL=PATH[=TYPE]; TYPE (marigold-*/crefnet[-e]/ordinal[-rendered-only]/17) is
            # authoritative when given, else the family is inferred from the label/path.
            parts = spec.split('=')
            if len(parts) >= 3:
                label, path, version = parts[0], parts[1], parts[2]
            elif len(parts) == 2:
                label, path = parts
                version = _infer_version(label, path)
            else:
                path = parts[0]
                label = Path(path).name
                version = _infer_version(label, path)
            ckpt_specs.append((label, _abs(path), version))
    else:
        for label, path, version in [
            ('V17 19k (pre-CARI baseline)', args.ckpt19k, '17'),
            ('V17 30k (+CARI P3 11k iters)', args.ckpt30k, '17'),
        ]:
            rpath = _abs(path)
            if not os.path.exists(rpath):
                print(f'[SKIP] checkpoint not found: {rpath}')
                continue
            ckpt_specs.append((label, rpath, version))

    if (not args.ckpts) or args.include_marigold:
        for mv in _MARIGOLD_VERSIONS:
            short = mv.split('-')[1]
            cp = str(_MARIGOLD_PATH / f'marigold-iid-{short}-v1-0')
            if os.path.isdir(cp):
                ckpt_specs.append((f'Marigold-{short}', cp, mv))

    results = []
    for label, ckpt_path, version in ckpt_specs:
        is_dir_model = os.path.isdir(str(ckpt_path))
        # Ordinal Shading fetches its own weights via torch.hub; its path arg is a placeholder.
        needs_path = version not in _ORDINAL_VERSIONS
        if needs_path and not is_dir_model and not os.path.exists(str(ckpt_path)):
            print(f'[SKIP] checkpoint not found: {ckpt_path}')
            continue
        res = run_eval(ckpt_path, label, scenes, args.device, version,
                       infer_max_size=args.infer_max_size)
        results.append(res)

        if args.save_vis:
            predictor = AlbedoPredictor(ckpt_path, version, args.device)
            safe = ''.join(c if c.isalnum() or c in '-_' else '_' for c in label)
            out = os.path.join(args.save_vis, f'mid_constancy_{safe}.jpg')
            save_mid_constancy_sheet(scenes, predictor, args.device, out, args.vis_scenes, display_mode=args.vis_mode)
        torch.cuda.empty_cache()

    if not results:
        print('No results — check checkpoint paths.')
        return

    print_table(results)

    if args.save_json:
        def _clean(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, (list, tuple)):
                return [_clean(x) for x in obj]
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            return obj

        save_path = _abs(args.save_json)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'timestamp': datetime.now().isoformat(),
            'split': args.split,
            'results': _clean(results),
        }
        with open(save_path, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f'\nResults saved → {save_path}')


if __name__ == '__main__':
    main()
