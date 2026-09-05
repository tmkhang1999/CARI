#!/usr/bin/env python3
"""Sweep SSIM computation variants on ARAP to find which one reproduces the
published Ordinal Shading figure (SSIM 0.761, Table 1 of Careaga & Aksoy 2023).

Motivation: the paper calls SSIM one of its "scale-invariant metrics", but plain
SSIM is NOT scale-invariant -- so a scale alignment step must exist that the paper
and supplement never spell out. The supplement fixes only the mask threshold
(albedo < 0.004 marked invalid, A.6 / B.1); everything else about the SSIM call is
unspecified. This sweeps the plausible combinations against the same 136 images the
main table uses, so the protocol gap is measured rather than guessed.

Usage: CUDA_VISIBLE_DEVICES=0 python tests/eval/sweep_arap_ssim.py --model ordinal
"""
import argparse
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity as sk_ssim
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests' / 'eval'))
os.chdir(str(ROOT / 'tests' / 'eval'))

from eval_arap import (  # noqa: E402
    get_arap_cases, load_gt, load_image, _white_balance_gt,
    _external_display_linear, _find_file, INPUT_EXTS,
)
from ordinal_adapter import load_ordinal, run_ordinal  # noqa: E402
from crefnet_adapter import load_crefnet, run_crefnet  # noqa: E402

ARAP = str(ROOT / 'tests/testing_data/ARAP_dataset')


def align_lstsq(pred, gt, mask):
    """Single global least-squares scalar, the convention their si-RMSE uses."""
    p = pred[mask]
    g = gt[mask]
    denom = float(np.sum(p * p))
    a = float(np.sum(p * g) / denom) if denom > 1e-12 else 0.0
    return pred * a


def align_lstsq_perchannel(pred, gt, mask):
    out = pred.copy()
    for c in range(pred.shape[-1]):
        p = pred[..., c][mask]
        g = gt[..., c][mask]
        denom = float(np.sum(p * p))
        a = float(np.sum(p * g) / denom) if denom > 1e-12 else 0.0
        out[..., c] = pred[..., c] * a
    return out


def ssim_variants(pred, gt, mask):
    """Return {variant_name: value} for one image."""
    out = {}
    eps = 1e-8

    def _ssim_multi(a, b, dr):
        # skimage renamed multichannel -> channel_axis
        return float(sk_ssim(a, b, data_range=dr, channel_axis=2))

    def _ssim_perchan(a, b, dr):
        return float(np.mean([sk_ssim(a[..., c], b[..., c], data_range=dr)
                              for c in range(a.shape[-1])]))

    # --- A: no alignment, GT as-is, clamp both to [0,1] ---
    p0 = np.clip(pred, 0, 1)
    g0 = np.clip(gt, 0, 1)
    out['A_raw_perchan'] = _ssim_perchan(g0, p0, 1.0)
    out['A_raw_multi'] = _ssim_multi(g0, p0, 1.0)

    # --- B: GT normalised by its own p99 (what we currently propose) ---
    v = gt[gt > 1e-8]
    s = float(np.percentile(v, 99.0)) if v.size else 1.0
    g1 = np.clip(gt / (s + eps), 0, 1)
    out['B_gtp99_perchan'] = _ssim_perchan(g1, p0, 1.0)
    out['B_gtp99_multi'] = _ssim_multi(g1, p0, 1.0)

    # --- C: least-squares align pred->gt (global), data_range from GT ---
    pa = align_lstsq(pred, gt, mask)
    dr = float(gt.max() - gt.min()) or 1.0
    out['C_lstsq_globalDR_perchan'] = _ssim_perchan(gt, pa, dr)
    out['C_lstsq_globalDR_multi'] = _ssim_multi(gt, pa, dr)

    # --- D: least-squares align per channel, data_range from GT ---
    pb = align_lstsq_perchannel(pred, gt, mask)
    out['D_lstsqPC_globalDR_perchan'] = _ssim_perchan(gt, pb, dr)
    out['D_lstsqPC_globalDR_multi'] = _ssim_multi(gt, pb, dr)

    # --- E: align, then BOTH renormalised to [0,1] by GT max ---
    gmax = float(gt.max()) or 1.0
    out['E_lstsq_bothNorm_perchan'] = _ssim_perchan(
        np.clip(gt / gmax, 0, 1), np.clip(pa / gmax, 0, 1), 1.0)
    out['E_lstsq_bothNorm_multi'] = _ssim_multi(
        np.clip(gt / gmax, 0, 1), np.clip(pa / gmax, 0, 1), 1.0)

    # --- F: grayscale, align, normalise ---
    def _gray(x):
        return 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
    gg, pg = _gray(gt), _gray(pa)
    gm = float(gg.max()) or 1.0
    out['F_gray_lstsq_norm'] = float(sk_ssim(np.clip(gg / gm, 0, 1),
                                             np.clip(pg / gm, 0, 1), data_range=1.0))

    # --- G: like B but pred also renormalised to its own p99 ---
    vp = pred[pred > 1e-8]
    sp = float(np.percentile(vp, 99.0)) if vp.size else 1.0
    p2 = np.clip(pred / (sp + eps), 0, 1)
    out['G_bothP99_perchan'] = _ssim_perchan(g1, p2, 1.0)
    out['G_bothP99_multi'] = _ssim_multi(g1, p2, 1.0)

    # --- H: Wang et al. reference config. skimage's defaults (7x7 uniform window,
    # sample covariance) do NOT match the original SSIM paper; reproducing it needs
    # gaussian_weights=True, sigma=1.5, use_sample_covariance=False. Many codebases
    # use this, so it is a prime suspect for the remaining gap.
    wang = dict(gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
    gn = np.clip(gt / gmax, 0, 1)
    pn = np.clip(pa / gmax, 0, 1)
    out['H_wang_lstsq_norm_multi'] = float(
        sk_ssim(gn, pn, data_range=1.0, channel_axis=2, **wang))
    out['H_wang_lstsq_norm_gray'] = float(
        sk_ssim(np.clip(gg / gm, 0, 1), np.clip(pg / gm, 0, 1), data_range=1.0, **wang))

    # --- J: sRGB-encode both before scoring. The ARAP GT is linear HDR; if their
    # harness gamma-encodes for display before measuring, dark regions get expanded
    # and SSIM rises substantially. Plausible unstated step.
    def _srgb(x):
        return np.clip(x, 0, 1) ** (1 / 2.2)
    out['J_srgb_lstsq_multi'] = _ssim_multi(_srgb(gn), _srgb(pn), 1.0)
    out['J_srgb_wang_multi'] = float(
        sk_ssim(_srgb(gn), _srgb(pn), data_range=1.0, channel_axis=2, **wang))

    # --- K: score at a reduced fixed resolution. Downscaling smooths high-frequency
    # disagreement, which systematically raises SSIM. ---
    for side in (512, 384):
        h, w = gn.shape[:2]
        sc = side / max(h, w)
        if sc < 1.0:
            nh, nw = max(8, int(h * sc)), max(8, int(w * sc))
            g_r = cv2.resize(gn, (nw, nh), interpolation=cv2.INTER_AREA)
            p_r = cv2.resize(pn, (nw, nh), interpolation=cv2.INTER_AREA)
            out[f'K_resize{side}_multi'] = _ssim_multi(g_r, p_r, 1.0)
            out[f'K_resize{side}_srgb_multi'] = _ssim_multi(_srgb(g_r), _srgb(p_r), 1.0)

    # --- I: crop to the mask's bounding box before scoring (drops the dead border
    # entirely rather than letting it contribute a trivial match) ---
    ys, xs = np.where(mask)
    if ys.size:
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        gc, pc = gn[y0:y1, x0:x1], pn[y0:y1, x0:x1]
        if min(gc.shape[:2]) >= 7:
            out['I_crop_lstsq_norm_multi'] = float(
                sk_ssim(gc, pc, data_range=1.0, channel_axis=2))
            out['I_crop_wang_multi'] = float(
                sk_ssim(gc, pc, data_range=1.0, channel_axis=2, **wang))

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='ordinal', choices=['ordinal', 'crefnet'])
    ap.add_argument('--max_size', type=int, default=1280)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.model == 'ordinal':
        model = load_ordinal(device, variant='ordinal')
        runner = run_ordinal
    else:
        model = load_crefnet(str(ROOT / 'checkpoints/CRefNet/final_real.pt'),
                             device, variant='crefnet')
        runner = run_crefnet

    cases = get_arap_cases(ARAP)
    if args.limit:
        cases = cases[:args.limit]

    acc = {}
    n_used = 0
    for base_name, input_stem, ext in tqdm(cases, desc='sweep'):
        img_path = _find_file(ARAP, input_stem, [f'.{ext}'] + list(INPUT_EXTS))
        if img_path is None:
            continue
        albedo_gt, _ = load_gt(ARAP, base_name, input_stem, ext)
        if albedo_gt is None:
            continue
        rgb_linear, is_hdr = load_image(img_path)
        H, W = rgb_linear.shape[:2]

        _a = albedo_gt
        if _a.shape[:2] != (H, W):
            _a = cv2.resize(_a, (W, H), interpolation=cv2.INTER_LINEAR)
        rgb_linear = _white_balance_gt(rgb_linear, _a)
        is_hdr = False
        if albedo_gt.shape[:2] != (H, W):
            albedo_gt = cv2.resize(albedo_gt, (W, H), interpolation=cv2.INTER_LINEAR)

        rgb_disp = _external_display_linear(rgb_linear, is_hdr)
        pred, _ = runner(model, rgb_disp, args.max_size, device)
        if pred.shape[:2] != (H, W):
            pred = cv2.resize(pred, (W, H), interpolation=cv2.INTER_LINEAR)

        if albedo_gt.ndim == 2:
            albedo_gt = np.repeat(albedo_gt[..., None], 3, axis=-1)
        if pred.ndim == 2:
            pred = np.repeat(pred[..., None], 3, axis=-1)

        lum = (0.2126 * albedo_gt[..., 0] + 0.7152 * albedo_gt[..., 1]
               + 0.0722 * albedo_gt[..., 2])
        mask = lum > 0.004
        if mask.sum() < 88:
            continue

        try:
            vals = ssim_variants(pred.astype(np.float32), albedo_gt.astype(np.float32), mask)
        except Exception as e:
            print('  variant error:', e)
            continue
        for k, v in vals.items():
            acc.setdefault(k, []).append(v)
        n_used += 1

    print(f'\n=== SSIM variant sweep: {args.model}, {n_used} images ===')
    print(f'{"variant":34s} {"mean SSIM":>10s}   {"|diff vs 0.761|":>15s}')
    rows = [(k, float(np.nanmean(v))) for k, v in acc.items()]
    rows.sort(key=lambda r: abs(r[1] - 0.761))
    for k, v in rows:
        print(f'{k:34s} {v:10.4f}   {abs(v-0.761):15.4f}')


if __name__ == '__main__':
    main()
