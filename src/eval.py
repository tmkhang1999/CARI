"""
Evaluation script for Stage 1 intrinsic decomposition.
Computes scale-invariant metrics on Hypersim validation split.
"""

import argparse
import inspect
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

# Make src importable
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.hypersim_dataset import get_hypersim_loader
from models import (
    IntrinsicDecompositionV6,
    IntrinsicDecompositionV9,
    IntrinsicDecompositionV10,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Intrinsic Decomposition Model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='hypersim', choices=['hypersim'], help='Dataset to evaluate on')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'], help='Dataset split')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--hypersim_root', type=str, default=None, help='Optional Hypersim root override')
    parser.add_argument('--eval_input_size', type=int, default=1024, help='Fixed evaluation resize resolution')
    parser.add_argument('--batch_size', type=int, default=1, help='Evaluation batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='Dataloader workers')
    parser.add_argument('--cache_max_items', type=int, default=64, help='Dataset cache size for eval')
    parser.add_argument('--max_batches', type=int, default=0, help='Limit evaluated batches for quick checks (0 = full split)')
    return parser.parse_args()


def scale_match(A_raw, A_pred, valid):
    v = valid.expand_as(A_raw).float()
    a = (A_raw * v).reshape(A_raw.shape[0], -1)
    b = (A_pred * v).reshape(A_pred.shape[0], -1)
    c = (a * b).sum(dim=1) / ((a * a).sum(dim=1) + 1e-6)
    c = torch.clamp_min(c, 0.05)
    return c.view(-1, 1, 1, 1)


def compute_targets(predictions, rgb, albedo_raw, valid_mask, illum_raw=None, m_diffuse=None):
    eps = 1e-6
    c = scale_match(albedo_raw, predictions['a_d'].detach(), valid_mask)
    A_star = c * albedo_raw
    A_star = torch.nan_to_num(A_star, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)

    # Dec-A/Dec-B targets always come from colorful shading S_c = I / A*
    S_c_star = rgb / (A_star + eps)
    S_c_star = torch.nan_to_num(S_c_star, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)

    # Dec-D target can route to diffuse illumination GT when available.
    S_d_star = S_c_star
    if illum_raw is not None and m_diffuse is not None:
        route = m_diffuse.view(-1, 1, 1, 1).to(device=rgb.device, dtype=rgb.dtype)
        illum = torch.nan_to_num(illum_raw, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        S_d_star = route * illum + (1.0 - route) * S_c_star

    S_g_star = 0.299 * S_c_star[:, 0:1] + 0.587 * S_c_star[:, 1:2] + 0.114 * S_c_star[:, 2:3]
    D_g_star = 1.0 / (S_g_star + 1.0)
    D_g_star = torch.nan_to_num(D_g_star, nan=1.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    C_RG = S_c_star[:, 0:1] / (S_c_star[:, 1:2] + eps)
    C_BG = S_c_star[:, 2:3] / (S_c_star[:, 1:2] + eps)
    xi_star = torch.cat([1.0 / (C_RG + 1.0), 1.0 / (C_BG + 1.0)], dim=1)
    xi_star = torch.nan_to_num(xi_star, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    pi_star = 1.0 / (S_d_star + 1.0)
    pi_star = torch.nan_to_num(pi_star, nan=1.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    return {
        'D_g_star': D_g_star,
        'xi_star': xi_star,
        'A_d_star': A_star,
        'pi_star': pi_star,
    }


def _forward_kwargs(model, m_diffuse, normals, seg, valid_mask):
    sig = inspect.signature(model.forward).parameters
    kwargs = {}
    if 'm_diffuse' in sig:
        kwargs['m_diffuse'] = m_diffuse
    if 'normals' in sig and normals is not None:
        kwargs['normals'] = normals
    if 'seg' in sig and seg is not None:
        kwargs['seg'] = seg
    if 'valid_mask' in sig and valid_mask is not None:
        kwargs['valid_mask'] = valid_mask
    return kwargs


def _apply_diffuse_detach(predictions, m_diffuse):
    mask = m_diffuse.view(-1, 1, 1, 1).to(predictions['pi'].device)
    predictions['pi'] = predictions['pi'] * mask + predictions['pi'].detach() * (1.0 - mask)
    return predictions


def _masked_norm(pred, target, valid_mask, eps=1e-7):
    v = valid_mask.bool().expand_as(pred)
    p = pred[v]
    t = target[v]
    if p.numel() == 0:
        return None, None
    p = p / (p.median() + eps)
    t = t / (t.median() + eps)
    return p, t


def _compute_lmse(pred, target, valid_mask, window_size=20, stride=10, min_valid_ratio=0.5):
    if pred.ndim == 3: pred = pred.unsqueeze(1)
    if target.ndim == 3: target = target.unsqueeze(1)
    if valid_mask.ndim == 3: valid_mask = valid_mask.unsqueeze(1)

    B, C, H, W = pred.shape
    if H < window_size or W < window_size:
        return 0.0

    # 1. Zero out invalid pixels in inputs
    pred = pred * valid_mask
    target = target * valid_mask

    # 2. Extract sliding windows
    unfold = torch.nn.Unfold(kernel_size=window_size, stride=stride)
    p_u = unfold(pred)        # (B, C*K*K, L)
    t_u = unfold(target)      # (B, C*K*K, L)
    m_u = unfold(valid_mask.float())  # (B, 1*K*K, L)

    k2 = window_size * window_size
    valid_count = m_u.sum(dim=1)  # (B, L)
    valid_patch_mask = (valid_count > (min_valid_ratio * k2))  # (B, L)

    if not valid_patch_mask.any():
        return 0.0

    # Revert to fixed N but ensure valid_patch_mask filters properly
    N = float(C * k2)
    sqrt_N = N ** 0.5

    # Only normalize patches where BOTH pred and target have sufficient signal
    p_energy = torch.norm(p_u, p=2, dim=1, keepdim=True)
    t_energy = torch.norm(t_u, p=2, dim=1, keepdim=True)

    # Skip patches where either has near-zero energy (all-invalid or flat)
    min_energy = 1e-3
    has_signal = (p_energy.squeeze(1) > min_energy) & \
                 (t_energy.squeeze(1) > min_energy) & \
                 valid_patch_mask

    if not has_signal.any():
        return 0.0

    p_sc = (p_u / (p_energy + 1e-7)) * sqrt_N
    t_sc = (t_u / (t_energy + 1e-7)) * sqrt_N

    # 5. Compute MSE per patch
    # MSE = mean((p - t)^2)
    diff_sq = (p_sc - t_sc) ** 2
    mse_per_patch = diff_sq.mean(dim=1)  # (B, L)

    # 6. Average over valid patches
    mse_flat = mse_per_patch[has_signal]
    return float(mse_flat.mean().item())


def _masked_scale_invariant_rmse(pred, target, valid_mask, eps=1e-7):
    """Scale-invariant RMSE following Grosse et al. / Careaga & Aksoy."""
    mask = valid_mask.bool().expand_as(pred)

    p = pred[mask]
    t = target[mask]
    if p.numel() == 0:
        return 0.0
    p = p / (p.mean() + eps)
    t = t / (t.mean() + eps)
    return float(torch.sqrt(((p - t) ** 2).mean()).item())


def compute_ssim_bounded(pred, target, valid_mask):
    """SSIM for bounded outputs (e.g., albedo in [0,1])."""
    pred_norm = torch.clamp(pred, 0.0, 1.0)
    target_norm = torch.clamp(target, 0.0, 1.0)

    mask_f = valid_mask.float().expand_as(target_norm)
    pred_norm = pred_norm * mask_f
    target_norm = target_norm * mask_f

    fn_p = pred_norm.squeeze().detach().cpu().numpy()
    fn_t = target_norm.squeeze().detach().cpu().numpy()

    if fn_p.ndim == 3:
        vals = []
        for c in range(fn_p.shape[0]):
            vals.append(ssim(fn_t[c], fn_p[c], data_range=1.0))
        return float(np.mean(vals))
    return float(ssim(fn_t, fn_p, data_range=1.0))



def build_stage1_model(model_cfg):
    version = float(model_cfg.get("version", 10))
    model_config = {
        "z_channels": model_cfg.get("z_channels", 1024),
        "freeze_stages": model_cfg.get("freeze_stages", [1, 2]),
        "backbone": model_cfg.get("backbone", "convnextv2_base"),
        "pretrained": model_cfg.get("pretrained", True),
        "num_seg_classes": model_cfg.get("num_seg_classes", 41),
        "input_size": int(model_cfg.get("input_size", 1024)),
    }
    model_map = {
        6: IntrinsicDecompositionV6,
        9: IntrinsicDecompositionV9,
        10: IntrinsicDecompositionV10,
    }
    if version not in model_map:
        raise ValueError(f"Unsupported Stage1 version: {version}")
    return model_map[version](model_config)


def evaluate_model(model, dataloader, device, max_batches=0):
    model.eval()

    metrics = {
        'a_d_lmse': [],
        'a_d_rmse': [],
        'a_d_ssim': [],
        's_d_lmse': [],
        's_d_rmse': [],
        's_d_ssim': [],
    }

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if max_batches > 0 and batch_idx >= max_batches:
                break

            rgb = batch['rgb'].to(device)
            albedo_raw = batch['albedo_raw'].to(device)
            illum_raw = batch.get('illum_raw', None)
            if illum_raw is not None:
                illum_raw = illum_raw.to(device)
            valid_mask = batch['valid_mask'].to(device)
            m_diffuse = batch['M_diffuse'].float().to(device)
            seg = batch.get('seg')
            if seg is not None:
                seg = seg.to(device)
            normals = batch.get('normals')
            if normals is not None:
                normals = normals.to(device)

            predictions = model(rgb, **_forward_kwargs(model, m_diffuse, normals, seg, valid_mask))
            predictions = _apply_diffuse_detach(predictions, m_diffuse)
            targets = compute_targets(
                predictions,
                rgb,
                albedo_raw,
                valid_mask,
                illum_raw=illum_raw,
                m_diffuse=m_diffuse,
            )

            if 'd_g_raw' in batch:
                targets['D_g_star'] = batch['d_g_raw'].to(device)
            if 'xi_raw' in batch:
                targets['xi_star'] = batch['xi_raw'].to(device)

            # Evaluate shading in inverse space (bounded).
            pi_pred = predictions['pi']
            pi_gt = targets['pi_star']

            has_dg = 'd_g' in predictions
            has_xi = 'xi' in predictions
            if has_dg and 's_g_lmse' not in metrics:
                metrics.update({'s_g_lmse': [], 's_g_rmse': [], 's_g_ssim': []})
            if has_xi and 'xi_mse' not in metrics:
                metrics.update({'xi_mse': []})

            if batch_idx == 0 and has_dg:
                d_g_gt = targets['D_g_star']
                print(f"Sanity Check [Batch 0]:")
                print(f"D_g_gt range: {d_g_gt.min().item():.4f} to {d_g_gt.max().item():.4f}")
                print(f"D_g_star range: {targets['D_g_star'].min().item():.4f} to {targets['D_g_star'].max().item():.4f}")

            for i in range(rgb.shape[0]):
                vm = valid_mask[i:i+1] # (1, 1, H, W)

                if has_dg:
                    d_g_pred = predictions['d_g']
                    d_g_gt = targets['D_g_star']
                    metrics['s_g_lmse'].append(_compute_lmse(d_g_pred[i:i+1], d_g_gt[i:i+1], vm))
                    metrics['s_g_rmse'].append(_masked_scale_invariant_rmse(
                        d_g_pred[i:i+1], d_g_gt[i:i+1], vm
                    ))
                    metrics['s_g_ssim'].append(compute_ssim_bounded(
                        d_g_pred[i:i+1], d_g_gt[i:i+1], vm
                    ))

                # Albedo (A_d)
                metrics['a_d_lmse'].append(_compute_lmse(predictions['a_d'][i:i+1], targets['A_d_star'][i:i+1], vm))
                metrics['a_d_rmse'].append(_masked_scale_invariant_rmse(
                    predictions['a_d'][i:i+1], targets['A_d_star'][i:i+1], vm
                ))
                metrics['a_d_ssim'].append(compute_ssim_bounded(
                    predictions['a_d'][i:i+1], targets['A_d_star'][i:i+1], vm
                ))

                # Diffuse Shading (inverse pi)
                metrics['s_d_lmse'].append(_compute_lmse(pi_pred[i:i+1], pi_gt[i:i+1], vm))
                metrics['s_d_rmse'].append(_masked_scale_invariant_rmse(
                    pi_pred[i:i+1], pi_gt[i:i+1], vm
                ))
                metrics['s_d_ssim'].append(compute_ssim_bounded(
                    pi_pred[i:i+1], pi_gt[i:i+1], vm
                ))

                if has_xi:
                    xi_v = vm.expand_as(predictions['xi'][i:i+1]).float()
                    xi_err = ((predictions['xi'][i:i+1] - targets['xi_star'][i:i+1]) ** 2 * xi_v).sum() / (xi_v.sum() + 1e-7)
                    metrics['xi_mse'].append(float(xi_err.item()))

    avg_metrics = {}
    for key, values in metrics.items():
        if values:
            avg_metrics[key] = float(np.mean(values))
            avg_metrics[f'{key}_std'] = float(np.std(values))
            avg_metrics[f'{key}_min'] = float(np.min(values))
            avg_metrics[f'{key}_max'] = float(np.max(values))
            avg_metrics[f'{key}_count'] = len(values)

    return avg_metrics


def _resolve_hypersim_root(args, config):
    cfg_root = config.get('data', {}).get('hypersim_root', '../datasets/hypersim')
    root = args.hypersim_root or cfg_root
    if not os.path.isabs(root):
        root = str(ROOT_DIR / root)
    return root


def main():
    args = parse_args()

    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint.get('config', {})

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model_cfg = dict(config.get("model", {}))
    model_cfg['input_size'] = int(args.eval_input_size)
    model = build_stage1_model(model_cfg).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")

    hypersim_root = _resolve_hypersim_root(args, config)
    print(f"Loading {args.dataset} dataset ({args.split} split) from {hypersim_root}")

    data_cfg = config.get('data', {})
    dataloader = get_hypersim_loader(
        root_dir=hypersim_root,
        batch_size=int(args.batch_size),
        split=args.split,
        num_workers=int(args.num_workers),
        input_size=int(args.eval_input_size),
        cache_max_items=int(args.cache_max_items),
        crop_mode_train='random',
        crop_mode_val='full',
        split_file=data_cfg.get('hypersim_split_file', 'hypersim_split.json'),
        split_seed=int(data_cfg.get('hypersim_split_seed', 42)),
        split_ratio=float(data_cfg.get('hypersim_split_ratio', 0.9)),
        strict_split=bool(data_cfg.get('hypersim_strict_split', True)),
        max_hdf5_retries=int(data_cfg.get('hypersim_max_hdf5_retries', 1)),
        skip_corrupt_samples=bool(data_cfg.get('hypersim_skip_corrupt_samples', True)),
        pin_memory=False,
    )

    print("\nStarting evaluation...")
    metrics = evaluate_model(
        model,
        dataloader,
        device,
        max_batches=int(args.max_batches),
    )

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    # Get sample count from first metric
    sample_count = int(metrics.get('a_d_lmse_count', 0))
    print(f"\nTotal samples evaluated: {sample_count}\n")

    if 's_g_lmse' in metrics:
        print("Gray Shading (inverse D_g):")
        print(f"  LMSE: {metrics['s_g_lmse']:.6f} (mean) | {metrics.get('s_g_lmse_min', 0):.6f} (min) | {metrics.get('s_g_lmse_max', 0):.6f} (max) ± {metrics.get('s_g_lmse_std', 0):.6f}")
        print(f"  RMSE: {metrics['s_g_rmse']:.6f} (mean) | {metrics.get('s_g_rmse_min', 0):.6f} (min) | {metrics.get('s_g_rmse_max', 0):.6f} (max) ± {metrics.get('s_g_rmse_std', 0):.6f}")
        print(f"  SSIM: {metrics['s_g_ssim']:.4f} (mean) | {metrics.get('s_g_ssim_min', 0):.4f} (min) | {metrics.get('s_g_ssim_max', 0):.4f} (max) ± {metrics.get('s_g_ssim_std', 0):.4f}")

    print("\nDiffuse Albedo (A_d):")
    if 'a_d_lmse' in metrics:
        print(f"  LMSE: {metrics['a_d_lmse']:.6f} (mean) | {metrics.get('a_d_lmse_min', 0):.6f} (min) | {metrics.get('a_d_lmse_max', 0):.6f} (max) ± {metrics.get('a_d_lmse_std', 0):.6f}")
        print(f"  RMSE: {metrics['a_d_rmse']:.6f} (mean) | {metrics.get('a_d_rmse_min', 0):.6f} (min) | {metrics.get('a_d_rmse_max', 0):.6f} (max) ± {metrics.get('a_d_rmse_std', 0):.6f}")
        print(f"  SSIM: {metrics['a_d_ssim']:.4f} (mean) | {metrics.get('a_d_ssim_min', 0):.4f} (min) | {metrics.get('a_d_ssim_max', 0):.4f} (max) ± {metrics.get('a_d_ssim_std', 0):.4f}")

    print("\nDiffuse Shading (inverse pi):")
    if 's_d_lmse' in metrics:
        print(f"  LMSE: {metrics['s_d_lmse']:.6f} (mean) | {metrics.get('s_d_lmse_min', 0):.6f} (min) | {metrics.get('s_d_lmse_max', 0):.6f} (max) ± {metrics.get('s_d_lmse_std', 0):.6f}")
        print(f"  RMSE: {metrics['s_d_rmse']:.6f} (mean) | {metrics.get('s_d_rmse_min', 0):.6f} (min) | {metrics.get('s_d_rmse_max', 0):.6f} (max) ± {metrics.get('s_d_rmse_std', 0):.6f}")
        print(f"  SSIM: {metrics['s_d_ssim']:.4f} (mean) | {metrics.get('s_d_ssim_min', 0):.4f} (min) | {metrics.get('s_d_ssim_max', 0):.4f} (max) ± {metrics.get('s_d_ssim_std', 0):.4f}")

    if 'xi_mse' in metrics:
        print("\nChroma (xi):")
        print(f"  MSE:  {metrics['xi_mse']:.6f} (mean) | {metrics.get('xi_mse_min', 0):.6f} (min) | {metrics.get('xi_mse_max', 0):.6f} (max) ± {metrics.get('xi_mse_std', 0):.6f}")

    print("\n" + "=" * 80)
    print("SUMMARY - Mean Values Only")
    print("=" * 80)
    if 's_g_lmse' in metrics:
        print(f"Gray Shading LMSE:   {metrics['s_g_lmse']:.6f}")
        print(f"Gray Shading RMSE:   {metrics['s_g_rmse']:.6f}")
        print(f"Gray Shading SSIM:   {metrics['s_g_ssim']:.4f}")
    print(f"Albedo LMSE:         {metrics['a_d_lmse']:.6f}")
    print(f"Albedo RMSE:         {metrics['a_d_rmse']:.6f}")
    print(f"Albedo SSIM:         {metrics['a_d_ssim']:.4f}")
    print(f"Diffuse Shading LMSE: {metrics['s_d_lmse']:.6f}")
    print(f"Diffuse Shading RMSE: {metrics['s_d_rmse']:.6f}")
    print(f"Diffuse Shading SSIM: {metrics['s_d_ssim']:.4f}")
    if 'xi_mse' in metrics:
        print(f"Chroma MSE:          {metrics['xi_mse']:.6f}")
    print("=" * 80)


if __name__ == '__main__':
    main()

