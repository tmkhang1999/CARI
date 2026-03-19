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
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# Make src importable
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.hypersim_dataset import get_hypersim_loader
from models import (
    IntrinsicDecompositionV1,
    IntrinsicDecompositionV2,
    IntrinsicDecompositionV3,
    IntrinsicDecompositionV4,
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


def compute_targets(predictions, rgb, albedo_raw, valid_mask):
    eps = 1e-6
    c = scale_match(albedo_raw, predictions['a_d'].detach(), valid_mask)
    A_star = c * albedo_raw
    S_star = rgb / (A_star + eps)

    S_g_star = 0.2126 * S_star[:, 0:1] + 0.7152 * S_star[:, 1:2] + 0.0722 * S_star[:, 2:3]
    D_g_star = 1.0 / (S_g_star + 1.0)

    C_RG = S_star[:, 0:1] / (S_star[:, 1:2] + eps)
    C_BG = S_star[:, 2:3] / (S_star[:, 1:2] + eps)
    xi_star = torch.cat([1.0 / (C_RG + 1.0), 1.0 / (C_BG + 1.0)], dim=1)

    return {
        'D_g_star': D_g_star,
        'xi_star': xi_star,
        'A_d_star': A_star,
        'pi_star': 1.0 / (S_star + 1.0),
    }


def _forward_kwargs(model, m_diffuse, normals, seg):
    sig = inspect.signature(model.forward).parameters
    kwargs = {}
    if 'm_diffuse' in sig:
        kwargs['m_diffuse'] = m_diffuse
    if 'normals' in sig and normals is not None:
        kwargs['normals'] = normals
    if 'seg' in sig and seg is not None:
        kwargs['seg'] = seg
    return kwargs


def _apply_diffuse_detach(predictions, m_diffuse):
    mask = m_diffuse.view(-1, 1, 1, 1).to(predictions['s_d'].device)
    predictions['s_d'] = predictions['s_d'] * mask + predictions['s_d'].detach() * (1.0 - mask)
    return predictions


def _masked_norm(pred, target, valid_mask, eps=1e-7):
    v = valid_mask.bool().expand_as(pred)
    p = pred[v]
    t = target[v]
    if p.numel() == 0:
        return None, None
    p = p / (p.mean() + eps)
    t = t / (t.mean() + eps)
    return p, t


def scale_invariant_mse(pred, target, valid_mask):
    p, t = _masked_norm(pred, target, valid_mask)
    if p is None:
        return 0.0
    return float(((torch.log(p + 1e-7) - torch.log(t + 1e-7)) ** 2).mean().item())


def compute_rmse(pred, target, valid_mask):
    p, t = _masked_norm(pred, target, valid_mask)
    if p is None:
        return 0.0
    return float(torch.sqrt(((p - t) ** 2).mean()).item())


def compute_ssim(pred, target):
    eps = 1e-7
    pred_norm = pred / (pred.mean() + eps)
    target_norm = target / (target.mean() + eps)

    pred_np = pred_norm.detach().cpu().numpy()
    target_np = target_norm.detach().cpu().numpy()

    ssim_values = []
    for c in range(pred_np.shape[0]):
        data_range = float(max(target_np[c].max() - target_np[c].min(), 1e-6))
        ssim_values.append(ssim(target_np[c], pred_np[c], data_range=data_range))
    return float(np.mean(ssim_values))


def build_stage1_model(model_cfg):
    version = int(model_cfg.get("version", 1))
    model_config = {
        "z_channels": model_cfg.get("z_channels", 1024),
        "freeze_stages": model_cfg.get("freeze_stages", [1, 2]),
        "backbone": model_cfg.get("backbone", "convnextv2_base"),
        "pretrained": model_cfg.get("pretrained", True),
        "num_seg_classes": model_cfg.get("num_seg_classes", 41),
        "input_size": int(model_cfg.get("input_size", 1024)),
    }
    model_map = {
        1: IntrinsicDecompositionV1,
        2: IntrinsicDecompositionV2,
        3: IntrinsicDecompositionV3,
        4: IntrinsicDecompositionV4,
    }
    if version not in model_map:
        raise ValueError(f"Unsupported Stage1 version: {version}")
    return model_map[version](model_config)


def evaluate_model(model, dataloader, device, max_batches=0):
    model.eval()

    metrics = {
        's_g_lmse': [],
        's_g_rmse': [],
        's_g_ssim': [],
        'a_d_lmse': [],
        'a_d_rmse': [],
        'a_d_ssim': [],
        's_d_lmse': [],
        's_d_rmse': [],
        's_d_ssim': [],
        'xi_mse': [],
    }

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if max_batches > 0 and batch_idx >= max_batches:
                break

            rgb = batch['rgb'].to(device)
            albedo_raw = batch['albedo_raw'].to(device)
            valid_mask = batch['valid_mask'].to(device)
            m_diffuse = batch['M_diffuse'].float().to(device)
            seg = batch.get('seg')
            if seg is not None:
                seg = seg.to(device)
            normals = batch.get('normals')
            if normals is not None:
                normals = normals.to(device)

            predictions = model(rgb, **_forward_kwargs(model, m_diffuse, normals, seg))
            predictions = _apply_diffuse_detach(predictions, m_diffuse)
            targets = compute_targets(predictions, rgb, albedo_raw, valid_mask)

            inv_s_g_pred = 1.0 / (predictions['s_g'] + 1.0 + 1e-6)
            inv_s_d_pred = 1.0 / (predictions['s_d'] + 1.0 + 1e-6)
            inv_s_g_gt = targets['D_g_star']
            inv_s_d_gt = targets['pi_star']

            for i in range(rgb.shape[0]):
                vm = valid_mask[i:i+1]

                metrics['s_g_lmse'].append(scale_invariant_mse(inv_s_g_pred[i:i+1], inv_s_g_gt[i:i+1], vm))
                metrics['s_g_rmse'].append(compute_rmse(inv_s_g_pred[i:i+1], inv_s_g_gt[i:i+1], vm))
                metrics['s_g_ssim'].append(compute_ssim(inv_s_g_pred[i], inv_s_g_gt[i]))

                metrics['a_d_lmse'].append(scale_invariant_mse(predictions['a_d'][i:i+1], targets['A_d_star'][i:i+1], vm))
                metrics['a_d_rmse'].append(compute_rmse(predictions['a_d'][i:i+1], targets['A_d_star'][i:i+1], vm))
                metrics['a_d_ssim'].append(compute_ssim(predictions['a_d'][i], targets['A_d_star'][i]))

                metrics['s_d_lmse'].append(scale_invariant_mse(inv_s_d_pred[i:i+1], inv_s_d_gt[i:i+1], vm))
                metrics['s_d_rmse'].append(compute_rmse(inv_s_d_pred[i:i+1], inv_s_d_gt[i:i+1], vm))
                metrics['s_d_ssim'].append(compute_ssim(inv_s_d_pred[i], inv_s_d_gt[i]))

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
    model.load_state_dict(checkpoint['model_state_dict'])
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
    sample_count = metrics.get('s_g_lmse_count', 0)
    print(f"\nTotal samples evaluated: {sample_count}\n")

    print("Gray Shading (inverse D_g = 1/(1+S_g)):")
    if 's_g_lmse' in metrics:
        print(f"  LMSE: {metrics['s_g_lmse']:.6f} (mean) | {metrics.get('s_g_lmse_min', 0):.6f} (min) | {metrics.get('s_g_lmse_max', 0):.6f} (max) ± {metrics.get('s_g_lmse_std', 0):.6f}")
        print(f"  RMSE: {metrics['s_g_rmse']:.6f} (mean) | {metrics.get('s_g_rmse_min', 0):.6f} (min) | {metrics.get('s_g_rmse_max', 0):.6f} (max) ± {metrics.get('s_g_rmse_std', 0):.6f}")
        print(f"  SSIM: {metrics['s_g_ssim']:.4f} (mean) | {metrics.get('s_g_ssim_min', 0):.4f} (min) | {metrics.get('s_g_ssim_max', 0):.4f} (max) ± {metrics.get('s_g_ssim_std', 0):.4f}")

    print("\nDiffuse Albedo (A_d):")
    if 'a_d_lmse' in metrics:
        print(f"  LMSE: {metrics['a_d_lmse']:.6f} (mean) | {metrics.get('a_d_lmse_min', 0):.6f} (min) | {metrics.get('a_d_lmse_max', 0):.6f} (max) ± {metrics.get('a_d_lmse_std', 0):.6f}")
        print(f"  RMSE: {metrics['a_d_rmse']:.6f} (mean) | {metrics.get('a_d_rmse_min', 0):.6f} (min) | {metrics.get('a_d_rmse_max', 0):.6f} (max) ± {metrics.get('a_d_rmse_std', 0):.6f}")
        print(f"  SSIM: {metrics['a_d_ssim']:.4f} (mean) | {metrics.get('a_d_ssim_min', 0):.4f} (min) | {metrics.get('a_d_ssim_max', 0):.4f} (max) ± {metrics.get('a_d_ssim_std', 0):.4f}")

    print("\nDiffuse Shading (inverse pi = 1/(1+S_d)):")
    if 's_d_lmse' in metrics:
        print(f"  LMSE: {metrics['s_d_lmse']:.6f} (mean) | {metrics.get('s_d_lmse_min', 0):.6f} (min) | {metrics.get('s_d_lmse_max', 0):.6f} (max) ± {metrics.get('s_d_lmse_std', 0):.6f}")
        print(f"  RMSE: {metrics['s_d_rmse']:.6f} (mean) | {metrics.get('s_d_rmse_min', 0):.6f} (min) | {metrics.get('s_d_rmse_max', 0):.6f} (max) ± {metrics.get('s_d_rmse_std', 0):.6f}")
        print(f"  SSIM: {metrics['s_d_ssim']:.4f} (mean) | {metrics.get('s_d_ssim_min', 0):.4f} (min) | {metrics.get('s_d_ssim_max', 0):.4f} (max) ± {metrics.get('s_d_ssim_std', 0):.4f}")

    print("\nChroma (xi):")
    if 'xi_mse' in metrics:
        print(f"  MSE:  {metrics['xi_mse']:.6f} (mean) | {metrics.get('xi_mse_min', 0):.6f} (min) | {metrics.get('xi_mse_max', 0):.6f} (max) ± {metrics.get('xi_mse_std', 0):.6f}")

    print("\n" + "=" * 80)
    print("SUMMARY - Mean Values Only")
    print("=" * 80)
    print(f"Gray Shading LMSE:   {metrics['s_g_lmse']:.6f}")
    print(f"Gray Shading RMSE:   {metrics['s_g_rmse']:.6f}")
    print(f"Gray Shading SSIM:   {metrics['s_g_ssim']:.4f}")
    print(f"Albedo LMSE:         {metrics['a_d_lmse']:.6f}")
    print(f"Albedo RMSE:         {metrics['a_d_rmse']:.6f}")
    print(f"Albedo SSIM:         {metrics['a_d_ssim']:.4f}")
    print(f"Diffuse Shading LMSE: {metrics['s_d_lmse']:.6f}")
    print(f"Diffuse Shading RMSE: {metrics['s_d_rmse']:.6f}")
    print(f"Diffuse Shading SSIM: {metrics['s_d_ssim']:.4f}")
    print(f"Chroma MSE:          {metrics['xi_mse']:.6f}")
    print("=" * 80)


if __name__ == '__main__':
    main()

