"""
Training script for Stage 1 intrinsic decomposition.
"""

import argparse
import inspect
import os
import re
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models import (
    IntrinsicDecompositionV1,
    IntrinsicDecompositionV2,
    IntrinsicDecompositionV3,
    IntrinsicDecompositionV4,
)
from losses.flexible_loss import FlexibleLoss
from data.hypersim_dataset import HypersimDataset, get_hypersim_loader
from data.midintrinsic_dataset import MIDIntrinsicDataset
from data.mixed_dataloader import MixedDataloader


TB_TAGS = {
    'loss_a': '01_Losses/1_a',
    'loss_b': '01_Losses/2_b',
    'loss_c': '01_Losses/3_c_total',
    'loss_d': '01_Losses/4_d',
    'loss_total': '01_Losses/5_total',
    'loss_c_l1': '01_Losses/3_c_L1',
    'loss_c_msg': '01_Losses/3_c_MSG',
    'loss_c_perceptual': '01_Losses/3_c_Perceptual',
    'loss_c_tv': '01_Losses/3_c_TV',
    'a_d_lmse': '02_Albedo_Ad/1_lmse',
    'a_d_rmse': '02_Albedo_Ad/2_rmse',
    'a_d_ssim': '02_Albedo_Ad/3_ssim',
    's_g_lmse': '03_GrayShading_Sg/1_lmse',
    's_g_rmse': '03_GrayShading_Sg/2_rmse',
    's_g_ssim': '03_GrayShading_Sg/3_ssim',
    'xi_mse': '04_Chroma_xi/1_mse',
    's_d_lmse': '05_DiffuseShading_Sd/1_lmse',
    's_d_rmse': '05_DiffuseShading_Sd/2_rmse',
    's_d_ssim': '05_DiffuseShading_Sd/3_ssim',
}


def _log_ordered_scalars(writer, values, global_step):
    """Write only requested TensorBoard tags in deterministic order."""
    ordered = [
        'loss_a', 'loss_b', 'loss_c', 'loss_d', 'loss_total',
        'loss_c_l1', 'loss_c_msg', 'loss_c_perceptual', 'loss_c_tv',
        'a_d_lmse', 'a_d_rmse', 'a_d_ssim',
        's_g_lmse', 's_g_rmse', 's_g_ssim',
        'xi_mse',
        's_d_lmse', 's_d_rmse', 's_d_ssim',
    ]
    for key in ordered:
        if key not in values:
            continue
        val = values[key]
        if isinstance(val, torch.Tensor):
            val = val.item()
        writer.add_scalar(TB_TAGS[key], float(val), global_step)


def scale_match(A_raw, A_pred, valid):
    """Least-squares per-image scalar c for c*A_raw ~= A_pred over valid pixels."""
    v = valid.expand_as(A_raw).float()
    a = (A_raw * v).reshape(A_raw.shape[0], -1)
    b = (A_pred * v).reshape(A_pred.shape[0], -1)
    c = (a * b).sum(dim=1) / ((a * a).sum(dim=1) + 1e-6)
    # Prevent degenerate target scaling that can zero-out Dec-C supervision.
    c = torch.clamp_min(c, 0.05)
    return c.view(-1, 1, 1, 1)


def compute_targets(predictions, rgb, albedo_raw, valid_mask):
    """Compute scale-matched training targets after forward pass."""
    eps = 1e-6
    c = scale_match(albedo_raw, predictions['a_d'].detach(), valid_mask)

    A_star = c * albedo_raw
    S_star = rgb / (A_star + eps)

    S_g_star = (
        0.2126 * S_star[:, 0:1]
        + 0.7152 * S_star[:, 1:2]
        + 0.0722 * S_star[:, 2:3]
    )
    D_g_star = 1.0 / (S_g_star + 1.0)

    C_RG = S_star[:, 0:1] / (S_star[:, 1:2] + eps)
    C_BG = S_star[:, 2:3] / (S_star[:, 1:2] + eps)
    xi_star = torch.cat([1.0 / (C_RG + 1.0), 1.0 / (C_BG + 1.0)], dim=1)

    A_d_star = A_star
    pi_star = 1.0 / (S_star + 1.0)

    return {
        'D_g_star': D_g_star,
        'xi_star': xi_star,
        'A_d_star': A_d_star,
        'pi_star': pi_star,
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Train Stage 1')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--version', type=int, default=None)
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint path or "latest"')
    parser.add_argument('--auto-resume', action='store_true', help='Resume from latest checkpoint in version checkpoint dir')
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


def _deep_merge(base, override):
    merged = base.copy()
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_config(config_path=None, version=None):
    base_path = SRC_DIR / 'configs' / 'base.yaml'
    with open(base_path, 'r') as f:
        config = yaml.safe_load(f)

    if config_path is None and version is not None:
        config_path = str(SRC_DIR / 'configs' / f'v{version}.yaml')

    if config_path is not None and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            override = yaml.safe_load(f) or {}
        config = _deep_merge(config, override)
        print(f"Config: base.yaml <- {os.path.basename(config_path)}")
    else:
        print("Config: base.yaml only")
    return config


def save_checkpoint(model, optimizer, losses, config, filename, global_step):
    ckpt = {
        'global_step': int(global_step),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
        'config': config,
    }
    torch.save(ckpt, filename)
    print(f"Saved checkpoint to {filename}")


def load_checkpoint(model, optimizer, checkpoint_path, map_location=None):
    ckpt = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    global_step = int(ckpt.get('global_step', 0))
    losses = ckpt.get('losses', {})
    print(f"Loaded checkpoint at global_step={global_step}: {checkpoint_path}")
    return global_step, losses


def _extract_iter_from_name(path):
    m = re.search(r'checkpoint_iter_(\d+)\.pth$', os.path.basename(path))
    return int(m.group(1)) if m else -1


def _find_latest_checkpoint(ckpt_dir):
    if not os.path.isdir(ckpt_dir):
        return None
    files = [
        os.path.join(ckpt_dir, f)
        for f in os.listdir(ckpt_dir)
        if f.startswith('checkpoint_iter_') and f.endswith('.pth')
    ]
    if not files:
        return None
    files.sort(key=_extract_iter_from_name)
    return files[-1]


def _resolve_resume_path(resume_arg, auto_resume, ckpt_dir):
    if auto_resume or (isinstance(resume_arg, str) and resume_arg.lower() == 'latest'):
        latest = _find_latest_checkpoint(ckpt_dir)
        if latest is None:
            raise FileNotFoundError(
                f"Resume requested but no checkpoint found in: {ckpt_dir}"
            )
        return latest

    if not resume_arg:
        return None

    # Resolve relative paths against project root for convenience.
    if not os.path.isabs(resume_arg):
        candidate = str(ROOT_DIR / resume_arg)
        if os.path.exists(candidate):
            return candidate
    return resume_arg


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
    """Detach only Dec-D output where diffuse supervision is unavailable."""
    mask = m_diffuse.view(-1, 1, 1, 1).to(predictions['s_d'].device)
    predictions['s_d'] = predictions['s_d'] * mask + predictions['s_d'].detach() * (1.0 - mask)
    return predictions


def _loss_seg(model, seg):
    return seg if 'seg' in inspect.signature(model.forward).parameters else None


def train_one_step(model, batch, criterion, optimizer, device):
    model.train()

    rgb = batch['rgb'].to(device)
    albedo_raw = batch['albedo_raw'].to(device)
    valid_mask = batch['valid_mask'].to(device)
    m_diffuse = batch['M_diffuse'].float().to(device)
    m_albedo = batch['M_albedo'].float().to(device)
    seg = batch.get('seg', None)
    if seg is not None:
        seg = seg.to(device)
    normals = batch.get('normals', None)
    if normals is not None:
        normals = normals.to(device)

    predictions = model(rgb, **_forward_kwargs(model, m_diffuse, normals, seg))
    predictions = _apply_diffuse_detach(predictions, m_diffuse)
    targets = compute_targets(predictions, rgb, albedo_raw, valid_mask)

    losses = criterion(
        predictions,
        targets,
        m_diffuse,
        m_albedo,
        valid_mask,
        _loss_seg(model, seg),
    )

    optimizer.zero_grad()
    losses['loss_total'].backward()
    optimizer.step()
    return losses


def _compute_lmse(pred, target, valid_mask, window_size=20, stride=10, min_valid_ratio=0.5):
    """
    Standard Local Mean Squared Error (LMSE) with sliding window.
    Based on Grosse et al. (2009) definition: local windows are rescaled
    so that their mean squared magnitude is 1.
    """
    if pred.ndim == 3: pred = pred.unsqueeze(1)
    if target.ndim == 3: target = target.unsqueeze(1)
    if valid_mask.ndim == 3: valid_mask = valid_mask.unsqueeze(1)

    B, C, H, W = pred.shape
    if H < window_size or W < window_size:
        return torch.tensor(0.0, device=pred.device)

    # 1. Zero out invalid pixels in inputs
    pred = pred * valid_mask
    target = target * valid_mask

    # 2. Extract sliding windows
    unfold = torch.nn.Unfold(kernel_size=window_size, stride=stride)
    p_u = unfold(pred)        # (B, C*K*K, L)
    t_u = unfold(target)      # (B, C*K*K, L)
    m_u = unfold(valid_mask)  # (B, 1*K*K, L)

    # 3. Identify valid patches (at least 50% valid pixels)
    # Note: mask is single channel, valid for all channels
    k2 = window_size * window_size
    valid_count = m_u.sum(dim=1)  # (B, L)
    valid_patch_mask = (valid_count > (min_valid_ratio * k2))  # (B, L)

    if not valid_patch_mask.any():
        return torch.tensor(0.0, device=pred.device)

    # 4. Normalize patches (Mean Squared Magnitude = 1)
    # MSM(x) = sum(x^2)/N. For MSM=1, sum(x^2)=N.
    # So we scale by sqrt(N) / norm2(x).
    N = float(C * k2)
    sqrt_N = N ** 0.5

    p_norm = torch.norm(p_u, p=2, dim=1, keepdim=True) + 1e-7
    t_norm = torch.norm(t_u, p=2, dim=1, keepdim=True) + 1e-7

    p_sc = (p_u / p_norm) * sqrt_N
    t_sc = (t_u / t_norm) * sqrt_N

    # 5. Compute MSE per patch
    # MSE = mean((p - t)^2)
    diff_sq = (p_sc - t_sc) ** 2
    mse_per_patch = diff_sq.mean(dim=1)  # (B, L)

    # 6. Average over valid patches
    # Flatten batch and L
    mse_flat = mse_per_patch[valid_patch_mask]
    return mse_flat.mean()


def _masked_scale_invariant_rmse(pred, target, valid_mask, eps=1e-7):
    """RMSE after per-sample mean normalization on valid pixels."""
    v = valid_mask.bool().expand_as(pred)
    p = pred[v]
    t = target[v]
    if p.numel() == 0:
        return pred.new_tensor(0.0)
    p = p / (p.mean() + eps)
    t = t / (t.mean() + eps)
    return torch.sqrt(((p - t) ** 2).mean())


def _ssim_from_dssim(criterion, pred, target):
    """Convert criterion DSSIM helper to SSIM in [0,1]."""
    dssim = criterion._compute_dssim(pred, target)
    return 1.0 - 2.0 * dssim


def _vis_tonemap(img, percentile=99.0, eps=1e-6, scale=None):
    """Inference-style tonemap to [0,1] per sample with NaN/Inf guards."""
    if img.ndim == 4:
        img = img[0]
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    img = torch.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    if scale is None:
        scale = torch.quantile(img.reshape(-1), percentile / 100.0)
    scale = torch.as_tensor(scale, device=img.device, dtype=img.dtype).clamp_min(eps)
    return torch.clamp(img / scale, 0.0, 1.0)


def _log_val_examples(writer, global_step, rgb, predictions, targets, max_items=2, full_rgb_list=None, sample_index=None):
    """Log qualitative validation examples in the exact requested tile order.
    
    Args:
        sample_index: Optional global sample index for labeling in TensorBoard.
                     If provided, logged under Examples/{sample_index} tag.
    """
    # Tile order must match user request: original_input, cropped_input, s_g_pred, s_g_gt, a_d_pred, a_d_gt, s_d_pred, s_d_gt
    def _tile_full(i):
        if full_rgb_list is None or i >= len(full_rgb_list) or full_rgb_list[i] is None:
            return None
        return _vis_tonemap(full_rgb_list[i])

    def _normalize_tile(tile, target_hw=None):
        """Move tile to CPU CHW float for safe concatenation and TensorBoard logging."""
        if tile is None:
            return None
        if tile.ndim == 4:
            if tile.shape[0] == 0:
                raise ValueError('tile has empty batch dimension')
            tile = tile[0]
        if tile.ndim == 2:
            tile = tile.unsqueeze(0)
        if tile.ndim != 3:
            raise ValueError(f'expected CHW tile, got shape={tuple(tile.shape)}')
        if tile.shape[0] == 1:
            tile = tile.repeat(3, 1, 1)
        elif tile.shape[0] > 3:
            tile = tile[:3]

        tile = tile.detach().to(device='cpu', dtype=torch.float32)
        tile = torch.clamp(tile, 0.0, 1.0)

        if target_hw is not None and (tile.shape[1] != target_hw[0] or tile.shape[2] != target_hw[1]):
            tile = torch.nn.functional.interpolate(
                tile.unsqueeze(0),
                size=target_hw,
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)
        return tile

    layout_names = [
        '00_original_input',
        '01_cropped_input',
        '02_inv_s_g_pred',
        '03_inv_s_g_gt',
        '04_a_d_pred',
        '05_a_d_gt',
        '06_inv_s_d_pred',
        '07_inv_s_d_gt',
    ] if full_rgb_list else [
        '00_cropped_input',
        '01_inv_s_g_pred',
        '02_inv_s_g_gt',
        '03_a_d_pred',
        '04_a_d_gt',
        '05_inv_s_d_pred',
        '06_inv_s_d_gt',
    ]

    # Set TensorBoard tag based on sample_index
    if sample_index is not None:
        example_tag = f'06_Examples/sample_{sample_index}'
    else:
        example_tag = '06_Examples'
    
    writer.add_text(
        f'{example_tag}/layout',
        'left-to-right strip order: ' + ' | '.join(layout_names),
        global_step,
    )

    b = min(int(rgb.shape[0]), int(max_items))
    for i in range(b):
        named_tiles = []

        inv_s_g_pred = 1.0 / (predictions['s_g'][i:i+1] + 1.0 + 1e-6)
        inv_s_g_gt = targets['D_g_star'][i:i+1]
        inv_s_d_pred = 1.0 / (predictions['s_d'][i:i+1] + 1.0 + 1e-6)
        inv_s_d_gt = targets['pi_star'][i:i+1]

        sample_tiles = []
        if full_rgb_list:
            sample_tiles.append(('00_original_input', _tile_full(i)))

        sample_tiles.extend([
            (f"{'01' if full_rgb_list else '00'}_cropped_input", _vis_tonemap(rgb[i:i+1])),
            (f"{'02' if full_rgb_list else '01'}_inv_s_g_pred", _vis_tonemap(inv_s_g_pred)),
            (f"{'03' if full_rgb_list else '02'}_inv_s_g_gt", _vis_tonemap(inv_s_g_gt)),
            (f"{'04' if full_rgb_list else '03'}_a_d_pred", _vis_tonemap(predictions['a_d'][i:i+1])),
            (f"{'05' if full_rgb_list else '04'}_a_d_gt", _vis_tonemap(targets['A_d_star'][i:i+1])),
            (f"{'06' if full_rgb_list else '05'}_inv_s_d_pred", _vis_tonemap(inv_s_d_pred)),
            (f"{'07' if full_rgb_list else '06'}_inv_s_d_gt", _vis_tonemap(inv_s_d_gt)),
        ])

        target_hw = None
        for name, tile in sample_tiles:
            try:
                if tile is None:
                    continue
                norm_tile = _normalize_tile(tile, target_hw=target_hw)
                if norm_tile is None:
                    continue
                if target_hw is None:
                    target_hw = (int(norm_tile.shape[1]), int(norm_tile.shape[2]))
                named_tiles.append((name, norm_tile))
            except Exception as exc:
                print(f"[warn][val step {global_step}] skipped tile '{name}' for sample_{i}: {exc}")

        if not named_tiles:
            continue

        strip = torch.cat([t for _, t in named_tiles], dim=2)
        writer.add_image(f'{example_tag}/sample_{i}', strip, global_step)


def validate(model, dataloader, criterion, device, global_step, writer, val_example_images=2, val_example_indices=None):
    """
    Validation for ablation comparison on fixed Hypersim val split.
    Logs both losses and per-decoder metrics from scale-matched targets.
    
    Args:
        val_example_indices: list of global sample indices to log, e.g., [100, 110, 120].
                           If provided, val_example_images is ignored.
    """
    model.eval()
    total_loss = {
        'loss_a': 0.0,
        'loss_b': 0.0,
        'loss_c': 0.0,
        'loss_d': 0.0,
        'loss_total': 0.0,
        'loss_c_l1': 0.0,
        'loss_c_msg': 0.0,
        'loss_c_perceptual': 0.0,
        'loss_c_tv': 0.0,
    }
    total_metric = {
        's_g_lmse': 0.0,
        's_g_rmse': 0.0,
        's_g_ssim': 0.0,
        'xi_mse': 0.0,
        'a_d_lmse': 0.0,
        'a_d_rmse': 0.0,
        'a_d_ssim': 0.0,
        's_d_lmse': 0.0,
        's_d_rmse': 0.0,
        's_d_ssim': 0.0,
    }
    n_samples = 0
    n_s_d_samples = 0
    
    # Prepare sample collection for visualization
    if val_example_indices is None:
        val_example_indices = []
    val_example_indices = list(val_example_indices) if val_example_indices else []
    use_indices = len(val_example_indices) > 0
    use_count = not use_indices
    
    # Maps global sample index to (batch_idx, sample_in_batch)
    indices_to_collect = {idx: None for idx in val_example_indices}
    collected_samples = {}  # Maps global index to (rgb, predictions, targets)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc='Validation')):
            rgb = batch['rgb'].to(device)
            albedo_raw = batch['albedo_raw'].to(device)
            valid_mask = batch['valid_mask'].to(device)
            m_diffuse = batch['M_diffuse'].float().to(device)
            m_albedo = batch['M_albedo'].float().to(device)
            seg = batch.get('seg', None)
            if seg is not None:
                seg = seg.to(device)
            normals = batch.get('normals', None)
            if normals is not None:
                normals = normals.to(device)

            predictions = model(rgb, **_forward_kwargs(model, m_diffuse, normals, seg))
            predictions = _apply_diffuse_detach(predictions, m_diffuse)
            targets = compute_targets(predictions, rgb, albedo_raw, valid_mask)

            # Collect samples at specified indices
            batch_size = rgb.shape[0]
            if use_indices:
                for i in range(batch_size):
                    global_idx = batch_idx * dataloader.batch_size + i
                    if global_idx in indices_to_collect:
                        collected_samples[global_idx] = {
                            'rgb': rgb[i:i+1],
                            'predictions': predictions,
                            'targets': targets,
                        }
            
            # For count-based mode, log from first batch only
            if use_count and batch_idx == 0:
                _log_val_examples(
                    writer,
                    global_step,
                    rgb,
                    predictions,
                    targets,
                    max_items=val_example_images,
                    full_rgb_list=None,
                )

            losses = criterion(predictions, targets, m_diffuse, m_albedo, valid_mask, _loss_seg(model, seg))
            for k in total_loss:
                if k in losses:
                    total_loss[k] += losses[k].item()

            # Reconstruct target-space shading tensors for requested metrics.
            s_g_star = 1.0 / (targets['D_g_star'] + 1e-6) - 1.0
            s_d_star = 1.0 / (targets['pi_star'] + 1e-6) - 1.0

            batch_size = rgb.shape[0]
            for i in range(batch_size):
                vm = valid_mask[i:i+1]

                # Dec A metrics on S_g
                total_metric['s_g_lmse'] += _compute_lmse(
                    predictions['s_g'][i:i+1], s_g_star[i:i+1], vm
                ).item()
                total_metric['s_g_rmse'] += _masked_scale_invariant_rmse(
                    predictions['s_g'][i:i+1], s_g_star[i:i+1], vm
                ).item()
                total_metric['s_g_ssim'] += float(_ssim_from_dssim(
                    criterion, predictions['s_g'][i:i+1], s_g_star[i:i+1]
                ))

                # Dec B metric on xi
                xi_v = vm.expand_as(predictions['xi'][i:i+1]).float()
                xi_err = ((predictions['xi'][i:i+1] - targets['xi_star'][i:i+1]) ** 2 * xi_v).sum() / (xi_v.sum() + 1e-7)
                total_metric['xi_mse'] += xi_err.item()

                # Dec C metrics on A_d
                total_metric['a_d_lmse'] += _compute_lmse(
                    predictions['a_d'][i:i+1], targets['A_d_star'][i:i+1], vm
                ).item()
                total_metric['a_d_rmse'] += _masked_scale_invariant_rmse(
                    predictions['a_d'][i:i+1], targets['A_d_star'][i:i+1], vm
                ).item()
                total_metric['a_d_ssim'] += float(_ssim_from_dssim(
                    criterion, predictions['a_d'][i:i+1], targets['A_d_star'][i:i+1]
                ))

                # Dec D metrics on S_d (count only when diffuse GT is available)
                if m_diffuse[i].item() > 0.5:
                    s_d_lmse = _compute_lmse(
                        predictions['s_d'][i:i+1], s_d_star[i:i+1], vm
                    )
                    s_d_rmse = _masked_scale_invariant_rmse(
                        predictions['s_d'][i:i+1], s_d_star[i:i+1], vm
                    )
                    s_d_ssim = torch.as_tensor(_ssim_from_dssim(
                        criterion, predictions['s_d'][i:i+1], s_d_star[i:i+1]
                    ), device=rgb.device)

                    # Guard against NaN/Inf so TensorBoard tags are always emitted.
                    if torch.isfinite(s_d_lmse):
                        total_metric['s_d_lmse'] += s_d_lmse.item()
                    if torch.isfinite(s_d_rmse):
                        total_metric['s_d_rmse'] += s_d_rmse.item()
                    if torch.isfinite(s_d_ssim):
                        total_metric['s_d_ssim'] += s_d_ssim.item()

                    n_s_d_samples += 1
                n_samples += 1

    # Log collected samples for index-based visualization
    if use_indices and collected_samples:
        # Sort by global index for consistent logging order
        sorted_indices = sorted(collected_samples.keys())
        for global_idx in sorted_indices:
            sample = collected_samples[global_idx]
            _log_val_examples(
                writer,
                global_step,
                sample['rgb'],
                sample['predictions'],
                sample['targets'],
                max_items=1,
                full_rgb_list=None,
                sample_index=global_idx,
            )

    denom_loss = max(len(dataloader), 1)
    for k in total_loss:
        total_loss[k] /= denom_loss

    denom_metric = max(n_samples, 1)
    for k in total_metric:
        if k.startswith('s_d_'):
            total_metric[k] /= max(n_s_d_samples, 1)
        else:
            total_metric[k] /= denom_metric

    val_out = {}
    val_out.update(total_loss)
    val_out.update(total_metric)
    _log_ordered_scalars(writer, val_out, global_step)


    # Return combined dict for terminal reporting.
    out = {}
    out.update(total_loss)
    out.update(total_metric)
    return out


def _phase_schedule(train_cfg, global_step):
    """Two-phase curriculum exactly as requested by config values."""
    p1 = int(train_cfg.get('phase1_iterations', 20000))
    w1 = train_cfg.get('sampling_weights_phase1', {'hypersim': 1.0, 'midintrinsic': 0.0})
    w2 = train_cfg.get('sampling_weights_phase2', {'hypersim': 0.6, 'midintrinsic': 0.4})
    if global_step < p1:
        return 'phase1', w1
    return 'phase2', w2


def build_stage1_model(config):
    version = int(config['model'].get('version', 1))
    model_cfg = {
        'z_channels': config['model'].get('z_channels', 1024),
        'freeze_stages': config['model'].get('freeze_stages', [1, 2]),
        'backbone': config['model'].get('backbone', 'convnextv2_base'),
        'pretrained': config['model'].get('pretrained', True),
        'num_seg_classes': config['model'].get('num_seg_classes', 41),
        'input_size': int(config['train'].get('input_size', 384)),
    }
    model_map = {
        1: IntrinsicDecompositionV1,
        2: IntrinsicDecompositionV2,
        3: IntrinsicDecompositionV3,
        4: IntrinsicDecompositionV4,
    }
    if version not in model_map:
        raise ValueError(f"Unsupported Stage1 version: {version}")
    return model_map[version](model_cfg)


def _subset_dataset(dataset, dataset_name, max_images, seed):
    """Return deterministic subset for dataset-size ablations."""
    total = len(dataset)
    max_images = int(max_images)
    if max_images <= 0:
        return dataset, total, total
    if max_images >= total:
        print(
            f"[subset] {dataset_name}: requested {max_images} >= total {total}; "
            "using full dataset"
        )
        return dataset, total, total

    gen = torch.Generator()
    gen.manual_seed(int(seed))
    indices = torch.randperm(total, generator=gen)[:max_images].tolist()
    subset = Subset(dataset, indices)
    print(
        f"[subset] {dataset_name}: using {len(subset)}/{total} images "
        f"(seed={int(seed)})"
    )
    return subset, total, len(subset)


def main():
    args = parse_args()
    config = load_config(args.config, args.version)
    print(yaml.dump(config, default_flow_style=False))

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    version = int(config['model']['version'])
    ckpt_dir = os.path.join(config['paths']['checkpoint_dir'], f'v{version}')
    log_dir = os.path.join(config['paths']['log_dir'], f'v{version}')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)
    model = build_stage1_model(config).to(device)
    criterion = FlexibleLoss(config['loss']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config['train']['lr']))

    hypersim_root = config['data']['hypersim_root']
    mid_root = config['data'].get('midintrinsic_root', '../datasets/MIDIntrinsics')
    if not os.path.isabs(hypersim_root):
        hypersim_root = str(ROOT_DIR / hypersim_root)
    if not os.path.isabs(mid_root):
        mid_root = str(ROOT_DIR / mid_root)

    split_file = config['data'].get('hypersim_split_file', 'hypersim_split.json')
    split_seed = int(config['data'].get('hypersim_split_seed', 42))
    split_ratio = float(config['data'].get('hypersim_split_ratio', 0.9))
    strict_split = bool(config['data'].get('hypersim_strict_split', True))
    hypersim_max_hdf5_retries = int(config['data'].get('hypersim_max_hdf5_retries', 1))
    hypersim_skip_corrupt_samples = bool(config['data'].get('hypersim_skip_corrupt_samples', True))

    enabled = set(config['data'].get('datasets', ['hypersim']))
    cache_max_items = int(config['data'].get('cache_max_items', 512))
    crop_mode_train = str(config['data'].get('crop_mode_train', 'random'))
    crop_mode_val = str(config['data'].get('crop_mode_val', 'center'))
    loader_seed = int(config['train'].get('dataloader_seed', 42))
    subset_seed = int(config['data'].get('subset_seed', loader_seed))
    hypersim_train_max_images = int(config['data'].get('hypersim_train_max_images', 0))
    midintrinsic_train_max_images = int(config['data'].get('midintrinsic_train_max_images', 0))

    train_hypersim_base = HypersimDataset(
        root_dir=hypersim_root,
        split='train',
        input_size=int(config['train']['input_size']),
        cache_max_items=cache_max_items,
        crop_mode_train=crop_mode_train,
        crop_mode_val=crop_mode_val,
        split_file=split_file,
        split_seed=split_seed,
        split_ratio=split_ratio,
        strict_split=strict_split,
        max_hdf5_retries=hypersim_max_hdf5_retries,
        skip_corrupt_samples=hypersim_skip_corrupt_samples,
    )
    train_hypersim, hypersim_total, hypersim_effective = _subset_dataset(
        train_hypersim_base,
        'hypersim',
        hypersim_train_max_images,
        subset_seed,
    )

    datasets = {'hypersim': train_hypersim}
    train_mid = None
    if 'midintrinsic' in enabled:
        train_mid = MIDIntrinsicDataset(
            root_dir=mid_root,
            split='train',
            input_size=int(config['train']['input_size']),
            cache_max_items=int(config['data'].get('mid_cache_max_items', cache_max_items)),
            crop_mode_train=crop_mode_train,
            crop_mode_val=crop_mode_val,
        )
        train_mid, mid_total, mid_effective = _subset_dataset(
            train_mid,
            'midintrinsic',
            midintrinsic_train_max_images,
            subset_seed,
        )
        datasets['midintrinsic'] = train_mid

    train_num_workers = int(config['train'].get('num_workers', 4))
    val_num_workers = max(1, int(config['data'].get('val_num_workers', config['train'].get('val_num_workers', 2))))
    val_cache_max_items = max(0, int(config['data'].get('val_cache_max_items', 64)))

    mixed_loader = MixedDataloader(
        datasets=datasets,
        weights=config['train'].get('sampling_weights_phase1', {'hypersim': 1.0, 'midintrinsic': 0.0}),
        batch_size=int(config['train']['batch_size']),
        num_workers=train_num_workers,
        seed=loader_seed,
    )

    val_loader = get_hypersim_loader(
        root_dir=hypersim_root,
        batch_size=int(config['train']['batch_size']),
        split='val',
        num_workers=val_num_workers,
        input_size=int(config['train']['input_size']),
        cache_max_items=val_cache_max_items,
        crop_mode_train=crop_mode_train,
        crop_mode_val=crop_mode_val,
        split_file=split_file,
        split_seed=split_seed,
        split_ratio=split_ratio,
        strict_split=strict_split,
        max_hdf5_retries=hypersim_max_hdf5_retries,
        skip_corrupt_samples=hypersim_skip_corrupt_samples,
    )

    phase1_iters = int(config['train'].get('phase1_iterations', 20000))
    phase2_iters = int(config['train'].get('phase2_iterations', 50000))
    max_iters = phase2_iters
    val_interval_iters = int(config['train'].get('val_interval_iters', 2000))
    ckpt_interval_iters = int(config['train'].get('checkpoint_interval_iters', 5000))
    val_example_images = int(config['train'].get('val_example_images', 2))
    val_example_indices = config['train'].get('val_example_indices', [])

    start_step = 0
    resume_path = _resolve_resume_path(args.resume, args.auto_resume, ckpt_dir)
    if resume_path:
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        start_step, _ = load_checkpoint(model, optimizer, resume_path, map_location=device)
        start_step += 1

    print(f"Curriculum: phase1 [0,{phase1_iters}), phase2 [{phase1_iters},{phase2_iters})")
    print(f"Start step: {start_step}, max step: {max_iters}")

    running = {'loss_a': 0.0, 'loss_b': 0.0, 'loss_c': 0.0, 'loss_d': 0.0, 'loss_total': 0.0}
    last_phase = None

    if train_mid is None:
        print(
            f"Train scenes: hypersim={hypersim_effective} "
            f"(base={hypersim_total}, midintrinsic disabled)"
        )
    else:
        print(
            f"Train scenes: hypersim={hypersim_effective} (base={hypersim_total}), "
            f"midintrinsic={mid_effective} (base={mid_total})"
        )
    # Explicitly notify dataset-size cap used for this run (0 means full train split).
    if hypersim_train_max_images > 0:
        print(
            f"Hypersim cap active: hypersim_train_max_images={hypersim_train_max_images} "
            f"-> effective {hypersim_effective}/{hypersim_total} train images"
        )
    else:
        print(
            f"Hypersim cap inactive: hypersim_train_max_images=0 "
            f"-> using full train split ({hypersim_effective}/{hypersim_total})"
        )
    print(f"Val frames (hypersim): {len(val_loader.dataset)}")

    train_pbar = tqdm(
        range(start_step, max_iters),
        desc='Training',
        total=max_iters - start_step,
        dynamic_ncols=True,
    )

    for step in train_pbar:
        phase, weights = _phase_schedule(config['train'], step)
        if phase != last_phase:
            mixed_loader.set_weights(weights)
            print(f"[{step}] switch -> {phase}, weights={weights}")
            last_phase = phase

        batch, dataset_name = mixed_loader.next_batch()  # homogeneous per-batch dataset
        losses = train_one_step(model, batch, criterion, optimizer, device)

        train_pbar.set_postfix({
            'phase': phase,
            'ds': dataset_name,
            'hyp_cap': (str(hypersim_train_max_images) if hypersim_train_max_images > 0 else 'all'),
            'loss': f"{losses['loss_total'].item():.4f}",
        })

        for k in running:
            running[k] += losses[k].item()

        if step % int(config['train']['log_interval']) == 0:
            _log_ordered_scalars(writer, losses, step)

        if (step + 1) % val_interval_iters == 0:
            torch.cuda.empty_cache()
            vloss = validate(
                model,
                val_loader,
                criterion,
                device,
                step + 1,
                writer,
                val_example_images=val_example_images,
                val_example_indices=val_example_indices,
            )
            print(f"[{step+1}] val: " + ", ".join([f"{k}={v:.4f}" for k, v in vloss.items()]))

        if (step + 1) % ckpt_interval_iters == 0:
            avg = {k: running[k] / ckpt_interval_iters for k in running}
            for k in running:
                running[k] = 0.0
            ckpt_path = os.path.join(ckpt_dir, f'checkpoint_iter_{step+1}.pth')
            save_checkpoint(model, optimizer, avg, config, ckpt_path, global_step=step)
            latest_path = os.path.join(ckpt_dir, 'checkpoint_latest.pth')
            save_checkpoint(model, optimizer, avg, config, latest_path, global_step=step)
            print(f"[{step+1}] train(avg): " + ", ".join([f"{k}={v:.4f}" for k, v in avg.items()]))

    print('Training completed')
    writer.close()


if __name__ == '__main__':
    main()

