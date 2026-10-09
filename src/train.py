"""Train the RGB-shading model with Cross-Illumination Albedo Invariance (CIAI).

    bash scripts/train.sh --config ciai --cuda 0             # the reported model (ablation row 4)
    python src/train.py --config ciai --seed 43              # another seed of the same config

--config takes a name under src/configs (ciai, stage_a, ablation_*) or a path to a .yaml
file; checkpoints and logs go to checkpoints/<name>/ and logs/<name>/.

Training follows the curriculum in the config: phase 1 Hypersim, phase 2 + InteriorVerse
(supervised only), phase 3 adds MID cross-illumination pairs. Each step runs the model on the primary frame for the
single-image losses (losses/rgb_shading_loss.py) and, for paired rows, a second time on the pair
frame for the CIAI terms (losses/ciai.py). Validation runs on the Hypersim val split.
"""

import argparse
import math
import os
import random
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models import RGBShadingNet, model_arch
from losses.rgb_shading_loss import RGBShadingLoss
from metrics import (_compute_lmse, _masked_scale_invariant_rmse,
                     _compute_shading_ssim, _compute_ssim_bounded)
from data.hypersim_dataset import get_hypersim_loader


TB_TAGS = {
    'loss_total': '1. Losses/Total_all',
    'loss_a': '1. Losses/A_total',
    'loss_c_l1': '1. Losses/A_MSE',
    'loss_c_msg': '1. Losses/A_MSG',
    'loss_c_dssim': '1. Losses/A_DSSIM',
    'loss_s': '1. Losses/S_total',
    'loss_shading_mse': '1. Losses/S_MSE',
    'loss_shading_msg': '1. Losses/S_MSG',
    'loss_r': '1. Losses/R_total',
    'loss_residual_mse': '1. Losses/R_MSE',
    'loss_residual_msg': '1. Losses/R_MSG',
    'loss_recon': '1. Losses/Recon_total',
    'loss_recon_l1': '1. Losses/Recon_L1',
    'loss_recon_msg': '1. Losses/Recon_MSG',
    # CIAI pair terms: absent from a step whose batch has no paired rows. A missing or
    # flat-zero curve is the first sign that pairs are not reaching the loss.
    'loss_alb_invariance': '1. Losses/CIAI_L_inv',
    'loss_explain': '1. Losses/CIAI_L_explain',
    # Unweighted chroma residual of the shading, measured without gradient: how much of the
    # colour change between the two frames the RGB shading does not explain.
    'diag_chr_explain': '1. Losses/CIAI_chr_residual_diag',
}


def _log_ordered_scalars(writer, values, global_step, tag_prefix=None):
    """Write only the known tags, in a fixed order."""
    for key in TB_TAGS:
        if key not in values:
            continue
        val = values[key]
        if isinstance(val, torch.Tensor):
            val = val.item()
        tag = TB_TAGS[key]
        if tag_prefix:
            tag = f"{tag_prefix}/{tag}"
        writer.add_scalar(tag, float(val), global_step)


def scale_match(A_raw, A_pred, valid, seg=None, stable_classes=[1, 2, 22]):
    """Least-squares per-image scalar c for c*A_raw ~= A_pred over valid pixels."""
    v = valid.expand_as(A_raw).float()

    if seg is not None and stable_classes:
        if seg.dim() == 4: seg = seg[:, 0]
        target = torch.tensor(stable_classes, device=seg.device)
        struct_mask = torch.isin(seg, target).unsqueeze(1).expand_as(A_raw).float()
        # Only use structural pixels if enough exist (>5% of image)
        struct_ratio = struct_mask.sum() / (v.sum() + 1e-7)
        if struct_ratio > 0.05:
            v = v * struct_mask

    a = (A_raw * v).reshape(A_raw.shape[0], -1)
    b = (A_pred * v).reshape(A_pred.shape[0], -1)
    c = (a * b).sum(dim=1) / ((a * a).sum(dim=1) + 1e-6)
    # Prevent degenerate target scaling that can zero-out the albedo supervision.
    c = torch.clamp_min(c, 0.05)
    return c.reshape(-1, 1, 1, 1)


def compute_targets(predictions, batch):
    """Scale-matched albedo target and the shading/residual targets it implies.

    Albedo is only defined up to a global scale, so the ground truth is first scaled to
    the prediction. Where a dataset has diffuse-shading ground truth (M_diffuse = 1) it
    is used, rescaled by 1/c so that A* x S* still equals the image; elsewhere the
    shading target is I / A*.
    """
    device = predictions['a_d'].device
    rgb = batch['rgb'].to(device)
    albedo_target = batch['albedo_raw'].to(device)

    loss_mask = batch.get('loss_mask', None)
    if loss_mask is None:
        raise KeyError("batch is missing required 'loss_mask'")
    loss_mask = loss_mask.to(device)
    seg = batch.get('seg', None)
    if seg is not None:
        seg = seg.to(device)

    c = scale_match(albedo_target, predictions['a_d'].detach(), loss_mask, seg)
    A_star = c * albedo_target
    A_star = torch.nan_to_num(A_star, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)

    S_c_star = rgb / A_star.clamp_min(1e-4)
    S_c_star = torch.nan_to_num(S_c_star, nan=0.0, posinf=60000.0, neginf=0.0).clamp_min(0.0)

    S_d_star = S_c_star
    illum_raw = batch.get('illum_raw', None)
    m_diffuse = batch.get('M_diffuse', None)
    if illum_raw is not None and m_diffuse is not None:
        route = m_diffuse.reshape(-1, 1, 1, 1).to(device=rgb.device, dtype=rgb.dtype)
        illum = torch.nan_to_num(illum_raw.to(device), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        illum_scaled = illum / c.clamp_min(1e-4)
        S_d_star = route * illum_scaled + (1.0 - route) * S_c_star

    R_star = rgb - (A_star * S_d_star)
    R_star = torch.nan_to_num(R_star, nan=0.0, posinf=0.0, neginf=0.0)

    return {
        'A_d_star': A_star,
        'S_d_star': S_d_star,
        'pi_star': 1.0 / (S_d_star + 1.0),
        'R_star': R_star,
        'loss_mask': loss_mask,
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Train the RGB-shading model with CIAI')
    parser.add_argument('--config', type=str, default='ciai',
                        help='config name under src/configs (e.g. ciai) or a path to a .yaml file')
    parser.add_argument('--run-name', type=str, default=None,
                        help='checkpoint/log directory name (default: the config name)')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint path or "latest"')
    parser.add_argument('--auto-resume', action='store_true', help='Resume from the latest checkpoint in this run directory')
    parser.add_argument('--reset-lr', action='store_true', help='Override checkpoint LR with value from config')
    parser.add_argument('--skip-optimizer', action='store_true', help='Resume from checkpoint but skip loading optimizer state')
    parser.add_argument('--seed', type=int, default=None,
                        help='Override train.seed. Replicates of one config differ only '
                             'in this, so N seeds need no new config files.')
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


def _resolve_config_path(ref):
    """A config name under src/configs, or a path to a .yaml file."""
    ref = str(ref)
    if ref.endswith(('.yaml', '.yml')):
        return Path(ref)
    return SRC_DIR / 'configs' / f'{ref}.yaml'


def _load_config_with_parents(config_path, seen=None):
    """Load a config override and recursively merge its `extends` chain.

    Layering for nested ablations is:
        parent-of-parent <- parent <- child
    The top-level load_config() merges this result over base.yaml.
    """
    path = Path(config_path)
    seen = set() if seen is None else seen
    resolved = path.resolve()
    if resolved in seen:
        chain = ' -> '.join(str(p) for p in seen) + f' -> {resolved}'
        raise RuntimeError(f'Config extends cycle detected: {chain}')
    seen.add(resolved)

    with open(path, 'r') as f:
        override = yaml.safe_load(f) or {}
    parent = override.pop('extends', None)
    if parent is None:
        return override, [path.name]

    parent_path = _resolve_config_path(parent)
    if not parent_path.exists():
        raise FileNotFoundError(f"Parent config not found for extends={parent!r}: {parent_path}")
    parent_cfg, chain = _load_config_with_parents(parent_path, seen)
    return _deep_merge(parent_cfg, override), chain + [path.name]


def load_config(config=None):
    """base.yaml merged with a config (a name under src/configs or a path) and its parents."""
    base_path = SRC_DIR / 'configs' / 'base.yaml'
    with open(base_path, 'r') as f:
        merged = yaml.safe_load(f)
    if config is None:
        print('Config: base.yaml only')
        return merged
    config_path = _resolve_config_path(config)
    if not config_path.exists():
        names = sorted(p.stem for p in (SRC_DIR / 'configs').glob('*.yaml'))
        raise FileNotFoundError(f'config not found: {config_path} (available: {names})')
    override, chain = _load_config_with_parents(config_path)
    print('Config: base.yaml <- ' + ' <- '.join(chain))
    return _deep_merge(merged, override)


def save_checkpoint(model, optimizer, losses, config, filename, global_step):
    ckpt = {
        'global_step': int(global_step),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
        'config': config,
        # Promoted out of config so a result can be traced to its replicate without
        # loading and walking the whole config dict. None means the run was unseeded.
        'seed': config.get('train', {}).get('seed'),
    }
    tmp_filename = f"{filename}.tmp.{os.getpid()}"
    try:
        torch.save(ckpt, tmp_filename)
        os.replace(tmp_filename, filename)
    except Exception:
        try:
            if os.path.exists(tmp_filename):
                os.remove(tmp_filename)
        except OSError:
            pass
        raise
    print(f"Saved checkpoint to {filename}")


def load_checkpoint(model, optimizer, checkpoint_path, map_location=None, skip_optimizer=False):
    ckpt = torch.load(checkpoint_path, map_location=map_location)
    # Shape-filtered non-strict load: plain strict=False only tolerates missing/extra keys,
    # not shape mismatches on shared keys (e.g. a head's arch grows when a skip path is
    # toggled on). Drop mismatched-shape keys so they cold-start instead of erroring.
    own = model.state_dict()
    model_state = ckpt['model_state_dict']
    filtered = {k: v for k, v in model_state.items() if k in own and v.shape == own[k].shape}
    if len(filtered) < len(own):
        dropped = sorted(set(own) - set(filtered))
        print(f"[warn] loaded {len(filtered)}/{len(own)} params from checkpoint "
              f"(rest cold-start, shape/key mismatch): {dropped}")
    model.load_state_dict(filtered, strict=False)
    
    if not skip_optimizer:
        try:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        except Exception as exc:
            print(
                "[warn] optimizer state not loaded (param-group mismatch). "
                f"Continuing with freshly initialized optimizer: {exc}"
            )
    else:
        print("[info] Skipping optimizer state loading as requested.")

    global_step = int(ckpt.get('global_step', 0))
    losses = ckpt.get('losses', {})
    print(f"Loaded checkpoint at global_step={global_step}: {checkpoint_path}")
    return global_step, losses


def _extract_iter_from_name(path):
    m = re.search(r'checkpoint_iter_(\d+)\.pth$', os.path.basename(path))
    return int(m.group(1)) if m else -1


def _is_readable_checkpoint(path):
    """Cheaply reject half-written PyTorch zip checkpoints before torch.load()."""
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0 and zipfile.is_zipfile(path)
    except OSError:
        return False


def _find_latest_checkpoint(ckpt_dir):
    if not os.path.isdir(ckpt_dir):
        return None
    iter_files = [
        os.path.join(ckpt_dir, f)
        for f in os.listdir(ckpt_dir)
        if f.startswith('checkpoint_iter_') and f.endswith('.pth')
    ]
    iter_files.sort(key=_extract_iter_from_name, reverse=True)
    fallback = os.path.join(ckpt_dir, 'checkpoint_latest.pth')

    candidates = []
    seen = set()
    for path in iter_files + [fallback]:
        if path in seen:
            continue
        seen.add(path)
        candidates.append(path)

    for path in candidates:
        if _is_readable_checkpoint(path):
            return path
        if os.path.exists(path):
            print(f"[warn] Skipping unreadable checkpoint during auto-resume: {path}")
    return None


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


def _backward(loss, scaler, grad_accum_steps):
    """Scale by 1/accum and backward (AMP-aware). Caller does optimizer.step on the accum boundary."""
    scaled = loss / grad_accum_steps
    if scaler is not None:
        scaler.scale(scaled).backward()
    else:
        scaled.backward()


def train_one_step(model, batch, criterion, device, global_step, ssi_warmup_iters,
                   scaler=None, grad_accum_steps=1):
    """One micro-batch: single-image losses on the primary frame, then the CIAI pair terms
    on paired rows using a second forward on rgb2. Backward is done here; the caller steps
    the optimizer on the accumulation boundary. Returns a detached loss dict."""
    model.train()

    rgb = batch['rgb'].to(device, non_blocking=True)
    loss_mask = batch.get('loss_mask', None)
    if loss_mask is None:
        raise KeyError("batch is missing required 'loss_mask'")
    loss_mask = loss_mask.to(device, non_blocking=True)
    m_diffuse = batch.get('M_diffuse', torch.zeros(rgb.shape[0])).float().to(device, non_blocking=True)
    m_residual = batch.get('m_residual', torch.ones(rgb.shape[0])).float().to(device, non_blocking=True)

    def _forward(x):
        with autocast(device_type='cuda', dtype=torch.float16):
            preds = model(x)
            return {k: (v.float() if isinstance(v, torch.Tensor) else v) for k, v in preds.items()}

    # -- 1. Primary frame: single-image losses --------------------------------------
    predictions = _forward(rgb)
    targets = compute_targets(predictions, batch)
    losses = criterion(
        predictions=predictions, targets=targets, loss_mask=loss_mask,
        m_diffuse=m_diffuse, m_residual=m_residual, rgb=rgb,
        use_ssi=(global_step >= ssi_warmup_iters),
    )

    # -- 2. CIAI: second forward on the pair frame, paired rows only -----------------
    lam_inv = criterion.lambda_alb_invariance
    lam_explain = criterion.lambda_explain
    rgb2 = batch.get('rgb2', None)
    m_invariant = batch.get('m_invariant', None)
    if (lam_inv > 0 or lam_explain > 0)             and rgb2 is not None and m_invariant is not None:
        m_invariant = m_invariant.float().to(device, non_blocking=True)
        if m_invariant.sum() > 0:
            rgb2 = rgb2.to(device, non_blocking=True)
            pair_valid = batch.get('pair_valid', loss_mask).float().to(device, non_blocking=True)
            pred2 = _forward(rgb2)
            cr_mask = loss_mask.float() * m_invariant.view(-1, 1, 1, 1) * pair_valid
            s1, s2 = predictions['shading_linear'], pred2['shading_linear'].float()
            if lam_inv > 0:
                losses['loss_alb_invariance'] = lam_inv * criterion.albedo_invariance(
                    predictions['a_d'], pred2['a_d'].float(), cr_mask)
                losses['loss_total'] = losses['loss_total'] + losses['loss_alb_invariance']
            if lam_explain > 0:
                losses['loss_explain'] = lam_explain * criterion.luminance_explain(
                    rgb, rgb2, s1, s2, cr_mask)
                losses['loss_total'] = losses['loss_total'] + losses['loss_explain']
            # Diagnostic only: the unexplained chroma residual, without gradient.
            with torch.no_grad():
                losses['diag_chr_explain'] = criterion.chroma_explain(
                    rgb, rgb2, s1.detach(), s2.detach(), cr_mask)

    _backward(losses['loss_total'], scaler, grad_accum_steps)
    return {k: (v.detach() if torch.is_tensor(v) else v) for k, v in losses.items()}


def _get_tonemap_scale(img, percentile=90.0, target_brightness=0.8, valid_mask=None):
    """Compute scale so the given percentile maps to target_brightness."""
    if img.ndim == 4:
        img = img[0]
    if valid_mask is not None and valid_mask.ndim == 4:
        valid_mask = valid_mask[0]
        
    img = torch.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    
    if valid_mask is not None and valid_mask.any():
        pixels = img[:, valid_mask.squeeze(0).bool()].reshape(-1)
        if pixels.numel() > 10:
            q = torch.quantile(pixels, percentile / 100.0)
            return torch.clamp_min(q / target_brightness, 1e-6)
    
    q = torch.quantile(img.reshape(-1), percentile / 100.0)
    return torch.clamp_min(q / target_brightness, 1e-6)


def _vis_tonemap(img, percentile=90.0, target_brightness=0.8, eps=1e-6, scale=None, valid_mask=None):
    """Inference-style tonemap to [0,1] per sample with NaN/Inf guards.
    Use scale=1.0 for already-tonemapped RGB inputs; keep scale=None for
    diagnostic tensors (reconstruction/derived albedo) that need auto exposure.
    If valid_mask is provided, computes tonemap scale only over valid pixels."""
    if img.ndim == 4:
        img = img[0]
    if valid_mask is not None and valid_mask.ndim == 4:
        valid_mask = valid_mask[0]
        
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
        
    img = torch.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    
    if scale is None:
        scale = _get_tonemap_scale(img, percentile=percentile, target_brightness=target_brightness, valid_mask=valid_mask)
            
    scale = torch.as_tensor(scale, device=img.device, dtype=img.dtype).clamp_min(eps)
    return torch.clamp(img / scale, 0.0, 1.0)


def _log_val_examples(writer, global_step, rgb, predictions, targets, max_items=2,
                      sample_index=None, example_root='3. Examples'):
    """Log input / GT / prediction strips (albedo, pi-shading, residual, reconstruction)."""
    example_tag = f'{example_root}/sample_{sample_index}' if sample_index is not None else example_root

    def _gamma(x):
        return torch.pow(torch.clamp(x, min=0.0, max=1.0), 1.0 / 3.0)

    def _tile(x, hw=None):
        x = x[0] if x.ndim == 4 else x
        x = x.repeat(3, 1, 1) if x.shape[0] == 1 else x[:3]
        x = x.detach().to('cpu', torch.float32).clamp(0.0, 1.0)
        x = F.interpolate(x.unsqueeze(0), scale_factor=0.5, mode='bilinear', align_corners=False)[0]
        return x

    for i in range(min(int(rgb.shape[0]), int(max_items))):
        v_mask = targets['loss_mask'][i:i+1]
        a_gt = targets['A_d_star'][i:i+1]
        s_gt_pi = targets['pi_star'][i:i+1].clamp(0.0, 1.0)
        s_gt_linear = 1.0 / (s_gt_pi + 1e-6) - 1.0
        r_gt = targets['R_star'][i:i+1]
        recon_gt = a_gt * s_gt_linear + r_gt

        a_pred = predictions['a_d'][i:i+1]
        s_pred_pi = predictions['shading'][i:i+1].clamp(0.0, 1.0)
        r_pred = predictions['residual'][i:i+1]
        recon_pred = predictions['rgb_reconstructed'][i:i+1]

        scale_a = torch.max(_get_tonemap_scale(a_pred, valid_mask=v_mask),
                            _get_tonemap_scale(a_gt, valid_mask=v_mask))
        scale_r = torch.max(_get_tonemap_scale(r_pred.abs(), valid_mask=v_mask),
                            _get_tonemap_scale(r_gt.abs(), valid_mask=v_mask))
        scale_c = torch.max(_get_tonemap_scale(recon_pred, valid_mask=v_mask),
                            _get_tonemap_scale(recon_gt, valid_mask=v_mask))

        gt_row = [
            _gamma(_vis_tonemap(rgb[i:i+1], scale=None, valid_mask=v_mask)),
            _gamma(_vis_tonemap(a_gt, scale=scale_a)),
            _gamma(s_gt_pi),
            _gamma(_vis_tonemap(r_gt.abs(), scale=scale_r)),
            _gamma(_vis_tonemap(recon_gt, scale=scale_c)),
        ]
        pred_row = [
            torch.ones_like(gt_row[0]),
            _gamma(_vis_tonemap(a_pred, scale=scale_a)),
            _gamma(s_pred_pi),
            _gamma(_vis_tonemap(r_pred.abs(), scale=scale_r)),
            _gamma(_vis_tonemap(recon_pred, scale=scale_c)),
        ]
        strip = torch.cat([torch.cat([_tile(t) for t in gt_row], dim=2),
                           torch.cat([_tile(t) for t in pred_row], dim=2)], dim=1)
        if i == 0:
            writer.add_text(f'{example_tag}/layout',
                            'top: input | albedo GT | shading GT (pi) | residual GT | recon GT; '
                            'bottom: predictions in the same order', global_step)
        tag = f'{example_tag}/index_{sample_index}' if sample_index is not None else f'{example_tag}/sample_{i}'
        writer.add_image(tag, strip, global_step)


def validate(model, dataloader, criterion, device, global_step, writer,
             val_example_images=2, val_example_indices=None, max_val_batches=None,
             compute_val_losses=True, example_root='3. Examples', config=None):
    """Hypersim val split: losses plus albedo / shading / residual metrics against
    scale-matched targets, and example strips."""
    model.eval()
    total_loss = {}
    total_metric = {k: 0.0 for k in ('a_d_lmse', 'a_d_rmse', 'a_d_ssim',
                                      's_d_lmse', 's_d_rmse', 's_d_ssim', 'r_rmse')}
    n_samples = 0
    val_example_indices = list(val_example_indices) if val_example_indices else []
    use_indices = len(val_example_indices) > 0
    indices_to_collect = set(int(idx) for idx in val_example_indices)
    collected = {}
    processed_batches = 0

    with torch.no_grad():
        n_metric = len(dataloader) if max_val_batches is None else min(max_val_batches, len(dataloader))
        progress_total = len(dataloader) if use_indices else n_metric
        for batch_idx, batch in enumerate(tqdm(dataloader, desc='Validation', total=progress_total)):
            budget_reached = max_val_batches is not None and batch_idx >= max_val_batches
            need_more = use_indices and len(collected) < len(indices_to_collect)
            if budget_reached and not need_more:
                break

            rgb = batch['rgb'].to(device, non_blocking=True)
            loss_mask = batch['loss_mask'].to(device, non_blocking=True)
            m_residual = batch.get('m_residual', torch.ones(rgb.shape[0])).float().to(device, non_blocking=True)
            m_diffuse = batch.get('M_diffuse', torch.zeros(rgb.shape[0])).float().to(device, non_blocking=True)
            predictions = {k: v.float() if isinstance(v, torch.Tensor) else v
                           for k, v in model(rgb).items()}
            targets = compute_targets(predictions, batch)

            batch_size = rgb.shape[0]
            sample_indices = batch.get('sample_idx', None)
            if use_indices:
                for i in range(batch_size):
                    gidx = (int(sample_indices[i].item()) if sample_indices is not None
                            else int(batch_idx * dataloader.batch_size + i))
                    if gidx in indices_to_collect:
                        collected[gidx] = (rgb[i:i+1], {k: v[i:i+1] for k, v in predictions.items()
                                                        if torch.is_tensor(v)},
                                           {k: v[i:i+1] for k, v in targets.items()})
            elif batch_idx == 0:
                _log_val_examples(writer, global_step, rgb, predictions, targets,
                                  max_items=val_example_images, example_root=example_root)

            if budget_reached:
                continue
            processed_batches += 1

            if compute_val_losses:
                losses = criterion(predictions=predictions, targets=targets, loss_mask=loss_mask,
                                   m_diffuse=m_diffuse, m_residual=m_residual, rgb=rgb, use_ssi=True)
                for k, v in losses.items():
                    total_loss[k] = total_loss.get(k, 0.0) + v.item()

            a_gt = targets['A_d_star']
            total_metric['a_d_lmse'] += _compute_lmse(predictions['a_d'], a_gt, loss_mask).item() * batch_size
            total_metric['a_d_rmse'] += _masked_scale_invariant_rmse(predictions['a_d'], a_gt, loss_mask).item() * batch_size
            total_metric['a_d_ssim'] += _compute_ssim_bounded(predictions['a_d'], a_gt, loss_mask) * batch_size
            s_d_gt, s_d_pred = targets['S_d_star'], predictions['shading']
            total_metric['s_d_lmse'] += _compute_lmse(s_d_pred, s_d_gt, loss_mask).item() * batch_size
            total_metric['s_d_rmse'] += _masked_scale_invariant_rmse(s_d_pred, s_d_gt, loss_mask).item() * batch_size
            total_metric['s_d_ssim'] += _compute_shading_ssim(s_d_pred, s_d_gt, loss_mask) * batch_size
            total_metric['r_rmse'] += _masked_scale_invariant_rmse(predictions['residual'], targets['R_star'], loss_mask).item() * batch_size
            n_samples += batch_size

    for gidx in sorted(collected):
        rgb_i, pred_i, tgt_i = collected[gidx]
        _log_val_examples(writer, global_step, rgb_i, pred_i, tgt_i, max_items=1,
                          sample_index=gidx, example_root=example_root)

    if compute_val_losses:
        for k in total_loss:
            total_loss[k] /= max(processed_batches, 1)
    for k in total_metric:
        total_metric[k] /= max(n_samples, 1)

    val_out = dict(total_loss) if compute_val_losses else {}
    val_out.update(total_metric)
    _log_ordered_scalars(writer, val_out, global_step, tag_prefix='2. Val')
    for k, v in total_metric.items():
        writer.add_scalar(f'2. Val/2. Metrics/{k}', float(v), global_step)

    return val_out


def build_model(config):
    """Build the RGB-shading model from config['model'] (+ train.input_size)."""
    arch = model_arch(config['model'])
    if arch != 'rgb_shading':
        raise ValueError(f'train.py builds the RGB-shading model only; got model.arch={arch!r}. '
                         'Use src/train_trifactor.py for the trifactor model.')
    m = config['model']
    return RGBShadingNet({
        'dpt_feat_ch': int(m.get('dpt_feat_ch', 256)),
        'dpt_fusion_ch': int(m.get('dpt_fusion_ch', 128)),
        'dpt_out_ch': int(m.get('dpt_out_ch', 128)),
        'detail_ch': int(m.get('detail_ch', 48)),
        'head_mid': int(m.get('head_mid', 64)),
        'pi_floor': float(m.get('pi_floor', 5e-3)),
        'albedo_chroma_skip': m.get('albedo_chroma_skip', False),
        'shading_lum_skip': m.get('shading_lum_skip', False),
        'albedo_rgb_skip': m.get('albedo_rgb_skip', False),
        'dino_variant': m.get('dino_variant', 'large'),
        'dino_pretrained': m.get('dino_pretrained', True),
    })


def build_optimizer(model, train_cfg, model_cfg):
    """Build Adam optimizer with optional lower LR for image encoder."""
    base_lr = float(train_cfg['lr'])
    multiplier = float(model_cfg.get('backbone_lr_multiplier', 1.0))

    if multiplier <= 0.0:
        raise ValueError(f"backbone_lr_multiplier must be > 0, got {multiplier}")

    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith('image_encoder.'):
            backbone_params.append(param)
        else:
            other_params.append(param)

    if not backbone_params or not other_params or abs(multiplier - 1.0) < 1e-12:
        params = (p for p in model.parameters() if p.requires_grad)
        return torch.optim.Adam(params, lr=base_lr)

    return torch.optim.Adam(
        [
            {'params': backbone_params, 'lr': base_lr * multiplier},
            {'params': other_params, 'lr': base_lr},
        ]
    )


def main():
    args = parse_args()
    config = load_config(args.config)
    print(yaml.dump(config, default_flow_style=False))

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Seeding fixes data order, augmentation and initialisation; --seed overrides the
    # config so replicates need no new config files. It does not make cuDNN kernels
    # bitwise reproducible unless train.deterministic is set (at a throughput cost).
    seed = args.seed if args.seed is not None else config['train'].get('seed', None)
    if seed is not None:
        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    config['train']['seed'] = seed
    deterministic = bool(config['train'].get('deterministic', False))
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True
    print(f"Seed: {seed if seed is not None else 'UNSEEDED (nondeterministic run)'} | "
          f"deterministic kernels: {deterministic}")

    run_name = args.run_name or Path(str(args.config)).stem
    ckpt_dir = os.path.join(config['paths']['checkpoint_dir'], run_name)
    log_dir = os.path.join(config['paths']['log_dir'], run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)
    model = build_model(config).to(device)
    criterion = RGBShadingLoss(config['loss']).to(device)
    optimizer = build_optimizer(model, config['train'], config['model'])

    data_cfg = config['data']
    hypersim_root = data_cfg['hypersim_root']
    if not os.path.isabs(hypersim_root):
        hypersim_root = str(ROOT_DIR / hypersim_root)
    strict_split = bool(data_cfg.get('hypersim_strict_split', True))
    from src.data.mixed_dataset import get_mixed_loader

    def _build_train_loader(weights_key):
        return get_mixed_loader(
            data_roots={
                'hypersim': hypersim_root,
                'midintrinsic': data_cfg.get('midintrinsic_root', '../datasets/MIDIntrinsics'),
                'interiorverse': data_cfg.get('interiorverse_root', '../datasets/InteriorVerse'),
            },
            batch_size=int(config['train']['batch_size']),
            split='train',
            num_workers=int(config['train'].get('num_workers', 4)),
            input_size=int(config['train']['input_size']),
            cache_max_items=int(data_cfg.get('cache_max_items', 512)),
            mix_weights=config['train'].get(weights_key, {'hypersim': 1.0}),
            seed=seed,
            strict_split=strict_split,
            use_mid_paired=bool(data_cfg.get('use_mid_paired', False)),
            mid_raw_color_pair=bool(data_cfg.get('mid_raw_color_pair', False)),
        )

    def infinite_loader(dl):
        while True:
            for b in dl:
                yield b

    val_loader = get_hypersim_loader(
        root_dir=hypersim_root,
        batch_size=int(config['train']['batch_size']),
        split='val',
        num_workers=max(1, int(data_cfg.get('val_num_workers', 2))),
        input_size=int(config['train']['input_size']),
        cache_max_items=max(0, int(data_cfg.get('val_cache_max_items', 64))),
        crop_mode_train=str(data_cfg.get('crop_mode_train', 'random')),
        crop_mode_val=str(data_cfg.get('crop_mode_val', 'center')),
        split_file=data_cfg.get('hypersim_split_file', 'hypersim_split.json'),
        split_seed=int(data_cfg.get('hypersim_split_seed', 42)),
        split_ratio=float(data_cfg.get('hypersim_split_ratio', 0.9)),
        strict_split=strict_split,
        max_hdf5_retries=int(data_cfg.get('hypersim_max_hdf5_retries', 1)),
        skip_corrupt_samples=bool(data_cfg.get('hypersim_skip_corrupt_samples', True)),
        load_geometry=False,
    )

    train_cfg = config['train']
    max_iters = int(train_cfg.get('extend_iterations', 25000))
    grad_accum_steps = max(1, int(train_cfg.get('grad_accum_steps', 1)))
    ssi_warmup = int(config['loss'].get('ssi_warmup_iters', 3000))
    grad_clip_max_norm = float(train_cfg.get('grad_clip_max_norm', 1.0))
    val_interval_iters = int(train_cfg.get('val_interval_iters', 2000))
    ckpt_interval_iters = int(train_cfg.get('checkpoint_interval_iters', 5000))
    max_val_batches = train_cfg.get('max_val_batches', None)
    max_val_batches = int(max_val_batches) if max_val_batches is not None else None

    start_step = 0
    resume_path = _resolve_resume_path(args.resume, args.auto_resume, ckpt_dir)
    if resume_path:
        skip_opt = args.skip_optimizer or train_cfg.get('skip_optimizer', False)
        start_step, _ = load_checkpoint(model, optimizer, resume_path, map_location=device, skip_optimizer=skip_opt)
        start_step += 1
        if args.reset_lr or train_cfg.get('reset_lr', False):
            base_lr = float(train_cfg['lr'])
            multiplier = float(config['model'].get('backbone_lr_multiplier', 1.0))
            if len(optimizer.param_groups) == 2:
                optimizer.param_groups[0]['lr'] = base_lr * multiplier
                optimizer.param_groups[1]['lr'] = base_lr
            else:
                for pg in optimizer.param_groups:
                    pg['lr'] = base_lr
            print(f"LR reset to {base_lr} from config")

    scheduler = None
    if bool(train_cfg.get('use_cosine_lr', True)):
        total_opt_steps = max(1, math.ceil(max_iters / grad_accum_steps))
        completed_opt_steps = max(0, start_step // grad_accum_steps)
        for pg in optimizer.param_groups:
            pg.setdefault('initial_lr', pg['lr'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_opt_steps, eta_min=float(train_cfg.get('lr_eta_min', 1.0e-7)),
            last_epoch=completed_opt_steps - 1)

    phase1_iters = int(train_cfg.get('phase1_iterations', -1))
    phase2_iters = int(train_cfg.get('phase2_iterations', -1))

    def _weights_key_for_step(step):
        if phase2_iters >= 0 and step >= phase2_iters and 'sampling_weights_phase3' in train_cfg:
            return 'sampling_weights_phase3'
        if phase1_iters >= 0 and step >= phase1_iters and 'sampling_weights_phase2' in train_cfg:
            return 'sampling_weights_phase2'
        return 'sampling_weights_phase1'

    active_weights_key = _weights_key_for_step(start_step)
    print(f"[data] train mix at step {start_step}: {active_weights_key} = "
          f"{train_cfg.get(active_weights_key, {})}")
    train_iter = iter(infinite_loader(_build_train_loader(active_weights_key)))
    print(f"Start step: {start_step}, max step: {max_iters}")

    running = {}
    scaler = GradScaler('cuda')
    keep_ckpts = max(1, int(train_cfg.get('keep_checkpoints', 2)))
    pbar = tqdm(range(start_step, max_iters), desc='Training', total=max_iters - start_step,
                dynamic_ncols=True)
    for step in pbar:
        desired = _weights_key_for_step(step)
        if desired != active_weights_key:
            active_weights_key = desired
            print(f"--- switching train mix at step {step}: {active_weights_key} = "
                  f"{train_cfg.get(active_weights_key, {})} ---")
            train_iter = iter(infinite_loader(_build_train_loader(active_weights_key)))

        losses = train_one_step(model, next(train_iter), criterion, device, step, ssi_warmup,
                                scaler=scaler, grad_accum_steps=grad_accum_steps)

        if (step + 1) % grad_accum_steps == 0 or step == max_iters - 1:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        pbar.set_postfix({'loss': f"{losses['loss_total'].item():.4f}"})
        for k, v in losses.items():
            running[k] = running.get(k, 0.0) + v.item()

        if step % int(train_cfg['log_interval']) == 0:
            _log_ordered_scalars(writer, losses, step)
            writer.add_scalar('0. Training/lr', float(optimizer.param_groups[-1]['lr']), step)

        if (step + 1) % ckpt_interval_iters == 0:
            avg = {k: running[k] / ckpt_interval_iters for k in running}
            running = {}
            save_checkpoint(model, optimizer, avg, config,
                            os.path.join(ckpt_dir, f'checkpoint_iter_{step+1}.pth'), global_step=step)
            save_checkpoint(model, optimizer, avg, config,
                            os.path.join(ckpt_dir, 'checkpoint_latest.pth'), global_step=step)
            # Keep the newest `keep_checkpoints` iteration files (each ~1.4 GB).
            iter_ckpts = sorted((os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir)
                                 if f.startswith('checkpoint_iter_') and f.endswith('.pth')),
                                key=_extract_iter_from_name)
            for old in iter_ckpts[:-keep_ckpts]:
                try:
                    os.remove(old)
                except OSError as e:
                    print(f"[warn] could not delete {old}: {e}")

        if (step + 1) % val_interval_iters == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            vloss = validate(
                model, val_loader, criterion, device, step + 1, writer,
                val_example_images=int(train_cfg.get('val_example_images', 2)),
                val_example_indices=train_cfg.get('val_example_indices', []),
                max_val_batches=max_val_batches,
                compute_val_losses=bool(train_cfg.get('compute_val_losses', True)),
                example_root='3. Examples', config=config,
            )
            print(f"[{step+1}] val: " + ", ".join(f"{k}={v:.4f}" for k, v in vloss.items()))

    print('Training completed')
    writer.close()


if __name__ == '__main__':
    main()
