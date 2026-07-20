"""
Training script for Stage 1 intrinsic decomposition.
"""

import argparse
import math
import inspect
import os
import re
import sys
import time
import zipfile
from pathlib import Path

import torch
import torch.nn.functional as F
import kornia
import yaml
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from torch.amp import autocast, GradScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models import IntrinsicDecompositionV17
from models.ccr_utils import compute_ccr
from models.iid_utils import invert, iuv_to_rgb, rgb_to_iuv
from losses.flexible_loss_v17 import V17Loss
from data.hypersim_dataset import HypersimDataset, get_hypersim_loader
from src.data.front3d_dataset import Front3DDataset


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
    # CARI cross-render terms: absent from a step's dict when the batch has no
    # m_invariant rows — a flat-zero/missing curve here is the loudest available
    # signal that MID/front3d pairing is not flowing (see 2026-07-07 wiring bug).
    'loss_alb_invariance': '1. Losses/CARI_L_inv',
    'loss_explain': '1. Losses/CARI_L_explain',
    # IIW ordinal-hinge fine-tune term (v17_26 and analogues): present only when
    # lambda_ordinal_iiw > 0. Was computed and backpropagated but never logged prior to
    # 2026-07-14 — this is the one term whose OWN trajectory needs to be watched during an
    # IIW fine-tune, separately from loss_total, especially across a lambda sweep.
    'loss_ordinal_iiw': '1. Losses/IIW_Ordinal',
}

def _log_ordered_scalars(writer, values, global_step, tag_prefix=None):
    """Write only requested TensorBoard tags in deterministic order."""
    ordered = [
        'loss_total',
        'loss_a', 'loss_c_l1', 'loss_c_msg', 'loss_c_dssim',
        'loss_s', 'loss_shading_mse', 'loss_shading_msg',
        'loss_r', 'loss_residual_mse', 'loss_residual_msg',
        'loss_recon', 'loss_recon_l1', 'loss_recon_msg',
        'loss_alb_invariance', 'loss_explain',
        'loss_ordinal_iiw',
    ]
    for key in ordered:
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
    # Prevent degenerate target scaling that can zero-out Dec-C supervision.
    c = torch.clamp_min(c, 0.05)
    return c.reshape(-1, 1, 1, 1)


def compute_targets(predictions, batch):
    eps = 1e-6
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
    
    # Scale matching parameter C is computed between predictions and truth
    c = scale_match(albedo_target, predictions['a_d'].detach(), loss_mask, seg)

    A_star = c * albedo_target
    A_star = torch.nan_to_num(A_star, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)

    # Colorful shading (S_c) is directly derived from RGB and Albedo: I / A*
    A_safe = A_star.clamp_min(1e-4) 
    S_c_star = rgb / A_safe
    S_c_star = torch.nan_to_num(S_c_star, nan=0.0, posinf=60000.0, neginf=0.0).clamp_min(0.0)

    # Diffuse shading target (S_d_star) routes from direct illumination GT if available
    S_d_star = S_c_star
    illum_raw = batch.get('illum_raw', None)
    m_diffuse = batch.get('M_diffuse', None)
    
    if illum_raw is not None and m_diffuse is not None:
        route = m_diffuse.reshape(-1, 1, 1, 1).to(device=rgb.device, dtype=rgb.dtype)
        illum = torch.nan_to_num(illum_raw.to(device), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        # --- CRITICAL PHYSICS FIX ---
        # Because A_star is scaled by 'c', S_d_star MUST be scaled by '1/c'
        # This guarantees that A_star * S_d_star = I
        illum_scaled = illum / c.clamp_min(1e-4)
        S_d_star = route * illum_scaled + (1.0 - route) * S_c_star

    # Target Residual R_star = I - A*S_d
    R_star = rgb - (A_star * S_d_star)
    R_star = torch.nan_to_num(R_star, nan=0.0, posinf=0.0, neginf=0.0)

    # π of the diffuse shading — fallback shading target for non-d_g (V17/V18) models.
    pi_star = 1.0 / (S_d_star + 1.0)

    # Illuminant chromaticity target = unit-luminance chroma of the COLORFUL shading S_c = I/A*.
    # Derived from the SAME frame the model saw (the primary rgb), so it is consistent with the
    # primary forward's chroma_field: ≈neutral on white Hypersim AND on white-balanced MID (the
    # anchor that prevents chroma green-collapse). NOTE: to teach the *colored* illuminant on MID,
    # supervise the chroma of the RAW (un-WB) rgb2 frame separately (see train_one_step CARI branch);
    # a colored target on this WB primary frame would be physically inconsistent.
    lum_sc = 0.299 * S_c_star[:, 0:1] + 0.587 * S_c_star[:, 1:2] + 0.114 * S_c_star[:, 2:3]
    chroma_field_gt = S_c_star / (lum_sc + eps)

    return {
        'A_d_star': A_star,
        'S_c_star': S_c_star,                # colorful shading I/A* — derive-consistent gray-SSI target
        'S_d_star': S_d_star,
        'pi_star': pi_star,
        'R_star': R_star,
        'chroma_field_gt': chroma_field_gt,  # unit-luminance illuminant colour — chroma-head target
        'loss_mask': loss_mask,
    }



def parse_args():
    parser = argparse.ArgumentParser(description='Train Stage 1')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--version', type=str, default='14')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint path or "latest"')
    parser.add_argument('--auto-resume', action='store_true', help='Resume from latest checkpoint in version checkpoint dir')
    parser.add_argument('--reset-lr', action='store_true', help='Override checkpoint LR with value from config')
    parser.add_argument('--skip-optimizer', action='store_true', help='Resume from checkpoint but skip loading optimizer state')
    parser.add_argument('--start-step', type=int, default=None,
                        help='Override the resume start step (micro-batch units). Use when changing '
                             'batch_size mid-run: the checkpoint global_step is in the OLD bs units, so '
                             'rescale it to the new schedule (e.g. bs2 iter_10000 = 20k samples = step '
                             '5000 at bs4). Keeps LR/phase/sample alignment continuous.')
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
    ref = str(ref)
    path = SRC_DIR / 'configs' / f'v{ref}.yaml'
    if not path.exists():
        path = SRC_DIR / 'configs' / f'{ref}.yaml'
    return path


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


def load_config(config_path=None, version=None):
    base_path = SRC_DIR / 'configs' / 'base.yaml'
    with open(base_path, 'r') as f:
        config = yaml.safe_load(f)

    if config_path is None and version is not None:
        config_path = str(SRC_DIR / 'configs' / f'v{version}.yaml')

    if config_path is not None and os.path.exists(config_path):
        override, chain = _load_config_with_parents(config_path)
        config = _deep_merge(config, override)
        print('Config: base.yaml <- ' + ' <- '.join(chain))
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


def _forward_kwargs(model, m_diffuse, normals, seg, valid_mask, ccr=None):
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
    if 'ccr' in sig and ccr is not None:
        kwargs['ccr'] = ccr
    return kwargs





def _loss_seg(model, seg):
    if 'seg' not in inspect.signature(model.forward).parameters:
        return None
    if seg is not None and seg.dim() == 4 and seg.shape[1] == 1:
        return seg[:, 0]
    return seg


def _backward(loss, scaler, grad_accum_steps):
    """Scale by 1/accum and backward (AMP-aware). Caller does optimizer.step on the accum boundary."""
    scaled = loss / grad_accum_steps
    if scaler is not None:
        scaler.scale(scaled).backward()
    else:
        scaled.backward()


def train_one_step(model, batch, criterion, device, global_step, ssi_warmup_iters,
                   scaler=None, grad_accum_steps=1, iiw_batch=None):
    """One micro-batch step with up to FOUR forwards, each with its OWN backward so the forward
    graphs never co-reside (peak GPU = 2 graphs): (1) main + Hypersim/MID losses, (2) CARI
    cross-render L_inv/L_explain on rgb2, (3) shadow/sun self-sup (Hypersim-gated), (4) IIW ordinal
    hinge (post-baseline FT). Backward is done HERE; the caller only runs optimizer.step()/zero_grad()
    on the accum boundary. Returns a detached losses dict. derive_anneal ramps the Q99 derive scale."""
    model.train()

    rgb = batch['rgb'].to(device, non_blocking=True)
    loss_mask = batch.get('loss_mask', None)
    if loss_mask is None:
        raise KeyError("batch is missing required 'loss_mask'")
    loss_mask = loss_mask.to(device, non_blocking=True)
    m_diffuse = batch.get('M_diffuse', torch.zeros(rgb.shape[0])).float().to(device, non_blocking=True)
    m_residual = batch.get('m_residual', torch.ones(rgb.shape[0])).float().to(device, non_blocking=True)
    seg = batch.get('seg', None)
    if seg is not None:
        seg = seg.to(device, non_blocking=True)
    normals = batch.get('normals', None)
    if normals is not None:
        normals = normals.to(device, non_blocking=True)

    use_ccr = getattr(model, 'use_ccr_albedo', False)
    def _ccr(x):
        if not use_ccr:
            return None
        with torch.no_grad():
            return compute_ccr(x)

    # Derive-anneal ramp: Q99 scale 0.8/q99 → identity (a=I/S) over derive_anneal_iters after warmup.
    _anneal_iters = int(getattr(criterion, 'derive_anneal_iters', 12000))
    _anneal_cap = float(getattr(criterion, 'derive_anneal_cap', 1.0))
    derive_anneal = (1.0 if _anneal_iters <= 0
                     else min(1.0, max(0.0, (global_step - ssi_warmup_iters) / _anneal_iters)))
    # Cap below 1.0 so training never sits on the pure clamp(I/S,0,1) rail (gradient-death → collapse).
    derive_anneal = min(derive_anneal, _anneal_cap)
    # Pass derive_anneal only to models that accept it (V20 has **kwargs; V17/V18 don't).
    _msig = inspect.signature(model.forward).parameters
    _accepts_anneal = ('derive_anneal' in _msig
                       or any(p.kind == p.VAR_KEYWORD for p in _msig.values()))

    def _forward(x, ccr=None):
        kw = _forward_kwargs(model, m_diffuse, normals, seg, loss_mask, ccr)
        if _accepts_anneal:
            kw['derive_anneal'] = derive_anneal
        with autocast(device_type='cuda', dtype=torch.float16):
            preds = model(x, **kw)
            return {k: (v.float() if isinstance(v, torch.Tensor) else v) for k, v in preds.items()}

    # ── 1. Main forward + Hypersim/MID losses ──────────────────────────────────────
    predictions = _forward(rgb, _ccr(rgb))
    targets = compute_targets(predictions, batch)
    use_ssi = (global_step >= ssi_warmup_iters)
    decorr_scale = min(1.0, max(0.0, (global_step - ssi_warmup_iters) / max(1, ssi_warmup_iters)))
    losses = criterion(
        predictions=predictions, targets=targets, loss_mask=loss_mask,
        m_diffuse=m_diffuse, m_residual=m_residual, rgb=rgb,
        use_ssi=use_ssi, decorr_scale=decorr_scale,
    )

    # ── 2. CARI cross-render (rgb2 = same scene, different REAL light; paired rows only) ──────
    lam_inv = float(getattr(criterion, 'lambda_alb_invariance', 0.0))
    lam_explain = float(getattr(criterion, 'lambda_explain', 0.0))
    lam_cf_pair = float(getattr(criterion, 'lambda_chroma_field_pair', 0.0))
    rgb2 = batch.get('rgb2', None)
    m_invariant = batch.get('m_invariant', None)
    if (lam_inv > 0 or lam_explain > 0 or lam_cf_pair > 0) and rgb2 is not None and m_invariant is not None:
        m_invariant = m_invariant.float().to(device, non_blocking=True)
        if m_invariant.sum() > 0:
            rgb2 = rgb2.to(device, non_blocking=True)
            pair_valid = batch.get('pair_valid', loss_mask).float().to(device, non_blocking=True)
            pred2 = _forward(rgb2, _ccr(rgb2))
            cr_mask = loss_mask.float() * m_invariant.view(-1, 1, 1, 1) * pair_valid
            if lam_inv > 0:
                l_inv = criterion.cari_albedo_invariance(predictions['a_d'], pred2['a_d'].float(), cr_mask)
                losses['loss_alb_invariance'] = lam_inv * l_inv
                losses['loss_total'] = losses['loss_total'] + losses['loss_alb_invariance']
            if lam_explain > 0:
                l_explain = criterion.cari_explain(
                    rgb, rgb2, predictions['shading_linear'], pred2['shading_linear'].float(), cr_mask)
                losses['loss_explain'] = lam_explain * l_explain
                losses['loss_total'] = losses['loss_total'] + losses['loss_explain']
            # Colored-illuminant chroma target on the pair frame (V20 thesis lever): teach the chroma
            # head the REAL cast from rgb2's colored shading chroma(rgb2/A*) — the signal the neutral
            # WB primary frame cannot give. Same albedo A* (same scene) so material cancels.
            if lam_cf_pair > 0 and pred2.get('chroma_field') is not None and targets.get('A_d_star') is not None:
                l_cf_pair = criterion.cari_chroma_field(
                    pred2['chroma_field'].float(), rgb2, targets['A_d_star'], cr_mask)
                losses['loss_chroma_field_pair'] = lam_cf_pair * l_cf_pair
                losses['loss_total'] = losses['loss_total'] + losses['loss_chroma_field_pair']

    # ── Backward #1 (main + CARI). Detach shadow anchors BEFORE the graph is freed. ──────────
    lam_sinv = float(getattr(criterion, 'lambda_shadow_inv', 0.0))
    lam_sexp = float(getattr(criterion, 'lambda_shadow_explain', 0.0))
    lam_sgt_f3d = float(getattr(criterion, 'lambda_shadow_gt_front3d', 0.0))
    shadow_start = int(getattr(criterion, 'shadow_start_iter', 0))
    shadow_on_hypersim = bool(getattr(criterion, 'shadow_on_hypersim', True))
    shadow_on_front3d = bool(getattr(criterion, 'shadow_on_front3d', False))
    is_front3d = batch.get('is_front3d', torch.zeros(rgb.shape[0])).float().to(device, non_blocking=True)
    # Row-selection mask for the self-sup terms (shadow_inv/shadow_explain): OR of the two
    # dataset gates, each independently switchable so (35−29) can be a front3d-only σ row
    # without also re-enabling the Hypersim lever. Defaults (True/False) reproduce every
    # existing config's mask exactly: (m_diffuse > 0), byte-for-byte.
    shadow_row_mask = torch.zeros_like(m_diffuse)
    if shadow_on_hypersim:
        shadow_row_mask = shadow_row_mask + (m_diffuse > 0).float()
    if shadow_on_front3d:
        shadow_row_mask = shadow_row_mask + (is_front3d > 0).float()
    shadow_row_mask = shadow_row_mask.clamp(max=1.0)
    # The GT-anchor term is independent of shadow_on_front3d (that flag only gates the
    # self-sup terms) — it is scoped to is_front3d rows directly, always, so a config could in
    # principle run the GT anchor without the self-sup σ-invariance term.
    has_active_rows = bool((shadow_row_mask > 0).any()) or (lam_sgt_f3d > 0 and bool((is_front3d > 0).any()))
    shadow_active = ((lam_sinv > 0 or lam_sexp > 0 or lam_sgt_f3d > 0) and decorr_scale > 0
                     and global_step >= shadow_start and has_active_rows)
    a_anchor = predictions['a_d'].detach() if shadow_active else None
    s_anchor = predictions['shading_linear'].detach() if shadow_active else None
    _backward(losses['loss_total'], scaler, grad_accum_steps)
    losses['loss_total'] = losses['loss_total'].detach()

    # ── 3. Shadow/sun self-sup (Hypersim and/or front3d rows, per config): own forward + backward ──
    if shadow_active:
        from data.shadow_aug import sample_relight_field
        Bn, _, Hn, Wn = rgb.shape
        M, c = sample_relight_field(Bn, Hn, Wn, rgb.device)
        rgb_s = (rgb * M * c).clamp(0.0, 1.0)
        pred_s = _forward(rgb_s, _ccr(rgb_s))
        a_s, s_s = pred_s['a_d'].float(), pred_s['shading_linear'].float()
        # Bottom-clip (new, front3d-shadow-aug plan): σ can blacken pixels on the shadow side
        # of a strong field; those must not be scored, matching the existing top-clip guard.
        # SCOPED to when the new front3d mechanism is actually active (measured impact when it
        # is: 0.0017% of top-clip-valid pixels on a Hypersim-like batch — negligible either
        # way, but gating it keeps every existing Hypersim-lever config's mask, including
        # v17_23/33/34, PROVABLY byte-identical to what it always was, not just "negligibly
        # different" — no reason to accept even a tiny mismatch when it costs nothing to avoid.
        top_clip = (rgb_s.amax(dim=1, keepdim=True) < 0.99)
        if shadow_on_front3d or lam_sgt_f3d > 0:
            sh_valid = (top_clip & (rgb_s.amax(dim=1, keepdim=True) > 1e-3)).float()
        else:
            sh_valid = top_clip.float()
        sh_mask_spatial = loss_mask.float() * sh_valid
        sh_mask = sh_mask_spatial * shadow_row_mask.reshape(-1, 1, 1, 1)
        loss_shadow = rgb.new_zeros(())
        if lam_sinv > 0:
            term = lam_sinv * float(decorr_scale) * criterion.shadow_invariance(a_anchor, a_s, sh_mask)
            loss_shadow = loss_shadow + term
            losses['loss_shadow_inv'] = term.detach()
        if lam_sexp > 0:
            term = lam_sexp * float(decorr_scale) * criterion.shadow_explain(s_anchor, s_s, M, c, sh_mask)
            loss_shadow = loss_shadow + term
            losses['loss_shadow_explain'] = term.detach()
        if lam_sgt_f3d > 0:
            gt_mask = sh_mask_spatial * (is_front3d > 0).float().reshape(-1, 1, 1, 1)
            a_gt = targets['A_d_star'].detach()
            term = lam_sgt_f3d * float(decorr_scale) * criterion.shadow_invariance(a_gt, a_s, gt_mask)
            loss_shadow = loss_shadow + term
            losses['loss_shadow_gt_f3d'] = term.detach()
        _backward(loss_shadow, scaler, grad_accum_steps)
        losses['loss_total'] = losses['loss_total'] + loss_shadow.detach()

    # ── 4. IIW WHDR ordinal hinge (post-baseline FT): own forward + backward ─────────────────
    lam_ord_iiw = float(getattr(criterion, 'lambda_ordinal_iiw', 0.0))
    if lam_ord_iiw > 0 and iiw_batch is not None:
        iiw_rgb = iiw_batch['rgb'].to(device, non_blocking=True)
        pred_iiw = _forward(iiw_rgb, None)
        l_ord = criterion.ordinal_hinge_loss(pred_iiw['a_d'].float(), iiw_batch['comparisons'])
        loss_ord = lam_ord_iiw * l_ord
        _backward(loss_ord, scaler, grad_accum_steps)
        losses['loss_ordinal_iiw'] = loss_ord.detach()
        losses['loss_total'] = losses['loss_total'] + loss_ord.detach()

    return {k: (v.detach() if torch.is_tensor(v) else v) for k, v in losses.items()}


def _compute_lmse(pred, target, valid_mask, window_size=20, stride=10):
    """
    Local Mean Squared Error (LMSE) aligned with chrislib benchmark.
    
    Key difference from naive per-channel: for RGB inputs (C=3), chrislib
    concatenates all channels into one 2D array per patch and computes ONE
    joint alpha. For gray inputs (C=1), it's equivalent to per-channel.
    
    This matches chrislib's lmse_rgb / lmse_gray exactly.
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
    p_u = unfold(pred)                # (B, C*K*K, L)
    t_u = unfold(target)              # (B, C*K*K, L)
    m_u = unfold(valid_mask.float())  # (B, 1*K*K, L)
    
    # 3. Reshape to separate Channels (C) from Spatial Pixels (K*K)
    k2 = window_size * window_size
    L = p_u.shape[-1]
    p_u = p_u.view(B, C, k2, L)       # (B, C, K*K, L)
    t_u = t_u.view(B, C, k2, L)       # (B, C, K*K, L)
    m_u = m_u.view(B, 1, k2, L)       # (B, 1, K*K, L)

    # 4. Compute ONE joint alpha across all channels per patch
    # chrislib concatenates [R, G, B] into a single 2D array, then calls
    # ssq_error which computes: alpha = sum(correct * estimate * mask) / sum(estimate^2 * mask)
    # This is equivalent to summing over both spatial (K*K) AND channel (C) dims.
    m_expanded = m_u.expand_as(p_u)    # (B, C, K*K, L) — same mask for all channels
    num = (t_u * p_u * m_expanded).sum(dim=(1, 2))   # (B, L) — joint over C and K*K
    den = (p_u ** 2 * m_expanded).sum(dim=(1, 2))     # (B, L)

    den_safe = torch.where(den > 1e-5, den, torch.ones_like(den))
    alpha = torch.where(den > 1e-5, num / den_safe, torch.zeros_like(den))
    alpha = alpha.unsqueeze(1).unsqueeze(2)            # (B, 1, 1, L) — broadcasts over C and K*K

    # 5. Compute sum squared error per patch (joint over all channels)
    diff = t_u - alpha * p_u
    diff_sq = (diff ** 2) * m_expanded
    ssq_per_patch = diff_sq.sum(dim=(1, 2))            # (B, L)

    # 6. Compute total actual squared target (joint over all channels)
    total_per_patch = ((t_u ** 2) * m_expanded).sum(dim=(1, 2))  # (B, L)

    # 7. Sum over ALL patches per image (no patch filtering)
    ssq_sum = ssq_per_patch.sum(dim=1)                 # (B,)
    total_sum = total_per_patch.sum(dim=1)             # (B,)

    # Final division with safety net
    total_safe = torch.where(total_sum > 1e-7, total_sum, torch.ones_like(total_sum))
    lmse_val = torch.where(total_sum > 1e-7, ssq_sum / total_safe, torch.zeros_like(total_sum))

    return lmse_val.mean()


def _masked_rmse(pred, target, valid_mask, eps=1e-7):
    """
    Standard RMSE matching chrislib/metrics.py rmse_error implementation.
    """
    if pred.ndim == 3: pred = pred.unsqueeze(1)
    if target.ndim == 3: target = target.unsqueeze(1)
    if valid_mask.ndim == 3: valid_mask = valid_mask.unsqueeze(1)

    mask = valid_mask.bool().expand_as(pred)

    diff_sq = ((pred - target) ** 2) * mask.float()
    valid_counts = mask.float().reshape(pred.shape[0], -1).sum(dim=1).clamp_min(1)
    rmse_per_image = torch.sqrt(diff_sq.reshape(pred.shape[0], -1).sum(dim=1) / valid_counts)
    
    return rmse_per_image.mean()


def _masked_scale_invariant_rmse(pred, target, valid_mask, eps=1e-7):
    """
    Scale-invariant RMSE.
    Matches infer_hypersim.py for scale-ambiguous targets.
    """
    if pred.ndim == 3: pred = pred.unsqueeze(1)
    if target.ndim == 3: target = target.unsqueeze(1)
    if valid_mask.ndim == 3: valid_mask = valid_mask.unsqueeze(1)

    mask_f = valid_mask.float().expand_as(pred)
    
    pred_masked = pred * mask_f
    target_masked = target * mask_f
    
    valid_counts = mask_f.reshape(pred.shape[0], -1).sum(dim=1).clamp_min(1)
    p_mean = pred_masked.reshape(pred.shape[0], -1).sum(dim=1) / valid_counts
    t_mean = target_masked.reshape(target.shape[0], -1).sum(dim=1) / valid_counts
    
    # Expand means to match spatial dimensions
    p_mean_exp = p_mean.view(-1, 1, 1, 1)
    t_mean_exp = t_mean.view(-1, 1, 1, 1)
    
    # Normalize
    p_norm = pred / (p_mean_exp + eps)
    t_norm = target / (t_mean_exp + eps)
    
    diff_sq = ((p_norm - t_norm) ** 2) * mask_f
    rmse_per_image = torch.sqrt(diff_sq.reshape(pred.shape[0], -1).sum(dim=1) / valid_counts)
    
    return rmse_per_image.mean()


def _ssim_from_dssim(criterion, pred, target):
    """Convert criterion DSSIM helper to SSIM in [0,1]."""
    dssim = criterion._compute_dssim(pred, target)
    return 1.0 - 2.0 * dssim


def _make_gaussian_window(win_size, sigma, channels, device, dtype):
    """Create a Gaussian window matching skimage SSIM defaults."""
    coords = torch.arange(win_size, dtype=dtype, device=device) - win_size // 2
    gauss_1d = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    gauss_1d = gauss_1d / gauss_1d.sum()
    gauss_2d = gauss_1d.unsqueeze(1) @ gauss_1d.unsqueeze(0)
    return gauss_2d.unsqueeze(0).unsqueeze(0).expand(channels, 1, -1, -1).contiguous()


def _compute_shading_ssim(pred, target, valid_mask, eps=1e-6, shading_cap=None):
    """Batch-safe SSIM for unbounded shading with per-image LS alignment + robust scale."""
    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
        valid_mask = valid_mask.unsqueeze(0)

    B = pred.shape[0]
    mask = valid_mask.bool().expand_as(pred)
    if shading_cap is not None:
        mask = mask & (target <= shading_cap)

    p_masked = pred * mask.float()
    t_masked = target * mask.float()

    # 1. Per-image optimal alpha (batched least-squares)
    p_flat = p_masked.reshape(B, -1)
    t_flat = t_masked.reshape(B, -1)
    num = (t_flat * p_flat).sum(dim=1)
    den = (p_flat ** 2).sum(dim=1)
    alpha = torch.where(den > 1e-5, num / den.clamp_min(1e-5), torch.zeros_like(den))
    pred_aligned = pred * alpha.view(B, 1, 1, 1)

    # 2. Per-image robust scale (mean + 2*std of valid GT pixels)
    valid_counts = mask.float().reshape(B, -1).sum(dim=1).clamp_min(1.0)
    t_mean = t_flat.sum(dim=1) / valid_counts
    t_sq_mean = (t_flat ** 2).sum(dim=1) / valid_counts
    t_var = (t_sq_mean - t_mean ** 2).clamp_min(0.0)
    t_std = torch.sqrt(t_var)
    scale = (t_mean + 2.0 * t_std).clamp_min(eps).view(B, 1, 1, 1)

    # 3. Normalize into [0,1] and mask
    pred_n = torch.clamp(pred_aligned / scale, 0.0, 1.0) * mask.float()
    tgt_n = torch.clamp(target / scale, 0.0, 1.0) * mask.float()

    return _pytorch_ssim_skimage(pred_n, tgt_n, data_range=1.0)


def _pytorch_ssim_skimage(pred, target, data_range=1.0, win_size=11, sigma=1.5):
    """GPU SSIM matching skimage.metrics.structural_similarity defaults.
    
    Uses Gaussian window (sigma=1.5, win_size=11) with Bessel-corrected
    variance, exactly matching skimage's implementation.
    """
    # Ensure batched format
    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
        
    B, C, H, W = pred.shape
    if H < win_size or W < win_size or C == 0:
        return 0.0
        
    pad = win_size // 2
    p_pad = F.pad(pred, (pad, pad, pad, pad), mode='reflect')
    t_pad = F.pad(target, (pad, pad, pad, pad), mode='reflect')
    
    # Gaussian window matching skimage defaults (sigma=1.5, win_size=11)
    weight = _make_gaussian_window(win_size, sigma, C, pred.device, pred.dtype)
    
    mu_x = F.conv2d(p_pad, weight, groups=C)
    mu_y = F.conv2d(t_pad, weight, groups=C)
    
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y
    
    sigma_x_sq = F.conv2d(p_pad * p_pad, weight, groups=C) - mu_x_sq
    sigma_y_sq = F.conv2d(t_pad * t_pad, weight, groups=C) - mu_y_sq
    sigma_xy = F.conv2d(p_pad * t_pad, weight, groups=C) - mu_xy
    
    # Bessel correction (N/(N-1)) to match skimage's unbiased variance
    NP = win_size ** 2
    # For Gaussian window, the effective NP is sum(w)^2 / sum(w^2)
    # But skimage uses a simpler correction: cov_norm = NP / (NP - 1)
    # which is applied as sigma = sigma * NP / (NP - 1)
    cov_norm = NP / (NP - 1)
    sigma_x_sq = sigma_x_sq * cov_norm
    sigma_y_sq = sigma_y_sq * cov_norm
    sigma_xy = sigma_xy * cov_norm
    
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
    
    return ssim_map.mean().item()


def _compute_ssim_bounded(pred, target, valid_mask):
    """SSIM for bounded outputs (e.g., albedo).
    """
    pred_n = torch.clamp(pred, 0.0, 1.0)
    tgt_n = torch.clamp(target, 0.0, 1.0)

    mask_f = valid_mask.float().expand_as(tgt_n)
    pred_n = pred_n * mask_f
    tgt_n = tgt_n * mask_f
    
    return _pytorch_ssim_skimage(pred_n, tgt_n, data_range=1.0)


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


def _log_val_examples(
    writer,
    global_step,
    rgb,
    predictions,
    targets,
    max_items=2,
    full_rgb_list=None,
    sample_index=None,
    example_root='06_Examples',
):
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

    # Layout names are written per-sample after we know whether V20 outputs are present.
    is_v20 = ('chroma_field' in predictions and 'd_g' in predictions)

    # Set TensorBoard tag based on sample_index
    if sample_index is not None:
        example_tag = f'{example_root}/sample_{sample_index}'
    else:
        example_tag = example_root

    def _gamma_correct(x):
        return torch.pow(torch.clamp(x, min=0.0, max=1.0), 1.0 / 3.0)

    def _chroma_vis(c):
        """Visualise a unit-luminance chroma field: neutral (all-1) → mid-gray (0.5)."""
        if c is None:
            return None
        return _gamma_correct((c.clamp(0.0, 2.0) * 0.5))

    b = min(int(rgb.shape[0]), int(max_items))
    for i in range(b):
        # Robust mask: validate() sets 'loss_mask'; _log_dataset_examples sets 'valid_mask'
        v_mask = targets.get('loss_mask', targets.get('valid_mask',
                    torch.ones(1, 1, rgb.shape[2], rgb.shape[3],
                               device=rgb.device, dtype=torch.bool)))[i:i+1]

        # GT shading — display the COLORFUL S_c = I/A* (the derive-consistent shading the gray-SSI
        # loss + chroma head are actually trained toward), NOT the Hypersim diffuse illumination S_d.
        # Pred 'shading' = invert(g·chroma) → invert(S_c) at convergence, so GT = invert(S_c) is the
        # apples-to-apples reference. Falls back to pi_star (=π(S_d)) only for models w/o S_c_star.
        # predictions['shading'] is already π-domain [0,1]; keep GT in π too (no shared linear scale).
        a_gt_all = targets.get('A_d_star')
        a_gt = a_gt_all[i:i+1] if a_gt_all is not None else None
        # ONLY V20 trains shading against S_c (gray-SSI vs gray(I/A*)); V17/V18 train against
        # pi_star = π(S_d), so for them the GT row MUST stay S_d (else the viz contradicts the loss).
        sc_star = targets.get('S_c_star') if is_v20 else None
        if sc_star is not None:
            s_gt_linear = sc_star[i:i+1].clamp_min(0.0)
            s_gt_pi = (1.0 / (s_gt_linear + 1.0)).clamp(0.0, 1.0)            # invert(S_c), colored
            # Residual/recon GT consistent with the S_c derive: R=(I−A*·S_c)₊≈0, recon=A*·S_c=I —
            # matches the analytic decomposition the Pred row shows (vs the S_d residual the model
            # never predicts). Falls back below when S_c_star is absent.
            r_gt = (rgb[i:i+1] - a_gt * s_gt_linear).clamp(min=0.0) if a_gt is not None else None
        else:
            pi_star_gt = targets.get('pi_star')
            s_gt_pi = pi_star_gt[i:i+1].clamp(0.0, 1.0) if pi_star_gt is not None else None
            s_gt_linear = (1.0 / (s_gt_pi + 1e-6) - 1.0) if s_gt_pi is not None else None
            r_gt_all = targets.get('R_star')
            r_gt = r_gt_all[i:i+1] if r_gt_all is not None else None

        recon_gt = None
        if a_gt is not None and s_gt_linear is not None and r_gt is not None:
            recon_gt = a_gt * s_gt_linear + r_gt

        # Predictions
        s_pred_pi = predictions['shading'][i:i+1].clamp(0.0, 1.0)  # already π-domain
        a_pred = predictions['a_d'][i:i+1]
        r_pred = predictions['residual'][i:i+1]
        recon_pred = predictions.get('rgb_reconstructed',
                        a_pred * (1.0 / (s_pred_pi + 1e-6) - 1.0) + r_pred)[i:i+1]

        # V20-specific outputs
        chroma_pred = predictions['chroma_field'][i:i+1] if is_v20 else None
        d_g_pred    = predictions['d_g'][i:i+1]          if is_v20 else None
        chroma_gt_all = targets.get('chroma_field_gt')
        chroma_gt   = chroma_gt_all[i:i+1]               if chroma_gt_all is not None else None

        # Tonemap scales — shared GT/pred within each channel for fair comparison
        scale_a = _get_tonemap_scale(a_pred, valid_mask=v_mask)
        if a_gt is not None:
            scale_a = torch.max(scale_a, _get_tonemap_scale(a_gt, valid_mask=v_mask))
        scale_r = _get_tonemap_scale(r_pred.abs(), valid_mask=v_mask)
        if r_gt is not None:
            scale_r = torch.max(scale_r, _get_tonemap_scale(r_gt.abs(), valid_mask=v_mask))
        scale_recon = _get_tonemap_scale(recon_pred, valid_mask=v_mask)
        if recon_gt is not None:
            scale_recon = torch.max(scale_recon, _get_tonemap_scale(recon_gt, valid_mask=v_mask))

        # Albedo (tonemap + gamma)
        a_pred_vis = _gamma_correct(_vis_tonemap(a_pred, scale=scale_a))
        a_gt_vis   = _gamma_correct(_vis_tonemap(a_gt, scale=scale_a)) if a_gt is not None else None

        # Shading: display in π-domain directly — values already [0,1], no shared scale needed.
        # HIGH π = bright = LOW linear shading; LOW π = dark = HIGH linear shading (shadow).
        s_pred_vis = _gamma_correct(s_pred_pi)
        s_gt_vis   = _gamma_correct(s_gt_pi) if s_gt_pi is not None else None

        # Residual and recon
        r_pred_vis    = _gamma_correct(_vis_tonemap(r_pred.abs(), scale=scale_r))
        r_gt_vis      = _gamma_correct(_vis_tonemap(r_gt.abs(), scale=scale_r)) if r_gt is not None else None
        recon_pred_vis = _gamma_correct(_vis_tonemap(recon_pred, scale=scale_recon))
        recon_gt_vis   = _gamma_correct(_vis_tonemap(recon_gt, scale=scale_recon)) if recon_gt is not None else None

        blank_tile = torch.ones_like(a_pred_vis)
        input_vis  = _gamma_correct(_vis_tonemap(rgb[i:i+1], scale=None, valid_mask=v_mask))

        if is_v20:
            # 12 tiles — 2 rows of 6
            # Row 1 (GT):   Input | Albedo GT | Shading GT (π) | Chroma GT | Residual GT | Recon GT
            # Row 2 (Pred): Gray d_g | Albedo Pred | Shading Pred (π) | Chroma Pred | Residual Pred | Recon Pred
            sample_tiles = [
                ('Input RGB',        input_vis),
                ('Albedo GT',        a_gt_vis if a_gt_vis is not None else blank_tile),
                ('Shading GT (π)',   s_gt_vis if s_gt_vis is not None else blank_tile),
                ('Chroma GT',        _chroma_vis(chroma_gt) if chroma_gt is not None else blank_tile),
                ('Residual GT',      r_gt_vis if r_gt_vis is not None else blank_tile),
                ('Recon GT',         recon_gt_vis if recon_gt_vis is not None else blank_tile),

                ('Gray d_g',         _gamma_correct(d_g_pred.clamp(0, 1))),
                ('Albedo Pred',      a_pred_vis),
                ('Shading Pred (π)', s_pred_vis),
                ('Chroma Pred',      _chroma_vis(chroma_pred) if chroma_pred is not None else blank_tile),
                ('Residual Pred',    r_pred_vis),
                ('Recon Pred',       recon_pred_vis),
            ]
        else:
            # 10 tiles — 2 rows of 5 (non-V20 models; blank separates GT from Pred row)
            sample_tiles = [
                ('Input RGB',        input_vis),
                ('Albedo GT',        a_gt_vis if a_gt_vis is not None else blank_tile),
                ('Shading GT (π)',   s_gt_vis if s_gt_vis is not None else blank_tile),
                ('Residual GT',      r_gt_vis if r_gt_vis is not None else blank_tile),
                ('Recon GT',         recon_gt_vis if recon_gt_vis is not None else blank_tile),

                ('',                 blank_tile),
                ('Albedo Pred',      a_pred_vis),
                ('Shading Pred (π)', s_pred_vis),
                ('Residual Pred',    r_pred_vis),
                ('Recon Pred',       recon_pred_vis),
            ]

        final_target_hw = None
        footer_h = 50 
        tile_scale = 0.5
        try:
            from PIL import ImageFont
            font = ImageFont.truetype("DejaVuSans.ttf", 15)
        except OSError:
            from PIL import ImageFont
            font = ImageFont.load_default()

        named_tiles = []
        for name, tile in sample_tiles:
            try:
                if tile is None:
                    continue
                # Normalize per-tile first, then force a single final HW after all
                # visualization transforms (scale + footer + text) so cat() is safe.
                norm_tile = _normalize_tile(tile, target_hw=None)
                if norm_tile is None:
                    continue
                
                # Apply tile scaling
                if tile_scale != 1.0:
                    norm_tile = F.interpolate(
                        norm_tile.unsqueeze(0),
                        scale_factor=tile_scale,
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                
                # Add footer with title
                c, h, w = norm_tile.shape
                canvas = torch.ones((c, h + footer_h, w), dtype=norm_tile.dtype, device=norm_tile.device)
                canvas[:, :h, :] = norm_tile
                
                # Render title text onto footer using PIL
                try:
                    canvas_np = (canvas.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                    from PIL import Image, ImageDraw
                    img = Image.fromarray(canvas_np)
                    draw = ImageDraw.Draw(img)
                    text_w = draw.textlength(name, font=font)
                    x = max(0, int((w - text_w) // 2))
                    y = h + 10
                    draw.text((x, y), name, fill=(0, 0, 0), font=font)
                    norm_tile = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).permute(2, 0, 1)
                except Exception as e:
                    norm_tile = canvas
                
                if final_target_hw is None:
                    final_target_hw = (int(norm_tile.shape[1]), int(norm_tile.shape[2]))
                elif (
                    norm_tile.shape[1] != final_target_hw[0]
                    or norm_tile.shape[2] != final_target_hw[1]
                ):
                    norm_tile = torch.nn.functional.interpolate(
                        norm_tile.unsqueeze(0),
                        size=final_target_hw,
                        mode='bilinear',
                        align_corners=False,
                    ).squeeze(0)
                named_tiles.append((name, norm_tile))
            except Exception as exc:
                print(f"[warn][val step {global_step}] skipped tile '{name}' for sample_{i}: {exc}")

        if not named_tiles:
            continue

        tile_names = [n for n, _ in named_tiles]
        tiles = [t for _, t in named_tiles]
        writer.add_text(
            f'{example_tag}/layout',
            'left-to-right strip order: ' + ' | '.join(tile_names),
            global_step,
        )
        n = len(tiles)
        if n >= 18:
            row1_img = torch.cat(tiles[:6], dim=2)
            row2_img = torch.cat(tiles[6:12], dim=2)
            row3_img = torch.cat(tiles[12:18], dim=2)
            strip = torch.cat([row1_img, row2_img, row3_img], dim=1)
        elif n >= 12:
            # V20: 2 rows of 6 (GT row / Pred row)
            row1_img = torch.cat(tiles[:6], dim=2)
            row2_img = torch.cat(tiles[6:12], dim=2)
            strip = torch.cat([row1_img, row2_img], dim=1)
        elif n >= 10:
            row1_img = torch.cat(tiles[:5], dim=2)
            row2_img = torch.cat(tiles[5:10], dim=2)
            strip = torch.cat([row1_img, row2_img], dim=1)
        else:
            strip = torch.cat(tiles, dim=2)

        if sample_index is not None:
            image_tag = f'{example_tag}/index_{sample_index}'
        else:
            image_tag = f'{example_tag}/sample_{i}'
        writer.add_image(image_tag, strip, global_step)


def _log_dataset_examples(dataset, model, device, writer, global_step, n_samples=3,
                          seed=42, example_root='3. Examples', dataset_tag='Dataset'):
    """Visualize n_samples from a dataset via model inference.

    GT row is built with the SAME compute_targets() path validate() uses, so the MID/OpenRooms
    sheets show the real GT: Albedo GT (pseudo-GT albedo.exr), Shading GT = π(S_c = I/A*), Chroma
    GT, Residual GT (≈0), Recon GT (= I) — matching how eval_mid_constancy.py reads MID albedo.
    (The old code looked for batch['albedo']/['shading'] keys these datasets never emit — they use
    'albedo_raw'/'illum_raw' — so the entire GT row rendered blank.) Each sample is logged under a
    UNIQUE sample_index tag so the n samples don't all collapse onto .../sample_0 and overwrite."""
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)

    model.eval()
    with torch.no_grad():
        for dataset_idx in indices:
            sample = dataset[int(dataset_idx)]
            # Collate the single sample → batch of 1 (so compute_targets sees the (B,C,H,W) it expects).
            batch = {k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v)
                     for k, v in sample.items()}
            rgb = batch['rgb'].float()

            fwd_kwargs = {}
            if 'M_diffuse' in batch:
                fwd_kwargs['m_diffuse'] = batch['M_diffuse'].float()
            predictions = model(rgb, **fwd_kwargs)
            predictions = {k: (v.float() if isinstance(v, torch.Tensor) else v)
                           for k, v in predictions.items()}

            # Full targets (A_d_star, S_c_star, chroma_field_gt, R_star, pi_star, loss_mask).
            try:
                targets = compute_targets(predictions, batch)
            except Exception as exc:
                print(f"[warn] {dataset_tag} viz compute_targets failed at step {global_step}: {exc}")
                lm = batch.get('loss_mask')
                targets = {'valid_mask': lm.bool() if lm is not None else
                           torch.ones(1, 1, rgb.shape[2], rgb.shape[3], device=device, dtype=torch.bool)}

            # Unique sample_index → distinct TensorBoard tag per sample (no overwrite).
            _log_val_examples(
                writer=writer,
                global_step=global_step,
                rgb=rgb,
                predictions=predictions,
                targets=targets,
                max_items=1,
                sample_index=int(dataset_idx),
                example_root=f'{example_root}/{dataset_tag}',
            )


_FRONT3D_VAL_CACHE = {}  # root -> (dataset, fixed view-index subset) — scanned once per process


def _get_front3d_val_probe_set(root_dir, input_size, n_views=8):
    """Lazily build (and cache) a small, FIXED subset of the front3d HELD-OUT val split
    (rooms hashed out of training, never seen regardless of mix weight). Fixed indices so
    the same views are compared checkpoint-to-checkpoint — a consistent smoke set, not a
    fresh random draw each time. Returns None if the dataset has no val views (e.g. render
    not finished yet) so callers can skip cleanly rather than crash a live training run."""
    key = (root_dir, input_size)
    if key not in _FRONT3D_VAL_CACHE:
        try:
            ds = Front3DDataset(root_dir=root_dir, split='val', input_size=input_size)
        except Exception as exc:
            print(f'[warn] front3d val probe: dataset init failed ({exc}); skipping.')
            _FRONT3D_VAL_CACHE[key] = None
            return None
        if len(ds) == 0:
            _FRONT3D_VAL_CACHE[key] = None
        else:
            idx = list(range(min(n_views, len(ds))))
            _FRONT3D_VAL_CACHE[key] = (ds, idx)
    return _FRONT3D_VAL_CACHE[key]


@torch.no_grad()
def _run_front3d_val_probe(model, device, config, global_step, writer):
    """Cheap (~8-view) live check of what the offline-sweep probe would have measured, made
    necessary because only the 2 latest checkpoints are kept on disk (no post-hoc sweep
    possible). Reports, on FRONT3D'S OWN held-out rooms:
      alb_si_rmse : scale-invariant albedo error vs true GT albedo (accuracy)
      inv_gap     : mean L1 between A(rgb_L0) and A(rgb_L1) on paired pixels (the raw
                    cross-illuminant invariance CARI trains for)
    Skipped entirely (no TB tag written) when front3d isn't in this run's mix, so rows that
    don't use it stay clean. Wrapped defensively: any failure here must never take down an
    otherwise-healthy training run.
    """
    try:
        front3d_weight = float(config.get('train', {}).get('sampling_weights_phase3', {}).get('front3d', 0.0))
        if front3d_weight <= 0:
            return
        root_dir = config.get('data', {}).get('front3d_root', '../datasets/front3d_iid')
        input_size = int(config.get('train', {}).get('input_size', 384))
        probe = _get_front3d_val_probe_set(root_dir, input_size, n_views=8)
        if probe is None:
            return
        ds, idx = probe

        was_training = model.training
        model.eval()
        use_ccr = getattr(model, 'use_ccr_albedo', False)
        accepts_ccr = 'ccr' in inspect.signature(model.forward).parameters

        def _fwd(x):
            kw = {}
            if accepts_ccr and use_ccr:
                kw['ccr'] = compute_ccr(x)
            with autocast(device_type='cuda', dtype=torch.float16):
                return model(x, **kw)['a_d'].float()

        # σ-invariance probe (front3d-shadow-aug plan, documents/design/FRONT3D_SHADOW_AUG_PLAN.md,
        # smoke test S2's statistic logged continuously): mean|A(I) − A(I⊙σ)| under a synthetic
        # hard-shadow field, on the SAME clean forward `a1` already computed for inv_gap (no extra
        # forward beyond the one σ-perturbed branch) — watches whether this channel is still
        # saturated (near-zero, as measured pre-fix) or has live gradient once training resumes.
        from data.shadow_aug import sample_relight_field

        si_rmses, inv_gaps, inv_gaps_sigma = [], [], []
        for i in idx:
            b = ds[i]
            rgb = b['rgb'].unsqueeze(0).to(device, non_blocking=True)
            rgb2 = b['rgb2'].unsqueeze(0).to(device, non_blocking=True)
            alb_gt = b['albedo_scaled'].unsqueeze(0).to(device, non_blocking=True)
            mask = b['loss_mask'].unsqueeze(0).to(device, non_blocking=True)
            pair_valid = b['pair_valid'].unsqueeze(0).to(device, non_blocking=True)

            a1, a2 = _fwd(rgb), _fwd(rgb2)

            si_rmses.append(_masked_scale_invariant_rmse(a1, alb_gt, mask).item())
            pv = (mask & pair_valid).float().expand_as(a1)
            inv_gaps.append(((a1 - a2).abs() * pv).sum().item() / (pv.sum().item() + 1e-6))

            M, c = sample_relight_field(1, rgb.shape[2], rgb.shape[3], rgb.device)
            rgb_sig = (rgb * M * c).clamp(0.0, 1.0)
            sig_valid = ((rgb_sig.amax(dim=1, keepdim=True) < 0.99)
                        & (rgb_sig.amax(dim=1, keepdim=True) > 1e-3))
            a_sig = _fwd(rgb_sig)
            pv_sig = (mask & sig_valid).float().expand_as(a1)
            inv_gaps_sigma.append(((a1 - a_sig).abs() * pv_sig).sum().item() / (pv_sig.sum().item() + 1e-6))

        if was_training:
            model.train()
        n = len(si_rmses)
        if n == 0:
            return
        writer.add_scalar('2. Val/3. Front3D/alb_si_rmse', sum(si_rmses) / n, global_step)
        writer.add_scalar('2. Val/3. Front3D/inv_gap', sum(inv_gaps) / n, global_step)
        writer.add_scalar('2. Val/3. Front3D/inv_gap_sigma', sum(inv_gaps_sigma) / n, global_step)
    except Exception as exc:
        print(f'[warn] front3d val probe failed at step {global_step} ({exc}); skipping this step.')


def validate(
    model,
    dataloader,
    criterion,
    device,
    global_step,
    writer,
    val_example_images=2,
    val_example_indices=None,
    max_val_batches=None,
    compute_val_losses=True,
    example_root='3_Examples',
    config=None,
):
    """
    Validation on fixed Hypersim val split.
    Logs both losses and per-decoder metrics from scale-matched targets.
    """
    model.eval()
    total_loss = {}
    total_metric = {
        'a_d_lmse': 0.0,
        'a_d_rmse': 0.0,
        'a_d_ssim': 0.0,
        's_d_lmse': 0.0,
        's_d_rmse': 0.0,
        's_d_ssim': 0.0,
        'r_rmse': 0.0,
    }
    n_samples = 0
    n_s_d_samples = 0
    
    # Prepare sample collection for visualization
    if val_example_indices is None:
        val_example_indices = []
    val_example_indices = list(val_example_indices) if val_example_indices else []
    use_indices = len(val_example_indices) > 0
    use_count = not use_indices
    
    indices_to_collect = set(int(idx) for idx in val_example_indices)
    collected_samples = {}  # Maps global index to (rgb, predictions, targets)
    processed_batches = 0

    with torch.no_grad():
        metric_batches_to_process = len(dataloader) if max_val_batches is None else min(max_val_batches, len(dataloader))
        progress_total = len(dataloader) if use_indices else metric_batches_to_process

        for batch_idx, batch in enumerate(tqdm(dataloader, desc='Validation', total=progress_total)):
            metric_budget_reached = max_val_batches is not None and batch_idx >= max_val_batches
            need_more_index_samples = use_indices and (len(collected_samples) < len(indices_to_collect))
            if metric_budget_reached and not need_more_index_samples:
                break

            rgb = batch['rgb'].to(device, non_blocking=True)
            loss_mask = batch.get('loss_mask', None)
            if loss_mask is None:
                raise KeyError("batch is missing required 'loss_mask'")
            loss_mask = loss_mask.to(device, non_blocking=True)
            m_residual = batch.get('m_residual', torch.ones(rgb.shape[0])).float().to(device, non_blocking=True)
            m_diffuse = batch.get('M_diffuse', torch.zeros(rgb.shape[0])).float().to(device, non_blocking=True)
            seg = batch.get('seg', None)
            if seg is not None:
                seg = seg.to(device, non_blocking=True)
            normals = batch.get('normals', None)
            if normals is not None:
                normals = normals.to(device, non_blocking=True)
            with torch.no_grad():
                ccr = compute_ccr(rgb)
            predictions = model(rgb, **_forward_kwargs(model, m_diffuse=None, normals=normals, seg=seg, valid_mask=loss_mask, ccr=ccr))
            
            # Ensure predictions are float32 for stable metric computation
            predictions = {k: v.float() if isinstance(v, torch.Tensor) else v for k, v in predictions.items()}
            
            targets = compute_targets(predictions, batch)

            # Collect samples at specified indices.
            batch_size = rgb.shape[0]
            sample_indices = batch.get('sample_idx', None)
            if use_indices:
                for i in range(batch_size):
                    global_idx = int(sample_indices[i].item()) if sample_indices is not None else int(batch_idx * dataloader.batch_size + i)
                    if global_idx in indices_to_collect:
                        collected_samples[global_idx] = {
                            'rgb': rgb[i:i+1],
                            'predictions': {k: v[i:i+1] for k, v in predictions.items()},
                            'targets': {k: v[i:i+1] for k, v in targets.items()},
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
                    example_root=example_root,
                )

            if metric_budget_reached:
                continue

            processed_batches += 1

            if compute_val_losses:
                losses = criterion(
                    predictions=predictions,
                    targets=targets,
                    loss_mask=loss_mask,
                    m_diffuse=m_diffuse,
                    m_residual=m_residual,
                    rgb=rgb,
                    use_ssi=True,
                )
                for k, v in losses.items():
                    if k not in total_loss:
                        total_loss[k] = 0.0
                    total_loss[k] += v.item()

            # Evaluate metrics
            # 1. Albedo metrics
            total_metric['a_d_lmse'] += _compute_lmse(predictions['a_d'], targets['A_d_star'], loss_mask).item() * batch_size
            total_metric['a_d_rmse'] += _masked_scale_invariant_rmse(predictions['a_d'], targets['A_d_star'], loss_mask).item() * batch_size
            total_metric['a_d_ssim'] += _compute_ssim_bounded(predictions['a_d'], targets['A_d_star'], loss_mask) * batch_size

            # 2. Diffuse Shading metrics (from Softplus Linear Shading)
            s_d_gt = targets['S_d_star']
            s_d_pred = predictions['shading']
            
            total_metric['s_d_lmse'] += _compute_lmse(s_d_pred, s_d_gt, loss_mask).item() * batch_size
            total_metric['s_d_rmse'] += _masked_scale_invariant_rmse(s_d_pred, s_d_gt, loss_mask).item() * batch_size
            total_metric['s_d_ssim'] += _compute_shading_ssim(s_d_pred, s_d_gt, loss_mask) * batch_size
            n_s_d_samples += batch_size

            # 3. Residual metrics
            r_pred = predictions['residual']
            r_gt = targets['R_star']
            total_metric['r_rmse'] += _masked_scale_invariant_rmse(r_pred, r_gt, loss_mask).item() * batch_size
            
            n_samples += batch_size

    if use_indices and collected_samples:
        for global_idx in sorted(collected_samples.keys()):
            sample = collected_samples[global_idx]
            _log_val_examples(
                writer, global_step, sample['rgb'], sample['predictions'], sample['targets'],
                max_items=1, sample_index=global_idx, example_root=example_root,
            )

    if compute_val_losses:
        denom_loss = max(processed_batches, 1)
        for k in total_loss: total_loss[k] /= denom_loss

    denom_metric = max(n_samples, 1)
    for k in total_metric:
        if k.startswith('s_d_'):
            total_metric[k] /= max(n_s_d_samples, 1)
        else:
            total_metric[k] /= denom_metric

    val_out = {}
    if compute_val_losses: val_out.update(total_loss)
    val_out.update(total_metric)
    # tag_prefix='2. Val': without it these averages land in the SAME '1. Losses/*' tags as
    # the per-batch train losses (periodic spikes in the train curves), and the metric keys
    # (a_d_rmse etc.) are silently dropped by _log_ordered_scalars' key filter.
    _log_ordered_scalars(writer, val_out, global_step, tag_prefix='2. Val')
    for k, v in total_metric.items():
        writer.add_scalar(f'2. Val/2. Metrics/{k}', float(v), global_step)

    if config is not None:
        _run_front3d_val_probe(model, device, config, global_step, writer)

    return val_out



def build_stage1_model(config):
    version = float(config['model'].get('version', 11))
    model_cfg = {
        'z_channels': config['model'].get('z_channels', 1024),
        'freeze_stages': config['model'].get('freeze_stages', [1, 2]),
        'backbone': config['model'].get('backbone', 'convnextv2_base'),
        'pretrained': config['model'].get('pretrained', True),
        'num_seg_classes': config['model'].get('num_seg_classes', 41),
        'input_size': int(config['train'].get('input_size', 384)),
        # DPT decoder + head dims — must match checkpoint architecture
        'dpt_feat_ch': int(config['model'].get('dpt_feat_ch', 256)),
        'dpt_fusion_ch': int(config['model'].get('dpt_fusion_ch', 128)),
        'dpt_out_ch': int(config['model'].get('dpt_out_ch', 128)),
        'detail_ch': int(config['model'].get('detail_ch', 48)),
        'head_mid': int(config['model'].get('head_mid', 64)),
        # Physics-typed skip gates
        'albedo_chroma_skip': config['model'].get('albedo_chroma_skip', True),
        'shading_lum_skip': config['model'].get('shading_lum_skip', True),
        'albedo_rgb_skip': config['model'].get('albedo_rgb_skip', False),
        'dino_variant': config['model'].get('dino_variant', 'large'),
        'dino_pretrained': config['model'].get('dino_pretrained', True),
        'refiner': config['model'].get('refiner', {}),
    }

    model_map = {
        17.0: IntrinsicDecompositionV17,
    }

    if version not in model_map:
        raise ValueError(f"Unsupported Stage1 version for single dataset mode: {version}. Supported: 13, 14, 15, 16, 17, 17.27")
    return model_map[version](model_cfg)


def _subset_dataset(dataset, dataset_name, max_images, seed=None):
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

    if seed is None:
        indices = torch.randperm(total)[:max_images].tolist()
    else:
        gen = torch.Generator()
        gen.manual_seed(int(seed))
        indices = torch.randperm(total, generator=gen)[:max_images].tolist()
    subset = Subset(dataset, indices)
    print(
        f"[subset] {dataset_name}: using {len(subset)}/{total} images "
        f"(seed={int(seed)})"
    )
    return subset, total, len(subset)


def build_optimizer_stage1(model, train_cfg, model_cfg):
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
    config = load_config(args.config, args.version)
    print(yaml.dump(config, default_flow_style=False))

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    torch.backends.cudnn.benchmark = True

    # Use command-line version for paths if provided, else fall back to config's model version
    path_version = args.version if args.version is not None else config['model']['version']
    ckpt_dir = os.path.join(config['paths']['checkpoint_dir'], f'v{path_version}')
    log_dir = os.path.join(config['paths']['log_dir'], f'v{path_version}')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)
    model = build_stage1_model(config).to(device)
    criterion = V17Loss(config['loss']).to(device)
    optimizer = build_optimizer_stage1(model, config['train'], config['model'])

    hypersim_root = config['data']['hypersim_root']
    if not os.path.isabs(hypersim_root):
        hypersim_root = str(ROOT_DIR / hypersim_root)

    split_file = config['data'].get('hypersim_split_file', 'hypersim_split.json')
    split_seed = int(config['data'].get('hypersim_split_seed', 42))
    split_ratio = float(config['data'].get('hypersim_split_ratio', 0.9))
    strict_split = bool(config['data'].get('hypersim_strict_split', True))
    hypersim_max_hdf5_retries = int(config['data'].get('hypersim_max_hdf5_retries', 1))
    hypersim_skip_corrupt_samples = bool(config['data'].get('hypersim_skip_corrupt_samples', True))

    cache_max_items = int(config['data'].get('cache_max_items', 512))
    crop_mode_train = str(config['data'].get('crop_mode_train', 'random'))
    crop_mode_val = str(config['data'].get('crop_mode_val', 'center'))
    subset_seed = None

    hypersim_train_max_images = int(config['data'].get('hypersim_train_max_images', 0))

    from src.data.mixed_dataset import get_mixed_loader
    # InteriorVerse is OPT-IN: the dataset is only instantiated when its sampling weight > 0
    # (see get_mixed_loader). Wiring the root here makes it a one-line config flip. NOTE: IV sets
    # M_diffuse=0 (interiorverse_dataset.py) so it is EXCLUDED from the Hypersim-gated shadow
    # self-sup (sh_mask) — it adds albedo-supervision diversity only, never shadow signal.
    iv_root = config['data'].get('interiorverse_root',
                                 '../datasets/IndoorInverseRendering/interiorverse/interiorverse/interverse')
    front3d_root = config['data'].get('front3d_root', '../../../datasets/front3d_iid')
    def _build_train_loader(weights_key):
        return get_mixed_loader(
            data_roots={
                'hypersim': hypersim_root,
                'midintrinsic': config['data'].get('midintrinsic_root', '../../../datasets/MIDIntrinsics'),
                'interiorverse': iv_root,
                'front3d': front3d_root,
            },
            batch_size=int(config['train']['batch_size']),
            split='train',
            num_workers=int(config['train'].get('num_workers', 4)),
            input_size=int(config['train']['input_size']),
            cache_max_items=cache_max_items,
            mix_weights=config['train'].get(weights_key, {'hypersim': 1.0, 'midintrinsic': 0.0}),
            strict_split=strict_split,
            load_geometry=False,
            load_normals=False,
            # MID CARI pairing + Hypersim tint-pair flags (config data:). get_mixed_loader
            # reads these from kwargs with OFF defaults, so omitting them silently disables
            # MID raw cross-render pairs no matter what v17.yaml says (v18/v20 trainers
            # already forward use_mid_paired; this one didn't).
            use_mid_paired=bool(config['data'].get('use_mid_paired', False)),
            mid_pair_mode=str(config['data'].get('mid_pair_mode', 'raw')),
            mid_chromatic_aug=bool(config['data'].get('mid_chromatic_aug', False)),
            mid_raw_color_pair=bool(config['data'].get('mid_raw_color_pair', False)),
            hypersim_color_pair_prob=float(config['data'].get('hypersim_color_pair_prob', 0.0)),
            hypersim_color_tint_min=float(config['data'].get('hypersim_color_tint_min', 0.8)),
            hypersim_color_tint_max=float(config['data'].get('hypersim_color_tint_max', 1.25)),
            # Per-worker LRU cache of decoded front3d EXRs. Conservative default (128 ≈
            # 0.4 GB/worker) because the machine is memory-pressured and per-worker caches
            # don't share; raise toward 512 for single-trainer runs with free RAM, or set 0
            # to disable. Hit rate ≈ this/n_views (uniform random sampling — see dataset).
            front3d_cache_max_items=int(config['data'].get('front3d_cache_max_items', 128)),
        )

    def infinite_loader(dl):
        while True:
            for b in dl:
                yield b

    val_num_workers = max(1, int(config['data'].get('val_num_workers', config['train'].get('val_num_workers', 2))))
    val_cache_max_items = max(0, int(config['data'].get('val_cache_max_items', 64)))
    val_batch_size = int(config['train']['batch_size'])

    val_loader = get_hypersim_loader(
        root_dir=hypersim_root,
        batch_size=val_batch_size,
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
        load_geometry=False,
    )

    max_iters = int(config['train'].get('extend_iterations', 25000))
    grad_accum_steps = max(1, int(config['train'].get('grad_accum_steps', 1)))
    ssi_warmup = int(config['loss'].get('ssi_warmup_iters', 3000))   # legacy V17 loop warmup gate
    grad_clip_max_norm = float(config['train'].get('grad_clip_max_norm', 1.0))
    use_cosine_lr = bool(config['train'].get('use_cosine_lr', True))
    lr_eta_min = float(config['train'].get('lr_eta_min', 1.0e-7))
    val_interval_iters = int(config['train'].get('val_interval_iters', 2000))
    ckpt_interval_iters = int(config['train'].get('checkpoint_interval_iters', 5000))
    val_example_images = int(config['train'].get('val_example_images', 2))
    val_example_indices = config['train'].get('val_example_indices', [])
    max_val_batches = config['train'].get('max_val_batches', None)
    if max_val_batches is not None:
        max_val_batches = int(max_val_batches)
    compute_val_losses = bool(config['train'].get('compute_val_losses', True))

    start_step = 0
    resume_path = _resolve_resume_path(args.resume, args.auto_resume, ckpt_dir)
    if resume_path:
        skip_opt = args.skip_optimizer or config['train'].get('skip_optimizer', False)
        start_step, _ = load_checkpoint(model, optimizer, resume_path, map_location=device, skip_optimizer=skip_opt)
        start_step += 1
        
        if args.reset_lr or config['train'].get('reset_lr', False):
            base_lr = float(config['train']['lr'])
            multiplier = float(config['model'].get('backbone_lr_multiplier', 1.0))
            if len(optimizer.param_groups) == 2:
                optimizer.param_groups[0]['lr'] = base_lr * multiplier
                optimizer.param_groups[1]['lr'] = base_lr
            else:
                for pg in optimizer.param_groups:
                    pg['lr'] = base_lr
            print(f"!!! LR Reset: Manual override to {base_lr} from config")

    # ── IIW ordinal-hinge fine-tune loader (off unless lambda_ordinal_iiw>0) ──────────────
    # Joint WHDR fine-tune (post-baseline): a dedicated IIW human-judgment batch per step feeds
    # the ordinal hinge in train_one_step (§4) while the Hypersim+MID constancy losses stay on.
    # Train split = COMPLEMENT of the eval test split (no leakage). See src/data/iiw_dataset.py
    # + src/configs/v20_iiw_ft.yaml (the V17 analog extends the final model's config). Ported
    # from train_v20.py's main loop so a V17 final-model IIW-ft runs through train_v17.py.
    iiw_iter = None
    if float(config['loss'].get('lambda_ordinal_iiw', 0.0)) > 0:
        from data.iiw_dataset import get_iiw_loader
        _iiw_root = config['data'].get('iiw_root', 'tests/testing_data/iiw-dataset/data')
        _iiw_root = _iiw_root if os.path.isabs(_iiw_root) else str(ROOT_DIR / _iiw_root)
        iiw_loader = get_iiw_loader(
            _iiw_root, split='train',
            batch_size=int(config['train'].get('iiw_batch_size', 2)),
            input_size=int(config['train']['input_size']),
            num_workers=max(1, int(config['train'].get('num_workers', 4)) // 2),
        )
        iiw_iter = iter(infinite_loader(iiw_loader))
        print(f"[IIW] ordinal-hinge fine-tune ENABLED: {len(iiw_loader.dataset)} train images "
              f"(lambda={config['loss']['lambda_ordinal_iiw']}), root={_iiw_root}")

    scheduler = None
    if use_cosine_lr:
        total_opt_steps = max(1, math.ceil(max_iters / grad_accum_steps))
        completed_opt_steps = max(0, start_step // grad_accum_steps)
        for pg in optimizer.param_groups:
            pg.setdefault('initial_lr', pg['lr'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_opt_steps, eta_min=lr_eta_min, last_epoch=completed_opt_steps - 1
        )

    phase1_iters = int(config['train'].get('phase1_iterations', -1))
    phase2_iters = int(config['train'].get('phase2_iterations', -1))

    def _weights_key_for_step(step):
        if phase2_iters >= 0 and step >= phase2_iters and 'sampling_weights_phase3' in config['train']:
            return 'sampling_weights_phase3'
        if phase1_iters >= 0 and step >= phase1_iters and 'sampling_weights_phase2' in config['train']:
            return 'sampling_weights_phase2'
        return 'sampling_weights_phase1'

    active_weights_key = _weights_key_for_step(start_step)
    print(
        f"[data] Initial train mix at step {start_step}: {active_weights_key} = "
        f"{config['train'].get(active_weights_key, {})}"
    )
    train_loader = _build_train_loader(active_weights_key)
    train_iter = iter(infinite_loader(train_loader))

    print(f"Start step: {start_step}, max step: {max_iters}")

    running = {}

    train_pbar = tqdm(range(start_step, max_iters), desc='Training', total=max_iters - start_step, dynamic_ncols=True)

    scaler = GradScaler('cuda')

    for step in train_pbar:
        desired_weights_key = _weights_key_for_step(step)
        if desired_weights_key != active_weights_key:
            active_weights_key = desired_weights_key
            print(
                f"--- Switching train mix at step {step}: {active_weights_key} = "
                f"{config['train'].get(active_weights_key, {})} ---"
            )
            train_loader = _build_train_loader(active_weights_key)
            train_iter = iter(infinite_loader(train_loader))

        batch = next(train_iter)
        iiw_batch = next(iiw_iter) if iiw_iter is not None else None
        # train_one_step now does its own (split) backward internally — do NOT backward again here.
        losses = train_one_step(model, batch, criterion, device, step, ssi_warmup,
                                scaler=scaler, grad_accum_steps=grad_accum_steps,
                                iiw_batch=iiw_batch)

        if (step + 1) % grad_accum_steps == 0 or step == max_iters - 1:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
        
        train_pbar.set_postfix({'loss': f"{losses['loss_total'].item():.4f}"})

        for k, v in losses.items():
            if k not in running:
                running[k] = 0.0
            running[k] += v.item()

        if step % int(config['train']['log_interval']) == 0:
            _log_ordered_scalars(writer, losses, step, tag_prefix=None)
            if len(optimizer.param_groups) > 1:
                writer.add_scalar('0. Training/lr_backbone', float(optimizer.param_groups[0]['lr']), step)
                writer.add_scalar('0. Training/lr_heads', float(optimizer.param_groups[1]['lr']), step)
            else:
                writer.add_scalar('0. Training/lr', float(optimizer.param_groups[0]['lr']), step)

        if (step + 1) % ckpt_interval_iters == 0:
            avg = {k: running[k] / ckpt_interval_iters for k in running}
            for k in running: running[k] = 0.0
            ckpt_path = os.path.join(ckpt_dir, f'checkpoint_iter_{step+1}.pth')
            save_checkpoint(model, optimizer, avg, config, ckpt_path, global_step=step)
            save_checkpoint(model, optimizer, avg, config, os.path.join(ckpt_dir, 'checkpoint_latest.pth'), global_step=step)

            # Keep maximum 2 checkpoints (exclude checkpoint_latest.pth)
            all_ckpts = [
                os.path.join(ckpt_dir, f)
                for f in os.listdir(ckpt_dir)
                if f.startswith('checkpoint_iter_') and f.endswith('.pth')
            ]
            all_ckpts.sort(key=_extract_iter_from_name)
            while len(all_ckpts) > 2:
                oldest_ckpt = all_ckpts.pop(0)
                try:
                    os.remove(oldest_ckpt)
                    print(f"Deleted old checkpoint: {oldest_ckpt}")
                except Exception as e:
                    print(f"Failed to delete {oldest_ckpt}: {e}")

        if (step + 1) % val_interval_iters == 0:
            torch.cuda.empty_cache()
            vloss = validate(
                model, val_loader, criterion, device, step + 1, writer,
                val_example_images=val_example_images, val_example_indices=val_example_indices,
                max_val_batches=max_val_batches, compute_val_losses=compute_val_losses,
                example_root='3. Examples', config=config,
            )
            pretty = ", ".join([f"{k}={v:.4f}" for k, v in vloss.items()])
            print(f"[{step+1}] val: {pretty}")

    print('Training completed')
    writer.close()


if __name__ == '__main__':
    main()

