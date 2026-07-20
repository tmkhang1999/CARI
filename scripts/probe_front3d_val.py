#!/usr/bin/env python3
"""Offline front3d-val probe: the memorization / generalization signal the trainer
does not log (train_v17 logs only pooled losses and validates on Hypersim only).

Runs a checkpoint on Front3DDataset's HELD-OUT val split (rooms hashed out of
training, val_fraction=0.05) and reports:
  - alb_si_rmse : scale-invariant RMSE of predicted albedo vs true GT albedo
  - inv_gap     : mean L1 between A(rgb_L0) and A(rgb_L1) over pair-valid pixels —
                  the raw cross-illuminant invariance the CARI loss trains for
Sweep several checkpoints of one run to see whether held-out front3d performance
keeps improving (fine) or stalls/regresses while the train loss falls (memorizing
the 2.4k-view corpus — back off the front3d mix weight for the ship model).

Usage (repeat --checkpoint to sweep; ~120 val views per checkpoint):
  python scripts/probe_front3d_val.py \
      --checkpoint checkpoints/v17_29/checkpoint_iter_50000.pth \
      --checkpoint checkpoints/v17_29/checkpoint_iter_60000.pth \
      --device cuda:0 [--front3d-root ../datasets/front3d_iid] [--max-views N]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / 'src'))

from src.data.front3d_dataset import Front3DDataset            # noqa: E402
from src.models import IntrinsicDecompositionV17               # noqa: E402
from src.train import _masked_scale_invariant_rmse             # noqa: E402


def load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model_cfg = ckpt.get('config', {}).get('model', {})
    model = IntrinsicDecompositionV17(model_cfg).to(device)
    sd = ckpt.get('model_state_dict', ckpt)
    own = model.state_dict()
    filtered = {k: v for k, v in sd.items() if k in own and v.shape == own[k].shape}
    model.load_state_dict(filtered, strict=False)
    model.eval()
    step = ckpt.get('global_step', '?')
    print(f'  loaded {ckpt_path} (step={step}, {len(filtered)}/{len(own)} tensors)')
    return model, step


@torch.no_grad()
def probe(model, ds, device, max_views=0):
    n = len(ds) if max_views <= 0 else min(max_views, len(ds))
    si_rmses, inv_gaps = [], []
    for i in range(n):
        b = ds[i]
        rgb = b['rgb'].unsqueeze(0).to(device)
        rgb2 = b['rgb2'].unsqueeze(0).to(device)
        alb_gt = b['albedo_scaled'].unsqueeze(0).to(device)
        mask = b['loss_mask'].unsqueeze(0).to(device)
        pair_valid = b['pair_valid'].unsqueeze(0).to(device)

        a1 = model(rgb)['a_d'].float()
        a2 = model(rgb2)['a_d'].float()

        si_rmses.append(_masked_scale_invariant_rmse(a1, alb_gt, mask).item())
        pv = (mask & pair_valid).float().expand_as(a1)
        inv_gaps.append(((a1 - a2).abs() * pv).sum().item() / (pv.sum().item() + 1e-6))
    mean = lambda xs: sum(xs) / max(len(xs), 1)
    return mean(si_rmses), mean(inv_gaps), n


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--checkpoint', action='append', required=True,
                    help='Checkpoint .pth (repeatable — sweep several steps of one run).')
    ap.add_argument('--front3d-root', default=str(ROOT_DIR / '../datasets/front3d_iid'))
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--max-views', type=int, default=0, help='0 = all (~120) val views.')
    args = ap.parse_args()

    ds = Front3DDataset(root_dir=args.front3d_root, split='val', input_size=384)
    if len(ds) == 0:
        raise SystemExit(f'No val views under {args.front3d_root} — wrong root?')

    print(f'\n{"checkpoint":55s} {"step":>7s} {"alb_si_rmse":>12s} {"inv_gap":>9s} {"views":>6s}')
    for ck in args.checkpoint:
        model, step = load_model(ck, args.device)
        si, gap, n = probe(model, ds, args.device, args.max_views)
        print(f'{Path(ck).parent.name + "/" + Path(ck).name:55s} {str(step):>7s} '
              f'{si:12.4f} {gap:9.4f} {n:6d}')
        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
