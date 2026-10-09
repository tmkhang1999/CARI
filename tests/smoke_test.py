"""Smoke test without data or pretrained weights: one training step for every V17 config and
one V21 loss evaluation, on random tensors with a small, randomly initialised DINOv2.

    python tests/smoke_test.py

Checks that each config builds, every loss is finite, the CIAI pair terms fire on paired
rows, gradients reach the trainable weights, and the V21 chroma term respects its gap gate.
Runs on CPU in about a minute.
"""

import sys
import warnings
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
warnings.filterwarnings('ignore')

import train_v17 as tv  # noqa: E402
from losses.v17_loss import V17Loss  # noqa: E402
from losses.v21_loss import V21Loss  # noqa: E402
from models import IntrinsicDecompositionV21  # noqa: E402

V17_CONFIGS = sorted(p.stem[1:] for p in (ROOT / 'src/configs').glob('v17_*.yaml'))


def v17_batch(B=3, H=112, W=112):
    g = torch.Generator().manual_seed(1)
    rgb = torch.rand(B, 3, H, W, generator=g) * 0.8 + 0.05
    albedo = torch.rand(B, 3, H, W, generator=g) * 0.9 + 0.05
    return {
        'rgb': rgb, 'albedo_raw': albedo, 'illum_raw': (rgb / albedo).clamp(0.0, 20.0),
        'rgb2': (rgb * (0.7 + 0.6 * torch.rand(B, 3, 1, 1, generator=g))).clamp(0.0, 1.0),
        'loss_mask': torch.ones(B, 1, H, W), 'pair_valid': torch.ones(B, 1, H, W),
        'M_diffuse': torch.tensor([1.0, 0.0, 0.0]), 'm_residual': torch.ones(B),
        'm_invariant': torch.tensor([0.0, 1.0, 1.0]),
    }


def check_v17():
    for ref in V17_CONFIGS:
        cfg = tv.load_config(None, ref)
        cfg['model'].update(dino_variant='small', dino_pretrained=False)
        torch.manual_seed(0)
        model = tv.build_model(cfg)
        criterion = V17Loss(cfg['loss'])
        losses = tv.train_one_step(model, v17_batch(), criterion, 'cpu', 25000, 3000)
        assert all(torch.isfinite(v).all() for v in losses.values()), f'{ref}: non-finite loss'
        if criterion.lambda_alb_invariance > 0:
            assert 'loss_alb_invariance' in losses, f'{ref}: invariance term did not fire'
        grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        assert grads and sum(float(g.abs().sum()) for g in grads) > 0, f'{ref}: no gradient'
        print(f'  v{ref}: total {float(losses["loss_total"]):.4f}')


def check_v21():
    cfg = yaml.safe_load(open(ROOT / 'src/configs/v21.yaml'))
    cfg['model'].update(dino_variant='small', dino_pretrained=False)
    torch.manual_seed(0)
    model = IntrinsicDecompositionV21(cfg['model'])
    criterion = V21Loss(cfg['loss'])
    B, H, W = 3, 112, 112
    g = torch.Generator().manual_seed(1)
    rgb = torch.rand(B, 3, H, W, generator=g) * 0.8 + 0.05
    alb = torch.rand(B, 3, H, W, generator=g) * 0.9 + 0.05
    sh = rgb / alb
    batch = {
        'rgb': rgb, 'rgb2': (rgb * (0.7 + 0.6 * torch.rand(B, 3, 1, 1, generator=g))).clamp(0, 1),
        'albedo_gt': alb, 'shading_lum_gt': 0.2126 * sh[:, 0:1] + 0.7152 * sh[:, 1:2] + 0.0722 * sh[:, 2:3],
        'shading_uv_gt': torch.zeros(B, 2, H, W), 'diffuse_gt': alb * sh,
        'residual_gt': torch.zeros(B, 3, H, W), 'shadow_edge': torch.zeros(B, 1, H, W),
        'loss_mask': torch.ones(B, 1, H, W, dtype=torch.bool),
        'pair_valid': torch.ones(B, 1, H, W, dtype=torch.bool),
        'pair_supervision': torch.tensor([0.0, 1.0, 1.0]),
        'factor_supervision': torch.tensor([1.0, 0.0, 1.0]),
        'residual_supervision': torch.ones(B),
    }
    for gaps, expect_chroma in (([0.0, 0.2, 0.01], True), ([0.0, -1.0, 0.01], False)):
        batch['pair_gap'] = torch.tensor(gaps)
        model.zero_grad()
        total, parts = criterion(model(batch['rgb']), batch, model(batch['rgb2']))
        total.backward()
        assert torch.isfinite(total), 'V21: non-finite loss'
        assert (float(parts['chr_explain']) > 0) == expect_chroma, 'V21: gap gate misbehaves'
    print(f'  v21: total {float(total):.4f}, gap gate OK')


if __name__ == '__main__':
    check_v17()
    check_v21()
    print('smoke test passed')
