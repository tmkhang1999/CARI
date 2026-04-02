"""Smoke tests for loss components."""

import sys
from pathlib import Path
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from losses.flexible_loss import FlexibleLoss


def test_flexible_loss_shapes():
    config = {
        'lambda_msg': 0.5,
        'lambda_tv': 0.1,
        'lambda_perceptual': 0.05,
        'lambda_dssim': 0.1,
    }
    loss_fn = FlexibleLoss(config)

    predictions = {
        'd_g': torch.rand(2, 1, 64, 64),
        'xi': torch.rand(2, 2, 64, 64),
        'c': torch.rand(2, 3, 64, 64),
        's_c': torch.rand(2, 3, 64, 64),
        'a_d': torch.rand(2, 3, 64, 64),
        'pi': torch.rand(2, 3, 64, 64),
    }

    # Scale-matched, target-space GTs (as computed in training loop)
    targets = {
        'D_g_star': torch.rand(2, 1, 64, 64),   # inverse gray shading
        'xi_star': torch.rand(2, 2, 64, 64),     # bounded chroma
        'A_d_star': torch.rand(2, 3, 64, 64),    # scale-matched albedo
        'pi_star': torch.rand(2, 3, 64, 64),     # inverse colorful shading
    }

    m_diffuse = torch.tensor([1.0, 0.0])
    m_albedo = torch.tensor([1.0, 1.0])
    valid_mask = torch.ones(2, 1, 64, 64).bool()
    seg = torch.randint(0, 40, (2, 64, 64))

    losses = loss_fn(predictions, targets, m_diffuse, m_albedo, valid_mask, seg)

    assert 'loss_total' in losses
    assert losses['loss_total'].shape == torch.Size([])
    assert losses['loss_total'].item() >= 0
    # loss_a/loss_b should be > 0; loss_c should be > 0 (m_albedo both 1)
    print(f"  loss_a={losses['loss_a'].item():.4f}")
    print(f"  loss_b={losses['loss_b'].item():.4f}")
    print(f"  loss_c={losses['loss_c'].item():.4f}")
    print(f"  loss_d={losses['loss_d'].item():.4f}")
    print(f"  total ={losses['loss_total'].item():.4f}")


def test_loss_routing_no_diffuse():
    """When m_diffuse=0 for all samples, only Dec D should be zeroed."""
    config = {'lambda_msg': 0.5, 'lambda_tv': 0.1,
              'lambda_perceptual': 0.05, 'lambda_dssim': 0.1}
    loss_fn = FlexibleLoss(config)

    preds = {k: torch.rand(2, c, 32, 32)
             for k, c in [('d_g', 1), ('xi', 2), ('c', 3),
                          ('s_c', 3), ('a_d', 3), ('pi', 3)]}
    targets = {
        'D_g_star': torch.rand(2, 1, 32, 32),
        'xi_star': torch.rand(2, 2, 32, 32),
        'A_d_star': torch.rand(2, 3, 32, 32),
        'pi_star': torch.rand(2, 3, 32, 32),
    }
    m_diffuse = torch.tensor([0.0, 0.0])
    m_albedo = torch.tensor([1.0, 1.0])
    valid = torch.ones(2, 1, 32, 32).bool()

    losses = loss_fn(preds, targets, m_diffuse, m_albedo, valid)
    assert losses['loss_a'].item() > 0.0
    assert losses['loss_b'].item() > 0.0
    assert losses['loss_d'].item() == 0.0
    assert losses['loss_c'].item() > 0.0


if __name__ == "__main__":
    test_flexible_loss_shapes()
    print("Loss shape test passed")
    test_loss_routing_no_diffuse()
    print("Loss routing test passed")
    print("All loss tests passed")
