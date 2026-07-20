"""V17 add-on albedo refiner.

This module deliberately leaves the V17 backbone/trunk/heads unchanged.  It
subclasses V17 only so checkpoint keys stay compatible with existing V17
checkpoints; the added parameters live under ``refiner.*`` and cold-start when
resuming a plain V17 checkpoint.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .v17 import IntrinsicDecompositionV17
from .decoders.dpt_decoder import _gn


def _luminance(rgb: torch.Tensor) -> torch.Tensor:
    return 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]


class LowResLogAlbedoRefiner(nn.Module):
    """Bounded low-resolution residual correction in log-albedo space.

    Inputs are base albedo and base shading only; raw RGB is intentionally not
    provided, because a full-resolution image path is the easiest way for a
    refiner to copy shadows back into albedo.
    """

    def __init__(
        self,
        hidden_ch: int = 48,
        num_blocks: int = 4,
        down: int = 8,
        beta: float = 0.10,
    ):
        super().__init__()
        self.down = max(1, int(down))
        self.beta = float(beta)

        in_ch = 7  # log A0 RGB (3) + log shading luminance (1) + shading chroma (3)
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, hidden_ch, 3, padding=1, bias=False),
            _gn(hidden_ch),
            nn.SiLU(inplace=True),
        ]
        for _ in range(max(0, int(num_blocks) - 1)):
            layers.extend([
                nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1, bias=False),
                _gn(hidden_ch),
                nn.SiLU(inplace=True),
            ])
        layers.append(nn.Conv2d(hidden_ch, 3, 1))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _features(self, a0: torch.Tensor, s0: torch.Tensor) -> torch.Tensor:
        eps = 1e-4
        log_a = torch.log(a0.clamp_min(eps)).clamp(-8.0, 0.0)
        s_lum = _luminance(s0).clamp_min(eps)
        log_s_lum = torch.log(s_lum).clamp(-6.0, 6.0)
        s_chroma = s0.clamp_min(eps) / s0.clamp_min(eps).sum(dim=1, keepdim=True).clamp_min(eps)
        return torch.cat([log_a, log_s_lum, s_chroma], dim=1)

    def forward(self, a0: torch.Tensor, s0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        H, W = a0.shape[-2:]
        feat = self._features(a0, s0)
        if self.down > 1:
            h = max(1, H // self.down)
            w = max(1, W // self.down)
            feat = F.interpolate(feat, size=(h, w), mode='area')

        delta_low = self.net(feat)
        delta = torch.tanh(delta_low)
        if delta.shape[-2:] != (H, W):
            delta = F.interpolate(delta, size=(H, W), mode='bilinear', align_corners=False)

        log_a1 = torch.log(a0.clamp_min(1e-4)) + self.beta * delta
        a1 = torch.exp(log_a1).clamp(1e-4, 1.0)
        return a1, delta


class IntrinsicDecompositionV17Refiner(IntrinsicDecompositionV17):
    """V17 plus a frozen-backbone low-res albedo residual refiner."""

    def __init__(self, config):
        super().__init__(config)
        ref_cfg = dict(config.get('refiner', {}))
        self.freeze_base = bool(ref_cfg.get('freeze_base', True))
        self.detach_refiner_inputs = bool(ref_cfg.get('detach_inputs', True))
        self.refiner = LowResLogAlbedoRefiner(
            hidden_ch=int(ref_cfg.get('hidden_ch', 48)),
            num_blocks=int(ref_cfg.get('num_blocks', 4)),
            down=int(ref_cfg.get('down', 8)),
            beta=float(ref_cfg.get('beta', 0.10)),
        )

        if self.freeze_base:
            for name, param in self.named_parameters():
                if not name.startswith('refiner.'):
                    param.requires_grad_(False)

    def _base_forward(self, rgb: torch.Tensor, **kwargs) -> dict:
        if self.freeze_base:
            with torch.no_grad():
                return super().forward(rgb, **kwargs)
        return super().forward(rgb, **kwargs)

    def forward(self, rgb: torch.Tensor, **kwargs):
        base = self._base_forward(rgb, **kwargs)
        a0 = base['a_d']
        s0 = base['shading_linear']
        a_in = a0.detach() if self.detach_refiner_inputs else a0
        s_in = s0.detach() if self.detach_refiner_inputs else s0

        a1, delta = self.refiner(a_in, s_in)
        diffuse = a1 * s0
        residual = (rgb - diffuse).clamp(min=0.0)

        out = dict(base)
        out['a_base'] = a0
        out['a_refine_delta'] = delta
        out['a_d'] = a1
        out['residual'] = residual
        out['rgb_reconstructed'] = diffuse + residual
        return out
