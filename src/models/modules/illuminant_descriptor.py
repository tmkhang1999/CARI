"""Illuminant descriptor producing FiLM gamma/beta from masked log-chroma stats."""

import torch
import torch.nn as nn


class IlluminantDescriptor(nn.Module):
    def __init__(self, in_dim=4, hidden_dim=128, out_dim=None, z_channels=None):
        super().__init__()
        # Backward compatibility: older callers passed z_channels and expected
        # FiLM output to be [gamma, beta] with 2 * z_channels channels.
        if out_dim is None:
            out_dim = 2 * int(z_channels) if z_channels is not None else 2048
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def _masked_mean_std(self, x, mask, eps=1e-6):
        mask_sum = mask.sum(dim=(-1, -2), keepdim=True).clamp_min(eps)
        mean = (x * mask).sum(dim=(-1, -2), keepdim=True) / mask_sum
        var = ((x - mean) ** 2 * mask).sum(dim=(-1, -2), keepdim=True) / mask_sum
        std = torch.sqrt(var.clamp_min(eps))
        return mean.squeeze(-1).squeeze(-1), std.squeeze(-1).squeeze(-1)

    def forward(self, rgb, valid_mask=None):
        eps = 1e-6
        if valid_mask is None:
            valid_mask = torch.ones_like(rgb[:, :1])
        if valid_mask.dim() == 3:
            valid_mask = valid_mask.unsqueeze(1)
        mask = (valid_mask > 0.5).float()

        log_rgb = torch.log(rgb.clamp_min(eps))
        log_rg = log_rgb[:, 0:1] - log_rgb[:, 1:2]
        log_bg = log_rgb[:, 2:3] - log_rgb[:, 1:2]

        mean_rg, std_rg = self._masked_mean_std(log_rg, mask)
        mean_bg, std_bg = self._masked_mean_std(log_bg, mask)
        stats = torch.cat([mean_rg, mean_bg, std_rg, std_bg], dim=1)
        stats = torch.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)

        film = self.mlp(stats)
        gamma, beta = torch.chunk(film, 2, dim=1)
        return gamma.unsqueeze(-1).unsqueeze(-1), beta.unsqueeze(-1).unsqueeze(-1)
