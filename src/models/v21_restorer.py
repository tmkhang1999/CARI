"""Deterministic noise-free albedo flow and pixel detail adapter for V21.

The module starts from the frozen base albedo instead of Gaussian noise. It is
therefore deterministic and remains tied to the measured decomposition. The
flow handles low/mid-frequency completion; the adapter can copy bounded detail
from the canonical image only where the base observability mask permits it.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoders.dpt_decoder import _gn


class TimeEmbedding(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.project = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.SiLU(),
            nn.Linear(channels * 2, channels),
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half = self.channels // 2
        frequency = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, device=time.device, dtype=time.dtype)
            / max(half - 1, 1)
        )
        phase = time[:, None] * frequency[None, :]
        embedding = torch.cat((phase.sin(), phase.cos()), dim=1)
        if embedding.shape[1] < self.channels:
            embedding = F.pad(embedding, (0, self.channels - embedding.shape[1]))
        return self.project(embedding)


class ConditionedBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm1 = _gn(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = _gn(out_channels)
        self.time = nn.Linear(time_channels, out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, tensor: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.norm1(self.conv1(tensor)))
        hidden = hidden + self.time(time)[:, :, None, None]
        hidden = self.norm2(self.conv2(F.silu(hidden)))
        return F.silu(hidden + self.skip(tensor))


class NoiseFreeFlow(nn.Module):
    def __init__(self, width: int = 48, time_channels: int = 128) -> None:
        super().__init__()
        self.time_embedding = TimeEmbedding(time_channels)
        self.enc0 = ConditionedBlock(13, width, time_channels)
        self.down1 = nn.Conv2d(width, width * 2, 4, stride=2, padding=1)
        self.enc1 = ConditionedBlock(width * 2, width * 2, time_channels)
        self.down2 = nn.Conv2d(width * 2, width * 4, 4, stride=2, padding=1)
        self.middle = nn.ModuleList(
            [ConditionedBlock(width * 4, width * 4, time_channels) for _ in range(3)]
        )
        self.dec1 = ConditionedBlock(width * 6, width * 2, time_channels)
        self.dec0 = ConditionedBlock(width * 3, width, time_channels)
        self.out = nn.Conv2d(width, 3, 3, padding=1)

    def forward(
        self,
        state: torch.Tensor,
        rgb: torch.Tensor,
        base_albedo: torch.Tensor,
        shading: torch.Tensor,
        observable: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        time_feature = self.time_embedding(time)
        tensor = torch.cat((state, rgb, base_albedo, shading, observable), dim=1)
        level0 = self.enc0(tensor, time_feature)
        level1 = self.enc1(self.down1(level0), time_feature)
        hidden = self.down2(level1)
        for block in self.middle:
            hidden = block(hidden, time_feature)
        hidden = F.interpolate(hidden, level1.shape[-2:], mode="bilinear", align_corners=False)
        hidden = self.dec1(torch.cat((hidden, level1), dim=1), time_feature)
        hidden = F.interpolate(hidden, level0.shape[-2:], mode="bilinear", align_corners=False)
        hidden = self.dec0(torch.cat((hidden, level0), dim=1), time_feature)
        return torch.tanh(self.out(hidden))


class PixelDetailAdapter(nn.Module):
    def __init__(self, width: int = 32, max_delta: float = 0.10) -> None:
        super().__init__()
        self.max_delta = float(max_delta)
        self.net = nn.Sequential(
            nn.Conv2d(10, width, 3, padding=1), _gn(width), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1), _gn(width), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1), _gn(width), nn.SiLU(),
            nn.Conv2d(width, 3, 1),
        )

    def forward(
        self,
        rgb: torch.Tensor,
        canonical: torch.Tensor,
        prior: torch.Tensor,
        observable: torch.Tensor,
    ) -> torch.Tensor:
        detail = torch.cat((rgb, canonical, prior, observable), dim=1)
        return self.max_delta * torch.tanh(self.net(detail)) * observable


class V21AlbedoRestorer(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.minimum_prior_gate = float(config.get("minimum_prior_gate", 0.25))
        self.inference_steps = int(config.get("inference_steps", 4))
        self.flow = NoiseFreeFlow(width=int(config.get("flow_width", 48)))
        self.detail = PixelDetailAdapter(
            width=int(config.get("adapter_width", 32)),
            max_delta=float(config.get("adapter_max_delta", 0.10)),
        )

    def prior_gate(self, observable: torch.Tensor) -> torch.Tensor:
        return self.minimum_prior_gate + (1.0 - self.minimum_prior_gate) * (1.0 - observable)

    def training_forward(
        self,
        rgb: torch.Tensor,
        base_output: dict[str, torch.Tensor],
        target_albedo: torch.Tensor,
        time: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        base = base_output["a_d"].detach()
        shading = base_output["shading_linear"].detach()
        observable = base_output["observable"].detach()
        canonical = base_output.get("canonical_linear", base_output["canonical"]).detach().clamp(0.0, 1.0)
        gate = self.prior_gate(observable)
        target_prior = base + gate * (target_albedo - base)
        if time is None:
            time = torch.rand(base.shape[0], device=base.device, dtype=base.dtype)
        state = base + time[:, None, None, None] * (target_prior - base)
        target_velocity = target_prior - base
        velocity = self.flow(state, rgb, base, shading, observable, time)
        prior = (base + velocity).clamp(0.0, 1.0)
        delta = self.detail(rgb, canonical, prior, observable)
        final = (prior + delta).clamp(0.0, 1.0)
        return {
            "albedo": final,
            "prior": prior,
            "detail_delta": delta,
            "velocity": velocity,
            "target_velocity": target_velocity,
            "observable": observable,
            "uncertainty": 1.0 - observable,
            "shading_linear": shading,
        }

    @torch.no_grad()
    def forward(
        self,
        rgb: torch.Tensor,
        base_output: dict[str, torch.Tensor],
        steps: int | None = None,
    ) -> dict[str, torch.Tensor]:
        base = base_output["a_d"]
        shading = base_output["shading_linear"]
        observable = base_output["observable"]
        canonical = base_output.get("canonical_linear", base_output["canonical"]).clamp(0.0, 1.0)
        state = base.clone()
        count = max(1, int(steps or self.inference_steps))
        dt = 1.0 / count
        for index in range(count):
            time = rgb.new_full((rgb.shape[0],), index / count)
            velocity = self.flow(state, rgb, base, shading, observable, time)
            state = (state + dt * velocity).clamp(0.0, 1.0)
        delta = self.detail(rgb, canonical, state, observable)
        return {
            "albedo": (state + delta).clamp(0.0, 1.0),
            "prior": state,
            "detail_delta": delta,
            "observable": observable,
            "uncertainty": 1.0 - observable,
        }
