"""Losses for the V21 tri-factor model: I = A * (S_lum * C) + R, Y(C) = 1.

Single-image terms supervise the albedo, the shading luminance S_lum and the shading
chroma C wherever ground truth exists. The pair terms are the CIAI losses of
losses/ciai.py, applied to the full shading S_lum * C:

  invariance   the albedos of the two frames agree
  lum_explain  the luminance ratio of the shadings matches that of the images
  chr_explain  the chromatic part of the ratio matches too; applied only to pairs whose
               measured illuminant gap reaches `chr_explain_min_gap`, because on pairs
               with no real colour change it fits noise and drains the albedo's colour
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import ciai


def _expand_mask(mask: torch.Tensor, value: torch.Tensor, gate: torch.Tensor | None = None) -> torch.Tensor:
    mask = mask.to(value.dtype)
    if gate is not None:
        mask = mask * gate.to(value.dtype).view(-1, 1, 1, 1)
    return mask.expand_as(value)


def masked_charbonnier(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    gate: torch.Tensor | None = None,
    eps: float = 1e-3,
) -> torch.Tensor:
    expanded = _expand_mask(mask, prediction, gate)
    error = torch.sqrt((prediction - target).square() + eps * eps)
    return (error * expanded).sum() / expanded.sum().clamp_min(1.0)


def gradient(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dx = tensor[..., :, 1:] - tensor[..., :, :-1]
    dy = tensor[..., 1:, :] - tensor[..., :-1, :]
    return dx, dy


def gradient_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    gate: torch.Tensor | None = None,
    scales: tuple[int, ...] = (1, 2, 4),
) -> torch.Tensor:
    total = prediction.new_zeros(())
    for scale in scales:
        if scale > 1:
            pred = F.avg_pool2d(prediction, scale, scale)
            tgt = F.avg_pool2d(target, scale, scale)
            current_mask = F.interpolate(mask.float(), pred.shape[-2:], mode="nearest").bool()
        else:
            pred, tgt, current_mask = prediction, target, mask
        pred_x, pred_y = gradient(pred)
        tgt_x, tgt_y = gradient(tgt)
        mask_x = current_mask[..., :, 1:] & current_mask[..., :, :-1]
        mask_y = current_mask[..., 1:, :] & current_mask[..., :-1, :]
        total = total + masked_charbonnier(pred_x, tgt_x, mask_x, gate)
        total = total + masked_charbonnier(pred_y, tgt_y, mask_y, gate)
    return total / (2.0 * len(scales))


def scale_align_log(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_log = torch.log(prediction.clamp_min(1e-5))
    target_log = torch.log(target.clamp_min(1e-5))
    expanded = mask.to(pred_log.dtype).expand_as(pred_log)
    count = expanded.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1.0)
    offset = ((target_log - pred_log) * expanded).sum(dim=(1, 2, 3), keepdim=True) / count
    return pred_log + offset, target_log


def ordinal_loss(
    pred_log: torch.Tensor,
    target_log: torch.Tensor,
    mask: torch.Tensor,
    gate: torch.Tensor,
    offsets: tuple[int, ...] = (4, 16, 64),
) -> torch.Tensor:
    total = pred_log.new_zeros(())
    terms = 0
    for offset in offsets:
        if offset >= min(pred_log.shape[-2:]):
            continue
        for axis in (-1, -2):
            if axis == -1:
                p0, p1 = pred_log[..., :, :-offset], pred_log[..., :, offset:]
                t0, t1 = target_log[..., :, :-offset], target_log[..., :, offset:]
                valid = mask[..., :, :-offset] & mask[..., :, offset:]
            else:
                p0, p1 = pred_log[..., :-offset, :], pred_log[..., offset:, :]
                t0, t1 = target_log[..., :-offset, :], target_log[..., offset:, :]
                valid = mask[..., :-offset, :] & mask[..., offset:, :]
            target_delta = (t1 - t0).clamp(-2.0, 2.0)
            pred_delta = (p1 - p0).clamp(-2.0, 2.0)
            total = total + masked_charbonnier(pred_delta, target_delta, valid, gate)
            terms += 1
    return total / max(terms, 1)


def ssim_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mu_x = F.avg_pool2d(prediction, 3, 1, 1)
    mu_y = F.avg_pool2d(target, 3, 1, 1)
    var_x = F.avg_pool2d(prediction.square(), 3, 1, 1) - mu_x.square()
    var_y = F.avg_pool2d(target.square(), 3, 1, 1) - mu_y.square()
    covariance = F.avg_pool2d(prediction * target, 3, 1, 1) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mu_x * mu_y + c1) * (2 * covariance + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (var_x + var_y + c2) + 1e-8
    )
    expanded = mask.float().expand_as(score)
    return (((1.0 - score.clamp(-1.0, 1.0)) * 0.5) * expanded).sum() / expanded.sum().clamp_min(1.0)


class V21Loss(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.weights = {key: float(value) for key, value in config.items() if key.startswith("lambda_")}
        self.chr_explain_min_gap = float(config.get("chr_explain_min_gap", 0.08))

    def weight(self, name: str) -> float:
        return self.weights.get(f"lambda_{name}", 0.0)

    def forward(
        self,
        output: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        pair_output: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mask = batch["loss_mask"]
        factor_gate = batch["factor_supervision"]
        residual_gate = batch["residual_supervision"]
        losses: dict[str, torch.Tensor] = {}

        losses["albedo"] = masked_charbonnier(output["a_d"], batch["albedo_gt"], mask)
        losses["albedo_grad"] = gradient_loss(output["a_d"], batch["albedo_gt"], mask)
        losses["albedo_ssim"] = ssim_loss(output["a_d"], batch["albedo_gt"], mask)

        pred_log_lum, target_log_lum = scale_align_log(
            output["shading_luminance"], batch["shading_lum_gt"], mask
        )
        losses["shading_lum"] = masked_charbonnier(
            pred_log_lum, target_log_lum, mask, factor_gate
        )
        losses["shading_lum_grad"] = gradient_loss(
            pred_log_lum, target_log_lum, mask, factor_gate
        )
        losses["shading_chroma"] = masked_charbonnier(
            output["shading_uv"], batch["shading_uv_gt"], mask, factor_gate
        )
        losses["shading_chroma_grad"] = gradient_loss(
            output["shading_uv"], batch["shading_uv_gt"], mask, factor_gate
        )
        losses["ordinal"] = ordinal_loss(
            pred_log_lum, target_log_lum, mask, factor_gate
        )
        losses["diffuse"] = masked_charbonnier(
            output["diffuse"], batch["diffuse_gt"], mask, factor_gate
        )
        losses["residual"] = masked_charbonnier(
            output["residual"], batch["residual_gt"], mask, residual_gate
        )

        albedo_dx, albedo_dy = gradient(torch.log(output["a_d"].clamp_min(1e-4)))
        edge_x = batch["shadow_edge"][..., :, 1:] * mask[..., :, 1:].float()
        edge_y = batch["shadow_edge"][..., 1:, :] * mask[..., 1:, :].float()
        losses["edge_ownership"] = (
            (albedo_dx.abs() * edge_x).sum() / edge_x.expand_as(albedo_dx).sum().clamp_min(1.0)
            + (albedo_dy.abs() * edge_y).sum() / edge_y.expand_as(albedo_dy).sum().clamp_min(1.0)
        ) * 0.5

        zero = output["a_d"].new_zeros(())
        losses["invariance"] = zero
        losses["lum_explain"] = zero
        losses["chr_explain"] = zero
        if pair_output is not None:
            pair_mask = ciai.pair_gate(mask, batch["pair_supervision"], batch["pair_valid"])
            s1, s2 = output["shading_linear"], pair_output["shading_linear"]
            losses["invariance"] = ciai.albedo_invariance(output["a_d"], pair_output["a_d"], pair_mask)
            losses["lum_explain"] = ciai.luminance_explain(batch["rgb"], batch["rgb2"], s1, s2, pair_mask)
            if self.weight("chr_explain") > 0:
                gate = ciai.gap_gate(batch["pair_gap"], self.chr_explain_min_gap)
                losses["chr_explain"] = ciai.chroma_explain(
                    batch["rgb"], batch["rgb2"], s1, s2, pair_mask * gate.view(-1, 1, 1, 1))

        total = sum(self.weight(name) * value for name, value in losses.items())
        return total, losses
