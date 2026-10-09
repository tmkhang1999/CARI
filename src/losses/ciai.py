"""Cross-Illumination Albedo Invariance (CIAI): the paired training losses.

CIAI is a training strategy, not an architecture. A model is run twice, on two
pixel-aligned photographs of one scene under different illumination (I1, I2), and the
two predictions are scored here:

  albedo_invariance   the two predicted albedos must agree.
  luminance_explain   the log luminance ratio of the two shadings must equal that of
                      the two images, so the change in light lands in the shading and a
                      flat albedo cannot satisfy the invariance term.
  chroma_explain      the chromatic part of the same ratio. The luminance term is blind
                      to a pure colour change between frames; this term scores exactly
                      the component it discards.

Every function takes plain tensors, so any model that outputs an albedo and a linear
(positive) shading can be trained with CIAI. `pair_gate` builds the per-pixel mask
the terms are averaged over, and optionally drops pairs whose illuminant colour barely
changes, where a chroma term has nothing to explain and only fits noise.
"""

from __future__ import annotations

import torch


def _masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target).abs()
    return (diff * mask).sum() / (mask.sum() + 1e-7)


def _lum601(x: torch.Tensor) -> torch.Tensor:
    return (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]).clamp_min(0.0)


def pair_gate(loss_mask: torch.Tensor, pair_flag: torch.Tensor,
              pair_valid: torch.Tensor | None = None) -> torch.Tensor:
    """Per-pixel weight for the pair losses.

    loss_mask : (B,1,H,W) valid pixels of the primary frame.
    pair_flag : (B,) 1 for rows that carry a second illumination, 0 otherwise.
    pair_valid: (B,1,H,W) pixels valid in BOTH frames (not clipped, not near black).
    """
    mask = loss_mask.float() * pair_flag.float().view(-1, 1, 1, 1)
    if pair_valid is not None:
        mask = mask * pair_valid.float()
    return mask


def gap_gate(gap: torch.Tensor, threshold: float) -> torch.Tensor:
    """(B,) 1 where the measured illuminant chromaticity gap of a pair reaches `threshold`."""
    return (gap >= float(threshold)).float()


def albedo_invariance(a1: torch.Tensor, a2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked L1 between the albedos predicted from the two frames."""
    return _masked_l1(a1, a2, mask)


def luminance_explain(rgb1: torch.Tensor, rgb2: torch.Tensor, s1: torch.Tensor,
                      s2: torch.Tensor, mask: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """| log(L(I1)/L(I2)) - log(L(S1)/L(S2)) | over the mask (Rec. 601 luminance).

    s1, s2 are LINEAR shadings: 3-channel, or 1-channel if the model predicts shading
    luminance directly.
    """
    s1l = s1.clamp_min(0.0) if s1.shape[1] == 1 else _lum601(s1)
    s2l = s2.clamp_min(0.0) if s2.shape[1] == 1 else _lum601(s2)
    ri = torch.log(_lum601(rgb1) + eps) - torch.log(_lum601(rgb2) + eps)
    rs = torch.log(s1l + eps) - torch.log(s2l + eps)
    return _masked_l1(ri, rs, mask)


def chroma_explain(rgb1: torch.Tensor, rgb2: torch.Tensor, s1: torch.Tensor,
                   s2: torch.Tensor, mask: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Chromatic part of the log ratio: || c(r_I) - c(r_S) ||_1, c(r) = r - mean_c(r).

    Removing the per-pixel channel mean makes this disjoint from the achromatic change,
    so it does not double-count what `luminance_explain` already scores. Needs
    3-channel shading; a 1-channel shading has no colour to explain.
    """
    if s1.shape[1] != 3 or s2.shape[1] != 3:
        raise ValueError(
            f'chroma_explain needs 3-channel shading, got {s1.shape[1]} and {s2.shape[1]}. '
            'Set the chroma explanation weight to 0 for a grey-shading model.')

    def chroma_resid(x1, x2):
        r = torch.log(x1.clamp_min(0.0) + eps) - torch.log(x2.clamp_min(0.0) + eps)
        return r - r.mean(dim=1, keepdim=True)

    return _masked_l1(chroma_resid(rgb1, rgb2), chroma_resid(s1, s2), mask)
