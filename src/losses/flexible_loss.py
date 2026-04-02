"""
Flexible supervised loss with routing masks.
Implements the complete loss formulation from project plan Section 3.

IMPORTANT: This loss expects **scale-matched, target-space** ground truths
that are computed in the training loop AFTER the forward pass.
See data_processing_supplement.md Phase 3.5 and Phase 6.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .msg_loss import MultiScaleGradientLoss
from .perceptual_loss import PerceptualLoss
from .semantic_tv_loss import SemanticTVLoss, NormalGuidedTVLoss


class FlexibleLoss(nn.Module):
    """
    Flexible loss that routes gradients based on available ground truth.
    Handles mixed datasets with different supervision signals.

    Expected inputs (computed in training loop after forward pass):
        D_g_star:  (N,1,H,W) inverse gray shading GT  = 1/(S_g_star+1)
        xi_star:   (N,2,H,W) bounded chroma GT
        A_d_star:  (N,3,H,W) scale-matched albedo GT
        pi_star:   (N,3,H,W) inverse colorful shading GT = 1/(S_star+1)
        valid_mask:(N,1,H,W) bool — excludes sky, mirrors, glass
    """

    def __init__(self, config):
        super().__init__()

        # Loss weights (plan Section 3.3)
        self.lambda_msg = config.get('lambda_msg', 0.5)
        self.lambda_tv = config.get('lambda_tv', 0.1)
        self.lambda_perceptual = config.get('lambda_perceptual', 0.05)
        self.lambda_dssim = config.get('lambda_dssim', 0.1)
        self.tv_type = self._normalize_tv_type(config.get('tv_type', 'segmentation'))
        self.tv_target_classes = config.get('tv_target_classes', [1, 22])

        # Loss modules
        self.msg_loss = MultiScaleGradientLoss(num_scales=4)
        self.perceptual_loss = PerceptualLoss()
        self.semantic_tv_loss = SemanticTVLoss(target_classes=self.tv_target_classes)
        self.normal_tv_loss = NormalGuidedTVLoss()

        # Cache Gaussian DSSIM windows by channel/device/dtype/size.
        self._dssim_window_cache = {}

    @staticmethod
    def _normalize_tv_type(tv_type):
        """Normalize TV mode. Supports disabling via None/'none'/'off'."""
        if tv_type is None:
            return None
        if isinstance(tv_type, str):
            mode = tv_type.strip().lower()
            if mode in ('none', 'off', ''):
                return None
            if mode in ('segmentation', 'normals'):
                return mode
        return tv_type

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def _masked_l1(self, pred, target, mask):
        """Element-wise L1 masked by valid * routing mask."""
        diff = F.l1_loss(pred, target, reduction='none')
        return (diff * mask).sum() / (mask.sum() + 1e-7)

    def _masked_mse(self, pred, target, mask):
        """Element-wise MSE masked by valid * routing mask."""
        diff = F.mse_loss(pred, target, reduction='none')
        return (diff * mask).sum() / (mask.sum() + 1e-7)

    def _masked_msg(self, pred, target, mask_ratio):
        """MSG loss scaled by the fraction of valid samples."""
        return self.msg_loss(pred, target) * mask_ratio

    def _get_dssim_window(self, channels, device, dtype, window_size):
        key = (int(channels), str(device), str(dtype), int(window_size))
        window = self._dssim_window_cache.get(key)
        if window is not None:
            return window

        sigma = 1.5
        coords = torch.arange(window_size, dtype=dtype, device=device)
        gauss = torch.exp(-((coords - window_size // 2) ** 2) / (2.0 * sigma ** 2))
        gauss = gauss / gauss.sum()

        window_1d = gauss.unsqueeze(1)
        window_2d = (window_1d @ window_1d.t()).unsqueeze(0).unsqueeze(0)
        window = window_2d.expand(channels, 1, window_size, window_size).contiguous()
        self._dssim_window_cache[key] = window
        return window

    def _compute_dssim(self, pred, target, window_size=11):
        """DSSIM = (1 - SSIM) / 2."""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        C = pred.shape[1]
        window = self._get_dssim_window(C, pred.device, pred.dtype, window_size)

        pad = window_size // 2
        mu1 = F.conv2d(pred, window, padding=pad, groups=C)
        mu2 = F.conv2d(target, window, padding=pad, groups=C)

        mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
        sigma1_sq = F.conv2d(pred * pred, window, padding=pad, groups=C) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=pad, groups=C) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=pad, groups=C) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ((1.0 - ssim_map) / 2.0).mean()

    # ------------------------------------------------------------------
    # Per-decoder losses
    # ------------------------------------------------------------------
    def loss_a(self, D_g_pred, D_g_star, valid_mask):
        """
        Dec A loss: Gray shading in inverse space.
        Always active on valid pixels when pseudo-GT albedo is available.
        """
        mask = valid_mask.float()
        l1 = self._masked_l1(D_g_pred, D_g_star, mask)
        # MSG is global by construction; weight by valid-pixel fraction.
        msg = self.msg_loss(D_g_pred, D_g_star) * mask.mean()
        return l1 + self.lambda_msg * msg

    def loss_b(self, xi_pred, xi_star, valid_mask):
        """
        Dec B loss: Chroma in bounded ratio space.
        Always active on valid pixels when pseudo-GT albedo is available.
        """
        mask = valid_mask.float()

        mse = self._masked_mse(xi_pred, xi_star, mask)
        # Match Dec A behavior so invalid regions do not drive MSG gradients.
        msg = self.msg_loss(xi_pred, xi_star) * mask.mean()
        return mse + self.lambda_msg * msg

    def loss_c(self, a_d_pred, A_d_star, valid_mask, m_albedo, seg_map=None, normals=None):
        """Dec C total loss (kept for backward compatibility)."""
        total, _ = self.loss_c_with_details(
            a_d_pred,
            A_d_star,
            valid_mask,
            m_albedo,
            seg_map,
            normals,
        )
        return total

    def loss_c_with_details(self, a_d_pred, A_d_star, valid_mask, m_albedo, seg_map=None, normals=None):
        """Dec C loss with per-term details for logging/debugging."""
        zero = a_d_pred.new_tensor(0.0)
        if m_albedo.sum() == 0:
            return zero, {
                'loss_c_l1': zero,
                'loss_c_msg': zero,
                'loss_c_tv': zero,
                'loss_c_perceptual': zero,
                'loss_c_dssim': zero,
                'loss_c_mask_ratio': zero,
            }

        route = m_albedo.view(-1, 1, 1, 1).to(valid_mask.device)
        mask = valid_mask.float() * route
        mask_ratio = (m_albedo.sum() / m_albedo.numel()).to(a_d_pred.dtype)

        l1 = self._masked_l1(a_d_pred, A_d_star, mask)
        msg = self._masked_msg(a_d_pred, A_d_star, mask_ratio)

        if self.tv_type is None:
            tv = zero
        elif self.tv_type == 'normals' and normals is not None:
            tv = self.normal_tv_loss(a_d_pred, normals, valid_mask=mask)
        else:
            tv = self.semantic_tv_loss(a_d_pred, seg_map) if seg_map is not None else zero

        perceptual = self.perceptual_loss(a_d_pred, A_d_star)
        dssim = self._compute_dssim(a_d_pred, A_d_star)

        total = mask_ratio * (
            l1
            + self.lambda_msg * msg
            + self.lambda_tv * tv
            + self.lambda_perceptual * perceptual
            + self.lambda_dssim * dssim
        )

        details = {
            'loss_c_l1': l1.detach(),
            'loss_c_msg': msg.detach(),
            'loss_c_tv': tv.detach(),
            'loss_c_perceptual': perceptual.detach(),
            'loss_c_dssim': dssim.detach(),
            'loss_c_mask_ratio': mask_ratio.detach(),
        }
        return total, details

    def loss_d(self, pi_pred, pi_star, valid_mask, m_diffuse):
        """
        Dec D loss: Diffuse shading in inverse space.
        L_D = M_diffuse * ( ||π − π*||₂² + λ_msg * L_MSG(π, π*) )
        Both terms are applied directly on inverse shading tensors (pi_pred, pi_star)
        with no additional conversion.
        No reconstruction loss — avoids absorbing specular residual R.
        """
        if m_diffuse.sum() == 0:
            return torch.tensor(0.0, device=pi_pred.device)

        route = m_diffuse.view(-1, 1, 1, 1).to(valid_mask.device)
        mask = valid_mask.float() * route
        mask_ratio = m_diffuse.sum() / m_diffuse.numel()

        mse = self._masked_mse(pi_pred, pi_star, mask)
        msg = self._masked_msg(pi_pred, pi_star, mask_ratio)
        return mse + self.lambda_msg * msg

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def forward(self, predictions, targets, m_diffuse, m_albedo,
                valid_mask, seg_map=None, normals=None):
        """
        Args:
            predictions: dict with keys [d_g, xi, c, s_c, a_d, pi]
            targets: dict with keys [D_g_star, xi_star, A_d_star, pi_star]
                     computed in training loop via scale_match → target-space conversion.
            m_diffuse: (N,) float routing mask for diffuse shading GT availability.
            m_albedo:  (N,) float routing mask for albedo GT availability.
            valid_mask: (N, 1, H, W) bool pixel mask (sky, mirrors excluded).
            seg_map: (N, H, W) integer seg labels, optional.

        Returns:
            dict with individual losses and total loss.
        """
        zero = predictions['d_g'].new_tensor(0.0)

        la = self.loss_a(predictions['d_g'], targets['D_g_star'], valid_mask) \
            if targets.get('D_g_star') is not None else zero

        lb = self.loss_b(predictions['xi'], targets['xi_star'], valid_mask) \
            if targets.get('xi_star') is not None else zero

        if targets.get('A_d_star') is not None:
            lc, lc_details = self.loss_c_with_details(
                predictions['a_d'],
                targets['A_d_star'],
                valid_mask,
                m_albedo,
                seg_map,
                normals,
            )
        else:
            lc = zero
            lc_details = {
                'loss_c_l1': zero,
                'loss_c_msg': zero,
                'loss_c_tv': zero,
                'loss_c_perceptual': zero,
                'loss_c_dssim': zero,
                'loss_c_mask_ratio': zero,
            }

        ld = self.loss_d(predictions['pi'], targets['pi_star'],
                         valid_mask, m_diffuse) \
            if targets.get('pi_star') is not None else zero

        out = {
            'loss_a': la,
            'loss_b': lb,
            'loss_c': lc,
            'loss_d': ld,
            'loss_total': la + lb + lc + ld,
        }
        out.update(lc_details)
        return out


if __name__ == '__main__':
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
    targets = {
        'D_g_star': torch.rand(2, 1, 64, 64),
        'xi_star': torch.rand(2, 2, 64, 64),
        'A_d_star': torch.rand(2, 3, 64, 64),
        'pi_star': torch.rand(2, 3, 64, 64),
    }
    m_diffuse = torch.tensor([1.0, 0.0])
    m_albedo = torch.tensor([1.0, 1.0])
    valid_mask = torch.ones(2, 1, 64, 64).bool()
    seg = torch.randint(0, 40, (2, 64, 64))

    losses = loss_fn(predictions, targets, m_diffuse, m_albedo, valid_mask, seg)
    print("Losses:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")
