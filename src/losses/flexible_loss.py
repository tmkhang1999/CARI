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

    Expected V11 inputs (computed in training loop after forward pass):
        A_d_star:  (N,3,H,W) scale-matched albedo GT
        pi_star:   (N,3,H,W) inverse diffuse shading GT = 1/(S_star+1)
        valid_mask:(N,1,H,W) bool — excludes sky, mirrors, glass
    """

    def __init__(self, config):
        super().__init__()

        # Branch weights
        self.w_a = config.get('lambda_branch_a', 1.0)
        self.w_b = config.get('lambda_branch_b', 1.0)
        self.w_c = config.get('lambda_branch_c', 1.0)
        self.w_d = config.get('lambda_branch_d', 1.0)

        # Loss weights (plan Section 3.3)
        self.lambda_msg = config.get('lambda_msg', 0.8)
        self.lambda_tv = config.get('lambda_tv', 0.05)
        self.lambda_perceptual = config.get('lambda_perceptual', 0.05)
        self.lambda_dssim = config.get('lambda_dssim', 0.4)
        self.lambda_semvar = config.get('lambda_semvar', 0.0)
        self.lambda_recon = config.get('lambda_recon', 0.2)
        self.lambda_recon_color = config.get('lambda_recon_color', 1.0)
        self.enable_dssim_a_d = config.get('enable_dssim_a_d', True)
        self.tv_type = self._normalize_tv_type(config.get('tv_type', 'segmentation'))
        self.tv_target_classes = config.get('tv_target_classes', [1, 22])
        self.semantic_var_classes = config.get('semantic_var_classes', [1, 2, 22])

        # Loss modules
        self.msg_loss = MultiScaleGradientLoss(num_scales=4)
        self.perceptual_loss = PerceptualLoss()
        self.semantic_tv_loss = SemanticTVLoss(target_classes=self.tv_target_classes)
        self.normal_tv_loss = NormalGuidedTVLoss()

        # Base Gaussian DSSIM window registered as buffer for safe device moves.
        window_size = 11
        sigma = 1.5
        coords = torch.arange(window_size, dtype=torch.float32)
        gauss = torch.exp(-((coords - window_size // 2) ** 2) / (2.0 * sigma ** 2))
        gauss = gauss / gauss.sum()
        window_2d = (gauss.unsqueeze(1) @ gauss.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
        self.register_buffer('dssim_base_window', window_2d, persistent=False)

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

    def _get_dssim_window(self, channels):
        return self.dssim_base_window.expand(channels, 1, -1, -1).contiguous()

    def _compute_dssim(self, pred, target, window_size=11):
        """DSSIM = (1 - SSIM) / 2."""
        pred = pred.to(torch.float32)
        target = target.to(torch.float32)
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        C = pred.shape[1]
        window = self._get_dssim_window(C).to(device=pred.device, dtype=pred.dtype)

        pad = window_size // 2
        mu1 = F.conv2d(pred, window, padding=pad, groups=C)
        mu2 = F.conv2d(target, window, padding=pad, groups=C)

        mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
        sigma1_sq = F.conv2d(pred * pred, window, padding=pad, groups=C) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=pad, groups=C) - mu2_sq
        sigma1_sq = sigma1_sq.clamp_min(0.0)
        sigma2_sq = sigma2_sq.clamp_min(0.0)
        sigma12 = F.conv2d(pred * target, window, padding=pad, groups=C) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ((1.0 - ssim_map) / 2.0).mean()

    # ------------------------------------------------------------------
    # Per-decoder losses
    # ------------------------------------------------------------------

    def loss_reconstruction_sc(self, a_d_pred, d_g_pred, xi_pred, rgb, valid_mask, m_albedo):
        if m_albedo.sum() == 0:
            return (a_d_pred * 0.0).sum(), {}
            
        route = m_albedo.view(-1, 1, 1, 1).to(valid_mask.device)
        mask = valid_mask.float() * route

        # Reconstruct S_c from Dec A and Dec B
        s_g = (1.0 / (d_g_pred.clamp(1e-6) + 1e-6)) - 1.0
        c_rg = (1.0 - xi_pred[:, 0:1]) / (xi_pred[:, 0:1].clamp(1e-6) + 1e-6)
        c_bg = (1.0 - xi_pred[:, 1:2]) / (xi_pred[:, 1:2].clamp(1e-6) + 1e-6)
        
        # Invert the luminance formula: S_g = 0.2126 * S_R + 0.7152 * S_G + 0.0722 * S_B
        denom = (0.2126 * c_rg + 0.7152 + 0.0722 * c_bg).clamp(1e-6)
        s_green = s_g / denom
        
        c = torch.cat([c_rg, torch.ones_like(c_rg), c_bg], dim=1)
        s_c_linear = (s_green * c).clamp(0.0, 20.0)
        
        recon = a_d_pred * s_c_linear
        
        l1 = self._masked_l1(recon, rgb, mask)
        return l1, {'loss_recon_l1': l1.detach()}

    def loss_a(self, D_g_pred, D_g_star, valid_mask):
        """
        Dec A loss: Gray shading in inverse space.
        Always active on valid pixels when pseudo-GT albedo is available.
        """
        mask = valid_mask.float()
        l1 = self._masked_l1(D_g_pred, D_g_star, mask)
        
        if self.lambda_msg > 0:
            msg = self.msg_loss(D_g_pred * mask, D_g_star * mask)
        else:
            msg = D_g_pred.new_tensor(0.0)

        # Conditionally restore DSSIM for sharp cast shadows
        if self.enable_dssim_a_d and self.lambda_dssim > 0:
            dssim = self._compute_dssim(D_g_pred * mask, D_g_star * mask)
        else:
            dssim = D_g_pred.new_tensor(0.0)
        
        total = l1 + self.lambda_msg * msg + self.lambda_dssim * dssim
        
        details = {
            'loss_a_l1': l1.detach(),
            'loss_a_msg': msg.detach(),
        }
        if self.enable_dssim_a_d:
            details['loss_a_dssim'] = dssim.detach()
            
        return total, details


    def loss_b(self, xi_pred, xi_star, valid_mask):
        """
        Dec B loss: Chroma in bounded ratio space.
        Always active on valid pixels when pseudo-GT albedo is available.
        """
        mask = valid_mask.float()

        mse = self._masked_mse(xi_pred, xi_star, mask)
        return mse


    def loss_c_with_details(self, a_d_pred, A_d_star, valid_mask, m_albedo, seg_map=None, normals=None):
        """Dec C loss with per-term details for logging/debugging."""
        zero = (a_d_pred * 0.0).sum()
        if m_albedo.sum() == 0:
            return zero, {
                'loss_c_l1': zero,
                'loss_c_msg': zero,
                'loss_c_tv': zero,
                'loss_c_perceptual': zero,
                'loss_c_dssim': zero,
                'loss_c_semvar': zero,
            }

        route = m_albedo.view(-1, 1, 1, 1).to(valid_mask.device)
        mask = valid_mask.float() * route

        l1 = self._masked_l1(a_d_pred, A_d_star, mask)
        
        if self.lambda_msg > 0:
            msg = self.msg_loss(a_d_pred * mask, A_d_star * mask)
        else:
            msg = zero

        if self.tv_type is None or self.lambda_tv <= 0:
            tv = zero
        elif self.tv_type == 'normals' and normals is not None:
            tv = self.normal_tv_loss(a_d_pred, normals, valid_mask=mask)
        else:
            tv = self.semantic_tv_loss(a_d_pred, seg_map) if seg_map is not None else zero

        route_idx = (m_albedo > 0.5)
        if route_idx.any() and (self.lambda_perceptual > 0 or self.lambda_dssim > 0):
            route_mask = valid_mask[route_idx].float()
            route_pred = a_d_pred[route_idx] * route_mask
            route_tgt = A_d_star[route_idx] * route_mask
            
            if self.lambda_perceptual > 0:
                perceptual = self.perceptual_loss(route_pred, route_tgt)
            else:
                perceptual = zero
                
            if self.lambda_dssim > 0:
                dssim = self._compute_dssim(route_pred, route_tgt)
            else:
                dssim = zero
        else:
            perceptual = zero
            dssim = zero

        total = (
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
            'loss_c_semvar': zero.detach(),
        }
        return total, details

    def _loss_semantic_variance(self, a_d_pred, seg_map, valid_mask):
        """Encourage piece-wise constant albedo within selected semantic regions."""
        if seg_map is None:
            return (a_d_pred * 0.0).sum()

        if seg_map.ndim == 4 and seg_map.shape[1] == 1:
            seg_map = seg_map[:, 0]
        valid = valid_mask.bool()

        total = (a_d_pred * 0.0).sum()
        count = 0
        target_set = set(int(x) for x in self.semantic_var_classes)

        for b in range(a_d_pred.shape[0]):
            seg_b = seg_map[b]
            valid_b = valid[b, 0]
            for label in torch.unique(seg_b):
                label_i = int(label.item())
                if target_set and label_i not in target_set:
                    continue
                pix = (seg_b == label) & valid_b
                n = int(pix.sum().item())
                if n < 10:
                    continue
                vals = a_d_pred[b, :, pix]
                mu = vals.mean(dim=1, keepdim=True)
                total = total + ((vals - mu) ** 2).mean()
                count += 1

        if count == 0:
            return (a_d_pred * 0.0).sum()
        return total / float(count)

    def loss_d(self, pi_pred, pi_star, valid_mask, m_diffuse):
        """
        Dec D loss: Diffuse shading in inverse space.
        L_D = M_diffuse * ( ||π − π*||₁ + λ_msg * L_MSG(π, π*) + λ_dssim * DSSIM )
        Both terms are applied directly on inverse shading tensors (pi_pred, pi_star)
        with no additional conversion.
        No reconstruction loss — avoids absorbing specular residual R.
        """
        if m_diffuse.sum() == 0:
            zero = (pi_pred * 0.0).sum()
            return zero, {'loss_d_l1': zero, 'loss_d_msg': zero, 'loss_d_dssim': zero}

        route = m_diffuse.view(-1, 1, 1, 1).to(valid_mask.device)
        mask = valid_mask.float() * route

        l1 = self._masked_l1(pi_pred, pi_star, mask)
        
        if self.lambda_msg > 0:
            msg = self.msg_loss(pi_pred * mask, pi_star * mask)
        else:
            msg = zero
        
        if self.enable_dssim_a_d and self.lambda_dssim > 0:
            dssim = self._compute_dssim(pi_pred * mask, pi_star * mask)
        else:
            dssim = pi_pred.new_tensor(0.0)
        
        total = l1 + self.lambda_msg * msg + self.lambda_dssim * dssim
        
        details = {
            'loss_d_l1': l1.detach(),
            'loss_d_msg': msg.detach(),
        }
        if self.enable_dssim_a_d:
            details['loss_d_dssim'] = dssim.detach()
            
        return total, details

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def forward(self, predictions, targets, m_diffuse, m_albedo, valid_mask, seg_map=None, normals=None, rgb=None):
        # V11 supervision uses only albedo (a_d) and diffuse shading (pi).
        zero = predictions['a_d'].new_tensor(0.0)

        # 1. Albedo Loss (L1, MSG, DSSIM, and Semantic Variance)
        if targets.get('A_d_star') is not None:
            lc, lc_details = self.loss_c_with_details(
                predictions['a_d'],
                targets['A_d_star'],
                valid_mask,
                m_albedo,
                seg_map,
                normals,
            )
            
            # Add the new Physics Prior!
            if self.lambda_semvar > 0.0 and seg_map is not None:
                lc_var = self._loss_semantic_variance(predictions['a_d'], seg_map, valid_mask)
                lc = lc + self.lambda_semvar * lc_var
                lc_details['loss_c_semvar'] = lc_var.detach()
            else:
                lc_details['loss_c_semvar'] = zero.detach()
                
            # LNorm is intentionally disabled for V11 because it can leak colored illumination into albedo.
        else:
            lc = zero
            lc_details = {}

        # 2. Shading Loss (MSE, MSG, DSSIM)
        if targets.get('pi_star') is not None:
            ld, ld_details = self.loss_d(predictions['pi'], targets['pi_star'], valid_mask, m_diffuse)
        else:
            ld = zero
            ld_details = {}
            
        # 3. Diffuse Reconstruction Loss (Anchor using GT product, NOT I)
        if targets.get('A_d_star') is not None and targets.get('pi_star') is not None and self.lambda_recon > 0.0:
            if m_diffuse.sum() > 0:
                s_d_pred = (1.0 / (predictions['pi'] + 1e-6)) - 1.0
                s_d_pred = s_d_pred.clamp(0.0, 20.0)
                s_d_star = (1.0 / (targets['pi_star'] + 1e-6)) - 1.0
                recon_pred = predictions['a_d'] * s_d_pred
                # Reconstruct the true latent diffuse RGB rather than forcing inclusion of specular residuals.
                recon_target = targets['A_d_star'] * s_d_star
                
                route = m_diffuse.view(-1, 1, 1, 1).to(valid_mask.device)
                mask = valid_mask.float() * route
                l_recon = self._masked_l1(recon_pred, recon_target, mask)
            else:
                l_recon = zero
        else:
            l_recon = zero

        la, la_details = zero, {}
        lb = zero
        
        if self.w_a > 0.0 and 'd_g' in predictions and targets.get('D_g_star') is not None:
            la, la_details = self.loss_a(predictions['d_g'], targets['D_g_star'], valid_mask)
            la = self.w_a * la

        if self.w_b > 0.0 and 'xi' in predictions and targets.get('xi_star') is not None:
            lb = self.loss_b(predictions['xi'], targets['xi_star'], valid_mask)
            lb = self.w_b * lb

        loss_total = la + lb + lc + ld + (self.lambda_recon * l_recon)
        
        if 'd_g' in predictions and 'xi' in predictions and rgb is not None and self.lambda_recon_color > 0.0:
            la_s, da_s = self.loss_reconstruction_sc(
                predictions['a_d'],
                predictions['d_g'],
                predictions['xi'],
                rgb,
                valid_mask,
                m_albedo
            )
            loss_total = loss_total + self.lambda_recon_color * la_s
            ld_details.update(da_s)

        out = {
            # Keep both keys for compatibility with old and new callers.
            'loss': loss_total,
            'loss_total': loss_total,
            'loss_a': la,
            'loss_b': lb,
            'loss_c': lc,
            'loss_d': ld,
            'loss_recon': l_recon,
        }
        out.update(la_details)
        out.update(lc_details)
        out.update(ld_details)
        return out