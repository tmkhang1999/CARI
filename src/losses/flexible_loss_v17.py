"""
V17 Heterogeneous Loss for Parallel Intrinsic Decomposition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .msg_loss import MultiScaleGradientLoss

def _scale_shift_align(pred, target, mask, eps=1e-6):
    """Align pred to target via least-squares scale and shift.

    Solves: argmin_{a,b} || mask * (a*pred + b - target) ||^2
    Returns the aligned prediction.
    """
    B = pred.shape[0]
    p_flat = pred.reshape(B, -1)
    t_flat = target.reshape(B, -1)
    m_flat = mask.expand_as(pred).reshape(B, -1).float()

    sum_m = m_flat.sum(dim=1, keepdim=True).clamp(min=1.0)
    p_mean = (p_flat * m_flat).sum(dim=1, keepdim=True) / sum_m
    t_mean = (t_flat * m_flat).sum(dim=1, keepdim=True) / sum_m

    p_cen = (p_flat - p_mean) * m_flat
    t_cen = (t_flat - t_mean) * m_flat

    covar = (p_cen * t_cen).sum(dim=1, keepdim=True)
    var_p = (p_cen * p_cen).sum(dim=1, keepdim=True)

    a = F.relu(covar / (var_p + eps)) + eps
    b = t_mean - a * p_mean

    shape = [-1] + [1] * (pred.ndim - 1)
    return pred * a.view(shape) + b.view(shape)


class V17Loss(nn.Module):
    """
    Clean, parallel loss design for V17.
    """
    def __init__(self, config):
        super().__init__()

        self.w_a = float(config.get('lambda_a', 1.0))
        self.w_s = float(config.get('lambda_s', 1.0))
        self.w_r = float(config.get('lambda_r', 1.0))
        self.w_recon = float(config.get('lambda_recon', 1.0))

        self.lambda_msg = float(config.get('lambda_msg', 1.0))
        self.lambda_dssim = float(config.get('lambda_dssim', 0.2))

        self.msg_loss = MultiScaleGradientLoss(scales=4)
        
        self._dssim_win_size = 11
        self._dssim_sigma = 1.5
        self._dssim_win = None

    def _masked_mse(self, pred, target, mask):
        diff = F.mse_loss(pred, target, reduction='none')
        return (diff * mask).sum() / (mask.sum() + 1e-7)

    def _masked_l1(self, pred, target, mask):
        diff = F.l1_loss(pred, target, reduction='none')
        return (diff * mask).sum() / (mask.sum() + 1e-7)

    def _get_dssim_window(self, channels, device, dtype):
        if (self._dssim_win is not None
                and self._dssim_win.shape[0] == channels
                and self._dssim_win.device == device
                and self._dssim_win.dtype == dtype):
            return self._dssim_win

        win_size = self._dssim_win_size
        sigma = self._dssim_sigma
        coords = torch.arange(win_size, dtype=dtype, device=device) - win_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window = g.unsqueeze(1) @ g.unsqueeze(0)
        window = window.unsqueeze(0).unsqueeze(0).expand(channels, 1, -1, -1).contiguous()
        self._dssim_win = window
        return window

    def _compute_dssim(self, pred, target, mask):
        C = pred.shape[1]
        win = self._get_dssim_window(C, pred.device, pred.dtype)
        pad = self._dssim_win_size // 2

        mu_x = F.conv2d(pred, win, padding=pad, groups=C)
        mu_y = F.conv2d(target, win, padding=pad, groups=C)

        mu_x_sq = mu_x * mu_x
        mu_y_sq = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x_sq = (F.conv2d(pred * pred, win, padding=pad, groups=C) - mu_x_sq).clamp_min(0)
        sigma_y_sq = (F.conv2d(target * target, win, padding=pad, groups=C) - mu_y_sq).clamp_min(0)
        sigma_xy = F.conv2d(pred * target, win, padding=pad, groups=C) - mu_xy

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
                   ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))

        dssim_map = (1.0 - ssim_map) / 2.0
        return (dssim_map * mask).sum() / (mask.sum() + 1e-7)

    def forward(self, predictions, targets, loss_mask, m_residual, rgb, use_ssi=True):
        """
        predictions: dict with 'a_d', 'shading', 'residual', 'rgb_reconstructed'
        targets: dict with 'A_d_star', 'S_d_star', 'R_star'
        loss_mask: (B, 1, H, W) float mask of valid pixels
        m_residual: (B,) float tensor gating the residual loss per-sample
        rgb: (B, 3, H, W) input image
        use_ssi: bool, if False, falls back to plain MSE for shading (warmup)
        """
        zero = torch.tensor(0.0, device=loss_mask.device)
        details = {}

        # 1. Albedo (MSE + MSG p=1 + DSSIM)
        a_pred, a_gt = predictions['a_d'], targets['A_d_star']
        l_a_mse = self._masked_mse(a_pred, a_gt, loss_mask)
        l_a_msg = self.msg_loss(a_pred, a_gt, mask=loss_mask, p=1) if self.lambda_msg > 0 else zero
        l_a_dssim = self._compute_dssim(a_pred, a_gt, loss_mask) if self.lambda_dssim > 0 else zero
        
        la = l_a_mse + self.lambda_msg * l_a_msg + self.lambda_dssim * l_a_dssim
        details.update({'loss_c_l1': l_a_mse.detach(), 'loss_c_msg': l_a_msg.detach(), 'loss_c_dssim': l_a_dssim.detach()})

        # 2. Shading (SSI-MSE + MSG p=2) - operates directly on linear shading
        s_pred, s_gt = predictions['shading'], targets['S_d_star']
        if use_ssi:
            s_aligned = _scale_shift_align(s_pred, s_gt, loss_mask)
        else:
            s_aligned = s_pred
            
        l_s_mse = self._masked_mse(s_aligned, s_gt, loss_mask)
        l_s_msg = self.msg_loss(s_aligned, s_gt, mask=loss_mask, p=2) if self.lambda_msg > 0 else zero
        
        ls = l_s_mse + self.lambda_msg * l_s_msg
        details.update({'loss_shading_mse': l_s_mse.detach(), 'loss_shading_msg': l_s_msg.detach()})

        # 3. Residual (L1 + MSG p=1)
        r_pred, r_gt = predictions['residual'], targets['R_star']
        m_res_mask = loss_mask * m_residual.view(-1, 1, 1, 1)
        
        if m_res_mask.sum() > 0:
            l_r_l1 = self._masked_l1(r_pred, r_gt, m_res_mask)
            l_r_msg = self.msg_loss(r_pred, r_gt, mask=m_res_mask, p=1) if self.lambda_msg > 0 else zero
            lr = l_r_l1 + self.lambda_msg * l_r_msg
        else:
            lr, l_r_l1, l_r_msg = zero, zero, zero
            
        details.update({'loss_residual_mse': l_r_l1.detach(), 'loss_residual_msg': l_r_msg.detach()})

        # 4. Reconstruction (L1 + MSG p=1)
        recon_pred = predictions['rgb_reconstructed']
        l_rec_l1 = self._masked_l1(recon_pred, rgb, loss_mask)
        l_rec_msg = self.msg_loss(recon_pred, rgb, mask=loss_mask, p=1) if self.lambda_msg > 0 else zero
        
        l_recon = l_rec_l1 + 0.5 * self.lambda_msg * l_rec_msg
        details.update({'loss_recon_l1': l_rec_l1.detach(), 'loss_recon_msg': l_rec_msg.detach()})

        # Weighting
        la_w = self.w_a * la
        ls_w = self.w_s * ls
        lr_w = self.w_r * lr
        l_recon_w = self.w_recon * l_recon

        loss_total = la_w + ls_w + lr_w + l_recon_w

        out = {
            'loss_total': loss_total,
            'loss_a': la_w,
            'loss_s': ls_w,
            'loss_r': lr_w,
            'loss_recon': l_recon_w,
        }
        out.update(details)
        return out
