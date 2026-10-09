"""Single-image losses of the V17 model, plus the CIAI pair terms it is trained with.

V17 predicts an albedo A, a three-channel inverse shading pi = 1/(S_d + 1) and an
analytic residual R = (I - A*S_d)_+. The single-image losses tie these to whatever
ground truth a dataset provides:

  albedo      MSE + multi-scale gradient + DSSIM + chroma-direction L1, all datasets
  shading     scale-and-shift-invariant MSE (+ gradient) in the pi domain, Hypersim only
  residual    L1 to the target residual + sparsity, rows with m_residual = 1
  recon       two-sided diffuse product A*S_d vs A*_gt * S_d_gt, Hypersim only
  material    DINOv2-gated within-material chroma consistency, GT-free
  shade_sign  relu(A*S_d - I): brightening beyond the image must come from albedo

The pair terms live in losses/ciai.py and are applied by the training step, which owns
the second forward pass.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import ciai
from .msg_loss import MultiScaleGradientLoss


# Settings of earlier experiments that this code no longer implements. Configs that
# switch any of them on would silently train a different objective, so refuse them.
_RETIRED = (
    'lambda_shade_abs', 'lambda_edge', 'lambda_grad_decorr', 'lambda_alb_flat',
    'lambda_alb_flat_lf', 'lambda_refiner_delta', 'lambda_refiner_hf', 'lambda_normal',
    'lambda_assign_tv', 'lambda_illum_prior', 'lambda_assign_entropy',
    'lambda_shadow_inv', 'lambda_shadow_explain', 'lambda_shadow_gt_front3d',
    'lambda_ordinal', 'lambda_ordinal_iiw', 'lambda_albedo_init', 'lambda_chroma_field',
    'lambda_chroma_field_pair', 'mat_intensity_weight', 'albedo_msg_target_grad_threshold',
)


def material_consistency_loss(albedo, dino_tokens, patch_hw, threshold=0.0):
    """Albedo chroma differences between adjacent DINOv2 patches, weighted by token similarity.

    Patches with similar DINOv2 tokens are likely the same material, so their pooled albedo
    chroma should agree; the weight falls to zero at material boundaries.
    """
    B, D = dino_tokens.shape[0], dino_tokens.shape[-1]
    H_p, W_p = patch_hw

    dino_spatial = dino_tokens.reshape(B, H_p, W_p, D).permute(0, 3, 1, 2)
    dino_norm = F.normalize(dino_spatial, dim=1)
    sim_h = (dino_norm[:, :, :, :-1] * dino_norm[:, :, :, 1:]).sum(dim=1)
    sim_v = (dino_norm[:, :, :-1, :] * dino_norm[:, :, 1:, :]).sum(dim=1)
    w_h = (sim_h - threshold).clamp(min=0.0)
    w_v = (sim_v - threshold).clamp(min=0.0)

    a_patch = F.adaptive_avg_pool2d(albedo, (H_p, W_p))
    a_chroma = F.normalize(a_patch.clamp(1e-6), dim=1)
    diff_h = (a_chroma[:, :, :, :-1] - a_chroma[:, :, :, 1:]).norm(dim=1)
    diff_v = (a_chroma[:, :, :-1, :] - a_chroma[:, :, 1:, :]).norm(dim=1)
    return (w_h * diff_h).mean() + (w_v * diff_v).mean()


def scale_shift_align(pred, target, mask, eps=1e-6):
    """Least-squares per-image scale and shift aligning pred to target over the mask."""
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
    def __init__(self, config):
        super().__init__()
        active = [k for k in _RETIRED if float(config.get(k, 0.0) or 0.0) != 0.0]
        if active:
            raise ValueError(f'config enables retired loss settings {active}; they were removed '
                             'from the codebase (see documents/history/DEVELOPMENT_HISTORY.md)')
        if str(config.get('recon_mode', 'diffuse')) != 'diffuse':
            raise ValueError("only recon_mode: diffuse is supported (the residual is analytic)")
        if bool(config.get('albedo_l1', False)):
            raise ValueError('albedo_l1 was never used by a reported run and is not supported')

        self.w_a = float(config.get('lambda_a', 1.0))
        self.w_s = float(config.get('lambda_s', 1.0))
        self.w_r = float(config.get('lambda_r', 0.0))
        self.w_recon = float(config.get('lambda_recon', 1.0))

        global_msg = float(config.get('lambda_msg', 0.5))
        self.lambda_msg_a = float(config.get('lambda_msg_albedo', global_msg))
        self.lambda_msg_s = float(config.get('lambda_msg_shading', global_msg))
        self.lambda_msg_r = float(config.get('lambda_msg_residual', global_msg))
        self.lambda_msg_recon = float(config.get('lambda_msg_recon', global_msg * 0.5))

        self.lambda_dssim = float(config.get('lambda_dssim', 0.2))
        self.lambda_res_sparse = float(config.get('lambda_res_sparse', 0.05))
        self.lambda_a_chroma = float(config.get('lambda_a_chroma', 0.0))
        self.lambda_mat_consist = float(config.get('lambda_mat_consist', 0.0))
        self.mat_sim_threshold = float(config.get('mat_sim_threshold', 0.0))
        self.lambda_shade_sign = float(config.get('lambda_shade_sign', 0.0))

        # CIAI pair weights; the training step reads them and calls losses/ciai.py.
        self.lambda_alb_invariance = float(config.get('lambda_alb_invariance', 0.0))
        self.lambda_explain = float(config.get('lambda_explain', 0.0))
        self.lambda_chr_explain = float(config.get('lambda_chr_explain', 0.0))

        self.msg_loss = MultiScaleGradientLoss(scales=4)
        self._dssim_win_size = 11
        self._dssim_sigma = 1.5
        self._dssim_win = None

    # ── helpers ────────────────────────────────────────────────────────────────
    @staticmethod
    def _masked_mse(pred, target, mask):
        diff = F.mse_loss(pred, target, reduction='none')
        return (diff * mask).sum() / (mask.sum() + 1e-7)

    @staticmethod
    def _masked_l1(pred, target, mask):
        diff = F.l1_loss(pred, target, reduction='none')
        return (diff * mask).sum() / (mask.sum() + 1e-7)

    def _get_dssim_window(self, channels, device, dtype):
        if (self._dssim_win is not None
                and self._dssim_win.shape[0] == channels
                and self._dssim_win.device == device
                and self._dssim_win.dtype == dtype):
            return self._dssim_win
        win_size = self._dssim_win_size
        coords = torch.arange(win_size, dtype=dtype, device=device) - win_size // 2
        g = torch.exp(-(coords ** 2) / (2 * self._dssim_sigma ** 2))
        g = g / g.sum()
        window = g.unsqueeze(1) @ g.unsqueeze(0)
        self._dssim_win = window.unsqueeze(0).unsqueeze(0).expand(channels, 1, -1, -1).contiguous()
        return self._dssim_win

    def _compute_dssim(self, pred, target, mask):
        C = pred.shape[1]
        win = self._get_dssim_window(C, pred.device, pred.dtype)
        pad = self._dssim_win_size // 2
        mu_x = F.conv2d(pred, win, padding=pad, groups=C)
        mu_y = F.conv2d(target, win, padding=pad, groups=C)
        mu_x_sq, mu_y_sq, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
        sigma_x_sq = (F.conv2d(pred * pred, win, padding=pad, groups=C) - mu_x_sq).clamp_min(0)
        sigma_y_sq = (F.conv2d(target * target, win, padding=pad, groups=C) - mu_y_sq).clamp_min(0)
        sigma_xy = F.conv2d(pred * target, win, padding=pad, groups=C) - mu_xy
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
                   ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
        return (((1.0 - ssim_map) / 2.0) * mask).sum() / (mask.sum() + 1e-7)

    def shade_sign(self, a, s_linear, rgb, mask):
        """relu(A*S_d - I): the residual is clamped non-negative, so overshoot is unmodelled."""
        overshoot = F.relu(a * s_linear - rgb)
        return (overshoot * mask).sum() / (mask.sum() + 1e-7)

    # ── CIAI pair terms (kept as methods so the training step reads one object) ──
    def albedo_invariance(self, a1, a2, mask):
        return ciai.albedo_invariance(a1, a2, mask)

    def luminance_explain(self, rgb1, rgb2, s1, s2, mask):
        return ciai.luminance_explain(rgb1, rgb2, s1, s2, mask)

    def chroma_explain(self, rgb1, rgb2, s1, s2, mask):
        return ciai.chroma_explain(rgb1, rgb2, s1, s2, mask)

    # ── single-image objective ─────────────────────────────────────────────────
    def forward(self, predictions, targets, loss_mask, m_diffuse, m_residual, rgb, use_ssi=True):
        """
        predictions: model outputs ('a_d', 'shading', 'shading_linear', 'residual', tokens)
        targets    : compute_targets() output ('A_d_star', 'S_d_star', 'pi_star', 'R_star')
        loss_mask  : (B,1,H,W) valid pixels
        m_diffuse  : (B,) 1 for rows with ground-truth diffuse shading (Hypersim)
        m_residual : (B,) 1 for rows whose residual is supervised
        use_ssi    : False during the shading warm-up (plain MSE in the pi domain)
        """
        zero = torch.tensor(0.0, device=loss_mask.device)
        details = {}
        # rgb is clipped to [0,1]; A*S is not, so R and recon targets are meaningless there.
        sat_ok = (rgb.amax(dim=1, keepdim=True) < 0.99).float()

        # 1. Albedo
        a_pred, a_gt = predictions['a_d'], targets['A_d_star']
        l_a_mse = self._masked_mse(a_pred, a_gt, loss_mask)
        l_a_msg = self.msg_loss(a_pred, a_gt, mask=loss_mask, p=1) if self.lambda_msg_a > 0 else zero
        l_a_dssim = self._compute_dssim(a_pred, a_gt, loss_mask) if self.lambda_dssim > 0 else zero
        if self.lambda_a_chroma > 0:
            # Unit-vector L1: prices hue/saturation error independently of brightness, so
            # desaturating is no longer a cheap way to lower the MSE.
            l_a_chroma = self._masked_l1(F.normalize(a_pred.clamp(1e-6), dim=1),
                                         F.normalize(a_gt.clamp(1e-6), dim=1), loss_mask)
        else:
            l_a_chroma = zero
        la = (l_a_mse + self.lambda_msg_a * l_a_msg + self.lambda_dssim * l_a_dssim
              + self.lambda_a_chroma * l_a_chroma)
        details.update({'loss_c_l1': l_a_mse.detach(), 'loss_c_msg': l_a_msg.detach(),
                        'loss_c_dssim': l_a_dssim.detach(), 'loss_c_chroma': l_a_chroma.detach()})

        # 2. Shading, pi domain, Hypersim only
        s_pred, s_gt = predictions['shading'], targets['pi_star']
        m_diff_mask = loss_mask * m_diffuse.view(-1, 1, 1, 1)
        if m_diff_mask.sum() > 0:
            s_aligned = scale_shift_align(s_pred, s_gt, m_diff_mask) if use_ssi else s_pred
            l_s_mse = self._masked_mse(s_aligned, s_gt, m_diff_mask)
            l_s_msg = self.msg_loss(s_aligned, s_gt, mask=m_diff_mask, p=2) if self.lambda_msg_s > 0 else zero
            ls = l_s_mse + self.lambda_msg_s * l_s_msg
        else:
            ls = l_s_mse = l_s_msg = zero
        details.update({'loss_shading_mse': l_s_mse.detach(), 'loss_shading_msg': l_s_msg.detach()})

        # 3. Residual: supervised where rgb is not clipped and the row asks for it
        r_pred, r_gt = predictions['residual'], targets['R_star']
        m_res_mask = (m_residual.view(-1, 1, 1, 1) * sat_ok * loss_mask).expand_as(r_pred)
        if m_res_mask.sum() > 0:
            l_r_l1 = self._masked_l1(r_pred, r_gt, m_res_mask)
            l_r_msg = self.msg_loss(r_pred, r_gt, mask=m_res_mask, p=1) if self.lambda_msg_r > 0 else zero
            l_r_sparse = (self._masked_l1(r_pred, torch.zeros_like(r_pred), loss_mask)
                          if self.lambda_res_sparse > 0 else zero)
            lr = l_r_l1 + self.lambda_msg_r * l_r_msg + self.lambda_res_sparse * l_r_sparse
        else:
            lr = l_r_l1 = l_r_msg = l_r_sparse = zero
        details.update({'loss_residual_mse': l_r_l1.detach(), 'loss_residual_msg': l_r_msg.detach(),
                        'loss_residual_sparse': l_r_sparse.detach()})

        # 4. Diffuse reconstruction, Hypersim only (elsewhere S_d* = I/A* makes it a tautology)
        recon_pred = predictions['a_d'] * predictions['shading_linear']
        recon_target = targets['A_d_star'] * targets['S_d_star']
        recon_mask = sat_ok * loss_mask * m_diffuse.view(-1, 1, 1, 1)
        l_rec_l1 = self._masked_l1(recon_pred, recon_target, recon_mask)
        l_rec_msg = (self.msg_loss(recon_pred, recon_target, mask=recon_mask, p=1)
                     if self.lambda_msg_recon > 0 else zero)
        l_recon = l_rec_l1 + self.lambda_msg_recon * l_rec_msg
        details.update({'loss_recon_l1': l_rec_l1.detach(), 'loss_recon_msg': l_rec_msg.detach()})

        # 5. DINOv2 material consistency (GT-free)
        if (self.lambda_mat_consist > 0 and predictions.get('dino_tokens') is not None
                and predictions.get('dino_patch_hw') is not None):
            l_mat = material_consistency_loss(a_pred, predictions['dino_tokens'],
                                              predictions['dino_patch_hw'],
                                              threshold=self.mat_sim_threshold)
        else:
            l_mat = zero

        # 6. Shading sign prior (data-free)
        if self.lambda_shade_sign > 0:
            l_shade_sign = self.shade_sign(predictions['a_d'], predictions['shading_linear'],
                                           rgb, loss_mask)
        else:
            l_shade_sign = zero

        out = {
            'loss_a': self.w_a * la,
            'loss_s': self.w_s * ls,
            'loss_r': self.w_r * lr,
            'loss_recon': self.w_recon * l_recon,
            'loss_mat_consist': self.lambda_mat_consist * l_mat,
            'loss_shade_sign': self.lambda_shade_sign * l_shade_sign,
        }
        out['loss_total'] = (out['loss_a'] + out['loss_s'] + out['loss_r'] + out['loss_recon']
                             + out['loss_mat_consist'] + out['loss_shade_sign'])
        out.update(details)       # component terms are logged unweighted
        return out
