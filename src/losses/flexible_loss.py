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
        self.lambda_dssim_c = config.get('lambda_dssim', 0.15)
        
        # V13 auxiliary losses
        self.lambda_boundary = config.get('lambda_boundary', 0.0)
        self.lambda_semvar = config.get('lambda_semvar', 0.0)
        self.lambda_exclusion = config.get('lambda_exclusion', 0.0)
        self.semantic_var_classes = config.get('semantic_var_classes', [1, 2, 22]) # Walls, floors, ceiling

        # V15 cross-decoder reconstruction constraint: L1(a_d * s_c - rgb).
        # Ties Dec C (albedo) and Dec A/B (colorful shading) into a self-consistent
        # pair. Unlike derive_albedo which is trivially zero, this uses the
        # *predicted* albedo against the fixed cascade shading.
        self.lambda_recon = config.get('lambda_recon', 0.0)

        # Auxiliary CCR Edge Loss (PIE-Net style edge supervision)
        self.lambda_ccr_edge = config.get('lambda_ccr_edge', 0.0)

        # Multi-scale gradient loss for structural fidelity
        self.msg_loss = MultiScaleGradientLoss(scales=4)
        self.bce_logits = nn.BCEWithLogitsLoss(reduction='none')

        # DSSIM window (registered as buffer so it moves to GPU with .to())
        self._dssim_win_size = 11
        self._dssim_sigma = 1.5
        self._dssim_win = None  # lazily created on first use

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def _masked_mse(self, pred, target, mask):
        """Element-wise MSE masked by valid * routing mask."""
        diff = F.mse_loss(pred, target, reduction='none')
        return (diff * mask).sum() / (mask.sum() + 1e-7)

    def _masked_l1(self, pred, target, mask):
        """Element-wise L1 masked by valid * routing mask."""
        diff = F.l1_loss(pred, target, reduction='none')
        return (diff * mask).sum() / (mask.sum() + 1e-7)

    def _get_dssim_window(self, channels, device, dtype):
        """Lazily create/cache the Gaussian SSIM window."""
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
        window = g.unsqueeze(1) @ g.unsqueeze(0)  # (win, win)
        window = window.unsqueeze(0).unsqueeze(0).expand(channels, 1, -1, -1).contiguous()
        self._dssim_win = window
        return window

    def _compute_dssim(self, pred, target, mask=None):
        """Compute masked DSSIM = (1 - SSIM) / 2 for [0, 1] bounded signals.

        Uses a conservative approach: SSIM is computed over the full image,
        then the per-pixel SSIM map is masked before averaging. This avoids
        boundary artifacts from masked convolutions.
        """
        C = pred.shape[1]
        win = self._get_dssim_window(C, pred.device, pred.dtype)
        pad = self._dssim_win_size // 2

        mu_x = F.conv2d(pred, win, padding=pad, groups=C)
        mu_y = F.conv2d(target, win, padding=pad, groups=C)

        mu_x_sq = mu_x * mu_x
        mu_y_sq = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x_sq = F.conv2d(pred * pred, win, padding=pad, groups=C) - mu_x_sq
        sigma_y_sq = F.conv2d(target * target, win, padding=pad, groups=C) - mu_y_sq
        sigma_xy = F.conv2d(pred * target, win, padding=pad, groups=C) - mu_xy

        # Clamp variances for numerical stability
        sigma_x_sq = sigma_x_sq.clamp(min=0)
        sigma_y_sq = sigma_y_sq.clamp(min=0)

        C1 = 0.01 ** 2  # data_range=1.0
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
                   ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))

        dssim_map = (1.0 - ssim_map) / 2.0

        if mask is not None:
            return (dssim_map * mask).sum() / (mask.sum() + 1e-7)
        return dssim_map.mean()

    @staticmethod
    def _detect_segment_boundaries(seg_map):
        """Detect pixels at segment boundaries via finite-difference on IDs.

        Args:
            seg_map: (N, 1, H, W) or (N, H, W) integer segment IDs.
        Returns:
            (N, 1, H, W) float tensor, 1.0 at boundaries, 0.0 interior.
            Dilated by 3x3 max-pool to cover the boundary region.
        """
        if seg_map.dim() == 3:
            seg = seg_map.unsqueeze(1).float()
        elif seg_map.dim() == 4:
            seg = seg_map[:, 0:1].float()
        else:
            seg = seg_map.float()

        # Boundary: adjacent pixels with different segment IDs
        dx = (seg[:, :, :, 1:] != seg[:, :, :, :-1]).float()
        dy = (seg[:, :, 1:, :] != seg[:, :, :-1, :]).float()
        dx = F.pad(dx, (0, 1))
        dy = F.pad(dy, (0, 0, 0, 1))
        boundary = (dx + dy).clamp(0.0, 1.0)

        # Dilate slightly to cover the transition region
        boundary = F.max_pool2d(boundary, kernel_size=3, stride=1, padding=1)
        return boundary

    def loss_boundary_consistency(self, a_d_pred, ccr_edge_gt, valid_mask, seg_map=None):
        """GT-guided albedo gradient suppression.
        
        Penalizes albedo spatial gradients where the Ground Truth albedo
        indicates no material boundary (ccr_edge_gt is low).
        This is perfectly illumination-invariant and highly stable.
        """
        if seg_map is not None:
            seg_boundary = self._detect_segment_boundaries(seg_map)
            if seg_boundary.shape[-2:] != ccr_edge_gt.shape[-2:]:
                seg_boundary = F.interpolate(
                    seg_boundary, size=ccr_edge_gt.shape[-2:], mode="nearest"
                )
            # Boost the edge map at semantic boundaries to be even more permissive
            # of gradients there, while relying on the clean GT albedo edges elsewhere.
            edge_map = torch.clamp(ccr_edge_gt + seg_boundary * 0.2, 0.0, 1.0)
        else:
            edge_map = ccr_edge_gt
            
        non_edge = (1.0 - edge_map) * valid_mask.float()

        # 3. Use the same discrete Laplacian cross-kernel as CCR for albedo
        kernel = torch.tensor(
            [[0, 1, 0],
             [1, 0, -1], 
             [0, -1, 0]], dtype=torch.float32, device=a_d_pred.device
        ).view(1, 1, 3, 3)
        
        r, g, b = a_d_pred[:, 0:1], a_d_pred[:, 1:2], a_d_pred[:, 2:3]
        diff_r = F.conv2d(r, kernel, padding=1).abs()
        diff_g = F.conv2d(g, kernel, padding=1).abs()
        diff_b = F.conv2d(b, kernel, padding=1).abs()
        
        a_grad = torch.max(torch.max(diff_r, diff_g), diff_b)

        return (a_grad * non_edge).sum() / (non_edge.sum() + 1e-7)

    def loss_gradient_exclusion(self, a_d_pred, pi_pred, valid_mask):
        """
        Soft Gradient Exclusion Loss (w-L1).
        Penalizes albedo gradients where diffuse shading is smooth.
        Uses an exponential escape hatch to forgive overlapping physical edges.
        """
        # 1. Use the true Diffuse Shading prediction, safely inverted to linear space.
        s_d_pred = (1.0 / pi_pred.clamp(min=1e-4) - 1.0).detach()
        
        # 2. Align the mask dimensions safely
        mask = valid_mask.float()
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
            
        # 3. Calculate spatial gradients
        grad_a_x = (a_d_pred[:,:,:,1:] - a_d_pred[:,:,:,:-1]).abs().mean(dim=1, keepdim=True)
        grad_a_y = (a_d_pred[:,:,1:,:] - a_d_pred[:,:,:-1,:]).abs().mean(dim=1, keepdim=True)
        
        grad_s_x = (s_d_pred[:,:,:,1:] - s_d_pred[:,:,:,:-1]).abs().mean(dim=1, keepdim=True)
        grad_s_y = (s_d_pred[:,:,1:,:] - s_d_pred[:,:,:-1,:]).abs().mean(dim=1, keepdim=True)
        
        # Pad back to original dimensions
        grad_a_x = F.pad(grad_a_x, (0,1))
        grad_a_y = F.pad(grad_a_y, (0,0,0,1))
        grad_s_x = F.pad(grad_s_x, (0,1))
        grad_s_y = F.pad(grad_s_y, (0,0,0,1))
        
        # 4. The Soft Exclusion Formulation
        alpha = 10.0
        
        # Where Shading Gradient is HIGH, weight drops to ZERO (Escape Hatch)
        # Where Shading Gradient is LOW, weight is 1.0 (Full Penalty applied to Albedo)
        weight_x = torch.exp(-alpha * grad_s_x)
        weight_y = torch.exp(-alpha * grad_s_y)
        
        excl_x = grad_a_x * weight_x
        excl_y = grad_a_y * weight_y
        
        return ((excl_x + excl_y) * mask).sum() / (mask.sum() + 1e-7)

    def _loss_semantic_variance(self, a_d_pred, seg_map, valid_mask):
        if seg_map is None or not self.semantic_var_classes:
            return (a_d_pred * 0.0).sum()

        if seg_map.ndim == 4:
            seg_map = seg_map.squeeze(1)
        
        valid = valid_mask.bool().squeeze(1)
        
        # 1. Create a unified binary mask for ALL target structural classes
        # Use torch.isin for fast vectorized matching
        target_tensor = torch.tensor(self.semantic_var_classes, device=seg_map.device)
        struct_mask = torch.isin(seg_map, target_tensor) & valid
        
        # 2. Compute variance globally over structural pixels per batch item
        # a_d_pred: (B, 3, H, W) -> (B, 3, H*W)
        B, C, H, W = a_d_pred.shape
        flat_a = a_d_pred.view(B, C, -1)
        flat_mask = struct_mask.view(B, 1, -1).expand(-1, C, -1)
        
        total_var = a_d_pred.new_tensor(0.0)
        valid_batches = 0
        
        for b in range(B):
            for c in range(C):
                pixel_vals = flat_a[b, c, flat_mask[b, c]]
                if pixel_vals.numel() > 16:  # Only compute if enough pixels exist
                    total_var += pixel_vals.var(unbiased=False)
                    valid_batches += 1
                    
        if valid_batches == 0:
            return (a_d_pred * 0.0).sum()
            
        return total_var / float(valid_batches)


    def loss_recon(self, a_d_pred, s_c, rgb, valid_mask):
        """L1 reconstruction constraint: a_d * s_c should equal the input rgb.

        Uses predicted albedo (a_d_pred from Dec C) and the detached colorful
        shading (s_c from Dec A+B). This is NOT trivially zero because a_d_pred
        is independently predicted, not derived from s_c.
        """
        recon = a_d_pred * s_c.clamp(min=1e-4)
        return self._masked_l1(recon, rgb.clamp(0.0, 1.0), valid_mask.float())

    # ------------------------------------------------------------------
    # Per-decoder losses
    # ------------------------------------------------------------------
    def loss_a(self, D_g_pred, D_g_star, valid_mask):
        """
        Dec A loss: Gray shading in inverse space.
        Always active on valid pixels when pseudo-GT albedo is available.
        Uses scale-and-shift invariant MSE (SSI-MSE).
        """
        mask = valid_mask.float()
        
        mse = self._masked_mse(D_g_pred, D_g_star, mask)
        
        if self.lambda_msg > 0:
            msg = self.msg_loss(D_g_pred, D_g_star, mask=mask, p=1)
        else:
            msg = D_g_pred.new_tensor(0.0)

        total = mse + self.lambda_msg * msg
        
        details = {
            'loss_a_mse': mse.detach(),
        }
        if self.lambda_msg > 0:
            details['loss_a_msg'] = msg.detach()
            
        return total, details


    def loss_b(self, xi_pred, xi_star, valid_mask):
        """
        Dec B loss: Chroma in bounded ratio space.
        Always active on valid pixels when pseudo-GT albedo is available.
        """
        mask = valid_mask.float()
        mse = self._masked_mse(xi_pred, xi_star, mask)

        if self.lambda_msg > 0:
            # Chroma uses p=2 (L2 MSG) for smooth gradients
            msg = self.msg_loss(xi_pred, xi_star, mask=mask, p=2)
        else:
            msg = xi_pred.new_tensor(0.0)

        total = mse + self.lambda_msg * msg
        details = {
            'loss_b_mse': mse.detach(),
        }
        if self.lambda_msg > 0:
            details['loss_b_msg'] = msg.detach()
        return total, details


    def loss_c_with_details(self, a_d_pred, A_d_star, valid_mask, seg_map=None, normals=None):
        """Dec C loss with per-term details for logging/debugging."""
        zero = (a_d_pred * 0.0).sum()
        mask = valid_mask.float()
        mse = self._masked_mse(a_d_pred, A_d_star, mask)
        
        if self.lambda_msg > 0:
            # Albedo uses p=1 (L1 MSG) to preserve sharp material discontinuities
            msg = self.msg_loss(a_d_pred, A_d_star, mask=mask, p=1)
        else:
            msg = zero

        total = mse + self.lambda_msg * msg

        # DSSIM: structural fidelity for sharp albedo textures
        if self.lambda_dssim_c > 0:
            dssim = self._compute_dssim(a_d_pred, A_d_star, mask)
            total = total + self.lambda_dssim_c * dssim
        else:
            dssim = zero

        details = {
            'loss_c_l1': mse.detach(),
        }
        if self.lambda_msg > 0:
            details['loss_c_msg'] = msg.detach()
        if self.lambda_dssim_c > 0:
            details['loss_c_dssim'] = dssim.detach()
        return total, details

    def loss_d(self, pi_pred, pi_star, valid_mask, m_diffuse):
        """
        Dec D loss: Diffuse shading in inverse space.
        L_D = M_diffuse * ( MSE(π, π*) + λ_msg * L_MSG(π, π*) )
        Both terms are applied directly on inverse shading tensors (pi_pred, pi_star)
        with no additional conversion.
        No reconstruction loss — avoids absorbing specular residual R.
        """
        if m_diffuse.sum() == 0:
            zero = (pi_pred * 0.0).sum()
            details = {'loss_d_mse': zero}
            if self.lambda_msg > 0:
                details['loss_d_msg'] = zero
            return zero, details

        route = m_diffuse.view(-1, 1, 1, 1).to(valid_mask.device)
        mask = valid_mask.float() * route

        mse = self._masked_mse(pi_pred, pi_star, mask)
        
        if self.lambda_msg > 0:
            # Diffuse Shading uses p=1 (L1 MSG) according to CD-IID
            msg = self.msg_loss(pi_pred, pi_star, mask=mask, p=1)
        else:
            msg = zero
        
        total = mse + self.lambda_msg * msg
        
        details = {
            'loss_d_mse': mse.detach(),
        }
        if self.lambda_msg > 0:
            details['loss_d_msg'] = msg.detach()
            
        return total, details

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def forward(self, predictions, targets, m_diffuse, loss_mask, seg_map=None, normals=None, rgb=None, ccr=None):
        # V11 supervision uses only albedo (a_d) and diffuse shading (pi).
        zero = predictions['a_d'].new_tensor(0.0)

        # 1. Albedo Loss (L1, MSG)
        if targets.get('A_d_star') is not None:
            lc, lc_details = self.loss_c_with_details(
                predictions['a_d'],
                targets['A_d_star'],
                loss_mask,
                seg_map,
                normals,
            )
        else:
            lc = zero
            lc_details = {}

        # 2. Shading Loss (MSE, MSG)
        if targets.get('pi_star') is not None:
            ld, ld_details = self.loss_d(predictions['pi'], targets['pi_star'], loss_mask, m_diffuse)
        else:
            ld = zero
            ld_details = {}

        # 3. Auxiliary losses (V13)
        if self.lambda_boundary > 0.0 and 'ccr_edge_gt' in targets:
            l_boundary = self.loss_boundary_consistency(predictions['a_d'], targets['ccr_edge_gt'], loss_mask, seg_map=seg_map)
            lc = lc + self.lambda_boundary * l_boundary
            lc_details['loss_c_boundary'] = l_boundary.detach()

        if self.lambda_semvar > 0.0 and seg_map is not None:
            l_semvar = self._loss_semantic_variance(predictions['a_d'], seg_map, loss_mask)
            lc = lc + self.lambda_semvar * l_semvar
            lc_details['loss_c_semvar'] = l_semvar.detach()
        else:
            lc_details['loss_c_semvar'] = zero.detach()

        if self.lambda_exclusion > 0:
            exclusion_loss = self.loss_gradient_exclusion(predictions['a_d'], predictions['pi'], loss_mask)
            lc = lc + self.lambda_exclusion * exclusion_loss
            lc_details['loss_c_exclusion_pi'] = exclusion_loss.detach()

        # 4. V15 reconstruction constraint: L1(a_d * s_c - rgb)
        l_recon = zero
        if self.lambda_recon > 0.0 and rgb is not None and 's_c' in predictions:
            l_recon = self.loss_recon(predictions['a_d'], predictions['s_c'], rgb, loss_mask)
            lc = lc + self.lambda_recon * l_recon
            lc_details['loss_c_recon'] = l_recon.detach()

        # 5. V15 auxiliary edge supervision for CCR Encoder
        # Moved out of loss_c so it is not affected by w_c multiplier.
        l_ccr_edge = zero
        ccr_edge_details = {'loss_ccr_edge': zero.detach()}
        if self.lambda_ccr_edge > 0.0 and 'ccr_edge_pred' in predictions and 'ccr_edge_gt' in targets:
            pred = predictions['ccr_edge_pred']
            gt = targets['ccr_edge_gt']
            # Compute BCE over valid pixels
            bce = self.bce_logits(pred, gt)
            bce_masked = bce * loss_mask.float()
            l_ccr_edge = bce_masked.sum() / (loss_mask.float().sum() + 1e-6)
            
            l_ccr_edge = self.lambda_ccr_edge * l_ccr_edge
            ccr_edge_details['loss_ccr_edge'] = l_ccr_edge.detach()

        la, la_details = zero, {}
        lb, lb_details = zero, {}
        
        if self.w_a > 0.0 and 'd_g' in predictions and targets.get('D_g_star') is not None:
            la, la_details = self.loss_a(predictions['d_g'], targets['D_g_star'], loss_mask)
            la = self.w_a * la

        if self.w_b > 0.0 and 'xi' in predictions and targets.get('xi_star') is not None:
            lb, lb_details = self.loss_b(predictions['xi'], targets['xi_star'], loss_mask)
            lb = self.w_b * lb

        lc = self.w_c * lc
        ld = self.w_d * ld
        loss_total = la + lb + lc + ld + l_ccr_edge
        
        out = {
            # Keep both keys for compatibility with old and new callers.
            'loss': loss_total,
            'loss_total': loss_total,
            'loss_a': la,
            'loss_b': lb,
            'loss_c': lc,
            'loss_d': ld,
        }
        
        out.update(la_details)
        out.update(lb_details)
        out.update(lc_details)
        out.update(ld_details)
        out.update(ccr_edge_details)
        return out