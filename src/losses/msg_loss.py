"""Multi-scale gradient (MSG) loss for structural fidelity.

Computes gradient magnitude of the pred-target difference at multiple
spatial scales, then aggregates with L1 (p=1) or L2 (p=2) norm.

Filter options:
  'farid': Farid-Simoncelli 3-tap filters (better gradient accuracy,
           optimal frequency response). Normalized so step-edge response
           matches Sobel, keeping lambda_msg consistent.
  'sobel': Standard 3x3 Sobel operators (proven baseline).

Farid filters are the default: they have better directional accuracy and
less angular error than Sobel, which matters for anisotropic textures in
albedo and smooth shading gradients. The normalization factor (9.16x)
ensures the loss magnitude is identical to Sobel at the same lambda.

p parameter controls task-specific behavior:
  p=1: L1 of gradient magnitudes. Constant penalty. Preserves sharp edges.
       Use for albedo and specular residual.
  p=2: L2 (squared) of gradient magnitudes. Penalizes large gradient errors
       harder. Enforces smoothness. Use for shading branches.

target_grad_threshold enables a CRefNet-style reflectance-gradient filter:
  gradients in the target below the threshold are treated as zero, so small
  target ramps/noise do not become albedo edges the model is asked to copy.
  Leave at 0.0 for the original MSG behavior.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_farid_kernels():
    """Farid-Simoncelli 3-tap separable filters, normalized to Sobel scale.

    Raw Farid step-edge response is 0.4632, Sobel is 4.2426.
    We scale by 4.2426/0.4632 = 9.16 so gradient magnitudes match Sobel,
    keeping lambda_msg interchangeable between filter choices.
    """
    smooth = torch.tensor([0.229879, 0.540242, 0.229879])
    diff = torch.tensor([0.425287, 0.0, -0.425287])

    # 2D kernels via outer product
    kx = smooth.unsqueeze(1) @ diff.unsqueeze(0)    # smooth vert, diff horiz
    ky = diff.unsqueeze(1) @ smooth.unsqueeze(0)     # diff vert, smooth horiz

    # Normalize: step-edge response from 0.4632 -> 4.2426 (Sobel equivalent)
    norm_factor = 4.0 / (smooth.sum() * diff[0]).abs()
    kx = kx * norm_factor
    ky = ky * norm_factor

    return kx.view(1, 1, 3, 3), ky.view(1, 1, 3, 3)


def _build_sobel_kernels():
    """Standard 3x3 Sobel operators."""
    kx = torch.tensor(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]], dtype=torch.float32
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1, -2, -1],
         [ 0,  0,  0],
         [ 1,  2,  1]], dtype=torch.float32
    ).view(1, 1, 3, 3)
    return kx, ky


class MultiScaleGradientLoss(nn.Module):
    def __init__(self, scales=4, erode=True, filters='farid'):
        """
        Args:
            scales: number of downsampling scales (0 = original resolution).
            erode: if True, erode the mask by 1px at each scale to avoid
                   boundary gradient artifacts. Uses max_pool2d (no kornia).
            filters: 'farid' (default, better accuracy) or 'sobel' (proven).
        """
        super().__init__()
        
        self.n_scale = scales
        self.erode = erode

        if filters == 'farid':
            kx, ky = _build_farid_kernels()
        elif filters == 'sobel':
            kx, ky = _build_sobel_kernels()
        else:
            raise ValueError(f"Unknown filter type: {filters}. Use 'farid' or 'sobel'.")

        self.register_buffer('kernel_x', kx, persistent=False)
        self.register_buffer('kernel_y', ky, persistent=False)

    @staticmethod
    def _erode_mask(mask):
        """Erode binary mask by 1px using inverted max_pool2d (no kornia)."""
        inv = 1.0 - mask
        dilated_inv = F.max_pool2d(inv, kernel_size=3, stride=1, padding=1)
        return (1.0 - dilated_inv).clamp(0.0, 1.0)

    def _compute_gradient(self, x):
        """Gradient via grouped convolution (one pass per axis)."""
        C = x.shape[1]
        weight_x = self.kernel_x.repeat(C, 1, 1, 1).to(x.dtype)
        weight_y = self.kernel_y.repeat(C, 1, 1, 1).to(x.dtype)
        grad_x = F.conv2d(x, weight_x, padding=1, groups=C)
        grad_y = F.conv2d(x, weight_y, padding=1, groups=C)
        return grad_x, grad_y

    def forward(self, pred, target, mask=None, p=1, target_grad_threshold=0.0):
        """
        Args:
            pred:   (N, C, H, W) prediction
            target: (N, C, H, W) ground truth
            mask:   (N, 1, H, W) valid pixel mask (None => all valid)
            p:      aggregation norm. 1 = L1 (sharp edges), 2 = L2 (smooth)
            target_grad_threshold: if > 0, suppress target gradients whose
                    magnitude is below this threshold before comparing against
                    prediction gradients. This preserves sharp reflectance
                    edges while discouraging copied low-contrast albedo ramps.
        """
        grad_threshold = float(target_grad_threshold or 0.0)
        use_target_filter = grad_threshold > 0.0
        diff = None if use_target_filter else pred - target

        if mask is None:
            mask = torch.ones(
                pred.shape[0], 1, pred.shape[2], pred.shape[3],
                device=pred.device, dtype=pred.dtype,
            )

        loss = torch.zeros((), device=pred.device, dtype=pred.dtype)
        for scale in range(self.n_scale):
            if scale > 0:
                scale_factor = 1.0 / (2 ** scale)
                if use_target_filter:
                    pred_scaled = F.interpolate(
                        pred, scale_factor=scale_factor,
                        mode='bilinear', align_corners=True, antialias=True,
                    )
                    target_scaled = F.interpolate(
                        target, scale_factor=scale_factor,
                        mode='bilinear', align_corners=True, antialias=True,
                    )
                else:
                    diff_scaled = F.interpolate(
                        diff, scale_factor=scale_factor,
                        mode='bilinear', align_corners=True, antialias=True,
                    )
                mask_scaled = F.interpolate(
                    mask.float(), scale_factor=scale_factor,
                    mode='bilinear', align_corners=True, antialias=True,
                )
                mask_scaled = torch.floor(mask_scaled + 0.001).clamp(0, 1)
            else:
                if use_target_filter:
                    pred_scaled = pred
                    target_scaled = target
                else:
                    diff_scaled = diff
                mask_scaled = mask.float()

            if self.erode:
                mask_scaled = self._erode_mask(mask_scaled)

            if use_target_filter:
                pred_x, pred_y = self._compute_gradient(pred_scaled)
                target_x, target_y = self._compute_gradient(target_scaled)
                target_mag = torch.sqrt(target_x.pow(2) + target_y.pow(2) + 1e-8)
                target_keep = (target_mag > grad_threshold).to(target_x.dtype)
                grad_x = pred_x - target_x * target_keep
                grad_y = pred_y - target_y * target_keep
            else:
                grad_x, grad_y = self._compute_gradient(diff_scaled)

            # Gradient magnitude (L2 norm of gradient vector per channel)
            grad_mag = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-8)

            # Average across channels before spatial aggregation
            grad_mag = torch.mean(grad_mag, dim=1, keepdim=True)

            # p=2: square the magnitudes to penalize large gradients harder
            if p == 2:
                grad_mag = grad_mag.pow(2)

            mask_sum = mask_scaled.sum()
            if mask_sum > 0:
                loss += (mask_scaled * grad_mag).sum() / mask_sum

        return loss / self.n_scale
