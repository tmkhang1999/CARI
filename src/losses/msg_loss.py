import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.filters as kn_filters
import kornia.morphology as kn_morph

def compute_scale_and_shift(prediction, target, mask):
    a_00 = torch.sum(mask * prediction * prediction, dim=(1, 2, 3))
    a_01 = torch.sum(mask * prediction, dim=(1, 2, 3))
    a_11 = torch.sum(mask, dim=(1, 2, 3))

    b_0 = torch.sum(mask * prediction * target, dim=(1, 2, 3))
    b_1 = torch.sum(mask * target, dim=(1, 2, 3))

    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1

def compute_ssi_pred(pred, grnd, mask):
    scale, shift = compute_scale_and_shift(pred, grnd, mask)
    scale = torch.nn.functional.relu(scale)
    
    # Broadcast to match pred shape (B, C, H, W)
    scale = scale.view(-1, 1, 1, 1)
    shift = shift.view(-1, 1, 1, 1)
    return (pred * scale) + shift

def resize_aa(img, scale: int):
    if scale == 0:
        return img
    scaled = torch.nn.functional.interpolate(
        img,
        scale_factor=1/(2**scale),
        mode='bilinear',
        align_corners=True,
        antialias=True
    )
    return scaled

class ImageDerivative(nn.Module):
    def __init__(self):
        super().__init__()
        tap_3 = torch.tensor([
            [0.425287, -0.0000, -0.425287],
            [0.229879, 0.540242, 0.229879]])
        self.register_buffer('kernel_p', tap_3[1:2, ...])
        self.register_buffer('kernel_d', tap_3[0:1, ...])

    def forward(self, img):
        grad_x = kn_filters.filter2d_separable(
            img,
            self.kernel_p,
            self.kernel_d,
            border_type='reflect',
            normalized=False,
            padding='same'
        )
        grad_y = kn_filters.filter2d_separable(
            img,
            self.kernel_d,
            self.kernel_p,
            border_type='reflect',
            normalized=False,
            padding='same'
        )
        return grad_x, grad_y

class MultiScaleGradientLoss(nn.Module):
    def __init__(self, scales=4):
        super().__init__()
        self.n_scale = scales
        self.imgDerivative = ImageDerivative()
        
        # Hardcode taps=1 (which corresponds to tap_3 in chrislib)
        self.erod_kernels = [torch.ones(3, 3) for _ in range(scales)]

    def forward(self, output, target, mask=None, p=1):
        diff = output - target

        if mask is None:
            mask = torch.ones(diff.shape[0], 1, diff.shape[2], diff.shape[3], device=diff.device)
            
        # Ensure erod_kernels are on the right device
        erod_kernels = [k.to(diff.device) for k in self.erod_kernels]

        loss = 0.0
        for i in range(self.n_scale):
            mask_resized = torch.floor(resize_aa(mask, i) + 0.001)
            mask_resized = kn_morph.erosion(mask_resized, erod_kernels[i])
            diff_resized = resize_aa(diff, i)

            grad_x, grad_y = self.imgDerivative(diff_resized)

            # Gradient magnitude: always L2 Euclidean norm of the gradient vector.
            # This matches chrislib's gradient_mag() exactly.
            grad_mag = torch.sqrt(torch.pow(grad_x, 2) + torch.pow(grad_y, 2) + 1e-8)

            # Mean over channels (C) before aggregation.
            grad_mag = torch.mean(grad_mag, dim=1, keepdim=True)

            # p controls the aggregation norm, not the magnitude formula:
            #   p=1 -> L1 mean of magnitudes (constant gradient, preserves sharp edges)
            #   p=2 -> MSE mean of squared magnitudes (larger errors penalized harder,
            #          enforces smoothness; appropriate for chroma/smooth signals)
            if p == 2:
                grad_mag = grad_mag.pow(2)

            mask_sum = torch.sum(mask_resized)
            if mask_sum > 0:
                loss += torch.sum(mask_resized * grad_mag) / mask_sum

        return loss / self.n_scale
