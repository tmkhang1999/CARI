import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

def random_color_shift(image, mean=0.5, std=0.05):
    """
    Simulates white balance errors while preserving luminance.
    MUST match chrislib implementation.
    """
    # Sample RGB gains
    temperature = np.random.normal(mean, std, size=3)

    # SAFEGUARD: prevent channel collapse
    temperature = temperature.clip(0.1, 1.0)

    # NORMALIZE: preserve global luminance
    temperature /= (np.sum(temperature) / 3.0)

    gain = torch.from_numpy(temperature.reshape(3, 1, 1)).to(device=image.device, dtype=image.dtype)

    augmented = (image * gain).clamp(0, 1).float()
    return augmented, gain

def random_hue_saturation_shifting(albedo):
    """
    Randomly shifts hue and saturation of the input albedo tensor.
    """
    hue_shft = (random.randint(0, 50) / 50.) - 0.5
    hue_img = TF.adjust_hue(albedo, hue_shft)
    sat_shft = (random.randint(0, 50) / 50.) + 0.5
    return TF.adjust_saturation(hue_img, sat_shft)

def random_scaling_red_blue_channels(albedo, scale_range=(0.8, 1.2)):
    """
    Randomly scales the red and blue channels of the input albedo tensor.
    """
    r = random.uniform(*scale_range)
    b = random.uniform(*scale_range)

    gain = torch.tensor([r, 1.0, b], device=albedo.device, dtype=albedo.dtype).view(3, 1, 1)

    # Normalize to prevent brightness drift
    gain = gain / (gain.mean() + 1e-6)

    return (albedo * gain).clamp(0, 1)

def random_colored_illumination(S_diff, max_tint=0.25, p_spatial=0.5):
    """Simulate colored shadows / multi-illuminant lighting on the shading layer.

    CCR (cross color ratio) cancels a GLOBAL colored cast (its zero-sum kernel
    differentiates a constant away) but provably CANNOT cancel SPATIALLY-VARYING
    colored illumination — at a single boundary a material edge and an
    illumination-color edge are indistinguishable. So colored shadows leak into
    albedo unless the decoder is shown such cases with flat-albedo / colored-shading
    supervision. Two modes:
      - 'ambient' (additive): a constant ambient color vector. Where shading is
        bright the color barely changes; where it is dark (shadow) it takes on the
        ambient tint — a first-order colored-shadow model.
      - 'spatial' (multiplicative): a smooth low-frequency per-channel gain field so
        different regions carry different illumination color (e.g. warm interior vs
        cool window light) — the case CCR cannot disentangle.
    Per-pixel luminance is preserved so albedo scale-matching stays stable.
    """
    C, H, W = S_diff.shape
    device, dtype = S_diff.device, S_diff.dtype
    if random.random() < p_spatial:
        # smooth 3x3 colored gain field, upsampled; chroma-only (luminance preserved)
        low = 1.0 + (torch.rand(1, 3, 3, 3, device=device, dtype=dtype) - 0.5) * 2.0 * max_tint
        field = F.interpolate(low, size=(H, W), mode='bilinear', align_corners=True)[0]
        field = field / field.mean(dim=0, keepdim=True).clamp_min(1e-6)
        return (S_diff * field).clamp_min(1e-4)
    tint = (torch.rand(3, 1, 1, device=device, dtype=dtype) - 0.25) * max_tint
    return (S_diff + tint).clamp_min(1e-4)


def apply_physical_augmentations(t, seg_t,
                                 p_hflip=0.5, p_vflip=0.5,
                                 force_type=None, apply_photometric=True):
    """
    Applies a structured suite of physical augmentations to the intrinsic image tensors.

    apply_photometric: when False, only the spatial flips run (they apply uniformly
    to every channel of t, including an appended rgb2). The photometric shifts
    (white balance / albedo / shadow) are skipped because they modify t[0:3] using
    the albedo/shading channels and would desynchronise a same-albedo image pair.
    """
    # ── 1. Spatial Augmentations (Can happen concurrently) ──
    if force_type == 'spatial' or (force_type is None and np.random.rand() < p_hflip):
        t     = torch.flip(t,     dims=[2])
        seg_t = torch.flip(seg_t, dims=[2])
    if force_type == 'spatial' or (force_type is None and np.random.rand() < p_vflip):
        t     = torch.flip(t,     dims=[1])
        seg_t = torch.flip(seg_t, dims=[1])

    if not apply_photometric:
        return t, seg_t

    # ── MUTUALLY EXCLUSIVE PHYSICAL AUGMENTATIONS ──
    # Default training: only apply ONE major photometric shift per image.
    # force_type: allows selecting specific ones for visualization summary.
    
    if force_type is not None:
        aug_roll = -1.0 # Bypass random roll
    else:
        aug_roll = np.random.rand()

    if force_type == 'wb' or (aug_roll >= 0 and aug_roll < 0.33):
        # ── 2. Global White Balance Augmentation ──
        rgb_view, illum_view = t[0:3], t[6:9]
        _, gain = random_color_shift(rgb_view, mean=0.5, std=0.05)

        # Global Max Normalization instead of Reinhard.
        # This keeps RGB in [0,1], perfectly preserves inter-channel color ratios
        # (for CCR), and prevents the target shading from being contaminated by
        # albedo edges (which happens when RGB is non-linearly compressed but
        # Illumination is not, or when I != A * S).
        rgb_gained = rgb_view * gain
        illum_gained = illum_view * gain

        max_val = rgb_gained.max()
        if max_val > 1.0:
            t[0:3] = rgb_gained / max_val
            t[6:9] = (illum_gained / max_val).clamp(0.0, 60000.0)
        else:
            t[0:3] = rgb_gained
            t[6:9] = illum_gained.clamp(0.0, 60000.0)

    elif force_type in ('albedo_hue', 'albedo_scale') or (aug_roll >= 0 and aug_roll < 0.66):
        # ── 3. Albedo Augmentation (Paint Changes) ──
        I_orig, A_orig, S_diff = t[0:3], t[3:6], t[6:9]
        R_specular = (I_orig - (A_orig * S_diff)).clamp_min(0.0)
        
        if force_type == 'albedo_hue' or (force_type is None and np.random.rand() < 0.5):
            A_new = random_hue_saturation_shifting(A_orig)
        else:
            A_new = random_scaling_red_blue_channels(A_orig)
            
        I_new = (A_new * S_diff) + R_specular

        t[3:6] = A_new
        # Global max normalization to maintain I = A * S
        max_val = I_new.max()
        if max_val > 1.0:
            t[0:3] = I_new / max_val
            # Must scale illumination too, otherwise I != A * S
            t[6:9] = S_diff / max_val
        else:
            t[0:3] = I_new
            # t[6:9] = S_diff  (already assigned above)
        
    elif force_type in ('shadow_weak', 'shadow_strong') or (aug_roll >= 0.66):
        # ── 4. Shadow & Illumination Augmentation ──
        I_orig, A_orig, S_diff = t[0:3], t[3:6], t[6:9]
        R_specular = (I_orig - (A_orig * S_diff)).clamp_min(0.0)
        
        S_new = S_diff.clone()
        
        # 4a. Global Contrast (Harsh vs Soft existing shadows)
        if force_type == 'shadow_weak':
            gamma = 0.6
        elif force_type == 'shadow_strong':
            gamma = 1.5
        else:
            gamma = np.random.uniform(0.6, 1.5)
            
        # Raise upper cap from 5.0 to preserve HDR illumination targets near lights.
        S_new = S_new.clamp(min=1e-4).pow(gamma).clamp(max=60000.0)

        # 4b. Colored illumination (teaches the decoder to keep albedo flat under
        # colored shadows / multi-illuminant scenes — CCR alone cannot, since it
        # only cancels a global cast, not spatially-varying illumination color).
        if force_type is None:
            if np.random.rand() < 0.8:
                S_new = random_colored_illumination(S_new, max_tint=0.25, p_spatial=0.5)
        else:
            # Forced visualization: spatial field for 'strong', ambient tint for 'weak'
            S_new = random_colored_illumination(
                S_new, max_tint=0.25,
                p_spatial=1.0 if force_type == 'shadow_strong' else 0.0,
            )

        # Recombine physically
        I_new = (A_orig * S_new) + R_specular

        # Global max normalization to maintain I = A * S
        max_val = I_new.max()
        if max_val > 1.0:
            t[0:3] = I_new / max_val
            t[6:9] = S_new / max_val
        else:
            t[0:3] = I_new
            t[6:9] = S_new
        
    return t, seg_t


def random_exposure_jitter(t, p=0.3, log_range=(-3.0, 3.0)): 
    """Simulate exposure variation by scaling rgb and illumination jointly.

    Uses global max normalization to bound RGB to [0,1] when the exposure
    scale pushes highlights above 1.0. Both RGB and illumination are divided
    by the same scalar, preserving the physical relationship I = A * S and
    inter-channel ratios for CCR.

    Args:
        t: stacked tensor (12, H, W) with layout [rgb(3), albedo(3), illum(3), normals(3)]
        p: probability of applying jitter
        log_range: (min, max) for log-uniform exposure scale.
    """
    if np.random.rand() > p:
        return t

    log_scale = np.random.uniform(*log_range)
    scale = float(np.exp(log_scale))

    rgb_scaled = t[0:3] * scale
    illum_scaled = t[6:9] * scale

    # Global max normalization to maintain I = A * S while bounding RGB to [0,1]
    max_val = rgb_scaled.max()
    if max_val > 1.0:
        t[0:3] = rgb_scaled / max_val
        t[6:9] = (illum_scaled / max_val).clamp(0.0, 60000.0)
    else:
        t[0:3] = rgb_scaled
        t[6:9] = illum_scaled.clamp(0.0, 60000.0)

    return t
