import random
import numpy as np
import torch
import torchvision.transforms.functional as TF
import cv2

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

def random_segmentation_degradation(seg_t, p_degrade=0.7):
    """
    Randomly erodes or dilates segmentation masks to simulate imperfect boundaries.
    Also adds translation shifts to simulate misaligned masks.
    Args:
        seg_t: Tensor (1, H, W) or (H, W)
    """
    if np.random.rand() > p_degrade:
        return seg_t
        
    device = seg_t.device
    seg_np = seg_t.detach().cpu().numpy().squeeze()
    
    classes = np.unique(seg_np)
    if len(classes) <= 1:
        return seg_t
        
    # Pick a few classes to distort
    counts = [np.sum(seg_np == c) for c in classes]
    dominant_class = classes[np.argmax(counts)]
    target_candidates = [c for c in classes if c != dominant_class]
    
    if not target_candidates:
        return seg_t
        
    # STRONGER: Distort more classes (up to 10)
    num_to_distort = min(np.random.randint(4, 10), len(target_candidates))
    targets = np.random.choice(target_candidates, num_to_distort, replace=False)
    
    seg_aug = seg_np.copy()
    for c in targets:
        bin_mask = (seg_np == c).astype(np.uint8)
        
        # STRONGER: Even larger kernel sizes (up to 20)
        kernel_size = np.random.randint(10, 20)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        if np.random.rand() > 0.5:
            # Erosion: class shrinks
            eroded = cv2.erode(bin_mask, kernel)
            seg_aug[(bin_mask == 1) & (eroded == 0)] = dominant_class
        else:
            # Dilation: class expands
            dilated = cv2.dilate(bin_mask, kernel)
            seg_aug[(bin_mask == 0) & (dilated == 1)] = c
            
    return torch.from_numpy(seg_aug).to(device=device, dtype=seg_t.dtype).reshape(seg_t.shape)

def apply_physical_augmentations(t, seg_t, 
                                 p_hflip=0.5, p_vflip=0.5,
                                 force_type=None):
    """
    Applies a structured suite of physical augmentations to the intrinsic image tensors.
    """
    # ── 1. Spatial Augmentations (Can happen concurrently) ──
    if force_type == 'spatial' or (force_type is None and np.random.rand() < p_hflip):
        t     = torch.flip(t,     dims=[2])
        seg_t = torch.flip(seg_t, dims=[2])
    if force_type == 'spatial' or (force_type is None and np.random.rand() < p_vflip):
        t     = torch.flip(t,     dims=[1])
        seg_t = torch.flip(seg_t, dims=[1])

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

        # 4b. Ambient Tint (Trains CCR to ignore colored shadows)
        if force_type is None:
            if np.random.rand() < 0.5:
                ambient_tint = torch.tensor([
                                   np.random.uniform(-0.02, 0.05),
                                   np.random.uniform(-0.02, 0.05),
                                   np.random.uniform(-0.02, 0.05)
                               ], dtype=torch.float32, device=S_new.device).view(3, 1, 1)
                S_new = (S_new + ambient_tint).clamp_min(1e-4)
        else:
            # For forced visualization, add a noticeable but realistic tint
            ambient_tint = torch.tensor([0.05, 0.01, 0.05], dtype=torch.float32, device=S_new.device).view(3, 1, 1)
            S_new = (S_new + ambient_tint).clamp_min(1e-4)

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
        
    # ── 5. Segmentation Augmentation ──
    if force_type == 'seg_degrade':
        seg_t = random_segmentation_degradation(seg_t, p_degrade=1.0)
        
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
