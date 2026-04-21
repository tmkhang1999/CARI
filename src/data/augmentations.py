import random
import numpy as np
import torch
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
