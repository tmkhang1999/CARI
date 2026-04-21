Here’s the **updated, stricter agent prompt** incorporating the official implementation details, safeguards, and correct logic:

---

## **Prompt (Updated with Official CD-IID Logic)**

Update the dataset pipeline to implement **CD-IID augmentations exactly as defined in `chrislib/data_util.py` and the supplementary material**, including all numerical safeguards and physically consistent re-derivations under:

[
I = A \times S
]

---

## **1. Implement Augmentations (Authoritative Code)**

```python
import torch
import numpy as np
import random
import torchvision.transforms.functional as TF

# --- 1. Chroma Stage: random_color_shift ---
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

    gain = torch.from_numpy(temperature.reshape(1, 3, 1, 1)).to(image.device)

    return (image * gain).clamp(0, 1).float()


# --- 2. Ordinal Stage: hue/saturation ---
def random_hue_saturation_shifting(albedo):
    return TF.ColorJitter(hue=0.1, saturation=0.2)(albedo)


# --- 3. Ordinal Stage: red/blue scaling ---
def random_scaling_red_blue_channels(albedo, scale_range=(0.6, 1.4)):
    r = random.uniform(*scale_range)
    b = random.uniform(*scale_range)

    gain = torch.tensor([r, 1.0, b], device=albedo.device).view(1, 3, 1, 1)

    # Normalize to prevent brightness drift
    gain = gain / (gain.mean() + 1e-6)

    return albedo * gain


# --- 4. Shared spatial augmentation ---
def random_flip(components, p=0.5):
    if random.random() > p:
        return [TF.hflip(x) for x in components]
    return components
```

---

## **2. Apply Augmentations in Dataset**

```python
# Inputs: I (image), A (albedo), S (shading)

if stage == "ordinal":
    # Apply ONLY to albedo
    A = random_hue_saturation_shifting(A)
    A = random_scaling_red_blue_channels(A)

    # Re-derive input image (critical)
    I = A * S

elif stage == "chroma":
    # Apply ONLY to image
    I = random_color_shift(I)

    # Re-derive shading (critical)
    S = I / (A + 1e-6)

# Apply shared spatial augmentation (must be identical)
I, A, S = random_flip([I, A, S])
```

---

## **3. Critical Constraints (Strict)**

* **Do NOT mix stages**

  * `random_color_shift` → ONLY in `chroma`
  * others → ONLY in `ordinal`

* **Always enforce physical consistency:**

  * If **I changes → recompute S**
  * If **A changes → recompute I**

* **Numerical safeguards are mandatory:**

  * Clip temperature to `[0.1, 1.0]`
  * Normalize gains to preserve luminance
  * Clamp outputs where specified

* **Apply augmentations in linear RGB space ONLY**

  * No gamma correction

* **Maintain tensor shape:** `(B, 3, H, W)`

* **Apply spatial transforms identically to all components**

---

## **4. Expected Outcome**

The updated pipeline must:

* Simulate **realistic white balance errors**
* Increase **material/color diversity**
* Prevent **RGB-ratio overfitting**
* Preserve **physical correctness of intrinsic decomposition**

---

## **5. Non-Negotiable Rule**

Every augmentation must respect:

[
I = A \times S
]

No exceptions. No shortcuts.

---

If needed, extend this to include masks, HDR handling, or crop/resize—but do not break the augmentation → re-derivation order.
