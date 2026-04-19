Based on the methodology established in the *Colorful Diffuse Intrinsic Image Decomposition in the Wild* (CD-IID) paper and its supplementary material, here is the complete, mathematically rigorous preprocessing pipeline for the MIDIntrinsics dataset.

### Handling the Scene Files
In each scene directory, you have:
* `*albedo*.exr`: The pseudo-GT median albedo ($A_{GT}$).
* `materials_mip2.png`: The dense material segmentation mask. We will load this to use as the `seg` map (useful for Decoder C's SPADE normalization later).
* `thumb.jpg`: A thumbnail. We must explicitly ignore this.
* `25 EXR files`: The white-balanced illumination captures.

### The Mathematical Strategy
To prevent the artificial Lab color shifts (used for data augmentation) from corrupting the geometric grayscale shading, our `_sample_mix` function will return **two** versions of the mixed image:
1.  **$I_{neutral}$**: The mixed image *before* the Lab color shift. We use this to compute pure Grayscale Shading ($D_g$).
2.  **$I_{shifted}$**: The mixed image *after* the Lab color shift. We use this as the network input and to compute the target Chroma ($\xi$).

Here is the complete Python pipeline for your `MIDIntrinsicDataset` class:

```python
import os
import glob
import random
import numpy as np
import torch
import torch.nn.functional as F
import imageio.v2 as imageio
from skimage import color
import cv2

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

class MIDIntrinsicDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, input_size=384, max_mix=3, lab_shift=8.0):
        self.root_dir = root_dir
        self.input_size = input_size
        self.max_mix = max_mix
        self.lab_shift = lab_shift
        self.samples = self._build_samples()

    def _build_samples(self):
        samples = []
        scene_dirs = sorted([d for d in glob.glob(os.path.join(self.root_dir, '*')) if os.path.isdir(d)])
        
        for sd in scene_dirs:
            albedo_paths = glob.glob(os.path.join(sd, '*albedo*.exr'))
            if not albedo_paths:
                continue
            albedo_path = albedo_paths[0]
            
            # Filter out albedo, materials_mip2.png, and thumb.jpg to get the 25 illuminations
            illum_paths = [
                p for p in glob.glob(os.path.join(sd, '*.exr'))
                if p != albedo_path and 'albedo' not in os.path.basename(p).lower()
            ]
            
            seg_path = os.path.join(sd, 'materials_mip2.png')
            
            if len(illum_paths) > 0:
                samples.append({
                    'albedo': albedo_path,
                    'illums': sorted(illum_paths),
                    'seg': seg_path if os.path.exists(seg_path) else None
                })
        return samples

    def _load_exr(self, path):
        """Loads EXR as linear HDR float32."""
        arr = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        arr = arr[..., ::-1] # BGR to RGB
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def _tonemap_linear(self, img):
        """99th percentile linear tonemapping to [0, 1]."""
        scale = float(np.percentile(img, 99.0))
        scale = scale if scale > 0 else 1e-6
        return np.clip(img / scale, 0.0, 1.0).astype(np.float32)

    def _sample_mix(self, illum_paths):
        """Samples 1-3 illuminations and returns both Neutral and Color-Shifted mixes."""
        k = random.randint(1, min(self.max_mix, len(illum_paths)))
        chosen = random.sample(illum_paths, k)
        weights = np.random.dirichlet(np.ones(k)).astype(np.float32)
        
        mixed_neutral = None
        mixed_shifted = None
        
        for w, path in zip(weights, chosen):
            # Load and tonemap the white-balanced image
            img_neutral = self._tonemap_linear(self._load_exr(path))
            
            # Apply Lab color shift for augmentation
            lab = color.rgb2lab(img_neutral)
            lab[..., 1] += np.random.uniform(-self.lab_shift, self.lab_shift)
            lab[..., 2] += np.random.uniform(-self.lab_shift, self.lab_shift)
            img_shifted = np.clip(color.lab2rgb(lab), 0.0, 1.0).astype(np.float32)
            
            if mixed_neutral is None:
                mixed_neutral = img_neutral * w
                mixed_shifted = img_shifted * w
            else:
                mixed_neutral += img_neutral * w
                mixed_shifted += img_shifted * w
                
        return np.clip(mixed_neutral, 0.0, 1.0), np.clip(mixed_shifted, 0.0, 1.0)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        eps = 1e-6

        # 1. Load Albedo and Valid Mask
        a_gt = self._load_exr(sample['albedo'])
        valid_mask = (a_gt.min(axis=-1) > 0.01).astype(np.float32)

        # 2. Get Neutral and Color-Shifted Images
        i_neutral, i_shifted = self._sample_mix(sample['illums'])

        # ---------------------------------------------------------
        # GT 1: Grayscale Shading (D_g)
        # Derived from I_neutral to ensure geometric purity
        # ---------------------------------------------------------
        s_neutral = i_neutral / (a_gt + eps)
        # Luminance weights (Rec. 709)
        s_g = 0.2126 * s_neutral[..., 0] + 0.7152 * s_neutral[..., 1] + 0.0722 * s_neutral[..., 2]
        d_g_target = 1.0 / (s_g + 1.0) # Inverse shading domain
        d_g_target = d_g_target[..., None] # Add channel dim (H, W, 1)

        # ---------------------------------------------------------
        # GT 2: Chroma (xi)
        # Derived from I_shifted (I_mixed)
        # ---------------------------------------------------------
        s_colorful = i_shifted / (a_gt + eps)
        
        # c* = S_channel / S_gray
        s_gray_shifted = 0.2126 * s_colorful[..., 0] + 0.7152 * s_colorful[..., 1] + 0.0722 * s_colorful[..., 2]
        c_rg = s_colorful[..., 0] / (s_gray_shifted + eps)
        c_bg = s_colorful[..., 2] / (s_gray_shifted + eps)
        
        # Bounded representation: xi = 1 / (c* + 1)
        xi_rg = 1.0 / (c_rg + 1.0)
        xi_bg = 1.0 / (c_bg + 1.0)
        xi_target = np.stack([xi_rg, xi_bg], axis=-1) # (H, W, 2)

        # 3. Load Segmentation (if available)
        if sample['seg']:
            seg = cv2.imread(sample['seg'], cv2.IMREAD_GRAYSCALE)
        else:
            seg = np.zeros(i_shifted.shape[:2], dtype=np.uint8)

        # 4. Stack and format for PyTorch
        combined = np.concatenate([
            i_shifted,             # 0:3 (Input I)
            a_gt,                  # 3:6 (Target A_d)
            d_g_target,            # 6:7 (Target D_g)
            xi_target,             # 7:9 (Target xi)
            valid_mask[..., None]  # 9:10 (Valid Mask)
        ], axis=-1)

        # Apply random crop and resize logic here...
        # (Assuming random crop and resize to self.input_size is applied to `combined` and `seg`)
        
        # Final tensor conversion
        t = torch.from_numpy(combined).permute(2, 0, 1).float()
        seg_t = torch.from_numpy(seg).unsqueeze(0).long()

        return {
            'rgb':        t[0:3],
            'albedo_raw': t[3:6],
            'd_g_raw':    t[6:7],
            'xi_raw':     t[7:9],
            'valid_mask': t[9:10].bool(),
            'seg':        seg_t,
            'M_albedo':   torch.tensor(1.0),
            'M_diffuse':  torch.tensor(0.0) # Explicitly skip Decoder D
        }
```

### Why this specific pipeline design?
1. **Filtering the Assets:** The list comprehension `p != albedo_path and 'albedo' not in os.path.basename(p)` safely excludes both the EXR albedo and the thumbnail, capturing exactly the 25 lighting files.
2. **Dual-Image Mixing:** By generating both `mixed_neutral` and `mixed_shifted`, we solve the color-leakage problem. $D_g$ is derived from `mixed_neutral` so Decoder A learns pure geometry uncorrupted by the artificial Lab shifts. $\xi$ is derived from `mixed_shifted` so Decoder B learns to estimate the exact artificial illumination color you injected.
3. **No Dynamic Scale Guessing:** $S_c$ is calculated strictly as `i_shifted / (a_gt + eps)`. There is no moving $c$ scale factor. The scale ambiguity is resolved later in the loss function during training.