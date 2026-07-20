# Data Processing & Dataloader Supplement

**Project:** Physically-Grounded Intrinsic Image Decomposition for HDR Real Estate Retouching

This document covers dataset downloading, HDR data handling, preprocessing, and dataloader implementation for Stage 1 training. Read alongside `IID_Project_Documentation.md`.

---

## Phase 0: Dataset Size Strategy

### Why You Do Not Need All 74,000 Hypersim Images

The full Hypersim dataset contains ~74,000 images across ~460 scenes. Each image requires 6 `.hdf5` files (Color, Albedo, Diffuse Illumination, Residual, Normals, Semantics), totalling ~440,000 files and ~1.9TB on disk.

However, Hypersim is generated from camera trajectories. `frame.0000`, `frame.0001`, and `frame.0002` are simply the camera moving a few inches across the same room. The visual and physical difference between consecutive frames is negligible, which means they provide near-zero additional learning value to the network while tripling training time and storage cost.

### Recommended Strategy: Trajectory Decimation

Keep every 3rd–4th frame per camera trajectory. This eliminates redundant frames while retaining 100% of scene diversity across all rooms and lighting conditions.

```
Full dataset:       ~74,000 images   ~1.9TB    (do not use)
After decimation:   ~20,000–25,000   ~100–150GB (production training)
Sanity check set:   ~500 images      ~3GB       (initial debugging)
```

**Stage 1 — Sanity Check Set (~500 images)**

Use during initial development to verify the dataloader, catch `NaN` errors, test VRAM limits, and confirm the loss decreases. Load time is near-instant, iteration is fast.

```python
# Keep only the first 500 frames from the dataset index
# e.g. scene ai_001_001 cam 00, frames 0000-0010 only
```

**Stage 2 — Production Set (~20,000–25,000 images)**

Use for thesis training. Apply decimation at download time or as a post-download cleanup:

```python
import os
import glob

def decimate_hypersim(root: str, keep_every: int = 4):
    """
    Remove redundant consecutive frames from Hypersim trajectories.
    Keeps frame.0000, frame.0004, frame.0008, etc.
    Deletes all other frames to save storage.

    keep_every: stride between kept frames (4 recommended)
    """
    all_color = sorted(glob.glob(
        os.path.join(root, '**', '*.color.hdf5'), recursive=True
    ))

    # Group by scene + camera trajectory
    from collections import defaultdict
    trajectories = defaultdict(list)
    for path in all_color:
        key = os.path.dirname(path)  # scene_cam_XX_final_hdf5 directory
        trajectories[key].append(path)

    suffixes = [
        'color.hdf5', 'diffuse_reflectance.hdf5',
        'diffuse_illumination.hdf5', 'residual.hdf5'
    ]
    geo_suffixes = ['normal_cam.hdf5', 'semantic.hdf5', 'render_entity_id.hdf5']

    kept, removed = 0, 0
    for traj_dir, frames in trajectories.items():
        frames = sorted(frames)
        geo_dir = traj_dir.replace('final_hdf5', 'geometry_hdf5')

        for i, color_path in enumerate(frames):
            base = color_path.replace('color.hdf5', '')
            geo_base = os.path.join(geo_dir, os.path.basename(base))

            if i % keep_every != 0:
                # Delete all 7 modality files for this frame
                for s in suffixes:
                    p = base + s
                    if os.path.exists(p): os.remove(p)
                for s in geo_suffixes:
                    p = geo_base + s
                    if os.path.exists(p): os.remove(p)
                removed += 1
            else:
                kept += 1

    print(f"Kept {kept} frames, removed {removed} frames.")
    print(f"Estimated storage: {kept * 6 * 0.006:.1f} GB")  # ~6MB per frame set

# Usage:
# decimate_hypersim('/path/to/hypersim', keep_every=4)
```

### Storage Reality Check

| Subset | Images | Files | Storage |
|---|---|---|---|
| Full Hypersim | ~74,000 | ~440,000 | ~1.9TB |
| Decimated (every 4th) | ~18,500 | ~111,000 | ~100–150GB |
| Sanity check | ~500 | ~3,000 | ~3GB |
| **Current download** | **59 (stride=5, 2 scenes)** | **~350** | **~25MB** |

100–150GB fits comfortably on a standard SSD and loads into RAM cache efficiently within a 24GB VRAM training setup.

---

## Phase 1: Selective Dataset Downloading

### Hypersim

**Setup:**
```bash
git clone https://github.com/apple/ml-hypersim.git
cd ml-hypersim
```

**Selective download** (community script at `contrib/99991/download.py`):
```bash
# Core physics: lighting and material
python contrib/99991/download.py -c color.hdf5               -d /path/to/hypersim
python contrib/99991/download.py -c diffuse_reflectance.hdf5  -d /path/to/hypersim
python contrib/99991/download.py -c diffuse_illumination.hdf5 -d /path/to/hypersim
python contrib/99991/download.py -c residual.hdf5             -d /path/to/hypersim

# Geometry priors
python contrib/99991/download.py -c normal_cam.hdf5           -d /path/to/hypersim
python contrib/99991/download.py -c semantic.hdf5             -d /path/to/hypersim
python contrib/99991/download.py -c render_entity_id.hdf5     -d /path/to/hypersim
```

**Skip entirely:** tone-mapped previews, depth maps, meshes, bounding boxes, camera metadata.

**Run decimation immediately after download:**
```bash
python decimate_hypersim.py --root /path/to/hypersim --keep_every 4
```

**Expected file structure:**
```
hypersim/
└── ai_001_001/
    └── images/
        ├── scene_cam_00_final_hdf5/
        │   ├── frame.0000.color.hdf5
        │   ├── frame.0000.diffuse_reflectance.hdf5
        │   ├── frame.0000.diffuse_illumination.hdf5
        │   └── frame.0000.residual.hdf5
        └── scene_cam_00_geometry_hdf5/
            ├── frame.0000.normal_cam.hdf5
            ├── frame.0000.semantic.hdf5
            └── frame.0000.render_entity_id.hdf5
```

Each `.hdf5` file contains a single array at key `'dataset'`.

---

### InteriorNet

Download from official source. Required modalities: RGB image, albedo ground truth.
- Resolution: 640×480
- `M_diffuse=0, M_albedo=1` — Dec C trains; Dec A/B train via pseudo targets; Dec D is gated off

### MIDIntrinsic

Download from official source. Required modalities: RGB images across 25 illuminations per scene, pre-computed pseudo-albedo.
- Resolution: ~1000×1500
- `M_diffuse=0, M_albedo=1`

---

## Phase 2: HDR Data Handling

### 2.1 Never Convert to PNG/JPEG

**Critical rule: load directly from `.hdf5` using `h5py`. Never convert to LDR formats.**

Hypersim stores all intrinsic data as uncompressed linear `float32`. Real indoor lighting is heavily unbounded — a sunlit window might have a pixel value of `450.0` while a dark corner has `0.005`. Converting to 8-bit PNG/JPEG clips these values to `[0, 255]`, destroying the linear energy relationship:

```
I = A_d * S_d + R
```

This equation only holds in linear HDR space. Once clipped, the intrinsic decomposition is physically invalid and the inverse shading transformation `D = 1/(S+1)` becomes meaningless.

**Always read with h5py:**
```python
import h5py
import numpy as np

def load_hdf5(path: str) -> np.ndarray:
    with h5py.File(path, 'r') as f:
        return f['dataset'][:]   # returns float32 array
```

### 2.2 Tonemapping the RGB Input (Without Gamma Correction)

The raw linear HDR `color.hdf5` values are unbounded and cannot be fed directly into ConvNeXt V2, which expects inputs in a normalized `[0, 1]` range. However, standard gamma correction (`x^(1/2.2)`) must NOT be applied because it breaks linearity — and linearity is required for the intrinsic equations to hold.

Instead, apply a **simple scale-clip tonemap** that compresses the HDR range to `[0, 1]` while preserving the linear relationship between pixel values:

```python
def tonemap_linear(rgb: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    """
    Tonemap linear HDR image to [0, 1] without gamma correction.
    Linearity is preserved — only the scale changes.

    rgb:        (H, W, 3) float32, linear HDR, unbounded
    percentile: clipping point (99th percentile avoids extreme specular outliers)
    returns:    (H, W, 3) float32, linear, in [0, 1]
    """
    scale = np.percentile(rgb, percentile) + 1e-6
    return np.clip(rgb / scale, 0.0, 1.0)
```

**What this does:**
- Divides by the 99th percentile intensity, so 99% of pixels land in `[0, 1]`
- Clips extreme specular outliers (the top 1%) that would otherwise dominate the encoder's dynamic range
- Does NOT apply gamma — the output remains linear, so `I = A_d * S_d + R` still holds

**Applied to:** RGB input image only (fed to ConvNeXt V2 and CCR computation).

**NOT applied to:** Albedo GT, diffuse illumination GT, normals, or any intrinsic targets. These are kept in their original linear HDR space and transformed by their own target-space mappings (inverse shading for shading targets, left as-is for albedo which is already in `[0, 1]` by definition).

```python
# In __getitem__:
rgb_tonemapped = tonemap_linear(rgb)    # input to encoder: linear [0,1]
# All GT arrays: kept as raw float32 HDR, transformed later by loss target functions
```

---

## Phase 3: Ground Truth Preparation

GT preparation has two stages with a critical distinction:

- **Static derivation** (`__getitem__`): load raw arrays, compute validity mask, apply augmentation. Raw GT arrays are returned as-is — no target-space mapping yet.
- **Dynamic scale matching** (training loop, after forward pass): resolve scale ambiguity by fitting GT scale to the network's current predictions, then map to target spaces (inverse shading, bounded chroma).

### 3.1 Why Raw GTs Cannot Be Used Directly — Scale Ambiguity

The intrinsic image formation model has an inherent scale ambiguity:

```
I = A_d * S_d   is equivalent to   I = (c * A_d) * (S_d / c)
```

Any scalar `c` applied to albedo and inverted on shading produces an equally valid decomposition. This means comparing the network's output directly to raw dataset GT values will produce massive, unstable gradients — the network is not wrong, it is just operating at a different arbitrary scale.

The solution, following CD-IID and Ordinal Shading, is to dynamically match the GT scale to the network's current predictions during each forward pass.

### 3.2 Validity Mask (Static, in `__getitem__`)

Computed once per sample before augmentation:

```python
# render_entity_id: (H, W) int32, -1 = sky or invalid geometry
# NOTE: render_entity_id.hdf5 is NOT present in the current sanity set
# (ai_001_001, ai_001_002). The dataset falls back to albedo-only masking,
# which is sufficient for sanity checking. Download render_entity_id.hdf5
# when scaling to the full production set.

if render_entity_id available:
    sky_mask = (render_id != -1)          # excludes sky, invalid geometry
else:
    sky_mask = np.ones((H, W), dtype=bool)  # fallback: trust albedo mask only

albedo_mask = (albedo_raw.mean(axis=-1) > 0.02)   # exclude mirrors, glass
valid_mask  = sky_mask & albedo_mask               # (H, W) bool
```

Current sanity set (59 frames, 2 scenes, `ai_001_001` + `ai_001_002`):
- `render_entity_id.hdf5` is **absent** → sky_mask falls back to all-True
- `albedo_mask` alone gives ~86% valid pixels per frame — adequate for sanity checks
- `scene_cam_01` in `ai_001_002` has only 11/20 geometry frames → 9 frames skipped by `_build_file_list`
- Total usable frames after filtering: **59 color** → **50 with full geometry** → 45 train / 5 val

### 3.3 What `__getitem__` Returns

`__getitem__` returns raw arrays only — no target-space mapping. Scale matching happens in the training loop after the forward pass.

```python
return {
    'rgb':        rgb_tonemapped,    # (3, H, W)  tonemapped linear, encoder input
    'normals':    normals_aug,       # (3, H, W)  unit vectors
    'seg':        seg_aug,           # (1, H, W)  long, NYU-40 labels
    'albedo_raw': albedo_aug,        # (3, H, W)  raw linear HDR albedo
    'illum_raw':  illum_aug,         # (3, H, W)  raw linear HDR diffuse illumination
    'valid_mask': valid_aug,         # (1, H, W)  bool
    'M_diffuse':  True,
    'M_albedo':   True,
}
```

---

## Phase 3.5: Dynamic Scale Matching (Training Loop)

This step runs **after the forward pass**, once the network has produced its current predictions.

### Why Scale on Albedo, Not Shading

Scale matching is performed on albedo rather than shading because albedo is a physical reflectance value bounded to `[0, 1]` by definition. This bounded range makes the least-squares fit numerically stable. Shading is unbounded HDR — using it for scale fitting would produce an unstable solution.

### Step 1: Least-Squares Scale Fit

```python
def scale_match(A_raw: torch.Tensor, A_pred: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
    """
    Compute per-image scalar c such that c * A_raw best matches A_pred
    in a least-squares sense, over valid pixels only.

    A_raw:  (N, 3, H, W)  raw GT albedo from dataset
    A_pred: (N, 3, H, W)  network's current albedo prediction
    valid:  (N, 1, H, W)  bool mask

    Returns: (N, 1, 1, 1)  per-image scale factor c
    """
    # Flatten spatial dims, apply mask
    v = valid.expand_as(A_raw)                          # (N, 3, H, W)
    a = (A_raw  * v).reshape(A_raw.shape[0], -1)        # (N, 3*H*W)
    b = (A_pred * v).reshape(A_pred.shape[0], -1)       # (N, 3*H*W)

    # Closed-form least-squares: c = (a^T b) / (a^T a)
    c = (a * b).sum(dim=1) / ((a * a).sum(dim=1) + 1e-6)  # (N,)
    return c.view(-1, 1, 1, 1)                              # (N, 1, 1, 1)
```

### Step 2: Compute Scale-Matched GTs

```python
# After forward pass produces A_d (albedo prediction):
c = scale_match(batch['albedo_raw'], A_d.detach(), batch['valid_mask'])

# Scale-matched albedo and shading GTs
A_star = c * batch['albedo_raw']       # (N, 3, H, W)  scaled albedo GT
S_star = batch['rgb'] / (A_star + 1e-6) # (N, 3, H, W)  implied scaled shading
```

Note: `S_star` is derived by dividing the input image by the scaled albedo — not loaded from disk. This ensures the shading GT is physically consistent with the albedo GT under the current scale.

### Step 3: Format into Decoder Target Spaces

**Dec A target — Inverse Grayscale Shading:**
```python
# Convert S_star to grayscale luminance
S_g_star = (0.2126 * S_star[:,0:1]
           + 0.7152 * S_star[:,1:2]
           + 0.0722 * S_star[:,2:3])     # (N, 1, H, W)

# Map to inverse space: compresses HDR range to (0, 1]
D_g_star = 1.0 / (S_g_star + 1.0)       # (N, 1, H, W)
```

**Dec B target — Bounded Chroma xi:**
```python
eps = 1e-6
# Chroma = per-pixel RGB ratio of S_star relative to green channel
C_RG = S_star[:,0:1] / (S_star[:,1:2] + eps)   # R/G
C_BG = S_star[:,2:3] / (S_star[:,1:2] + eps)   # B/G

# Map unbounded ratio to bounded (0, 1] — following CD-IID
xi_star = torch.cat([
    1.0 / (C_RG + 1.0),
    1.0 / (C_BG + 1.0)
], dim=1)                                        # (N, 2, H, W)
```

**Dec C target — Diffuse Albedo:**
```python
# No spatial mapping needed — use scale-matched albedo directly
A_d_star = A_star                                # (N, 3, H, W)
```

**Dec D target — Inverse Colorful Shading:**
```python
# Map RGB shading to inverse space
pi_star = 1.0 / (S_star + 1.0)                  # (N, 3, H, W)
```

**Why inverse shading space:** A window pixel with `S_g = 450.0` maps to `D_g = 0.0022`. A dark corner with `S_g = 0.005` maps to `D_g = 0.995`. The transformation compresses the entire unbounded HDR range to `(0, 1]` while preserving ordinality, preventing bright specular pixels from dominating gradients.

### Step 4: MIDIntrinsic Pseudo-GT (Real-World Dataset)

MIDIntrinsic provides 25 different illumination conditions per scene but no ground truth shading. The pseudo-albedo GT is generated by running all 25 images through the network and taking the per-pixel median:

```python
def compute_midintrinsic_pseudo_gt(model, scene_images: list,
                                   valid_mask: torch.Tensor) -> torch.Tensor:
    """
    scene_images: list of 25 tensors, each (1, 3, H, W), same scene
                  under different illuminations
    Returns: (1, 3, H, W) robust pseudo-albedo GT via per-pixel median
    """
    albedo_predictions = []
    with torch.no_grad():
        for img in scene_images:
            preds  = model(img, normals=None, seg=None)
            A_pred = preds['a_d']                # (1, 3, H, W)
            # Scale-match each prediction to a common reference scale
            c = scale_match(A_pred, scene_images[0], valid_mask)
            albedo_predictions.append(c * A_pred)

    # Stack and take per-pixel median across all 25 illuminations
    stacked = torch.stack(albedo_predictions, dim=0)  # (25, 1, 3, H, W)
    pseudo_gt, _ = stacked.median(dim=0)              # (1, 3, H, W)
    return pseudo_gt

# For MIDIntrinsic batches:
# M_diffuse = False → only L_C is computed using pseudo_gt
# Dec A, B, D losses are zeroed, cascade inputs detached
```

The median is robust to outliers: if one of the 25 illuminations causes a specular highlight on a surface, 24 other predictions vote it down. The result is a clean, stable albedo estimate despite having no ground truth shading.

---

## Phase 4: CCR Computation (Online, During Training)

**Do not precompute CCR offline.** CCR is a single GPU operation (log + 2D conv) with negligible cost (~0.5ms per batch). Computing it online inside the model forward pass eliminates an entire preprocessing pipeline and avoids storing extra files on disk.

### Correct CCR Formulation

CCR computes cross-channel ratios between neighboring pixels in log space. The descriptor is illumination-invariant by construction: values are nonzero only where the physical reflectance (albedo) changes, not where lighting changes.

**Mathematical definition** for neighboring pixels p1, p2:
```
M_RG = (R_p1 * G_p2) / (R_p2 * G_p1)

In log space:
log(M_RG) = log(R_p1) - log(G_p1) + log(G_p2) - log(R_p2)
           = diff_filter(log_R) - diff_filter(log_G)
```

This is computed efficiently via spatial convolution rather than pixel-by-pixel loops.

### PyTorch Implementation

The implementation returns **6 channels**, combining two complementary
illumination-invariant cues validated by PIE-Net and SIGNet:

- **Channels 0–2 — Clamped log cross-colour ratios (PIE-Net):** change only at
  reflectance boundaries. **Clamped to `[-1, 1]`** to prevent exploding values at
  near-black pixels (`img ≈ 0 → log → −∞`). Without the clamp, CCREncoder receives
  NaN/Inf gradients at dark image regions.
- **Channels 3–5 — Normalized RGB chromaticity (SIGNet):** `c / (R+G+B)` encodes
  per-pixel hue independently of illumination intensity.

```python
import torch
import torch.nn.functional as F

def compute_ccr(img: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    6-channel illumination-invariant descriptor.

    Ch 0-2: clamped log cross-colour ratios (PIE-Net)
      log_M_RG = clamp(diff(log R) - diff(log G), -1, 1)
      log_M_RB = clamp(diff(log R) - diff(log B), -1, 1)
      log_M_GB = clamp(diff(log G) - diff(log B), -1, 1)

    Ch 3-5: normalized RGB chromaticity (SIGNet)
      norm_c = c / (R + G + B + eps)    in [0, 1]

    Args:
        img:  (N, 3, H, W)  linear RGB float32, values > 0
        eps:  small constant to prevent log(0) and division by zero
    Returns:
        (N, 6, H, W)
    """
    # Clamped log CCR — channels 0-2
    log_img = torch.log(img + eps)
    R, G, B = log_img[:, 0:1], log_img[:, 1:2], log_img[:, 2:3]

    kernel = torch.tensor(
        [[0,  1,  0],
         [1,  0, -1],
         [0, -1,  0]], dtype=torch.float32
    ).view(1, 1, 3, 3).to(img.device)

    def diff(ch):
        return F.conv2d(ch, kernel, padding=1)

    log_M_RG = torch.clamp(diff(R) - diff(G), -1.0, 1.0)
    log_M_RB = torch.clamp(diff(R) - diff(B), -1.0, 1.0)
    log_M_GB = torch.clamp(diff(G) - diff(B), -1.0, 1.0)

    # Normalized RGB chromaticity — channels 3-5
    intensity = img[:, 0:1] + img[:, 1:2] + img[:, 2:3] + eps
    norm_rgb  = img / intensity                              # (N, 3, H, W)

    return torch.cat([log_M_RG, log_M_RB, log_M_GB, norm_rgb], dim=1)  # (N, 6, H, W)
```

**Call site — inside model forward pass (CCREncoder expects 6ch input):**
```python
def forward(self, rgb, ...):
    ccr      = compute_ccr(rgb)     # ~0.5 ms, no disk I/O, (N, 6, H, W)
    F_ccr_i  = self.ccr_encoder(ccr)  # CCREncoder(in_channels=6, ...)
    ...
```

**Comparison to old / wrong approaches:**

| Metric | Old (3ch unclamped) | Current (6ch clamped + norm-RGB) |
|---|---|---|
| Illumination invariant | Yes | Yes |
| Near-black stability | ❌ log→−∞ at dark pixels | ✅ clamped to [-1,1] |
| Per-pixel hue info | ❌ No | ✅ ch 3-5 (norm-RGB) |
| Changes at lighting edges | No | No |
| Changes at material edges | Yes | Yes |
| Output channels | 3 | **6** |
| CCREncoder `in_channels` | 3 | **6** |
| Requires preprocessing | No | No (still online GPU op) |

---

## Phase 5: Dataloader Implementation

### 5.1 RAM Caching Strategy

Hypersim `.hdf5` files are large float32 arrays. Repeated disk reads across training epochs can bottleneck multi-worker dataloaders, so we use a bounded decode cache per dataset worker.

- First read: load from disk, decode, and store in RAM cache
- Subsequent reads of the same file path: return cached decoded array
- Cache policy: LRU eviction (`cache_max_items`)
- `cache_max_items = 0` disables caching
- Sample selection remains index-driven by the dataloader sampler (no replay-style random index substitution)

```python
import threading
from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F

class HypersimDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, split: str = 'train',
                 cache_max_items: int = 512, target_size: int = 384,
                 split_file: str = 'hypersim_split.json',
                 split_seed: int = 42,
                 split_ratio: float = 0.9,
                 strict_split: bool = True):
        self.root = root
        self.target_size = target_size
        self.cache_max_items = max(0, int(cache_max_items))
        self.split_file = split_file
        self.split_seed = split_seed
        self.split_ratio = split_ratio
        self.strict_split = strict_split
        self.cache = OrderedDict()
        self.lock = threading.Lock()
        self.samples = self._build_sample_list(split)

    def _load_or_cache(self, key: str, path: str) -> np.ndarray:
        if self.cache_max_items <= 0:
            return load_hdf5(path)

        with self.lock:
            cached = self.cache.get(key)
            if cached is not None:
                self.cache.move_to_end(key)
                return cached

        data = load_hdf5(path)
        with self.lock:
            self.cache[key] = data
            self.cache.move_to_end(key)
            while len(self.cache) > self.cache_max_items:
                self.cache.popitem(last=False)
        return data

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        rgb = self._load_or_cache(f"rgb_{idx}", s['color'])
        albedo = self._load_or_cache(f"alb_{idx}", s['albedo'])
        illum = self._load_or_cache(f"ill_{idx}", s['illum'])
        normals = self._load_or_cache(f"nrm_{idx}", s['normal'])
        seg = self._load_or_cache(f"seg_{idx}", s['seg'])
        rid = self._load_or_cache(f"rid_{idx}", s['render_id'])
        return self._process(rgb, albedo, illum, normals, seg, rid)
```

### 5.2 Full Processing Pipeline Inside `__getitem__`

```python
    def _process(self, rgb, albedo, illum, normals, seg, render_id):
        # --- Tonemap RGB input (linear, no gamma) ---
        rgb = tonemap_linear(rgb)

        # --- Validity mask (computed before augmentation) ---
        sky_mask    = (render_id != -1)
        albedo_mask = (albedo.mean(axis=-1) > 0.02)
        valid       = (sky_mask & albedo_mask).astype(np.float32)  # (H,W)

        # --- Stack all modalities for joint augmentation ---
        # Layout: rgb(3) | albedo(3) | illum(3) | normals(3) | valid(1) = 13ch
        # Seg handled separately (nearest resize to avoid interpolating class labels)
        combined = np.concatenate([
            rgb.astype(np.float32),
            albedo.astype(np.float32),
            illum.astype(np.float32),
            normals.astype(np.float32),
            valid[..., None],
        ], axis=-1)                                                  # (H,W,13)

        # --- Random square crop (stochastic size) ---
        H, W  = combined.shape[:2]
        max_size = min(H, W)
        min_size = max(int(0.6 * max_size), self.target_size)
        size  = np.random.randint(min_size, max_size + 1) if max_size > min_size else max_size
        top   = np.random.randint(0, max(1, H - size + 1))
        left  = np.random.randint(0, max(1, W - size + 1))
        combined = combined[top:top+size, left:left+size]
        seg_crop = seg[top:top+size, left:left+size]

        # --- Resize to target (384x384) ---
        t     = torch.from_numpy(combined).permute(2,0,1).unsqueeze(0).float()
        t_r   = F.interpolate(t, size=(self.target_size,)*2,
                              mode='bilinear', align_corners=False).squeeze(0)

        seg_t = torch.from_numpy(seg_crop.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        seg_r = F.interpolate(seg_t, size=(self.target_size,)*2,
                              mode='nearest').squeeze(0).long()

        # --- Random flips (identical across all modalities) ---
        if np.random.rand() > 0.5:
            t_r   = torch.flip(t_r,   dims=[2])
            seg_r = torch.flip(seg_r, dims=[2])
        if np.random.rand() > 0.5:
            t_r   = torch.flip(t_r,   dims=[1])
            seg_r = torch.flip(seg_r, dims=[1])

        # --- Return raw arrays — scale matching happens in training loop ---
        return {
            'rgb':        t_r[0:3],           # (3, H, W)  tonemapped linear input
            'albedo_raw': t_r[3:6],           # (3, H, W)  raw albedo (unscaled)
            'illum_raw':  t_r[6:9],           # (3, H, W)  raw linear HDR diffuse illumination (unscaled)
            'normals':    t_r[9:12],          # (3, H, W)  unit vectors
            'valid_mask': t_r[12:13].bool(),  # (1, H, W)  applied in all losses
            'seg':        seg_r,              # (1, H, W)  long, NYU-40 labels
            'M_diffuse':  True,
            'M_albedo':   True,
        }
```

### 5.3 Mixed Dataset Dataloader

Each batch is homogeneous (single dataset) to ensure consistent `M_diffuse`/`M_albedo` flags:

```python
import numpy as np
import torch

class MixedDataloader:
    """
    Weighted homogeneous-batch sampler.
    Uses one dataset per step; supports dynamic phase weight updates.
    """
    def __init__(self, datasets: dict, weights: dict, batch_size: int, num_workers: int = 4):
        self.loaders = {
            name: torch.utils.data.DataLoader(
                ds, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=True, drop_last=True
            )
            for name, ds in datasets.items()
        }
        self.iterators = {n: iter(l) for n, l in self.loaders.items()}
        self.set_weights(weights)

    def set_weights(self, weights: dict):
        names = [n for n, w in weights.items() if n in self.loaders and float(w) > 0.0]
        probs = np.array([float(weights[n]) for n in names], dtype=np.float64)
        self.names = names
        self.probs = probs / probs.sum()

    def next_batch(self):
        name = str(np.random.choice(self.names, p=self.probs))
        try:
            batch = next(self.iterators[name])
        except StopIteration:
            self.iterators[name] = iter(self.loaders[name])
            batch = next(self.iterators[name])
        return batch, name
```

---

## Phase 6: Loss Routing by Dataset

```python
MASK_CONFIG = {
    'hypersim':     {'M_diffuse': True,  'M_albedo': True},
    'interiornet':  {'M_diffuse': False, 'M_albedo': True},
    'midintrinsic': {'M_diffuse': False, 'M_albedo': True},
}

for batch, dataset_name in mixed_loader:
    masks = MASK_CONFIG[dataset_name]
    v     = batch['valid_mask']   # (N, 1, H, W)

    m_diffuse_t = torch.tensor([float(masks['M_diffuse'])] * len(batch['rgb']))
    preds  = model(batch['rgb'], m_diffuse=m_diffuse_t,
                   normals=batch.get('normals'), seg=batch.get('seg'))
    S_g = preds['s_g']
    xi  = preds['xi']
    A_d = preds['a_d']
    S_d = preds['s_d']

    # --- Scale matching (after forward pass, before loss) ---
    # Fit scalar c per image: c = argmin_x ||x * A_raw - A_pred||^2
    # Done on albedo because it is bounded [0,1] → numerically stable fit
    c      = scale_match(batch['albedo_raw'], A_d.detach(), v)   # (N,1,1,1)

    A_star = c * batch['albedo_raw']                              # (N,3,H,W) scaled albedo GT
    S_star = batch['rgb'] / (A_star + 1e-6)                      # (N,3,H,W) implied scaled shading GT

    # --- Format into decoder target spaces ---
    # Dec A: inverse grayscale shading
    S_g_star = (0.2126*S_star[:,0:1] + 0.7152*S_star[:,1:2] + 0.0722*S_star[:,2:3])
    D_g_star = 1.0 / (S_g_star + 1.0)                            # (N,1,H,W) in (0,1]

    # Dec B: bounded 2-channel chroma xi
    eps      = 1e-6
    C_RG     = S_star[:,0:1] / (S_star[:,1:2] + eps)
    C_BG     = S_star[:,2:3] / (S_star[:,1:2] + eps)
    xi_star  = torch.cat([1./(C_RG+1.), 1./(C_BG+1.)], dim=1)   # (N,2,H,W) in (0,1]

    # Dec C: scale-matched albedo directly
    A_d_star = A_star                                             # (N,3,H,W)

    # Dec D: inverse colorful shading
    pi_star  = 1.0 / (S_star + 1.0)                              # (N,3,H,W) in (0,1]

    # --- Reconstruct S_c for cascade (from predictions, not GT) ---
    C_R = (1.0 / (xi[:,0:1] + eps)) - 1.0
    C_B = (1.0 / (xi[:,1:2] + eps)) - 1.0
    C   = torch.cat([C_R, torch.ones_like(C_R), C_B], dim=1)
    S_c = S_g * C

    # --- Detach cascade when shading GT unavailable ---
    if not masks['M_diffuse']:
        S_d = S_d.detach()
        S_c = S_c.detach()

    # --- Compute masked losses ---
    L_A = masked_loss(dec_a_loss(S_g, D_g_star), v)
    L_B = masked_loss(dec_b_loss(xi,  xi_star),  v)
    L_C = masked_loss(dec_c_loss(A_d, A_d_star, batch['seg']), v) * masks['M_albedo']
    L_D = masked_loss(dec_d_loss(S_d, pi_star),  v) * masks['M_diffuse']

    loss = L_A + L_B + L_C + L_D
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

---

## Summary: What Lives Where

| Component | Where computed | Format |
|---|---|---|
| RGB image | Loaded from `.hdf5`, tonemapped in `__getitem__` | float32, linear [0,1] |
| Albedo raw | Loaded from `.hdf5`, returned as-is | float32, linear HDR |
| Diffuse illumination raw | Loaded from `.hdf5`, returned as-is | float32, linear HDR |
| Normals | Loaded from `.hdf5` | float32, unit vectors |
| Segmentation | Loaded from `.hdf5` | long, NYU-40 labels |
| Validity mask | Derived in `__getitem__` from render_id + albedo | bool (H,W) |
| Scale factor c | Derived in training loop after forward pass | float32 scalar per image |
| A_star (scaled albedo GT) | Derived in training loop: `c * albedo_raw` | float32 (N,3,H,W) |
| S_star (implied shading GT) | Derived in training loop: `rgb / A_star` | float32 (N,3,H,W) |
| D_g_star (Dec A target) | Derived in training loop: `1/(S_g_star+1)` | float32 (N,1,H,W), in (0,1] |
| xi_star (Dec B target) | Derived in training loop from S_star ratios | float32 (N,2,H,W), in (0,1] |
| A_d_star (Dec C target) | A_star directly | float32 (N,3,H,W) |
| pi_star (Dec D target) | Derived in training loop: `1/(S_star+1)` | float32 (N,3,H,W), in (0,1] |
| CCR | Computed online in `model.forward()` via `compute_ccr(rgb)` | float32 **(N,6,H,W)**: ch 0-2 clamped log-CCR [-1,1], ch 3-5 norm-RGB [0,1] |
| S_c = S_g × C | Computed in training loop from predictions | float32 (N,3,H,W) |
| MIDIntrinsic pseudo-GT | Median of 25 per-scene predictions | float32 (N,3,H,W) |

**Key principle:** `__getitem__` returns only raw arrays. All target-space transformations (inverse shading, bounded chroma, scale matching) happen in the training loop **after** the forward pass, because the GT scale must be matched to the network's current predictions.
