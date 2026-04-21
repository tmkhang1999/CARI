# V11 Update To-Do List

Two versions are defined:
- **V11_single** — Hypersim only. All architecture and loss fixes. No MIDIntrinsic changes.
- **V11_mix** — Hypersim + MIDIntrinsic. Everything in V11_single, plus all dataset and training-loop changes below.

Priority levels: 🔴 Critical (breaks training or produces wrong gradients) · 🟡 Important (degrades quality) · 🟢 Minor (clean-up / robustness)

---

## 1. `src/models/decoders/progressive_decoder.py` — both versions

### 1.1 🔴 Replace ConvTranspose2d with bilinear upsample + Conv3x3 in DecoderBlock

`ConvTranspose2d(kernel=4, stride=2)` produces checkerboard artifacts at boundaries —
the exact edges IID needs to get right. Replace `DecoderBlock` in `decoder.py`:

```python
# Before
nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
nn.BatchNorm2d(out_channels),
nn.ReLU(inplace=True)

# After
nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
nn.GroupNorm(1, out_channels),
nn.GELU(),
```

This affects every `ProgressiveDecoder` (all four decoders in V10).

### 1.2 🟡 Widen final head bottleneck: 96 → 64 → out_channels

```python
# Before
nn.Conv2d(96, 32, kernel_size=3, padding=1),
nn.ReLU(inplace=True),
nn.Conv2d(32, out_channels, kernel_size=1),

# After
nn.Conv2d(96, 64, kernel_size=3, padding=1, bias=False),
nn.GroupNorm(1, 64),
nn.GELU(),
nn.Conv2d(64, out_channels, kernel_size=1),
```

---

## 2. `src/models/modules/spade.py` — both versions

### 2.1 🔴 Remove InstanceNorm — it destroys CCR modulation and albedo scale

InstanceNorm zeroes the mean and normalises variance of every feature map, immediately
undoing the magnitude adjustments SFM just applied. Make SPADE purely additive:

```python
class SPADE(nn.Module):
    def __init__(self, num_channels, num_classes=41, hidden_channels=64):
        super().__init__()
        # Remove: self.norm = nn.InstanceNorm2d(num_channels, affine=False)
        self.embed = nn.Sequential(
            nn.Conv2d(num_classes, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.beta = nn.Conv2d(hidden_channels, num_channels, kernel_size=3, padding=1)
        # Remove gamma head — additive residual only
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)
        self.num_classes = num_classes

    def forward(self, x, seg):
        seg_oh = self._prepare_seg(seg, x.shape[-2:])
        if seg_oh is None:
            return x
        return x + self.beta(self.embed(seg_oh))   # residual, no norm
```

### 2.2 🟡 Add SPADE at s3 (H/16) in Dec C inside `stage1_v11.py`

Semantic context is most useful at the coarsest spatial level. Add `spade_c3`:

```python
# In __init__
self.spade_c3 = SPADE(num_channels=768, num_classes=num_seg_classes)

# In forward stage_ops for decoder_c
lambda x: self.sfm_c_g3(self.spade_c3(self.sfm_c_ccr3(x, ccr_prior3), seg), g_c[3]),
```

---

## 3. `src/models/encoders/ccr_encoder.py` — both versions

### 3.1 🟡 Replace BatchNorm with GroupNorm throughout

BN running stats will mix Hypersim and MIDIntrinsic activation distributions
(more critical in V11_mix, but good practice in V11_single too):

```python
def _conv_gn_gelu(in_c, out_c, stride):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(1, out_c),
        nn.GELU(),
    )
```

### 3.2 🟡 Use ccr_feats[0] (H/2, 64ch) in dec0 of each decoder

Currently ccr_feats[0] is computed but discarded. Add SFM at the finest decoder stage
in Dec A and Dec C, which currently receive no CCR guidance:

```python
# In __init__ — new SFMs for dec0 level
self.sfm_a_ccr0 = SpatialFeatureModulation(96, 64)   # dec0 output ch=96, ccr ch=64
self.sfm_c_ccr0 = SpatialFeatureModulation(96, 64)

# In forward — dec0 is after the ProgressiveDecoder's final upscale, so apply in head:
# Pass ccr_feats[0] through forward() as a new argument and apply after dec0 block.
```

> Note: this requires exposing a post-dec0 hook in `ProgressiveDecoder`. Simplest approach
> is adding a 4th entry to `stage_ops` that is called after `dec0` before `head`.

---

## 4. `src/models/encoders/normal_encoder.py` — both versions

### 4.1 🟢 Add standard Conv3x3 as first layer before depthwise blocks

DWConv cannot mix x/y/z channels. A single full conv at the input fixes this cheaply:

```python
class NormalEncoder(nn.Module):
    def __init__(self, in_channels=3, channels=(64, 128, 256, 512)):
        super().__init__()
        c1, c2, c3, c4 = channels
        # New: one standard conv to mix x/y/z before going depthwise
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, c1),
            nn.GELU(),
        )
        self.block1 = _DWBlock(c1, c1)   # stride-2 from here
        self.block2 = _DWBlock(c1, c2)
        self.block3 = _DWBlock(c2, c3)
        self.block4 = _DWBlock(c3, c4)

    def forward(self, normals):
        x  = self.input_proj(normals)    # H, c1
        n1 = self.block1(x)              # H/2, c1
        n2 = self.block2(n1)             # H/4, c2
        n3 = self.block3(n2)             # H/8, c3
        n4 = self.block4(n3)             # H/16, c4
        return [n1, n2, n3, n4]
```

---

## 5. `src/models/modules/illuminant_descriptor.py` — both versions

### 5.1 🟢 Add a second hidden layer to the MLP

The current 4 → 128 → 2048 has a 16× expansion at the output. Add a middle layer:

```python
self.mlp = nn.Sequential(
    nn.Linear(in_dim, 64),
    nn.ReLU(inplace=True),
    nn.Linear(64, 256),
    nn.ReLU(inplace=True),
    nn.Linear(256, out_dim),
)
```

---

## 6. `src/models/modules/spatial_feature_modulation.py` — both versions

### 6.1 🟢 Make hidden_channels adaptive to prior size

Hardcoded 128 is too narrow for the s3 stage where prior is 512ch:

```python
def __init__(self, x_channels, prior_channels, hidden_channels=None, gamma_scale=1.0):
    super().__init__()
    if hidden_channels is None:
        hidden_channels = min(prior_channels, x_channels // 2)
        hidden_channels = max(hidden_channels, 64)   # floor
    ...
```

---

## 7. `src/losses/flexible_loss.py` — both versions

### 7.1 🔴 Apply w_c and w_d in forward()

```python
# Before — w_c and w_d silently ignored
loss_total = la + lb + lc + ld

# After
lc = self.w_c * lc
ld = self.w_d * ld
loss_total = la + lb + lc + ld
```

---

## 8. `src/data/midintrinsic_dataset.py` — V11_mix only

### 8.1 🔴 Fix d_g_raw scale mismatch

`d_g_raw` is computed as `rgb_mix_neutral / albedo_raw_HDR`. Neutral is tonemapped
[0,1] but albedo is raw HDR → division is meaningless. Fix: tonemap albedo with the
same per-illumination scale before division, OR drop d_g_raw and let compute_targets
derive D_g_star from the neutral branch at training time using a shared tonemap scale.

Recommended fix — compute shared scale and pass it through:

```python
# In _sample_mix: return also the neutral mix's tonemap scale
# In __getitem__: tonemap albedo with that scale before computing d_g_raw
tonemap_scale = float(np.percentile(rgb_mix_neutral, 99.0)) + 1e-6
albedo_tm = np.clip(albedo / tonemap_scale, 0.0, None)   # keep HDR but same scale
s_neutral = rgb_mix_neutral / (albedo_tm + eps)
s_g = 0.2126*s_neutral[...,0:1] + 0.7152*s_neutral[...,1:2] + 0.0722*s_neutral[...,2:3]
d_g_target = 1.0 / (s_g + 1.0)
```

### 8.2 🔴 Route d_g_raw and xi_raw into targets in training loop

These keys are returned in the batch dict but `compute_targets` never reads them. Add
routing logic in the training script after `compute_targets` is called:

```python
if dataset_name == 'midintrinsic':
    if 'd_g_raw' in batch:
        targets['D_g_star'] = batch['d_g_raw'].to(device)
    if 'xi_raw' in batch:
        targets['xi_star'] = batch['xi_raw'].to(device)
```

### 8.3 🟡 Re-normalise normals after bilinear resize

After `F.interpolate(..., mode='bilinear')`, normal vectors are no longer unit length:

```python
# After t_img interpolation, before concat with t_mask:
norm_len = t_img[6:9].norm(dim=0, keepdim=True).clamp_min(1e-6)
t_img = torch.cat([t_img[:6], t_img[6:9] / norm_len, t_img[9:]], dim=0)
```

### 8.4 🟡 Guard cv2.resize calls with None check

If OpenCV is unavailable and geometry files exist, this raises AttributeError:

```python
if cv2 is not None:
    seg_np = cv2.resize(...)
else:
    import torch.nn.functional as F
    seg_t_tmp = torch.from_numpy(seg_np).unsqueeze(0).unsqueeze(0).float()
    seg_np = F.interpolate(seg_t_tmp, size=(albedo.shape[0], albedo.shape[1]),
                           mode='nearest').squeeze().numpy().astype(np.int32)
```

---

## 9. `src/data/mixed_dataloader.py` — V11_mix only

### 9.1 🟡 Confirm lazy iterator initialisation is deployed

Verify the file on disk uses `self._iterators = {}` (lazy, from the new version in
document 4), NOT `self._iterators = {name: iter(loader) for ...}` (eager, old version).
Eager init spawns MIDIntrinsic workers during phase 1 when weight is 0.

---

## 10. New Loss Design and Functionality for the 4 Decoders

| Network | Target Variable | Primary Loss | Structural Loss | Domain |
| :--- | :--- | :--- | :--- | :--- |
| **1. Ordinal Shading** | Grayscale $D_g$ | **SSI-MSE** | **MSG** | Inverse Shading |
| **2. Shading Chroma** | Chroma $\xi$ (2-ch) | **MSE** | **MSG** | Bounded Ratios |
| **3. Diffuse Albedo** | Albedo $A_d$ | **MSE** | **MSG** | Linear RGB |
| **4. Diffuse Shading** | Diffuse $\pi$ (3-ch) | **MSE** | **MSG** | Inverse Shading |

**NOTE**: Still keep DSSIM, Perceptual, TVloss, and Semvar loss for Albedo. But make sure, when we set lambda weight = 0, don't calculate it and visualize on tensorboard

#### Detailed Loss Function Implementations
*   **SSI-MSE (Scale-and-Shift Invariant MSE):** Used for Network 1 to ensure correct linear ordering of shading values without penalizing global scale offsets. It estimates affine parameters ($a, b$) using least squares to align predictions to ground truth.
    *   **Code Location:** `intrinsic_decomposition/common/midas_loss.py` → `ssi_mse_loss` and `compute_scale_and_shift`.
*   **MSE (Mean-Squared Error):** Standard pixel-wise loss used for chroma, albedo, and diffuse shading stages to exploit the statistical sparsity of materials.
*   **MSG (Multi-Scale Gradient):** Computes $L_1$ loss on spatial gradients across a 4-scale pyramid. This is essential for generating spatially consistent estimations and sharp material/shadow boundaries.
    *   **Code Location:** `chrislib/loss.py` → `MSGLoss`.

---

### 2. Four-Stage Ground Truth (GT) Preparation

For **Hypersim**, all components are provided as uncompressed High Dynamic Range (HDR) images in the EXR format.

1.  **Network 1 (Ordinal GT - $D_g$):** Shading $S$ is derived by dividing the linear image by the ground-truth albedo ($S = I/R$). This is then mapped to the **Inverse Shading Domain** to best utilize the range and handle specular outliers:
    *   $D_g = 1 / (S + 1)$.
2.  **Network 2 (Chroma GT - $\xi$):** Colorful shading chromaticity is defined as channel ratios relative to the green channel: $C^* = (S_{c,R}/S_{c,G}, S_{c,B}/S_{c,G})$. These are then mapped to the range:
    *   $\xi = 1 / (C^* + 1)$.
3.  **Network 3 (Diffuse Albedo GT - $A_d$):** Hypersim provides a "diffuse reflectance" map. This HDR map is tonemapped to LDR without performing gamma compression to maintain linearity for the model.
4.  **Network 4 (Diffuse Shading GT - $\pi$):** Hypersim provides a "diffuse illumination" component. It is also mapped to the 3-channel inverse domain:
    *   $\pi = 1 / (S_d + 1)$.

---

### 3. Data Preprocessing and Augmentations (Hypersim Focus)

The pipeline relies on several critical utility functions to bridge the gap between synthetic data and "in-the-wild" performance.

#### Core Code Functions
*   **`compute_scale_and_shift`:**
    *   **Function:** Estimates the optimal scalar and offset using least squares to handle scale ambiguity during the SSI loss computation.
    *   **Implementation:** Found in `chrislib/loss.py`.
*   **`random_color_shift`:**
    *   **Function:** Specifically used for **Chroma Network** training to simulate inaccurate white balance. It generates a random RGB vector, normalizes it, and multiplies it by the input image.
    *   **Implementation:** Found in `chrislib/data_util.py`.
*   **`base_resize`:**
    *   **Function:** Resizes input images to the network's training resolution (default **384**) or receptive field size to ensure global lighting consistency during the "Base Pass" of inference.
    *   **Implementation:** Found in `intrinsic/ordinal_util.py`.

#### Specific Hypersim Preprocessing Steps
*   **Linearity Maintenance:** Images are tonemapped to LDR using a simple scheme (typically dividing by a percentile scale) but **no gamma correction** is applied, ensuring training occurs in linear RGB space.
*   **Masking:** Ground truth is masked for pixels with albedo values below **0.004**. These pixels represent mirror surfaces, glass, or skyboxes where information is typically unreliable or clipped.
*   **Albedo Scale Matching:** During training, the arbitrary scale of ground-truth albedo ($A^{**}$) is matched to the network's low-resolution input albedo ($\tilde{A}_!$) using **least-squares scale matching** to anchor the decomposition.

The **`get_brightness`** function is a core utility in the authors' `chrislib` library used to convert colorful RGB signals into a single-channel luminance baseline. As discussed, this is the critical first step for deriving the **ordinal shading ground truth** ($\pi^*$) from the ratio of the input image and albedo ($I/R$).

### **The Luminance Implementation**
The function utilizes the standard **CCIR601 YIQ method** to compute brightness, ensuring that the grayscale shading accurately reflects human perception of light intensity.

```python
def get_brightness(rgb, mode='numpy', keep_dim=True):
    """
    Uses the CCIR601 YIQ method to compute brightness of an RGB image.
    Formula: Y = 0.299*R + 0.587*G + 0.114*B
    """
    if mode == 'numpy':
        # Weights for R, G, B channels respectively
        weights = np.array([0.299, 0.587, 0.114])
        brightness = np.dot(rgb[..., :3], weights)
        if keep_dim:
            return brightness[..., np.newaxis]
        return brightness
        
    elif mode == 'torch':
        # PyTorch implementation for GPU-accelerated preprocessing
        weights = torch.tensor([0.299, 0.587, 0.114], device=rgb.device).view(1, 3, 1, 1)
        brightness = (rgb[:, :3, ...] * weights).sum(dim=1, keepdim=keep_dim)
        return brightness
```
*(Implementation referenced from `chrislib/general.py` and `chrislib/data_util.py`)*

### **How it is used in the GT Pipeline**
During the preparation of the **Hypersim** or **MIDIntrinsics** data, the `get_brightness` function is applied within the training loop or data loader as follows:

1.  **Compute Linear Shading:** The colorful shading is derived via $S_{color} = I / (A + \epsilon)$.
2.  **Calculate Luminance:** `S_gray = get_brightness(S_color)` is called to flatten the 3-channel shading into a grayscale map.
3.  **Inverse Mapping:** To create the final stable target for **Network 1**, the luminance is mapped to the inverse domain: 
    $$\pi^* = \frac{1}{S_{gray} + 1.0}$$
    This resulting value is what the **ordinal network** (and your **Decoder A**) is supervised to predict using the SSI-MSE loss.

### **Code Location**
In your pipeline and the authors' libraries, this function can be found at:
*   **`chrislib/general.py`** (line 464)
*   **`intrinsic_decomposition/common/general.py`** (line 349)

Would you like to see how the authors combine this grayscale output with the **chroma ratios** ($\xi$) in the second network to reconstruct the full colorful shading layer?