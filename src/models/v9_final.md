### Part 1: V9 - The Updated V8 Pipeline (Targeting Cast Shadows)

### The Fatal Flaw in Current V8: `ResidualAttention`

In `src/models/modules/residual_attention.py`, your attention mechanism is defined as:
```python
gate = torch.sigmoid(self.proj(prior))
return x * (1.0 + gate)
```
Let's analyze the bounds of this operation. `torch.sigmoid` outputs values in the range $[0, 1]$. Therefore, $(1.0 + \text{gate})$ strictly operates in the range $[1.0, 2.0]$. 

**Why this ruins cast shadow removal:**
If the ConvNeXt feature map $x$ contains a strong gradient from a cast shadow, and Decoder C (Albedo) needs to **remove** it using the CCR prior, it mathematically *cannot*. The minimum multiplier it can apply to the shadow edge is $1.0$. It can only amplify the shadow (up to $2.0\times$); it can never suppress or erase it down to $0$. 

You recognized this limitation in your `v8_ideas.md` when you introduced `ConcatProjection`, but you only applied it to *Guidance C*, while leaving `ResidualAttention` in place for the CCR injection (`self.attn_c_ccr3`). Since CCR is your primary tool for distinguishing material edges from shadow edges, gating it with a mechanism that cannot suppress features renders the CCR prior useless.

---

### The True V8.1 Pipeline (Fixing the Architecture)

To actually fix this, we must eradicate `ResidualAttention` and `ConcatProjection` entirely. We replace them with a unified **Spatial Feature Modulation (SFM)** module. SFM predicts affine parameters ($\gamma, \beta$) from the prior, allowing the network to completely zero out features ($\gamma \to -1$) or shift their baseline ($\beta$).

Here is the exact, corrected pipeline you should implement:

#### 1. The Unified Modulator: `SpatialFeatureModulation`
Create this new module to replace `ResidualAttention` and `ConcatProjection`.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialFeatureModulation(nn.Module):
    """
    Affine feature modulation allowing both amplification and strict suppression.
    Replaces ResidualAttention and ConcatProjection.
    """
    def __init__(self, x_channels, prior_channels):
        super().__init__()
        # Predicts gamma (scale) and beta (shift)
        self.proj = nn.Sequential(
            nn.Conv2d(prior_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, x_channels * 2, kernel_size=3, padding=1)
        )
        
        # Initialize to identity: gamma = 0, beta = 0 -> x' = x * (1+0) + 0
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x, prior):
        if prior is None:
            return x
        if prior.shape[-2:] != x.shape[-2:]:
            prior = F.interpolate(prior, size=x.shape[-2:], mode="bilinear", align_corners=False)
        
        affine_params = self.proj(prior)
        gamma, beta = affine_params.chunk(2, dim=1)
        
        # Network can learn to suppress features completely by driving gamma towards -1
        return x * (1.0 + gamma) + beta
```

#### 2. Decoder A (Gray Shading $D_g$)
We use SFM to inject Normals and CCR. 
* **The Goal:** $D_g$ must *absorb* the cast shadows so they don't leak into the albedo.
* **The Mechanism:** SFM uses CCR to identify material edges and drives $\gamma \to -1$ to suppress them from the shading map. Image edges that *lack* a CCR response (shadows) are preserved or amplified ($\gamma > 0$).
* **Code Update:** Replace `self.attn_a_ccr3` and `self.attn_a_n3` with `SpatialFeatureModulation`.

#### 3. Decoder B (Chroma $\xi$)
* **The Goal:** Prevent chroma from shifting in shadowed regions (e.g., shadows turning blue/orange).
* **The Mechanism:** Modulate with `GuidanceEncoder_B` (which now takes both $S_g$ and the safely derived implied albedo $A_g$).
* **Code Update:** Replace `self.attn_b3` with `SpatialFeatureModulation`. Inject the FiLM parameters ($\gamma, \beta$ from the `IlluminantDescriptor`) into the inner stages of this decoder, not just the global bottleneck $Z$.

#### 4. Decoder C (Diffuse Albedo $A_d$)
* **The Goal:** Strictly output material colors, erasing all shadows and shading.
* **The Mechanism:** 1.  Apply SFM with CCR (amplifies material edges, suppresses shadow edges).
    2.  Apply SPADE with Seg map (enforces region homogeneity).
    3.  Apply SFM with `GuidanceEncoder_C` (takes $S_c$ and $A_c$ to perform a final targeted subtraction of any residual shadow frequencies).
* **Code Update:** Chain the modulators logically:
    ```python
    # Inside Decoder C's stage_ops
    lambda x: self.sfm_c_guidance(self.spade_c(self.sfm_c_ccr(x, ccr_feats), seg), g_c)
    ```

#### 5. Decoder D (Diffuse Shading $\pi$)
* **The Goal:** Separate specular highlights from diffuse shading.
* **The Mechanism:** SFM with Normals (for geometry boundaries) and `GuidanceEncoder_D` (taking $S_c$ and $A_d$).

---

### Summary of the Strategy

By switching to `SpatialFeatureModulation`, you align your mathematical implementation with your theoretical claims. 

1.  **The Prior is Respected:** When CCR indicates "this is a shadow, not a material", the SFM layer can actually multiply those feature activations by $0$, eradicating the shadow from the Albedo pathway. Your previous V8 was multiplying them by $1.0$, leaving the shadow completely intact.
2.  **Stability via Initialization:** Because the final convolution of the SFM is zero-initialized, $\gamma=0$ and $\beta=0$ at step 0. The forward pass is $x \times (1.0 + 0) + 0 = x$. The network starts by behaving exactly like a baseline UNet and slowly learns to suppress artifacts as it minimizes the multi-scale gradient ($L_{msg}$) and DSSIM losses.
3.  **DDP & Memory Fixes:** Apply the exact code snippets from my previous response for `FlexibleLoss` (the `zero_loss = (a_d_pred * 0.0).sum()` DDP fix, the batched CCR computation, and the $FP32$ casting in DSSIM) alongside this architectural rewrite.

This is the true "V8.1" that reviewers will buy, because the math finally supports the story you are telling in the text.


#### 2. Numerical Instability in Shading/Albedo Derivation
Your mathematical formulation uses $S_g = 1 / (D_g + 1e-6) - 1$. If the network outputs $D_g$ near zero, $S_g$ spikes to millions, causing exploding gradients in the implied albedo losses.

**Fix (in your model's `forward` passes):**
Calculate a `safe` version of the shading for downstream feature extraction, and explicitly clamp the albedo derivation so gradients don't explode.
```python
# Derive linear shading and clamp for downstream Encoders
s_g = 1.0 / (d_g + 1e-6) - 1.0
s_g_safe = s_g.clamp(0.0, 20.0) # Use this for GuidanceEncoder inputs

# Derive implied albedo safely
# Use s_g_safe.detach() if Dec B becomes unstable, otherwise use s_g.clamp
a_g = (rgb / (s_g.clamp(1e-3, 20.0) + 1e-6)).clamp(0.0, 1.0)
a_g = torch.nan_to_num(a_g, nan=0.0, posinf=1.0, neginf=0.0)
```

#### 3. Inefficient CCR Computation
In `v5.py`, you compute spatial gradients using three separate `F.conv2d` calls. This is slow and memory-inefficient. Use a grouped convolution to compute it in a single pass.

**Fix in `compute_ccr`:**
```python
def compute_ccr(img, eps=1e-7):
    log_img = torch.log(img + eps)
    
    # Base 3x3 Sobel-like kernel
    base_kernel = torch.tensor([[0, 1, 0], [1, 0, -1], [0, -1, 0]], dtype=img.dtype, device=img.device)
    
    # Reshape for grouped convolution: (out_channels, in_channels/groups, kH, kW)
    kernel = base_kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    
    # Compute diffs for R, G, B in one pass
    diffs = F.conv2d(log_img, kernel, padding=1, groups=3)
    diff_r, diff_g, diff_b = diffs[:, 0:1], diffs[:, 1:2], diffs[:, 2:3]

    m_rg = torch.clamp(diff_r - diff_g, -1.0, 1.0)
    m_rb = torch.clamp(diff_r - diff_b, -1.0, 1.0)
    m_gb = torch.clamp(diff_g - diff_b, -1.0, 1.0)

    intensity = img.sum(dim=1, keepdim=True) + eps
    norm_rgb = img / intensity

    return torch.cat([m_rg, m_rb, m_gb, norm_rgb], dim=1)
```

#### 4. DSSIM Window Caching Memory Leak
Caching tensors in a standard Python dictionary `self._dssim_window_cache` inside an `nn.Module` prevents PyTorch from managing memory correctly when moving models across devices.

**Fix in `src/losses/flexible_loss.py`:**
Create a 1D base window and register it as a buffer. Expand it dynamically based on the channel size during the forward pass.
```python
class FlexibleLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        # ... other init code ...
        
        # Create a 1D Gaussian window ONCE and register as a buffer
        window_size = 11
        sigma = 1.5
        coords = torch.arange(window_size, dtype=torch.float32)
        gauss = torch.exp(-((coords - window_size // 2) ** 2) / (2.0 * sigma ** 2))
        gauss = gauss / gauss.sum()
        
        window_2d = (gauss.unsqueeze(1) @ gauss.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("dssim_base_window", window_2d, persistent=False)

    def _get_dssim_window(self, channels):
        # Dynamically expand the registered buffer to match channels
        return self.dssim_base_window.expand(channels, 1, -1, -1).contiguous()
```

#### 5. Mixed Precision Hazards in DSSIM
If you are using PyTorch AMP (Automatic Mixed Precision) for faster training, calculating variance via $E[X^2] - E[X]^2$ in `FP16` can result in catastrophic cancellation, yielding negative variances and `NaN` losses.

**Fix in `_compute_dssim`:**
Explicitly cast the predictions and targets to `float32` before the convolution operations.
```python
def _compute_dssim(self, pred, target, window_size=11):
    # Force float32 to prevent FP16 catastrophic cancellation in variance calculation
    pred = pred.to(torch.float32)
    target = target.to(torch.float32)
    
    C = pred.shape[1]
    window = self._get_dssim_window(C)
    # ... rest of the DSSIM math ...
```

#### 6. Nomenclature and Theoretical Mismatch (Reviewer Defense)
In your V8 design, you replaced a sigmoid-gated `ResidualAttention` with a `ConcatProjection`. Calling this "ConcatProjection" or "Attention" will result in rejection. Reviewers will ask: *"Why are you concatenating features and calling it attention?"*

**Fix:** Rename `ConcatProjection` to `SpatialFeatureModulation` (SFM). Mathematically, you are using the guidance features to modulate the target features, conceptually similar to SPADE but via direct fusion. 

```python
class SpatialFeatureModulation(nn.Module):
    """
    Replaces ConcatProjection. Fuses guidance signals (e.g., from GuidanceEncoder_C) 
    into decoder features, allowing for additive and subtractive shadow suppression.
    """
    def __init__(self, x_channels, guide_channels):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(x_channels + guide_channels, x_channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, x_channels),
            nn.GELU(),
        )
        # Identity initialization: ensures early training stability
        nn.init.zeros_(self.fusion[0].weight)
        with torch.no_grad():
            for i in range(x_channels):
                self.fusion[0].weight[i, i, 0, 0] = 1.0

    def forward(self, x, guide):
        if guide.shape[-2:] != x.shape[-2:]:
            guide = F.interpolate(guide, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.fusion(torch.cat([x, guide], dim=1))
```

#### 7. CCR Prior Justification (Writing the Paper)
When you write your ablation study, you *must* justify why CCR is fed into Decoder A (Shading). 
* **Your argument should be:** Prior works (like PIE-Net) use CCR exclusively to reconstruct albedo because CCR represents invariant material edges. We propose a novel inversion of this logic: by feeding CCR into the *shading* decoder alongside the raw image features, the network learns to isolate illumination edges via elimination. If a strong image gradient exists without a corresponding CCR gradient, it is mathematically guaranteed to be a shading/shadow edge. This allows Decoder A to aggressively absorb cast shadows, keeping the downstream albedo map pure.