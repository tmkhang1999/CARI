# Project Documentation: Multi-Stage Intrinsic Image Decomposition & Retouching

> **Last updated:** 2026-03-15 — aligned with current Stage 1 code and dataloader behavior.
> Key updates in this refresh: bounded per-worker LRU decode cache (`cache_max_items` / `mid_cache_max_items`), index-driven sampling only (no replay-style sample substitution), deterministic val crop option (`crop_mode_val`), mixed-loader seed (`dataloader_seed`), and updated base loss/config defaults.
> 2026-03-06 additions: backbone switched to ConvNeXt V2 Base (89M, z=1024); `compute_ccr` now returns 6ch (3 clamped log-CCR + 3 norm-RGB chromaticity), CCR encoder `in_channels=6`.

---

## 0. Overall Pipeline (Concise)

**Stage 1: Intrinsic Decomposition**

```
Encoders:
  RGB     → ConvNeXt V2 Base (fine-tuned, 89M params)
            → Z_global (N, 1024, H/32, W/32)
            → F_img_i  at 4 scales [(N,128,H/4,W/4), (N,256,H/8,W/8),
                                    (N,512,H/16,W/16), (N,1024,H/32,W/32)]

  Normals → DW Pyramid Encoder (GroupNorm + GELU)
            → F_N_i at 4 scales [(N,64,H/2,W/2), (N,128,H/4,W/4),
                                  (N,256,H/8,W/8), (N,512,H/16,W/16)]

  CCR     → 5-stage CNN Encoder (BN + ReLU)
            → F_CCR_i at 4 scales [(N,64,H/2,W/2), (N,128,H/4,W/4),
                                    (N,256,H/8,W/8), (N,512,H/16,W/16)]
            → computed online via compute_ccr(rgb) inside model.forward()

  Seg     → integer class IDs (N, H, W), no encoder
            → nearest-resize to decoder stage resolution inside SPADE

Cascade (V4 full):
  Z + F_img_i + ResidualAttn(F_N_i)                          → Dec A → S_g (N,1,H,W)
                                                               → ShadingAdapter → S_g_i[4 scales]
  Z + F_img_i + S_g_i                                        → Dec B → xi (N,2,H,W)
                                                               → C = f(xi)  → S_c = S_g×C (N,3,H,W)
                                                               → ColorfulAdapter → S_c_i[4 scales]
  Z + F_img_i + ResidualAttn(F_CCR_i) + SPADE(Seg) + S_c_i  → Dec C → A_d (N,3,H,W)
                                                               → AlbedoAdapter → A_d_i[4 scales]
  Z + F_img_i + ResidualAttn(F_N_i) + A_d_i + S_c_i         → Dec D → S_d (N,3,H,W)
```

**Training Loop (after forward pass — scale matching required):**
```
A_pred (detached) + albedo_raw + valid_mask
  → scale_match() → c (N,1,1,1)
  → A_star = c * albedo_raw              → A_d_star (Dec C target)
  → S_star = rgb / (A_star + ε)
  → S_g_star = luminance(S_star)         → D_g_star = 1/(S_g_star+1)  (Dec A target)
  → xi_star  = [1/(C_RG+1), 1/(C_BG+1)] (Dec B target)
  → pi_star  = 1/(S_star+1)             (Dec D target)
  → FlexibleLoss(predictions, targets, m_diffuse, m_albedo, valid_mask, seg)
```

**Stage 2: Diffuse Shadow Retouching**
```
  S_d(low-res, 256×256) → Coefficient Predictor → Bilateral Grid (N, 12, D=8, 16, 16)
  S_d(full-res) + Seg   → Pixel-level network  → Guide Coords (N, H, W, 3) in [-1,1]
                                                 → slice_grid (trilinear) → (N, 12, H, W)
                                                 → apply_affine(sliced, S_d) → S_r (N,3,H,W)
  Final: Image_final = A_d × S_r
```

---

## 1. Development Roadmap (Architecture Versions)

### Version 1 — Bottleneck-Only Cross-Decoder

No attention, no SPADE, no per-stage adapter features.
Cross-decoder signals encoded to bottleneck resolution and concatenated into Z only.
F_img skips are raw-concatenated at every decoder stage.

```
BottleneckAdapter(x):  Conv3x3(in_ch→64, s=2) → BN → ReLU
                       Conv3x3(64→128, s=2)    → BN → ReLU
                       AdaptiveAvgPool2d(H/32, W/32)
                       Conv1x1(128 → Z_channels)
                       Output: (N, Z_channels, H/32, W/32)

Dec A: Z                                              → Dec A → S_g (N,1,H,W)
Dec B: cat[Z, BottleneckAdapter(S_g)] (N,Z×2,H/32)   → Dec B → xi (N,2,H,W)
Dec C: cat[Z, BottleneckAdapter(S_c)] (N,Z×2,H/32)   → Dec C → A_d (N,3,H,W)
Dec D: cat[Z, BottleneckAdapter(S_c), BottleneckAdapter(A_d)] (N,Z×3,H/32) → Dec D → S_d (N,3,H,W)

Skip connections: raw F_img concat at every stage (no attention)
No F_N, no F_CCR, no Seg
```

Decoder stages (all V1 decoders, using DecoderA/B/C/D classes):
```
Stage 3 (H/32→H/16): ConvTranspose → BN → ReLU                  out: 768ch
Stage 2 (H/16→H/8):  ConvTranspose → BN → ReLU + cat[F_img[2]] out: 384ch
Stage 1 (H/8→H/4):   ConvTranspose → BN → ReLU + cat[F_img[1]] out: 192ch
Stage 0 (H/4→H/2):   ConvTranspose → BN → ReLU + cat[F_img[0]] out: 96ch
Final:   bilinear ×2 → Head: Conv3x3(96→32) → ReLU → Conv1x1(32→out_ch) → activation
```

Validates: multi-task loss design, cascade data flow, scale-matching pipeline.

---

### Version 2 — Adapter Pyramid Cross-Decoder

Replace bottleneck-only injection with full adapter pyramids so cross-decoder signals
are scale-matched at every decoder stage. Uses ProgressiveDecoder.

```
ShadingAdapter(S_g):  1ch  → [32@H/2, 64@H/4, 128@H/8, 256@H/16]
ColorfulAdapter(S_c): 3ch  → [64@H/2, 128@H/4, 256@H/8, 512@H/16]
AlbedoAdapter(A_d):   3ch  → [64@H/2, 128@H/4, 256@H/8, 512@H/16]

Dec A inputs: Z (N, Z_ch, H/32)   [Z_ch=1024 for Base; 768 for Tiny in tests]
Dec B inputs per stage: cat[x_dec, F_img_i, S_g_i]
  stage extras: S_g_pyr[3]=256@H/16, S_g_pyr[2]=128@H/8, S_g_pyr[1]=64@H/4
Dec C inputs per stage: cat[x_dec, F_img_i, S_c_i]
  stage extras: S_c_pyr[3]=512@H/16, S_c_pyr[2]=256@H/8, S_c_pyr[1]=128@H/4
Dec D inputs per stage: cat[x_dec, F_img_i, S_c_i, A_d_i]
  stage extras: cat[S_c_pyr,A_d_pyr] = 1024@H/16, 512@H/8, 256@H/4
```

Validates: whether scale-matched priors outperform bottleneck-only injection.

---

### Version 3 — Normal Guidance (Attention)

Adds NormalEncoder and per-stage ResidualAttention in Dec A and Dec D.

```
NormalEncoder(normals):
  Input:  (N, 3, H, W) unit-vector normals
  Output: [F_N_1=(N,64,H/2), F_N_2=(N,128,H/4), F_N_3=(N,256,H/8), F_N_4=(N,512,H/16)]

ResidualAttention(x_channels, prior_channels):
  Proj1x1(prior_channels → x_channels) → sigmoid → gate
  x_att = x * (1 + gate)
  Input:  x (N, C_dec, h, w), prior (N, C_prior, h', w') [auto-resized]
  Output: (N, C_dec, h, w)

Dec A per stage:
  x      = deconv(prev)                         # (N, 768/384/192/96, ...)
  x_att  = ResidualAttention(x, F_N_i)          # gate from normal features
  x_cat  = cat[x_att, F_img_i, S_g_i_or_none]  → next deconv

Dec D per stage:
  x      = deconv(prev)
  x_att  = ResidualAttention(x, F_N_i)          # gate from normal features
  x_cat  = cat[x_att, F_img_i, S_c_i, A_d_i]   → next deconv

normals defaults to zeros if not provided (graceful degradation).
```

Validates: whether geometry priors improve gray shading and diffuse shading accuracy.

---

### Version 4 — CCR Guidance + Semantic Homogeneity

Adds CCREncoder, ResidualAttention in Dec C, and SPADE in Dec C.

```
CCREncoder(ccr):
  Input:  (N, 6, H, W) from compute_ccr(rgb):
            ch 0-2: clamped log cross-colour ratios [-1,1] (PIE-Net)
            ch 3-5: normalized RGB chromaticity [0,1] (SIGNet)
  Output: [F_CCR_1=(N,64,H/2), F_CCR_2=(N,128,H/4), F_CCR_3=(N,256,H/8), F_CCR_4=(N,512,H/16)]

SPADE(num_channels, num_classes=41):
  Input:  x (N, C_dec, h, w) + seg (N, H, W) integer class IDs
  Internal: nearest-resize seg → one-hot (N,41,h,w) → Conv3x3→ReLU → gamma,beta
  Output: InstanceNorm(x) * (1 + gamma) + beta  → (N, C_dec, h, w)    (N, C_dec, h, w)
  Fallback: returns x unchanged if seg is None

Dec C per stage:
  x      = deconv(prev)                         # (N, 768/384/192/96, ...)
  x_att  = ResidualAttention(x, F_CCR_i)        # gate from CCR features
  x_norm = SPADE(x_att, seg)                    # class-conditional norm
  x_cat  = cat[x_norm, F_img_i, S_c_i]          → next deconv

Activates Semantic TV Loss in L_C.
ccr computed online: compute_ccr(rgb) called inside model.forward() (no disk I/O).
seg defaults to None → SPADE becomes identity if seg is None.
```

Validates: whether reflectance boundary guidance and semantic homogeneity improve albedo.

---

### Version 5 — Stage 2 Integration

Freeze Stage 1. Attach and train DiffRetouch bilateral grid network on aesthetic/internal datasets.

---

## 2. Stage 1: Final Architecture (V4)

### 2.1 Encoders

**Image Encoder** (`src/models/encoders/image_encoder.py`)
| Property | Value |
|---|---|
| Input | `(N, 3, H, W)` linear RGB, tonemapped to [0,1] |
| Backbone | ConvNeXt V2 **Base** (`convnextv2_base.fcmae_ft_in22k_in1k`, 89M params) |
| Frozen | stages 1–2; fine-tuned stages 3–4 |
| Z_global | `(N, 1024, H/32, W/32)` |
| F_img skips (4 scales) | `[(N,128,H/4), (N,256,H/8), (N,512,H/16), (N,1024,H/32)]` |

> **Why Base, not Large:** The full V4 model has 4 decoders + 3 adapters + Normal/CCR encoders + SPADE.
> ConvNeXt V2 Large (198M, z=1536) exceeds 24 GB VRAM at bs=8 with all decoders active.
> ConvNeXt V2 Base (89M, z=1024) fits ~15 GB, leaving headroom for the guidance modules.
> The SOTA reference CD-IID uses MiDaS (~45M encoder); Base at 89M already exceeds this.
> No IID paper has demonstrated that encoder capacity beyond ~100M helps — the bottleneck
> is in loss formulation, cascade design, and data quality. Code uses `feature_channels`
> dynamically, so switching to Large later requires only a config change.
> ConvNeXt Tiny (used in tests) produces `[96, 192, 384, 768]` — still handled correctly.

Supported backbone aliases (set via `config['backbone']`):
```
"convnextv2_base"   → convnextv2_base.fcmae_ft_in22k_in1k   (production default)
"convnextv2_large"  → convnextv2_large.fcmae_ft_in22k_in1k   (if >24GB VRAM available)
"convnextv2_tiny"   → convnextv2_tiny.fcmae_ft_in22k_in1k    (tests / ablation)
"convnext_large"    → convnext_large.fb_in22k_ft_in1k         (V1 fallback)
"convnext_base"     → convnext_base.fb_in22k_ft_in1k
"convnext_tiny"     → convnext_tiny.fb_in22k_ft_in1k          (tests)
```

---

**Normal Encoder** (`src/models/encoders/normal_encoder.py`)
| Property | Value |
|---|---|
| Input | `(N, 3, H, W)` unit-vector surface normals |
| Architecture | DWConv3x3(s=2,pad=1) → PWConv1x1 → **GroupNorm(1,C)** → **GELU** (×4) |
| Normalization | GroupNorm(1, out_ch) ≡ LayerNorm over (C,H,W). **Not** BatchNorm. |
| Output F_N_i | `[(N,64,H/2), (N,128,H/4), (N,256,H/8), (N,512,H/16)]` |

```
Stage 1: DWConv3x3(3→3, s=2)  → PWConv1x1(3→64)   → GN(1,64)  → GELU  → (N,64,H/2,W/2)
Stage 2: DWConv3x3(64→64,s=2) → PWConv1x1(64→128)  → GN(1,128) → GELU  → (N,128,H/4,W/4)
Stage 3: DWConv3x3(128→128,s=2)→ PWConv1x1(128→256)→ GN(1,256) → GELU  → (N,256,H/8,W/8)
Stage 4: DWConv3x3(256→256,s=2)→ PWConv1x1(256→512)→ GN(1,512) → GELU  → (N,512,H/16,W/16)
```

---

**CCR Encoder** (`src/models/encoders/ccr_encoder.py`)
| Property | Value |
|---|---|
| Input | `(N, 6, H, W)` invariant descriptor from `compute_ccr(rgb)` — see table below |
| Architecture | **5 stages**, standard Conv3x3 + BN + ReLU |
| Stage 0 | Conv3x3(**6**→64, **s=1**), BN, ReLU → `(N,64,H,W)` — not returned |
| Stage 1 | Conv3x3(64→64, s=2), BN, ReLU  → F_CCR_1 `(N,64,H/2,W/2)` |
| Stage 2 | Conv3x3(64→128, s=2), BN, ReLU → F_CCR_2 `(N,128,H/4,W/4)` |
| Stage 3 | Conv3x3(128→256,s=2), BN, ReLU → F_CCR_3 `(N,256,H/8,W/8)` |
| Stage 4 | Conv3x3(256→512,s=2), BN, ReLU → F_CCR_4 `(N,512,H/16,W/16)` |
| Output F_CCR_i | `[(N,64,H/2), (N,128,H/4), (N,256,H/8), (N,512,H/16)]` |

> **Critical:** Stage 0 must use **stride=1** to preserve full spatial resolution before downsampling.
> Standard conv (not DW) — CCR is spatially rich and needs full channel mixing (PIE-Net validated).
> CCR computed **online** inside `model.forward()` via `compute_ccr(rgb)`. No preprocessing files needed.

---

**Segmentation Input**
| Property | Value |
|---|---|
| Input | `(N, H, W)` integer class IDs (NYU-40: 0–40, i.e. 41 classes) |
| No encoder | Directly passed to SPADE |
| Inside SPADE | Nearest-resize to decoder stage resolution → one-hot `(N,41,h,w)` |
| Used in | Dec C only |

---

### 2.2 Adapter Pyramids (Cross-Decoder Sub-networks)

(`src/models/modules/adapters.py`)

**BottleneckAdapter** (Version 1 only)
```
Input:  (N, in_ch, H, W)
Conv3x3(in_ch→64, s=2) → BN → ReLU  →  (N, 64,  H/2, W/2)
Conv3x3(64→128,   s=2) → BN → ReLU  →  (N, 128, H/4, W/4)
AdaptiveAvgPool2d(Z_h, Z_w)           →  (N, 128, Z_h, Z_w)
Conv1x1(128→Z_channels)               →  (N, Z_channels, Z_h, Z_w)
Output: (N, Z_channels, H/32, W/32)  [Z spatial size matched at first forward]
```

**ShadingAdapter** (Dec A → Dec B, V2+)
```
Input:  S_g (N, 1, H, W)
Conv3x3(1→32,    s=2) → BN → ReLU  → S_g_1 (N, 32,  H/2,  W/2)
Conv3x3(32→64,   s=2) → BN → ReLU  → S_g_2 (N, 64,  H/4,  W/4)   ← injected at Dec B stage 1
Conv3x3(64→128,  s=2) → BN → ReLU  → S_g_3 (N, 128, H/8,  W/8)   ← injected at Dec B stage 2
Conv3x3(128→256, s=2) → BN → ReLU  → S_g_4 (N, 256, H/16, W/16)  ← injected at Dec B stage 3
```
Dec B receives: `stage_extra_channels = (256, 128, 64)` at stages (s3, s2, s1).

**ColorfulAdapter** (Dec B → Dec C/D, V2+)
```
Input:  S_c = S_g × C  (N, 3, H, W)  [elementwise multiply, no params]
Conv3x3(3→64,    s=2) → BN → ReLU  → S_c_1 (N, 64,  H/2,  W/2)
Conv3x3(64→128,  s=2) → BN → ReLU  → S_c_2 (N, 128, H/4,  W/4)   ← injected at Dec C/D stage 1
Conv3x3(128→256, s=2) → BN → ReLU  → S_c_3 (N, 256, H/8,  W/8)   ← injected at Dec C/D stage 2
Conv3x3(256→512, s=2) → BN → ReLU  → S_c_4 (N, 512, H/16, W/16)  ← injected at Dec C/D stage 3
```

**AlbedoAdapter** (Dec C → Dec D, V2+)
```
Input:  A_d (N, 3, H, W)
Conv3x3(3→64,    s=2) → BN → ReLU  → A_d_1 (N, 64,  H/2,  W/2)
Conv3x3(64→128,  s=2) → BN → ReLU  → A_d_2 (N, 128, H/4,  W/4)   ← injected at Dec D stage 1
Conv3x3(128→256, s=2) → BN → ReLU  → A_d_3 (N, 256, H/8,  W/8)   ← injected at Dec D stage 2
Conv3x3(256→512, s=2) → BN → ReLU  → A_d_4 (N, 512, H/16, W/16)  ← injected at Dec D stage 3
```
Dec D stage extras: `cat[S_c_pyr, A_d_pyr]` = `(1024, 512, 256)` at (s3, s2, s1).

---

### 2.3 Residual Attention (`src/models/modules/residual_attention.py`)

```python
class ResidualAttention(x_channels, prior_channels):
  Proj1x1: Conv2d(prior_channels → x_channels, kernel=1, bias=True)

  forward(x, prior):
    Input:  x     (N, x_channels, h, w)
            prior (N, prior_channels, h', w')  [bilinear-resized if h'≠h]
    gate  = sigmoid(Proj1x1(prior))            (N, x_channels, h, w)
    Output: x * (1 + gate)                     (N, x_channels, h, w)
```

Applied to decoder features **before** concatenating with skips:
```
order: deconv(prev) → ResidualAttention(x, prior) → cat[x_att, skip, extras]
```

| Decoder | Prior    | prior_channels | Stage widths gated |
|---------|----------|----------------|--------------------|
| Dec A   | F_N_i    | 512/256/128    | 768/384/192        |
| Dec C   | F_CCR_i  | 512/256/128    | 768/384/192        |
| Dec D   | F_N_i    | 512/256/128    | 768/384/192        |
| Dec B   | —        | —              | none               |

The Proj1x1 also auto-resizes the prior spatially (bilinear) when prior and decoder
feature spatial sizes differ.

---

### 2.4 SPADE Normalization (`src/models/modules/spade.py`)

```python
class SPADE(num_channels, num_classes=41, hidden_channels=64):

  Components:
    InstanceNorm2d(num_channels, affine=False)
    embed: Conv2d(num_classes, 64, k=3, pad=1) → ReLU
    gamma: Conv2d(64, num_channels, k=3, pad=1)
    beta:  Conv2d(64, num_channels, k=3, pad=1)

  forward(x, seg):
    Input:
      x:   (N, num_channels, h, w)  decoder features
      seg: (N, H, W) or (N, 1, H, W) integer class IDs (long)
    Processing:
      1. nearest-resize seg to (h, w)
      2. clamp to [0, num_classes-1]
      3. one-hot encode → (N, num_classes, h, w) float
      4. embed → (N, 64, h, w)
      5. gamma, beta from separate Conv2d heads
    Output:
      InstanceNorm(x) * (1 + gamma) + beta    (N, num_channels, h, w)
    Fallback: returns x unchanged if seg is None
```

Applied in Dec C only, **after** CCR attention and **before** concatenating with skips:
```
order: deconv → ResidualAttention(x, F_CCR_i) → SPADE(x_att, seg) → cat[x_norm, skip, S_c_i]
```

num_classes must match dataset seg labels. Default 41 = NYU-40 classes (0–40).
Set via `config['num_seg_classes']` in V4 model config.

---

### 2.5 Decoder Cascade — ProgressiveDecoder (`src/models/decoders/progressive_decoder.py`)

Used by V2, V3, V4. Each decoder is a `ProgressiveDecoder` instance with
version-specific `stage_extra_channels` and `stage_ops`.

**Internal channel widths (fixed, independent of backbone):**
```
dec3: in_channels → 768
dec2: (768 + skip[2] + e3) → 384
dec1: (384 + skip[1] + e2) → 192
dec0: (192 + skip[0] + e1) → 96
```
Where `skip[i]` comes from the image encoder (dynamic), `e3/e2/e1` are extra adapter channels.

**Spatial flow (input H×W = 384×384):**
```
z (H/32=12)  → dec3 → 768@H/16 → [op3] → cat[skip[2], extra3] → dec2
               → 384@H/8  → [op2] → cat[skip[1], extra2] → dec1
               → 192@H/4  → [op1] → cat[skip[0], extra1] → dec0
               → 96@H/2   → bilinear×2 → 96@H
               → head: Conv3x3(96→32) → ReLU → Conv1x1(32→out_ch) → activation
Output: (N, out_ch, H, W)
```

**Stage ops** (`stage_ops=[op3, op2, op1]`):
- Run on `x_dec` **after** deconv, **before** cat with skips
- Each op is a callable: `x = op(x)` → returns same shape
- In V3/V4: lambdas closing over `ResidualAttention` or `ResidualAttention + SPADE`

**Output activations by decoder:**
| Dec | out_ch | activation | range |
|-----|--------|-----------|-------|
| A   | 1      | Softplus  | (0, ∞) → represents S_g in log-safe positive space |
| B   | 2      | Sigmoid   | (0, 1) → bounded xi ratio |
| C   | 3      | Sigmoid   | (0, 1) → albedo is bounded by definition |
| D   | 3      | Softplus  | (0, ∞) → unbounded HDR shading |

---

### 2.6 Per-Version Decoder Inputs Summary

**Version 1 (V1)** — uses `DecoderA/B/C/D` from `decoder.py`
```
Dec A: Z (N,Z_ch,H/32)                                        → (N,1,H,W)  S_g
Dec B: cat[Z, BN_adapt(S_g)] (N,2·Z_ch,H/32)                 → (N,2,H,W)  xi
Dec C: cat[Z, BN_adapt(S_c)] (N,2·Z_ch,H/32)                 → (N,3,H,W)  A_d
Dec D: cat[Z, BN_adapt(S_c), BN_adapt(A_d)] (N,3·Z_ch,H/32)  → (N,3,H,W)  S_d
```

**Version 2 (V2)** — uses `ProgressiveDecoder`
```
Dec A: Z, no extras                                            → (N,1,H,W)  S_g
Dec B: Z, extras=[S_g_pyr[3](256), S_g_pyr[2](128), S_g_pyr[1](64)]
Dec C: Z, extras=[S_c_pyr[3](512), S_c_pyr[2](256), S_c_pyr[1](128)]
Dec D: Z, extras=[cat(S_c,A_d)_pyr=(1024,512,256)]
```

**Version 3 (V3)** — V2 + ResidualAttention in A and D
```
Dec A: Z, no extras, stage_ops=[attn_a3(F_N[3]), attn_a2(F_N[2]), attn_a1(F_N[1])]
Dec B: same as V2
Dec C: same as V2 (no attention)
Dec D: Z, extras same as V2, stage_ops=[attn_d3(F_N[3]), attn_d2(F_N[2]), attn_d1(F_N[1])]
```

**Version 4 (V4)** — V3 + ResidualAttention+SPADE in C, CCR encoder
```
Dec A: same as V3
Dec B: same as V2/V3
Dec C: Z, extras same as V2, stage_ops=[CCR_attn+SPADE at each stage]
Dec D: same as V3
```

---

## 3. Flexible Supervised Loss Design

### 3.1 Routing Masks

| Dataset      | M_diffuse | M_albedo |
|--------------|-----------|----------|
| Hypersim     | 1         | 1        |
| InteriorNet  | 0         | 1        |
| MIDIntrinsic | 0         | 1        |

Routing: Dec C is gated by `M_albedo`; Dec D is gated by `M_diffuse`; Dec A/B always train on valid pixels (using pseudo-GT chain when diffuse GT is absent).

**Detach scope when `M_diffuse=0`:** only detach `S_d` to block Dec D gradients; keep `S_g`, `xi`, `S_c`, and `A_d` in the graph so Dec A/B/C still learn from pseudo-GT.

### 3.2 Scale Matching (Training Loop, After Forward Pass)

**Why scale matching is required:** The intrinsic decomposition has an inherent scale ambiguity:
`I = A_d × S_d  ≡  (c·A_d) × (S_d/c) for any scalar c. Raw dataset GTs and network predictions
operate at different arbitrary scales. Comparing directly produces unstable gradients.

**Implementation** (`train_stage1.py: scale_match() + compute_targets()`):
```python
# After forward pass, before loss:
c      = scale_match(albedo_raw, A_pred.detach(), valid_mask)
         # c = (a·b)/(a·a) least-squares, over valid pixels only
         # Returns (N,1,1,1) per-image scalar

A_star = c * albedo_raw                    # (N,3,H,W) scale-matched albedo GT
S_star = rgb / (A_star + ε)               # (N,3,H,W) implied shading GT

# Target-space conversions:
S_g_star = 0.2126·S[:,0] + 0.7152·S[:,1] + 0.0722·S[:,2]   # (N,1,H,W) luminance
D_g_star = 1/(S_g_star + 1)             # (N,1,H,W)  Dec A target, ∈(0,1]
xi_star  = [1/(C_RG+1), 1/(C_BG+1)]      # (N,2,H,W)  Dec B target, ∈(0,1]
A_d_star = A_star                          # (N,3,H,W)  Dec C target
pi_star  = 1 / (S_star + 1)               # (N,3,H,W)  Dec D target, ∈(0,1]
```

### 3.3 Loss Formulations

**FlexibleLoss forward signature:**
```python
FlexibleLoss.forward(predictions, targets, m_diffuse, m_albedo, valid_mask, seg_map=None)

predictions: dict  { s_g: (N,1,H,W), xi: (N,2,H,W), c: (N,3,H,W),
                     s_c: (N,3,H,W), a_d: (N,3,H,W), s_d: (N,3,H,W) }

targets: dict      { D_g_star: (N,1,H,W),   ← 1/(S_g_star+1), from scale_match
                     xi_star:  (N,2,H,W),   ← [1/(C_RG+1), 1/(C_BG+1)]
                     A_d_star: (N,3,H,W),   ← c * albedo_raw
                     pi_star:  (N,3,H,W) }  ← 1/(S_star+1)

m_diffuse:  (N,)  float, per-sample diffuse-shading availability (routes Dec D)
m_albedo:   (N,)  float, per-sample albedo availability (routes Dec C)

valid_mask: (N,1,H,W) bool; seg_map optional for SemanticTVLoss
```

**Dec A — Gray Shading (L_A)**
```
D_g_pred = 1 / (S_g_pred + 1)          ← inverse-space mapping inside loss
mask = valid_mask                      ← always active on valid pixels

L_A = masked_L1(D_g_pred, D_g_star, mask)
    + λ_msg · L_MSG(D_g_pred, D_g_star)
```

**Dec B — Chroma (L_B)**
```
xi_pred already in (0,1) from Sigmoid output
mask = valid_mask

L_B = masked_MSE(xi_pred, xi_star, mask)
    + λ_msg · L_MSG(xi_pred, xi_star)
```

**Dec C — Diffuse Albedo (L_C)**
```
mask = valid_mask × m_albedo            ← m_albedo always 1 for all datasets

L_C = mask_ratio · (
    masked_L1(A_d_pred, A_d_star, mask)
  + λ_msg · L_MSG(A_d, A_d_star)
  + λ_TV  · L_TV_walls(A_d_pred, seg)  ← TV on wall/floor/ceiling only (NYU-40: 1,2,22)
  + λ_p   · L_VGG(A_d_pred, A_d_star)  ← VGG16 relu1_2,relu2_2,relu3_3,relu4_3 features
  + λ_d   · DSSIM(A_d_pred, A_d_star)  ← (1-SSIM)/2, Gaussian window 11×11
)
```

**Dec D — Diffuse Shading (L_D)**
```
pi_pred = 1 / (S_d_pred + 1)           ← inverse-space mapping inside loss
mask = valid_mask × m_diffuse

L_D = masked_MSE(pi_pred, pi_star, mask)
    + λ_msg · L_MSG(pi_pred, pi_star) · (Σm_diffuse / N)
```
No reconstruction loss — avoids forcing Dec D to absorb specular residual R.

**Total:** `L_total = L_A + L_B + L_C + L_D`

### 3.4 Loss Weights
```yaml
lambda_msg:        0.5
lambda_tv:         0.1
lambda_perceptual: 0.05
lambda_dssim:      0.4
```

---

## 4. Dataset Loading Strategy

### 4.1 Dataset Configurations

**Directory layout for mixing (do not flatten/merge):**
```
../datasets/
    hypersim/          <- current sanity + production split source
        ai_001_001/images/scene_cam_00_final_hdf5/...
        ai_001_001/images/scene_cam_00_geometry_hdf5/...
        ai_001_002/...
    interiornet/       <- optional future dataset (M_diffuse=0, M_albedo=1)
    MIDIntrinsics/     <- real-world multi-illumination dataset (M_diffuse=0, M_albedo=1)
```
`MixedDataloader` handles mixing at training time by probability weight.
Each batch is sampled by dataloader index only; cache affects decode reuse, not sample selection.
No need to reorganise directories as more datasets are added.

**Current sanity set status (`../datasets/hypersim/`):**
```
Scenes:   ai_001_001, ai_001_002  (2 scenes, 3 cameras)
Frames:   59 color (every 5th frame = already decimated at stride=5)
Complete: 50 frames have full geometry (normal_cam + semantic)
  → 9 frames in scene_cam_01/ai_001_002 skipped (incomplete geometry)
Missing:  render_entity_id.hdf5 not downloaded
  → valid_mask falls back to albedo > 0.02 (~86% valid pixels — OK for sanity)
Split:    45 train / 5 val  (90/10)
```

**Hypersim production set** (`M_diffuse=1, M_albedo=1`)
- Type: Synthetic indoor, 1024×768, linear HDR, HDF5 format
- RGB → `images/scene_cam_XX_final_hdf5/frame.XXXX.color.hdf5` (float32, linear)
- Albedo → `frame.XXXX.diffuse_reflectance.hdf5` (float32, linear [0,1] by def)
- Illum → `frame.XXXX.diffuse_illumination.hdf5` (float32, linear HDR)
- Normals → `frame.XXXX.normal_cam.hdf5` (float32, unit vectors, camera space)
- Seg → `frame.XXXX.semantic.hdf5` (uint8, NYU-40: 1–40, 0=unlabelled)
- Valid mask → derived from `render_entity_id.hdf5` (exclude -1 = sky/invalid) + albedo > 0.02
- CCR → computed online in model.forward(), not loaded from disk
- Fixed split for all ablations: scene-level `hypersim_split.json` (seed=42, 90/10 scenes)
- Current production counts: 18,698 frames total = 16,866 train + 1,832 val

**InteriorNet** (`M_diffuse=0, M_albedo=1`) — albedo GT only
**MIDIntrinsic** (`M_diffuse=0, M_albedo=1`) — pseudo-albedo (median of 25 illuminations)

**Batch homogeneity:** Each batch comes from a single dataset to ensure consistent M_diffuse/M_albedo flags.

### 4.2 `__getitem__` Return Format

Returns **raw arrays only** — no scale matching or target-space conversion here.

```python
{
    'rgb':        (3, H, W)   float32, tonemapped linear [0,1]  ← input to encoder
    'albedo_raw': (3, H, W)   float32, raw linear HDR           ← unscaled, for scale_match
    'illum_raw':  (3, H, W)   float32, raw linear HDR           ← optional reference
    'normals':    (3, H, W)   float32, unit vectors
    'valid_mask': (1, H, W)   bool    = sky_mask & (albedo > 0.02)
    'seg':        (1, H, W)   long    NYU-40 class IDs
    'M_albedo':   bool        dataset-level flag
    'M_diffuse':  bool        dataset-level flag
}
```

> **Key principle:** Scale matching (`c = argmin_c ‖c·A_raw − A_pred‖²`) happens in the
> training loop AFTER the forward pass, not in the dataset. Targets are derived dynamically
> each iteration because the GT scale must be matched to the network's current predictions.

### 4.3 CCR: Online vs Offline

CCR is computed **online** inside `model.forward()`:
```python
ccr = compute_ccr(rgb)   # log space + 3×3 diff filter, ~0.5ms per batch
F_CCR_i = self.ccr_encoder(ccr)
```
This eliminates preprocessing files and disk I/O. The `preprocessor/compute_ccr.py` file
provides the same function for offline analysis/visualization, but is not used during training.

### 4.4 Tonemapping

Apply to RGB **input only** (fed to encoder + CCR computation). Not applied to GT arrays.
```python
def tonemap_linear(rgb, percentile=99.0):
    scale = np.percentile(rgb, percentile) + 1e-6
    return np.clip(rgb / scale, 0.0, 1.0)    # linear, no gamma
```

### 4.5 Caching and Augmentation Policy

- `cache_max_items` (Hypersim) and `mid_cache_max_items` (MIDIntrinsic) control bounded per-worker LRU decode caches.
- `0` disables decode caching for that dataset.
- Cache stores decoded arrays only; it does not override sampler indices.
- Train split uses stochastic crop/flip policy (`crop_mode_train`), val split uses deterministic crop policy (`crop_mode_val`) for stable ablations.

---

## 5. Training & Evaluation Pipeline

### 5.1 Configuration
```yaml
model:
  version: 4
  backbone: convnextv2_base
  z_channels: 1024
  freeze_stages: [1, 2]
  pretrained: true
  num_seg_classes: 41

train:
  optimizer: adam
  lr: 1.0e-5
  batch_size: 8
  num_workers: 4
  dataloader_seed: 42
  input_size: 384
  log_interval: 50
  val_interval_iters: 2000
  checkpoint_interval_iters: 5000

  phase1_iterations: 20000
  phase2_iterations: 50000
  extend_iterations: 75000
  sampling_weights_phase1: {hypersim: 1.0, midintrinsic: 0.0}
  sampling_weights_phase2: {hypersim: 0.6, midintrinsic: 0.4}

data:
  datasets: [hypersim, midintrinsic]
  hypersim_root: ../datasets/hypersim
  midintrinsic_root: ../datasets/MIDIntrinsics

  cache_max_items: 512
  mid_cache_max_items: 2048
  crop_mode_train: random
  crop_mode_val: center

  hypersim_split_file: hypersim_split.json
  hypersim_split_seed: 42
  hypersim_split_ratio: 0.9
  hypersim_strict_split: true
```

### 5.2 Training Loop (Iteration-Based Curriculum)

```python
for step in range(start_step, phase2_iterations):
    phase, weights = phase_schedule(step)  # phase1: 0-19999, phase2: 20000-49999
    mixed_loader.set_weights(weights)       # homogeneous batches by dataset

    batch, dataset_name = mixed_loader.next_batch()
    preds = model(batch['rgb'], m_diffuse=batch['M_diffuse'],
                  normals=batch.get('normals'), seg=batch.get('seg'))

    # detach only Dec D when diffuse GT unavailable
    if (batch['M_diffuse'] == 0).any():
        preds['s_d'] = detach_where_no_diffuse(preds['s_d'], batch['M_diffuse'])

    targets = compute_targets(preds, batch['rgb'], batch['albedo_raw'], batch['valid_mask'])
    losses = criterion(preds, targets, batch['M_diffuse'], batch['M_albedo'],
                       batch['valid_mask'], seg=batch.get('seg'))
```

### 5.3 Tensorboard Logging

**Scalars (every `log_interval` iterations):**
- `train/loss_a`, `train/loss_b`, `train/loss_c`, `train/loss_d`, `train/loss_total`
- `train/is_hypersim_batch`

**Validation scalars (every `val_interval_iters` iterations, fixed Hypersim val split):**
- `val/s_g_lmse`, `val/s_g_rmse`, `val/s_g_ssim`
- `val/xi_mse`
- `val/a_d_lmse`, `val/a_d_rmse`, `val/a_d_ssim`
- `val/s_d_lmse`, `val/s_d_rmse`, `val/s_d_ssim`
- plus `val/loss_a`, `val/loss_b`, `val/loss_c`, `val/loss_d`, `val/loss_total`

**Image grid (every `val_interval_iters` iterations):**
```
val/examples/sample_0, val/examples/sample_1
[Input | Pred_S_g | GT_S_g | Pred_A_d | GT_A_d | Pred_S_d | GT_S_d]
```

> These validation numbers/images are for controlled V1/V2/V3/V4 ablation comparison on one fixed split, not primary thesis claim metrics.

### 5.4 Evaluation Protocol

Scale-invariant metrics (Careaga & Aksoy 2023): normalize by mean before computing.
Benchmarks: Hypersim val split, MAW (real albedo), ARAP (OOD synthetic).

---

## 6. Stage 2: Diffuse Shadow Retouching (DiffRetouch)

*Train after Stage 1 is complete. Stage 1 weights fully frozen.*

### 6.1 Input / Output
```
Input:  S_d (N,3,H,W)   unbounded HDR from frozen Stage 1
        Seg (N,H,W)     integer class IDs (same as Stage 1)
        A_d (N,3,H,W)   from frozen Stage 1 (for final reconstruction)
Output: S_r (N,3,H,W)   retouched shading
Final:  Image_final = A_d × S_r
```

### 6.2–6.5 Architecture (unchanged)
See previous versions — bilateral grid, trilinear slicing, per-pixel 3×4 affine apply.

---

## 7. File Structure

```
IR-IID/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── setup.py
├── fig/
├── checkpoints/
├── datasets/
│   └── examples/                    # sample HDF5 frames for quick smoke tests
├── preprocessor/
│   └── compute_ccr.py               # offline CCR analysis/visualization only
├── scripts/
│   ├── train_stage1.sh
│   ├── train_stage2.sh
│   └── eval.sh
├── src/
│   ├── configs/
│   │   └── base.yaml
│   ├── data/
│   │   ├── __init__.py
│   │   ├── hypersim_dataset.py      # returns albedo_raw, valid_mask, M_diffuse/M_albedo
│   │   ├── interiornet_dataset.py
│   │   ├── midintrinsic_dataset.py
│   │   └── mixed_dataloader.py
│   ├── models/
│   │   ├── __init__.py              # exports V1-V4
│   │   ├── encoders/
│   │   │   ├── image_encoder.py     # ConvNeXt V2 wrapper, dynamic skip_channels
│   │   │   ├── normal_encoder.py    # DW Pyramid, GroupNorm(1,C), GELU
│   │   │   └── ccr_encoder.py       # 5-stage CNN, stride-1 first conv
│   │   ├── decoders/
│   │   │   ├── decoder.py           # DecoderA/B/C/D for V1
│   │   │   └── progressive_decoder.py  # ProgressiveDecoder for V2-V4
│   │   ├── modules/
│   │   │   ├── adapters.py          # BottleneckAdapter(V1), Shading/Colorful/AlbedoAdapter
│   │   │   ├── residual_attention.py # Proj1x1 gate: x*(1+sigmoid(Proj(prior)))
│   │   │   └── spade.py             # one-hot seg → Conv embed → InstanceNorm affine
│   │   ├── stage1_v1.py             # bottleneck-only, DecoderA/B/C/D
│   │   ├── stage1_v2.py             # adapter pyramid, ProgressiveDecoder
│   │   ├── stage1_v3.py             # +NormalEncoder, +ResidualAttention (Dec A/D)
│   │   ├── stage1_v4.py             # +CCREncoder, +ResidualAttention (Dec C), +SPADE
│   │   └── stage2_retouch.py
│   ├── losses/
│   │   ├── msg_loss.py              # 4-scale Sobel gradient L1
│   │   ├── perceptual_loss.py       # VGG16 relu1_2/2_2/3_3/4_3 MSE
│   │   ├── semantic_tv_loss.py      # TV on wall/floor/ceiling pixels
│   │   └── flexible_loss.py         # routes losses; expects scale-matched targets
│   ├── train_stage1.py              # includes scale_match() + compute_targets()
│   ├── train_stage2.py
│   └── eval.py
├── tests/
│   ├── test_models.py               # smoke tests V1-V4 forward, module shapes
│   └── test_losses.py               # FlexibleLoss with scale-matched targets
└── documents/
    ├── project_plan.md              # this file
    ├── data_processing_supplement.md
    └── references/
        ├── CD-IID.txt
        ├── DiffRetouch_code.txt
        ├── PIE-Net.txt
        └── SIGNet.txt
```

---

## 8. Key Physical Equations Reference

```
Physical models:
  Grayscale diffuse:     I = A_g × S_g
  RGB diffuse:           I = A_c × S_c          where S_c = S_g × C
  Intrinsic residual:    I = A_d × S_d + R

Scale ambiguity (resolved via dynamic scale matching):
  I = A_d × S_d  ≡  (c·A_d) × (S_d/c)   for any scalar c > 0
  Solution: c = argmin_c ‖c·A_raw − A_pred‖²  (least-squares, over valid pixels)

Chroma:
  C = [C_R/G, C_B/G]    = [S_c_r/S_c_g, S_c_b/S_c_g]
  xi = [1/(C_R/G+1), 1/(C_B/G+1)]   ∈ (0,1]²   (bounded training target)
  Recovery: C_R/G = (1/xi_0 − 1),  C_B/G = (1/xi_1 − 1)
  S_c = S_g × C                      (elementwise, no learnable parameters)

Inverse shading (compresses unbounded HDR to (0,1]):
  D_g   = 1/(S_g + 1)    Dec A target
  pi    = 1/(S_d + 1)    Dec D target
  A window pixel S_g=450 → D_g=0.0022; dark corner S_g=0.005 → D_g=0.995

Scale-matched GT derivation (after forward pass):
  A_star   = c × albedo_raw
  S_star   = rgb / (A_star + ε)
  S_g_star = 0.2126·S_r + 0.7152·S_g + 0.0722·S_b    (BT.709 luminance)
  D_g_star = 1/(S_g_star + 1)         → Dec A target  (N,1,H,W)
  xi_star  = [1/(S_r/S_g+1), 1/(S_b/S_g+1)] → Dec B target  (N,2,H,W)
  A_d_star = A_star                    → Dec C target  (N,3,H,W)
  pi_star  = 1/(S_star + 1)           → Dec D target  (N,3,H,W)

Stage 2:
  Residual:          R = I − (A_d × S_d)
  Stage 2 final:     Image_final = A_d × S_r
```

---

## 9. Known Design Decisions & Rationale

| Decision | Rationale |
|---|---|
| **ConvNeXt V2 Base** (not Large) as default backbone | Large (198M, z=1536) exceeds 24GB VRAM at bs=8 with 4 decoders+adapters. Base (89M, z=1024) fits ~15GB. SOTA CD-IID uses MiDaS (~45M); Base already exceeds this. No IID paper shows >100M encoder capacity helps. Switching to Large later is a single config change. |
| **`compute_ccr` clamps log-CCR to [-1, 1]** | Near-black pixels (img ≈ 0) produce log→−∞, making CCR values explode. PIE-Net applies this clamp explicitly. Without it the CCREncoder receives NaN/Inf gradients at dark image regions. |
| **`compute_ccr` outputs 6ch** (3 log-CCR + 3 norm-RGB) | Normalized RGB chromaticity (`c/(R+G+B)`) is a second illumination-invariant cue used by SIGNet. It encodes per-pixel hue independently of intensity, complementing log-CCR which encodes *boundary edges* between materials. Both are zero-parameter ops, so the only cost is 3 extra input channels to CCREncoder's first conv. |
| GroupNorm(1,C) in NormalEncoder instead of BatchNorm | Equivalent to channel-last LayerNorm; more stable on small batches; GELU consistent with ConvNeXt design |
| CCR encoder has 5 stages (stride-1 first) | First conv preserves full resolution for fine-grained boundary detection before downsampling; matches PIE-Net validated design |
| SPADE uses one-hot class embedding, not raw class ID | Integer IDs are not ordinal; one-hot allows independent per-class affine parameters |
| SPADE num_classes=41 (not 40) | NYU-40 labels are 1–40; class 0 = unlabelled background; 41 total |
| Scale matching uses albedo (not shading) | Albedo ∈ [0,1] by definition → numerically stable least-squares fit; shading is unbounded HDR |
| valid_mask applied to ALL 4 losses | Sky, mirror, glass pixels have invalid GT for all intrinsic components, not just albedo |
| `xi_star` derived from scale-matched S_star, not raw GT | Raw chroma GT would be at wrong scale; using S_star ensures consistency with the albedo scale |
| CCR computed online | Single GPU op (~0.5ms), eliminates preprocessing pipeline and disk storage overhead |
| ProgressiveDecoder hardcodes widths (768,384,192,96) | Architecture hyper-parameters independent of backbone; skip channels read dynamically from encoder |

