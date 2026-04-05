## Analysis of V7 Results

Four specific failures visible in the image:

**1. Gray Shading Pred is too flat and overexposed.** The window grid shadow pattern (sharp cast shadows) is largely missing. This is the root cause of everything else failing downstream — if Dec A doesn't capture the shadow, it can never be removed from albedo.

**2. Albedo Pred is far too dark and brownish.** GT is bright cream/white. This is a direct consequence of failure 1 — Dec A over-estimates shading brightness globally, so `A_d = I/S_d` comes out dark. The scale matching is working but Dec A is systematically wrong.

**3. Derived A_g still shows the window grid shadow pattern.** This means Dec A is not putting the cast shadow into D_g at all. The GuidanceEncoder chain cannot fix what the first decoder misses.

**4. Derived A_c is better than A_g but still has warm orange cast.** Dec B is partially correcting illumination color but the chroma estimate is incomplete without a clean S_g to work from.

The cascade failure is clear: Dec A is the weak link and everything propagates from there.

---

## Root Cause

Dec A currently has only normal guidance via ResidualAttention(F_N). Surface normals encode geometry — but the window grid cast shadow is not a geometric feature, it is a **lighting discontinuity on a flat surface with constant normal**. The normal is identical on both sides of the shadow boundary. So normal guidance gives Dec A zero signal about where the shadow boundary is.

What Dec A actually needs is a signal that distinguishes **shadow edges** from **material edges**. This is exactly what CCR encodes: CCR fires strongly at material boundaries (color ratios change) and fires weakly at shadow boundaries (same material on both sides, color ratios stay constant under neutral illuminant). Giving CCR to Dec A tells it: "this edge has low CCR response → it is a shadow, put it in shading."

---

## V8 Design

### Key Changes from V7

```
V7 → V8 changes:
1. Add CCR guidance to Dec A  ← most critical, fixes root cause
2. Add GuidanceEncoder_A(A_g) to Dec A  ← explicit implied albedo loop-back
3. Replace ShadowGuidanceInjection(tanh) with ConcatProjection in Dec C  ← stronger suppression
4. Remove SPADE from Dec C stage s3  ← SPADE at coarsest scale fights with CCR+guidance
5. GuidanceEncoder_B now takes cat[S_g, A_g] (2ch) not just S_g  ← A_g helps chroma see shadows
```

---

### Encoders

```
─────────────────────────────────────────────────────────────────────
SHARED ENCODERS  (unchanged from V7)
─────────────────────────────────────────────────────────────────────
RGB → ConvNeXt V2 Base → Z (N,1024,H/32,W/32)
                       → F_img_0 (N,128,H/4)
                       → F_img_1 (N,256,H/8)
                       → F_img_2 (N,512,H/16)

Normals → NormalEncoder → F_N_1..4 (64,128,256,512 at H/2..H/16)

CCR = compute_ccr(RGB) → CCREncoder → F_CCR_1..4 (64,128,256,512 at H/2..H/16)

valid_mask → IlluminantDescriptor → gamma, beta (N,1024,1,1)
z_b = Z * (1 + gamma) + beta
```

---

### Dec A — Gray Shading (V8 Key Change)

```
─────────────────────────────────────────────────────────────────────
DEC A  — Normals + CCR  ← NEW: CCR added
─────────────────────────────────────────────────────────────────────
Input:  Z (N, 1024, H/32, W/32)

Stage s3  (H/32 → H/16):
  x = ConvTranspose(Z,         1024→768)   BN ReLU
  x = ResidualAttn(x, F_N_4)              sigmoid  [geometry — where shading varies]
  x = ResidualAttn(x, F_CCR_4)            sigmoid  [material boundary — NOT shadow]
  x = cat[x, F_img_2]                     768+512 = 1280ch

Stage s2  (H/16 → H/8):
  x = ConvTranspose(x,         1280→384)   BN ReLU
  x = ResidualAttn(x, F_N_3)              sigmoid
  x = ResidualAttn(x, F_CCR_3)            sigmoid
  x = cat[x, F_img_1]                     384+256 = 640ch

Stage s1  (H/8 → H/4):
  x = ConvTranspose(x,          640→192)   BN ReLU
  x = ResidualAttn(x, F_N_2)              sigmoid
  x = ResidualAttn(x, F_CCR_2)            sigmoid
  x = cat[x, F_img_0]                     192+128 = 320ch

Stage s0  (H/4 → H/2):
  x = ConvTranspose(x,          320→96)    BN ReLU

Final: bilinear ×2 → Conv3×3(96→32) ReLU → Conv1×1(32→1) → Sigmoid

Output: D_g (N,1,H,W) ∈ (0,1)

Derive:
  S_g     = 1 / (D_g + 1e-6) - 1
  S_g_safe = S_g.clamp(0, 20)
  A_g     = (I / (S_g.clamp(1e-3) + 1e-6)).clamp(0,1).nan_to_num(0,1,0)
```

**Why CCR helps Dec A specifically:** At a window grid shadow boundary, CCR is near-zero (same wood material on both sides). At a material boundary (wood frame vs paper panel), CCR is large. Dec A can now learn: "large CCR = material boundary, keep this out of shading; small CCR + intensity edge = shadow boundary, put this in shading." This is the missing signal in V7.

---

### Dec B — Chroma (V8 Change: A_g added)

```
─────────────────────────────────────────────────────────────────────
DEC B  — FiLM + GuidanceEncoder_B(S_g, A_g)  ← NEW: A_g added
─────────────────────────────────────────────────────────────────────
GuidanceEncoder_B input:
  cat[S_g_safe, A_g]  (N, 4ch, H, W)   ← was 1ch in V7
  → DW-sep pyramid → G_B_1..4

Input: z_b (FiLM-modulated)

Per stage: same structure as V7
  x = ConvTranspose(prev)
  x = ResidualAttn(x, G_B_i)   sigmoid
  x = cat[x, F_img_i]

Output: xi → C → S_c → A_c  (same derivation as V7)
```

---

### Dec C — Diffuse Albedo (V8 Key Change: Stronger injection)

```
─────────────────────────────────────────────────────────────────────
DEC C  — CCR + SPADE + ConcatProjection(G_C)  ← STRONGER than V7 tanh
─────────────────────────────────────────────────────────────────────
GuidanceEncoder_C input:
  cat[S_c_safe, A_c]  (N, 6ch, H, W)   unchanged
  → DW-sep pyramid → G_C_1..4

ConcatProjection module:
  cat[x, G_C_i] → Conv1×1(x_ch + guide_ch → x_ch) → LayerNorm → GELU
  This allows full additive AND subtractive correction unlike tanh gate.

Input: Z (unmodified)

Stage s3  (H/32 → H/16):
  x = ConvTranspose(Z,         1024→768)   BN ReLU
  x = ResidualAttn(x, F_CCR_4)            sigmoid  [material boundaries]
  x = ConcatProjection(x, G_C_4)          768+512→768, LayerNorm, GELU
  x = cat[x, F_img_2]                     768+512 = 1280ch
  ← NO SPADE at s3: coarsest scale, SPADE here fights with ConcatProjection

Stage s2  (H/16 → H/8):
  x = ConvTranspose(x,         1280→384)   BN ReLU
  x = ResidualAttn(x, F_CCR_3)            sigmoid
  x = SPADE(x, seg)                        ← SPADE at s2/s1 only
  x = ConcatProjection(x, G_C_3)          384+256→384, LayerNorm, GELU
  x = cat[x, F_img_1]                     384+256 = 640ch

Stage s1  (H/8 → H/4):
  x = ConvTranspose(x,          640→192)   BN ReLU
  x = ResidualAttn(x, F_CCR_2)            sigmoid
  x = SPADE(x, seg)
  x = ConcatProjection(x, G_C_2)          192+128→192, LayerNorm, GELU
  x = cat[x, F_img_0]                     192+128 = 320ch

Stage s0  (H/4 → H/2):
  x = ConvTranspose(x,          320→96)    BN ReLU

Final: bilinear ×2 → Conv3×3(96→32) ReLU → Conv1×1(32→3) → Sigmoid

Output: A_d (N,3,H,W) ∈ (0,1)
```

**Why ConcatProjection is stronger than tanh:** The tanh gate is `x * (1 + tanh(proj(G_C)))`. Even with tanh gate = -1 (maximum suppression), x becomes `x * 0 = 0` only if tanh saturates, which requires large proj outputs and rarely happens early in training. ConcatProjection `cat[x, G_C] → Conv1×1 → LayerNorm` lets the network learn any linear combination of the two signals including full replacement of x with information from G_C. The shadow boundary signal can completely override the decoder features rather than just modulating them.

---

### Dec D — Diffuse Shading (unchanged from V7)

```
─────────────────────────────────────────────────────────────────────
DEC D  — Normals + GuidanceEncoder_D  (V7 design, no changes)
─────────────────────────────────────────────────────────────────────
GuidanceEncoder_D input: cat[S_c_safe, A_d.detach()] (6ch)

Per stage:
  x = ConvTranspose(prev)
  x = ResidualAttn(x, F_N_i)    sigmoid
  x = ResidualAttn(x, G_D_i)    sigmoid  (tanh not needed — amplifying, not suppressing)
  x = cat[x, F_img_i]

Output: pi (N,3,H,W)
```

---

### Summary Table

```
Decoder │ Bottleneck │ FiLM │ Normals │ CCR      │ SPADE    │ GuidanceEnc input  │ Injection
────────┼────────────┼──────┼─────────┼──────────┼──────────┼────────────────────┼──────────────────
Dec A   │ Z          │  ✗   │ ✓ s3,2,1│ ✓ s3,2,1 │  ✗       │ —                  │ sigmoid ResAttn
Dec B   │ z_b        │  ✓   │  ✗      │  ✗       │  ✗       │ S_g, A_g   (4ch)   │ sigmoid ResAttn
Dec C   │ Z          │  ✗   │  ✗      │ ✓ s3,2,1 │ ✓ s2,s1  │ S_c, A_c   (6ch)   │ ConcatProjection
Dec D   │ Z          │  ✗   │ ✓ s3,2,1│  ✗       │  ✗       │ S_c, A_d.det (6ch) │ sigmoid ResAttn
```

---

### New Modules Required

**ConcatProjection:**
```python
class ConcatProjection(nn.Module):
    def __init__(self, x_channels, guide_channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(x_channels + guide_channels, x_channels,
                      kernel_size=1, bias=False),
            nn.GroupNorm(1, x_channels),   # LayerNorm equivalent
            nn.GELU(),
        )
        # Initialize as near-identity: output ≈ x at start of training
        # Prevents guidance from destabilizing early training
        nn.init.zeros_(self.proj[0].weight[:, :x_channels])
        nn.init.eye_(  self.proj[0].weight[:x_channels, :x_channels])

    def forward(self, x, guide):
        if guide.shape[-2:] != x.shape[-2:]:
            guide = F.interpolate(guide, size=x.shape[-2:],
                                  mode="bilinear", align_corners=False)
        return self.proj(torch.cat([x, guide], dim=1))
```

**GuidanceEncoder (single class, configurable in_channels):**
```python
class GuidanceEncoder(nn.Module):
    # Identical architecture to NormalEncoder
    # Only difference: in_channels is configurable (1, 4, or 6)
    def __init__(self, in_channels, channels=(64, 128, 256, 512)):
        # DW-sep blocks, GroupNorm, GELU
        # Same as NormalEncoder but takes in_channels argument
```

---

### Critical Notes

**Note 1 — ConcatProjection initialization is mandatory**

Without the near-identity initialization, `ConcatProjection` at the start of training outputs random features and immediately destroys the decoder's ability to learn from the loss signal. The initialization sets the first `x_channels` weights to approximate an identity mapping so output ≈ x at step 0.

**Note 2 — CCR in Dec A: use sigmoid not tanh**

Dec A uses CCR to identify material boundaries so it can correctly NOT put them in shading. This is an amplification task (amplify material boundary response), not suppression. Sigmoid ResidualAttention is correct here.

**Note 3 — SPADE removed from stage s3 in Dec C**

At H/16 spatial resolution, SPADE's InstanceNorm normalizes across the entire feature map. At this coarse scale, the shadow gradient spans the whole feature spatially — InstanceNorm zeroes it out before ConcatProjection can act on it. SPADE is only effective at s2 (H/8) and s1 (H/4) where spatial resolution is high enough that shadows and materials occupy distinct regions.

**Note 4 — A_g in GuidanceEncoder_B**

`A_g` tells Dec B where the illuminated vs shadowed regions are so chroma estimation avoids being biased by shadow color temperature. Without A_g, Dec B may predict warm chroma for shadow regions (which appear orange in the tonemapped image) rather than neutral. The orange cast visible in V7's Colorful Shading Pred is partially this effect.