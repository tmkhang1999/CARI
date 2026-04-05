## V7 Complete Pipeline

### Encoders

```
─────────────────────────────────────────────────────────────────────
SHARED ENCODER
─────────────────────────────────────────────────────────────────────
RGB (N,3,H,W)
  → ConvNeXt V2 Base (freeze stages 1,2)
  → Z          (N, 1024, H/32, W/32)   ← global bottleneck
  → F_img_0    (N,  128, H/4,  W/4)    ← skip 0
  → F_img_1    (N,  256, H/8,  W/8)    ← skip 1
  → F_img_2    (N,  512, H/16, W/16)   ← skip 2
  → F_img_3    (N, 1024, H/32, W/32)   ← skip 3 (same as Z, kept for API)

Normals (N,3,H,W)
  → NormalEncoder (DW-sep, GroupNorm, GELU)
  → F_N_1  (N,  64, H/2,  W/2)
  → F_N_2  (N, 128, H/4,  W/4)
  → F_N_3  (N, 256, H/8,  W/8)
  → F_N_4  (N, 512, H/16, W/16)

CCR = compute_ccr(RGB)  (N,6,H,W)  [online, no extra memory]
  → CCREncoder (standard Conv, BN, ReLU)
  → F_CCR_1  (N,  64, H/2,  W/2)
  → F_CCR_2  (N, 128, H/4,  W/4)
  → F_CCR_3  (N, 256, H/8,  W/8)
  → F_CCR_4  (N, 512, H/16, W/16)

valid_mask (N,1,H,W)
  → IlluminantDescriptor
      log-chroma stats over valid pixels → (N,4)
      MLP: 4 → 128 → 2048
      split → gamma (N,1024,1,1), beta (N,1024,1,1)
  → z_b = Z * (1 + gamma) + beta       ← FiLM for Dec B only
```

---

### Dec A — Gray Shading

```
─────────────────────────────────────────────────────────────────────
DEC A
─────────────────────────────────────────────────────────────────────
Input to stage s3:
  Z  (N, 1024, H/32, W/32)

Stage s3  (H/32 → H/16):
  x = ConvTranspose(Z,          1024→768)   BN ReLU
  x = ResidualAttn(x, F_N_4)               768ch ← 512ch gate
  x = cat[x, F_img_2]                       768+512 = 1280ch

Stage s2  (H/16 → H/8):
  x = ConvTranspose(x,          1280→384)   BN ReLU
  x = ResidualAttn(x, F_N_3)               384ch ← 256ch gate
  x = cat[x, F_img_1]                       384+256 = 640ch

Stage s1  (H/8 → H/4):
  x = ConvTranspose(x,          640→192)    BN ReLU
  x = ResidualAttn(x, F_N_2)               192ch ← 128ch gate
  x = cat[x, F_img_0]                       192+128 = 320ch

Stage s0  (H/4 → H/2):
  x = ConvTranspose(x,          320→96)     BN ReLU

Final (H/2 → H):
  x = bilinear ×2
  x = Conv3×3(96→32) ReLU Conv1×1(32→1) Sigmoid

Output:
  D_g  (N, 1, H, W)   ∈ (0,1)

Derive:
  S_g  = 1 / (D_g + 1e-6) - 1              (N, 1, H, W)  unbounded HDR
  S_g_safe = S_g.clamp(0, 20)              ← clamp for encoder input only
  A_g  = (I / (S_g.clamp(1e-3) + 1e-6))
         .clamp(0,1)
         .nan_to_num(nan=0, posinf=1)       (N, 3, H, W)
```

---

### Dec B — Chroma

```
─────────────────────────────────────────────────────────────────────
DEC B  — GuidanceEncoder_B + FiLM
─────────────────────────────────────────────────────────────────────
GuidanceEncoder_B input:
  cat[S_g_safe]  (N, 1, H, W)
  → DW-sep pyramid (GroupNorm, GELU)
  → G_B_1  (N,  64, H/2,  W/2)
  → G_B_2  (N, 128, H/4,  W/4)
  → G_B_3  (N, 256, H/8,  W/8)
  → G_B_4  (N, 512, H/16, W/16)

Input to stage s3:
  z_b  (N, 1024, H/32, W/32)   ← FiLM-modulated Z

Stage s3  (H/32 → H/16):
  x = ConvTranspose(z_b,        1024→768)   BN ReLU
  x = ResidualAttn(x, G_B_4)               768ch ← 512ch gate
  x = cat[x, F_img_2]                       768+512 = 1280ch

Stage s2  (H/16 → H/8):
  x = ConvTranspose(x,          1280→384)   BN ReLU
  x = ResidualAttn(x, G_B_3)               384ch ← 256ch gate
  x = cat[x, F_img_1]                       384+256 = 640ch

Stage s1  (H/8 → H/4):
  x = ConvTranspose(x,          640→192)    BN ReLU
  x = ResidualAttn(x, G_B_2)               192ch ← 128ch gate
  x = cat[x, F_img_0]                       192+128 = 320ch

Stage s0  (H/4 → H/2):
  x = ConvTranspose(x,          320→96)     BN ReLU

Final (H/2 → H):
  x = bilinear ×2
  x = Conv3×3(96→32) ReLU Conv1×1(32→2) Sigmoid

Output:
  xi   (N, 2, H, W)   ∈ (0,1)

Derive:
  C_rg = (1 - xi[:,0:1]) / (xi[:,0:1] + 1e-7)
  C_bg = (1 - xi[:,1:2]) / (xi[:,1:2] + 1e-7)
  C    = cat[C_rg, ones, C_bg]             (N, 3, H, W)
  S_c  = S_g * C                           (N, 3, H, W)  unbounded HDR
  S_c_safe = S_c.clamp(0, 20)             ← clamp for encoder input only
  A_c  = (I / (S_c.clamp(1e-3) + 1e-6))
         .clamp(0,1)
         .nan_to_num(nan=0, posinf=1)       (N, 3, H, W)
```

---

### Dec C — Diffuse Albedo

```
─────────────────────────────────────────────────────────────────────
DEC C  — GuidanceEncoder_C + CCR + SPADE
─────────────────────────────────────────────────────────────────────
GuidanceEncoder_C input:
  cat[S_c_safe, A_c]  (N, 6, H, W)
  → DW-sep pyramid (GroupNorm, GELU)
  → G_C_1  (N,  64, H/2,  W/2)
  → G_C_2  (N, 128, H/4,  W/4)
  → G_C_3  (N, 256, H/8,  W/8)
  → G_C_4  (N, 512, H/16, W/16)

Input to stage s3:
  Z  (N, 1024, H/32, W/32)   ← unmodified Z, not z_b

Stage s3  (H/32 → H/16):
  x = ConvTranspose(Z,          1024→768)   BN ReLU
  x = ResidualAttn(x, F_CCR_4)             768ch ← 512ch gate  [material boundaries]
  x = SPADE(x, seg)                         768ch               [class homogeneity]
  x = ResidualAttn(x, G_C_4)               768ch ← 512ch gate  [shadow removal]
  x = cat[x, F_img_2]                       768+512 = 1280ch

Stage s2  (H/16 → H/8):
  x = ConvTranspose(x,          1280→384)   BN ReLU
  x = ResidualAttn(x, F_CCR_3)             384ch ← 256ch gate
  x = SPADE(x, seg)                         384ch
  x = ResidualAttn(x, G_C_3)               384ch ← 256ch gate
  x = cat[x, F_img_1]                       384+256 = 640ch

Stage s1  (H/8 → H/4):
  x = ConvTranspose(x,          640→192)    BN ReLU
  x = ResidualAttn(x, F_CCR_2)             192ch ← 128ch gate
  x = SPADE(x, seg)                         192ch
  x = ResidualAttn(x, G_C_2)               192ch ← 128ch gate
  x = cat[x, F_img_0]                       192+128 = 320ch

Stage s0  (H/4 → H/2):
  x = ConvTranspose(x,          320→96)     BN ReLU

Final (H/2 → H):
  x = bilinear ×2
  x = Conv3×3(96→32) ReLU Conv1×1(32→3) Sigmoid

Output:
  A_d  (N, 3, H, W)   ∈ (0,1)
```

---

### Dec D — Diffuse Shading

```
─────────────────────────────────────────────────────────────────────
DEC D  — GuidanceEncoder_D + Normals
─────────────────────────────────────────────────────────────────────
GuidanceEncoder_D input:
  cat[S_c_safe, A_d.detach()]  (N, 6, H, W)
  → DW-sep pyramid (GroupNorm, GELU)
  → G_D_1  (N,  64, H/2,  W/2)
  → G_D_2  (N, 128, H/4,  W/4)
  → G_D_3  (N, 256, H/8,  W/8)
  → G_D_4  (N, 512, H/16, W/16)

Input to stage s3:
  Z  (N, 1024, H/32, W/32)   ← unmodified Z

Stage s3  (H/32 → H/16):
  x = ConvTranspose(Z,          1024→768)   BN ReLU
  x = ResidualAttn(x, F_N_4)               768ch ← 512ch gate  [geometry]
  x = ResidualAttn(x, G_D_4)               768ch ← 512ch gate  [diffuse separation]
  x = cat[x, F_img_2]                       768+512 = 1280ch

Stage s2  (H/16 → H/8):
  x = ConvTranspose(x,          1280→384)   BN ReLU
  x = ResidualAttn(x, F_N_3)               384ch ← 256ch gate
  x = ResidualAttn(x, G_D_3)               384ch ← 256ch gate
  x = cat[x, F_img_1]                       384+256 = 640ch

Stage s1  (H/8 → H/4):
  x = ConvTranspose(x,          640→192)    BN ReLU
  x = ResidualAttn(x, F_N_2)               192ch ← 128ch gate
  x = ResidualAttn(x, G_D_2)               192ch ← 128ch gate
  x = cat[x, F_img_0]                       192+128 = 320ch

Stage s0  (H/4 → H/2):
  x = ConvTranspose(x,          320→96)     BN ReLU

Final (H/2 → H):
  x = bilinear ×2
  x = Conv3×3(96→32) ReLU Conv1×1(32→3) Sigmoid

Output:
  pi   (N, 3, H, W)   ∈ (0,1)
```

---

### Summary Table

```
Decoder  │ Bottleneck │ FiLM │ Normals │ CCR  │ SPADE │ GuidanceEnc input
─────────┼────────────┼──────┼─────────┼──────┼───────┼──────────────────
Dec A    │ Z          │  ✗   │   ✓     │  ✗   │  ✗    │ —
Dec B    │ z_b        │  ✓   │   ✗     │  ✗   │  ✗    │ S_g (1ch)
Dec C    │ Z          │  ✗   │   ✗     │  ✓   │  ✓    │ S_c, A_c (6ch)
Dec D    │ Z          │  ✗   │   ✓     │  ✗   │  ✗    │ S_c, A_d.det (6ch)
```

---

### Critical Implementation Notes

**Note 1 — ProgressiveDecoder `stage_extra_channels` stays `(0,0,0)` for all decoders**

All guidance enters through `stage_ops`, not `extra_features`. The `extra_features` mechanism from V2-V6 (pyramid adapters injected as cat) is completely removed in V7. Never set `stage_extra_channels` to anything other than `(0,0,0)` in V7.

**Note 2 — ConvTranspose input channels must be recomputed**

With the old pyramid extras removed, `dec2` input is now `768 + skip_channels[2]` with no extra channels. Verify `ProgressiveDecoder.__init__` matches exactly:

```python
# V7 Dec C example
self.decoder_c = ProgressiveDecoder(
    in_channels=1024,          # Z only, no adapter cat
    skip_channels=skip_channels,
    out_channels=3,
    stage_extra_channels=(0, 0, 0),
    activation="sigmoid",
)
```

**Note 3 — GuidanceEncoder is a new module, not NormalEncoder reused**

NormalEncoder takes 3ch normals. GuidanceEncoder_B takes 1ch, GuidanceEncoder_C and D take 6ch. Write a single `GuidanceEncoder(in_channels)` class with a configurable first conv, identical otherwise to NormalEncoder. Do not pass wrong channel counts.

**Note 4 — Gradient flow through implied albedo**

`S_g = 1/D_g - 1` is differentiable. `A_g = I/S_g` is differentiable through `S_g` back into Dec A. This is intentional — the guidance signal for Dec B carries implicit supervision about how well Dec A separated shading. However `A_c = I/S_c` is differentiable through Dec B. Whether to allow this gradient is a design choice. Recommend allowing it initially and watching if Dec B training destabilizes, then detaching if needed:

```python
# Start with gradients flowing (recommended)
a_c = (rgb / (s_c.clamp(1e-3) + 1e-6)).clamp(0, 1)
# If Dec B becomes unstable after ~5k steps, switch to:
# a_c = (rgb / (s_c.detach().clamp(1e-3) + 1e-6)).clamp(0, 1)
```

**Note 5 — SPADE seg input shape**

SPADE expects `(N, H, W)` or `(N, 1, H, W)`. Your dataloader returns `(N, 1, H, W)`. SPADE's `_prepare_seg` already handles both. But verify that when `seg=None` (e.g. MIDIntrinsics batch), SPADE returns `x` unchanged — this is already implemented but easy to break if you refactor SPADE.

**Note 6 — ResidualAttention ordering in Dec C**

The order CCR → SPADE → GuidanceC within each stage is fixed and meaningful. CCR gates on material boundaries first, SPADE normalizes by class, then GuidanceC removes residual shadows. If you flip GuidanceC before SPADE, the InstanceNorm in SPADE normalizes away the guidance signal before it can act.

**Note 7 — V7 is a standalone class, no inheritance**

```python
class IntrinsicDecompositionV7(nn.Module):
    def __init__(self, config):
        super().__init__()
        # build everything from scratch
        # no super().__init__() from V1-V6
```

This avoids accidentally inheriting adapters, old decoders, or wrong forward signatures from previous versions.