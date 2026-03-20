# IR-IID: Intrinsic Image Decomposition & HDR Retouching

A multi-stage deep learning pipeline for physically-grounded intrinsic decomposition
of indoor HDR images, with diffuse shadow retouching.

**Stage 1** decomposes a linear HDR image into gray shading, colorful shading, diffuse
albedo, and diffuse shading via a cascade of 4 decoders sharing a single ConvNeXt V2
encoder. Four progressive architecture versions are implemented:

| Version | What's added |
|---|---|
| V1 | Bottleneck-only cross-decoder (baseline) |
| V2 | Adapter pyramids — scale-matched cross-decoder priors at all stages |
| V3 | V2 + Normal encoder with ResidualAttention in Dec A and Dec D |
| V4 | V3 + CCR encoder (6ch) + ResidualAttention in Dec C + SPADE semantic normalization |

**Stage 2** (stub) retouches the predicted diffuse shading via a bilateral grid network.

---

## Architecture at a Glance

```
RGB (N,3,H,W)
  └─► ConvNeXt V2 Base (89M, frozen stages 1-2)
        → Z_global (N, 1024, H/32, W/32)
        → F_img skips [(N,128,H/4), (N,256,H/8), (N,512,H/16), (N,1024,H/32)]

  ┌─► Dec A → S_g (N,1,H,W)      gray shading      [Softplus]
  │     ↓ ShadingAdapter
  ├─► Dec B → xi  (N,2,H,W)      bounded chroma     [Sigmoid]
  │     ↓ ColorfulAdapter  →  S_c = S_g × C
  ├─► Dec C → A_d (N,3,H,W)      diffuse albedo     [Sigmoid]
  │     ↓ AlbedoAdapter
  └─► Dec D → S_d (N,3,H,W)      diffuse shading    [Softplus]

V3/V4 also uses:
  Normals → DW Pyramid Encoder → F_N_i → ResidualAttention in Dec A, Dec D
V4 also uses:
  compute_ccr(rgb) → 6ch (3 clamped log-CCR + 3 norm-RGB) → CCR Encoder
                  → F_CCR_i → ResidualAttention + SPADE in Dec C
```

Training uses **dynamic scale matching** after each forward pass to resolve the inherent
scale ambiguity of intrinsic decomposition. All four decoder targets are derived from
the scale-matched albedo — not loaded from disk.

---

## Project Layout

```
IR-IID/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── setup.py
├── fig/                           # figures / qualitative results
├── checkpoints/
├── datasets/
│   └── examples/                  # sample HDF5 frames for smoke tests
├── preprocessor/
│   └── compute_ccr.py             # offline CCR analysis (training uses inline version)
├── scripts/
│   ├── train_stage1.sh
│   ├── train_stage2.sh
│   ├── eval.sh
│   └── infer_one_image.sh
├── src/
│   ├── configs/
│   │   └── base.yaml              # backbone, z_channels, loss weights, etc.
│   ├── data/
│   │   ├── hypersim_dataset.py    # M_diffuse=1, M_albedo=1
│   │   ├── interiornet_dataset.py # M_diffuse=0, M_albedo=1
│   │   ├── midintrinsic_dataset.py# M_diffuse=0, M_albedo=1
│   │   └── mixed_dataloader.py    # homogeneous batches per dataset
│   ├── models/
│   │   ├── encoders/
│   │   │   ├── image_encoder.py   # ConvNeXt V2 Base, dynamic skip_channels
│   │   │   ├── normal_encoder.py  # DW Pyramid, GroupNorm+GELU
│   │   │   └── ccr_encoder.py     # 5-stage CNN, in_channels=6
│   │   ├── decoders/
│   │   │   ├── decoder.py         # DecoderA/B/C/D (V1)
│   │   │   └── progressive_decoder.py  # ProgressiveDecoder (V2-V4)
│   │   ├── modules/
│   │   │   ├── adapters.py        # BottleneckAdapter (V1), Shading/Colorful/AlbedoAdapter
│   │   │   ├── residual_attention.py  # x*(1+sigmoid(Proj(prior)))
│   │   │   └── spade.py           # one-hot seg → InstanceNorm affine
│   │   ├── stage1_v1.py
│   │   ├── stage1_v2.py
│   │   ├── stage1_v3.py
│   │   ├── stage1_v4.py
│   │   └── stage2_retouch.py
│   ├── losses/
│   │   ├── flexible_loss.py       # routes L_A/B/C/D with m_diffuse/m_albedo masks
│   │   ├── msg_loss.py            # 4-scale Sobel gradient L1
│   │   ├── perceptual_loss.py     # VGG16 relu1_2/2_2/3_3/4_3
│   │   └── semantic_tv_loss.py    # TV on wall/floor/ceiling (NYU-40)
│   ├── train_stage1.py            # scale_match() + compute_targets() + training loop
│   ├── train_stage2.py
│   └── eval_hypersim.py
├── tests/
│   ├── test_models.py             # V1-V4 forward pass + module shape checks
│   └── test_losses.py             # FlexibleLoss with scale-matched targets
└── documents/
    ├── project_plan.md            # full architecture, loss design, training pipeline
    ├── data_processing_supplement.md  # dataset download, HDR handling, dataloader
    └── references/
        ├── CD-IID.txt             # Careaga & Aksoy 2023/2024 (primary reference)
        ├── PIE-Net.txt            # Das et al. CVPR 2022
        ├── SIGNet.txt             # Das et al. ECCV 2022
        └── DiffRetouch.txt        # Stage 2 reference
```

---

## Quickstart

**1. Install dependencies** (Python ≥ 3.8, CUDA recommended):
```bash
pip install -r requirements.txt
```

**2. Run smoke tests** (CPU, no dataset required, ~30 seconds):
```bash
python tests/test_models.py
python tests/test_losses.py
```

**3. Train Stage 1** (requires Hypersim dataset):
```bash
bash scripts/train_stage1.sh
```

**4. Evaluate a checkpoint**:
```bash
torchrun src/eval_hypersim.py --checkpoint checkpoints/checkpoint_epoch_10.pth
```
The evaluation script now prints aggregate metrics to stdout only (no TensorBoard scalars/images and no saved eval preview images).

**5. Run full-size inference for one Hypersim frame**:
```bash
bash scripts/infer_one_image.sh \
  --checkpoint checkpoints/v1/checkpoint_latest.pth \
  --split val \
  --sample_idx 0 \
  --output_dir outputs/infer_one
```

Use `--match frame.0001` if you want to select by frame path substring.

---

## Key Design Decisions

| Decision | Why |
|---|---|
| **ConvNeXt V2 Base** (89M, z=1024) | Large (198M) exceeds 24GB VRAM at bs=8 with 4 decoders. Base fits ~15GB. SOTA reference CD-IID uses a ~45M MiDaS backbone. |
| **6-channel CCR** (ch 0-2: clamped log-CCR, ch 3-5: norm-RGB) | PIE-Net clamp prevents NaN at near-black pixels. SIGNet norm-RGB adds illumination-invariant hue. |
| **Dynamic scale matching** after forward pass | Intrinsic decomposition has inherent `I=(c·A)×(S/c)` ambiguity. GT must be matched to current predictions each step. |
| **Inverse shading space** `D=1/(S+1)` | Compresses unbounded HDR shading to (0,1], prevents bright windows from dominating gradients. |
| **M_diffuse / M_albedo routing** | Hypersim uses both flags as 1. MIDIntrinsic/InteriorNet use `M_albedo=1, M_diffuse=0`: Dec A/B/C still train (via scale-matched targets), while Dec D is gated by `M_diffuse`. |
| **Bounded decode cache (LRU)** | Dataloaders cache decoded arrays per worker with `cache_max_items`/`mid_cache_max_items`; cache does not replay or override sampler indices. |

---

## References

- **CD-IID:** Careaga & Aksoy, *Colorful Diffuse Intrinsic Image Decomposition in the Wild*, ACM ToG 2024
- **PIE-Net:** Das et al., *Photometric Invariant Edge Guided Network for IID*, CVPR 2022
- **SIGNet:** Das et al., *Semantic and Invariant Gradient Driven Network for IID*, ECCV 2022
- **DiffRetouch:** Stage 2 latent diffusion retouching reference

For full architecture documentation see [`documents/project_plan.md`](documents/project_plan.md).
For dataset setup and HDR data handling see [`documents/data_processing_supplement.md`](documents/data_processing_supplement.md).
