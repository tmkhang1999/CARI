# Should You Use ConvNeXt Large? Analysis for Your Setup

## Your Constraints

- **GPU:** 2 × Quadro RTX 6000 (24 GB VRAM each)
- **Input:** 384 × 384
- **Batch size:** 8 (config)
- **Architecture:** 1 shared encoder → 4 decoders + 3 adapters + guidance encoders (V4)
- **Frozen:** stages 1–2 of the backbone (only stages 3–4 fine-tuned)

## ConvNeXt Family Comparison

| Variant | Params | Skip Channels | z_channels | FP16 Enc VRAM (bs=8, 384²) | Pretrain Quality |
|---|---|---|---|---|---|
| **Tiny** | 28M | [96, 192, 384, 768] | 768 | ~2.5 GB | IN-22k→1k |
| **Small** | 50M | [96, 192, 384, 768] | 768 | ~3.5 GB | IN-22k→1k |
| **Base** | 89M | [128, 256, 512, 1024] | 1024 | ~5 GB | IN-22k→1k |
| **Large** | 198M | [192, 384, 768, 1536] | 1536 | ~9 GB | IN-22k→1k |

## The Real Problem: Total Model VRAM

The encoder is only **one part** of VRAM. Your full V4 model also runs:

| Component | Large | Base |
|---|---|---|
| Image Encoder (198M / 89M) | ~9 GB | ~5 GB |
| 4× ProgressiveDecoder | ~4 GB | ~2.5 GB |
| 3× Adapter pyramids | ~0.5 GB | ~0.5 GB |
| Normal Encoder | ~0.1 GB | ~0.1 GB |
| CCR Encoder | ~0.1 GB | ~0.1 GB |
| 3× SPADE | ~0.1 GB | ~0.1 GB |
| Optimizer states (Adam: 2× params) | ~6 GB | ~3 GB |
| Activations / gradients (bs=8, 384²) | ~6 GB | ~4 GB |
| **Total estimated** | **~26 GB** | **~15 GB** |

**ConvNeXt Large at bs=8 will NOT fit in 24 GB.** You'd need to drop batch size to 4 or use gradient checkpointing, both of which hurt training quality.

## What the Reference Papers Actually Use

| Paper | Encoder | Params | Task |
|---|---|---|---|
| **CD-IID** (Careaga 2023) | MiDaS (ResNeXt-101) | ~45M | Intrinsic decomposition (same task) |
| **PIE-Net** (Das 2022) | VGG-16 BN | ~15M | Intrinsic decomposition |
| **SIGNet** (Das 2024) | VGG-16 BN | ~15M | Intrinsic decomposition |
| **Omnidata** | DPT-Large (ViT-L) | ~307M | Multi-task dense prediction |

The **state-of-the-art CD-IID** — which your project directly builds upon — uses **MiDaS with a ~45M ResNeXt-101 backbone**. Not a 198M backbone. They achieve their results with a decoder cascade similar to yours.

## Recommendation: **ConvNeXt V2 Base**

| Criterion | Large | **Base** | Small/Tiny |
|---|---|---|---|
| Fits in 24GB at bs=8? | ❌ Marginal/No | ✅ Yes (~15 GB) | ✅ Yes |
| Feature richness for IID? | Overkill | Sufficient | May underfit |
| Pretrain quality (IN-22k)? | ✅ Best | ✅ Very good | ⚠️ Weaker |
| Aligned with SOTA references? | Oversized vs CD-IID | Comparable to CD-IID | Undersized |
| z_channels | 1536 | 1024 | 768 |
| Skip channels | [192,384,768,1536] | [128,256,512,1024] | [96,192,384,768] |

### Why Base, not Small/Tiny:
- Intrinsic decomposition requires **multi-scale boundary understanding** — you need enough channels at H/16 and H/32 to separate material vs lighting edges. Base's 1024-dim bottleneck is strong enough; Tiny's 768 may struggle at the highest-frequency boundaries.
- Base's `[128, 256, 512, 1024]` skip channels are a **natural match** for the decoder widths `[768→384→192→96]` — the concatenation ratios stay balanced. With Large's `[192, 384, 768, 1536]`, the skips dominate at later stages.
- Base has 89M params — roughly 2× CD-IID's 45M encoder, giving you headroom for the extra guidance modules (Normal, CCR, SPADE) that CD-IID doesn't have.

### Why not Large:
- You have **4 decoders**, not 1. Each decoder stores its own activation maps. Large's wider channels blow up decoder VRAM quadratically.
- Stages 1–2 are frozen anyway — those are the "low-level feature" stages. The semantic understanding that matters comes from stages 3–4, and Base's pretrained representations at those stages are nearly as discriminative as Large's for a task like IID that operates at 384² (not 1024²+ like detection/segmentation benchmarks where Large pulls ahead).
- No IID paper has demonstrated that encoder capacity beyond ~100M params helps — the bottleneck is almost always in the loss formulation, cascade design, and data quality.
