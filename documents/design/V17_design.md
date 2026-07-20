# V17 — Depth-Anything-for-IID: frozen DINOv2-L backbone + DPT, direct albedo, analytic residual

> **Status:** Design (2026-06). Replaces the old ConvNeXt parallel-head / `derive_albedo`
> V17 and the deleted V19 (DINOv2 cross-attention bolt-on). Standalone feed-forward IID;
> V18 (one-step SD diffusion) is kept separately as a potential teacher. This doc records
> the architecture, the design rationale, the data preprocessing contract, the thesis
> novelty framing, and the planned ablation study. Decisions marked **[open]** are parked.

---

## 0. TL;DR — the thesis

**A discriminative, self-supervised, illumination-*invariant* prior (DINOv2) is sufficient
for real-world intrinsic decomposition — recovering the shadow/albedo separation that a
generative diffusion prior (Marigold) achieves, but feed-forward and trained on synthetic
data only.** We graft an IID-specific physics decode (direct albedo + inverse-domain diffuse
shading, analytic residual) onto a frozen DINOv2-L backbone via a DPT decoder, and exploit
DINOv2's invariance both as the backbone and inside a GT-free material-consistency loss.

---

## 1. Architecture

```
I (linear) ─┬─ gamma → sRGB → DINOv2-L/14 (FROZEN, no_grad)
            │      get_intermediate_layers([4,11,17,23]) → 4× (B,1024,Hp,Wp)  + tokens
            │            │
            │      DPT Reassemble → {/4,/8,/16,/32} → RefineNet fusion → /2
            │            │
            └─ gamma → DetailStem (trainable conv) → /2 high-freq ──► concat → trunk F (/2)
                                                                          │
                                          ┌────────────────────────────────┴───────────┐
                                          ▼                                             ▼
                              Albedo head (sigmoid)                       Shading head (sigmoid)
                                          ▼                                             ▼
                                      A ∈ [0,1]                          π ∈ [π_floor,1]; S_d=(1−π)/π
                                          R = (I − A·S_d)₊  (analytic)    I = A·S_d + R
```

Files: `src/models/v17.py`, `src/models/encoders/dino_encoder.py`,
`src/models/decoders/dpt_decoder.py`, `src/configs/v17.yaml`. ~5M trainable (DPT+heads+stem);
DINOv2-L = 304M frozen. ~2.6 GB at bs=2/384².

---

## 2. Design rationale (why each choice)

- **Frozen DINOv2-L backbone — the core bet.** DINOv2's augmentation-invariant SSL objective
  makes a shadowed and a lit patch of the *same* material map to near-identical tokens, so
  albedo constancy across a shadow is a *feature property*, not something the decoder must
  hallucinate. Frozen preserves that invariance (and excludes 304M params from the optimizer).
- **DPT decoder + trainable DetailStem.** DPT reassembles 4 ViT layers into a {/4,/8,/16,/32}
  pyramid and fuses top-down; the /14 tokens are semantically rich but spatially coarse, so a
  small conv stem on the input restores high-frequency albedo edges. GroupNorm throughout
  (BatchNorm would strip the per-image illumination statistics IID needs).
- **Direct albedo, plain sigmoid (no rgI split).** The albedo loss scale-matches GT to the
  prediction with a **per-image scalar** (`scale_match`, all channels jointly), so chroma is
  already constrained — there is no per-channel scale freedom to desaturate into. The rgI
  chroma/intensity split only added a collapse mode (the V19 dark-blue albedo).
- **No normal prior.** Both training sets ship normals, but wild inference has none → a
  train/test modality mismatch that undercuts generalization. DINOv2 already encodes surface
  orientation. Dropped. (Re-add path if shading underperforms: modality-dropout training +
  Metric3D normals at inference — not a naive toggle.)
- **Inverse-domain shading π = 1/(S_d+1) ∈ [0,1].** Puts unbounded HDR diffuse shading inside a
  bounded sigmoid head; `π_floor=5e-3` ⇒ S_d ≤ ~199 headroom.
- **3-channel COLORED shading.** The shading head emits 3 channels and `S_d_star` is colored on
  both datasets (Hypersim: GT `illum`; InteriorVerse: `I/A`). So `I=A·S_d` with colored `S_d`
  already gives colored illumination a physical home in shading — the model has CCR's core
  property *for free*; no separate illuminant-chromaticity head is needed (a grayscale×chroma
  re-factor of an already-colored shading adds zero capability).
- **No hand-crafted color skips (off by default; ablation only).** A prior attempt fed
  chromaticity `c(I)=I/(R+G+B)`→albedo and luminance `L(I)`→shading to fight DINOv2's
  color-invariance desaturation. Dropped because (i) the typing was asymmetric (a 3-ch colored
  shading head fed only 1-ch luminance, color hint on the wrong head), (ii) `c(I)` cancels only
  *white* shading so colored light/clipping leaks the illuminant into albedo (a desaturation fix
  that becomes a color-leak), and (iii) given colored `S_d` above it adds no capability. The
  real-image desaturation is a **training-signal gap**, not a missing input prior: DINOv2
  discards absolute color by design and synthetic recon cannot correct it on real photos. The
  fix is a real-image signal (§7/§9, MLLM-judge), **measured after the synthetic baseline trains**.
- **Analytic residual `R=(I−A·S_d)₊`** — see §3.

---

## 3. Residual: analytic vs. learned decoder  **[open — default analytic; keep as ablation]**

**Decision: keep analytic `R=(I−A·S_d)₊`.** It is physics-exact, costs zero capacity, and
cannot leak (it is a deterministic function of `A,S_d,I`). A learned `[-max,max]` decoder
absorbs reconstruction error into arbitrary blobs (the old V17 R-map looked like the whole
image); its two justifications are redundant/harmful here — `L1(R,R_gt)` re-expresses
`A·S_d≈A_gt·S_d_gt` (already the diffuse-recon loss), and residual-sparsity pushes `A·S_d→I`
everywhere *including* speculars → forces the diffuse term to swallow highlights (leakage).

Add the learned decoder back only if scope expands to **explicit specular/material modeling
for relighting/editing**, or to represent overshoot (`A·S_d>I`, which the clamp discards).
Either way, "analytic vs learned-decoder vs none" is an **ablation row** — showing analytic ≥
learned defends the choice.

---

## 4. Data preprocessing contract  **(answers: do we still need gamma?)**

**Yes — gamma-encode the backbone/stem input; keep everything physical in linear.**

| Quantity | Space | Why |
|---|---|---|
| Dataset `rgb` into the model | **linear** [0,1] | LDR loader does `(px/255)^2.2`; HDR EXR is linear (tonemapped to [0,1]). |
| IID physics: targets `A,S_d,R`, `S=I/A`, `R=I−A·S_d`, recon, losses | **linear** | Light adds/multiplies linearly. |
| **DINOv2 encoder input** | **sRGB = rgb^(1/2.2)**, then ImageNet `(x−μ)/σ` | DINOv2 pretrained on sRGB internet images (LVD-142M, ImageNet-style transforms). Linear input is off-distribution (dark midtones) → degrades the invariance/material features the thesis depends on. |
| **DetailStem input** | **sRGB = rgb^(1/2.2)** | Same backbone-distribution logic; keeps the two encoder paths consistent. |

This mirrors the *old* V17, which gamma-encoded the ConvNeXt input (`x_enc=rgb.pow(1/2.2)`)
while keeping the physics linear. **Code note:** the DINOv2 encoder must gamma-encode before
normalization (it was inherited from V19 feeding linear — fixed). No double-gamma risk: the
data pipeline does not gamma the model input; the model owns the backbone-space conversion.

---

## 5. Loss & per-dataset reconstruction

`V17Loss` in `recon_mode: diffuse` (reused as-is): albedo MSE+MSG+DSSIM (both sets) · SSI
shading in π (Hypersim only, `M_diffuse`-gated) · two-sided diffuse reconstruction `A·S_d ≈
A_gt·S_d_gt` · GT-free DINOv2 material-consistency (`lambda_mat_consist=0.05`). Analytic-R
weights (`lambda_r/res_sparse/albedo_init/edge/grad_decorr/alb_invariance`) = 0.

**Recon target is `c`-independent and correct on both datasets:**
- **Hypersim** (`M_diffuse=1`): `S_d_star=illum/c` → target `A_star·S_d_star = A_gt·illum` =
  true diffuse render; specular falls into analytic R.
- **InteriorVerse** (`M_diffuse=0`): `S_d_star=I/A` → target `= (c·A_gt)·(I/(c·A_gt)) = I`
  **exactly** (albedo cancels) → robust to IV's render noise / imperfect albedo GT.

Both teach the *same thing about the output we care about — clean albedo*; speculars route to
shading on IV vs. R on Hypersim (benign, since albedo is supervised cleanly on both and
`sat_ok` masks blown highlights). Recon is the **only** shading constraint on IV (SSI is
Hypersim-gated) → keep it enabled, equal weight.

---

## 6. Relationship to DepthAnything — borrowed vs. novel

The **backbone+decoder *template* (frozen DINOv2 + DPT) is borrowed** from DPT/DepthAnything
and is *not* claimed as a contribution (reusing a proven dense-prediction skeleton, like
U-Net/ResNet). DepthAnything does single-task scale-invariant monocular depth; nothing in it
addresses the ill-posed albedo/shading split. The contribution lives in what we do *on top of
and because of* that backbone (§7).

---

## 7. Thesis novelty claims

Central hypothesis (§0). Supporting, defensible novelties:
1. **Invariance made explicit and exploited** — not "use a good backbone." We *measure* that
   DINOv2 collapses same-material patches across shadows, and *use* it as a
   DINOv2-similarity-weighted material-consistency loss (GT-free → transfers to real). The
   coupling is ours.
2. **Physics-in-the-loop analytic-residual decode on a frozen SSL backbone**: `[A | π]` +
   `R=(I−A·S_d)₊` + two-sided diffuse-recon. This decode/loss suite is not DepthAnything's.
3. **Synthetic-only real-world generalization without a generative model**: CD-IID/CRefNet
   wrangle real data; Marigold pays diffusion cost. We claim the frozen SSL prior suffices,
   feed-forward.

**Honest assessment:** the synthetic-only scope = solid *empirical/systems* contribution
(hypothesis + clean design + strong eval) but methodologically incremental, and DINOv2's
color-invariance leaves a real-image **desaturation** gap that no input prior closes (above).
The headline mechanism that makes synth→real *actually* work — and the planned contribution —
is a **real-image, label-free training signal**:

- **MLLM-judge fine-tuning (ReasonX, Adobe CVPR'26; arXiv:2512.04222).** A frozen multimodal
  LLM answers *pairwise* "are these two points the same material / which is more lit?" queries
  on unlabelled real photos; agreement between the judge and the analytic ordering of the model's
  own outputs becomes a **GRPO reward** (backbone + VAE/judge frozen, only the decoder updates).
  Reported **+9–25% WHDR on IIW**. Decisive fit here: it injects real-image supervision **without
  a saturated albedo target** (only ordinal comparisons), so unlike MID-as-target it cannot
  teach over-saturation — it *replaces MID's role* and removes its harm. Model-agnostic, so it
  bolts onto this frozen-DINOv2 feed-forward decoder; applying it to a non-diffusion student is
  itself novel (they used diffusion bases). **No distillation, no generative model, no labels.**
- Alternative if multi-illumination real data were available: SAIL-style albedo-constancy
  self-supervision (arXiv:2505.19751). Not pursued — no multi-illum data on hand.

**Decision: train the synthetic baseline first, measure the IIW/ARAP gap, then add the judge.**
The invariance-probe (§8) is the cheap synthetic-only evidence that motivates why the gap exists.

---

## 8. Ablation study (each row isolates one variable; same decoder/loss/data otherwise)

| # | Ablation | Variants | Claim tested | Priority |
|---|---|---|---|---|
| A | **Backbone (killer)** | DINOv2-L vs **supervised ImageNet ViT-L** vs ConvNeXt-V2 vs random — all frozen, same DPT | isolates SSL *invariance* from "ViT/DPT". DINOv2 ≫ supervised-ViT ⇒ thesis holds | must |
| B | Backbone frozen vs tuned | frozen / LoRA / full-FT | does FT destroy invariance/generalization? | must |
| C | Residual | analytic / learned-decoder+sparsity / none | defends §3 | must |
| D | Material-consistency loss | ± `lambda_mat_consist` | does exploiting invariance in the loss help on real? | high |
| E | Backbone size | DINOv2 S / B / L | capacity vs generalization | high |
| F | Data | Hypersim / +InteriorVerse / +real(MID) | "synthetic-only suffices" | high |
| G | Decoder/detail | DPT vs old progressive; ± DetailStem; 4 vs 2 ViT layers | high-freq source | med |
| H | Shading domain | π=1/(S+1) vs linear | HDR parametrization | med |
| I | Gamma input | sRGB (gamma) vs linear into DINOv2 | validates §4 | med |
| J | **MLLM-judge (the payoff)** | ± ReasonX GRPO fine-tune on real | the headline synth→real mechanism | high (planned) |
| K | Color skips | ± `albedo_chroma_skip`/`shading_lum_skip` | shows hand-crafted color routing does NOT generalize | med |

**Evaluation (not ablations):**
- Real benchmarks: IIW (WHDR), SAW, MAW, ARAP (`tests/eval/eval_arap.py`) — train synthetic, test real.
- **Shadow-constancy metric** (headline): albedo variance within a same-material region
  straddling a shadow (lower = better); report vs Marigold/CD-IID.
- **Mechanism evidence**: DINOv2 token cosine-similarity across shadow boundaries vs
  ConvNeXt/supervised-ViT (empirically shows the invariance claimed).
- **Efficiency**: params / latency vs Marigold (the feed-forward speed claim).

Most rows are config flags except A/B/E (backbone swap) and C (learned-R) — wire a
`backbone: {dinov2_*|vit_large|convnextv2_base}` switch + `residual_mode: {analytic|decoder}`.

---

## 9. Deferred levers / open decisions

- **[planned — headline] MLLM-judge GRPO fine-tuning on unlabelled real images (ReasonX,
  arXiv:2512.04222).** Phase-3 after the synthetic baseline. Replaces MID (drops its saturated
  targets), label-free, feed-forward-compatible. Scope before building: pick the judge VLM
  (InternVL2.5-4B / Qwen2.5-VL), implement the pairwise albedo/irradiance queries + analytic
  ordering of `a_d`/`S_d`, and the GRPO reward with KL-to-baseline. Ablation row J becomes
  "± MLLM-judge" instead of "± distillation".
- **[dropped]** Diffusion→DINOv2 distillation — superseded by the judge (no teacher model, no
  generative cost, and the user ruled out distillation for the thesis).
- **[open]** Hand-crafted color skips (`albedo_chroma_skip`/`shading_lum_skip`) — code retained,
  off by default; keep purely as the ablation that shows they do *not* generalize (white-light
  color-leak), motivating the judge.
- **[open]** Learned residual decoder (only if specular/relighting scope).
- **[open]** Normal prior via modality-dropout + inference normals (only if shading underperforms).
- **[open]** DINOv2-perceptual / light GAN loss for crispness if regression blur appears.
- **[open]** LoRA-unfreeze DINOv2 if frozen features underfit.
