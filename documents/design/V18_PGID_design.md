# V18 — Physics-Grounded One-Step Colorful-Diffuse Intrinsic Diffusion

> **Status:** Design (2026-06). Standalone latent-diffusion IID that does **not**
> depend on V17. This revision pivots the *training paradigm* from multi-step
> ε-diffusion (a Marigold-IID clone) to a **one-step, physics-grounded** model,
> after studying the Marigold reference (`documents/references/marigold`).
> Self-contained: motivation, the Marigold baseline (grounded in its code), the
> one-step lesson, the V18 design and its three novelties, formulation, loss,
> architecture, training/inference, datasets, the 24 GB plan, what changes from
> the current code, risks, and open decisions.

---

## 0. TL;DR — the thesis

**V18 fine-tunes the Stable-Diffusion U-Net as a *single-step* feed-forward
predictor that jointly outputs colorful albedo + diffuse shading, trained
*end-to-end with the project's physics-loss suite in image space*.**

The single-step choice is not just for speed — it is the *enabler*: you cannot
backprop a reconstruction / cross-task physics loss through a 4-step diffusion
unroll cheaply, so Marigold-IID is forced to train with **latent MSE only** and
never enforces `I = A·S_d + R`. One step makes image-space physics losses (a
two-sided **diffuse reconstruction** `A·S_d≈A_gt·S_d_gt`, SSI shading, MSG, DSSIM)
tractable in the training loop. **That combination — an SD generative prior +
physics-in-the-loop one-step fine-tuning for the colorful-diffuse split — is what
no existing method does.**

---

## 1. What Marigold-IID actually does (grounded in its code)

Read from `documents/references/marigold`. There are **two separate models**:

| | Appearance (`train_marigold_iid_appearance.yaml`) | Lighting (`train_marigold_iid_lighting.yaml`) |
|---|---|---|
| Targets | `albedo, material(roughness, metallicity)` | **`albedo, shading, residual`** |
| Space | albedo sRGB, material linear | all **linear**, shading/residual *up-to-scale* |
| Dataset | **InteriorVerse** | **Hypersim** |
| Latent | 2 targets → 8ch out | 3 targets → 12ch out |

Mechanics (`marigold_iid_trainer.py`, `marigold_iid_pipeline.py`):
- `conv_in` widened to `(n_targets+1)·4` (image latent + all target latents),
  weights repeated and scaled `1/(n+1)`; `conv_out` to `n_targets·4`.
- **Multi-step ε/v diffusion**: add noise to the GT target latents, U-Net predicts
  noise, **loss = plain `mse_loss` on the latent** (`config.loss.name: mse_loss`),
  optional multi-res noise. **No image-space loss. No reconstruction. No physics.**
- Decode each modality's 4-ch latent **independently** through the VAE; clip to
  `[-1,1]`. Inference = 4 DDIM steps, `rescale_betas_zero_snr=True`,
  `timestep_spacing="trailing"`.

So the **Lighting model is literally `I → (A, S, R)` as three independent
generative channels** with no constraint tying them back to `I`. R is spent as a
full 4-ch diffusion target yet `A·S+R` is never required to equal `I`.

**Takeaways we exploit:**
1. The latent-concat multi-target recipe works and is proven — keep the *interface*.
2. Its weaknesses are precisely our opening: (a) no reconstruction/physics, (b) R
   wasted as a generated channel, (c) two models, (d) multi-step.

---

## 2. The one-step lesson (Lotus / GenPercept / E2E-FT)

A line of 2024–25 work — **GenPercept**, **Lotus**, and *"Fine-tuning
image-conditional diffusion models is easier than you think"* (E2E-FT) — found
that for image-conditional dense prediction:

- The multi-step diffusion formulation was **partly a liability** (a timestep bug
  made multi-step *worse*); a **single fixed-timestep, x0-prediction** fine-tune
  of the *same* SD U-Net matches or beats multi-step and is **~10–50× faster**.
- Trained **end-to-end with the task loss directly on the decoded output**
  (e.g. affine-invariant / scale-shift-invariant depth loss), not noise MSE.
- The SD prior survives this deterministic fine-tune (web-scale generalization
  retained), and inference is **one forward pass**.

**Why this matters for IID specifically:** one step ⇒ the graph is *one* U-Net +
one VAE decode, so we can **afford to decode to image space and apply the
project's physics losses every iteration** — the thing Marigold-IID structurally
cannot do.

---

## 3. V18 design — three novelties

**(N1) One-step x0 feed-forward, not multi-step ε.** SD U-Net as a deterministic
predictor (prediction_type `sample`, fixed terminal timestep, zero-terminal-SNR).
One forward at inference.

**(N2) Physics-in-the-loop, image-space losses — the core contribution.** Because
it is one-step, decode the predicted x0 to `A, S_d` every step and apply the
project's physics losses in image space, **adapted to the analytic-residual split**
(§5): albedo `MSE+MSG+DSSIM` · `SSI shading` · a **two-sided diffuse reconstruction**
`A·S_d ≈ A_gt·S_d_gt`. (V17's residual-supervision, residual-sparsity, full-`I`
reconstruction and gradient-decorrelation are *dropped* — redundant, one-sided, or
harmful once R is analytic.) No diffusion-IID method (Marigold-IID, RGB↔X,
IntrinsicAnything) trains with reconstruction + cross-task physics in image space.
This grafts the project's hard-won inductive biases onto a web-scale generative prior.

**(N3) Colorful-diffuse split with analytic residual + HDR-safe π.** Generate
only **`[albedo | π]`** (8ch), where `π = 1/(S_d+1) ∈ [0,1]` puts HDR diffuse
shading inside the LDR SD VAE. Derive `S_d=(1−π)/π` and
`R=(I−A·S_d).clamp(min=0)` analytically ⇒ **physics exact** and **R costs zero
generative capacity** (Marigold spends a 4-ch target on R *and still* doesn't
reconstruct). One **unified** model on clean synthetic (Hypersim + InteriorVerse),
not two (§9).

**Bonus — dual mode.** The same U-Net remains a valid diffusion model: run it
**one-step deterministic** (primary: fast, physics-grounded, for metrics/
deployment) or **few-step stochastic** to sample a *distribution* of albedo in
ambiguous regions (uncertainty), Lotus-style discriminative/generative switch.

---

## 4. Formulation

**Targets (fixed canonical scale — no scale-match).** From `compute_diffusion_targets`:
```
A_gt = albedo_raw                              # dataset reflectance, [0,1]
S_d  = diffuse-illum GT (Hypersim, M_diffuse=1; A_gt·illum = I_diffuse)
       else colorful shading I / A             # R≈0 fallback (no diffuse GT)
π_gt = 1/(S_d+1)  ∈ [π_floor, 1]
```

**One-step prediction (exact recipe, validated against `diffusion-e2e-ft/training/train.py`).**
```
z_I    = VAE.encode(I)                          # condition (frozen)
z_T    = zeros_like(z_A)                         # E2E-FT default ("zeros"); fully deterministic
t      = T-1 = 999                               # single fixed terminal timestep
v_hat  = UNet( [z_I | z_T], t, ∅ )              # v-prediction (8ch → [v_A | v_π])
x0_hat = √ᾱ_t · z_T − √(1−ᾱ_t) · v_hat          # v→x0 ; at t=999, ᾱ_t=0 ⇒ x0 = −v_hat
A   = VAE.decode(z_A, gamma=True).clamp(0,1)
π   = VAE.decode(z_π, gamma=False).clamp(π_floor, 1−ε)
S_d = (1−π)/π ;   R = (I − A·S_d).clamp(min=0)  #  ⇒  I = A·S_d + R  exact
```
**Why these exact settings (this is the part SD1.5 gets wrong):** the one-step trick
needs the scheduler at `rescale_betas_zero_snr=True` + `timestep_spacing="trailing"`,
and **v-prediction**. At `t=999` with zero terminal SNR, `ᾱ_t=0`; the ε→x0 formula
(`x0=(z_T−√(1−ᾱ)ε)/√ᾱ`) **divides by zero**, while v→x0 stays well-defined
(`x0=−v`). SD-1.5 is ε-native; **SD-2.1 is v-native** → V18 should use **SD-2.1**
(`cross_attention_dim=1024`) to match the proven E2E-FT/Marigold recipe. (Alt: keep
SD-1.5 with Lotus's `t=1` where `ᾱ≈1`, but SD-2.1+v is the documented path.)
Feeding zeros (not random noise) makes the map deterministic — Lotus even drops the
noise channels entirely (`unet_input = rgb_latent`).

---

## 5. Loss (image space — the novelty)

`L = L_physics(A, S_d, I) + λ_lat · ‖x0_hat − [z_A_gt | z_π_gt]‖²`

**`L_physics` = `V17Loss` *adapted* to the 2-output analytic-residual model — not
reused verbatim.** V17 has three *independent* decoder outputs (A, S, R), so its
loss supervises each. V18 generates only `[A | π]` and derives `R=(I−A·S_d)₊`
analytically, which makes three of V17's terms inert, redundant, or harmful. The
fix is to supervise the quantity the model actually controls — the **diffuse
product `A·S_d`** — not the analytic identity `A·S_d+R`.

**Kept (the physics-in-the-loop signal):**
- **Albedo** MSE + MSG + DSSIM — primary, both datasets (albedo GT always present).
- **SSI shading** (π-domain, Hypersim-gated) — the only source with diffuse-shading GT.
- **Diffuse reconstruction** `L1(A·S_d, A_gt·S_d_gt) + MSG`, saturation-guarded
  (`recon_mode: diffuse`). Two-sided and meaningful on both sets: Hypersim →
  `A_gt·S_d_gt` is the true diffuse render; InteriorVerse → `S_d=I/A_gt` ⇒ target `=I`,
  so it becomes a genuine full reconstruction. This is the coupling term.
- **`λ_lat` latent anchor** (small): Marigold-style latent MSE on the x0 estimate;
  stabilizes early training, keeps the SD prior on-manifold. Optional/annealed.

**Dropped (and *why* — this is the correction over the first draft):**

| Term | Why it fails under analytic R |
|---|---|
| Residual `L1(R,R_gt)` | `R=(I−A·S_d)₊` is a deterministic fn of `(A,S_d,I)`, **not** an independent output. `L1(R,R_gt)` only re-expresses "`A·S_d≈A_gt·S_d_gt`" → fully redundant with albedo+shading. |
| Full-`I` reconstruction | `recon_pred = A·S_d + (I−A·S_d)₊ = max(A·S_d, I)` ⇒ loss `= relu(A·S_d−I)`: **one-sided**, zero gradient in the normal case `A·S_d≤I`. Replaced by the two-sided diffuse recon above. |
| Residual sparsity `‖R‖₁` | pushes `A·S_d→I` at **every** pixel incl. speculars → forces the diffuse term to swallow specular highlights (**leakage**). Harmful. |
| Gradient-decorrelation (Retinex) | logged ≈0 in V17 (never bit) and **no modern supervised/diffusion IID SOTA uses it** (see §5.1). Albedo GT + SSI + diffuse-recon subsume it; GT-free, so it could only matter on unsupervised data, which V18 has none of. |

- **Multi-illumination invariance** — available but **off** by default (MID excluded,
  §9). Re-enable if a paired real set is added.
- **Detail preserver (Lotus):** Lotus adds an RGB→RGB self-reconstruction to retain
  high-frequency detail in one-step models. Our diffuse-reconstruction term already
  serves this (it reconstructs the diffuse image from the decomposition), so no
  extra head is needed.

**Feasibility validated** against `diffusion-e2e-ft` (decode x0 → image-space loss →
backprop through the frozen VAE is exactly their training loop), Lotus-D (one-step
latent prediction + detail preserver), GenPercept (same SSI loss family). The novelty
is unchanged — physics-in-the-loop on an SD prior — but the loss is now *correctly*
adapted to the analytic colorful-diffuse split rather than ported wholesale from V17.

### 5.1 Has IID used a gradient-decorrelation / edge-exclusion loss before?

Yes — the *family* has a clear lineage, which is exactly why dropping it is informed,
not careless:
- **Retinex (Land & McCann '71), SIRFS (Barron–Malik), Intrinsic Images in the Wild
  (Bell '14), CGIntrinsics (Li–Snavely '18)** — reflectance-sparse / shading-smooth
  gradient priors; the classical ancestor of "edges belong to one layer."
- **Exclusion loss (Zhang et al., *Single Image Reflection Separation*, CVPR '18)** —
  the direct form: `‖tanh(λ_T|∇T|) ⊙ tanh(λ_R|∇R|)‖`, minimizing **co-located edges**
  between two layers. Our `mean(|∇A|·|∇S|)` is the same idea for the A/S split.
- Used mostly in **layer/reflection separation and *unsupervised* IID** (Liu et al.
  CVPR '20), where GT is absent.

**But modern strongly-supervised / diffusion IID dropped it:** CD-IID (Careaga–Aksoy),
CRefNet, and **Marigold-IID** use direct GT + reconstruction (and ordinal/multi-stage
shading) and carry **no** gradient-decorrelation term — supervision provides stronger,
less-fragile edge ownership. V18 (full albedo GT on both sets) is squarely in that
regime, so it follows SOTA and omits it.

---

## 6. Architecture

```
   I ──VAE.encode──▶ z_I (4ch) ─┐
   noise z_T (8ch) ─────────────┤  concat → [12ch]
                                ▼
              ┌──────── SD-1.5 U-Net (fine-tuned, x0) ────────┐
              │ conv_in 8+4=12→320 ; conv_out 320→8           │
              │ single fixed timestep t=T ; null cross-attn   │
              └───────────────┬───────────────────────────────┘
                              ▼ x0 = [z_A | z_π]
  A=dec(z_A)  π=dec(z_π)→S_d=(1−π)/π  → V17Loss(A,S_d): albedo + SSI + diffuse-recon
                              R=(I−A·S_d)₊  analytic, derived for output (not supervised)
```
- **VAE** frozen SD-1.5; deterministic `mode()` encode; gamma per-modality
  (albedo on, π off). Backprop flows through the frozen decoder into the U-Net
  (cheap at one step).
- **U-Net** SD-1.5; conv_in 4→12 / conv_out 4→8, pretrained weights copied into
  each modality slot (warm start). prediction_type `sample`; scheduler with
  `rescale_betas_zero_snr=True`, `timestep_spacing="trailing"`.
- **Conditioning** image-latent concat only; learned null `(1,77,768)` embedding.

---

## 7. Training & 8. Inference

**Train** (`src/train_v18.py`): one-step forward → decode → `V17Loss`
(`recon_mode: diffuse`) + small `λ_lat` anchor; AdamW, grad-checkpointing, fp16.
Two-phase curriculum, clean synthetic only: P1 Hypersim-only → P2
Hypersim+InteriorVerse (MID dropped, §9).

**Infer:** one forward → `{A, S_d, R}` with `I=A·S_d+R` exact (≈20–50× faster than
4-step Marigold). Optional **few-step stochastic** sampling for albedo uncertainty.

---

## 9. Datasets & per-dataset reconstruction (decided)

**What every diffusion / one-step SOTA does:** train on **clean SYNTHETIC only** —
E2E-FT & Lotus use Hypersim + Virtual-KITTI (90/10); Marigold-IID uses Hypersim
(lighting) *or* InteriorVerse (appearance), one synthetic set each. **None use noisy
real data** (the SD prior provides wild generalization, so the model needs little
data, and noisy pseudo-GT *poisons* the prior).

**V18 decision — Hypersim primary, InteriorVerse for albedo, DROP MID.**
- **Hypersim** — full `A, S_d(illum)` linear GT → drives albedo + shading SSI +
  diffuse reconstruction (`A·S_d` vs the true diffuse render `A_gt·S_d_gt`). The only
  source with true diffuse-shading GT. (R falls out analytically; not supervised.)
- **InteriorVerse** — clean albedo GT (the renders are slightly noisy with a blown
  window, but the *albedo* is excellent). Use for **albedo supervision**; derive
  `S=I/A` for the shading channel (no diffuse GT → treat as R≈0).
- **MID** — **excluded from V18.** Its 3-EXR input is heavily clipped/multi-illuminant
  and the albedo is *pseudo*-GT; feeding it to a generative prior degrades it. MID's
  only unique signal (multi-illumination invariance) is optional and not worth the
  noise. This is exactly the SOTA choice and answers *"can we use only
  Hypersim/InteriorVerse like other diffusion models" → yes, we should.*

**Reconstruction loss `L1(A·S_d+R, I)` — keep only where `I` is trustworthy:**

| Dataset | Input quality | Albedo GT | Recon in **V17** (CNN) | Recon in **V18** (diffusion) |
|---|---|---|---|---|
| Hypersim | clean synthetic | true `A,S,R` | **keep** (physics anchor) | **keep** (the novelty) |
| InteriorVerse | render noise + blown window | clean/pretty | **keep** w/ saturation guard, secondary to albedo loss | keep w/ guard (albedo is primary signal) |
| MID | heavily clipped, multi-illum | pseudo | **down-weight ≈0.1 or gate off**; lean on albedo + invariance | **N/A (excluded)** |

**Principle:** recon is only as reliable as `I`. The saturation guard
(`sat_ok = rgb.amax<0.99`, already in `V17Loss`) masks clipped highlights, but on MID
the *whole* input is unreliable, so trust the clean albedo GT (direct supervision)
and minimise/skip recon. Where the albedo GT is "pretty" and the input is noisy, the
albedo loss should dominate and recon should be down-weighted — never the reverse.

---

## 10. Novelty vs prior work

| Axis | Marigold-IID (Lighting) | Lotus / GenPercept / E2E-FT | CD-IID (Careaga) | **V18** |
|---|---|---|---|---|
| Backbone | SD diffusion | SD one-step | regression U-Net cascade | **SD one-step** |
| Steps | 4 (ε/v) | 1 | 1 (feed-fwd) | **1 (+ optional stochastic)** |
| Loss | latent MSE | task loss (1 modality) | image regression | **image-space physics suite (multi-task)** |
| Reconstruction `I=A·S+R` | ✗ | n/a | partial | **✓ exact (analytic R)** |
| Residual | generated 4-ch target | n/a | separate | **analytic, 0 capacity** |
| HDR shading | linear up-to-scale | n/a | multi-stage | **π=1/(S+1) in LDR VAE** |
| Cross-task coupling | ✗ (per-modality) | ✗ (single task) | via cascade | **✓ two-sided diffuse-recon `A·S_d`** |
| #models | 2 (appearance+lighting) | 1 | cascade | **1 unified** |

**One-line:** *V18 = a physics-grounded one-step colorful-diffuse intrinsic
diffusion — an SD prior fine-tuned in a single step with albedo + SSI-shading + a
two-sided diffuse-reconstruction (`A·S_d`) loss in image space, generating
`[albedo | π]` and closing `I=A·S_d+R` analytically.*

---

## 11. 24 GB-VRAM plan

One-step training = 1 U-Net fwd/bwd + 2 VAE decodes (frozen, but activations) +
`V17Loss`. No multi-step unroll → cheaper than it sounds.

| Item | VRAM |
|---|---|
| fp32 master + fp16 U-Net params | ~5.1 GB |
| AdamW states | ~6.9 GB |
| Grads (fp16) | ~1.7 GB |
| Activations + VAE-decode graph (bs1–2, **grad-checkpoint**) | ~4–6 GB |
| Frozen VAE | ~1 GB |
| **Total** | **~19–21 GB ✓** |

Defaults: `batch_size 2`, `grad_accum 8`, `grad_checkpointing true`. Headroom:
**8-bit AdamW** (−5 GB) or **LoRA** (−10 GB). **SD-1.5, not SDXL.** Inference is
trivial on 24 GB (one pass, no grads). Lighter than CD-IID's U-Net cascade.

---

## 12. What changes from the current code

The current `src/models/v18_pgid.py` / `train_v18.py` implement **multi-step
v-prediction with latent MSE** (the Marigold clone). To realize this design:

1. **Backbone** → switch `sd_pretrained` to **`stabilityai/stable-diffusion-2-1`**,
   `cross_attn_dim: 1024` (v-native; required for the zero-SNR one-step trick).
2. **Scheduler** → keep `prediction_type="v_prediction"`, add
   `rescale_betas_zero_snr=True`, `timestep_spacing="trailing"`.
3. **`forward`** → single fixed `t=999`; feed `z_T=zeros`; predict v; convert
   `x0 = √ᾱ·z_T − √(1−ᾱ)·v` (= `−v` at t=999); **decode** `z_A,z_π` → `A,S_d,R`.
4. **Loss** → swap `V18Loss` (latent v-MSE) for **`V17Loss` on decoded `A,S_d`** with
   `recon_mode: diffuse` (two-sided `A·S_d≈A_gt·S_d_gt`); **drop** R-supervision,
   residual-sparsity, full-`I` recon and gradient-decorrelation (§5); small `λ_lat`
   latent anchor. (`lambda_r=lambda_res_sparse=lambda_grad_decorr=0` in `v18.yaml`.)
5. **`sample`** → one-step path (encode I, one U-Net call, v→x0, decode) as default;
   keep the DDIM loop as the optional stochastic mode.
6. **Trainer** → reuse `compute_diffusion_targets`; backprop through VAE decode; keep
   grad-checkpointing; **drop MID** from the mix (Hypersim + InteriorVerse only).

Reused unchanged: `VAEWrapper`, `IntrinsicLatentDenoiser` (conv surgery),
`compute_diffusion_targets`, `V17Loss`, datasets / `use_mid_paired`, metric/viz.

---

## 13. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Backprop through VAE decode unstable / heavy | one step only; grad-checkpoint; `λ_lat` anchor keeps latents on-manifold |
| One-step over-smooths detail | MSG + DSSIM already penalise it; optional Lotus "detail preserver" on `I`'s high-freq |
| Losing the SD prior in deterministic FT | E2E-FT shows it survives; option LoRA; small LR + warmup |
| π VAE round-trip on flat shading | `shading_gamma` flag; light VAE finetune if visible |
| No reconstruction-maskable latent | losses are in **image space** (maskable); only `λ_lat` is latent (small) |

---

## 14. Open decisions

1. **x0 vs v at one step** — start x0 (`sample`); v as fallback.
2. **`λ_lat`** schedule — start ~0.1, anneal toward 0 as physics losses take over.
3. **Full-FT vs LoRA vs 8-bit AdamW** — start full-FT; switch for VRAM/iteration speed.
4. **Detail preserver / CFG** — add if albedo under-commits or detail softens.
5. **Stochastic mode** — auxiliary ε/v-consistency term to keep multi-step sampling
   valid for uncertainty maps (defer to v2).
6. **VAE light-finetune on albedo+π** — only if round-trip error shows in flats.

---

## 15. Summary

V18 keeps Marigold's proven latent-concat interface but rejects its core choices:
instead of 4-step ε-diffusion with latent-MSE and two models, it is a **one-step
x0 fine-tune of a single SD U-Net**, generating **`[albedo | π]`** and closing
`I=A·S_d+R` analytically, **trained end-to-end with the project's physics-loss
suite in image space**. The one-step design is what makes those physics losses
affordable — and physics-in-the-loop on an SD prior for the colorful-diffuse split
is the novel contribution, distinct from Marigold-IID, Lotus/GenPercept/E2E-FT,
and CD-IID. Fits 24 GB; ~10–50× faster inference than multi-step.

## 16. Some papers can improve the pixel accuracy
https://github.com/vivoCameraResearch/AdaRefSR
https://github.com/Adam-duan/DiT4SR
https://github.com/kunncheng/DiT-SR
https://github.com/lose4578/SAM-DiffSR

