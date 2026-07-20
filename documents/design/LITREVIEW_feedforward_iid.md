# Feed-Forward (Non-Diffusion) Intrinsic Decomposition on Hypersim

Research synthesis (2026-06-17) for the V20 design decision. Grounded in the actual
code of the methods cloned under `documents/references/` plus the IDT paper. Question:
how do the strong non-diffusion methods (CD-IID, IDT, CRefNet, PIE-Net) get **sharp,
accurate albedo (high LMSE) on Hypersim** without a generative/diffusion decoder?

---

## TL;DR — two camps, one physical model

| | **DERIVE camp** | **REGRESS camp** |
|---|---|---|
| Method | CD-IID (Careaga & Aksoy '23/'24) | IDT ('25), CRefNet ('24), PIE-Net |
| Albedo | **A = I / S** (division) | **A = head(features)** (direct) |
| Sharpness from | arithmetic (÷ smooth S keeps I's detail) | L1 loss + gradient/edge/DSSIM + capacity |
| Backbone | MiDaS CNNs, multi-stage | Transformer (VGGT / Swin-V2) |
| Views | single | IDT multi-view; CRefNet/PIE single |

**Convergent physical model (all of them):**

> `I = A ⊙ S_diff + S_spec`  — view-invariant albedo × diffuse (Lambertian) shading + specular/non-Lambertian residual.

This is exactly the model V20 should target. IDT Eq.(2) and CD-IID's final stage
(`dif_img = alb·dif_shd; residual = img − dif_img`) are the same equation.

---

## CD-IID (Careaga & Aksoy, TOG 2024) — the DERIVE cascade

From `references/Intrinsic CD-IID/intrinsic/pipeline.py` (actual code):

1. **Ordinal shading (stage 0)** — a MiDaS net predicts *scale-invariant* (ordinal)
   grayscale shading at **base AND full resolution**, then `equalize_predictions`
   least-squares-aligns them. Scale-invariance = robustness/generalization.
2. **Gray shading (stage 1)** — `iid_model` (MidasNet_small, in=5ch [rgb+ord_base+ord_full],
   out=1ch) → `inv_shd`; `shd = uninvert(inv_shd)`; **`alb = img / shd`** (DERIVE).
3. **Chroma (stage 2)** — `col_model` predicts shading chroma `uv_shd` (2ch) at **BASE
   resolution** (chroma is low-frequency), upsampled and combined with the full-res gray
   shading → `rough_shd`; **`rough_alb = img / rough_shd`** (DERIVE), Q99-normalized.
4. **High-res albedo (stage 3)** — `alb_model` (in=9ch [img, invert(rough_shd), rough_alb],
   out=3ch) regresses the final albedo **with `last_residual=True`** in v2.1: i.e.
   `pred_alb = rough_alb + correction`. A **residual refinement of the DERIVED albedo** —
   cleans up without re-blurring (it starts from the sharp derived layer).
5. **Diffuse shading (stage 4)** — `dif_model` predicts `dif_shd`; `residual = img − alb·dif_shd`,
   split into `pos_res` (specular/lights) and `neg_res` (saturated). The colorful
   decomposition (1–4) lets them train the diffuse split on **one synthetic dataset**.

**Key techniques:** derive albedo (sharpness is arithmetic); multi-scale base+full;
**scale separation** (gray high-res = sharp shadows, chroma low-res = smooth color);
ordinal (scale-invariant) shading; **residual refinement** instead of from-scratch
regression; explicit diffuse+specular residual split.

## IDT (Du et al., arXiv 2512.23667, Dec 2025) — the REGRESS transformer

- **Physical model Eq.(2):** `I_v = A ⊙ S_diff_v + S_spec_v`, A view-invariant.
- **Albedo is REGRESSED** (`A = h_alb(Z_alb)`, Eq.6) — no division. Sharpness comes from
  the **L1 albedo loss** (Eq.9 `‖A − A*‖₁`, "L1 for sharp boundaries") + transformer
  capacity + multi-view consistency.
- **Architecture:** VGGT-style multi-view transformer encoder → shared tokens → per-factor
  cross-attention "appearance adapters" (albedo / diffuse / specular); illumination as a
  Spherical Gaussian Mixture. Multi-view consistency via shared-albedo + scene-conditioned
  attention (single feed-forward pass over V views).
- **Losses:** `L_alb=‖A−A*‖₁`; `L_diff=‖log S_diff − log S_diff*‖²` (log-L2);
  `L_spec` log-L2; `L_recon=‖A·S_diff+S_spec − I‖₁`; `L_illum`.
- **Data:** **Hypersim primary** (uses diffuse_reflectance, diffuse_illumination,
  non-Lambertian residual — same channels we use) + InteriorVerse. Code not public yet.

## Supporting regression methods (local)

- **CRefNet** (TVCG '24): Swin-V2 transformer, decoder-sharing; fights blur with
  **DSSIM + L1 image-gradient loss + a fixed edge-conv loss** (`solver/loss.py`).
- **PIE-Net**: **edge-guided** decoder (`EdgeGuidedNetwork`) — predicts an edge map to keep
  albedo discontinuities sharp.

---

## How the field handles Hypersim specifically

- Albedo GT = `diffuse_reflectance`; diffuse shading GT = `diffuse_illumination`;
  non-Lambertian = `residual` (the channels our loader already reads).
- The **residual is large** (we measured 26–37% energy). Both camps model it explicitly:
  CD-IID `pos_res/neg_res`, IDT `S_spec`. Nobody folds it into shading.
- Shading is supervised **scale-invariantly**: CD-IID ordinal (relative), IDT log-space L2.
  (Our SSI in π-domain is the same family.)
- Sharpness lever, ranked by leverage: **derive (÷)** > **L1 albedo loss** (not L2/MSE) >
  multi-scale gradient + DSSIM + edge losses > scale separation (chroma low-res) >
  residual refinement > transformer/multi-view capacity.

---

## Implications for V20 (the decision)

1. **Physical model `I = A·S_diff + S_spec` is the field consensus** — matches the
   physical-inverse derive proposal (`A = (I−R)/S_d`).
2. **Derive is the strongest sharpness mechanism** (CD-IID). Our derive direction is the
   proven recipe, not a gamble.
3. **The "decoder_c" should return — but as a RESIDUAL refinement** of the derived albedo
   (CD-IID `last_residual`), never a from-scratch regression (that is exactly the V12/V16
   re-blur). This resolves the earlier removal: derive (sharp base) → `+correction` (clean).
4. **Switch albedo MSE → L1** (IDT + CD-IID + CRefNet all use L1/sharp losses; our V17Loss
   uses `_masked_mse` for albedo). L1 alone measurably reduces mean-blur.
5. **Scale separation is free LMSE**: predict shading **chroma at low resolution**, gray
   shading at full resolution (CD-IID). Cheap to adopt on our trunk.
6. **Keep the explicit residual/specular** (we already have analytic `R=(I−A·S_d)₊` = pos_res).
7. **Single-view caveat**: IDT's headline gains are multi-view consistency, which our
   single-view benchmarks (IIW/ARAP/MAW/MID) don't exercise — borrow its model + L1 losses,
   not its multi-view machinery.
