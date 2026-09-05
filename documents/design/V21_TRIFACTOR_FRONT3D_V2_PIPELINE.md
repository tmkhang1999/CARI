# V21 TriFactor-IID: Three-Factor Decomposition, 3D-Front v2, and Detail Recovery

Date: 2026-07-20

Status: design only. This is a new model family and must not change V17, its
configs, checkpoints, or thesis ablation rows.

## 0. Decision

Build the next pipeline as three independently testable components:

1. A deterministic, three-decoder base model that predicts scalar shading,
   shading chroma, and diffuse albedo.
2. A new Blender/Cycles 3D-Front corpus with real geometry-cast hard shadows,
   multi-colored illumination, and exact factor passes.
3. A separately trained, confidence-gated generative albedo restorer. It
   preserves observed detail where the image contains evidence and synthesizes
   only where deep shadow or clipping destroyed that evidence.

Do not simply add a third head to the current two-head model and continue the
old training recipe. Without a normalized factorization, the new chroma head
would give lighting another route into albedo. Without harder rendered
illumination, the third head would learn the same soft-shadow distribution that
already failed to transfer.

The recommended build order is:

```text
3D-Front v2 pilot and validation
        -> three-decoder deterministic base
        -> hard-light curriculum
        -> frozen-base generative restorer
        -> optional short joint tuning
```

## 1. Evidence Behind the Design

### 1.1 What to retain from V12

V12 made one important distinction that the current two-head model removed:

```text
scalar shading S_lum -> shading chroma C -> albedo A
```

This is the right task decomposition for colored illumination. Estimating
illumination chroma is easier than directly deciding whether every RGB change is
material or light, and CD-IID independently reports that shading chromaticity is
predominantly low frequency and is easier to estimate than albedo.

Do not retain the fragile parts of V12:

- Do not produce the final albedo as `I / S`.
- Do not use four large cascaded decoders.
- Do not require ground-truth normals or segmentation at inference.
- Do not feed unconstrained raw chromaticity directly to the albedo output.
- Do not use repeated detach operations as the main way to stabilize an
  otherwise unidentifiable cascade.

The new model may use a guarded `I / S` image as an internal feature, but the
albedo itself must be predicted directly and supervised directly.

### 1.2 What the first 3D-Front corpus taught us

The first corpus covered illuminant color but not the target shadow geometry:

- median illuminant-chroma gap: 0.151, comparable to or broader than ARAP;
- hard shading-edge fraction: 0.022, versus 0.115 for ARAP indoor and 0.186
  for ARAP outdoor;
- the large area-key lights had size `0.25-0.55 * room_radius`, with an even
  larger fill and a view-facing fill.

This explains the measured result: the data helped color-related metrics but
could not teach removal of hard cast shadows. The attempted image-space shadow
augmentation also failed its pre-registered gate because a few smooth analytic
blobs cannot reproduce the boundary density of shadows cast by complex scene
geometry.

The next dataset must change the rendered content, not merely its sampling
weight or learning rate.

### 1.3 What SOTA methods contribute

CD-IID solves a difficult decomposition as a sequence of simpler variables:
grayscale shading, shading chroma, albedo, then diffuse shading. Its albedo
network receives an initial colorful decomposition so its main task is removal
of remaining illumination artifacts. This validates factorized conditioning,
but its full four-network cascade is heavier than needed here.

Diffusion IID methods provide a strong learned prior and can generate sharp,
plausible material explanations instead of averaging ambiguous solutions.
DNF-Intrinsic further shows that starting from the source-image latent instead
of Gaussian noise better preserves the structure and appearance needed for
inverse rendering. These models cannot recover information that was never
recorded, however. In a clipped white patch or a channel-black deep shadow,
generated texture is a plausible completion, not a measured recovery.

The proposed system combines these lessons: deterministic factors for physical
accountability, then a generative prior only for the remaining ambiguous detail.

## 2. Identifiable Three-Factor Image Model

Use the intrinsic residual model in linear RGB:

```text
I = A_d * S_rgb + R
S_rgb = S_lum * C
Y(C) = 1
```

where:

- `A_d` is three-channel diffuse albedo;
- `S_lum` is a positive scalar shading field;
- `C` is a positive RGB shading-chroma field;
- `R = I - A_d * S_rgb` is a signed analytic residual.

The constraint `Y(C) = 1` is essential. It prevents the luminance and chroma
decoders from changing each other's scale. Parameterize chroma with two bounded
log ratios:

```text
u = log(C_R / C_G)
v = log(C_B / C_G)
C_raw = [exp(u), 1, exp(v)]
C = C_raw / (Y(C_raw) + eps)
```

Predict scalar shading in log space:

```text
g = log(S_lum + eps)
S_lum = exp(g)
```

This avoids the saturation and extreme gradients of the inverse-shading
parameterization. Clamp `g` only for numerical safety, not to force shading into
the display range. HDR shading must remain representable.

Use a signed residual. The current positive-only residual cannot represent
overexposure: when the estimated diffuse reconstruction contains information
that the captured image clipped, the residual is negative. CD-IID explicitly
uses this behavior for HDR-aware editing.

## 3. Base Architecture: Exactly Three Decoders

### 3.1 Shared features

Keep the proven V17 backbone shape as an initialization path:

```text
linear RGB
  |-> frozen DINOv2 semantic/content features F_inv
  |-> trainable photometric stem F_rgb (color and fine detail)
  `-> shared DPT fusion trunk F
```

The two streams have different roles:

- `F_inv` provides long-range material and scene context that is relatively
  stable under illumination changes.
- `F_rgb` preserves color and high-frequency image evidence discarded by the
  invariant backbone.

Do not let every decoder consume both streams without restriction. That would
recreate the current raw-RGB illumination leak.

### 3.2 Decoder 1: scalar shading

Inputs:

- shared semantic features `F`;
- a full-resolution log-luminance stem `log(Y(I) + eps)`;
- optional predicted normals from a frozen RGB-only estimator during training
  experiments, but never clean ground-truth normals as a required model input.

Outputs:

- multi-resolution `g = log(S_lum + eps)`;
- no color channels.

This decoder owns intensity gradients, attached shadows, cast shadows, and
large illumination ramps. It must be supervised at both low resolution for
global ordering and high resolution for narrow shadow boundaries.

### 3.3 Decoder 2: shading chroma

Inputs:

- shared semantic features `F`;
- color-preserving features `F_rgb`;
- stop-gradient scalar shading features from Decoder 1;
- a global illuminant token pooled from the image.

Outputs:

- a global two-channel log-chroma estimate;
- a low-resolution spatial residual in `u,v`;
- normalized `C` with `Y(C)=1`.

Use a global-plus-spatial parameterization instead of an unrestricted
full-resolution RGB map. Most illuminant chroma is low frequency, while the
spatial residual is still needed for mixed lights, colored interreflection, and
colored cast-shadow boundaries. Decode the residual at one-quarter or
one-eighth resolution and edge-aware upsample it using shading features, not raw
RGB edges.

### 3.4 Canonicalized detail feature

After Decoders 1 and 2, construct an internal, guarded de-lit feature:

```text
S_hat = stopgrad(S_lum * C)
J = I / max(S_hat, s_min)
M_obs = valid exposure and sufficient predicted illumination
J_guarded = M_obs * clamp(J) + (1 - M_obs) * neutral_fill
```

`J_guarded` is a feature, not the albedo output. Its purpose is to carry observed
material color and texture after the two lighting factors have been removed.
The mask prevents division noise in deep shadows and clipped channels from
entering the albedo decoder.

### 3.5 Decoder 3: direct diffuse albedo

Inputs:

- illumination-invariant semantic features `F_inv` and DPT context;
- `J_guarded` through a small high-resolution detail stem;
- stop-gradient `g`, `C`, and `M_obs`.

Output:

- direct three-channel `A_0` in linear RGB.

Do not feed unrestricted raw RGB directly to this decoder. Color reaches albedo
through the predicted de-lit feature. The output remains direct, so errors in
small shading values do not automatically create the exploding `I/S` artifacts
seen in V12.

### 3.6 Forward summary

```text
                          +-> D_lum    -> S_lum --+
I -> dual-stream encoder -+                        +-> S_rgb
                          +-> D_chroma -> C -------+
                                      |
                   guarded canonical I / S_rgb
                                      |
                                      +-> D_albedo -> A_0

R = I - A_0 * S_rgb
```

The base model is deterministic and feed-forward. It should remain independently
usable if the generative restorer fails its gates.

## 4. 3D-Front v2 Renderer

### 4.1 Renderer objective

The renderer must produce the joint condition missing from all current training
sources:

```text
colored illumination x hard, moving, geometry-consistent cast shadows
```

The old renderer's soft area lights are retained only as one reference mode.
Image-space shadow overlays are not used.

### 4.2 Cycles and color pipeline

Use Blender 4.5 LTS or the newest validated LTS available on the render machine,
with Cycles and GPU rendering. Version changes are secondary to the lighting and
pass changes, but an LTS build gives a stable Python API for a multi-day render.

Required settings:

- save all supervision as linear, scene-referred, 32-bit multilayer EXR;
- use AgX or sRGB only for PNG previews, never for training labels;
- disable direct-light clamping and keep indirect clamping high enough not to
  erase intended highlights;
- use at least 4 diffuse bounces and 2 glossy/transmission bounces;
- use adaptive sampling with a pilot-calibrated noise threshold;
- denoise preview RGB if useful, but do not denoise factor labels unless a
  reconstruction test proves that the pass arithmetic remains unbiased;
- use persistent data between same-camera lighting variants.

Cycles distinguishes diffuse, glossy, transmission, and shadow paths and exposes
their bounce controls. Blender's light model also makes the crucial renderer
change explicit: area lights create soft borders, while point/spot lights with a
small radius create sharp object shadows.

### 4.3 Lighting rigs

Render six same-camera variants per view. Sample the rig type with a fixed
stratified roster rather than six unconstrained random draws:

| Variant | Rig | Purpose |
|---|---|---|
| L0 | neutral large-area key and weak fill | recoverable reference and old-domain anchor |
| L1 | small-radius point or spot key | sharp indoor cast shadows |
| L2 | small-radius key from a substantially different azimuth | moving shadow correspondence |
| L3 | sun through a procedural window and mullion gobo | long, dense directional boundaries |
| L4 | warm hard key plus cool soft fill | mixed-color shadow and illuminant separation |
| L5 | low-key practical lights plus indirect bounce | deep shadow, local color, and residual effects |

Concrete geometric requirements:

- point/spot radius: `0.005-0.04 * room_radius`;
- spot blend: `0.0-0.15`, with randomized cone angle;
- sun angle: include `0.1-2.0` degrees, with random elevation and azimuth;
- key movement between L1 and L2: at least 45 degrees around the scene center;
- fill-to-key power ratio: sample `0.02-0.25`, not the old always-bright range;
- preserve a small fraction of soft old-style examples to prevent a new bias
  toward every edge being a shadow.

The current solid procedural room shell blocks meaningful outside sunlight.
Replace it with a shell that has one or two procedural window apertures, or use a
camera-invisible but shadow-visible window/gobo assembly inside the room. Window
mullions, blinds, and a small library of silhouette occluders create realistic
boundary complexity. Furniture geometry remains the primary occluder.

Do not use hundreds of random blobs. They can match an edge-density number while
teaching an implausible illumination prior.

### 4.4 Light color sampling

Keep the successful color range from v1, but make source roles explicit:

- 50% blackbody sources from 2200 K to 10,000 K;
- 30% moderate-saturation RGB sources with a per-channel energy floor;
- 20% mixed complementary or practical-light configurations;
- pairwise `rg` chromaticity gap target: `0.08-0.20` for at least two variants;
- randomize direction, intensity, size, and color independently enough that the
  network cannot identify a rig from brightness alone.

Keep one neutral anchor in every group. Fully monochromatic frames are invalid
because material color is not observable when one or more channels carry nearly
zero signal.

### 4.5 Exact outputs

For every view, export enough passes to verify and supervise the proposed image
model:

```text
combined RGB
diffuse color / base color A_gt
diffuse direct
diffuse indirect
glossy direct and indirect
transmission
emission
normal, depth, object/material ID
```

Derive and store:

```text
diffuse_rgb_gt
S_rgb_gt
S_lum_gt = Y(S_rgb_gt)
C_gt = S_rgb_gt / (S_lum_gt + eps)
R_gt = combined_rgb - diffuse_rgb_gt
observable/confidence mask
hard-shadow mask and boundary map
```

Blender pass conventions must be validated empirically before the full render.
For ten pilot frames, the selected diffuse passes and base color must reconstruct
`diffuse_rgb_gt` to less than 1e-3 relative error in valid pixels. Do not assume
whether a pass includes the material color without this test.

### 4.6 Dataset size and split

Recommended first full version:

```text
997 rooms x 4 views x 6 lighting variants = 23,928 rendered RGB images
3,988 aligned multi-light groups
```

Split by room or house before rendering. All views and lighting variants from one
room must remain in one split. Keep a frozen 50-room renderer-development split
that never enters training.

### 4.7 Pre-render gates

Render only 50 rooms first. Continue to the full corpus only if all gates pass:

1. Median hard-edge fraction for hard rigs falls inside the ARAP indoor interval
   `[0.07, 0.18]` and is at least 3 times the old 0.022 value.
2. At least 70% of L1/L2 pairs have materially different shadow masks, measured
   by shadow-mask IoU below 0.7.
3. Median illuminant-chroma gap remains at least 0.10.
4. At least 85% of frames retain valid signal in all three channels over 60% of
   diffuse foreground pixels.
5. Factor-pass reconstruction error is below 1e-3 relative error.
6. Human inspection of 100 random sheets finds no dominant failure from blocked
   rooms, floating lights, camera-inside-geometry, noisy tiny lights, or implausible
   gobo patterns.

These gates prevent another multi-day render whose training gradient is already
known to miss the target failure.

## 5. Losses for the Three-Decoder Base

### 5.1 Supervised factor losses

On sources with exact factors:

```text
L_A = log-L1 + multi-scale gradient + DSSIM + small Lab/DeltaE term
L_lum = scale/shift-invariant log-L1 + multi-scale gradient
L_chroma = L1(u,v) + angular RGB-chroma loss + low-resolution gradient loss
L_diffuse = robust L1(A_0 * S_rgb, diffuse_rgb_gt)
```

Use direct albedo supervision on clean synthetic data. The DeltaE term should be
small and must not replace linear-RGB accuracy.

### 5.2 Ordinal scalar shading

Generate ordinal pairs directly from `S_lum_gt` at multiple distances and
resolutions. Train Decoder 1 to predict whether point `p` is darker, equal, or
brighter than point `q`, with a confidence margin that ignores near-equal noisy
pairs.

This gives global ordering plus high-resolution shadow boundaries. It is more
targeted than applying an ordinal loss to the final albedo and follows the useful
part of Ordinal Shading.

### 5.3 Edge ownership

The renderer provides exact albedo and shading boundaries. Define:

- shadow edges: high `grad(S_lum or C)`, low `grad(A_gt)`;
- material edges: high `grad(A_gt)`;
- mixed edges: both high, excluded from the exclusive penalties.

Penalize albedo gradients on shadow-only edges and require albedo gradients on
material-only edges. This is safer than generic albedo flattening, which produced
good benchmark values by blurring legitimate texture.

### 5.4 Cross-render supervision

For same-camera variants `i,j`:

```text
L_A_inv = |align(A_i) - align(A_j)|
L_lum_explain = |Delta log Y(I_diffuse) - Delta log S_lum|
L_chr_explain = |Delta chroma(I_diffuse) - Delta chroma(C)|
```

Apply these only on pixels valid in both frames. Supervised absolute albedo
prevents the gray constant solution. Splitting the explain loss into scalar and
chroma terms directly supervises which of the first two decoders owns each
illumination change.

Retain raw MID pairs as real-image invariance supervision, but do not use MID as
the only source of absolute albedo color. Use the corrected Cast_rel and
Chroma_fid guards, not the old pooled Cast_RMS metric that rewarded gray output.

### 5.5 Residual treatment

Do not use total reconstruction `I = A*S + R` as a loss when `R` is defined
analytically, since it is an identity. Supervise the diffuse product against the
renderer diffuse pass and supervise `R` against the renderer non-diffuse pass.
On real data without `R_gt`, use pair consistency and weak priors rather than
forcing specularities into shading.

## 6. Generative Detail Restorer

### 6.1 Why a separate stage

The deterministic base should remove lighting and produce metrically stable
albedo. A generative prior has a different job: restore plausible material
structure after the light has been removed. Combining both jobs in one decoder
makes it difficult to know whether sharper output is genuine detail or lighting
copied back into albedo.

Train the restorer only after the three-head base passes its own benchmark gates.
Freeze the base and precompute its outputs to disk. This reduces 24 GB VRAM usage
and prevents the generative loss from destabilizing the physical factors.

### 6.2 Observability mask

Compute a per-channel confidence map from:

- input underexposure and saturation;
- predicted `S_lum` and `C`;
- disagreement between same-scene variants during synthetic training;
- renderer visibility and clipping labels when available.

Classify pixels as:

```text
observed: detail is present and may be transferred from the canonicalized image
uncertain: detail is noisy or partly color-clipped
unobserved: deep shadow, full clipping, or no diffuse signal
```

This mask controls both training and inference. It also provides an uncertainty
visualization for honest downstream use.

### 6.3 Hybrid restorer architecture

Use a deterministic, one-step source-image-to-albedo latent flow model for low-
and mid-frequency albedo, followed by a small pixel-space detail adapter:

```text
conditions = [input I, coarse A_0, S_lum, C, M_obs, semantic features]
latent prior -> A_prior
pixel adapter(J_guarded, A_prior, M_obs) -> bounded high-frequency delta
A_final = compose(A_prior, delta, M_obs)
```

Recommended implementation:

- follow DNF-Intrinsic's noise-free principle: initialize the flow from the
  source/canonical image latent, not Gaussian noise;
- initialize the latent prior from the existing SD/Marigold-compatible IID code
  or a compatible pretrained diffusion transformer;
- predict albedo only, not all intrinsic layers again;
- use one deterministic output for training, evaluation, and deployment;
- condition through lightweight adapters or LoRA before attempting full U-Net
  fine-tuning;
- use tiled latent inference plus an overlapping pixel adapter for long-edge
  resolution above 1280.

The pixel adapter is essential. A frozen VAE and latent network can soften small
textures. The adapter transfers input-aligned high frequencies only where
`M_obs` says the evidence is reliable. In unobserved regions, it cannot copy the
input and the latent prior supplies a plausible completion. Report `M_obs` as
the confidence output; do not imply pixel-level certainty from a deterministic
prediction in unobserved areas.

### 6.4 Restorer losses

Train against exact `A_gt` from 3D-Front v2, Hypersim, and InteriorVerse:

```text
L_ref_rgb       = robust linear-RGB and log-RGB loss
L_ref_perc      = low-weight LPIPS/DISTS-like perceptual loss
L_ref_grad      = multi-band gradient/Laplacian loss to A_gt
L_ref_inv       = same-scene albedo consistency across lighting variants
L_ref_identity  = preserve A_0 on already-clean, high-confidence pixels
L_ref_halluc    = penalize changes outside the unobserved/uncertain mask
L_ref_physics   = diffuse reconstruction using frozen S_rgb where valid
```

Never train the refiner mainly with `A_final * S ~= I`. That objective rewards
copying shadows and specularities back into albedo. The exact clean target and
cross-render invariance must dominate.

### 6.5 What "restore" means

- Texture visible in at least one channel is recoverable and should be preserved
  deterministically.
- Texture attenuated but above the sensor/noise floor can be enhanced with the
  clean-albedo prior and multi-light training.
- Texture lost through clipping or zero signal cannot be recovered uniquely from
  one image. The diffusion output is a context-consistent hypothesis.

The implementation and paper should call the last case plausible completion or
prior-based restoration, not exact recovery.

## 7. Training Curriculum

### Stage 0: renderer pilot

- Render 50 rooms with all factor passes.
- Run every gate in Section 4.7.
- Stop if the hard-shadow distribution still misses ARAP.

### Stage 1: initialize three decoders

- Load the current frozen DINO encoder and compatible DPT/detail trunk weights.
- Initialize the direct albedo decoder from the current albedo head where shapes
  match.
- Initialize scalar and chroma decoders separately.
- Freeze DINO and the albedo decoder for about 5k steps while scalar/chroma heads
  learn valid ranges.

### Stage 2: supervised joint base training

- Unfreeze DPT, photometric stem, and all three decoders.
- Keep DINO frozen initially.
- Suggested starting mix:
  Hypersim 0.30, InteriorVerse 0.20, 3D-Front v2 0.30, MID pairs 0.20.
- Train with exact factor losses and moderate cross-render losses.

### Stage 3: hard-light curriculum

- Increase hard 3D-Front pair sampling, not total 3D-Front epoch reuse.
- Ensure each synthetic pair contains one neutral/soft anchor and one hard or
  mixed-color variant.
- Ramp ordinal and edge-ownership losses after direct factor losses stabilize.
- Unfreeze only the final DINO blocks with LoRA if the model lacks real-image
  adaptation; do not fully fine-tune the backbone first.

### Stage 4: frozen-base restorer

- Precompute `A_0`, `S_lum`, `C`, `M_obs`, and semantic features.
- Train the latent prior and pixel adapter separately from the base.
- Use deterministic one-step image-to-albedo flow as the primary path.
- Consider extra flow steps only as an ablation. Do not use stochastic
  ensembling or sample selection for the reported benchmark result.

### Stage 5: optional short joint tuning

Proceed only if Stage 4 passes. Unfreeze the albedo decoder at 5-10 times lower
learning rate than the restorer for at most 5k steps. Keep both shading decoders
frozen so the restorer cannot move the decomposition boundary to improve its own
loss.

## 8. Required Ablations

The minimum causal table is:

| Row | Three factors | 3D-Front v2 hard rigs | Restorer | Confidence gate |
|---|---:|---:|---:|---:|
| A | no, current two-head base | no | no | no |
| B | yes | no, old soft corpus | no | no |
| C | yes | yes | no | no |
| D | yes | yes | deterministic pixel refiner | yes |
| E | yes | yes | latent prior plus pixel adapter | no |
| F | yes | yes | latent prior plus pixel adapter | yes |

This separates:

- whether the third decoder improves color ownership;
- whether geometry-correct hard shadows improve shadow removal;
- whether diffusion adds detail beyond a small deterministic adapter;
- whether confidence gating prevents illumination from being copied back.

Also report old soft 3D-Front versus v2 at equal groups and equal optimization
steps. Otherwise a gain could be attributed only to more data.

## 9. Evaluation and Kill Gates

### 9.1 Dataset-level gates

Use Section 4.7 before training.

### 9.2 Base-model metrics

Primary:

- ARAP standard white-balanced LMSE, scale-invariant RMSE, and SSIM;
- ARAP raw-color `C_arap`, Cast_rel, and the colored/indoor subsets;
- MAW DeltaE and intensity;
- MID `C_mat`, Cast_rel, and Chroma_fid;
- IIW WHDR as a guard, not the optimization target.

Add a pre-registered ARAP hard-shadow subset based on ground-truth shading-edge
density. The renderer hypothesis succeeds only if gains concentrate on the
strong-shadow scenes that currently fail.

### 9.3 Detail metrics

Report detail separately from decomposition accuracy:

- scale-aligned LPIPS or DISTS on ARAP/Hypersim albedo;
- Laplacian-band or high-frequency gradient error to albedo GT;
- SSIM on texture-bearing material regions;
- cross-light variance of the restored albedo;
- a contact sheet containing strong shadow, fine texture, saturated light, and
  failure cases selected by fixed scene IDs.

The restorer passes only if it improves perceptual/high-frequency metrics while
not worsening ARAP/MID constancy or MAW color by more than 5%. A sharper image
with higher cross-light variance is a failed refiner.

### 9.4 Go/no-go thresholds

Proceed from three-head base to restorer only if:

- raw-color ARAP colored-subset constancy improves at least 10% over the matched
  two-head baseline;
- MAW DeltaE and intensity regress less than 5%;
- Chroma_fid remains near 1 and does not improve only through desaturation;
- hard-shadow albedo leakage is visibly reduced on fixed ARAP scenes.

Promote the generative restorer only if:

- high-frequency albedo error improves at least 10%;
- ARAP and MID constancy regress less than 3%;
- MAW DeltaE regresses less than 5%;
- unobserved-region uncertainty is higher than observed-region uncertainty;
- no repeated texture, tiling, semantic replacement, or shadow copying appears
  in the fixed visual audit.

## 10. Resource Plan for Two 24 GB GPUs

GPU 0:

- render and validate 3D-Front v2 in batches;
- then train the deterministic base with paired batch size 2 and gradient
  accumulation.

GPU 1:

- run base-model validation and data probes while rendering;
- later train the restorer from precomputed conditions.

Practical constraints:

- never keep trainable base and diffusion U-Net in memory together during the
  first restorer experiment;
- use gradient checkpointing and mixed precision for the latent prior;
- cache base predictions as FP16 tensors with factor metadata;
- use one deterministic flow prediction for all quantitative evaluation;
- use auto-resume with atomic checkpoint writes because previous jobs were lost
  to host RAM pressure.

## 11. Main Risks

### Chroma still leaks into albedo

Cause: unrestricted RGB access or an unnormalized chroma field.

Mitigation: `Y(C)=1`, strict decoder input routing, direct chroma supervision,
and split luminance/chroma explain losses.

### Hard-shadow training removes material texture

Cause: a generic smooth-albedo prior cannot distinguish texture from shadow.

Mitigation: exact edge ownership from rendered passes and a texture/detail guard
against `A_gt`.

### The generative restorer hides errors with plausible texture

Cause: diffusion can improve appearance without preserving scene evidence.

Mitigation: confidence-gated composition, deterministic evaluation, uncertainty
maps, cross-render consistency, and separate detail versus physical metrics.

### 3D-Front remains out of domain

Cause: hard shadows alone do not fix asset realism, camera distribution, or
material quality.

Mitigation: keep Hypersim, InteriorVerse, and real MID pairs; validate transfer on
a hard-shadow holdout before full-scale training; expand renderer diversity only
after the 997-room version proves the mechanism.

### Three heads increase ambiguity rather than reduce it

Cause: the same RGB shading can be represented by different scalar/chroma scales.

Mitigation: enforce `Y(C)=1` analytically and supervise each factor from exact
render passes.

## 12. Recommendation

Implement the renderer pilot first, then the deterministic three-decoder base.
Do not start with diffusion. The existing failure is primarily missing
hard-shadow supervision and weak factor ownership, and a generative refiner
cannot repair either if the coarse decomposition is wrong.

If the three-head plus 3D-Front v2 row improves the hard-shadow subset, train the
confidence-gated restorer. If it does not, stop before diffusion: that result
would show that the remaining problem is real-domain supervision or the encoder,
not missing decoder capacity.

## 13. Primary References

- Careaga and Aksoy, "Colorful Diffuse Intrinsic Image Decomposition in the
  Wild," ACM TOG 2024: https://arxiv.org/abs/2409.13690
- Careaga and Aksoy, "Intrinsic Image Decomposition via Ordinal Shading," ACM
  TOG 2023: https://arxiv.org/abs/2311.12792
- Ke et al., "Marigold: Affordable Adaptation of Diffusion-Based Image Generators
  for Image Analysis," 2025: https://arxiv.org/abs/2505.09358
- Kocsis, Sitzmann, and Niessner, "Intrinsic Image Diffusion for Indoor
  Single-view Material Estimation," CVPR 2024:
  https://openaccess.thecvf.com/content/CVPR2024/html/Kocsis_Intrinsic_Image_Diffusion_for_Indoor_Single-view_Material_Estimation_CVPR_2024_paper.html
- Djeghim et al., "SAIL: Self-supervised Albedo Estimation from Real Images with
  a Latent Diffusion Model," 2025: https://arxiv.org/abs/2505.19751
- Zheng et al., "DNF-Intrinsic: Deterministic Noise-Free Diffusion for Indoor
  Inverse Rendering," ICCV 2025:
  https://openaccess.thecvf.com/content/ICCV2025/html/Zheng_DNF-Intrinsic_Deterministic_Noise-Free_Diffusion_for_Indoor_Inverse_Rendering_ICCV_2025_paper.html
- Dirik et al., "ReasonX: MLLM-Guided Intrinsic Image Decomposition," CVPR 2026:
  https://openaccess.thecvf.com/content/CVPR2026/html/Dirik_ReasonX_MLLM-Guided_Intrinsic_Image_Decomposition_CVPR_2026_paper.html
- Blender Manual, Light Objects:
  https://docs.blender.org/manual/en/4.3/render/lights/light_object.html
- Blender Manual, Cycles Light Paths:
  https://docs.blender.org/manual/en/4.5/render/cycles/render_settings/light_paths.html
- Blender Manual, Cycles Sampling:
  https://docs.blender.org/manual/en/4.0/render/cycles/render_settings/sampling.html

## 14. Implementation Boundaries

Create new modules rather than extending the already measured V17 code paths:

```text
scripts/render_3dfront_dataset_v2.py
scripts/validate_3dfront_v2.py
src/data/front3d_v2_dataset.py
src/models/v21_trifactor.py
src/models/v21_restorer.py
src/losses/v21_loss.py
src/configs/v21.yaml
src/configs/v21_restorer.yaml
src/train_v21.py
```

The v2 dataset loader should expose named tensors instead of overloading the old
`illum` field:

```text
rgb, albedo_gt, shading_lum_gt, shading_chroma_gt, diffuse_gt, residual_gt,
pair_rgb, pair_valid, observable_mask, shadow_edge, material_edge
```

Keep renderer metadata versioned in every sample. At minimum record Blender
version, Cycles device, scene/room ID, camera, seed, rig type, every light's
transform/color/power/size, path settings, sample count, and pass schema version.

The first implementation milestone is not a trained network. It is:

```text
50 rooms rendered
all six rigs visually audited
factor arithmetic verified
hard-edge and chroma gates passed
one fixed pilot manifest committed
```

Only after that milestone should `v21_trifactor.py` be implemented. The first
model experiment is Row B versus Row C from Section 8 at matched groups and
steps. If v2 hard lighting does not improve the pre-registered hard-shadow
subset, do not spend compute on the generative restorer.

## 15. Implemented Runbook (2026-07-20)

The isolated implementation is now available in the paths listed in Section
14. The albedo restorer is a deterministic noise-free image flow plus a bounded
pixel adapter; it does not require a new diffusion dependency. The tri-factor
base is trainable. The restorer loads a completed base checkpoint strictly,
sets every base parameter to `requires_grad=False`, and optimizes only the flow
and detail adapter.

Run the stages in this order.

### 15.1 Render the 50-room pilot

```bash
CUDA_VISIBLE_DEVICES=0 /home/khang/miniconda3/envs/IR/bin/python \
  scripts/render_3dfront_dataset_v2.py \
  --glb-root ../datasets/3D-Front-HF/scenes/3D-FRONT-TEST-SCENE \
  --out ../datasets/front3d_iid_v2 \
  --blender tools/blender/blender-4.2.0-linux-x64/blender \
  --views 3 --lightings 6 --resolution 512 --samples 64 \
  --device GPU --limit 50
```

### 15.2 Enforce the corpus gates

```bash
OPENCV_IO_ENABLE_OPENEXR=1 /home/khang/miniconda3/envs/IR/bin/python \
  scripts/validate_3dfront_v2.py \
  --root ../datasets/front3d_iid_v2 \
  --report tests/visualizations/front3d_v2_pilot/validation_v2.json
```

Do not add `--no-fail` for the real pilot. A non-zero exit status means the
corpus must not be used for a full training run.

### 15.3 Dry-run and train the tri-factor base

```bash
bash scripts/train.sh --version 21 --cuda 0 --dry-run
bash scripts/train.sh --version 21 --cuda 0 --auto-resume
```

### 15.4 Train the frozen-base restorer only after base benchmark gates pass

```bash
CUDA_VISIBLE_DEVICES=0 /home/khang/miniconda3/envs/IR/bin/python \
  src/train_v21_restorer.py --device cuda --auto-resume \
  --base-checkpoint checkpoints/v21/checkpoint_latest.pth
```

### 15.5 Implementation smoke result

One deliberately sparse local room was rendered through all six rigs at 512
pixels and 16 Cycles samples. All files, material overrides, factor arithmetic,
loader paths, exposure checks, metadata, chroma, and moving-shadow checks
worked. Its hard-edge fraction was `0.034`, compared with `0.022` for the old
corpus, but below the pre-registered `0.070` pilot gate. This is not a reason to
lower the gate: a representative 50-room pilot is still required before any
base or restorer compute is committed.
