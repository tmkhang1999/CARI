# Development history

Internal record of how the project reached its current form: what was tried, what was
measured, and why each direction was kept or dropped. It replaces the design documents,
evaluation notes and thesis working notes that used to live in `documents/design/`,
`documents/evals/*.md` and `documents/thesis/*.md`. Their full text is in git history;
the last commit containing all of them is `c4162a8` (2026-10-08). Result files that the
report still cites are in `documents/results/`.

Dates are 2026.

---

## 1. Starting point: a retouching pipeline (March - May)

The project began as "multi-stage intrinsic image decomposition and retouching" for
real-estate photographs. The early models (V1 to V16) were ConvNeXt-V2 encoders with
extra inputs: surface normals from Metric3D, NYU-40 segmentation from Mask2Former, and a
cross-colour-ratio (CCR) image, fused through SPADE/FiLM blocks. Albedo was *derived* as
A = I / S and then refined by a cascade. The shading was moved to the inverse domain
pi = 1/(S+1) in March to bound the output (this survives in V17).

Why it was dropped: the extra inputs were not available reliably on real photographs, the
division A = I / S exploded where shading was small (dark blobs), and the hand-built
priors leaked lighting into albedo anyway. (Design notes: `project_plan.md`,
`data_processing_supplement.md`.)

## 2. Three paradigms compared (June 9-10)

A code-grounded review compared three backbones built in the project:

| | V12 (CNN) | V17 (transformer) | V18 (diffusion) |
|---|---|---|---|
| Backbone | ConvNeXt-V2 + priors | frozen DINOv2-L + DPT | Stable Diffusion U-Net, one step |
| Albedo | derived I/S + cascade | direct head | direct, VAE-decoded |
| Main failure | fragile division, leaks | desaturated albedo | pixel accuracy, cast splitting |

A two-part thesis plan around V18 plus an image-enhancement application was written and
abandoned within days in favour of the idea that became the project:

**V19 / CARI (June 10).** Import the disambiguation that multi-view or multi-light capture
gives, at training time only: two pixel-aligned photographs of one MID scene under
different light, one shared network, a loss that ties the two albedos and a loss that makes
the shading explain the change. Built on the V17 skeleton (frozen DINOv2-L + DPT, direct
albedo, analytic residual R = (I - A*S_d)+, three-channel shading). This is the model and
strategy the report describes, renamed CIAI in October.

## 3. Making V17 + CARI work (June 12 - July 9)

- **Evaluation harness (June 12-19).** MID constancy (C_mat, a pooled chroma variance
  Cast_RMS), ARAP constancy and accuracy, MAW, IIW. A double-gamma bug had been
  desaturating Marigold's albedo and flattering its constancy; after the fix, V17 and
  Marigold split the metrics roughly evenly.
- **White balance (June 15).** Pair frames had been probe-white-balanced. Keeping them raw
  was adopted. At the time this was believed to expose a 15-25% illuminant colour change;
  that figure was wrong (see section 6). Probe white balance rescales only R and B, so the
  raw pair adds just the small bounce-colour difference.
- **A synthetic tint augmentation** on the pair frame (U[0.6,1.4] per channel) was tried
  and dropped.
- **Shading-first / flatness (June 16).** A pointwise "A = I/S_d" consistency term was
  rejected (it is algebraically the reconstruction loss and is satisfied by the leak it
  targets). A chroma-gated, low-frequency albedo flatness prior was designed instead.
- **V20 (June 29), rejected.** Shading-first analytic derive: grey shading g plus a dense
  unit-luminance chroma field c, albedo = clamp(I/(g*c)). It collapsed (S -> floor,
  a -> white) when the shadow loss perturbed the free shading scale, and was dropped. It is
  the closest ancestor of V21's factorisation; V21 differs by predicting albedo directly
  and feeding the head I/(S_lum*C) as a skip rather than deriving it.
- **Refinement levers (July 4).** On top of the 50k model: flatness prior, synthetic
  shadow invariance, ordinal shading, a frozen low-resolution refiner (V17.27). Rows were
  compared in a "Table B".
- **Wiring bug (July 7-9).** The checkpoint labelled "full CARI" had been trained with the
  cross-render losses never firing (two independent bugs). Table A (v17_41..44) was
  retrained from the 19k Stage-A fork with the losses verified in TensorBoard, and Table B
  was rebuilt on the corrected base.
- **3D-Front-IID v1.** 960 rooms, 5,056 coloured-light pairs with exact albedo, rendered
  with Blender. Added to the refinement mix (v17_29).

## 4. Thesis (July 11 - September 1)

Benchmark protocol fixed (ARAP cites Ordinal Shading Table 1; MAW cites CD-IID Table 1;
MID has no external citation). A full review (July 14) and a reference audit (July 15)
found and fixed fabricated or wrong citations. Submitted July 15, defended September 1.

What the thesis claimed, and what was later found to be wrong:

- It was titled "Coloured Illumination Constancy" and described MID as coloured-illuminant
  pairs. MID's flash is white (section 6).
- It called the pooled Cast_RMS "the chroma-cast score used throughout this literature".
  It was the project's own early metric.
- It presented the colour path (RGB skip + chroma loss on albedo) as made safe by, and
  needed because of, CARI. Table A shows the base model desaturates with or without CARI.
- The headline model was the refinement-study row v17_34 (CARI + 3D-Front + gentle
  flatness + shadow invariance), which mixed several levers that the matched studies could
  not attribute.

Also explored and not kept: hard-shadow augmentation on 3D-Front rows (July 13; the
pre-registered smoke test failed by 25-60x), an IIW ordinal-hinge fine-tune (v17_26;
WHDR 0.264 -> 0.220 but C_mat +47%, MAW dE +30%).

## 5. After the defence: re-checking the colour claim (September 2-10)

- **More baselines (Sept 2).** The full CD-IID cascade and RGB->X were run locally. CD-IID
  beats the model on hue drift, chroma error and MAW without desaturating; the claim was
  narrowed to lightness stability.
- **Phase A, no GPU (Sept 5).**
  - ARAP stores ground-truth albedo in three encodings (Radiance /179, 8-bit sRGB,
    unscaled); the evaluator's fixed mask kept ~1.5% of the frame on 40 scenes and dropped
    21 images. ARAP numbers are provisional until regenerated with `arap_preprocess.py`.
  - Illuminant colour gaps measured on probes and I/A: MID 0.030 median (86% of pairs
    below 0.08), ARAP indoor 0.070, 3D-Front v1 0.153 (10th percentile 0.069, i.e. never
    subtle), v2 pilot 0.123 (spans 0.008-0.35).
  - Hue drift split by scene illuminant gap: our high/low ratio 1.42-1.43 (as grey-shading
    CRefNet), CD-IID 1.02. CARI does not change it.
- **Phase B (Sept 5).** Pairs sampled by measured gap plus a calibrated 3000-6000 K tint on
  the pair frame, on top of the refinement recipe (v17_50/51). High/low ratio 1.474 ->
  1.447, single seed, below the 0.035 threshold. Dropped with the refinement study.
- **Study 1 (Sept 8-10), pre-registered.** A chroma explanation loss on the existing
  three-channel shading (v17_61..64). Seed noise of the ratio 0.014. With 3D-Front pairs in
  the mix it desaturated (spread ratio -0.10); on MID pairs only (seed 42) drift -3.5% but
  model chroma -2.9%, Cast_rel +0.0006: a null.
- **V21 scaffolding (Sept 10).** Tri-factor model I = A*(S_lum*C) + R with Y(C) = 1,
  3D-Front v2 renderer and validator. A generative albedo restorer was designed and
  deferred, and later removed.

## 6. Reframing and cleanup (October 8-9)

- Renamed CARI -> CIAI (Cross-Illumination Albedo Invariance), including the repository and project page URL.
- Claim narrowed to lightness stability; colour limits reported as findings; contributions
  rewritten; report, README, project page and portfolio corrected (revised edition).
- Refinement study (Table B), Phase B and the IIW fine-tune removed from the report and the
  code. "Ours" is now the base CIAI model v17_44 at 40k.
- Code reduced to one purpose: V17 + CIAI as reported (Table A, Study 1, 3D-Front colour
  study) and V21 for the next stage. CIAI pair losses moved to `src/losses/ciai.py` so any
  architecture can use them. The cleaned V17 training step was checked against the old
  code on synthetic batches for all eleven kept configs: every loss value and every
  parameter gradient is bitwise identical.

## Lessons kept

1. A stability score computed on predictions alone can be improved by removing colour;
   always report it beside an accuracy score.
2. Pre-register the endpoint and measure the seed noise floor before reading a contrast.
3. Check what a dataset actually varies before claiming what a model learns from it.
4. A synthetic signal the model learns perfectly (3D-Front inv_gap 0.0013) can still fail
   to transfer; measure transfer, not the training signal.
5. Verify that a loss fires (TensorBoard tag present and non-zero) before training rows
   that depend on it.
