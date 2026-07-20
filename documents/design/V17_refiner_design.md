# V17.27 Frozen Low-Resolution Albedo Refiner Design

Date: 2026-07-04  
Status: diagnostic experiment, not thesis spine unless gates pass.

## Context

Current measured best backbone candidate is `B_combo` / `v17_23@60k`, trained from the V17 50k checkpoint with:

- `lambda_alb_flat_lf: 4.0`
- `lambda_shadow_inv: 1.0`
- `lambda_shadow_explain: 0.0`

Draft25 subset result summary:

| row | MID Cmat | ARAP C | ARAP colored | ARAP siRMSE | MAW dE | IIW WHDR |
|---|---:|---:|---:|---:|---:|---:|
| B_ctrl | 0.215 | 0.109 | 0.122 | 0.282 | 3.786 | 0.282 |
| B_flat | 0.185 | 0.097 | 0.113 | 0.252 | 4.032 | 0.271 |
| B_shinv | 0.194 | 0.118 | 0.140 | 0.337 | 3.651 | 0.292 |
| B_combo | 0.199 | 0.087 | 0.095 | 0.229 | 3.979 | 0.264 |
| B_killgate | 0.252 | 0.131 | 0.150 | 0.338 | 4.920 | 0.340 |
| B_ordinal | 0.266 | 0.160 | 0.174 | 0.392 | 3.550 | 0.354 |

`B_combo` is not best on every metric, but it is the strongest balanced row for the current pain points: ARAP constancy, ARAP si-RMSE, and IIW WHDR. The refiner starts from `B_combo`, not from the original V17 50k checkpoint.

## Goal

Test whether a small extra module can improve the remaining structure/homogeneity weakness without changing the V17 backbone/trunk/heads and without reopening the illumination-to-albedo leak.

The refiner is allowed to make only a bounded low-frequency correction to V17's predicted albedo. It is not a new albedo decoder.

## Non-Goals

- Do not modify `src/models/v17.py`.
- Do not train DINOv2, DPT trunk, albedo head, or shading head in the first diagnostic stage.
- Do not provide raw RGB or image chromaticity to the refiner.
- Do not use a reconstruction-to-input objective as the refiner's main signal.
- Do not add a learned residual image branch.
- Do not use `shadow_explain`; previous ablations mark it as a kill-gate failure.

## Implemented Files

- `src/models/v17_refiner.py`
  - `LowResLogAlbedoRefiner`
  - `IntrinsicDecompositionV17Refiner`
- `src/configs/v17_27.yaml`
- `src/train_v17.py`
  - adds version `17.27` to the model factory
- `src/models/__init__.py`
  - exports `IntrinsicDecompositionV17Refiner`
- `src/losses/flexible_loss_v17.py`
  - adds optional refiner guard losses, default off
- eval/inference routing:
  - `tests/eval/eval_mid_constancy.py`
  - `tests/eval/eval_maw.py`
  - `tests/eval/eval_arap.py`
  - `tests/infer/infer_wild.py` for IIW/shared inference loading

## Architecture

The wrapper subclasses `IntrinsicDecompositionV17` only to preserve checkpoint key compatibility. Existing V17 module names remain unchanged. New parameters live under `refiner.*`.

Forward pass:

```text
base = V17(rgb)
A0 = base['a_d']
S0 = base['shading_linear']

features = [log(A0), log(luminance(S0)), chroma(S0)]
delta = Refiner(features_downsampled_to_H/8)
A1 = exp(log(A0) + beta * tanh(delta))
```

Default config:

```yaml
model:
  version: 17.27
  refiner:
    freeze_base: true
    detach_inputs: true
    hidden_ch: 48
    num_blocks: 4
    down: 8
    beta: 0.10
```

Output dictionary:

- `a_d`: refined albedo `A1`, used by existing losses/evals
- `a_base`: frozen V17 albedo `A0`, for guard losses/debug
- `a_refine_delta`: bounded correction map, for inspection/debug
- `shading`, `shading_linear`, `dino_tokens`, etc.: inherited from V17
- `residual` and `rgb_reconstructed`: recomputed from `A1` and original `S0`

## Why Low-Resolution Log Residual

The target failure is mostly low-frequency shadow/ramp leak on homogeneous regions. A high-resolution or raw-RGB-conditioned refiner can simply copy material/shadow edges from the input and preserve the failure. Low-resolution correction forces the module to adjust broad albedo fields rather than hallucinate new texture.

Log-albedo makes the correction multiplicative and scale-aware:

```text
A1 / A0 = exp(beta * tanh(delta))
```

With `beta = 0.10`, the initial maximum correction range is about ±10.5% multiplicative. `beta` can be increased to `0.20` only if the frozen diagnostic passes the gates.

The final convolution is zero-initialized, so `A1 == A0` at step 0.

## Losses

The refiner reuses the existing V17/CARI objective because it replaces `predictions['a_d']`. Therefore albedo L1/MSG/DSSIM/chroma, CARI L_inv, flat-LF, material consistency, shadow_inv, IIW ordinal hinge if enabled, and eval metrics all naturally score `A1`.

Two optional guard losses are added:

```yaml
loss:
  lambda_refiner_delta: 0.02
  lambda_refiner_hf: 0.05
  refiner_hf_down: 8
```

- `loss_refiner_delta`: masked L1 between `log(A1)` and `log(A0)`, discourages global drift.
- `loss_refiner_hf`: preserves high-frequency log-albedo residual of `A0`; allows low-frequency correction while guarding texture/detail.

These losses are no-ops for normal V17/V20 models because they only fire when predictions include `a_base`.

## Training Command

```bash
bash scripts/train.sh --version 17.27 --cuda 0   --resume checkpoints/v17_23/checkpoint_iter_60000.pth
```

The config sets:

```yaml
train:
  lr: 1.0e-4
  use_cosine_lr: false
  reset_lr: true
  skip_optimizer: true
  extend_iterations: 65000
  checkpoint_interval_iters: 1000
  val_interval_iters: 1000
```

This is a 5k-step diagnostic from 60k to 65k. Because base V17 is frozen, only `refiner.*` parameters train.

## Go / No-Go Gates

Compare against `B_combo@60k`.

Go only if all hold:

- MID `C_mat` worsens less than 5% from `0.199`.
- MAW dE worsens less than 5-10% from `3.979`.
- ARAP `C_arap`, colored `C_arap`, and Cast_RMS do not worsen.
- ARAP si-RMSE improves at least about 5%, or IIW WHDR improves at least 1 absolute point.
- Visual sheets show actual shadow/ramp removal, not only smoothing/desaturation.

Immediate no-go if any hold:

- `a_refine_delta` saturates spatially or globally.
- Homogeneous regions flatten but material boundaries smear.
- MAW chromaticity regresses sharply.
- IIW improves while MID/ARAP constancy collapses.
- Shadow edges are visible in `a_refine_delta`, indicating the refiner is copying an input shadow pattern.

## Specific Residual ARAP Scenes To Watch

From draft25 per-scene ARAP, `B_combo` improves many hard scenes but leaves `villa` essentially unchanged:

- Major wins vs `B_ctrl`: `chocofur`, `conference`, `bread`, `violin`.
- Residual failure: `villa` remains about `0.37` C_arap.
- Small regressions: `postit`, `villa`, `workshop`.

Use the generated comparison sheet:

```text
tests/visualizations/draft25/arap_Bctrl_Bflat_Bcombo_worst_scenes.jpg
```

The refiner is only useful if it helps residual failures like `villa` without undoing the gains on `chocofur`, `conference`, `bread`, and `violin`.

## Risks

1. **Refiner copies illumination leak.**  
   Mitigation: no raw RGB input, low-res correction, bounded log residual, CARI losses still active.

2. **Refiner becomes a blur module.**  
   Mitigation: high-frequency preservation against `A0`; inspect texture-heavy MAW/ARAP scenes.

3. **Frozen base prevents enough correction.**  
   This is intentional for the first diagnostic. If frozen refiner cannot help, joint finetuning is unlikely to be a safe thesis bet. If frozen refiner passes, a second-stage low-LR unfreeze of albedo head only can be considered.

4. **Metrics improve by desaturation.**  
   Mitigation: MAW dE/chromaticity and MID cast gates; visual inspection of `a_refine_delta`.

## Possible Stage 2 Only If Stage 1 Passes

If frozen-refiner stage passes, consider:

- unfreeze only albedo head + refiner
- keep DINO and DPT trunk frozen
- lower LR on albedo head by 5-10x relative to refiner
- keep raw RGB paths unchanged
- continue strict gates

Do not jump directly to a generative decoder for the thesis version. It may improve visual sharpness but risks damaging the current constancy contribution.
