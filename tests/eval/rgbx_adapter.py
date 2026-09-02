"""RGB->X adapter — Zeng et al., "RGB<->X: Image Decomposition and Synthesis Using
Material- and Lighting-aware Diffusion Models" (SIGGRAPH 2024).

Why this baseline: our two diffusion comparisons are both Marigold variants, so they
share a backbone, a training recipe and a failure mode. RGB->X is an independent
diffusion decomposer, which tests whether our colour-constancy result holds against
the diffusion family generally rather than against Marigold specifically.

Weights: `zheng95z/rgb-to-x` on HuggingFace. The pipeline class is NOT in diffusers —
it is `StableDiffusionAOVMatEstPipeline`, defined in the authors' repo, so that repo
must be on disk; set RGBX_REPO or clone it next to this project (see _import_pipeline).

The model is prompt-conditioned: one full denoising run per requested channel, chosen
by an "AOV" prompt string. We request albedo only, so a decomposition costs one
diffusion run at the configured step count — by far the slowest adapter here.

    COLOUR SPACES — both verified by reading the authors' code, not assumed
INPUT is LINEAR. Their demo loads PNGs with `load_ldr_image(..., from_srgb=True)`,
which does `image ** 2.2` (rgb2x/load_image.py:86-88) before the tensor reaches the
pipeline. Our harness already hands us display-linear, so it is passed through
unchanged — gamma-encoding it first would be a double transform.

OUTPUT is GAMMA-ENCODED. `VaeImageProcessorAOV.postprocess` applies
`image = torch.pow(image, 1.0 / 2.2)` under `do_gamma_correction: bool = True`
(rgb2x/pipeline_rgb2x.py:95-97), and the pipeline calls it without overriding that
default. So the returned albedo must be raised to 2.2 to return it to the linear
convention every metric in this harness expects.

Getting that second one backwards is exactly the double-gamma bug that once inflated
Marigold's constancy scores by silently desaturating its albedo (documents/evals/,
2026-06-19..20), which is why it is read off the source here rather than guessed.
`probe_albedo_gamma()` re-derives the same verdict empirically as a cross-check.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

_PIPE_ID = 'zheng95z/rgb-to-x'
_PROMPT = {
    'albedo': 'Albedo (diffuse basecolor)',
    'normal': 'Camera-space Normal',
    'roughness': 'Roughness',
    'metallic': 'Metallicness',
    'irradiance': 'Irradiance (diffuse lighting)',
}

# Where the authors' repo lives (needs pipeline_rgb2x.py). Override with RGBX_REPO.
_DEFAULT_REPO = Path(os.environ.get('RGBX_REPO', Path.home() / 'rgbx'))

_PipelineCls = None


def _import_pipeline():
    global _PipelineCls
    if _PipelineCls is not None:
        return _PipelineCls
    repo = _DEFAULT_REPO / 'rgb2x'
    if not (repo / 'pipeline_rgb2x.py').exists():
        raise FileNotFoundError(
            f'RGB->X pipeline code not found at {repo}. Clone it with\n'
            f'  git clone https://github.com/zheng95z/rgbx.git {_DEFAULT_REPO}\n'
            f'or point RGBX_REPO at an existing checkout.')
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from pipeline_rgb2x import StableDiffusionAOVMatEstPipeline
    _PipelineCls = StableDiffusionAOVMatEstPipeline
    return _PipelineCls


def load_rgbx(device, dtype=None):
    """Build the RGB->X pipeline (weights come from the HuggingFace cache)."""
    import torch
    from diffusers import DDIMScheduler
    cls = _import_pipeline()
    if dtype is None:
        dtype = torch.float16 if str(device).startswith('cuda') else torch.float32
    pipe = cls.from_pretrained(_PIPE_ID, torch_dtype=dtype)
    # The demo replaces the scheduler; keep that, since the released weights were
    # tuned with zero-terminal-SNR rescaling and trailing timestep spacing.
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing='trailing')
    pipe = pipe.to(str(device))
    pipe.set_progress_bar_config(disable=True)
    print(f'  Loaded RGB->X ({_PIPE_ID})')
    return pipe


def _run_aov(pipe, rgb_linear: np.ndarray, aov: str, max_size: int, steps: int,
             seed: int, device):
    """One denoising run for a single AOV. Input linear HWC; returns linear HWC float32."""
    import cv2
    import torch

    h, w = rgb_linear.shape[:2]
    scale = min(1.0, float(max_size) / max(h, w))
    nh = max(8, int(round(h * scale)) // 8 * 8)
    nw = max(8, int(round(w * scale)) // 8 * 8)
    small = cv2.resize(np.clip(rgb_linear, 0, 1), (nw, nh), interpolation=cv2.INTER_AREA)

    photo = torch.from_numpy(small).permute(2, 0, 1).to(str(device))
    gen = torch.Generator(device=str(device)).manual_seed(seed)
    out = pipe(prompt=_PROMPT[aov], photo=photo, num_inference_steps=steps,
               height=nh, width=nw, generator=gen, required_aovs=[aov])

    img = out.images[0][0]
    arr = np.asarray(img).astype(np.float32) / 255.0
    # Undo the pipeline's own 1/2.2 postprocess to get back to linear.
    arr = np.power(np.clip(arr, 0.0, 1.0), 2.2)
    if arr.shape[:2] != (h, w):
        arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(arr, 0.0, 1.0)


def probe_albedo_gamma(pipe, rgb_display_linear: np.ndarray, max_size: int = 768,
                       steps: int = 50, seed: int = 0, device='cuda'):
    """Cross-check the documented output space against the model's own factorisation.

    For a diffuse surface the model should satisfy image ~= albedo * irradiance in
    the space it predicts in. We undo the pipeline's gamma (as run_rgbx does) and
    compare that reconstruction against the alternative of leaving it applied;
    the smaller scale-invariant residual indicates the correct handling.
    """
    lin = np.clip(rgb_display_linear, 0, 1).astype(np.float32)
    alb_lin = _run_aov(pipe, lin, 'albedo', max_size, steps, seed, device)
    irr_lin = _run_aov(pipe, lin, 'irradiance', max_size, steps, seed, device)

    res = {}
    for name, (a, i) in (
        ('gamma_undone (what run_rgbx does)', (alb_lin, irr_lin)),
        ('gamma_left_applied', (alb_lin ** (1 / 2.2), irr_lin ** (1 / 2.2))),
    ):
        recon = np.clip(a * i, 0, 1)
        s = float((recon * lin).sum() / max(float((recon * recon).sum()), 1e-8))
        res[name] = float(np.abs(s * recon - lin).mean())
    return res, alb_lin, irr_lin


def run_rgbx(pipe, rgb_display_linear: np.ndarray, max_size: int, device,
             steps: int = 50, seed: int = 0):
    """Predict albedo.

    Args:
        rgb_display_linear: display-space LINEAR [0,1] HWC float array — the same
            convention every other adapter in this harness takes, and already the
            space RGB->X wants, so it is passed through unchanged.

    Returns:
        (albedo_hwc, None): float32 HWC [0,1] LINEAR albedo. Shading is not returned:
        RGB->X predicts irradiance, which is not the diffuse shading our other
        adapters produce, and conflating the two would corrupt the shading metrics.
    """
    alb = _run_aov(pipe, np.clip(rgb_display_linear, 0, 1).astype(np.float32),
                   'albedo', max_size, steps, seed, device)
    return alb.astype(np.float32), None
