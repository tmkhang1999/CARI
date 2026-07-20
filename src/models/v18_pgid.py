"""V18 — Physics-Grounded One-Step Colorful-Diffuse Intrinsic Diffusion.

A single SD-2.1 U-Net fine-tuned as a ONE-STEP feed-forward predictor that jointly
outputs colorful albedo + diffuse shading (shading in the inverse domain π), trained
end-to-end with the project's physics losses in image space (V17Loss). No V17 CNN.

One-step recipe (validated against `documents/references/diffusion-e2e-ft`):
  - SD-2.1 (v-prediction, native), scheduler rescale_betas_zero_snr + trailing.
  - Fixed terminal timestep t=999; feed z_T = zeros (E2E-FT default → deterministic).
  - Predict v; convert x0 = √ᾱ·z_T − √(1−ᾱ)·v   (= −v at t=999 where ᾱ=0).
  - Decode [z_A | z_π] → A, π → S_d=(1−π)/π → R=(I−A·S_d)₊  (physics exact).

Design doc: documents/design/V18_PGID_design.md
"""

import torch
import torch.nn as nn

from .diffusion.vae import VAEWrapper
from .diffusion.denoiser import IntrinsicLatentDenoiser


class V18PGID(nn.Module):
    """Standalone one-step intrinsic latent diffusion.

    Training  : `forward(rgb)` runs one step, decodes, and returns a V17-style
                predictions dict (consumed by V17Loss in image space) plus the raw
                8-ch x0 latent (for the optional latent anchor).
    Inference : `sample(rgb, num_steps=1)` = one forward; num_steps>1 = optional
                stochastic DDIM (uncertainty mode).
    """

    def __init__(self, config: dict):
        super().__init__()

        sd_path = config.get("sd_pretrained", "Manojb/stable-diffusion-2-1-base")

        # ── VAE (frozen) ─────────────────────────────────────────────────────
        self.vae = VAEWrapper(sd_path, subfolder="vae")
        self.vae.requires_grad_(False)
        self.latent_ch: int = self.vae.latent_channels        # 4
        self.latent_stride: int = self.vae.downscale_factor   # 8

        # ── U-Net (only trainable component) ────────────────────────────────
        self.denoiser = IntrinsicLatentDenoiser(sd_path, subfolder="unet")

        self.shading_gamma = bool(config.get("shading_gamma", False))
        self.pi_floor = float(config.get("pi_floor", 5e-3))
        self.noise_type = str(config.get("noise_type", "zeros"))  # zeros|gaussian

        # Learned null cross-attention embedding (empty prompt). SD-2.1 → 1024.
        cross_attn_dim = int(config.get("cross_attn_dim", 1024))
        null_seq_len = int(config.get("null_seq_len", 77))
        self.null_embedding = nn.Parameter(torch.zeros(1, null_seq_len, cross_attn_dim))

        # ── Scheduler: v-prediction + zero-terminal-SNR + trailing ───────────
        from diffusers import DDPMScheduler, DDIMScheduler

        sched_kwargs = dict(
            num_train_timesteps=int(config.get("num_train_timesteps", 1000)),
            beta_start=float(config.get("beta_start", 0.00085)),
            beta_end=float(config.get("beta_end", 0.012)),
            beta_schedule="scaled_linear",
            prediction_type="v_prediction",
            clip_sample=False,
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
        )
        self.noise_scheduler = DDPMScheduler(**sched_kwargs)
        self.ddim_scheduler = DDIMScheduler(**sched_kwargs)
        self.fixed_timestep = int(config.get("num_train_timesteps", 1000)) - 1  # 999

    # ────────────────────────────────────────────────────────────────────────
    # Latent helpers
    # ────────────────────────────────────────────────────────────────────────

    def _null_embed(self, batch_size: int) -> torch.Tensor:
        return self.null_embedding.expand(batch_size, -1, -1)

    def _enc_img(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.vae.encode(x, gamma=True)

    def _enc_shading(self, pi: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.vae.encode(pi, gamma=self.shading_gamma)

    def _dec_img(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z, gamma=True)

    def _dec_shading(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z, gamma=self.shading_gamma)

    def _x0_from_v(self, z_t, v_pred, timesteps):
        """x0 = √ᾱ_t·z_t − √(1−ᾱ_t)·v.  At t=999 (zero-SNR) ᾱ_t=0 ⇒ x0=−v."""
        alphas = self.noise_scheduler.alphas_cumprod.to(z_t.device, z_t.dtype)
        a_t = alphas[timesteps].sqrt().view(-1, 1, 1, 1)
        s_t = (1.0 - alphas[timesteps]).sqrt().view(-1, 1, 1, 1)
        return a_t * z_t - s_t * v_pred

    def encode_targets(self, albedo_gt: torch.Tensor, shading_pi_gt: torch.Tensor) -> torch.Tensor:
        """[z_A | z_π] of the GT — used for the optional latent anchor loss."""
        z_A = self._enc_img(albedo_gt)
        z_pi = self._enc_shading(shading_pi_gt)
        return torch.cat([z_A, z_pi], dim=1)

    def _initial_latent(self, rgb: torch.Tensor) -> torch.Tensor:
        B, _, H, W = rgb.shape
        Hl, Wl = H // self.latent_stride, W // self.latent_stride
        shape = (B, 2 * self.latent_ch, Hl, Wl)
        if self.noise_type == "zeros":
            return torch.zeros(shape, device=rgb.device, dtype=rgb.dtype)
        return torch.randn(shape, device=rgb.device, dtype=rgb.dtype)

    # ────────────────────────────────────────────────────────────────────────
    # One-step prediction core
    # ────────────────────────────────────────────────────────────────────────

    def _predict_x0(self, rgb: torch.Tensor, z_T: torch.Tensor | None = None) -> torch.Tensor:
        """One U-Net call → x0 latent [z_A | z_π]. Inherits grad context of caller."""
        B = rgb.shape[0]
        z_I = self._enc_img(rgb)
        if z_T is None:
            z_T = self._initial_latent(rgb)
        t = torch.full((B,), self.fixed_timestep, device=rgb.device, dtype=torch.long)
        v = self.denoiser(z_T, z_I, t, self._null_embed(B))
        return self._x0_from_v(z_T, v, t)

    def _decode_decompose(self, x0: torch.Tensor, rgb: torch.Tensor) -> dict:
        z_A, z_pi = x0[:, :self.latent_ch], x0[:, self.latent_ch:]
        A = self._dec_img(z_A).clamp(0.0, 1.0)
        pi = self._dec_shading(z_pi).clamp(self.pi_floor, 1.0 - 1e-4)
        S_d = (1.0 - pi) / pi
        R = (rgb - A * S_d).clamp(min=0.0)
        return {
            "a_d": A,
            "shading": pi,                  # π domain — V17Loss shading term uses this
            "shading_linear": S_d,
            "residual": R,
            "rgb_reconstructed": A * S_d + R,
            "x0_latent": x0,
        }

    def forward(self, rgb: torch.Tensor) -> dict:
        """Training forward: one step → decode → V17-style predictions dict."""
        x0 = self._predict_x0(rgb)
        return self._decode_decompose(x0, rgb)

    # ────────────────────────────────────────────────────────────────────────
    # Inference
    # ────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(self, rgb: torch.Tensor, num_steps: int = 1,
               guidance_lambda: float = 0.0, seed: int | None = None) -> dict:
        """num_steps<=1 → one-step (default). num_steps>1 → stochastic DDIM."""
        if num_steps <= 1:
            out = self._decode_decompose(self._predict_x0(rgb), rgb)
            out.pop("x0_latent", None)
            return out

        # Optional stochastic multi-step (uncertainty mode)
        device = rgb.device
        B = rgb.shape[0]
        z_I = self._enc_img(rgb)
        enc = self._null_embed(B)
        gen = torch.Generator(device=device)
        if seed is not None:
            gen.manual_seed(seed)
        z_t = torch.randn(B, 2 * self.latent_ch, *z_I.shape[-2:], generator=gen, device=device)
        self.ddim_scheduler.set_timesteps(num_steps, device=device)
        for step_t in self.ddim_scheduler.timesteps:
            v = self.denoiser(z_t, z_I, step_t.expand(B), enc)
            z_t = self.ddim_scheduler.step(v, step_t, z_t).prev_sample
        out = self._decode_decompose(z_t, rgb)
        out.pop("x0_latent", None)
        return out
