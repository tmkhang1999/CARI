"""SD-style VAE wrapper for V18.

All inputs assumed to be in [0, 1] linear space. Gamma encoding (x^1/2.2)
aligns the albedo distribution with the VAE's sRGB training distribution.
"""

import torch
import torch.nn as nn
from diffusers import AutoencoderKL


class VAEWrapper(nn.Module):
    """Frozen SD-1.5 VAE with gamma encode/decode for albedo.

    encode(x) → z: applies gamma, normalises to [-1,1], returns z * scale_factor
    decode(z) → x: divides by scale_factor, decodes, un-gammas to [0,1]

    The scale_factor (0.18215 for SD-1.5) is the standard diffusers convention
    so that latents have unit variance during training.
    """

    def __init__(self, pretrained_name_or_path: str, subfolder: str = "vae"):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_name_or_path, subfolder=subfolder
        )
        self.vae.requires_grad_(False)
        self.scale_factor: float = float(self.vae.config.scaling_factor)

    @property
    def latent_channels(self) -> int:
        return int(self.vae.config.latent_channels)

    @property
    def downscale_factor(self) -> int:
        """Spatial downsampling of the VAE (8 for SD-1.5: 4 block_out_channels)."""
        return 2 ** (len(self.vae.config.block_out_channels) - 1)

    def encode(self, x: torch.Tensor, gamma: bool = True) -> torch.Tensor:
        """Encode image [0,1] → scaled latent (deterministic posterior mean).

        Uses the posterior MODE (mean), not a stochastic sample: as a frozen
        encoder the VAE should give a stable, jitter-free target/condition
        latent (Marigold convention). Stochastic sampling would add noise to
        the diffusion target every step.

        Args:
            x: Image tensor in [0, 1]
            gamma: Apply x^(1/2.2) before encoding (True for albedo and RGB to
                   match the VAE sRGB training distribution; False for the
                   pi-domain shading map, which is already a bounded [0,1] map).
        Returns:
            Scaled latent z_mode * scale_factor.
        """
        x = x.clamp(0.0, 1.0)
        if gamma:
            x = x.pow(1.0 / 2.2)
        x_norm = x * 2.0 - 1.0  # VAE expects [-1, 1]
        posterior = self.vae.encode(x_norm)
        z = posterior.latent_dist.mode()
        return z * self.scale_factor

    def decode(self, z: torch.Tensor, gamma: bool = True) -> torch.Tensor:
        """Decode scaled latent → image [0,1].

        Args:
            z: Scaled latent (as returned by encode, i.e. z * scale_factor)
            gamma: Un-apply gamma (x^2.2) after decoding.
        Returns:
            Image in [0, 1].
        """
        z_unscaled = z / self.scale_factor
        x = self.vae.decode(z_unscaled).sample
        x = ((x + 1.0) / 2.0).clamp(0.0, 1.0)
        if gamma:
            x = x.pow(2.2)
        return x.clamp(0.0, 1.0)
