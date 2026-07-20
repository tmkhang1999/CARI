"""SD-1.5 U-Net adapted for joint albedo + diffuse-shading latent diffusion (V18).

The diffusion target is an 8-channel latent = [z_A (4) | z_pi (4)]:
  z_A  — VAE latent of RGB albedo A
  z_pi — VAE latent of diffuse shading in the inverse domain pi = 1/(S+1)

Conditioning is the input-image latent z_I (4ch), concatenated on the input.

  conv_in : 8 (z_t) + 4 (z_I) = 12  ->  320
  conv_out: 320  ->  8

Warm start: the pretrained 4-channel SD weights are copied into each
4-channel modality slot, so at init the U-Net behaves like the pretrained
noise predictor for both the albedo and shading latent groups. Full
fine-tuning then specialises them. No CNN prior (standalone — Marigold/IID
recipe) so V18 does not depend on V17.
"""

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel


class IntrinsicLatentDenoiser(nn.Module):
    """SD-1.5 U-Net with 8-channel intrinsic latent target + 4-channel image cond."""

    TARGET_CH = 8   # [z_A (4) | z_pi (4)]
    COND_CH = 4     # z_I

    def __init__(self, pretrained_name_or_path: str, subfolder: str = "unet"):
        super().__init__()
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_name_or_path, subfolder=subfolder
        )
        self._latent_ch = int(self.unet.config.in_channels)  # 4 for SD-1.5
        self._expand_conv_in()
        self._expand_conv_out()
        # Keep config metadata consistent with the new conv shapes.
        self.unet.config.in_channels = self.TARGET_CH + self.COND_CH
        self.unet.config.out_channels = self.TARGET_CH

    def _expand_conv_in(self):
        """conv_in: latent(4) -> 12ch, copy pretrained into each 4ch group."""
        old = self.unet.conv_in
        w = old.weight.data                       # (320, 4, 3, 3)
        L = self._latent_ch                        # 4
        new_in = self.TARGET_CH + self.COND_CH     # 12
        new = nn.Conv2d(
            new_in, old.out_channels,
            kernel_size=old.kernel_size, stride=old.stride,
            padding=old.padding, bias=old.bias is not None,
        )
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, 0 * L:1 * L] = w   # z_t albedo slot
            new.weight[:, 1 * L:2 * L] = w   # z_t shading slot
            new.weight[:, 2 * L:3 * L] = w   # z_I condition slot
            if old.bias is not None:
                new.bias.copy_(old.bias)
        self.unet.conv_in = new

    def _expand_conv_out(self):
        """conv_out: 320 -> 8ch, duplicate pretrained 4ch head for both modalities."""
        old = self.unet.conv_out
        w = old.weight.data                       # (4, 320, 3, 3)
        L = self._latent_ch                        # 4
        new = nn.Conv2d(
            old.in_channels, self.TARGET_CH,
            kernel_size=old.kernel_size, stride=old.stride,
            padding=old.padding, bias=old.bias is not None,
        )
        with torch.no_grad():
            new.weight.zero_()
            new.weight[0 * L:1 * L] = w   # albedo output
            new.weight[1 * L:2 * L] = w   # shading output
            if old.bias is not None:
                new.bias[0 * L:1 * L] = old.bias
                new.bias[1 * L:2 * L] = old.bias
        self.unet.conv_out = new

    def forward(
        self,
        z_t: torch.Tensor,                  # (B, 8, h, w) noisy intrinsic latent
        z_I: torch.Tensor,                  # (B, 4, h, w) image conditioning latent
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Return v-prediction (B, 8, h, w)."""
        x = torch.cat([z_t, z_I], dim=1)    # (B, 12, h, w)
        return self.unet(
            x, timesteps, encoder_hidden_states=encoder_hidden_states
        ).sample
