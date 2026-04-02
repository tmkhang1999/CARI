"""
Version 1 Model: Bottleneck-Only Cross-Decoder Architecture
Simple baseline with no attention, no SPADE, no additional encoders.
Cross-decoder signals are encoded to bottleneck resolution and concatenated to Z only.
"""

import torch
import torch.nn as nn

from .encoders.image_encoder import ImageEncoder
from .decoders.decoder import DecoderA, DecoderB, DecoderC, DecoderD
from .modules.adapters import BottleneckAdapter


class IntrinsicDecompositionV1(nn.Module):
    """
    Version 1: Bottleneck-only cross-decoder architecture.

    Pipeline:
        Z + F_img                                -> Dec A -> D_g
        Z + BottleneckAdapter(S_g) + F_img       -> Dec B -> xi (chroma)
        Z + BottleneckAdapter(S_c) + F_img       -> Dec C -> A_d
        Z + BottleneckAdapter(S_c, A_d) + F_img  -> Dec D -> pi
    """
    def __init__(self, config):
        super().__init__()

        # Extract config
        z_channels = config.get('z_channels', 1024)
        freeze_stages = config.get('freeze_stages', [1, 2])
        model_name = config.get('backbone', config.get('model_name', 'convnextv2_base'))
        pretrained = config.get('pretrained', True)
        input_size = int(config.get('input_size', 384))
        if input_size < 32:
            raise ValueError(f"input_size must be >= 32, got {input_size}")

        # Image encoder
        self.image_encoder = ImageEncoder(
            model_name=model_name,
            freeze_stages=freeze_stages,
            pretrained=pretrained,
        )
        skip_channels = self.image_encoder.feature_channels

        # Decoders
        self.decoder_a = DecoderA(z_channels=z_channels, skip_channels=skip_channels)
        self.decoder_b = DecoderB(z_channels=z_channels, skip_channels=skip_channels)
        self.decoder_c = DecoderC(z_channels=z_channels, skip_channels=skip_channels)
        self.decoder_d = DecoderD(z_channels=z_channels, skip_channels=skip_channels)

        # Pre-initialize adapters so optimizer captures their parameters.
        z_hw = input_size // 32
        self.z_spatial_size = (z_hw, z_hw)
        self.z_channels = z_channels
        self.adapter_s_g = BottleneckAdapter(
            in_channels=1,
            z_channels=self.z_channels,
            z_spatial_size=self.z_spatial_size,
        )
        self.adapter_s_c = BottleneckAdapter(
            in_channels=3,
            z_channels=self.z_channels,
            z_spatial_size=self.z_spatial_size,
        )
        self.adapter_a_d = BottleneckAdapter(
            in_channels=3,
            z_channels=self.z_channels,
            z_spatial_size=self.z_spatial_size,
        )

    def _validate_bottleneck_shape(self, z_global):
        """Fail fast if runtime bottleneck shape differs from configured input size."""
        _, _, h, w = z_global.shape
        if (h, w) != self.z_spatial_size:
            raise RuntimeError(
                f"V1 adapter bottleneck mismatch: got {(h, w)}, expected {self.z_spatial_size}. "
                "Set model.input_size to match training input_size."
            )

    def forward(self, rgb, m_diffuse=None, **kwargs):
        """
        Args:
            rgb: (N, 3, H, W) input RGB image
            m_diffuse: (N,) optional routing mask for diffuse GT availability
                       (unused in model; detach routing is handled in training)

        Returns:
            Dictionary with keys:
                s_g: (N, 1, H, W) inverse gray shading D_g
                xi: (N, 2, H, W) bounded chroma ratio
                c: (N, 3, H, W) chroma map (converted from xi)
                s_c: (N, 3, H, W) colorful shading = S_g * C
                a_d: (N, 3, H, W) diffuse albedo
                pi: (N, 3, H, W) inverse diffuse shading pi
        """
        # Encode image
        z_global, skip_features = self.image_encoder(rgb)
        self._validate_bottleneck_shape(z_global)

        # Dec A: inverse gray shading D_g
        d_g = self.decoder_a(z_global, skip_features)

        # Convert to linear S_g for the cascade branches (Dec B/C/D).
        s_g = 1.0 / (d_g + 1e-6) - 1.0

        # Adapt S_g to bottleneck
        s_g_adapted = self.adapter_s_g(s_g)
        z_b = torch.cat([z_global, s_g_adapted], dim=1)

        # Dec B: Chroma (bounded ratio)
        xi = self.decoder_b(z_b, skip_features)

        # Convert xi to chroma C: C = [C_R/G, C_B/G]
        # xi = [1/(C_R/G+1), 1/(C_B/G+1)] -> xi = 1/(C+1) -> C+1 = 1/xi -> C = 1/xi - 1
        eps = 1e-7
        c_rg = (1 - xi[:, 0:1, :, :]) / (xi[:, 0:1, :, :] + eps)
        c_bg = (1 - xi[:, 1:2, :, :]) / (xi[:, 1:2, :, :] + eps)

        # Build RGB chroma: C = [C_R/G * S_g, S_g, C_B/G * S_g]
        # Simplified: C_R = C_R/G, C_G = 1, C_B = C_B/G
        c = torch.cat([c_rg, torch.ones_like(c_rg), c_bg], dim=1)

        # Colorful shading S_c = S_g * C
        s_c = s_g * c

        # Adapt S_c to bottleneck
        s_c_adapted = self.adapter_s_c(s_c)
        z_c = torch.cat([z_global, s_c_adapted], dim=1)

        # Dec C: Diffuse albedo
        a_d = self.decoder_c(z_c, skip_features)

        # Adapt A_d to bottleneck
        # Dec D should not backprop into Dec C through this branch.
        a_d_adapted = self.adapter_a_d(a_d.detach())
        z_d = torch.cat([z_global, s_c_adapted, a_d_adapted], dim=1)

        # Dec D: inverse diffuse shading pi
        pi = self.decoder_d(z_d, skip_features)

        return {
            'd_g': d_g,
            'xi': xi,
            'c': c,
            's_c': s_c,
            'a_d': a_d,
            'pi': pi
        }


if __name__ == '__main__':
    # Test
    config = {
        'z_channels': 1024,
        'backbone': 'convnext_base',
        'freeze_stages': [1, 2]
    }

    model = IntrinsicDecompositionV1(config)
    x = torch.randn(2, 3, 384, 384)

    # Test without masking
    outputs = model(x)
    print("Model outputs:")
    for k, v in outputs.items():
        print(f"  {k}: {v.shape}")

    # Test with masking
    m_diffuse = torch.tensor([1, 0], dtype=torch.float32)
    outputs = model(x, m_diffuse=m_diffuse)
    print("\nWith masking:")
    for k, v in outputs.items():
        print(f"  {k}: {v.shape}")
