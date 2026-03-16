"""Version 2: adapter-pyramid cross-decoder architecture."""

import torch
import torch.nn as nn

from .encoders.image_encoder import ImageEncoder
from .decoders.progressive_decoder import ProgressiveDecoder
from .modules.adapters import ShadingAdapter, ColorfulAdapter, AlbedoAdapter


class IntrinsicDecompositionV2(nn.Module):
    def __init__(self, config):
        super().__init__()

        z_channels = config.get("z_channels", 1024)
        freeze_stages = config.get("freeze_stages", [1, 2])
        model_name = config.get("backbone", config.get("model_name", "convnextv2_base"))
        pretrained = config.get("pretrained", True)

        self.image_encoder = ImageEncoder(
            model_name=model_name,
            freeze_stages=freeze_stages,
            pretrained=pretrained,
        )
        skip_channels = self.image_encoder.feature_channels

        self.decoder_a = ProgressiveDecoder(
            in_channels=z_channels,
            skip_channels=skip_channels,
            out_channels=1,
            stage_extra_channels=(0, 0, 0),
            activation="softplus",
        )
        self.decoder_b = ProgressiveDecoder(
            in_channels=z_channels,
            skip_channels=skip_channels,
            out_channels=2,
            stage_extra_channels=(256, 128, 64),
            activation="sigmoid",
        )
        self.decoder_c = ProgressiveDecoder(
            in_channels=z_channels,
            skip_channels=skip_channels,
            out_channels=3,
            stage_extra_channels=(512, 256, 128),
            activation="sigmoid",
        )
        self.decoder_d = ProgressiveDecoder(
            in_channels=z_channels,
            skip_channels=skip_channels,
            out_channels=3,
            stage_extra_channels=(1024, 512, 256),
            activation="softplus",
        )

        self.shading_adapter = ShadingAdapter()
        self.colorful_adapter = ColorfulAdapter()
        self.albedo_adapter = AlbedoAdapter()

    @staticmethod
    def _to_chroma(xi):
        eps = 1e-7
        c_rg = (1.0 - xi[:, 0:1]) / (xi[:, 0:1] + eps)
        c_bg = (1.0 - xi[:, 1:2]) / (xi[:, 1:2] + eps)
        return torch.cat([c_rg, torch.ones_like(c_rg), c_bg], dim=1)

    def forward(self, rgb, m_diffuse=None, **kwargs):
        z_global, skip_features = self.image_encoder(rgb)

        s_g = self.decoder_a(z_global, skip_features)

        s_g_pyr = self.shading_adapter(s_g)
        xi = self.decoder_b(
            z_global,
            skip_features,
            extra_features=[s_g_pyr[3], s_g_pyr[2], s_g_pyr[1]],
        )

        c = self._to_chroma(xi)
        s_c = s_g * c

        s_c_pyr = self.colorful_adapter(s_c)
        a_d = self.decoder_c(
            z_global,
            skip_features,
            extra_features=[s_c_pyr[3], s_c_pyr[2], s_c_pyr[1]],
        )

        a_d_pyr = self.albedo_adapter(a_d)
        s_d = self.decoder_d(
            z_global,
            skip_features,
            extra_features=[
                torch.cat([s_c_pyr[3], a_d_pyr[3]], dim=1),
                torch.cat([s_c_pyr[2], a_d_pyr[2]], dim=1),
                torch.cat([s_c_pyr[1], a_d_pyr[1]], dim=1),
            ],
        )

        return {
            "s_g": s_g,
            "xi": xi,
            "c": c,
            "s_c": s_c,
            "a_d": a_d,
            "s_d": s_d,
        }
