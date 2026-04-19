"""Stage 1 V6: self-contained CCR/SPADE + normal-guided decoders."""

import torch
import torch.nn as nn

from .encoders.image_encoder import ImageEncoder
from .encoders.normal_encoder import NormalEncoder
from .encoders.ccr_encoder import CCREncoder
from .decoders.decoder import DecoderB
from .decoders.progressive_decoder import ProgressiveDecoder
from .modules.adapters import BottleneckAdapter
from .modules.residual_attention import ResidualAttention
from .modules.spade import SPADE
from .ccr_utils import compute_ccr


class IntrinsicDecompositionV6(nn.Module):
    def __init__(self, config):
        super().__init__()

        z_channels = config.get('z_channels', 1024)
        freeze_stages = config.get('freeze_stages', [1, 2])
        model_name = config.get('backbone', config.get('model_name', 'convnextv2_base'))
        pretrained = config.get('pretrained', True)
        input_size = int(config.get('input_size', 384))
        if input_size < 32:
            raise ValueError(f"input_size must be >= 32, got {input_size}")

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
            activation='sigmoid',
        )
        self.decoder_b = DecoderB(z_channels=z_channels, skip_channels=skip_channels)
        self.decoder_c = ProgressiveDecoder(
            in_channels=z_channels * 2,
            skip_channels=skip_channels,
            out_channels=3,
            stage_extra_channels=(0, 0, 0),
            activation='sigmoid',
        )
        self.decoder_d = ProgressiveDecoder(
            in_channels=z_channels * 3,
            skip_channels=skip_channels,
            out_channels=3,
            stage_extra_channels=(0, 0, 0),
            activation='sigmoid',
        )

        z_hw = input_size // 32
        self.z_spatial_size = (z_hw, z_hw)
        self.adapter_s_g = BottleneckAdapter(1, z_channels, self.z_spatial_size)
        self.adapter_s_c = BottleneckAdapter(3, z_channels, self.z_spatial_size)
        self.adapter_a_d = BottleneckAdapter(3, z_channels, self.z_spatial_size)

        num_classes = int(config.get('num_seg_classes', 41))
        self.ccr_encoder = CCREncoder(in_channels=6, channels=(64, 128, 256, 512))
        self.spade_c3 = SPADE(768, num_classes=num_classes)
        self.spade_c2 = SPADE(384, num_classes=num_classes)
        self.spade_c1 = SPADE(192, num_classes=num_classes)
        self.attn_c3 = ResidualAttention(768, 512)
        self.attn_c2 = ResidualAttention(384, 256)
        self.attn_c1 = ResidualAttention(192, 128)

        self.normal_encoder = NormalEncoder(in_channels=3, channels=(64, 128, 256, 512))
        self.attn_a3 = ResidualAttention(768, 512)
        self.attn_a2 = ResidualAttention(384, 256)
        self.attn_a1 = ResidualAttention(192, 128)
        self.attn_d3 = ResidualAttention(768, 512)
        self.attn_d2 = ResidualAttention(384, 256)
        self.attn_d1 = ResidualAttention(192, 128)

    def _validate_bottleneck_shape(self, z_global):
        _, _, h, w = z_global.shape
        if (h, w) != self.z_spatial_size:
            raise RuntimeError(
                f"V6 adapter bottleneck mismatch: got {(h, w)}, expected {self.z_spatial_size}. "
                "Set model.input_size to match training input_size."
            )

    def forward(self, rgb, m_diffuse=None, normals=None, seg=None,
                valid_mask=None, ccr=None, **kwargs):
        z_global, skip_features = self.image_encoder(rgb)
        self._validate_bottleneck_shape(z_global)

        if normals is None:
            normals = torch.zeros_like(rgb)
        normal_feats = self.normal_encoder(normals)

        d_g = self.decoder_a(
            z_global,
            skip_features,
            stage_ops=[
                lambda x: self.attn_a3(x, normal_feats[3]),
                lambda x: self.attn_a2(x, normal_feats[2]),
                lambda x: self.attn_a1(x, normal_feats[1]),
            ]
        )
        s_g = 1.0 / (d_g + 1e-6) - 1.0

        s_g_adapted = self.adapter_s_g(s_g)
        z_b = torch.cat([z_global, s_g_adapted], dim=1)
        xi = self.decoder_b(z_b, skip_features)

        eps = 1e-7
        c_rg = (1.0 - xi[:, 0:1]) / (xi[:, 0:1] + eps)
        c_bg = (1.0 - xi[:, 1:2]) / (xi[:, 1:2] + eps)
        c = torch.cat([c_rg, torch.ones_like(c_rg), c_bg], dim=1)
        s_c = s_g * c

        if ccr is None:
            ccr = compute_ccr(rgb)
        ccr_feats = self.ccr_encoder(ccr)

        s_c_adapted = self.adapter_s_c(s_c)
        z_c = torch.cat([z_global, s_c_adapted], dim=1)
        a_d = self.decoder_c(
            z_c,
            skip_features,
            stage_ops=[
                lambda x: self.spade_c3(self.attn_c3(x, ccr_feats[3]), seg),
                lambda x: self.spade_c2(self.attn_c2(x, ccr_feats[2]), seg),
                lambda x: self.spade_c1(self.attn_c1(x, ccr_feats[1]), seg),
            ]
        )

        a_d_adapted = self.adapter_a_d(a_d.detach())
        z_d = torch.cat([z_global, s_c_adapted, a_d_adapted], dim=1)
        pi = self.decoder_d(
            z_d,
            skip_features,
            stage_ops=[
                lambda x: self.attn_d3(x, normal_feats[3]),
                lambda x: self.attn_d2(x, normal_feats[2]),
                lambda x: self.attn_d1(x, normal_feats[1]),
            ]
        )

        return {
            'd_g': d_g,
            'xi': xi,
            'c': c,
            's_c': s_c,
            'a_d': a_d,
            'pi': pi,
        }
