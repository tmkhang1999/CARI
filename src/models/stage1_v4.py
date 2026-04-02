"""Stage 1 V4: Version-1 baseline + segmentation-guided SPADE pyramids in Dec C."""

import torch

from .stage1_v1 import IntrinsicDecompositionV1
from .modules.spade import SPADE
from .decoders.progressive_decoder import ProgressiveDecoder


class IntrinsicDecompositionV4(IntrinsicDecompositionV1):
    def __init__(self, config):
        super().__init__(config)
        z_channels = config.get('z_channels', 1024)
        num_classes = config.get('num_seg_classes', 41)
        
        # Replace Decoder C with ProgressiveDecoder to support stage_ops (pyramid injection)
        skip_channels = self.image_encoder.feature_channels
        self.decoder_c = ProgressiveDecoder(
            in_channels=z_channels * 2,  # z_global + s_c_adapted
            skip_channels=skip_channels,
            out_channels=3,
            stage_extra_channels=(0, 0, 0),
            activation="sigmoid",
        )
        
        # SPADE Pyramids applied after dec3, dec2, and dec1
        self.spade_c3 = SPADE(768, num_classes=num_classes)
        self.spade_c2 = SPADE(384, num_classes=num_classes)
        self.spade_c1 = SPADE(192, num_classes=num_classes)

    def forward(self, rgb, m_diffuse=None, normals=None, seg=None,
                valid_mask=None, ccr=None, **kwargs):
        z_global, skip_features = self.image_encoder(rgb)
        self._validate_bottleneck_shape(z_global)

        # Dec A
        d_g = self.decoder_a(z_global, skip_features)
        s_g = 1.0 / (d_g + 1e-6) - 1.0

        # Dec B
        s_g_adapted = self.adapter_s_g(s_g)
        z_b = torch.cat([z_global, s_g_adapted], dim=1)
        xi = self.decoder_b(z_b, skip_features)

        # Convert xi to chroma map C
        eps = 1e-7
        c_rg = (1.0 - xi[:, 0:1]) / (xi[:, 0:1] + eps)
        c_bg = (1.0 - xi[:, 1:2]) / (xi[:, 1:2] + eps)
        c = torch.cat([c_rg, torch.ones_like(c_rg), c_bg], dim=1)
        s_c = s_g * c

        # Dec C: segmentation-guided SPADE pyramids
        s_c_adapted = self.adapter_s_c(s_c)
        z_c = torch.cat([z_global, s_c_adapted], dim=1)
        a_d = self.decoder_c(
            z_c, 
            skip_features,
            stage_ops=[
                lambda x: self.spade_c3(x, seg),
                lambda x: self.spade_c2(x, seg),
                lambda x: self.spade_c1(x, seg),
            ]
        )

        # Dec D
        a_d_adapted = self.adapter_a_d(a_d.detach())
        z_d = torch.cat([z_global, s_c_adapted, a_d_adapted], dim=1)
        pi = self.decoder_d(z_d, skip_features)

        return {
            'd_g': d_g,
            'xi': xi,
            'c': c,
            's_c': s_c,
            'a_d': a_d,
            'pi': pi,
        }

