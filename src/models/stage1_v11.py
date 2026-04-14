"""Stage 1 V11: Physics-driven dual-branch architecture predicting only Albedo and Shading."""

import torch
import torch.nn as nn

from .encoders.image_encoder import ImageEncoder
from .encoders.normal_encoder import NormalEncoder
from .encoders.guidance_encoder import GuidanceEncoder
from .encoders.ccr_encoder import CCREncoder
from .decoders.progressive_decoder import ProgressiveDecoder
from .modules.spatial_feature_modulation import SpatialFeatureModulation
from .modules.spade import SPADE
from .stage1_v5 import compute_ccr

class IntrinsicDecompositionV11(nn.Module):
    def __init__(self, config):
        super().__init__()

        z_channels = config.get('z_channels', 1024)
        freeze_stages = config.get('freeze_stages', [1, 2])
        model_name = config.get('backbone', config.get('model_name', 'convnextv2_base'))
        pretrained = config.get('pretrained', True)
        num_seg_classes = int(config.get('num_seg_classes', 41))

        self.image_encoder = ImageEncoder(
            model_name=model_name,
            freeze_stages=freeze_stages,
            pretrained=pretrained,
        )
        skip_channels = self.image_encoder.feature_channels

        self.normal_encoder = NormalEncoder(in_channels=3, channels=(64, 128, 256, 512))
        self.ccr_encoder = CCREncoder(in_channels=6, channels=(64, 128, 256, 512))

        # We only need ONE Guidance Encoder now (to feed Albedo into Shading)
        self.guidance_d = GuidanceEncoder(in_channels=3, channels=(64, 128, 256, 512))

        self.decoder_c = ProgressiveDecoder(
            in_channels=z_channels,
            skip_channels=skip_channels,
            out_channels=3,
            stage_extra_channels=(0, 0, 0),
            activation='sigmoid',
        )

        self.decoder_d = ProgressiveDecoder(
            in_channels=z_channels,
            skip_channels=skip_channels,
            out_channels=3,
            stage_extra_channels=(0, 0, 0),
            activation='sigmoid',
        )

        # Decoder C (Albedo) GUIDANCE
        self.sfm_c_ccr3 = SpatialFeatureModulation(768, 512)
        self.sfm_c_ccr2 = SpatialFeatureModulation(384, 256)
        self.spade_c2 = SPADE(num_channels=384, num_classes=num_seg_classes)
        self.sfm_c_ccr1 = SpatialFeatureModulation(192, 128)
        self.spade_c1 = SPADE(num_channels=192, num_classes=num_seg_classes)

        # Decoder D (Shading) GUIDANCE
        self.sfm_d_n3 = SpatialFeatureModulation(768, 512)
        self.sfm_d_a3 = SpatialFeatureModulation(768, 512)
        self.sfm_d_n2 = SpatialFeatureModulation(384, 256)
        self.sfm_d_a2 = SpatialFeatureModulation(384, 256)
        self.sfm_d_n1 = SpatialFeatureModulation(192, 128)
        self.sfm_d_a1 = SpatialFeatureModulation(192, 128)

    def forward(self, rgb, normals=None, seg=None, ccr=None, valid_mask=None, **kwargs):
        z_global, skip_features = self.image_encoder(rgb)
        
        if normals is None:
            normals = torch.zeros_like(rgb)
        normal_feats = self.normal_encoder(normals)
        
        if ccr is None:
            ccr = compute_ccr(rgb)
        ccr_feats = self.ccr_encoder(ccr)

        # 1. Predict Albedo FIRST (Guided purely by CCR boundaries and Semantics)
        a_d = self.decoder_c(
            z_global, skip_features,
            stage_ops=[
                lambda x: self.sfm_c_ccr3(x, ccr_feats[3]),
                lambda x: self.spade_c2(self.sfm_c_ccr2(x, ccr_feats[2]), seg),
                lambda x: self.spade_c1(self.sfm_c_ccr1(x, ccr_feats[1]), seg),
            ]
        )

        # 2. Predict Shading SECOND (Guided by Normals and the Predicted Albedo)
        g_d = self.guidance_d(a_d.detach())

        pi = self.decoder_d(
            z_global, skip_features,
            stage_ops=[
                lambda x: self.sfm_d_a3(self.sfm_d_n3(x, normal_feats[3]), g_d[3]),
                lambda x: self.sfm_d_a2(self.sfm_d_n2(x, normal_feats[2]), g_d[2]),
                lambda x: self.sfm_d_a1(self.sfm_d_n1(x, normal_feats[1]), g_d[1]),
            ]
        )

        return {'a_d': a_d, 'pi': pi}
