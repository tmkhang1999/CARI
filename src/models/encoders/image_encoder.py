"""
Image encoder using ConvNeXt with partial fine-tuning.
Outputs global bottleneck Z and multi-scale skip features F_img_i.
"""

import torch
import torch.nn as nn
import timm


def _resolve_model_name(model_name):
    aliases = {
        # ConvNeXt V2 (default: Base)
        "convnextv2_large": "convnextv2_large.fcmae_ft_in22k_in1k",
        "convnextv2_base": "convnextv2_base.fcmae_ft_in22k_in1k",
        "convnextv2_small": "convnextv2_nano.fcmae_ft_in22k_in1k",
        "convnextv2_tiny": "convnextv2_tiny.fcmae_ft_in22k_in1k",
        # ConvNeXt V1 fallback
        "convnext_large": "convnext_large.fb_in22k_ft_in1k",
        "convnext_base": "convnext_base.fb_in22k_ft_in1k",
        "convnext_small": "convnext_small.fb_in22k_ft_in1k",
        "convnext_tiny": "convnext_tiny.fb_in22k_ft_in1k",
    }
    return aliases.get(model_name, model_name)


class ImageEncoder(nn.Module):
    def __init__(
        self,
        model_name="convnextv2_base",
        freeze_stages=[1, 2],
        pretrained=True,
    ):
        """
        Args:
            model_name: ConvNeXt model variant from timm or shorthand alias.
            freeze_stages: List of stages to freeze (1-indexed).
            pretrained: Whether to load pretrained weights.
        """
        super().__init__()

        resolved_name = _resolve_model_name(model_name)

        try:
            self.backbone = timm.create_model(
                resolved_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=(0, 1, 2, 3),
            )
        except Exception as exc:
            if pretrained:
                print(f"Warning: failed to load pretrained weights ({exc}); retrying without pretrained.")
                self.backbone = timm.create_model(
                    resolved_name,
                    pretrained=False,
                    features_only=True,
                    out_indices=(0, 1, 2, 3),
                    checkpoint_grad=True,  # for better OOM error messages if model is too large
                )
            else:
                raise

        # Channel counts depend on backbone variant (read dynamically from timm)
        # ConvNeXt V2 Base: [128, 256, 512, 1024]
        # ConvNeXt V2 Large: [192, 384, 768, 1536]
        # ConvNeXt Tiny:     [96, 192, 384, 768]
        self.feature_channels = self.backbone.feature_info.channels()

        # Freeze specified stages
        self._freeze_stages(freeze_stages)

    def _freeze_stages(self, freeze_stages):
        """
        Freeze parameters in specified stages (1-indexed).

        timm's features_only=True returns a FeatureListNet whose stage modules
        are stored as attributes named 'stages_0', 'stages_1', etc. — NOT as a
        list called 'stages'. We resolve them by name.
        """
        if not freeze_stages:
            return

        frozen = []
        for stage_num in freeze_stages:
            stage_idx = stage_num - 1   # 1-indexed → 0-indexed

            # Try attribute names used by timm FeatureListNet
            for attr in (f'stages_{stage_idx}',    # ConvNeXt V2 / ConvNeXt V1
                         f'layer{stage_num}',       # ResNet-style fallback
                         f'blocks_{stage_idx}'):    # other timm variants
                module = getattr(self.backbone, attr, None)
                if module is not None:
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False
                    frozen.append(attr)
                    break

        # Also freeze the stem / patch-embed when stage 1 is in freeze list
        if 1 in freeze_stages:
            for stem_attr in ('stem', 'patch_embed', 'downsample_layers_0'):
                stem = getattr(self.backbone, stem_attr, None)
                if stem is not None:
                    stem.eval()
                    for param in stem.parameters():
                        param.requires_grad = False
                    frozen.append(stem_attr)
                    break

        print(f"Frozen modules: {frozen}")

    def forward(self, x):
        """
        Args:
            x: (N, 3, H, W) RGB image

        Returns:
            z_global: (N, C, H/32, W/32) bottleneck features
            skip_features: List of 4 tensors at scales [H/4, H/8, H/16, H/32]
        """
        features = self.backbone(x)

        # features[0]: H/4   (Base: 128, Large: 192, Tiny: 96)
        # features[1]: H/8   (Base: 256, Large: 384, Tiny: 192)
        # features[2]: H/16  (Base: 512, Large: 768, Tiny: 384)
        # features[3]: H/32  (Base: 1024, Large: 1536, Tiny: 768) = z_global

        z_global = features[-1]
        skip_features = features

        return z_global, skip_features


if __name__ == '__main__':
    # Test
    encoder = ImageEncoder(freeze_stages=[1, 2])
    x = torch.randn(2, 3, 128, 384)
    z, skips = encoder(x)

    print(f"Z_global shape: {z.shape}")
    for i, skip in enumerate(skips):
        print(f"Skip {i} shape: {skip.shape}")
