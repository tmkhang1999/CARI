"""SPADE-style physical prior injector with GroupNorm normalization.

Replaces SpatialFeatureModulation for V16/V17. The key difference from SFM is
GroupNorm before modulation: this standardizes the feature distribution so the
learned gamma/beta operate on a consistent scale regardless of spatial position
or training iteration. Without this, SFM's affine transform effectiveness
depends on absolute feature magnitudes, which vary unpredictably.

Supports optional segmentation embedding (fused with the spatial prior).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicalPriorInjector(nn.Module):
    """GroupNorm + SPADE-style affine modulation from spatial prior maps.

    Given decoder features x and a prior feature map (e.g. CCR, normals):
      1. Normalize x via GroupNorm (zero mean, unit variance per group)
      2. Predict spatially-varying gamma, beta from the prior
      3. Output = x_norm * (1 + gamma) + beta

    Initialization: gamma=0, beta=0 => identity at start.
    """

    def __init__(
        self,
        feature_channels,
        prior_channels,
        hidden_channels=128,
        num_groups=8,
        use_seg=False,
        num_seg_classes=41,
        normalize=True,
    ):
        super().__init__()
        self.use_seg = use_seg
        self.normalize = normalize

        # Normalization: standardize features before modulation
        if self.normalize:
            self.norm = nn.GroupNorm(
                num_groups=min(num_groups, feature_channels),
                num_channels=feature_channels,
                affine=False,
            )

        input_channels = prior_channels
        if use_seg:
            self.seg_embed = nn.Sequential(
                nn.Conv2d(num_seg_classes, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
            input_channels += 32

        self.prior_mlp = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv_gamma = nn.Conv2d(
            hidden_channels, feature_channels, kernel_size=3, padding=1
        )
        self.conv_beta = nn.Conv2d(
            hidden_channels, feature_channels, kernel_size=3, padding=1
        )

        # Identity initialization: gamma=0, beta=0 => output = x_norm
        nn.init.zeros_(self.conv_gamma.weight)
        nn.init.zeros_(self.conv_gamma.bias)
        nn.init.zeros_(self.conv_beta.weight)
        nn.init.zeros_(self.conv_beta.bias)

    def forward(self, x, prior, seg=None):
        x_norm = self.norm(x) if self.normalize else x

        if prior.shape[-2:] != x.shape[-2:]:
            prior = F.interpolate(
                prior, size=x.shape[-2:], mode="bilinear", align_corners=False
            )

        if self.use_seg and seg is not None:
            seg = seg.long()
            if seg.dim() == 4:
                seg = seg[:, 0]
            if seg.shape[-2:] != x.shape[-2:]:
                seg = (
                    F.interpolate(
                        seg.unsqueeze(1).float(), size=x.shape[-2:], mode="nearest"
                    )
                    .squeeze(1)
                    .long()
                )
            seg_oh = F.one_hot(seg.clamp(0, 40), 41).permute(0, 3, 1, 2).float()
            seg_feat = self.seg_embed(seg_oh)
            prior = torch.cat([prior, seg_feat], dim=1)

        h = self.prior_mlp(prior)
        gamma = self.conv_gamma(h)
        beta = self.conv_beta(h)

        return x_norm * (1.0 + gamma) + beta
