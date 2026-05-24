"""Unified VGG-style prior encoder for V16.

Each stage: Conv3x3(stride=2) -> BN -> ReLU -> Conv3x3(stride=1) -> BN -> ReLU.
This matches the CCR and semantic encoder architecture used in PIE-Net and SIGNet exactly.

Replaces CCREncoder, NormalEncoder, and GuidanceEncoder for V16 only.
V12-V15 still use their own encoder files and are completely unaffected.

Output contract (same as V15 encoders so SFM dimensions remain unchanged):
  PriorEncoder with full_res_stage=True  (CCR, Normal use case):
    [0] H,    c1 ch  -- full-res features for edge head or skip connections
    [1] H/2,  c1 ch
    [2] H/4,  c2 ch
    [3] H/8,  c3 ch
    [4] H/16, c4 ch

  PriorEncoder with full_res_stage=False (Guidance use case):
    [0] H/2,  c1 ch
    [1] H/4,  c2 ch
    [2] H/8,  c3 ch
    [3] H/16, c4 ch
"""

import torch.nn as nn


def _vgg_block(in_c, out_c, stride):
    """Two-conv VGG block matching PIE-Net/SIGNet auxiliary encoder stages.

    First conv performs spatial downsampling (stride=2) and channel expansion.
    Second conv refines features at the new resolution (stride=1).
    """
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class PriorEncoder(nn.Module):
    """VGG-style 2-conv-per-stage prior encoder for V16.

    Args:
        in_channels: Number of input channels (6 for CCR, 3 for normals,
                     6-7 for guidance tensors).
        channels: Tuple of (c1, c2, c3, c4) output channels per stage.
                  Default (64, 128, 256, 512) matches V15 SFM prior_channels.
        full_res_stage: If True, prepend a stride-1 input projection block at
                        full resolution. Required for CCR (feeds ccr_edge_head)
                        and Normal encoders. Set False for guidance encoders
                        whose inputs are already downscaled to base_size.
    """

    def __init__(self, in_channels, channels=(64, 128, 256, 512), full_res_stage=True):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.full_res_stage = full_res_stage

        if full_res_stage:
            # Stride-1 input projection: preserves spatial resolution.
            # CCR: this output feeds ccr_edge_head for auxiliary edge supervision.
            # Normal: gives the network a full-res feature before downsampling.
            self.stage0 = _vgg_block(in_channels, c1, stride=1)
            stage1_in = c1
        else:
            # Guidance inputs are already at base_size -- start downsampling immediately.
            # Use a single projection conv to handle variable in_channels (6 or 7).
            self.stage0 = nn.Sequential(
                nn.Conv2d(in_channels, c1, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(c1),
                nn.ReLU(inplace=True),
            )
            stage1_in = c1

        # H -> H/2
        self.stage1 = _vgg_block(stage1_in, c1, stride=2)
        # H/2 -> H/4
        self.stage2 = _vgg_block(c1, c2, stride=2)
        # H/4 -> H/8
        self.stage3 = _vgg_block(c2, c3, stride=2)
        # H/8 -> H/16
        self.stage4 = _vgg_block(c3, c4, stride=2)

    def forward(self, x):
        """
        Returns:
            full_res_stage=True:  [f0(H,c1), f1(H/2,c1), f2(H/4,c2), f3(H/8,c3), f4(H/16,c4)]
            full_res_stage=False: [f1(H/2,c1), f2(H/4,c2), f3(H/8,c3), f4(H/16,c4)]

        The indexing matches V15's ccr_feats and normal_feats / guidance encoder
        outputs exactly, so no changes to SFM wiring are needed.
        """
        f0 = self.stage0(x)
        f1 = self.stage1(f0)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)

        if self.full_res_stage:
            return [f0, f1, f2, f3, f4]
        else:
            return [f1, f2, f3, f4]
