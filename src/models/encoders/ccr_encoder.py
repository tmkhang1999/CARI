"""
CCR encoder using standard convolutions for boundary-rich guidance.
Architecture (plan Section 2.1):
  Conv3x3(6→64,  s=1) → BN → ReLU   (H,   64ch)  ← not returned
  Conv3x3(64→64, s=2) → BN → ReLU   (H/2, 64ch)  ← F_CCR_1
  Conv3x3(64→128,s=2) → BN → ReLU   (H/4, 128ch) ← F_CCR_2
  Conv3x3(128→256,s=2)→ BN → ReLU   (H/8, 256ch) ← F_CCR_3
  Conv3x3(256→512,s=2)→ BN → ReLU   (H/16,512ch) ← F_CCR_4

Input: 6-channel descriptor from compute_ccr():
  ch 0-2: clamped log cross-colour ratios (PIE-Net) — reflectance boundary edges
  ch 3-5: normalized RGB chromaticity (SIGNet)      — illumination-invariant hue

Standard conv (not DW) — CCR is spatially rich and needs full channel mixing (PIE-Net validated).
"""

import torch.nn as nn


def _conv_gn_gelu(in_c, out_c, stride):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
        # nn.GroupNorm(1, out_c),
        # nn.GELU()
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class CCREncoder(nn.Module):
    """Produces CCR features at [H/2, H/4, H/8, H/16] with channels [64, 128, 256, 512].

    Accepts 6-channel input: [log_M_RG, log_M_RB, log_M_GB, norm_R, norm_G, norm_B]
    as produced by compute_ccr(). Pass in_channels=3 for legacy 3-channel CCR maps.
    """

    def __init__(self, in_channels=6, channels=(64, 128, 256, 512)):
        super().__init__()
        c1, c2, c3, c4 = channels

        # Stage 0: full resolution, stride-1  (H → H)
        self.conv0 = _conv_gn_gelu(in_channels, c1, stride=1)
        # Stage 1: H → H/2
        self.conv1 = _conv_gn_gelu(c1, c1, stride=2)
        # Stage 2: H/2 → H/4
        self.conv2 = _conv_gn_gelu(c1, c2, stride=2)
        # Stage 3: H/4 → H/8
        self.conv3 = _conv_gn_gelu(c2, c3, stride=2)
        # Stage 4: H/8 → H/16
        self.conv4 = _conv_gn_gelu(c3, c4, stride=2)

    def forward(self, ccr):
        """
        Args:
            ccr: (N, 6, H, W) invariant descriptor from compute_ccr()
        Returns:
            List of 5 feature maps:
              [0] x0: H,    64ch  (full-res, stride-1) -- high-freq edge details
              [1] c1: H/2,  64ch
              [2] c2: H/4,  128ch
              [3] c3: H/8,  256ch
              [4] c4: H/16, 512ch
        """
        x0 = self.conv0(ccr)   # H,   64ch  (stride-1, full-res)
        c1 = self.conv1(x0)    # H/2, 64ch
        c2 = self.conv2(c1)    # H/4, 128ch
        c3 = self.conv3(c2)    # H/8, 256ch
        c4 = self.conv4(c3)    # H/16,512ch
        return [x0, c1, c2, c3, c4]


# Backward-compatible alias
CCLEncoder = CCREncoder
