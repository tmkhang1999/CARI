"""
Decoder modules for intrinsic decomposition cascade.
Each decoder follows: deconv -> (attention) -> (SPADE) -> concat skips -> repeat
"""

import torch
import torch.nn as nn


class DecoderBlock(nn.Module):
    """
    Single decoder block: ConvTranspose -> BN -> ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # nn.GroupNorm(1, out_channels),
            # nn.GELU()
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DecoderA(nn.Module):
    """
    Decoder A: Predicts inverse gray shading D_g (1 channel).
    Input: Z_global + F_img skips
    Version 1: No attention, raw skip concat
    """
    def __init__(self, z_channels=1536, skip_channels=[192, 384, 768, 1536]):
        super().__init__()

        # Build 4-stage decoder: 1536 -> 768 -> 384 -> 192 -> 96
        # Stage 3: from bottleneck
        self.dec3 = DecoderBlock(z_channels, 768)

        # Stage 2
        in_ch2 = 768 + skip_channels[2]  # dec output + skip
        self.dec2 = DecoderBlock(in_ch2, 384)

        # Stage 1
        in_ch1 = 384 + skip_channels[1]
        self.dec1 = DecoderBlock(in_ch1, 192)

        # Stage 0
        in_ch0 = 192 + skip_channels[0]
        self.dec0 = DecoderBlock(in_ch0, 96)

        # Final prediction head
        self.head = nn.Sequential(
            nn.Conv2d(96, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()  # Inverse shading D_g in (0, 1)
        )

    def forward(self, z, skip_features):
        """
        Args:
            z: (N, z_channels, H/32, W/32) bottleneck
            skip_features: List of 4 tensors [H/4, H/8, H/16, H/32]
        Returns:
            d_g: (N, 1, H, W) inverse gray shading prediction
        """
        # Stage 3: H/32 -> H/16
        x = self.dec3(z)
        x = torch.cat([x, skip_features[2]], dim=1)

        # Stage 2: H/16 -> H/8
        x = self.dec2(x)
        x = torch.cat([x, skip_features[1]], dim=1)

        # Stage 1: H/8 -> H/4
        x = self.dec1(x)
        x = torch.cat([x, skip_features[0]], dim=1)

        # Stage 0: H/4 -> H
        x = self.dec0(x)

        # Upsample to full resolution
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        # Predict
        d_g = self.head(x)

        return d_g


class DecoderB(nn.Module):
    """
    Decoder B: Predicts chroma C (2 channels as bounded ratio xi).
    Input: Z_global + F_img skips + S_g_adapted
    Version 1: Bottleneck adapter only, raw skip concat
    """
    def __init__(self, z_channels=1536, skip_channels=[192, 384, 768, 1536], adapter_channels=None):
        super().__init__()

        # In V1, S_g is adapted to bottleneck and concatenated to Z
        # So first stage input = z_channels + z_channels (adapted)
        in_z = z_channels * 2 if adapter_channels is None else z_channels + adapter_channels

        # Build 4-stage decoder
        self.dec3 = DecoderBlock(in_z, 768)

        in_ch2 = 768 + skip_channels[2]
        self.dec2 = DecoderBlock(in_ch2, 384)

        in_ch1 = 384 + skip_channels[1]
        self.dec1 = DecoderBlock(in_ch1, 192)

        in_ch0 = 192 + skip_channels[0]
        self.dec0 = DecoderBlock(in_ch0, 96)

        # Final prediction head for bounded chroma xi
        self.head = nn.Sequential(
            nn.Conv2d(96, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=1),
            nn.Sigmoid()  # Bounded in (0, 1)
        )

    def forward(self, z_combined, skip_features):
        """
        Args:
            z_combined: (N, z_channels*2, H/32, W/32) z + adapted S_g
            skip_features: List of 4 tensors
        Returns:
            xi: (N, 2, H, W) bounded chroma ratio [1/(C_R/G+1), 1/(C_B/G+1)]
        """
        # Stage 3
        x = self.dec3(z_combined)
        x = torch.cat([x, skip_features[2]], dim=1)

        # Stage 2
        x = self.dec2(x)
        x = torch.cat([x, skip_features[1]], dim=1)

        # Stage 1
        x = self.dec1(x)
        x = torch.cat([x, skip_features[0]], dim=1)

        # Stage 0
        x = self.dec0(x)

        # Upsample to full resolution
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        # Predict bounded chroma
        xi = self.head(x)

        return xi


class DecoderC(nn.Module):
    """
    Decoder C: Predicts diffuse albedo A_d (3 channels).
    Input: Z_global + F_img skips + S_c_adapted
    Version 1: Bottleneck adapter only, no CCR, no SPADE
    """
    def __init__(self, z_channels=1536, skip_channels=[192, 384, 768, 1536]):
        super().__init__()

        # In V1, S_c is adapted to bottleneck
        in_z = z_channels * 2

        # Build decoder
        self.dec3 = DecoderBlock(in_z, 768)

        in_ch2 = 768 + skip_channels[2]
        self.dec2 = DecoderBlock(in_ch2, 384)

        in_ch1 = 384 + skip_channels[1]
        self.dec1 = DecoderBlock(in_ch1, 192)

        in_ch0 = 192 + skip_channels[0]
        self.dec0 = DecoderBlock(in_ch0, 96)

        # Final prediction head
        self.head = nn.Sequential(
            nn.Conv2d(96, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Sigmoid()  # Albedo in [0, 1]
        )

    def forward(self, z_combined, skip_features):
        """
        Args:
            z_combined: (N, z_channels*2, H/32, W/32) z + adapted S_c
            skip_features: List of 4 tensors
        Returns:
            a_d: (N, 3, H, W) diffuse albedo
        """
        # Stage 3
        x = self.dec3(z_combined)
        x = torch.cat([x, skip_features[2]], dim=1)

        # Stage 2
        x = self.dec2(x)
        x = torch.cat([x, skip_features[1]], dim=1)

        # Stage 1
        x = self.dec1(x)
        x = torch.cat([x, skip_features[0]], dim=1)

        # Stage 0
        x = self.dec0(x)

        # Upsample
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        # Predict
        a_d = self.head(x)

        return a_d


class DecoderD(nn.Module):
    """
    Decoder D: Predicts inverse diffuse shading pi (3 channels, bounded).
    Input: Z_global + F_img skips + S_c_adapted + A_d_adapted
    Version 1: Bottleneck adapters only
    """
    def __init__(self, z_channels=1536, skip_channels=[192, 384, 768, 1536]):
        super().__init__()

        # In V1, S_c and A_d are both adapted to bottleneck
        in_z = z_channels * 3  # z + S_c + A_d

        # Build decoder
        self.dec3 = DecoderBlock(in_z, 768)

        in_ch2 = 768 + skip_channels[2]
        self.dec2 = DecoderBlock(in_ch2, 384)

        in_ch1 = 384 + skip_channels[1]
        self.dec1 = DecoderBlock(in_ch1, 192)

        in_ch0 = 192 + skip_channels[0]
        self.dec0 = DecoderBlock(in_ch0, 96)

        # Final prediction head for inverse diffuse shading
        self.head = nn.Sequential(
            nn.Conv2d(96, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Sigmoid()  # Inverse shading pi in (0, 1)
        )

    def forward(self, z_combined, skip_features):
        """
        Args:
            z_combined: (N, z_channels*3, H/32, W/32) z + adapted S_c + adapted A_d
            skip_features: List of 4 tensors
        Returns:
            pi: (N, 3, H, W) inverse diffuse shading pi in (0, 1)
        """
        # Stage 3
        x = self.dec3(z_combined)
        x = torch.cat([x, skip_features[2]], dim=1)

        # Stage 2
        x = self.dec2(x)
        x = torch.cat([x, skip_features[1]], dim=1)

        # Stage 1
        x = self.dec1(x)
        x = torch.cat([x, skip_features[0]], dim=1)

        # Stage 0
        x = self.dec0(x)

        # Upsample
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        # Predict
        pi = self.head(x)

        return pi


if __name__ == '__main__':
    # Test decoders (ConvNeXt V2 Base channels)
    z = torch.randn(2, 1024, 12, 12)
    skips = [
        torch.randn(2, 128, 96, 96),
        torch.randn(2, 256, 48, 48),
        torch.randn(2, 512, 24, 24),
        torch.randn(2, 1024, 12, 12)
    ]

    dec_a = DecoderA()
    d_g = dec_a(z, skips)
    print(f"Dec A output (D_g): {d_g.shape}")

    z_b = torch.randn(2, 2048, 12, 12)  # z + adapted S_g
    dec_b = DecoderB()
    xi = dec_b(z_b, skips)
    print(f"Dec B output (xi): {xi.shape}")
