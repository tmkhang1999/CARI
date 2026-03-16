"""
Adapter modules for cross-decoder communication.
"""

import torch
import torch.nn as nn


class BottleneckAdapter(nn.Module):
    """
    Version 1 adapter: Encodes full-resolution prediction into Z bottleneck space.
    Used to inject S_g/S_c/A_d into downstream decoders.
    """
    def __init__(self, in_channels, z_channels, z_spatial_size):
        """
        Args:
            in_channels: Input channels (1 for S_g, 3 for S_c/A_d)
            z_channels: Target bottleneck channels (e.g., 1536 for ConvNeXt Large)
            z_spatial_size: Target spatial size tuple (H, W)
        """
        super().__init__()
        self.z_spatial_size = z_spatial_size

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.pool = nn.AdaptiveAvgPool2d(z_spatial_size)
        self.proj = nn.Conv2d(128, z_channels, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: (N, C_in, H, W) full-resolution prediction
        Returns:
            (N, z_channels, z_H, z_W) bottleneck features
        """
        x = self.encoder(x)
        x = self.pool(x)
        x = self.proj(x)
        return x


class ShadingAdapter(nn.Module):
    """
    Version 2+ adapter: Encodes S_g into multi-scale features for Dec B.
    """
    def __init__(self):
        super().__init__()

        # Build pyramid: 1 -> 32 -> 64 -> 128 -> 256 channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

    def forward(self, s_g):
        """
        Args:
            s_g: (N, 1, H, W) gray shading prediction
        Returns:
            List of 4 tensors at scales [H/2, H/4, H/8, H/16]
        """
        s_g_1 = self.conv1(s_g)
        s_g_2 = self.conv2(s_g_1)
        s_g_3 = self.conv3(s_g_2)
        s_g_4 = self.conv4(s_g_3)

        return [s_g_1, s_g_2, s_g_3, s_g_4]


class ColorfulAdapter(nn.Module):
    """
    Version 2+ adapter: Encodes S_c into multi-scale features for Dec C/D.
    """
    def __init__(self):
        super().__init__()

        # Build pyramid: 3 -> 64 -> 128 -> 256 -> 512 channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

    def forward(self, s_c):
        """
        Args:
            s_c: (N, 3, H, W) colorful shading prediction
        Returns:
            List of 4 tensors at scales [H/2, H/4, H/8, H/16]
        """
        s_c_1 = self.conv1(s_c)
        s_c_2 = self.conv2(s_c_1)
        s_c_3 = self.conv3(s_c_2)
        s_c_4 = self.conv4(s_c_3)

        return [s_c_1, s_c_2, s_c_3, s_c_4]


class AlbedoAdapter(nn.Module):
    """
    Version 2+ adapter: Encodes A_d into multi-scale features for Dec D.
    """
    def __init__(self):
        super().__init__()

        # Build pyramid: 3 -> 64 -> 128 -> 256 -> 512 channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

    def forward(self, a_d):
        """
        Args:
            a_d: (N, 3, H, W) diffuse albedo prediction
        Returns:
            List of 4 tensors at scales [H/2, H/4, H/8, H/16]
        """
        a_d_1 = self.conv1(a_d)
        a_d_2 = self.conv2(a_d_1)
        a_d_3 = self.conv3(a_d_2)
        a_d_4 = self.conv4(a_d_3)

        return [a_d_1, a_d_2, a_d_3, a_d_4]


if __name__ == '__main__':
    # Test BottleneckAdapter
    adapter = BottleneckAdapter(in_channels=1, z_channels=1536, z_spatial_size=(12, 12))
    s_g = torch.randn(2, 1, 384, 384)
    z_adapted = adapter(s_g)
    print(f"BottleneckAdapter output: {z_adapted.shape}")

    # Test ShadingAdapter
    shading_adapter = ShadingAdapter()
    s_g_pyramid = shading_adapter(s_g)
    print(f"ShadingAdapter outputs: {[f.shape for f in s_g_pyramid]}")

