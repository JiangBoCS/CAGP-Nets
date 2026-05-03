import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.conv(x))


class FeatureCapturer(nn.Module):
    """
    Stage 1: Self-supervised feature capturer inspired by MAE.
    Uses masked image reconstruction with long-distance residual design.
    Architecture: 18 Conv Blocks + 1 final conv layer.
    """

    def __init__(self, in_channels=3, mid_channels=32, num_blocks=18):
        super().__init__()
        layers = []
        layers.append(ConvBlock(in_channels, mid_channels))
        for _ in range(num_blocks - 1):
            layers.append(ConvBlock(mid_channels, mid_channels))
        self.blocks = nn.Sequential(*layers)
        self.final_conv = nn.Conv2d(mid_channels, in_channels, 3, padding=1)

    def forward(self, x_masked):
        feat = self.blocks(x_masked)
        out = self.final_conv(feat)
        return out

    def forward_with_features(self, x_masked):
        """Return intermediate features for injection into CAGP-Net."""
        features = []
        x = x_masked
        for block in self.blocks:
            x = block(x)
            features.append(x)
        out = self.final_conv(x)
        return out, features

    def reconstruct(self, x_masked):
        """Full reconstruction with residual: I_r = F(I_m) + I_m"""
        return self.forward(x_masked) + x_masked
