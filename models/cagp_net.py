import torch
import torch.nn as nn

from .feature_capturer import ConvBlock
from .cagp_block import CAGPBlock


class CAGPNet(nn.Module):
    """
    Clustered Adaptive Graph Priors Network for Few-shot Image Denoising.
    Architecture: 1 Conv Block + 17 CAGP Blocks + 1 final conv layer.
    """

    def __init__(self, in_channels=3, mid_channels=32, num_cagp_blocks=17,
                 patch_size=4, k=12, num_clusters=8):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels

        # Initial Conv Block
        self.init_block = ConvBlock(in_channels, mid_channels)

        # 17 CAGP Blocks
        self.cagp_blocks = nn.ModuleList([
            CAGPBlock(channels=mid_channels, patch_size=patch_size,
                      k=k, num_clusters=num_clusters)
            for _ in range(num_cagp_blocks)
        ])

        # Final conv layer
        self.final_conv = nn.Conv2d(mid_channels, in_channels, 3, padding=1)

    def forward(self, x, capturer_features=None):
        """
        Args:
            x: (B, C, H, W) - noisy input image
            capturer_features: list of (B, mid_channels, H, W) features from Feature Capturer
                             One feature per block (18 total: 1 init + 17 CAGP)
        Returns:
            denoised: (B, C, H, W)
        """
        # Initial Conv Block
        feat = self.init_block(x)

        # Inject first capturer feature if available
        if capturer_features is not None and len(capturer_features) > 0:
            feat = feat + capturer_features[0]

        # 17 CAGP Blocks with feature injection
        for i, cagp_block in enumerate(self.cagp_blocks):
            feat = cagp_block(feat)
            # Inject capturer feature (offset by 1 for the init block)
            if capturer_features is not None and (i + 1) < len(capturer_features):
                feat = feat + capturer_features[i + 1]

        # Final conv
        out = self.final_conv(feat)

        return out
