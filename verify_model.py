"""Quick verification that models can do a forward pass."""
import torch
from models import FeatureCapturer, CAGPNet


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Test Feature Capturer
    print("\n--- Feature Capturer ---")
    MID_CH = 64
    capturer = FeatureCapturer(in_channels=3, mid_channels=MID_CH, num_blocks=18).to(device)
    print(f"Parameters: {count_parameters(capturer) / 1e6:.3f}M")

    x = torch.randn(2, 3, 128, 128).to(device)
    out, feats = capturer.forward_with_features(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Number of intermediate features: {len(feats)}")
    print(f"Feature shapes: {feats[0].shape}")

    # Test CAGP-Net
    print("\n--- CAGP-Net ---")
    model = CAGPNet(
        in_channels=3, mid_channels=MID_CH, num_cagp_blocks=17,
        patch_size=8, k=12, num_clusters=8
    ).to(device)
    print(f"Parameters: {count_parameters(model) / 1e6:.3f}M")

    noisy = torch.randn(2, 3, 128, 128).to(device)
    with torch.no_grad():
        _, capturer_feats = capturer.forward_with_features(noisy)
    output = model(noisy, capturer_feats)
    print(f"Input: {noisy.shape}")
    print(f"Output: {output.shape}")

    # Test backward pass
    loss = output.mean()
    loss.backward()
    print("\nBackward pass: OK")

    # Test without capturer features
    output2 = model(noisy)
    print(f"Output (no capturer features): {output2.shape}")

    print("\n=== All checks passed! ===")


if __name__ == '__main__':
    main()
