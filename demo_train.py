"""
Quick demo: creates synthetic data and runs a few training iterations
to verify the full pipeline works end-to-end.
"""
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image

from models import FeatureCapturer, CAGPNet
from utils import compute_psnr


class SyntheticDataset(Dataset):
    """Generate synthetic image pairs on-the-fly for testing."""

    def __init__(self, num_samples=32, patch_size=128, noise_level=25, channels=3):
        self.num_samples = num_samples
        self.patch_size = patch_size
        self.noise_level = noise_level
        self.channels = channels

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        torch.manual_seed(idx + 1000)
        clean = torch.rand(self.channels, self.patch_size, self.patch_size)
        noise = torch.randn_like(clean) * (self.noise_level / 255.0)
        noisy = (clean + noise).clamp(0, 1)
        return noisy, clean


class SyntheticBaseDataset(Dataset):
    """Generate synthetic clean images with masks for Stage 1."""

    def __init__(self, num_samples=64, patch_size=128, mask_ratio=0.3,
                 mask_patch_size=16, channels=3):
        self.num_samples = num_samples
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.mask_patch_size = mask_patch_size
        self.channels = channels

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        torch.manual_seed(idx)
        img = torch.rand(self.channels, self.patch_size, self.patch_size)

        # Generate mask
        ps = self.patch_size
        mps = self.mask_patch_size
        num_patches = (ps // mps) ** 2
        num_masked = int(num_patches * self.mask_ratio)

        mask = torch.ones(1, ps, ps)
        g = torch.Generator().manual_seed(idx + 9999)
        perm = torch.randperm(num_patches, generator=g)
        masked_indices = perm[:num_masked]

        patches_per_row = ps // mps
        for mi in masked_indices:
            row = mi // patches_per_row
            col = mi % patches_per_row
            mask[:, row * mps:(row + 1) * mps, col * mps:(col + 1) * mps] = 0

        masked_img = img * mask
        return masked_img, img, mask


def charbonnier_loss(pred, target, eps=1e-6):
    diff = pred - target
    return torch.mean(torch.sqrt(diff * diff + eps * eps))


def demo():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("=" * 60)
    print("CAGP-Net Demo: Full Pipeline Verification")
    print("=" * 60)

    in_channels = 3
    mid_channels = 64
    patch_size = 64  # smaller for demo speed

    # ===== Stage 1: Feature Capturer =====
    print("\n[Stage 1] Training Feature Capturer...")
    capturer = FeatureCapturer(in_channels=in_channels, mid_channels=mid_channels).to(device)
    optimizer1 = AdamW(capturer.parameters(), lr=1e-3)

    base_dataset = SyntheticBaseDataset(
        num_samples=32, patch_size=patch_size, channels=in_channels
    )
    base_loader = DataLoader(base_dataset, batch_size=4, shuffle=True)

    capturer.train()
    for epoch in range(5):
        total_loss = 0
        for masked_img, clean_img, mask in base_loader:
            masked_img = masked_img.to(device)
            clean_img = clean_img.to(device)

            output = capturer(masked_img) + masked_img
            loss = nn.functional.mse_loss(output, clean_img)

            optimizer1.zero_grad()
            loss.backward()
            optimizer1.step()
            total_loss += loss.item()

        print(f"  Epoch {epoch+1}/5 - Loss: {total_loss/len(base_loader):.6f}")

    # ===== Stage 2: CAGP-Net =====
    print("\n[Stage 2] Training CAGP-Net...")
    capturer.eval()
    for p in capturer.parameters():
        p.requires_grad = False

    model = CAGPNet(
        in_channels=in_channels, mid_channels=mid_channels,
        num_cagp_blocks=4,  # fewer blocks for demo speed
        patch_size=8, k=8, num_clusters=4
    ).to(device)
    optimizer2 = AdamW(model.parameters(), lr=1e-3)

    novel_dataset = SyntheticDataset(
        num_samples=16, patch_size=patch_size, noise_level=25, channels=in_channels
    )
    novel_loader = DataLoader(novel_dataset, batch_size=4, shuffle=True)

    model.train()
    for epoch in range(10):
        total_loss = 0
        for noisy, clean in novel_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            with torch.no_grad():
                _, capturer_feats = capturer.forward_with_features(noisy)

            output = model(noisy, capturer_feats)
            loss = charbonnier_loss(output, clean)

            optimizer2.zero_grad()
            loss.backward()
            optimizer2.step()
            total_loss += loss.item()

        print(f"  Epoch {epoch+1}/10 - Loss: {total_loss/len(novel_loader):.6f}")

    # ===== Evaluate =====
    print("\n[Evaluation]")
    model.eval()
    psnr_noisy_list = []
    psnr_denoised_list = []

    with torch.no_grad():
        for noisy, clean in novel_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            _, capturer_feats = capturer.forward_with_features(noisy)
            output = model(noisy, capturer_feats).clamp(0, 1)

            for i in range(noisy.shape[0]):
                psnr_noisy_list.append(compute_psnr(noisy[i:i+1], clean[i:i+1]))
                psnr_denoised_list.append(compute_psnr(output[i:i+1], clean[i:i+1]))

    avg_noisy = np.mean(psnr_noisy_list)
    avg_denoised = np.mean(psnr_denoised_list)
    print(f"  Average PSNR (noisy):    {avg_noisy:.2f} dB")
    print(f"  Average PSNR (denoised): {avg_denoised:.2f} dB")
    print(f"  Improvement: {avg_denoised - avg_noisy:+.2f} dB")

    print("\n" + "=" * 60)
    print("Demo complete! Full pipeline works correctly.")
    print("=" * 60)


if __name__ == '__main__':
    demo()
