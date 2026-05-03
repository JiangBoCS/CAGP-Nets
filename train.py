import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

from models import FeatureCapturer, CAGPNet
from datasets import BaseDataset, NovelDataset
from utils import AverageMeter, save_checkpoint, load_checkpoint


def set_seed(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def charbonnier_loss(pred, target, eps=1e-6):
    """Charbonnier loss: sqrt(||pred - target||^2 + eps^2)"""
    diff = pred - target
    return torch.mean(torch.sqrt(diff * diff + eps * eps))


def train_stage1(args):
    """Stage 1: Self-supervised training of Feature Capturer."""
    print("=" * 60)
    print("Stage 1: Training Feature Capturer")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = BaseDataset(
        root_dir=args.base_dir,
        patch_size=args.patch_size,
        mask_ratio=args.mask_ratio,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True
    )

    model = FeatureCapturer(
        in_channels=args.in_channels, mid_channels=args.mid_channels
    ).to(device)

    # Kaiming initialization
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    start_epoch = 0
    if args.resume_stage1:
        start_epoch = load_checkpoint(args.resume_stage1, model, optimizer)

    for epoch in range(start_epoch, args.stage1_epochs):
        model.train()
        loss_meter = AverageMeter()

        for batch_idx, (masked_img, clean_img, mask) in enumerate(dataloader):
            masked_img = masked_img.to(device)
            clean_img = clean_img.to(device)

            # Reconstruct: I_r = F(I_m) + I_m
            output = model(masked_img) + masked_img

            # MSE loss (Eq. 7)
            loss = nn.functional.mse_loss(output, clean_img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), masked_img.size(0))

            if (batch_idx + 1) % args.print_freq == 0:
                print(f"  Epoch [{epoch+1}/{args.stage1_epochs}] "
                      f"Batch [{batch_idx+1}/{len(dataloader)}] "
                      f"Loss: {loss_meter.avg:.6f}")

        print(f"Epoch [{epoch+1}/{args.stage1_epochs}] Loss: {loss_meter.avg:.6f}")

        if (epoch + 1) % args.save_freq == 0 or (epoch + 1) == args.stage1_epochs:
            save_checkpoint(
                os.path.join(args.save_dir, f'feature_capturer_epoch{epoch+1}.pth'),
                model, optimizer, epoch + 1
            )

    # Save final model
    save_checkpoint(
        os.path.join(args.save_dir, 'feature_capturer_final.pth'),
        model, optimizer, args.stage1_epochs
    )
    print("Stage 1 training complete!")
    return model


def train_stage2(args, capturer=None):
    """Stage 2: Few-shot training of CAGP-Net with Feature Capturer features."""
    print("=" * 60)
    print("Stage 2: Training CAGP-Net")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Feature Capturer (frozen)
    if capturer is None:
        capturer = FeatureCapturer(
            in_channels=args.in_channels, mid_channels=args.mid_channels
        ).to(device)
        ckpt = torch.load(args.capturer_path, map_location=device)
        capturer.load_state_dict(ckpt['model_state_dict'])
    capturer.eval()
    for param in capturer.parameters():
        param.requires_grad = False

    # Dataset
    dataset = NovelDataset(
        clean_dir=args.novel_clean_dir,
        noisy_dir=args.novel_noisy_dir,
        patch_size=args.patch_size,
        noise_level=args.noise_level,
        K=args.K,
        seed=3407,
        is_real_noise=args.real_noise,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True
    )

    # CAGP-Net
    model = CAGPNet(
        in_channels=args.in_channels,
        mid_channels=args.mid_channels,
        num_cagp_blocks=args.num_cagp_blocks,
        patch_size=args.graph_patch_size,
        k=args.k_neighbors,
        num_clusters=args.num_clusters,
    ).to(device)

    # Kaiming initialization
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    start_epoch = 0
    if args.resume_stage2:
        start_epoch = load_checkpoint(args.resume_stage2, model, optimizer)

    for epoch in range(start_epoch, args.stage2_epochs):
        model.train()
        loss_meter = AverageMeter()

        for batch_idx, (noisy, clean) in enumerate(dataloader):
            noisy = noisy.to(device)
            clean = clean.to(device)

            # Extract features from Feature Capturer (frozen)
            with torch.no_grad():
                _, capturer_feats = capturer.forward_with_features(noisy)

            # Forward through CAGP-Net with injected features
            output = model(noisy, capturer_feats)

            # Charbonnier loss (Eq. 8)
            loss = charbonnier_loss(output, clean)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), noisy.size(0))

            if (batch_idx + 1) % args.print_freq == 0:
                print(f"  Epoch [{epoch+1}/{args.stage2_epochs}] "
                      f"Batch [{batch_idx+1}/{len(dataloader)}] "
                      f"Loss: {loss_meter.avg:.6f}")

        print(f"Epoch [{epoch+1}/{args.stage2_epochs}] Loss: {loss_meter.avg:.6f}")

        if (epoch + 1) % args.save_freq == 0 or (epoch + 1) == args.stage2_epochs:
            save_checkpoint(
                os.path.join(args.save_dir, f'cagp_net_K{args.K}_sigma{args.noise_level}_epoch{epoch+1}.pth'),
                model, optimizer, epoch + 1
            )

    save_checkpoint(
        os.path.join(args.save_dir, f'cagp_net_K{args.K}_sigma{args.noise_level}_final.pth'),
        model, optimizer, args.stage2_epochs
    )
    print("Stage 2 training complete!")


def main():
    parser = argparse.ArgumentParser(description='CAGP-Net Training')

    # Data paths
    parser.add_argument('--base_dir', type=str, default='./data/Flickr2K',
                        help='Path to base image dataset (Flickr2K)')
    parser.add_argument('--novel_clean_dir', type=str, default='./data/CBSD400/clean',
                        help='Path to novel clean images')
    parser.add_argument('--novel_noisy_dir', type=str, default=None,
                        help='Path to novel noisy images (for real noise)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')

    # Training parameters
    parser.add_argument('--stage1_epochs', type=int, default=400)
    parser.add_argument('--stage2_epochs', type=int, default=800)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=100)

    # Model parameters
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--mid_channels', type=int, default=64)
    parser.add_argument('--num_cagp_blocks', type=int, default=17)
    parser.add_argument('--graph_patch_size', type=int, default=8)
    parser.add_argument('--k_neighbors', type=int, default=12)
    parser.add_argument('--num_clusters', type=int, default=8)

    # Stage 1 parameters
    parser.add_argument('--mask_ratio', type=float, default=0.3)

    # Stage 2 parameters
    parser.add_argument('--noise_level', type=int, default=25)
    parser.add_argument('--K', type=int, default=20, help='Number of few-shot pairs')
    parser.add_argument('--real_noise', action='store_true')

    # Resume
    parser.add_argument('--resume_stage1', type=str, default=None)
    parser.add_argument('--resume_stage2', type=str, default=None)
    parser.add_argument('--capturer_path', type=str, default=None,
                        help='Path to pretrained Feature Capturer (skip stage 1)')

    # Mode
    parser.add_argument('--stage', type=int, default=0,
                        help='0: both stages, 1: stage1 only, 2: stage2 only')

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(3407)

    if args.stage == 0:
        capturer = train_stage1(args)
        train_stage2(args, capturer)
    elif args.stage == 1:
        train_stage1(args)
    elif args.stage == 2:
        if args.capturer_path is None:
            args.capturer_path = os.path.join(args.save_dir, 'feature_capturer_final.pth')
        train_stage2(args)


if __name__ == '__main__':
    main()
