import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms.functional as TF

from models import FeatureCapturer, CAGPNet
from datasets import TestDataset
from utils import compute_psnr, compute_ssim


def test(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load Feature Capturer
    capturer = FeatureCapturer(
        in_channels=args.in_channels, mid_channels=args.mid_channels
    ).to(device)
    ckpt = torch.load(args.capturer_path, map_location=device)
    capturer.load_state_dict(ckpt['model_state_dict'])
    capturer.eval()

    # Load CAGP-Net
    model = CAGPNet(
        in_channels=args.in_channels,
        mid_channels=args.mid_channels,
        num_cagp_blocks=args.num_cagp_blocks,
        patch_size=args.graph_patch_size,
        k=args.k_neighbors,
        num_clusters=args.num_clusters,
    ).to(device)
    ckpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Dataset
    dataset = TestDataset(
        clean_dir=args.test_dir,
        noise_level=args.noise_level,
        is_real_noise=args.real_noise,
        noisy_dir=args.test_noisy_dir,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    psnr_list = []
    ssim_list = []

    os.makedirs(args.output_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (noisy, clean, fname) in enumerate(dataloader):
            noisy = noisy.to(device)
            clean = clean.to(device)

            # Ensure dimensions are divisible by graph_patch_size
            B, C, H, W = noisy.shape
            ps = args.graph_patch_size
            pad_h = (ps - H % ps) % ps
            pad_w = (ps - W % ps) % ps
            if pad_h > 0 or pad_w > 0:
                noisy_padded = torch.nn.functional.pad(noisy, (0, pad_w, 0, pad_h), mode='reflect')
            else:
                noisy_padded = noisy

            # Extract capturer features
            _, capturer_feats = capturer.forward_with_features(noisy_padded)

            # Denoise
            output = model(noisy_padded, capturer_feats)

            # Remove padding
            if pad_h > 0 or pad_w > 0:
                output = output[:, :, :H, :W]

            output = output.clamp(0, 1)

            # Compute metrics
            psnr = compute_psnr(output, clean)
            ssim = compute_ssim(output, clean)
            psnr_list.append(psnr)
            ssim_list.append(ssim)

            print(f"  [{idx+1}/{len(dataset)}] {fname[0]} - PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")

            # Save denoised image
            if args.save_images:
                out_img = output.squeeze(0).cpu()
                out_img = TF.to_pil_image(out_img)
                out_img.save(os.path.join(args.output_dir, fname[0]))

    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    print(f"\n{'='*60}")
    print(f"Results on {args.test_dir}:")
    print(f"  Average PSNR: {avg_psnr:.2f} dB")
    print(f"  Average SSIM: {avg_ssim:.4f}")
    print(f"  Noise level: {args.noise_level}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='CAGP-Net Testing')

    parser.add_argument('--test_dir', type=str, required=True,
                        help='Path to test clean images')
    parser.add_argument('--test_noisy_dir', type=str, default=None,
                        help='Path to test noisy images (for real noise)')
    parser.add_argument('--capturer_path', type=str, required=True,
                        help='Path to Feature Capturer checkpoint')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to CAGP-Net checkpoint')
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--save_images', action='store_true')

    # Model parameters (must match training)
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--mid_channels', type=int, default=64)
    parser.add_argument('--num_cagp_blocks', type=int, default=17)
    parser.add_argument('--graph_patch_size', type=int, default=8)
    parser.add_argument('--k_neighbors', type=int, default=12)
    parser.add_argument('--num_clusters', type=int, default=8)

    # Noise parameters
    parser.add_argument('--noise_level', type=int, default=25)
    parser.add_argument('--real_noise', action='store_true')

    args = parser.parse_args()
    test(args)


if __name__ == '__main__':
    main()
