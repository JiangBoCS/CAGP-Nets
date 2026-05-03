"""
Data preparation script for CAGP-Net.
Downloads and organizes datasets used in the paper:
- Flickr2K (base dataset for Stage 1)
- CBSD400/CBSD68 (synthetic noise experiments)
- SIDD (real noise experiments)
- KODAK, Set12 (additional test sets)
"""
import os
import argparse
import shutil


def create_directory_structure(root='./data'):
    """Create the expected directory structure."""
    dirs = [
        os.path.join(root, 'Flickr2K'),
        os.path.join(root, 'CBSD400', 'clean'),
        os.path.join(root, 'CBSD68'),
        os.path.join(root, 'KODAK'),
        os.path.join(root, 'Set12'),
        os.path.join(root, 'SIDD', 'train', 'clean'),
        os.path.join(root, 'SIDD', 'train', 'noisy'),
        os.path.join(root, 'SIDD', 'test', 'clean'),
        os.path.join(root, 'SIDD', 'test', 'noisy'),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"  Created: {d}")


def main():
    parser = argparse.ArgumentParser(description='Prepare data directories for CAGP-Net')
    parser.add_argument('--root', type=str, default='./data',
                        help='Root directory for datasets')
    args = parser.parse_args()

    print("Creating directory structure...")
    create_directory_structure(args.root)

    print(f"""
{'='*60}
Directory structure created at: {os.path.abspath(args.root)}
{'='*60}

Please download and place datasets as follows:

1. Flickr2K (Stage 1 base dataset - 2650 HR images):
   Download from: https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar
   Place images in: {args.root}/Flickr2K/

2. CBSD400 (Stage 2 training - synthetic noise):
   Download from: https://github.com/cszn/DnCNN/tree/master/TrainingCodes/dncnn_pytorch/data
   Place clean images in: {args.root}/CBSD400/clean/

3. CBSD68 (Test - synthetic noise):
   Download from: https://github.com/cszn/FFDNet/tree/master/testsets/BSD68
   Place images in: {args.root}/CBSD68/

4. KODAK (Test - 24 images):
   Download from: http://r0k.us/graphics/kodak/
   Place images in: {args.root}/KODAK/

5. Set12 (Test - 12 images):
   Download from: https://github.com/cszn/DnCNN/tree/master/testsets/Set12
   Place images in: {args.root}/Set12/

6. SIDD (Real noise dataset - 160 pairs):
   Download from: https://www.eecs.yorku.ca/~kalf/sidd/
   Place in: {args.root}/SIDD/train/clean/ and {args.root}/SIDD/train/noisy/
   Test split: {args.root}/SIDD/test/clean/ and {args.root}/SIDD/test/noisy/

{'='*60}
After placing datasets, run training:

  # Full training (both stages):
  python train.py --base_dir ./data/Flickr2K --novel_clean_dir ./data/CBSD400/clean --noise_level 25 --K 20

  # Stage 1 only:
  python train.py --stage 1 --base_dir ./data/Flickr2K

  # Stage 2 only (requires pretrained Feature Capturer):
  python train.py --stage 2 --novel_clean_dir ./data/CBSD400/clean --capturer_path ./checkpoints/feature_capturer_final.pth --noise_level 25 --K 20

  # Test:
  python test.py --test_dir ./data/CBSD68 --capturer_path ./checkpoints/feature_capturer_final.pth --model_path ./checkpoints/cagp_net_K20_sigma25_final.pth --noise_level 25 --save_images
{'='*60}
""")


if __name__ == '__main__':
    main()
