"""
Batch experiment runner for CAGP-Net.
Reproduces the paper's main results across different noise levels and K values.
"""
import os
import subprocess
import sys


def run_cmd(cmd):
    print(f"\n{'='*60}")
    print(f"Running: {cmd}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        sys.exit(1)


def main():
    # Configuration
    base_dir = './data/Flickr2K'
    novel_clean_dir = './data/CBSD400/clean'
    save_dir = './checkpoints'
    test_dirs = {
        'CBSD68': './data/CBSD68',
        'KODAK': './data/KODAK',
        'Set12': './data/Set12',
    }
    noise_levels = [15, 25, 50]
    K_values = [20, 40, 60]

    os.makedirs(save_dir, exist_ok=True)

    # Stage 1: Train Feature Capturer (only once)
    capturer_path = os.path.join(save_dir, 'feature_capturer_final.pth')
    if not os.path.exists(capturer_path):
        run_cmd(
            f'python train.py --stage 1 '
            f'--base_dir {base_dir} '
            f'--save_dir {save_dir} '
            f'--stage1_epochs 400 '
            f'--batch_size 8 '
            f'--patch_size 128'
        )
    else:
        print(f"Feature Capturer already trained: {capturer_path}")

    # Stage 2: Train CAGP-Net for each (noise_level, K) combination
    for sigma in noise_levels:
        for K in K_values:
            model_path = os.path.join(save_dir, f'cagp_net_K{K}_sigma{sigma}_final.pth')
            if os.path.exists(model_path):
                print(f"Already trained: {model_path}")
                continue

            run_cmd(
                f'python train.py --stage 2 '
                f'--novel_clean_dir {novel_clean_dir} '
                f'--capturer_path {capturer_path} '
                f'--save_dir {save_dir} '
                f'--noise_level {sigma} '
                f'--K {K} '
                f'--stage2_epochs 800 '
                f'--batch_size 8 '
                f'--patch_size 128'
            )

    # Test all combinations
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    for sigma in noise_levels:
        for K in K_values:
            model_path = os.path.join(save_dir, f'cagp_net_K{K}_sigma{sigma}_final.pth')
            if not os.path.exists(model_path):
                print(f"  [SKIP] Model not found: {model_path}")
                continue

            for test_name, test_dir in test_dirs.items():
                if not os.path.exists(test_dir):
                    print(f"  [SKIP] Test dir not found: {test_dir}")
                    continue

                output_dir = f'./results/sigma{sigma}_K{K}/{test_name}'
                run_cmd(
                    f'python test.py '
                    f'--test_dir {test_dir} '
                    f'--capturer_path {capturer_path} '
                    f'--model_path {model_path} '
                    f'--noise_level {sigma} '
                    f'--output_dir {output_dir} '
                    f'--save_images'
                )


if __name__ == '__main__':
    main()
