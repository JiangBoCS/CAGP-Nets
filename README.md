# 1. Prepare data directories
python prepare_data.py

# 2. Download datasets (Flickr2K, CBSD400, CBSD68, KODAK, Set12, SIDD)

# 3. Train both stages
python train.py --base_dir ./data/Flickr2K --novel_clean_dir ./data/CBSD400/clean --noise_level 25 --K 20

# 4. Test
python test.py --test_dir ./data/CBSD68 --capturer_path ./checkpoints/feature_capturer_final.pth --model_path ./checkpoints/cagp_net_K20_sigma25_final.pth --noise_level 25 --save_images

# 5. Run all experiments from the paper
python run_experiments.py
