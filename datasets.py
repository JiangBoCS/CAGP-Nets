import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class BaseDataset(Dataset):
    """
    Stage 1: Base image dataset (D_base) for self-supervised training.
    Returns masked/unmasked clean image pairs.
    """

    def __init__(self, root_dir, patch_size=128, mask_ratio=0.3, mask_patch_size=16):
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.mask_patch_size = mask_patch_size

        self.image_paths = []
        for fname in sorted(os.listdir(root_dir)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                self.image_paths.append(os.path.join(root_dir, fname))

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def _random_crop(self, img):
        w, h = img.size
        ps = self.patch_size
        if w < ps or h < ps:
            img = img.resize((max(w, ps), max(h, ps)), Image.BICUBIC)
            w, h = img.size
        x = random.randint(0, w - ps)
        y = random.randint(0, h - ps)
        return img.crop((x, y, x + ps, y + ps))

    def _generate_mask(self):
        """Generate a patch-based random mask."""
        ps = self.patch_size
        mps = self.mask_patch_size
        num_patches = (ps // mps) ** 2
        num_masked = int(num_patches * self.mask_ratio)

        mask = torch.ones(1, ps, ps)
        indices = list(range(num_patches))
        random.shuffle(indices)
        masked_indices = indices[:num_masked]

        patches_per_row = ps // mps
        for idx in masked_indices:
            row = idx // patches_per_row
            col = idx % patches_per_row
            mask[:, row * mps:(row + 1) * mps, col * mps:(col + 1) * mps] = 0

        return mask

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img = self._random_crop(img)

        # Random augmentation
        if random.random() > 0.5:
            img = TF.hflip(img)
        if random.random() > 0.5:
            img = TF.vflip(img)

        img_tensor = TF.to_tensor(img)  # (3, H, W), range [0,1]
        mask = self._generate_mask()  # (1, H, W)
        masked_img = img_tensor * mask

        return masked_img, img_tensor, mask


class NovelDataset(Dataset):
    """
    Stage 2: Novel image dataset (D_novel) with clean/noisy pairs.
    For synthetic noise: adds Gaussian noise to clean images.
    For real noise: loads pre-existing noisy images.
    """

    def __init__(self, clean_dir, noisy_dir=None, patch_size=128,
                 noise_level=25, K=20, seed=3407, is_real_noise=False):
        self.patch_size = patch_size
        self.noise_level = noise_level
        self.is_real_noise = is_real_noise

        # Collect image pairs
        clean_images = sorted([
            os.path.join(clean_dir, f) for f in os.listdir(clean_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
        ])

        if noisy_dir is not None:
            noisy_images = sorted([
                os.path.join(noisy_dir, f) for f in os.listdir(noisy_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
            ])
        else:
            noisy_images = None

        # Sample K pairs with fixed seed
        rng = random.Random(seed)
        indices = list(range(len(clean_images)))
        rng.shuffle(indices)
        selected = indices[:K]

        self.clean_paths = [clean_images[i] for i in selected]
        if noisy_images is not None:
            self.noisy_paths = [noisy_images[i] for i in selected]
        else:
            self.noisy_paths = None

    def __len__(self):
        return len(self.clean_paths)

    def _random_crop_pair(self, clean, noisy):
        w, h = clean.size
        ps = self.patch_size
        if w < ps or h < ps:
            clean = clean.resize((max(w, ps), max(h, ps)), Image.BICUBIC)
            noisy = noisy.resize((max(w, ps), max(h, ps)), Image.BICUBIC)
            w, h = clean.size
        x = random.randint(0, w - ps)
        y = random.randint(0, h - ps)
        return clean.crop((x, y, x + ps, y + ps)), noisy.crop((x, y, x + ps, y + ps))

    def __getitem__(self, idx):
        clean = Image.open(self.clean_paths[idx]).convert('RGB')

        if self.is_real_noise and self.noisy_paths is not None:
            noisy = Image.open(self.noisy_paths[idx]).convert('RGB')
        else:
            noisy = clean.copy()

        clean, noisy = self._random_crop_pair(clean, noisy)

        # Random augmentation
        if random.random() > 0.5:
            clean = TF.hflip(clean)
            noisy = TF.hflip(noisy)
        if random.random() > 0.5:
            clean = TF.vflip(clean)
            noisy = TF.vflip(noisy)

        clean_tensor = TF.to_tensor(clean)

        if not self.is_real_noise:
            # Add synthetic Gaussian noise
            noise = torch.randn_like(clean_tensor) * (self.noise_level / 255.0)
            noisy_tensor = clean_tensor + noise
            noisy_tensor = noisy_tensor.clamp(0, 1)
        else:
            noisy_tensor = TF.to_tensor(noisy)

        return noisy_tensor, clean_tensor


class TestDataset(Dataset):
    """Test dataset for evaluation."""

    def __init__(self, clean_dir, noise_level=25, is_real_noise=False, noisy_dir=None):
        self.noise_level = noise_level
        self.is_real_noise = is_real_noise

        self.clean_paths = sorted([
            os.path.join(clean_dir, f) for f in os.listdir(clean_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
        ])

        if noisy_dir is not None:
            self.noisy_paths = sorted([
                os.path.join(noisy_dir, f) for f in os.listdir(noisy_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
            ])
        else:
            self.noisy_paths = None

    def __len__(self):
        return len(self.clean_paths)

    def __getitem__(self, idx):
        clean = Image.open(self.clean_paths[idx]).convert('RGB')
        clean_tensor = TF.to_tensor(clean)

        if self.is_real_noise and self.noisy_paths is not None:
            noisy = Image.open(self.noisy_paths[idx]).convert('RGB')
            noisy_tensor = TF.to_tensor(noisy)
        else:
            torch.manual_seed(idx)
            noise = torch.randn_like(clean_tensor) * (self.noise_level / 255.0)
            noisy_tensor = (clean_tensor + noise).clamp(0, 1)

        return noisy_tensor, clean_tensor, os.path.basename(self.clean_paths[idx])
