import torch
import numpy as np


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(filepath, model, optimizer, epoch):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filepath)
    print(f"  Checkpoint saved: {filepath}")


def load_checkpoint(filepath, model, optimizer=None):
    ckpt = torch.load(filepath, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    epoch = ckpt.get('epoch', 0)
    print(f"  Loaded checkpoint from epoch {epoch}: {filepath}")
    return epoch


def compute_psnr(pred, target):
    """Compute PSNR between two tensors (values in [0,1])."""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(1.0 / mse).item()


def compute_ssim(pred, target, window_size=11):
    """Compute SSIM between two image tensors."""
    from torch.nn.functional import conv2d, pad

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Create Gaussian window
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g.unsqueeze(1) * g.unsqueeze(0)
    window = window.unsqueeze(0).unsqueeze(0)  # (1, 1, ws, ws)

    channels = pred.shape[1] if pred.dim() == 4 else pred.shape[0]
    window = window.expand(channels, 1, -1, -1).to(pred.device)

    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    mu1 = conv2d(pred, window, padding=window_size // 2, groups=channels)
    mu2 = conv2d(target, window, padding=window_size // 2, groups=channels)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = conv2d(pred * pred, window, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = conv2d(target * target, window, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = conv2d(pred * target, window, padding=window_size // 2, groups=channels) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()
