"""Evaluation metrics for SIRM. Kept dependency-light (no external
metrics package) so the environment stays simple - just torch/numpy."""

import torch
import numpy as np
from src.losses.losses import ssim as ssim_fn, spectral_angle_mapper


def psnr(pred, target, data_range=1.5):
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return float("inf")
    return 20 * np.log10(data_range) - 10 * np.log10(mse)


def per_band_rmse(pred, target):
    # pred, target: (B, C, H, W) -> returns list of length C
    diffs = (pred - target) ** 2
    return torch.sqrt(diffs.mean(dim=(0, 2, 3))).detach().cpu().numpy().tolist()


def ergas(pred, target, scale_ratio):
    """Global relative dimensionless synthesis error. Lower is better."""
    C = pred.shape[1]
    band_terms = []
    for c in range(C):
        rmse_c = torch.sqrt(torch.mean((pred[:, c] - target[:, c]) ** 2)).item()
        mean_c = torch.mean(target[:, c]).item()
        if mean_c == 0:
            continue
        band_terms.append((rmse_c / mean_c) ** 2)
    if not band_terms:
        return float("nan")
    return 100.0 * (1.0 / scale_ratio) * np.sqrt(np.mean(band_terms))


def evaluate_batch(pred, target, scale_ratio=10.0 / 3.0):
    with torch.no_grad():
        return {
            "psnr": psnr(pred, target),
            "ssim": ssim_fn(pred, target).item(),
            "sam": spectral_angle_mapper(pred, target).item(),
            "ergas": ergas(pred, target, scale_ratio),
            "per_band_rmse": per_band_rmse(pred, target),
        }
