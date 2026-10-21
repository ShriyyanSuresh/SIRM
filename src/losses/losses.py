"""
Loss functions for SIRM training.

Baseline loss (Exp 1-4) mirrors the official WorldStrat benchmark's
weighted MSE + MAE + SSIM combination (see their supplementary Table 1:
weight_mse=0.3, weight_mae=0.4, weight_ssim=0.3) so our results are
directly comparable to a documented reference point.

SAM (spectral angle mapper) and the degradation-consistency term are
implemented here but weighted to 0.0 in the baseline config - they get
switched on for later ablation experiments (Exp 5, Exp 6), not before.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def ssim(img1, img2, window_size=11, C1=0.01 ** 2, C2=0.03 ** 2):
    """Simple single-scale SSIM, computed per-batch, averaged over channels.
    Assumes img1/img2 in the same normalized range (we use ~[0, 1.5])."""
    channels = img1.shape[1]
    pad = window_size // 2

    def gaussian_window(size, sigma=1.5):
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g

    device = img1.device
    g1d = gaussian_window(window_size).to(device)
    window = (g1d[:, None] @ g1d[None, :]).unsqueeze(0).unsqueeze(0)
    window = window.expand(channels, 1, window_size, window_size).contiguous()

    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


def spectral_angle_mapper(pred, target, eps=1e-8):
    """Per-pixel spectral angle across the channel dimension, averaged.
    pred, target: (B, C, H, W)."""
    dot = (pred * target).sum(dim=1)
    pred_norm = pred.norm(dim=1).clamp_min(eps)
    target_norm = target.norm(dim=1).clamp_min(eps)
    cos_angle = (dot / (pred_norm * target_norm)).clamp(-1 + 1e-7, 1 - 1e-7)
    angle = torch.acos(cos_angle)
    return angle.mean()


def degradation_consistency_loss(sr, lr_reference, degrade_fn):
    """Soft degradation-consistency penalty: downsample the SR output back
    to LR resolution and compare against the real observed input.
    degrade_fn: callable(sr_tensor) -> tensor at LR resolution/band-space."""
    degraded = degrade_fn(sr)
    return F.l1_loss(degraded, lr_reference)


class SIRMLoss(nn.Module):
    def __init__(
        self,
        weight_mse: float = 0.3,
        weight_mae: float = 0.4,
        weight_ssim: float = 0.3,
        weight_sam: float = 0.0,
        weight_degradation: float = 0.0,
    ):
        super().__init__()
        self.w_mse = weight_mse
        self.w_mae = weight_mae
        self.w_ssim = weight_ssim
        self.w_sam = weight_sam
        self.w_deg = weight_degradation

    def forward(self, pred, target, lr_reference=None, degrade_fn=None):
        loss_mse = F.mse_loss(pred, target)
        loss_mae = F.l1_loss(pred, target)
        loss_ssim = 1.0 - ssim(pred, target)

        total = self.w_mse * loss_mse + self.w_mae * loss_mae + self.w_ssim * loss_ssim
        components = {"mse": loss_mse.item(), "mae": loss_mae.item(), "ssim_loss": loss_ssim.item()}

        if self.w_sam > 0:
            loss_sam = spectral_angle_mapper(pred, target)
            total = total + self.w_sam * loss_sam
            components["sam"] = loss_sam.item()

        if self.w_deg > 0 and lr_reference is not None and degrade_fn is not None:
            loss_deg = degradation_consistency_loss(pred, lr_reference, degrade_fn)
            total = total + self.w_deg * loss_deg
            components["degradation"] = loss_deg.item()

        components["total"] = total.item()
        return total, components
