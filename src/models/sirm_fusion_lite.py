"""
SIRM-Fusion-Lite: the first trainable baseline (Ablation "Exp 4" from our
design doc - RCAN backbone + fusion, no spectral/degradation/uncertainty
losses yet).

Design choices, and why, given the 6 GB VRAM budget:

- Multi-frame fusion via a learned per-frame attention weight (softmax over
  T revisits, computed from cheap global-pooled features), rather than a
  recurrent/ConvLSTM fusion like the original HighRes-Net. This is O(T)
  and fully parallelizable across frames instead of sequential, which
  matters a lot when T=8 frames all need to fit in memory alongside
  activations for backprop on a 6 GB card.
- Channel attention (RCAN-style squeeze-and-excite) throughout the
  backbone - cheap (a few FC layers per block), does the spectral-fidelity
  work.
- No spatial self-attention / transformer blocks in THIS baseline - that's
  reserved for the "Full SIRM" model (Exp 6/7) once this baseline is
  confirmed to train stably. Per the project's own instruction: don't
  overengineer the first version.
- Non-integer scale factor (10/3 ≈ 3.333) handled via bicubic upsampling
  of the fused LR features to the exact target size, followed by a
  learned residual refinement - avoids the complexity of a fractional
  PixelShuffle.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """RCAN-style squeeze-and-excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(self.pool(x))
        return x * w


class ResidualChannelAttentionBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.ca = ChannelAttention(channels, reduction)

    def forward(self, x):
        res = self.ca(self.body(x))
        return x + res


class ResidualGroup(nn.Module):
    def __init__(self, channels: int, num_blocks: int, reduction: int = 16):
        super().__init__()
        blocks = [ResidualChannelAttentionBlock(channels, reduction) for _ in range(num_blocks)]
        blocks.append(nn.Conv2d(channels, channels, 3, padding=1))
        self.body = nn.Sequential(*blocks)

    def forward(self, x):
        return x + self.body(x)


class FrameAttentionFusion(nn.Module):
    """Collapse (B, T, C, H, W) -> (B, C, H, W) via a learned per-frame,
    per-pixel-shared attention weight. Cheap: global-pooled features only."""

    def __init__(self, channels: int, hidden: int = 32):
        super().__init__()
        self.score = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, feats):
        # feats: (B, T, C, H, W)
        B, T, C, H, W = feats.shape
        flat = feats.view(B * T, C, H, W)
        scores = self.score(flat).view(B, T, 1, 1, 1)
        weights = torch.softmax(scores, dim=1)
        fused = (feats * weights).sum(dim=1)  # (B, C, H, W)
        return fused, weights.squeeze(-1).squeeze(-1).squeeze(-1)  # also return weights for inspection


class SIRMFusionLite(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        base_channels: int = 48,
        num_residual_groups: int = 4,
        num_residual_blocks_per_group: int = 4,
        reduction_ratio: int = 16,
        scale_factor: float = 10.0 / 3.0,
    ):
        super().__init__()
        self.scale_factor = scale_factor

        # Shared per-frame feature extractor (same weights across revisits).
        self.head = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.fusion = FrameAttentionFusion(base_channels)

        self.body = nn.Sequential(
            *[
                ResidualGroup(base_channels, num_residual_blocks_per_group, reduction_ratio)
                for _ in range(num_residual_groups)
            ]
        )
        self.body_conv = nn.Conv2d(base_channels, base_channels, 3, padding=1)

        # Refinement after upsampling to the exact target resolution.
        self.refine = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, out_channels, 3, padding=1),
        )

        # A cheap bicubic-upsampled reconstruction of the *mean* input frame,
        # added as a global residual connection - the network only has to
        # learn the correction on top of a naive baseline, not the whole
        # image from scratch. Standard practice in SR (EDSR/RCAN use the
        # analogous single-frame version of this).
        self._out_channels = out_channels

    def forward(self, lr_stack, target_hw=None):
        """
        lr_stack: (B, T, C, H, W)
        target_hw: (H_out, W_out) - required because our scale factor is
                   non-integer, so the exact output size must be specified
                   rather than derived purely from scale_factor * input.
        """
        B, T, C, H, W = lr_stack.shape
        flat = lr_stack.view(B * T, C, H, W)
        feats = self.head(flat).view(B, T, -1, H, W)

        fused, frame_weights = self.fusion(feats)  # (B, base_channels, H, W)

        x = self.body_conv(self.body(fused)) + fused

        if target_hw is None:
            target_hw = (int(round(H * self.scale_factor)), int(round(W * self.scale_factor)))
        x_up = F.interpolate(x, size=target_hw, mode="bicubic", align_corners=False)

        out = self.refine(x_up)

        # Global residual: bicubic-upsampled attention-weighted mean of the
        # raw input frames, in the ORIGINAL band space, so the network's
        # job is to add detail rather than reconstruct radiometry from zero.
        with torch.no_grad():
            mean_frame = (lr_stack * frame_weights.view(B, T, 1, 1, 1)).sum(dim=1)  # (B, C, H, W)
        mean_frame_up = F.interpolate(mean_frame, size=target_hw, mode="bicubic", align_corners=False)

        return out + mean_frame_up, frame_weights


def build_model(config: dict) -> SIRMFusionLite:
    m = config["model"]
    return SIRMFusionLite(
        in_channels=m["in_channels"],
        out_channels=m["out_channels"],
        base_channels=m["base_channels"],
        num_residual_groups=m["num_residual_groups"],
        num_residual_blocks_per_group=m["num_residual_blocks_per_group"],
        reduction_ratio=m["reduction_ratio"],
        scale_factor=m["scale_factor"],
    )
