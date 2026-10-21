import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Standard residual block.
    """

    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
            ),
        )

    def forward(self, x):
        return x + self.block(x)


class TemporalAttention(nn.Module):
    """
    Learns how important each temporal revisit is.

    Input:
        [B, T, C, H, W]

    Output:
        fused features [B, C, H, W]
        attention weights [B, T, H, W]
    """

    def __init__(self, channels):
        super().__init__()

        # Produces one attention score per spatial location
        # for every revisit.
        self.score = nn.Conv2d(
            channels,
            1,
            kernel_size=1,
        )

    def forward(self, x):

        # x:
        # [B, T, C, H, W]

        B, T, C, H, W = x.shape

        scores = []

        for t in range(T):
            score = self.score(x[:, t])
            scores.append(score)

        # [B, T, 1, H, W]
        scores = torch.stack(scores, dim=1)

        # Normalize across temporal dimension
        attention = torch.softmax(
            scores,
            dim=1,
        )

        # Weighted temporal fusion
        fused = torch.sum(
            x * attention,
            dim=1,
        )

        # Return attention without channel dimension
        attention = attention.squeeze(2)

        return fused, attention


class TemporalSRM(nn.Module):
    """
    Temporal multispectral super-resolution model
    for WorldStrat.

    Input:
        [B, 8, 12, 160, 160]   (z-score normalized, per worldstrat_dataset.py)

    Output:
        [B, 4, 500, 500]       (in the same [0, 1]-clipped range as the
                                 HR target produced by _read_hr)

    CHANGES FROM PREVIOUS VERSION, matching the fixes applied to
    baseline_srm.py so the two models are comparable under the same
    recipe:

    1. Global residual connection. The network now only has to learn a
       correction on top of a cheap bilinear-upsampled, denormalized
       input baseline, rather than reconstruct the whole image from
       scratch. Same implementation approach as BaselineSRM - see that
       file's docstring for the full reasoning. Same documented
       ASSUMPTION carried over: the HR target's 4 output bands are
       assumed to correspond to the first 4 of the 12 L2A input bands.

    2. Final activation changed from sigmoid to a hard clamp to [0, 1],
       for the same reason as BaselineSRM: sigmoid doesn't combine
       sensibly with an additive residual already roughly in the
       target's numeric range.

    3. The per-revisit encoder loop is now vectorized (reshape to
       (B*T, C, H, W) and run the encoder once) instead of looping in
       Python T times. Same result, faster - this was flagged as a
       free speedup early in the project and never applied here.

    4. forward() can now also return lr_reference (the real, observed
       averaged LR signal in output band-space, at native 160x160
       resolution) via return_lr_reference=True - used by the
       degradation-consistency loss term in train_temporal.py.
    """

    _L2A_MEAN = [
        0.0008, 0.0117, 0.0211, 0.0198,
        0.0671, 0.1274, 0.1449, 0.1841,
        0.1738, 0.1803, 0.0566, -0.0559,
    ]
    _L2A_STD = [
        0.0892, 0.0976, 0.1001, 0.1169,
        0.1154, 0.1066, 0.1097, 0.1151,
        0.1122, 0.1176, 0.1298, 0.1176,
    ]

    def __init__(
        self,
        num_revisits=8,
        num_input_bands=12,
        num_output_bands=4,
        hidden_channels=64,
        num_residual_blocks=4,
    ):
        super().__init__()

        self.num_revisits = num_revisits
        self.num_output_bands = num_output_bands

        self.register_buffer(
            "l2a_mean",
            torch.tensor(self._L2A_MEAN, dtype=torch.float32).view(1, num_input_bands, 1, 1),
        )
        self.register_buffer(
            "l2a_std",
            torch.tensor(self._L2A_STD, dtype=torch.float32).view(1, num_input_bands, 1, 1),
        )

        # --------------------------------------------------
        # 1. Shared spectral-spatial encoder
        # --------------------------------------------------

        self.encoder = nn.Sequential(

            nn.Conv2d(
                num_input_bands,
                hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

        # --------------------------------------------------
        # 2. Temporal attention
        # --------------------------------------------------

        self.temporal_attention = TemporalAttention(
            hidden_channels
        )

        # --------------------------------------------------
        # 3. Feature fusion
        # --------------------------------------------------

        self.fusion = nn.Sequential(

            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

        # --------------------------------------------------
        # 4. Residual reconstruction
        # --------------------------------------------------

        residual_blocks = []

        for _ in range(num_residual_blocks):
            residual_blocks.append(
                ResidualBlock(hidden_channels)
            )

        self.residual_blocks = nn.Sequential(
            *residual_blocks
        )

        # --------------------------------------------------
        # 5. Final reconstruction
        # --------------------------------------------------

        self.reconstruction = nn.Sequential(

            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                hidden_channels,
                num_output_bands,
                kernel_size=3,
                padding=1,
            ),
        )

    def _build_residual_base(self, x, output_size):
        """
        x: [B, T, C, H, W] - z-score normalized input.

        Returns:
            base: [B, num_output_bands, out_H, out_W] upsampled residual
                  base, same as before.
            lr_reference: [B, num_output_bands, H, W] at NATIVE input
                  resolution (not upsampled) - the real, observed
                  averaged LR signal in the output band-space. This is
                  the target for the degradation-consistency loss:
                  if we downsample our SR prediction back down, it
                  should match this.
        """

        denorm = x * self.l2a_std.unsqueeze(1) + self.l2a_mean.unsqueeze(1)

        mean_frame = denorm.mean(dim=1)  # [B, 12, H, W]

        lr_reference = mean_frame[:, : self.num_output_bands]  # [B, 4, H, W]

        base = F.interpolate(
            lr_reference,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

        return base, lr_reference

    def forward(
        self,
        x,
        return_attention=False,
        return_lr_reference=False,
    ):
        """
        x:
            [B, T, C, H, W]

        returns:
            [B, 4, 500, 500], plus optionally attention and/or
            lr_reference depending on the flags above.
        """

        B, T, C, H, W = x.shape

        if T != self.num_revisits:
            raise ValueError(
                f"Expected {self.num_revisits} revisits, "
                f"but received {T}"
            )

        # --------------------------------------------------
        # Encode every revisit using the SAME encoder,
        # vectorized across the temporal dimension instead of
        # looping in Python.
        # --------------------------------------------------

        flat = x.reshape(B * T, C, H, W)
        encoded_flat = self.encoder(flat)
        features = encoded_flat.reshape(B, T, -1, H, W)

        # --------------------------------------------------
        # Temporal attention + fusion
        # --------------------------------------------------

        fused, attention = self.temporal_attention(
            features
        )

        fused = self.fusion(
            fused
        )

        # --------------------------------------------------
        # Residual feature reconstruction
        # --------------------------------------------------

        fused = self.residual_blocks(
            fused
        )

        # --------------------------------------------------
        # Upsample 160 -> 500
        # --------------------------------------------------

        fused = F.interpolate(
            fused,
            size=(500, 500),
            mode="bilinear",
            align_corners=False,
        )

        # --------------------------------------------------
        # HR reconstruction (correction, pre-residual)
        # --------------------------------------------------

        correction = self.reconstruction(
            fused
        )

        # --------------------------------------------------
        # Global residual: the network only has to learn this
        # correction on top of a cheap bilinear-upsampled baseline,
        # not the whole image from scratch.
        # --------------------------------------------------

        base, lr_reference = self._build_residual_base(x, output_size=(500, 500))

        output = base + correction

        # Enforce the same valid range the HR target was clipped to.
        # Hard clamp, not sigmoid - see class docstring.
        output = torch.clamp(output, 0.0, 1.0)

        if return_attention and return_lr_reference:
            return output, attention, lr_reference
        if return_attention:
            return output, attention
        if return_lr_reference:
            return output, lr_reference

        return output


# ==========================================================
# Quick model test
# ==========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("TEMPORAL SRM MODEL TEST (with global residual)")
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    model = TemporalSRM(
        num_revisits=8,
        num_input_bands=12,
        num_output_bands=4,
        hidden_channels=64,
        num_residual_blocks=4,
    ).to(device)

    # Count parameters
    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Parameters: {parameters:,}"
    )

    # Dummy WorldStrat input
    x = torch.randn(
        1,
        8,
        12,
        160,
        160,
        device=device,
    )

    print(
        "Input:",
        tuple(x.shape)
    )

    # Forward pass - test all return combinations
    with torch.no_grad():

        output, attention = model(
            x,
            return_attention=True,
        )

        output2, attention2, lr_reference = model(
            x,
            return_attention=True,
            return_lr_reference=True,
        )

    print(
        "Output:",
        tuple(output.shape)
    )

    print(
        "Attention:",
        tuple(attention.shape)
    )

    print(
        "LR reference:",
        tuple(lr_reference.shape)
    )

    print(
        "Output range:",
        f"{output.min().item():.4f}",
        "-",
        f"{output.max().item():.4f}",
    )

    print(
        "Attention sum:",
        attention[:, :, 0, 0].sum().item()
    )

    assert output.shape == (1, 4, 500, 500), "Output shape mismatch!"
    assert output.min() >= 0.0 and output.max() <= 1.0, "Output not clamped to [0, 1]!"
    assert lr_reference.shape == (1, 4, 160, 160), "lr_reference shape mismatch!"

    print()
    print("Shape and range checks passed.")
    print("=" * 70)
    print("MODEL TEST PASSED")
    print("=" * 70)