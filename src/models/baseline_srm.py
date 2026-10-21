import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineSRM(nn.Module):
    """
    Baseline model for WorldStrat multi-temporal
    multispectral super-resolution.

    Input:
        [B, 8, 12, 160, 160]   (z-score normalized, per worldstrat_dataset.py)

    Output:
        [B, 4, 500, 500]       (in the same [0, 1]-clipped range as the
                                 HR target produced by _read_hr)

    CHANGE FROM PREVIOUS VERSION: added a global residual connection.
    Every SR architecture we reviewed (EDSR, RCAN, DSen2, and the official
    WorldStrat benchmark) adds a cheaply-upsampled version of the input
    back onto the network's output, so the network only has to learn the
    *correction* on top of a sane starting point, rather than reconstruct
    basic radiometry and structure from scratch. This was the single
    highest-leverage, most literature-validated change identified as
    missing from the original architecture.

    Implementation detail that matters: the model's input `x` is
    z-score normalized (WorldStrat's per-band mean/std), NOT in the same
    [0, 1] reflectance-like range as the HR target. Using the normalized
    input directly as the residual base would be wrong - it isn't in the
    target's numeric range. So this model stores the same normalization
    constants used in worldstrat_dataset.py and inverts them first, to
    build the residual base in approximately the same physical range as
    the HR target, before upsampling it.

    ASSUMPTION (documented, unverified - same one flagged earlier in the
    project): the HR target's 4 output bands correspond to the first 4
    of the 12 L2A input bands (roughly B01/B02/B03/B04 in the assumed
    SentinelHub band order). This is only used to build a *residual
    starting point* for the network to refine, not as a hard constraint,
    so even if the assumption is imperfect the network can still learn
    to correct for it - but it's worth eventually verifying against the
    actual band semantics.

    Because the output is now (base + learned correction) rather than a
    raw sigmoid, the final activation changed from sigmoid to a hard
    clamp to [0, 1] - sigmoid doesn't combine sensibly with an additive
    residual already roughly in the target's numeric range, whereas a
    clamp simply enforces the same valid range the HR target was clipped
    to in the dataset loader.
    """

    # Official WorldStrat Sentinel-2 L2A per-band normalization constants,
    # duplicated here (not imported) so this file has no dependency on the
    # dataset module - must match worldstrat_dataset.py exactly.
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
    ):
        super().__init__()

        self.num_revisits = num_revisits
        self.num_output_bands = num_output_bands

        # Registered as buffers (not parameters) so they move with
        # .to(device) automatically and are saved/loaded with the
        # checkpoint, but are never trained.
        self.register_buffer(
            "l2a_mean",
            torch.tensor(self._L2A_MEAN, dtype=torch.float32).view(1, num_input_bands, 1, 1),
        )
        self.register_buffer(
            "l2a_std",
            torch.tensor(self._L2A_STD, dtype=torch.float32).view(1, num_input_bands, 1, 1),
        )

        # --------------------------------------------------
        # 1. Spectral + spatial encoder
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
        # 2. Temporal fusion
        # --------------------------------------------------

        self.temporal_fusion = nn.Sequential(
            nn.Conv2d(
                hidden_channels * num_revisits,
                hidden_channels,
                kernel_size=1,
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
        # 3. Feature reconstruction
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
                hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

        # --------------------------------------------------
        # 4. Output layer
        # --------------------------------------------------

        self.output = nn.Conv2d(
            hidden_channels,
            num_output_bands,
            kernel_size=3,
            padding=1,
        )

    def _build_residual_base(self, x, output_size):
        """
        x: [B, T, C, H, W] - z-score normalized input.

        Returns a [B, num_output_bands, out_H, out_W] tensor in
        approximately the same [0, 1]-ish range as the HR target,
        built by: denormalize -> average across revisits -> keep the
        assumed output bands -> bicubic upsample.
        """

        B, T, C, H, W = x.shape

        # Denormalize: reverse the z-score normalization applied in
        # worldstrat_dataset.py, per band.
        denorm = x * self.l2a_std.unsqueeze(1) + self.l2a_mean.unsqueeze(1)

        # Average across the 8 temporal revisits.
        mean_frame = denorm.mean(dim=1)  # [B, 12, H, W]

        # ASSUMPTION (see class docstring): first num_output_bands input
        # bands correspond to the HR target's bands.
        base = mean_frame[:, : self.num_output_bands]  # [B, 4, H, W]

        base = F.interpolate(
            base,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

        return base

    def forward(self, x):
        """
        x:
            [B, T, C, H, W]

        returns:
            [B, 4, 500, 500]
        """

        batch_size, timesteps, channels, height, width = x.shape

        temporal_features = []

        # Encode every revisit independently
        for t in range(timesteps):

            revisit = x[:, t]

            features = self.encoder(revisit)

            temporal_features.append(features)

        # --------------------------------------------------
        # Concatenate temporal features
        # --------------------------------------------------

        fused = torch.cat(
            temporal_features,
            dim=1,
        )

        fused = self.temporal_fusion(fused)

        # --------------------------------------------------
        # Reconstruction
        # --------------------------------------------------

        features = self.reconstruction(fused)

        # --------------------------------------------------
        # Upsample 160 -> 500
        # --------------------------------------------------

        features = F.interpolate(
            features,
            size=(500, 500),
            mode="bilinear",
            align_corners=False,
        )

        correction = self.output(features)

        # --------------------------------------------------
        # Global residual: the network only has to learn this
        # correction on top of a cheap bicubic-upsampled baseline,
        # not the whole image from scratch.
        # --------------------------------------------------

        base = self._build_residual_base(x, output_size=(500, 500))

        output = base + correction

        # Enforce the same valid range the HR target was clipped to
        # (np.clip(image, 0.0, 1.0) in worldstrat_dataset.py). A hard
        # clamp, not sigmoid, since the sum is already roughly in that
        # range rather than an unbounded logit.
        output = torch.clamp(output, 0.0, 1.0)

        return output


if __name__ == "__main__":

    print("=" * 60)
    print("SIRM BASELINE MODEL TEST (with global residual)")
    print("=" * 60)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # Small test batch - realistic-ish input: zero-mean-ish noise,
    # standing in for z-score normalized data.
    x = torch.randn(
        1,
        8,
        12,
        160,
        160,
        device=device,
    )

    model = BaselineSRM().to(device)

    print()
    print("INPUT:")
    print(x.shape)

    with torch.no_grad():
        y = model(x)

    print()
    print("OUTPUT:")
    print(y.shape)

    print()
    print("OUTPUT RANGE:")
    print(
        float(y.min()),
        "to",
        float(y.max())
    )

    assert y.shape == (1, 4, 500, 500), "Output shape mismatch!"
    assert y.min() >= 0.0 and y.max() <= 1.0, "Output not clamped to [0, 1]!"

    print()
    print("Shape and range checks passed.")
    print("=" * 60)
    print("MODEL TEST COMPLETE")
    print("=" * 60)