import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# Residual Block
# =========================================================

class ResidualBlock(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1
            )
        )

    def forward(self, x):

        return x + self.block(x)


# =========================================================
# Temporal Attention
# =========================================================

class TemporalAttentionV2(nn.Module):

    def __init__(self, channels):

        super().__init__()

        # Produces one quality score per revisit
        self.score = nn.Sequential(
            nn.Conv2d(
                channels,
                channels // 2,
                kernel_size=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels // 2,
                1,
                kernel_size=1
            )
        )

    def forward(self, x):

        # x:
        # [B, T, C, H, W]

        B, T, C, H, W = x.shape

        scores = []

        for t in range(T):

            score = self.score(
                x[:, t]
            )

            scores.append(score)

        scores = torch.stack(
            scores,
            dim=1
        )

        # [B, T, 1, H, W]

        attention = torch.softmax(
            scores,
            dim=1
        )

        # Weighted temporal fusion
        weighted = (
            x * attention
        ).sum(dim=1)

        # Maximum feature response across revisits
        maximum = x.max(
            dim=1
        ).values

        return (
            weighted,
            maximum,
            attention.squeeze(2)
        )


# =========================================================
# Temporal SRM V2
# =========================================================

class TemporalSRMV2(nn.Module):

    def __init__(
        self,
        num_revisits=8,
        num_input_bands=12,
        num_output_bands=4,
        hidden_channels=64,
        num_residual_blocks=6,
    ):

        super().__init__()

        self.num_revisits = num_revisits

        # -------------------------------------------------
        # Shared spectral-spatial encoder
        # -------------------------------------------------

        self.encoder = nn.Sequential(

            nn.Conv2d(
                num_input_bands,
                hidden_channels,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),
        )

        # -------------------------------------------------
        # Temporal attention
        # -------------------------------------------------

        self.temporal_attention = TemporalAttentionV2(
            hidden_channels
        )

        # -------------------------------------------------
        # Fusion
        #
        # weighted feature = 64
        # max feature      = 64
        #
        # total             = 128
        # -------------------------------------------------

        self.fusion = nn.Sequential(

            nn.Conv2d(
                hidden_channels * 2,
                hidden_channels,
                kernel_size=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),
        )

        # -------------------------------------------------
        # Residual refinement
        # -------------------------------------------------

        self.residual_blocks = nn.Sequential(

            *[
                ResidualBlock(
                    hidden_channels
                )
                for _ in range(num_residual_blocks)
            ]
        )

        # -------------------------------------------------
        # Learned 2x upsampling
        #
        # 160 x 160
        #       ↓
        # 320 x 320
        # -------------------------------------------------

        self.upsample = nn.Sequential(

            nn.ConvTranspose2d(
                hidden_channels,
                hidden_channels,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

        )

        # -------------------------------------------------
        # Reconstruction
        # -------------------------------------------------

        self.reconstruction = nn.Sequential(

            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                hidden_channels,
                num_output_bands,
                kernel_size=3,
                padding=1
            )
        )

    # =====================================================
    # Forward
    # =====================================================

    def forward(
        self,
        x,
        return_attention=False
    ):

        # -------------------------------------------------
        # Input
        #
        # [B, T, C, H, W]
        # -------------------------------------------------

        B, T, C, H, W = x.shape

        # -------------------------------------------------
        # Encode every revisit using shared weights
        # -------------------------------------------------

        x = x.reshape(
            B * T,
            C,
            H,
            W
        )

        features = self.encoder(x)

        features = features.reshape(
            B,
            T,
            -1,
            H,
            W
        )

        # -------------------------------------------------
        # Temporal fusion
        # -------------------------------------------------

        weighted, maximum, attention = (
            self.temporal_attention(
                features
            )
        )

        fused = torch.cat(
            [
                weighted,
                maximum
            ],
            dim=1
        )

        fused = self.fusion(
            fused
        )

        # -------------------------------------------------
        # Residual refinement
        # -------------------------------------------------

        fused = self.residual_blocks(
            fused
        )

        # -------------------------------------------------
        # Learned upsampling
        # -------------------------------------------------

        fused = self.upsample(
            fused
        )

        # 320 x 320 → 500 x 500
        fused = F.interpolate(
            fused,
            size=(500, 500),
            mode="bilinear",
            align_corners=False
        )

        # -------------------------------------------------
        # Reconstruction
        # -------------------------------------------------

        output = self.reconstruction(
            fused
        )

        output = torch.sigmoid(
            output
        )

        if return_attention:

            return output, attention

        return output


# =========================================================
# Quick model test
# =========================================================

if __name__ == "__main__":

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

    model = TemporalSRMV2(
        num_revisits=8,
        num_input_bands=12,
        num_output_bands=4,
        hidden_channels=64,
        num_residual_blocks=6,
    ).to(device)

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Parameters: {parameters:,}"
    )

    dummy = torch.randn(
        1,
        8,
        12,
        160,
        160,
        device=device
    )

    with torch.no_grad():

        output, attention = model(
            dummy,
            return_attention=True
        )

    print(
        "Input:",
        tuple(dummy.shape)
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
        "Output range:",
        output.min().item(),
        output.max().item()
    )

    print(
        "Attention sum:",
        attention.sum(dim=1).mean().item()
    )

    print("MODEL TEST PASSED")