import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# =========================================================
# Project root
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


from src.data.worldstrat_dataset import WorldStratDataset
from src.models.temporal_srm_v2 import TemporalSRMV2


# =========================================================
# Configuration
# =========================================================

DATASET_ROOT = (
    PROJECT_ROOT /
    "src" /
    "datasets" /
    "worldstrat"
)

SPLIT_CSV = (
    DATASET_ROOT /
    "stratified_train_val_test_split.csv"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT /
    "checkpoints"
)

CHECKPOINT_DIR.mkdir(
    exist_ok=True
)

BATCH_SIZE = 1
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_WORKERS = 0
HIDDEN_CHANNELS = 96
NUM_RESIDUAL_BLOCKS = 6

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =========================================================
# Gradient loss
# =========================================================

def gradient_loss(
    prediction,
    target
):

    # Horizontal gradients
    pred_x = (
        prediction[:, :, :, 1:]
        -
        prediction[:, :, :, :-1]
    )

    target_x = (
        target[:, :, :, 1:]
        -
        target[:, :, :, :-1]
    )

    # Vertical gradients
    pred_y = (
        prediction[:, :, 1:, :]
        -
        prediction[:, :, :-1, :]
    )

    target_y = (
        target[:, :, 1:, :]
        -
        target[:, :, :-1, :]
    )

    loss_x = F.l1_loss(
        pred_x,
        target_x
    )

    loss_y = F.l1_loss(
        pred_y,
        target_y
    )

    return (
        loss_x +
        loss_y
    )


# =========================================================
# Total loss
# =========================================================

def calculate_loss(
    prediction,
    target
):

    l1 = F.l1_loss(
        prediction,
        target
    )

    grad = gradient_loss(
        prediction,
        target
    )

    # Main reconstruction loss
    # + edge preservation
    total = (
        0.8 * l1 +
        0.2 * grad
    )

    return total, l1, grad


# =========================================================
# PSNR
# =========================================================

def calculate_psnr(
    prediction,
    target
):

    mse = torch.mean(
        (prediction - target) ** 2
    )

    if mse.item() <= 1e-12:

        return 100.0

    return (
        10.0 *
        torch.log10(
            1.0 / mse
        )
    ).item()


# =========================================================
# Validation
# =========================================================

def validate(
    model,
    loader
):

    model.eval()

    total_loss = 0.0
    total_l1 = 0.0
    total_grad = 0.0
    total_psnr = 0.0

    count = 0

    with torch.no_grad():

        for batch in loader:

            lr = batch["lr"].to(
                DEVICE,
                non_blocking=True
            )

            hr = batch["hr"].to(
                DEVICE,
                non_blocking=True
            )

            with torch.amp.autocast(
                device_type="cuda",
                enabled=(
                    DEVICE.type == "cuda"
                )
            ):

                prediction = model(
                    lr
                )

                loss, l1, grad = (
                    calculate_loss(
                        prediction,
                        hr
                    )
                )

            total_loss += loss.item()

            total_l1 += l1.item()

            total_grad += grad.item()

            total_psnr += calculate_psnr(
                prediction.float(),
                hr.float()
            )

            count += 1

    return (
        total_loss / count,
        total_l1 / count,
        total_grad / count,
        total_psnr / count
    )


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 70)
    print("TEMPORAL SRM V2 TRAINING")
    print("=" * 70)

    print(
        "Device:",
        DEVICE
    )

    if DEVICE.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "VRAM:",
            round(
                torch.cuda.get_device_properties(0)
                .total_memory /
                (1024 ** 3),
                2
            ),
            "GB"
        )

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    print()
    print("Loading datasets...")

    train_dataset = WorldStratDataset(
        root=DATASET_ROOT,
        split_csv=SPLIT_CSV,
        split="train",
        revisits=8,
        input_size=160,
        output_size=500,
    )

    val_dataset = WorldStratDataset(
        root=DATASET_ROOT,
        split_csv=SPLIT_CSV,
        split="val",
        revisits=8,
        input_size=160,
        output_size=500,
    )

    print(
        "Train AOIs:",
        len(train_dataset)
    )

    print(
        "Validation AOIs:",
        len(val_dataset)
    )

    # -----------------------------------------------------
    # DataLoaders
    # -----------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(
            DEVICE.type == "cuda"
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            DEVICE.type == "cuda"
        ),
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    print()
    print("Creating Temporal SRM V2...")

    model = TemporalSRMV2(
        num_revisits=8,
        num_input_bands=12,
        num_output_bands=4,
        hidden_channels=HIDDEN_CHANNELS,
        num_residual_blocks=NUM_RESIDUAL_BLOCKS,
    ).to(DEVICE)

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{parameters:,}"
    )

    # -----------------------------------------------------
    # Optimizer
    # -----------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            DEVICE.type == "cuda"
        )
    )

    # -----------------------------------------------------
    # Checkpoints
    # -----------------------------------------------------

    latest_path = (
        CHECKPOINT_DIR /
        "temporal_v2_latest.pth"
    )

    best_path = (
        CHECKPOINT_DIR /
        "temporal_v2_best.pth"
    )

    start_epoch = 1

    best_val_loss = float(
        "inf"
    )

    # -----------------------------------------------------
    # Resume
    # -----------------------------------------------------

    if latest_path.exists():

        print()
        print(
            "Resuming from:",
            latest_path
        )

        checkpoint = torch.load(
            latest_path,
            map_location=DEVICE,
            weights_only=False
        )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        if (
            "scaler_state_dict"
            in checkpoint
        ):

            scaler.load_state_dict(
                checkpoint[
                    "scaler_state_dict"
                ]
            )

        start_epoch = (
            checkpoint["epoch"] + 1
        )

        best_val_loss = checkpoint.get(
            "best_val_loss",
            float("inf")
        )

        print(
            "Resuming from epoch:",
            start_epoch
        )

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    for epoch in range(
        start_epoch,
        NUM_EPOCHS + 1
    ):

        model.train()

        epoch_start = time.time()

        running_loss = 0.0

        running_l1 = 0.0

        running_grad = 0.0

        batches = 0

        print()
        print(
            f"Epoch {epoch}/{NUM_EPOCHS}"
        )

        print(
            "-" * 70
        )

        # -------------------------------------------------
        # Training batches
        # -------------------------------------------------

        for batch_idx, batch in enumerate(
            train_loader,
            start=1
        ):

            lr = batch["lr"].to(
                DEVICE,
                non_blocking=True
            )

            hr = batch["hr"].to(
                DEVICE,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.amp.autocast(
                device_type="cuda",
                enabled=(
                    DEVICE.type == "cuda"
                )
            ):

                prediction = model(
                    lr
                )

                loss, l1, grad = (
                    calculate_loss(
                        prediction,
                        hr
                    )
                )

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

            running_loss += loss.item()

            running_l1 += l1.item()

            running_grad += grad.item()

            batches += 1

            if batch_idx % 100 == 0:

                print(
                    f"  Batch "
                    f"{batch_idx}/"
                    f"{len(train_loader)} "
                    f"Loss: "
                    f"{loss.item():.5f}"
                )

        train_loss = (
            running_loss /
            batches
        )

        train_l1 = (
            running_l1 /
            batches
        )

        train_grad = (
            running_grad /
            batches
        )

        # -------------------------------------------------
        # Validation
        # -------------------------------------------------

        print()
        print(
            "Running validation..."
        )

        (
            val_loss,
            val_l1,
            val_grad,
            val_psnr
        ) = validate(
            model,
            val_loader
        )

        elapsed = (
            time.time() -
            epoch_start
        ) / 60.0

        print()
        print(
            f"Epoch {epoch} complete"
        )

        print(
            f"Train Loss : "
            f"{train_loss:.6f}"
        )

        print(
            f"Train L1   : "
            f"{train_l1:.6f}"
        )

        print(
            f"Train Grad : "
            f"{train_grad:.6f}"
        )

        print(
            f"Val Loss   : "
            f"{val_loss:.6f}"
        )

        print(
            f"Val L1     : "
            f"{val_l1:.6f}"
        )

        print(
            f"Val Grad   : "
            f"{val_grad:.6f}"
        )

        print(
            f"Val PSNR   : "
            f"{val_psnr:.2f} dB"
        )

        print(
            f"Time       : "
            f"{elapsed:.1f} min"
        )

        # -------------------------------------------------
        # Latest checkpoint
        # -------------------------------------------------

        torch.save(
            {
                "epoch": epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "scaler_state_dict":
                    scaler.state_dict(),

                "val_loss":
                    val_loss,

                "val_psnr":
                    val_psnr,

                "best_val_loss":
                    best_val_loss,
            },
            latest_path
        )

        # -------------------------------------------------
        # Best checkpoint
        # -------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                {
                    "epoch": epoch,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "scaler_state_dict":
                        scaler.state_dict(),

                    "val_loss":
                        val_loss,

                    "val_psnr":
                        val_psnr,

                    "best_val_loss":
                        best_val_loss,
                },
                best_path
            )

            print(
                "New best model saved:"
            )

            print(
                best_path
            )

    print()
    print("=" * 70)
    print(
        "TEMPORAL SRM V2 TRAINING FINISHED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()