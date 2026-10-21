import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ---------------------------------------------------------
# Reproducibility - same seed as train_baseline.py so the
# comparison isn't confounded by different random inits
# ---------------------------------------------------------
torch.manual_seed(42)

# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.worldstrat_dataset import WorldStratDataset
from src.models.temporal_srm import TemporalSRM
from src.losses import SIRMLoss


# =========================================================
# Configuration
# =========================================================
DATASET_ROOT = PROJECT_ROOT / "src" / "datasets" / "worldstrat"

SPLIT_CSV = (
    DATASET_ROOT /
    "stratified_train_val_test_split.csv"
)

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Physical batch size stays 1 (VRAM-limited), but gradients are
# accumulated over ACCUM_STEPS batches before each optimizer
# step, giving an effective batch size of
# BATCH_SIZE * ACCUM_STEPS = 16 - matches train_baseline.py so
# the comparison is controlled on this axis too.
BATCH_SIZE = 1
ACCUM_STEPS = 16

# Upper bound on epochs - early stopping below will usually end
# the run well before this if val loss stops improving.
NUM_EPOCHS = 30

LEARNING_RATE = 1e-4

# Max gradient norm, applied after unscaling, before every
# optimizer step. Same fix as train_baseline.py - prevents a
# single bad batch from permanently corrupting the model.
GRAD_CLIP_NORM = 5.0

# --------------------------------------------------------
# LR scheduling + early stopping - identical setup to
# train_baseline.py, so the comparison stays controlled.
# --------------------------------------------------------
LR_SCHEDULER_PATIENCE = 2        # epochs with no val improvement before halving LR
LR_SCHEDULER_FACTOR = 0.5
EARLY_STOPPING_PATIENCE = 5      # epochs with no new best before stopping entirely

# Threshold for the high-loss AOI diagnostic. Gated to start
# at epoch 2 (epoch index >= 1) since epoch 1's loss is
# naturally elevated across nearly every batch while the model
# is still essentially untrained - checking then just reports
# noise, not genuinely bad tiles.
HIGH_LOSS_THRESHOLD = 0.6

NUM_WORKERS = 0

HIDDEN_CHANNELS = 64
NUM_RESIDUAL_BLOCKS = 4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================================================
# PSNR
# =========================================================

def calculate_psnr(pred, target):
    mse = torch.mean(
        (pred - target) ** 2
    )

    if mse.item() <= 1e-12:
        return 100.0

    return (
        10.0 *
        torch.log10(
            1.0 / mse
        )
    ).item()

def degrade_fn(sr_output):
    """
    Downsample a 500x500 SR prediction back down to 160x160 using
    area averaging (a box filter) - this is the standard Wald-protocol
    style degradation operator. Comparing this against lr_reference
    (the real, observed averaged input) is the degradation-consistency
    check: if the SR output is truthful, degrading it should
    reproduce what was actually seen, not something fabricated.
    """
    return F.interpolate(sr_output, size=(160, 160), mode="area")
# =========================================================
# Validation
# =========================================================

def validate(model, loader, criterion):

    model.eval()

    total_loss = 0.0
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
                enabled=(DEVICE.type == "cuda")
            ):

                prediction, lr_reference = model(lr, return_lr_reference=True)

                loss, _components = criterion(
                    prediction,
                    hr,
                    lr_reference=lr_reference,
                    degrade_fn=degrade_fn,
                )

            total_loss += loss.item()

            total_psnr += calculate_psnr(
                prediction.float(),
                hr.float()
            )

            count += 1

    return (
        total_loss / count,
        total_psnr / count
    )


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 70)
    print("TEMPORAL SRM TRAINING")
    print("=" * 70)

    print("Device:", DEVICE)

    if DEVICE.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "VRAM:",
            round(
                torch.cuda.get_device_properties(0).total_memory
                / (1024 ** 3),
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
        pin_memory=(DEVICE.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    print()
    print("Creating Temporal SRM model...")

    model = TemporalSRM(
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
        f"Trainable parameters: {parameters:,}"
    )

    # -----------------------------------------------------
    # Loss + optimizer + scheduler
    # -----------------------------------------------------

    criterion = SIRMLoss(weight_sam=0.1, weight_degradation=0.1)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(DEVICE.type == "cuda")
    )

    # -----------------------------------------------------
    # Resume from previous checkpoint if available
    #
    # IMPORTANT: for a clean comparison against a fresh
    # train_baseline.py run, delete/rename
    # checkpoints/temporal_latest.pth and temporal_best.pth
    # before starting a new diagnostic run, so this doesn't
    # silently resume from weights trained under an older
    # recipe (different loss, no clipping, no scheduler, old
    # architecture without the residual connection).
    # -----------------------------------------------------

    latest_path = (
        CHECKPOINT_DIR /
        "temporal_latest.pth"
    )

    start_epoch = 0
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    if latest_path.exists():

        print()
        print("Resuming from:", latest_path)

        checkpoint = torch.load(
            latest_path,
            map_location=DEVICE,
            weights_only=False
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )

        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(
                checkpoint["scaler_state_dict"]
            )

        start_epoch = checkpoint["epoch"]

        print(
            "Resuming from epoch:",
            start_epoch + 1
        )

    best_path = (
        CHECKPOINT_DIR /
        "temporal_best.pth"
    )

    if best_path.exists():

        best_checkpoint = torch.load(
            best_path,
            map_location=DEVICE,
            weights_only=False
        )

        best_val_loss = best_checkpoint.get(
            "val_loss",
            float("inf")
        )

        print(
            "Best val loss so far:",
            best_val_loss
        )

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    for epoch in range(
        start_epoch,
        NUM_EPOCHS
    ):

        model.train()

        epoch_start = time.time()

        running_loss = 0.0
        batches = 0

        optimizer.zero_grad(set_to_none=True)

        print()
        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS}"
        )
        print("-" * 70)

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

            with torch.amp.autocast(
                device_type="cuda",
                enabled=(DEVICE.type == "cuda")
            ):

                prediction, lr_reference = model(lr, return_lr_reference=True)

                loss, _components = criterion(
                    prediction,
                    hr,
                    lr_reference=lr_reference,
                    degrade_fn=degrade_fn,
                )

                loss_scaled = loss / ACCUM_STEPS

            # Gated to epoch >= 1 (i.e. the 2nd epoch onward) -
            # epoch 1's loss is naturally elevated across nearly
            # every batch while the model is still untrained, so
            # checking then just reports noise, not genuinely
            # problematic tiles.
            if epoch >= 1 and loss.item() > HIGH_LOSS_THRESHOLD:
                print(f"  [high loss] AOI={batch['aoi']} loss={loss.item():.4f}")

            scaler.scale(loss_scaled).backward()

            if batch_idx % ACCUM_STEPS == 0:
                # Unscale before clipping - clipping must see the
                # true gradient magnitudes, not the AMP-scaled ones.
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=GRAD_CLIP_NORM,
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()
            batches += 1

            if batch_idx % 100 == 0:

                print(
                    f"  Batch {batch_idx}/{len(train_loader)} "
                    f"Loss: {loss.item():.5f}"
                )

        # Flush any remaining accumulated gradients if the
        # number of batches isn't an exact multiple of ACCUM_STEPS.
        if batches % ACCUM_STEPS != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=GRAD_CLIP_NORM,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        train_loss = (
            running_loss / batches
        )

        # -------------------------------------------------
        # Validation
        # -------------------------------------------------

        print()
        print("Running validation...")

        val_loss, val_psnr = validate(
            model,
            val_loader,
            criterion
        )

        elapsed = (
            time.time() - epoch_start
        ) / 60.0

        # -------------------------------------------------
        # Step the LR scheduler on val loss
        # -------------------------------------------------

        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]["lr"]

        print()
        print(
            f"Epoch {epoch + 1} complete"
        )

        print(
            f"Train Loss : {train_loss:.6f}"
        )

        print(
            f"Val Loss   : {val_loss:.6f}"
        )

        print(
            f"Val PSNR   : {val_psnr:.2f} dB"
        )

        print(
            f"LR         : {new_lr:.2e}"
        )

        print(
            f"Time       : {elapsed:.1f} min"
        )

        if new_lr < prev_lr:
            print(
                f"  LR reduced: {prev_lr:.2e} -> {new_lr:.2e} "
                f"(val loss plateaued for {LR_SCHEDULER_PATIENCE} epochs)"
            )

        # -------------------------------------------------
        # Latest checkpoint
        # -------------------------------------------------

        latest_path = (
            CHECKPOINT_DIR /
            "temporal_latest.pth"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "val_loss": val_loss,
                "val_psnr": val_psnr,
            },
            latest_path
        )

        # -------------------------------------------------
        # Best checkpoint + early stopping bookkeeping
        # -------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            epochs_without_improvement = 0

            best_path = (
                CHECKPOINT_DIR /
                "temporal_best.pth"
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "val_loss": val_loss,
                    "val_psnr": val_psnr,
                },
                best_path
            )

            print(
                f"New best model saved: {best_path}"
            )

        else:
            epochs_without_improvement += 1
            print(
                f"  No improvement for {epochs_without_improvement} "
                f"epoch(s) (best val loss so far: {best_val_loss:.6f})"
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # -------------------------------------------------
        # Early stopping
        # -------------------------------------------------

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping: no improvement for "
                f"{EARLY_STOPPING_PATIENCE} consecutive epochs."
            )
            break

    print()
    print("=" * 70)
    print("TEMPORAL SRM TRAINING FINISHED")
    print("=" * 70)


if __name__ == "__main__":
    main()