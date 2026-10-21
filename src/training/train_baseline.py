import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ---------------------------------------------------------
# Reproducibility - same seed as train_temporal.py so the
# comparison isn't confounded by different random inits
# ---------------------------------------------------------
torch.manual_seed(42)

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
    )
)

SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from data.worldstrat_dataset import WorldStratDataset
from models.baseline_srm import BaselineSRM
from losses import SIRMLoss


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = os.path.join(
    PROJECT_ROOT,
    "src",
    "datasets",
    "worldstrat",
)

SPLIT_CSV = os.path.join(
    DATASET_ROOT,
    "stratified_train_val_test_split.csv",
)

CHECKPOINT_DIR = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
)

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True,
)


# ============================================================
# TRAINING SETTINGS
# ============================================================

# Physical batch size stays 1 (VRAM-limited), but gradients are
# accumulated over ACCUM_STEPS batches before each optimizer
# step, giving an effective batch size of
# BATCH_SIZE * ACCUM_STEPS = 16 - matches train_temporal.py so
# the comparison is controlled on this axis too.
BATCH_SIZE = 1
ACCUM_STEPS = 16

# Upper bound on epochs - early stopping below will usually end
# the run well before this if val loss stops improving.
NUM_EPOCHS = 15

LEARNING_RATE = 1e-4

# Max gradient norm, applied after unscaling, before every
# optimizer step. Prevents a single bad batch from permanently
# corrupting the model (this is what fixed the epoch-10 collapse
# seen in an earlier run: loss jumped from ~0.12 to a sustained
# ~0.3-0.4 and never recovered, PSNR fell from 25.9 dB to 7.55 dB).
GRAD_CLIP_NORM = 5.0

# --------------------------------------------------------
# LR scheduling + early stopping
#
# Previous run (with clipping, but a fixed LR): val loss hit
# its best at epoch 3 (0.1282, PSNR 25.38) then got steadily
# WORSE every epoch through epoch 9 - both train and val loss
# rising together, not classic overfitting. That pattern means
# the fixed LR is too large to stay near a good minimum once
# found. ReduceLROnPlateau halves the LR whenever val loss
# stalls, and early stopping ends the run instead of wasting
# compute on a model that's drifting away from its best point.
# --------------------------------------------------------
LR_SCHEDULER_PATIENCE = 2        # epochs with no val improvement before halving LR
LR_SCHEDULER_FACTOR = 0.5
EARLY_STOPPING_PATIENCE = 5      # epochs with no new best before stopping entirely

NUM_WORKERS = 0

HIDDEN_CHANNELS = 64

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# PSNR
# ============================================================

def calculate_psnr(prediction, target):
    """
    Calculate PSNR assuming images are in [0, 1].
    """

    mse = torch.mean(
        (prediction - target) ** 2
    )

    if mse == 0:
        return 100.0

    psnr = 10 * torch.log10(
        1.0 / mse
    )

    return psnr.item()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SIRM BASELINE TRAINING")
    print("=" * 70)

    print("Device:", DEVICE)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "GPU VRAM:",
            round(
                torch.cuda.get_device_properties(0).total_memory
                / 1024**3,
                2
            ),
            "GB"
        )

    # ========================================================
    # DATASETS
    # ========================================================

    print()
    print("Loading datasets...")
    print("-" * 70)

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

    print()
    print("Train samples:", len(train_dataset))
    print("Val samples:  ", len(val_dataset))

    # ========================================================
    # DATALOADERS
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # ========================================================
    # MODEL
    # ========================================================

    print()
    print("Creating model...")
    print("-" * 70)

    model = BaselineSRM(
        num_revisits=8,
        num_input_bands=12,
        num_output_bands=4,
        hidden_channels=HIDDEN_CHANNELS,
    ).to(DEVICE)

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Parameters:",
        f"{total_parameters:,}"
    )

    # ========================================================
    # LOSS
    # ========================================================

    criterion = SIRMLoss()

    # ========================================================
    # OPTIMIZER + SCHEDULER
    # ========================================================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
    )

    # ========================================================
    # MIXED PRECISION
    # ========================================================

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=torch.cuda.is_available(),
    )

    latest_path = os.path.join(CHECKPOINT_DIR, "baseline_latest.pth")
    start_epoch = 0
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    if os.path.exists(latest_path):
        print()
        print("Resuming from:", latest_path)
        checkpoint = torch.load(latest_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        print("Resuming from epoch:", start_epoch + 1)

    best_path = os.path.join(CHECKPOINT_DIR, "baseline_best.pth")
    if os.path.exists(best_path):
        best_checkpoint = torch.load(best_path, map_location=DEVICE, weights_only=False)
        best_val_loss = best_checkpoint.get("val_loss", float("inf"))
        print("Best val loss so far:", best_val_loss)

    # ========================================================
    # TRAINING
    # ========================================================

    epochs_without_improvement = 0

    print()
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    for epoch in range(start_epoch, NUM_EPOCHS):

        epoch_start = time.time()

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model.train()

        train_loss = 0.0

        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(train_loader, start=1):

            lr = batch["lr"].to(
                DEVICE,
                non_blocking=True,
            )

            hr = batch["hr"].to(
                DEVICE,
                non_blocking=True,
            )

            # ----------------------------------------------
            # Forward pass with mixed precision
            # ----------------------------------------------

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=torch.cuda.is_available(),
            ):

                prediction = model(lr)

                loss, _components = criterion(
                    prediction,
                    hr,
                )

                loss_scaled = loss / ACCUM_STEPS

            # <-- ADD IT HERE, right after the autocast block closes,
            #     before scaler.scale(loss_scaled).backward()

            if epoch >= 1 and loss.item() > 0.6:
                print(f"  [high loss] AOI={batch['aoi']} loss={loss.item():.4f}")


            # ----------------------------------------------
            # Backpropagation (accumulated)
            # ----------------------------------------------

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

            train_loss += loss.item()

            # ----------------------------------------------
            # Progress
            # ----------------------------------------------

            if batch_idx % 100 == 0:

                print(
                    f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
                    f"Batch {batch_idx}/{len(train_loader)} | "
                    f"Loss: {loss.item():.6f}"
                )

        # Flush any remaining accumulated gradients if the
        # number of batches isn't an exact multiple of ACCUM_STEPS.
        if len(train_loader) % ACCUM_STEPS != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=GRAD_CLIP_NORM,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        train_loss /= len(train_loader)

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()

        val_loss = 0.0
        val_psnr = 0.0

        with torch.no_grad():

            for batch in val_loader:

                lr = batch["lr"].to(
                    DEVICE,
                    non_blocking=True,
                )

                hr = batch["hr"].to(
                    DEVICE,
                    non_blocking=True,
                )

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=torch.cuda.is_available(),
                ):

                    prediction = model(lr)

                    loss, _components = criterion(
                        prediction,
                        hr,
                    )

                val_loss += loss.item()

                val_psnr += calculate_psnr(
                    prediction.float(),
                    hr.float(),
                )

        val_loss /= len(val_loader)
        val_psnr /= len(val_loader)

        epoch_time = time.time() - epoch_start

        # ----------------------------------------------------
        # Step the LR scheduler on val loss
        # ----------------------------------------------------

        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]["lr"]

        # ----------------------------------------------------
        # Epoch summary
        # ----------------------------------------------------

        print()
        print(
            f"Epoch [{epoch + 1:02d}/{NUM_EPOCHS}] "
            f"| Train Loss: {train_loss:.6f} "
            f"| Val Loss: {val_loss:.6f} "
            f"| Val PSNR: {val_psnr:.2f} dB "
            f"| LR: {new_lr:.2e} "
            f"| Time: {epoch_time / 60:.1f} min"
        )

        if new_lr < prev_lr:
            print(
                f"  LR reduced: {prev_lr:.2e} -> {new_lr:.2e} "
                f"(val loss plateaued for {LR_SCHEDULER_PATIENCE} epochs)"
            )

        # ----------------------------------------------------
        # Save latest checkpoint
        # ----------------------------------------------------

        latest_path = os.path.join(
            CHECKPOINT_DIR,
            "baseline_latest.pth",
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_psnr": val_psnr,
            },
            latest_path,
        )

        # ----------------------------------------------------
        # Save best checkpoint + early stopping bookkeeping
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            epochs_without_improvement = 0

            best_path = os.path.join(
                CHECKPOINT_DIR,
                "baseline_best.pth",
            )

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_psnr": val_psnr,
                },
                best_path,
            )

            print(
                "\u2713 New best model saved."
            )

        else:
            epochs_without_improvement += 1
            print(
                f"  No improvement for {epochs_without_improvement} "
                f"epoch(s) (best val loss so far: {best_val_loss:.6f})"
            )

        print()

        # ----------------------------------------------------
        # Clear unused GPU memory
        # ----------------------------------------------------

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping: no improvement for "
                f"{EARLY_STOPPING_PATIENCE} consecutive epochs."
            )
            break

    # ========================================================
    # COMPLETE
    # ========================================================

    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        "Best validation loss:",
        best_val_loss,
    )

    print(
        "Checkpoint directory:",
        CHECKPOINT_DIR,
    )


if __name__ == "__main__":
    main()