import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# ------------------------------------------------------------
# Make src/ available for imports
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# Project imports
# ------------------------------------------------------------

from data.worldstrat_dataset import WorldStratDataset
from models.baseline_srm import BaselineSRM


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

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

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_SAMPLES = 2
BATCH_SIZE = 1
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print("SIRM BASELINE — TINY OVERFIT TEST")
    print("=" * 70)

    print("Device:", DEVICE)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = WorldStratDataset(
        root=DATASET_ROOT,
        split_csv=SPLIT_CSV,
        split="train",
        revisits=8,
        input_size=160,
        output_size=500,
    )

    # Only use two samples
    tiny_dataset = Subset(
        dataset,
        list(range(NUM_SAMPLES)),
    )

    loader = DataLoader(
        tiny_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    print()
    print("Full training dataset:", len(dataset))
    print("Tiny overfit dataset:", len(tiny_dataset))

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = BaselineSRM(
        num_revisits=8,
        num_input_bands=12,
        num_output_bands=4,
        hidden_channels=64,
    ).to(DEVICE)

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.L1Loss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print()
    print("Starting training...")
    print("-" * 70)

    for epoch in range(NUM_EPOCHS):

        model.train()

        epoch_loss = 0.0

        for batch in loader:

            lr = batch["lr"].to(
                DEVICE,
                non_blocking=True,
            )

            hr = batch["hr"].to(
                DEVICE,
                non_blocking=True,
            )

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            prediction = model(lr)

            # ------------------------------------------------
            # Loss
            # ------------------------------------------------

            loss = criterion(
                prediction,
                hr,
            )

            # ------------------------------------------------
            # Backpropagation
            # ------------------------------------------------

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(loader)

        print(
            f"Epoch [{epoch + 1:02d}/{NUM_EPOCHS}] "
            f"Loss: {epoch_loss:.6f}"
        )

    # --------------------------------------------------------
    # Final test
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("OVERFIT TEST COMPLETE")
    print("=" * 70)

    model.eval()

    with torch.no_grad():

        batch = next(iter(loader))

        lr = batch["lr"].to(DEVICE)

        hr = batch["hr"].to(DEVICE)

        prediction = model(lr)

        final_loss = criterion(
            prediction,
            hr,
        )

    print()
    print("Final loss:", final_loss.item())

    print()
    print("Prediction shape:", prediction.shape)
    print("Target shape:    ", hr.shape)

    print()
    print("Prediction range:")
    print(
        float(prediction.min()),
        "to",
        float(prediction.max()),
    )

    print()
    print("Target range:")
    print(
        float(hr.min()),
        "to",
        float(hr.max()),
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
