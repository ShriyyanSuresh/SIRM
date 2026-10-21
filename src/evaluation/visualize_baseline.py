import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.worldstrat_dataset import WorldStratDataset
from src.models.baseline_srm import BaselineSRM


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
DATASET_ROOT = PROJECT_ROOT / "src" / "datasets" / "worldstrat"
SPLIT_CSV = DATASET_ROOT / "stratified_train_val_test_split.csv"
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "baseline_best.pth"

OUTPUT_FILE = PROJECT_ROOT / "baseline_visual_comparison.png"


# ---------------------------------------------------------
# WorldStrat normalization constants
# ---------------------------------------------------------
MEAN = np.array([
    0.0008, 0.0117, 0.0211, 0.0198,
    0.0671, 0.1274, 0.1449, 0.1841,
    0.1738, 0.1803, 0.0566, -0.0559
], dtype=np.float32)

STD = np.array([
    0.0892, 0.0976, 0.1001, 0.1169,
    0.1154, 0.1066, 0.1097, 0.1151,
    0.1122, 0.1176, 0.1298, 0.1176
], dtype=np.float32)


# ---------------------------------------------------------
# Convert an image to displayable RGB
# ---------------------------------------------------------
def make_rgb(image, bands):
    """
    image:
        numpy array [C, H, W]

    bands:
        3 band indices used as RGB
    """

    rgb = image[bands].astype(np.float32)

    # Percentile stretch independently for each channel
    output = np.zeros_like(rgb)

    for c in range(3):
        channel = rgb[c]

        low = np.percentile(channel, 2)
        high = np.percentile(channel, 98)

        if high <= low:
            output[c] = np.clip(channel, 0, 1)
        else:
            output[c] = np.clip(
                (channel - low) / (high - low),
                0,
                1
            )

    return np.transpose(output, (1, 2, 0))


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("BASELINE VISUAL EVALUATION")
    print("=" * 70)

    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # -----------------------------------------------------
    # Load test dataset
    # -----------------------------------------------------
    test_dataset = WorldStratDataset(
        root=DATASET_ROOT,
        split_csv=SPLIT_CSV,
        split="test",
        revisits=8,
        input_size=160,
        output_size=500,
    )

    print(f"TEST AOIs: {len(test_dataset)}")

    # -----------------------------------------------------
    # Pick first test AOI
    # -----------------------------------------------------
    sample_index = 0

    sample = test_dataset[sample_index]

    lr = sample["lr"]
    hr = sample["hr"]
    aoi = sample["aoi"]

    print(f"AOI: {aoi}")
    print(f"LR shape: {lr.shape}")
    print(f"HR shape: {hr.shape}")

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------
    model = BaselineSRM(
        num_revisits=8,
        num_input_bands=12,
        num_output_bands=4,
        hidden_channels=64,
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False
    )

    # Support both normal state_dict and checkpoint dictionaries
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    print(f"Loaded checkpoint: {CHECKPOINT}")

    # -----------------------------------------------------
    # Run inference
    # -----------------------------------------------------
    lr_input = lr.unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(lr_input)

    prediction = prediction.squeeze(0).cpu().numpy()
    lr = lr.cpu().numpy()
    hr = hr.cpu().numpy()

    print(f"Prediction shape: {prediction.shape}")
    print(
        f"Prediction range: "
        f"{prediction.min():.4f} - {prediction.max():.4f}"
    )

    # -----------------------------------------------------
    # Select first revisit for LR visualization
    # -----------------------------------------------------
    lr_first = lr[0]

    # Denormalize Sentinel-2 LR
    lr_first = (
        lr_first
        * STD[:, None, None]
        + MEAN[:, None, None]
    )

    # -----------------------------------------------------
    # Create RGB images
    # -----------------------------------------------------
    # Sentinel-2:
    # B4 = Red -> index 3
    # B3 = Green -> index 2
    # B2 = Blue -> index 1
    lr_rgb = make_rgb(
        lr_first,
        [3, 2, 1]
    )

    # Model output / HR target:
    # WorldStrat HR RGBN:
    # RGB = first three bands
    pred_rgb = make_rgb(
        prediction,
        [0, 1, 2]
    )

    hr_rgb = make_rgb(
        hr,
        [0, 1, 2]
    )

    # -----------------------------------------------------
    # Calculate L1 error
    # -----------------------------------------------------
    l1_error = np.mean(
        np.abs(prediction - hr)
    )

    mse = np.mean(
        (prediction - hr) ** 2
    )

    psnr = 10 * np.log10(
        1.0 / max(mse, 1e-12)
    )

    print(f"L1 error: {l1_error:.6f}")
    print(f"PSNR: {psnr:.2f} dB")

    # -----------------------------------------------------
    # Plot
    # -----------------------------------------------------
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6)
    )

    axes[0].imshow(lr_rgb)
    axes[0].set_title(
        "LR Sentinel-2\nFirst Revisit",
        fontsize=14
    )
    axes[0].axis("off")

    axes[1].imshow(pred_rgb)
    axes[1].set_title(
        f"Baseline Prediction\nPSNR: {psnr:.2f} dB",
        fontsize=14
    )
    axes[1].axis("off")

    axes[2].imshow(hr_rgb)
    axes[2].set_title(
        "HR SPOT Ground Truth",
        fontsize=14
    )
    axes[2].axis("off")

    fig.suptitle(
        f"WorldStrat SRM Baseline — {aoi}",
        fontsize=17
    )

    plt.tight_layout()

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------
    plt.savefig(
        OUTPUT_FILE,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("=" * 70)
    print(f"Saved: {OUTPUT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()