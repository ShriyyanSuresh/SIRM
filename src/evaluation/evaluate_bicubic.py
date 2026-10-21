import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Add SIRM project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.worldstrat_dataset import WorldStratDataset


def psnr(pred, target):
    mse = np.mean((pred - target) ** 2)

    if mse == 0:
        return 100

    return 20 * np.log10(1.0 / np.sqrt(mse))


def rgb(x):
    """
    Convert 4-band image to RGB.

    Bands:
    R=2
    G=1
    B=0
    """

    x = np.transpose(x, (1, 2, 0))

    rgb = x[:, :, [2, 1, 0]]

    rgb = np.clip(rgb, 0, 1)

    return rgb


def main():

    print("=" * 70)
    print("BICUBIC BASELINE")
    print("=" * 70)

    dataset_root = os.path.join(
        "src",
        "datasets",
        "worldstrat"
    )

    split_csv = os.path.join(
        dataset_root,
        "stratified_train_val_test_split.csv"
    )

    dataset = WorldStratDataset(
        root=dataset_root,
        split_csv=split_csv,
        split="test",
        revisits=8,
        input_size=160,
        output_size=500,
    )

    sample = dataset[0]

    lr = sample["lr"]
    hr = sample["hr"]

    print("AOI:", sample["aoi"])

    # ---------------------------------------------------------
    # Denormalize Sentinel-2 LR
    # ---------------------------------------------------------
    MEAN = torch.tensor([
        0.0008, 0.0117, 0.0211, 0.0198,
        0.0671, 0.1274, 0.1449, 0.1841,
        0.1738, 0.1803, 0.0566, -0.0559
    ], dtype=torch.float32)

    STD = torch.tensor([
        0.0892, 0.0976, 0.1001, 0.1169,
        0.1154, 0.1066, 0.1097, 0.1151,
        0.1122, 0.1176, 0.1298, 0.1176
    ], dtype=torch.float32)

    # First revisit
    lr_first = lr[0]

    # Denormalize all 12 Sentinel-2 bands
    lr_first = (
        lr_first * STD[:, None, None]
        + MEAN[:, None, None]
    )

    # Use the first 4 bands as a simple spatial interpolation baseline
    lr_first = lr_first[:4]

    # Bicubic upsampling
    bicubic = F.interpolate(
        lr_first.unsqueeze(0),
        size=(500, 500),
        mode="bicubic",
        align_corners=False,
    ).squeeze(0)

    # Keep values in the HR target range
    bicubic = torch.clamp(bicubic, 0.0, 1.0)

    pred = bicubic.numpy()
    target = hr.numpy()

    p = psnr(pred, target)

    print("PSNR:", round(p, 2), "dB")

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(rgb(lr_first.numpy()))
    ax[0].set_title("LR")

    ax[1].imshow(rgb(pred))
    ax[1].set_title(
        f"Bicubic\nPSNR {p:.2f}"
    )

    ax[2].imshow(rgb(target))
    ax[2].set_title("HR")

    for a in ax:
        a.axis("off")

    plt.tight_layout()

    save_path = "bicubic_comparison.png"

    plt.savefig(save_path, dpi=300)

    print()
    print("Saved:", save_path)
    print("=" * 70)


if __name__ == "__main__":
    main()