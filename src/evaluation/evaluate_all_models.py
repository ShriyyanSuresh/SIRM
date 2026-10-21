import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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
from src.models.baseline_srm import BaselineSRM
from src.models.temporal_srm import TemporalSRM
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

OUTPUT_CSV = (
    PROJECT_ROOT /
    "test_all_models_comparison.csv"
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

BATCH_SIZE = 1


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
# SSIM
# =========================================================

def calculate_ssim(
    prediction,
    target,
    window_size=11,
    C1=0.01 ** 2,
    C2=0.03 ** 2
):

    # Average pooling approximation of SSIM
    mu_x = F.avg_pool2d(
        prediction,
        window_size,
        stride=1,
        padding=window_size // 2
    )

    mu_y = F.avg_pool2d(
        target,
        window_size,
        stride=1,
        padding=window_size // 2
    )

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = (
        F.avg_pool2d(
            prediction ** 2,
            window_size,
            stride=1,
            padding=window_size // 2
        )
        - mu_x_sq
    )

    sigma_y_sq = (
        F.avg_pool2d(
            target ** 2,
            window_size,
            stride=1,
            padding=window_size // 2
        )
        - mu_y_sq
    )

    sigma_xy = (
        F.avg_pool2d(
            prediction * target,
            window_size,
            stride=1,
            padding=window_size // 2
        )
        - mu_xy
    )

    numerator = (
        (2 * mu_xy + C1)
        *
        (2 * sigma_xy + C2)
    )

    denominator = (
        (mu_x_sq + mu_y_sq + C1)
        *
        (sigma_x_sq + sigma_y_sq + C2)
    )

    ssim_map = (
        numerator /
        (denominator + 1e-12)
    )

    return ssim_map.mean().item()


# =========================================================
# Load model checkpoints
# =========================================================

def load_baseline():

    model = BaselineSRM(
        num_revisits=8,
        num_input_bands=12,
        num_output_bands=4,
        hidden_channels=64,
    ).to(DEVICE)

    path = (
        CHECKPOINT_DIR /
        "baseline_best.pth"
    )

    checkpoint = torch.load(
        path,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    print(
        "Loaded:",
        path.name
    )

    print(
        "Checkpoint epoch:",
        checkpoint.get(
            "epoch",
            "unknown"
        )
    )

    return model


def load_temporal_v1():

    model = TemporalSRM(
        num_revisits=8,
        num_input_bands=12,
        num_output_bands=4,
        hidden_channels=64,
        num_residual_blocks=4,
    ).to(DEVICE)

    path = (
        CHECKPOINT_DIR /
        "temporal_best.pth"
    )

    checkpoint = torch.load(
        path,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    print(
        "Loaded:",
        path.name
    )

    print(
        "Checkpoint epoch:",
        checkpoint.get(
            "epoch",
            "unknown"
        )
    )

    return model


def load_temporal_v2():

    model = TemporalSRMV2(
        num_revisits=8,
        num_input_bands=12,
        num_output_bands=4,
        hidden_channels=64,
        num_residual_blocks=6,
    ).to(DEVICE)

    path = (
        CHECKPOINT_DIR /
        "temporal_v2_best.pth"
    )

    checkpoint = torch.load(
        path,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    print(
        "Loaded:",
        path.name
    )

    print(
        "Checkpoint epoch:",
        checkpoint.get(
            "epoch",
            "unknown"
        )
    )

    return model


# =========================================================
# Evaluate one model on entire test set
# =========================================================

def evaluate_model(
    model,
    loader,
    model_name,
    results
):

    print()
    print(
        "=" * 70
    )

    print(
        f"EVALUATING: {model_name}"
    )

    print(
        "=" * 70
    )

    with torch.no_grad():

        for idx, batch in enumerate(
            loader,
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
                enabled=(
                    DEVICE.type == "cuda"
                )
            ):

                if model_name == "Temporal V2":

                    prediction = model(
                        lr
                    )

                else:

                    prediction = model(
                        lr
                    )

            prediction = prediction.float()
            hr = hr.float()

            l1 = torch.mean(
                torch.abs(
                    prediction - hr
                )
            ).item()

            psnr = calculate_psnr(
                prediction,
                hr
            )

            ssim = calculate_ssim(
                prediction,
                hr
            )

            aoi = batch["aoi"][0]

            results.append(
                {
                    "aoi": aoi,
                    "model": model_name,
                    "l1": l1,
                    "psnr": psnr,
                    "ssim": ssim,
                }
            )

            if idx % 25 == 0:

                print(
                    f"Evaluated "
                    f"{idx}/"
                    f"{len(loader.dataset)} AOIs"
                )


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 70)
    print(
        "BASELINE vs TEMPORAL V1 vs TEMPORAL V2"
    )
    print(
        "TEST SET EVALUATION"
    )
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

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    test_dataset = WorldStratDataset(
        root=DATASET_ROOT,
        split_csv=SPLIT_CSV,
        split="test",
        revisits=8,
        input_size=160,
        output_size=500,
    )

    print(
        "Test AOIs:",
        len(test_dataset)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            DEVICE.type == "cuda"
        )
    )

    # -----------------------------------------------------
    # Load models
    # -----------------------------------------------------

    print()
    print("Loading models...")

    baseline = load_baseline()

    temporal_v1 = load_temporal_v1()

    temporal_v2 = load_temporal_v2()

    # -----------------------------------------------------
    # Evaluate
    # -----------------------------------------------------

    results = []

    start_time = time.time()

    evaluate_model(
        baseline,
        test_loader,
        "Baseline",
        results
    )

    evaluate_model(
        temporal_v1,
        test_loader,
        "Temporal V1",
        results
    )

    evaluate_model(
        temporal_v2,
        test_loader,
        "Temporal V2",
        results
    )

    elapsed = (
        time.time() -
        start_time
    ) / 60.0

    # -----------------------------------------------------
    # DataFrame
    # -----------------------------------------------------

    df = pd.DataFrame(
        results
    )

    # -----------------------------------------------------
    # Aggregate statistics
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print(
        "AGGREGATE TEST RESULTS"
    )
    print("=" * 70)

    summary = {}

    for model_name in [
        "Baseline",
        "Temporal V1",
        "Temporal V2"
    ]:

        subset = df[
            df["model"] == model_name
        ]

        summary[model_name] = {
            "l1_mean":
                subset["l1"].mean(),

            "l1_std":
                subset["l1"].std(),

            "psnr_mean":
                subset["psnr"].mean(),

            "psnr_std":
                subset["psnr"].std(),

            "ssim_mean":
                subset["ssim"].mean(),

            "ssim_std":
                subset["ssim"].std(),
        }

        print()
        print(
            model_name
        )

        print(
            f"  L1   : "
            f"mean={subset['l1'].mean():.5f}, "
            f"std={subset['l1'].std():.5f}"
        )

        print(
            f"  PSNR : "
            f"mean={subset['psnr'].mean():.5f}, "
            f"std={subset['psnr'].std():.5f}"
        )

        print(
            f"  SSIM : "
            f"mean={subset['ssim'].mean():.5f}, "
            f"std={subset['ssim'].std():.5f}"
        )

        print(
            f"  PSNR range: "
            f"{subset['psnr'].min():.2f} "
            f"to "
            f"{subset['psnr'].max():.2f} dB"
        )

    # -----------------------------------------------------
    # Pairwise comparisons
    # -----------------------------------------------------

    pivot_psnr = df.pivot(
        index="aoi",
        columns="model",
        values="psnr"
    )

    pivot_l1 = df.pivot(
        index="aoi",
        columns="model",
        values="l1"
    )

    pivot_ssim = df.pivot(
        index="aoi",
        columns="model",
        values="ssim"
    )

    print()
    print("=" * 70)
    print(
        "PAIRWISE COMPARISONS"
    )
    print("=" * 70)

    # -----------------------------------------------------
    # Temporal V1 vs Baseline
    # -----------------------------------------------------

    v1_psnr_diff = (
        pivot_psnr["Temporal V1"]
        -
        pivot_psnr["Baseline"]
    )

    v1_l1_diff = (
        pivot_l1["Baseline"]
        -
        pivot_l1["Temporal V1"]
    )

    v1_ssim_diff = (
        pivot_ssim["Temporal V1"]
        -
        pivot_ssim["Baseline"]
    )

    print()
    print(
        "Temporal V1 vs Baseline"
    )

    print(
        f"  PSNR difference: "
        f"{v1_psnr_diff.mean():+.5f} dB"
    )

    print(
        f"  SSIM difference: "
        f"{v1_ssim_diff.mean():+.5f}"
    )

    print(
        f"  L1 improvement: "
        f"{v1_l1_diff.mean():+.5f}"
    )

    print(
        "  V1 better PSNR:",
        int(
            (v1_psnr_diff > 0).sum()
        ),
        "/",
        len(v1_psnr_diff)
    )

    # -----------------------------------------------------
    # V2 vs Baseline
    # -----------------------------------------------------

    v2_psnr_diff = (
        pivot_psnr["Temporal V2"]
        -
        pivot_psnr["Baseline"]
    )

    v2_l1_diff = (
        pivot_l1["Baseline"]
        -
        pivot_l1["Temporal V2"]
    )

    v2_ssim_diff = (
        pivot_ssim["Temporal V2"]
        -
        pivot_ssim["Baseline"]
    )

    print()
    print(
        "Temporal V2 vs Baseline"
    )

    print(
        f"  PSNR difference: "
        f"{v2_psnr_diff.mean():+.5f} dB"
    )

    print(
        f"  SSIM difference: "
        f"{v2_ssim_diff.mean():+.5f}"
    )

    print(
        f"  L1 improvement: "
        f"{v2_l1_diff.mean():+.5f}"
    )

    print(
        "  V2 better PSNR:",
        int(
            (v2_psnr_diff > 0).sum()
        ),
        "/",
        len(v2_psnr_diff)
    )

    # -----------------------------------------------------
    # V2 vs V1
    # -----------------------------------------------------

    v2_v1_psnr_diff = (
        pivot_psnr["Temporal V2"]
        -
        pivot_psnr["Temporal V1"]
    )

    v2_v1_l1_diff = (
        pivot_l1["Temporal V1"]
        -
        pivot_l1["Temporal V2"]
    )

    v2_v1_ssim_diff = (
        pivot_ssim["Temporal V2"]
        -
        pivot_ssim["Temporal V1"]
    )

    print()
    print(
        "Temporal V2 vs Temporal V1"
    )

    print(
        f"  PSNR difference: "
        f"{v2_v1_psnr_diff.mean():+.5f} dB"
    )

    print(
        f"  SSIM difference: "
        f"{v2_v1_ssim_diff.mean():+.5f}"
    )

    print(
        f"  L1 improvement: "
        f"{v2_v1_l1_diff.mean():+.5f}"
    )

    print(
        "  V2 better PSNR:",
        int(
            (v2_v1_psnr_diff > 0).sum()
        ),
        "/",
        len(v2_v1_psnr_diff)
    )

    # -----------------------------------------------------
    # Save detailed results
    # -----------------------------------------------------

    df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print()
    print("=" * 70)

    print(
        f"Evaluated AOIs: "
        f"{len(test_dataset)}"
    )

    print(
        f"Elapsed time: "
        f"{elapsed:.1f} min"
    )

    print(
        "CSV saved to:"
    )

    print(
        OUTPUT_CSV
    )

    print("=" * 70)
    print(
        "Evaluation complete."
    )


# =========================================================
# Run
# =========================================================

if __name__ == "__main__":

    main()