
import sys
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.worldstrat_dataset import WorldStratDataset
from src.models.baseline_srm import BaselineSRM
from src.models.temporal_srm import TemporalSRM

# SSIM calculation
try:
    from skimage.metrics import structural_similarity
    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False


# =========================================================
# Configuration
# =========================================================

DATASET_ROOT = PROJECT_ROOT / "src" / "datasets" / "worldstrat"
SPLIT_CSV = DATASET_ROOT / "stratified_train_val_test_split.csv"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

BASELINE_CHECKPOINT = CHECKPOINT_DIR / "baseline_best.pth"
TEMPORAL_CHECKPOINT = CHECKPOINT_DIR / "temporal_best.pth"

OUTPUT_CSV = PROJECT_ROOT / "test_model_comparison.csv"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 1
NUM_WORKERS = 0


# =========================================================
# Metrics
# =========================================================

def calculate_metrics(prediction, target):
    """
    Inputs: tensors shaped [1, 4, 500, 500].
    Both prediction and target should be in [0, 1].
    """

    pred = prediction.float().clamp(0, 1)
    target = target.float()

    mae = torch.mean(torch.abs(pred - target)).item()

    mse = torch.mean((pred - target) ** 2).item()

    psnr = 100.0 if mse <= 1e-12 else 10.0 * np.log10(1.0 / mse)

    ssim = None

    if SSIM_AVAILABLE:
        pred_np = pred[0].detach().cpu().numpy()
        target_np = target[0].detach().cpu().numpy()

        band_scores = []

        for band in range(pred_np.shape[0]):
            score = structural_similarity(
                target_np[band],
                pred_np[band],
                data_range=1.0,
                gaussian_weights=True,
                sigma=1.5,
                use_sample_covariance=False,
            )
            band_scores.append(score)

        ssim = float(np.mean(band_scores))

    return {
        "L1": mae,
        "PSNR": float(psnr),
        "SSIM": ssim,
    }


# =========================================================
# Checkpoint loading
# =========================================================

def load_checkpoint(model, checkpoint_path):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    saved_epoch = checkpoint.get("epoch", "unknown")

    print(f"Loaded: {checkpoint_path.name}")
    print(f"Checkpoint epoch: {saved_epoch}")

    return model


# =========================================================
# Main evaluation
# =========================================================

def main():
    print("=" * 72)
    print("BASELINE vs TEMPORAL SRM: TEST SET EVALUATION")
    print("=" * 72)

    print("Device:", DEVICE)

    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    if not SSIM_AVAILABLE:
        print(
            "\nSSIM unavailable. Install it with:\n"
            "pip install scikit-image\n"
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    print("Test AOIs:", len(test_dataset))

    # -----------------------------------------------------
    # Models
    # -----------------------------------------------------

    baseline = BaselineSRM(
        num_revisits=8,
        num_input_bands=12,
        num_output_bands=4,
        hidden_channels=64,
    ).to(DEVICE)

    temporal = TemporalSRM(
        num_revisits=8,
        num_input_bands=12,
        num_output_bands=4,
        hidden_channels=64,
        num_residual_blocks=4,
    ).to(DEVICE)

    baseline = load_checkpoint(
        baseline, BASELINE_CHECKPOINT
    )

    temporal = load_checkpoint(
        temporal, TEMPORAL_CHECKPOINT
    )

    # -----------------------------------------------------
    # Evaluation
    # -----------------------------------------------------

    rows = []
    start_time = time.time()

    with torch.inference_mode():
        for index, batch in enumerate(test_loader, start=1):

            lr = batch["lr"].to(
                DEVICE, non_blocking=True
            )

            hr = batch["hr"].to(
                DEVICE, non_blocking=True
            )

            aoi = batch["aoi"][0]

            # Run both models on identical input and target.
            with torch.amp.autocast(
                device_type=DEVICE.type,
                enabled=(DEVICE.type == "cuda"),
            ):
                baseline_pred = baseline(lr)
                temporal_pred = temporal(lr)

            baseline_metrics = calculate_metrics(
                baseline_pred, hr
            )

            temporal_metrics = calculate_metrics(
                temporal_pred, hr
            )

            row = {
                "AOI": aoi,
                "Baseline_L1": baseline_metrics["L1"],
                "Baseline_PSNR": baseline_metrics["PSNR"],
                "Temporal_L1": temporal_metrics["L1"],
                "Temporal_PSNR": temporal_metrics["PSNR"],
            }

            if SSIM_AVAILABLE:
                row["Baseline_SSIM"] = baseline_metrics["SSIM"]
                row["Temporal_SSIM"] = temporal_metrics["SSIM"]

            rows.append(row)

            if index % 25 == 0 or index == len(test_loader):
                print(
                    f"Evaluated {index}/{len(test_loader)} AOIs"
                )

            del baseline_pred, temporal_pred, lr, hr

    # -----------------------------------------------------
    # Save per-AOI results
    # -----------------------------------------------------

    if not rows:
        raise RuntimeError("No test samples were evaluated.")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    # -----------------------------------------------------
    # Aggregate results
    # -----------------------------------------------------

    print("\n" + "=" * 72)
    print("AGGREGATE TEST RESULTS")
    print("=" * 72)

    metric_names = ["L1", "PSNR"]
    if SSIM_AVAILABLE:
        metric_names.append("SSIM")

    for metric in metric_names:
        baseline_values = np.array(
            [row[f"Baseline_{metric}"] for row in rows],
            dtype=np.float64,
        )

        temporal_values = np.array(
            [row[f"Temporal_{metric}"] for row in rows],
            dtype=np.float64,
        )

        print(f"\n{metric}")
        print(
            f"  Baseline: mean={baseline_values.mean():.5f}, "
            f"std={baseline_values.std():.5f}"
        )
        print(
            f"  Temporal: mean={temporal_values.mean():.5f}, "
            f"std={temporal_values.std():.5f}"
        )

        if metric == "PSNR" or metric == "SSIM":
            difference = temporal_values - baseline_values
            print(
                f"  Temporal minus baseline: "
                f"{difference.mean():+.5f}"
            )
            print(
                f"  AOIs where temporal is better: "
                f"{np.sum(difference > 0)}/{len(rows)}"
            )

        elif metric == "L1":
            difference = baseline_values - temporal_values
            print(
                f"  Baseline minus temporal: "
                f"{difference.mean():+.5f} "
                f"(positive favors temporal)"
            )
            print(
                f"  AOIs where temporal is better: "
                f"{np.sum(difference > 0)}/{len(rows)}"
            )

    baseline_psnr = np.array(
        [row["Baseline_PSNR"] for row in rows]
    )
    temporal_psnr = np.array(
        [row["Temporal_PSNR"] for row in rows]
    )

    print("\nPSNR range")
    print(
        f"  Baseline: {baseline_psnr.min():.2f} to "
        f"{baseline_psnr.max():.2f} dB"
    )
    print(
        f"  Temporal: {temporal_psnr.min():.2f} to "
        f"{temporal_psnr.max():.2f} dB"
    )

    print(f"\nEvaluated AOIs: {len(rows)}")
    print(f"Elapsed time: {(time.time() - start_time) / 60:.1f} min")
    print(f"CSV saved to: {OUTPUT_CSV}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()