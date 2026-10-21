
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.worldstrat_dataset import WorldStratDataset
from src.models.temporal_srm import TemporalSRM


# =========================================================
# CONFIGURATION
# =========================================================

DATASET_ROOT = PROJECT_ROOT / "src" / "datasets" / "worldstrat"

SPLIT_CSV = DATASET_ROOT / "stratified_train_val_test_split.csv"

CHECKPOINT = PROJECT_ROOT / "checkpoints" / "temporal_best.pth"

SAMPLE_INDEX = 10     # Change to 25, 50, etc.
NUM_REVISITS = 8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Sentinel-2 normalization constants
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


# =========================================================
# SELF-ENSEMBLE INFERENCE
# =========================================================

def self_ensemble_predict(model, lr, device):
    """
    Run inference on 4 geometric transforms of the input (identity,
    horizontal flip, vertical flip, both), undo each transform on the
    output, and average. Reduces directional bias in the model's
    predictions - standard practice in SR literature (used since
    EDSR). Costs 4x inference time, zero retraining, no model changes.
    """
    transforms = [
        (lambda t: t, lambda t: t),
        (lambda t: t.flip(-1), lambda t: t.flip(-1)),
        (lambda t: t.flip(-2), lambda t: t.flip(-2)),
        (lambda t: t.flip(-1).flip(-2), lambda t: t.flip(-1).flip(-2)),
    ]

    predictions = []
    with torch.inference_mode():
        for fwd, inv in transforms:
            lr_t = fwd(lr)
            pred = model(lr_t, return_attention=False)
            predictions.append(inv(pred))

    return torch.stack(predictions, dim=0).mean(dim=0)
# =========================================================
# VISUALIZATION HELPERS
# =========================================================


def stretch_rgb(img, low=0.0, high=0.25):
    """Use the same display range for every image."""

    img = np.asarray(img, dtype=np.float32)

    img = np.nan_to_num(
        img,
        nan=0.0,
        posinf=high,
        neginf=low
    )

    return np.clip(
        (img - low) / (high - low),
        0.0,
        1.0
    )


def make_rgb(image):
    """Convert a [4, H, W] RGBN image to a displayable RGB image."""

    rgb = np.transpose(image[[0, 1, 2]], (1, 2, 0))
    return stretch_rgb(rgb)


# =========================================================
# METRICS
# =========================================================

def calculate_metrics(prediction, target, label):
    """
    Calculate PSNR and SSIM when prediction and target
    share the same shape and compatible value ranges.

    Assumes reflectance values are expected in [0, 1].
    Does not silently clip predictions.
    """

    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    print(f"\n{label}")
    print("-" * 45)

    if prediction.shape != target.shape:
        print("Cannot calculate metrics: shape mismatch.")
        print("Prediction:", prediction.shape)
        print("Target:", target.shape)
        return None

    if not (
        np.isfinite(prediction).all()
        and np.isfinite(target).all()
    ):
        print("Cannot calculate metrics: NaN/Inf values found.")
        return None

    print(
        f"Prediction range: {prediction.min():.6f} "
        f"to {prediction.max():.6f}"
    )
    print(
        f"Target range:     {target.min():.6f} "
        f"to {target.max():.6f}"
    )

    # This script expects reflectance data on a [0, 1] scale.
    if (
        target.min() < -1e-4
        or target.max() > 1.0001
    ):
        print(
            "WARNING: target is outside [0, 1]. "
            "Verify the dataset's value scale before trusting PSNR."
        )

    mse = np.mean((prediction - target) ** 2)

    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 10.0 * np.log10(1.0 / mse)

    print(f"MSE:  {mse:.8f}")
    print(f"PSNR: {psnr:.3f} dB (data_range=1.0)")

    # SSIM is optional; install scikit-image if needed.
    try:
        from skimage.metrics import structural_similarity

        ssim_values = []

        for channel in range(target.shape[0]):
            value = structural_similarity(
                target[channel],
                prediction[channel],
                data_range=1.0
            )
            ssim_values.append(value)

        mean_ssim = float(np.mean(ssim_values))
        print(f"SSIM: {mean_ssim:.5f} (mean across bands)")

    except ImportError:
        mean_ssim = None
        print("SSIM unavailable. Install with: pip install scikit-image")

    return {
        "psnr": float(psnr),
        "ssim": mean_ssim,
        "mse": float(mse)
    }


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 72)
    print("TEMPORAL SRM: TEST AOI + BICUBIC COMPARISON")
    print("=" * 72)

    print("Device:", DEVICE)

    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    if not SPLIT_CSV.exists():
        raise FileNotFoundError(f"Split file not found: {SPLIT_CSV}")

    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    # -----------------------------------------------------
    # 1. Load test dataset
    # -----------------------------------------------------

    dataset = WorldStratDataset(
        root=DATASET_ROOT,
        split_csv=SPLIT_CSV,
        split="test",
        revisits=NUM_REVISITS,
        input_size=160,
        output_size=500,
    )

    print("\nTest AOIs:", len(dataset))

    if not 0 <= SAMPLE_INDEX < len(dataset):
        raise IndexError(
            f"SAMPLE_INDEX={SAMPLE_INDEX} is invalid. "
            f"Choose an index from 0 to {len(dataset) - 1}."
        )

    sample = dataset[SAMPLE_INDEX]

    lr_sample = sample["lr"].float()
    hr = sample["hr"].detach().cpu().numpy().astype(np.float32)

    print("Selected index:", SAMPLE_INDEX)
    print("AOI:", sample["aoi"])
    print("LR tensor shape:", tuple(lr_sample.shape))
    print("HR tensor shape:", tuple(sample["hr"].shape))

    # Expected LR shape: [revisits, 12, H, W]
    # Expected HR shape: [4, H, W]
    if lr_sample.ndim != 4 or lr_sample.shape[1] != 12:
        raise ValueError(
            "Unexpected LR shape. Expected [revisits, 12, H, W], "
            f"got {tuple(lr_sample.shape)}"
        )

    if hr.ndim != 3 or hr.shape[0] != 4:
        raise ValueError(
            "Unexpected HR shape. Expected [4, H, W], "
            f"got {hr.shape}"
        )

    lr = lr_sample.unsqueeze(0).to(DEVICE)

    # -----------------------------------------------------
    # 2. Load model and checkpoint
    # -----------------------------------------------------

    model = TemporalSRM(
        num_revisits=NUM_REVISITS,
        num_input_bands=12,
        num_output_bands=4,
        hidden_channels=64,
        num_residual_blocks=4,
    ).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False
    )

    state_dict = checkpoint.get(
        "model_state_dict", checkpoint
    )

    model.load_state_dict(state_dict)
    model.eval()

    print("\nLoaded checkpoint:", CHECKPOINT)

    if isinstance(checkpoint, dict):
        if "epoch" in checkpoint:
            print("Checkpoint epoch:", checkpoint["epoch"])
        if "best_val_loss" in checkpoint:
            print("Best validation loss:", checkpoint["best_val_loss"])

    # -----------------------------------------------------
    # 3. Run temporal SRM inference
    # -----------------------------------------------------

    # Self-ensembled prediction (for the reported metrics/image) -
    # averages 4 geometric transforms, no attention map available
    # from this path since it's not tied to one single forward pass.
    prediction_tensor = self_ensemble_predict(model, lr, DEVICE)

    # Separate, single plain forward pass just to get the attention
    # map for the attention visualization later in this script -
    # doesn't affect the reported prediction/metrics above.
    with torch.inference_mode():
        _, attention_tensor = model(lr, return_attention=True)

    prediction = (
        prediction_tensor[0]
        .detach().cpu().numpy().astype(np.float32)
    )

    attention = (
        attention_tensor[0]
        .detach().cpu().numpy().astype(np.float32)
    )

    print("\nPrediction shape:", prediction.shape)
    print("Attention shape:", attention.shape)

    print(
        "Prediction range:",
        float(np.nanmin(prediction)),
        "to",
        float(np.nanmax(prediction))
    )

    print(
        "HR range:",
        float(np.nanmin(hr)),
        "to",
        float(np.nanmax(hr))
    )

    # -----------------------------------------------------
    # 4. Build bicubic baseline
    # -----------------------------------------------------
    #
    # Assumes the 12 LR bands follow the common order:
    # B01, B02, B03, B04, B05, B06, B07, B08,
    # B8A, B09, B11, B12.
    #
    # HR is assumed to have RGBN channel order:
    # Red, Green, Blue, Near-Infrared.
    #
    # -----------------------------------------------------

    lr_mean = torch.tensor(
        MEAN, dtype=lr_sample.dtype
    )[:, None, None]

    lr_std = torch.tensor(
        STD, dtype=lr_sample.dtype
    )[:, None, None]

    # Reverse normalization for every revisit.
    lr_raw = lr_sample.cpu() * lr_std + lr_mean

    # Use the first revisit for the single-image baseline.
    # S2 indices: B04=3, B03=2, B02=1, B08=7.
    selected_bands = lr_raw[0, [3, 2, 1, 7]]

    # Resize to exactly match HR spatial dimensions.
    bicubic_tensor = F.interpolate(
        selected_bands.unsqueeze(0),
        size=hr.shape[-2:],
        mode="bicubic",
        align_corners=False
    )[0]

    bicubic = bicubic_tensor.numpy().astype(np.float32)

    print("\nBicubic shape:", bicubic.shape)

    # -----------------------------------------------------
    # 5. Evaluate both predictions
    # -----------------------------------------------------

    print("\n" + "=" * 72)
    print("NUMERICAL EVALUATION")
    print("=" * 72)

    bicubic_metrics = calculate_metrics(
        bicubic, hr, "BICUBIC BASELINE"
    )

    srm_metrics = calculate_metrics(
        prediction, hr, "TEMPORAL SRM"
    )

    if bicubic_metrics and srm_metrics:
        print("\n" + "=" * 45)
        print("MODEL VS BICUBIC")
        print("=" * 45)

        delta = (
            srm_metrics["psnr"]
            - bicubic_metrics["psnr"]
        )

        print(f"Bicubic PSNR: {bicubic_metrics['psnr']:.3f} dB")
        print(f"SRM PSNR:     {srm_metrics['psnr']:.3f} dB")
        print(f"PSNR change:  {delta:+.3f} dB")

        if delta > 0:
            print("SRM has higher PSNR on this AOI.")
        elif delta < 0:
            print("Bicubic has higher PSNR on this AOI.")
        else:
            print("Both have the same PSNR on this AOI.")

    # -----------------------------------------------------
    # 6. Visualize four images
    # -----------------------------------------------------

    lr_first = lr_raw[0].numpy()

    # Display Sentinel-2 B04, B03, B02.
    lr_rgb = stretch_rgb(
    np.transpose(lr_first[[3, 2, 1]], (1, 2, 0)),
    low=0.0,
    high=0.35
)

    bicubic_rgb = make_rgb(bicubic)
    pred_rgb = make_rgb(prediction)
    hr_rgb = make_rgb(hr)

    fig, axes = plt.subplots(1, 4, figsize=(20, 6))

    axes[0].imshow(lr_rgb)
    axes[0].set_title("Sentinel-2 input\nFirst revisit")

    axes[1].imshow(bicubic_rgb)
    axes[1].set_title(
        f"Bicubic baseline\n"
        f"PSNR: {bicubic_metrics['psnr']:.2f} dB"
        if bicubic_metrics else "Bicubic baseline"
    )

    axes[2].imshow(pred_rgb)
    axes[2].set_title(
        f"Temporal SRM\n"
        f"PSNR: {srm_metrics['psnr']:.2f} dB"
        if srm_metrics else "Temporal SRM"
    )

    axes[3].imshow(hr_rgb)
    axes[3].set_title("SPOT HR ground truth")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"WorldStrat Test AOI: {sample['aoi']} | "
        f"Test index: {SAMPLE_INDEX}",
        fontsize=13
    )

    plt.tight_layout()

    comparison_path = (
        PROJECT_ROOT / "temporal_visual_comparison.png"
    )

    plt.savefig(
        comparison_path,
        dpi=180,
        bbox_inches="tight"
    )
    plt.close(fig)

    print("\nSaved comparison:", comparison_path)

    # -----------------------------------------------------
    # 7. Visualize attention maps
    # -----------------------------------------------------

    if attention.ndim == 3 and attention.shape[0] == NUM_REVISITS:

        fig, axes = plt.subplots(
            2, 4, figsize=(15, 8)
        )

        for i, ax in enumerate(axes.flat):
            im = ax.imshow(attention[i], cmap="viridis")
            ax.set_title(f"Revisit {i + 1}")
            ax.axis("off")

            fig.colorbar(
                im, ax=ax, fraction=0.046, pad=0.04
            )

        fig.suptitle("Temporal Attention Maps")
        plt.tight_layout()

        attention_path = (
            PROJECT_ROOT / "temporal_attention_maps.png"
        )

        plt.savefig(
            attention_path,
            dpi=160,
            bbox_inches="tight"
        )
        plt.close(fig)

        print("Saved attention maps:", attention_path)

        print("\nMean attention per revisit:")

        mean_attention = attention.mean(axis=(1, 2))

        for i, value in enumerate(mean_attention):
            print(f"  Revisit {i + 1}: {value:.5f}")

    else:
        print(
            "\nAttention shape differs from [8, H, W]. "
            "Skipping attention-map plotting."
        )

    print("\n" + "=" * 72)
    print("VISUALIZATION COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
