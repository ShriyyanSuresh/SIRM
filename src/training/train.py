"""
SIRM training entry point.

    python -m src.training.train --config configs/baseline.yaml

Implements: AMP mixed precision, gradient accumulation, cosine annealing
with warm restarts, checkpointing (best + last), early stopping, and
per-epoch validation metrics. Deliberately straightforward - this is the
"strong reproducible baseline" step, not the full research pipeline with
uncertainty/degradation-consistency/downstream tasks (those come after
this trains correctly).
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.datasets.split_utils import clean_split_csv
from src.datasets.worldstrat_dataset import WorldStratDataset
from src.models.sirm_fusion_lite import build_model
from src.losses.losses import SIRMLoss
from src.evaluation.metrics import evaluate_batch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_dataloaders(cfg: dict):
    d = cfg["data"]
    split_csv_path = os.path.join(d["root"], d["split_csv"])
    clean_df, report = clean_split_csv(split_csv_path)
    print("[split cleanup]", json.dumps(report.__dict__, default=str)[:500], "...")

    id_col = clean_df.columns[0]
    split_col = [c for c in clean_df.columns if c != id_col][0]

    common_kwargs = dict(
        root=d["root"],
        split_df=clean_df,
        id_col=id_col,
        split_col=split_col,
        hr_dir=d["hr_dir"],
        lr_dir=d["lr_dir"],
        input_band_indices=d["input_band_indices"],
        num_revisits=d["num_revisits"],
        revisit_selection=d["revisit_selection"],
        lr_patch_size=d["lr_patch_size"],
        hr_patch_size=d["hr_patch_size"],
    )

    train_ds = WorldStratDataset(split_name=d["train_split_name"], train_mode=True, **common_kwargs)
    val_ds = WorldStratDataset(split_name=d["val_split_name"], train_mode=False, **common_kwargs)

    print(f"Train tiles: {len(train_ds)} | Val tiles: {len(val_ds)}")

    t = cfg["training"]
    train_loader = DataLoader(
        train_ds, batch_size=t["batch_size"], shuffle=True,
        num_workers=t["num_workers"], pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=t["batch_size"], shuffle=False,
        num_workers=t["num_workers"], pin_memory=True,
    )
    return train_loader, val_loader


def run_validation(model, val_loader, loss_fn, device):
    model.eval()
    agg = {}
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            lr = batch["lr"].to(device, non_blocking=True)
            hr = batch["hr"].to(device, non_blocking=True)
            target_hw = (hr.shape[-2], hr.shape[-1])
            pred, _ = model(lr, target_hw=target_hw)
            _, components = loss_fn(pred, hr)
            metrics = evaluate_batch(pred, hr)
            for k, v in {**components, **{f"metric_{k}": v for k, v in metrics.items() if k != "per_band_rmse"}}.items():
                agg[k] = agg.get(k, 0.0) + v
            n += 1
    return {k: v / max(n, 1) for k, v in agg.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--resume", default=None, help="path to a checkpoint to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = build_dataloaders(cfg)

    model = build_model(cfg).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    loss_fn = SIRMLoss(**{
        "weight_mse": cfg["loss"]["weight_mse"],
        "weight_mae": cfg["loss"]["weight_mae"],
        "weight_ssim": cfg["loss"]["weight_ssim"],
        "weight_sam": cfg["loss"]["weight_sam"],
        "weight_degradation": cfg["loss"]["weight_degradation_consistency"],
    })

    t = cfg["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=t["cosine_t0"])
    scaler = torch.cuda.amp.GradScaler(enabled=t["amp"])

    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", best_val_loss)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    os.makedirs(t["checkpoint_dir"], exist_ok=True)

    for epoch in range(start_epoch, t["max_epochs"]):
        model.train()
        optimizer.zero_grad()
        running_loss = 0.0

        for step, batch in enumerate(train_loader):
            lr = batch["lr"].to(device, non_blocking=True)
            hr = batch["hr"].to(device, non_blocking=True)
            target_hw = (hr.shape[-2], hr.shape[-1])

            with torch.cuda.amp.autocast(enabled=t["amp"]):
                pred, _ = model(lr, target_hw=target_hw)
                loss, components = loss_fn(pred, hr)
                loss = loss / t["grad_accumulation_steps"]

            scaler.scale(loss).backward()

            if (step + 1) % t["grad_accumulation_steps"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip_norm"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += components["total"]

            if step % t["log_every_n_steps"] == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)} loss {components['total']:.4f}")

        scheduler.step()
        avg_train_loss = running_loss / max(len(train_loader), 1)

        if epoch % t["val_every_n_epochs"] == 0:
            val_metrics = run_validation(model, val_loader, loss_fn, device)
            print(f"[epoch {epoch}] train_loss={avg_train_loss:.4f} val={json.dumps(val_metrics, indent=None)}")

            val_loss = val_metrics.get("total", avg_train_loss)
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "config": cfg,
            }
            torch.save(ckpt, os.path.join(t["checkpoint_dir"], "last.pt"))
            if is_best:
                torch.save(ckpt, os.path.join(t["checkpoint_dir"], "best.pt"))
                print(f"  -> new best checkpoint (val_loss={val_loss:.4f})")

            if patience_counter >= t["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience_counter} validations)")
                break

    print("Training complete.")


if __name__ == "__main__":
    main()
