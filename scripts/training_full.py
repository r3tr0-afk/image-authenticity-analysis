import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
from sklearn.metrics import roc_auc_score

from src.dataset import TrainDataset, EvalDataset
from src.model_full import FullForensicModel


CONFIG = {
    "csv_path": str(PROJECT_ROOT / "data" / "metadata.csv"),
    "batch_size": 8,
    "max_epochs": 20,
    "patience": 5,
    "learning_rate": 1e-4,
    "num_workers": 0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def train_one_epoch(model, loader, optimizer, criterion, device, epoch_idx):
    model.train()
    losses = []

    for i, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.float().to(device, non_blocking=True).unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)
        logit, _ = model(x)
        loss = criterion(logit, y)

        if not torch.isfinite(loss):
            print("Non-finite loss detected, skipping batch")
            continue

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if i % 26 == 0:
            print(
                f"Epoch {epoch_idx+1} | "
                f"Batch {i}/{len(loader)} | "
                f"Loss: {loss.item():.4f}"
            )

    return float(np.mean(losses))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)

        B, n_crops, C, H, W = x.shape
        x_flat = x.view(-1, C, H, W)

        logits_flat, _ = model(x_flat)
        logits_grouped = logits_flat.view(B, n_crops)
        logit_avg = logits_grouped.mean(dim=1)

        all_logits.append(logit_avg.cpu())
        all_labels.append(y)

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()

    if len(np.unique(labels)) < 2:
        return 0.5

    probs = 1.0 / (1.0 + np.exp(-logits))
    return float(roc_auc_score(labels, probs))


def main():
    device = CONFIG["device"]
    print(f"Running on device: {device}")

    train_ds = TrainDataset(CONFIG["csv_path"], split="train")
    val_ds   = EvalDataset(CONFIG["csv_path"], split="val", jpeg_quality=100)

    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
    )

    model = FullForensicModel(device=device).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=1e-4,
    )

    best_val_auc = 0.0
    epochs_no_improve = 0
    best_model_path = PROJECT_ROOT / "reports" / "best_model.pth"

    print("Starting full training")

    for epoch in range(CONFIG["max_epochs"]):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )

        val_auc = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{CONFIG['max_epochs']} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val AUC: {val_auc:.4f}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved (AUC={best_val_auc:.4f})")
        else:
            epochs_no_improve += 1
            print(f"No improvement ({epochs_no_improve}/{CONFIG['patience']})")

        if epochs_no_improve >= CONFIG["patience"]:
            print("Early stopping triggered")
            break

    print(f"Training finished | Best Val AUC: {best_val_auc:.4f}")


if __name__ == "__main__":
    main()
