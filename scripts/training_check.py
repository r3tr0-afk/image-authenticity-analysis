import sys
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


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    losses = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.float().to(device, non_blocking=True).unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)
        logit, _ = model(x)
        loss = criterion(logit, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

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

    final_logits = torch.cat(all_logits).numpy()
    final_labels = torch.cat(all_labels).numpy()

    if len(np.unique(final_labels)) < 2:
        return 0.5

    probs = 1.0 / (1.0 + np.exp(-final_logits))
    return float(roc_auc_score(final_labels, probs))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_ds = TrainDataset("../data/metadata_debug.csv", split="train")
    val_ds   = EvalDataset("../data/metadata_debug.csv", split="val", jpeg_quality=100)

    train_loader = DataLoader(
        train_ds,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=4, 
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    model = FullForensicModel(device=device)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )

    for epoch in range(2):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_auc = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch+1} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val AUC (5-Crop): {val_auc:.4f}"
        )


if __name__ == "__main__":
    main()