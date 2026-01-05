import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

from src.dataset import EvalDataset
from src.model_full import FullForensicModel


@torch.no_grad()
def evaluate_test(model, loader, device):
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

    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)

    auc = roc_auc_score(labels, probs)
    acc = accuracy_score(labels, preds)

    return auc, acc


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("test evaluation on device:", device)

    test_ds = EvalDataset(
        PROJECT_ROOT / "data" / "metadata.csv",
        split="test",
        jpeg_quality=100
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    model = FullForensicModel(device=device).to(device)

    ckpt_path = PROJECT_ROOT / "reports" / "best_model.pth"
    assert ckpt_path.exists(), "best model not found"

    
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)
    print("loaded best model")

    test_auc, test_acc = evaluate_test(model, test_loader, device)

    print(f"Test ROC-AUC : {test_auc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
