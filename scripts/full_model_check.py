import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.model_full import FullForensicModel


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model = FullForensicModel(device=device)
    model.eval()

    x = torch.rand(2, 3, 256, 256).to(device)

    with torch.no_grad():
        logit, alphas = model(x)

    print("Logit shape:", logit.shape)
    print("Attention shape:", alphas.shape)
    print("Attention sums:", alphas.sum(dim=1))

if __name__ == "__main__":
    main()
