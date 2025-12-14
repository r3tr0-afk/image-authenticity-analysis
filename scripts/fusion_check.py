import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.model_fusion import GatedFusion


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    B = 4

    f_clip  = torch.randn(B, 256).to(device)
    f_freq  = torch.randn(B, 64).to(device)
    f_noise = torch.randn(B, 128).to(device)
    f_stats = torch.randn(B, 32).to(device)

    model = GatedFusion().to(device)
    model.eval()

    with torch.no_grad():
        logit, alphas = model(f_clip, f_freq, f_noise, f_stats)

    print("Logit shape:", logit.shape)
    print("Attention shape:", alphas.shape)
    print("Attention sums:", alphas.sum(dim=1))


if __name__ == "__main__":
    main()
