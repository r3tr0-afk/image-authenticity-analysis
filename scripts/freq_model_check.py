import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.model_frequency import FrequencyBranch


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model = FrequencyBranch(img_size=256, max_radius=127, out_dim=64)
    model = model.to(device)

    x = torch.rand(4, 3, 256, 256, device=device)

    with torch.no_grad():
        out = model(x)

    print("Output shape:", out.shape)
    print("Output device:", out.device)
    print("Output dtype:", out.dtype)


if __name__ == "__main__":
    main()
