import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.model_noise import NoiseResidualBranch


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model = NoiseResidualBranch(
        img_size=256,
        kernels=None,
        use_minmax=True,
        compress_channels=32,
        cnn_channels=(64, 128),
        pool_alpha=0.5,
        out_dim=128,
        dropout=0.2,
        train_srm=False,
    )
    model = model.to(device)

    x = torch.rand(4, 3, 256, 256, device=device)

    with torch.no_grad():
        out = model(x)

    print("Output shape:", out.shape)
    print("Output device:", out.device)
    print("Output dtype:", out.dtype)


if __name__ == "__main__":
    main()
