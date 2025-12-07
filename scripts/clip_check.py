import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.model_clip import CLIPBranch


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPBranch(device=device)

    x = torch.randn(4, 3, 256, 256)

    out = model(x)
    print("Output shape:", out.shape)
    print("Output device:", out.device)
    print("Output dtype:", out.dtype)


if __name__ == "__main__":
    main()
