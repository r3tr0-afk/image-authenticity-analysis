import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T

from src.model_stats_n_color import StatsColorBranch


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model = StatsColorBranch().to(device)
    model.eval()

    tfm = T.Compose([T.ToTensor()])

    meta = pd.read_csv("../data/metadata.csv")

    X, y = [], []

    with torch.no_grad():
        for _, row in tqdm(meta.iterrows(), total=len(meta)):
            img = Image.open(row["image_path"]).convert("RGB")
            x = tfm(img).unsqueeze(0).to(device)

            f = model(x).cpu().numpy().squeeze()
            X.append(f)
            y.append(1 if row["label"] == "fake" else 0)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    np.save("../data/stats_features/X_stats.npy", X)
    np.save("../data/stats_features/y.npy", y)

if __name__ == "__main__":
    main()
