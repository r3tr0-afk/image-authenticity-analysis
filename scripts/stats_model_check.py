import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image
import torchvision.transforms as T

from src.model_stats_n_color import StatsColorBranch

def load_image(path):
    tfm = T.Compose([
        T.ToTensor(),
    ])
    img = Image.open(path).convert("RGB")
    return tfm(img).unsqueeze(0)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model = StatsColorBranch().to(device)
    model.eval()

    img_real = load_image("../data/raw/real/real_000246.jpg").to(device)
    img_fake = load_image("../data/raw/fake/fake_000010.jpg").to(device)

    with torch.no_grad():
        f_real = model(img_real)
        f_fake = model(img_fake)

    print("Real output shape:", f_real.shape)
    print("Fake output shape:", f_fake.shape)

    print("Real stats:",
          f_real.min().item(),
          f_real.max().item(),
          f_real.mean().item(),
          f_real.std().item())

    print("Fake stats:",
          f_fake.min().item(),
          f_fake.max().item(),
          f_fake.mean().item(),
          f_fake.std().item())

if __name__ == "__main__":
    main()
