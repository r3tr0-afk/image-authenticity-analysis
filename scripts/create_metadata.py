import csv
import random
from pathlib import Path


REAL_DIR = Path("../data/raw/real")
FAKE_DIR = Path("../data/raw/fake")
OUT_CSV = Path("../data/metadata.csv")

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

def list_images(folder):
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS]
    )

def split_paths(paths, seed=42):
    random.Random(seed).shuffle(paths)
    n = len(paths)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val
    train = paths[:n_train]
    val = paths[n_train:n_train + n_val]
    test = paths[n_train + n_val:]
    return train, val, test

def main():
    real_paths = list_images(REAL_DIR)
    fake_paths = list_images(FAKE_DIR)

    print(f"Found {len(real_paths)} real images")
    print(f"Found {len(fake_paths)} fake images")

    real_train, real_val, real_test = split_paths(real_paths, seed=42)
    fake_train, fake_val, fake_test = split_paths(fake_paths, seed=123)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUT_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label", "split"])

        for p in real_train:
            writer.writerow([str(p), "real", "train"])
        for p in real_val:
            writer.writerow([str(p), "real", "val"])
        for p in real_test:
            writer.writerow([str(p), "real", "test"])

        for p in fake_train:
            writer.writerow([str(p), "fake", "train"])
        for p in fake_val:
            writer.writerow([str(p), "fake", "val"])
        for p in fake_test:
            writer.writerow([str(p), "fake", "test"])

    print("Done. Metadata written to", OUT_CSV)

if __name__ == "__main__":
    main()
