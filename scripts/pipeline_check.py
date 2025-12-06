import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from pathlib import Path
from src.dataset import TrainDataset, EvalDataset

def main():
    meta = Path("../data/metadata.csv")

    train_ds = TrainDataset(meta, split="train")
    val_ds_clean = EvalDataset(meta, split="val", jpeg_quality=100)
    val_ds_compressed = EvalDataset(meta, split="val", jpeg_quality=75)

    print("Train size:", len(train_ds))
    print("Val (clean) size:", len(val_ds_clean))
    print("Val (compressed) size:", len(val_ds_compressed))

    x_train, y_train = train_ds[0]
    print("Train sample:", x_train.shape, y_train)

    x_val, y_val = val_ds_clean[0]
    print("Val sample (clean):", x_val.shape, y_val)

if __name__ == "__main__":
    main()
