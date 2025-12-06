from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from .pipelines import preprocess_train, preprocess_eval_five_crops


LABEL_MAP = {"real": 0, "fake": 1}


@dataclass
class Record:
    path: Path
    label: int


class TrainDataset(Dataset):
    def __init__(self, metadata_csv: str | Path, split: str = "train"):
        self.metadata_csv = Path(metadata_csv)
        df = pd.read_csv(self.metadata_csv)

        if "image_path" not in df.columns or "label" not in df.columns or "split" not in df.columns:
            raise ValueError("metadata.csv must contain 'image_path', 'label', 'split' columns")

        df = df[df["split"] == split].copy()
        if df.empty:
            raise ValueError(f"No rows found for split={split!r} in {self.metadata_csv}")

        self.records: List[Record] = []
        for _, row in df.iterrows():
            rel_path = Path(str(row["image_path"]))
            path = rel_path if rel_path.is_absolute() else Path(".") / rel_path

            label_str = str(row["label"]).strip().lower()
            if label_str not in LABEL_MAP:
                raise ValueError(f"Unknown label: {label_str}")
            label = LABEL_MAP[label_str]

            self.records.append(Record(path=path, label=label))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        rec = self.records[idx]
        img_path = rec.path
        label = rec.label

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")
        x = preprocess_train(img)
        return x, label


class EvalDataset(Dataset):
    def __init__(self, metadata_csv: str | Path, split: str, jpeg_quality: int = 100):
        self.metadata_csv = Path(metadata_csv)
        self.split = split
        self.jpeg_quality = jpeg_quality

        df = pd.read_csv(self.metadata_csv)

        if "image_path" not in df.columns or "label" not in df.columns or "split" not in df.columns:
            raise ValueError("metadata.csv must contain 'image_path', 'label', 'split' columns")

        df = df[df["split"] == split].copy()
        if df.empty:
            raise ValueError(f"No rows found for split={split!r} in {self.metadata_csv}")

        self.records: List[Record] = []
        for _, row in df.iterrows():
            rel_path = Path(str(row["image_path"]))
            path = rel_path if rel_path.is_absolute() else Path(".") / rel_path

            label_str = str(row["label"]).strip().lower()
            if label_str not in LABEL_MAP:
                raise ValueError(f"Unknown label: {label_str}")
            label = LABEL_MAP[label_str]

            self.records.append(Record(path=path, label=label))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        rec = self.records[idx]
        img_path = rec.path
        label = rec.label

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")
        crops = preprocess_eval_five_crops(img, crop_size=256, jpeg_quality=self.jpeg_quality)
        # Stack into [5, 3, 256, 256]
        x = torch.stack(crops, dim=0)
        return x, label
