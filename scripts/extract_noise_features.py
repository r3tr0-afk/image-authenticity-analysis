import sys
from pathlib import Path
import csv
import numpy as np
from tqdm import tqdm

import torch
from torchvision import transforms
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.model_noise import NoiseResidualBranch

def pil_load(path, size=256):
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BICUBIC)
    arr = np.array(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2,0,1)
    return t

def main():
    METADATA_PATH = "../data/metadata.csv"
    OUT_DIR = "../data/noise_features"
    BATCH_SIZE = 32
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    metadata = Path(METADATA_PATH)
    outdir = Path(OUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    with metadata.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append((r["image_path"], r["label"]))

    model = NoiseResidualBranch().to(DEVICE)
    model.eval()

    features = []
    labels = []

    batch_paths = []
    batch_tensors = []
    
    current_idx = 0

    for img_path, label in tqdm(rows, desc="Images"):
        pth = Path(img_path)
        if not pth.exists():
            pth = PROJECT_ROOT / img_path
            if not pth.exists():
                tqdm.write(f"Missing: {img_path} — skipping")
                current_idx += 1 
                continue
        
        t = pil_load(str(pth)).unsqueeze(0)
        batch_paths.append(str(pth))
        batch_tensors.append(t)

        if len(batch_tensors) >= BATCH_SIZE:
            batch = torch.cat(batch_tensors, dim=0).to(DEVICE)
            with torch.no_grad():
                out = model(batch)
            features.append(out.cpu().numpy())
            
            end_idx = current_idx + 1
            start_idx = end_idx - len(batch_tensors)
            
            chunk_labels = []
            for i in range(len(batch_tensors)):
                r_idx = current_idx - (len(batch_tensors) - 1) + i
                lab = rows[r_idx][1]
                chunk_labels.append(1 if lab.lower().startswith("fake") else 0)
                
            labels.extend(chunk_labels)
            batch_paths, batch_tensors = [], []
        
        current_idx += 1

    if batch_tensors:
        batch = torch.cat(batch_tensors, dim=0).to(DEVICE)
        with torch.no_grad():
            out = model(batch)
        features.append(out.cpu().numpy())
        
        chunk_labels = []
        for i in range(len(batch_tensors)):
            r_idx = current_idx - len(batch_tensors) + i 
            lab = rows[r_idx][1]
            chunk_labels.append(1 if lab.lower().startswith("fake") else 0)
        labels.extend(chunk_labels)

    feats = np.vstack(features)
    labs = np.array(labels, dtype=np.int64)

    np.save(outdir / "noise_features.npy", feats)
    np.save(outdir / "noise_labels.npy", labs)
    print("Saved:", outdir / "noise_features.npy", feats.shape, labs.shape)

if __name__ == "__main__":
    main()