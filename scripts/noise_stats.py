from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model_noise import NoiseResidualBranch


def load_rgb(path: Path, size=256) -> torch.Tensor:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return t


def to_numpy_img(t: torch.Tensor) -> np.ndarray:
    a = t.copy()
    mn = float(a.min())
    mx = float(a.max())
    if mx - mn < 1e-6:
        res = np.zeros_like(a, dtype=np.uint8)
        return res
    a = (a - mn) / (mx - mn + 1e-12)
    a = (a * 255.0).round().astype(np.uint8)
    return a


def save_map(t: torch.Tensor, out_path: Path, cmap="gray"):
    if t.ndim == 3 and t.shape[0] == 1:
        arr = t[0].detach().cpu().numpy()
    elif t.ndim == 2:
        arr = t.detach().cpu().numpy()
    elif t.ndim == 4:
        arr = t[0, 0].detach().cpu().numpy()
    else:
        arr = t.squeeze().detach().cpu().numpy()
        
    im = to_numpy_img(arr)
    Image.fromarray(im).save(out_path)


def print_stats(name: str, x: torch.Tensor):
    x_cpu = x.detach().cpu()
    print(f"{name}: shape={tuple(x_cpu.shape)} min={float(x_cpu.min()):.6f} max={float(x_cpu.max()):.6f} mean={float(x_cpu.mean()):.6f} std={float(x_cpu.std()):.6f}")


def visualize_channel_grid(tensor: torch.Tensor, out_path: Path, max_ch=9):
    b, c, h, w = tensor.shape
    channels = min(c, max_ch)
    imgs = tensor[0, :channels]
    
    imgs_norm = []
    for i in range(imgs.shape[0]):
        ch = imgs[i]
        ch_np = ch.detach().cpu().numpy()
        ch_img = to_numpy_img(ch_np)
        imgs_norm.append(torch.from_numpy(ch_img).unsqueeze(0).float() / 255.0)
    
    batch_grid = torch.stack(imgs_norm, dim=0)
    
    grid = make_grid(batch_grid, nrow=channels, normalize=False)
    
    nd = (grid.permute(1, 2, 0).mul(255).byte().cpu().numpy())
    Image.fromarray(nd).save(out_path)


def main():
    REAL_IMG_PATH = "../data/raw/real/real_000246.jpg"
    FAKE_IMG_PATH = "../data/raw/fake/fake_000010.jpg"
    OUT_DIR = "../reports/noise_debug"
    DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"

    real_path = Path(REAL_IMG_PATH)
    fake_path = Path(FAKE_IMG_PATH)
    outdir = Path(OUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Device:", DEVICE_STR)
    device = torch.device(DEVICE_STR)

    model = NoiseResidualBranch()
    model = model.to(device)
    model.eval()

    for name, path in [("real", real_path), ("fake", fake_path)]:
        print("\n=== Processing", name, path)
        try:
            x = load_rgb(path).to(device)
        except Exception as e:
            print(f"Skipping {name}: {e}")
            continue

        if model._srm_conv is None or next(model._srm_conv.parameters()).device != device:
            model._build_srm_conv(device)

        srm_out = model._srm_conv(x)
        print_stats("SRM_out", srm_out)
        visualize_channel_grid(srm_out, outdir / f"{name}_srm_channels.png", max_ch=9)

        if model.use_minmax:
            mm = model._minmax_per_channel(x)
            print_stats("MinMax", mm)
            save_map(mm[0, 0], outdir / f"{name}_minmax_ch0.png")
            visualize_channel_grid(mm, outdir / f"{name}_minmax_grid.png", max_ch=3)
        else:
            mm = None

        combined = torch.cat([srm_out, mm], dim=1) if mm is not None else srm_out
        print_stats("Combined (pre-1x1)", combined)
        visualize_channel_grid(combined, outdir / f"{name}_combined_channels.png", max_ch=9)

        z = model.conv1x1(combined)
        print_stats("After 1x1 (pre-BN)", z)
        z_bn = model.bn0(z)
        z_act = F.gelu(z_bn)
        print_stats("After BN + GELU", z_act)
        save_map(z_act[0, 0], outdir / f"{name}_compressed_ch0.png")
        visualize_channel_grid(z_act, outdir / f"{name}_compressed_grid.png", max_ch=9)

        h = model.conv_block1(z_act)
        h = model.conv_block2(h)
        print_stats("After conv blocks", h)
        visualize_channel_grid(h, outdir / f"{name}_after_conv_blocks.png", max_ch=9)

        gmp = F.adaptive_max_pool2d(h, (1, 1)).reshape(1, -1)
        gap = F.adaptive_avg_pool2d(h, (1, 1)).reshape(1, -1)
        pooled = model.pool_alpha * gmp + (1.0 - model.pool_alpha) * gap
        print_stats("GMP pooled", gmp)
        print_stats("GAP pooled", gap)
        print_stats("Hybrid pooled", pooled)

        out = model.projector(pooled)
        print_stats("Final f_noise", out)

        stats = {
            "srm_min": float(srm_out.min().cpu()), "srm_max": float(srm_out.max().cpu()),
            "srm_mean": float(srm_out.mean().cpu()), "srm_std": float(srm_out.std().cpu()),
            "mm_min": float(mm.min().cpu()) if mm is not None else None,
            "mm_max": float(mm.max().cpu()) if mm is not None else None,
            "pooled_mean": float(pooled.mean().cpu()),
            "pooled_std": float(pooled.std().cpu()),
            "final_mean": float(out.mean().cpu()),
            "final_std": float(out.std().cpu()),
        }
        np.savez(outdir / f"{name}_stats.npz", **stats)
        print("Saved debug images + stats to", outdir)
        
        
if __name__ == "__main__":
    main()