from __future__ import annotations

import torch
import torch.nn as nn


class FrequencyBranch(nn.Module):
    def __init__(
        self,
        img_size: int = 256,
        max_radius: int = 127,
        out_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        assert img_size == 256
        self.img_size = img_size
        self.max_radius = max_radius
        self.out_dim = out_dim

        H = W = img_size
        cy, cx = H // 2, W // 2

        ys = torch.arange(H, dtype=torch.float32)
        xs = torch.arange(W, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")

        rr = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        rr_round = rr.round().to(torch.long)

        rr_round = rr_round.clamp(min=0, max=max_radius)

        radius_index_flat = rr_round.reshape(-1)

        counts = torch.bincount(radius_index_flat, minlength=max_radius + 1).float()

        self.n_bins = max_radius + 1

        self.register_buffer("radius_index", radius_index_flat)
        self.register_buffer("radial_counts", counts)    

        weights = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32)
        self.register_buffer("lum_weights", weights.view(1, 3, 1, 1))

        in_dim = self.n_bins - 1

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )

    def _radial_profile(self, mag: torch.Tensor) -> torch.Tensor:
        B, H, W = mag.shape
        assert H == self.img_size and W == self.img_size

        mag_flat = mag.reshape(B, -1)  # [B, H*W]

        idx = self.radius_index.unsqueeze(0).expand(B, -1)  # [B, H*W]

        acc = torch.zeros(
            B,
            self.n_bins,
            device=mag.device,
            dtype=mag.dtype,
        )
        acc.scatter_add_(dim=1, index=idx, src=mag_flat)

        prof = acc / (self.radial_counts.unsqueeze(0) + 1e-8)
        return prof

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C == 3
        assert H == self.img_size and W == self.img_size

        y = (x * self.lum_weights).sum(dim=1)
        y = y - y.mean(dim=(1, 2), keepdim=True)  

        spec = torch.fft.fft2(y)
        mag = spec.abs()
        mag_shift = torch.fft.fftshift(mag, dim=(-2, -1))

        prof = self._radial_profile(mag_shift) 

        prof_log = torch.log10(1.0 + prof)

        dprof = prof_log[:, 1:] - prof_log[:, :-1]

        f_freq = self.mlp(dprof)

        return f_freq
