from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _default_kernels() -> List[torch.Tensor]:
    import numpy as np

    KV3 = np.array([
        [-1,  0,  1],
        [-1,  0,  1],
        [-1,  0,  1],
    ], dtype=np.float32)

    KB3 = np.array([
        [-1, -1, -1],
        [ 0,  0,  0],
        [ 1,  1,  1],
    ], dtype=np.float32)

    SPAM11 = np.array([
        [ 1., -1.,  1.],
        [-1.,  1., -1.],
        [ 1., -1.,  1.],
    ], dtype=np.float32)

    SQUARE_3 = np.array([
        [ 0., -1.,  0.],
        [-1.,  4., -1.],
        [ 0., -1.,  0.],
    ], dtype=np.float32)

    arrs = [KV3, KB3, SPAM11, SQUARE_3]
    return [torch.from_numpy(a).float() for a in arrs]


class NoiseResidualBranch(nn.Module):
    def __init__(
        self,
        img_size: int = 256,
        kernels: Optional[List[torch.Tensor]] = None,
        use_minmax: bool = True,
        compress_channels: int = 32,
        cnn_channels: List[int] = (64, 128),
        pool_alpha: float = 0.5,
        out_dim: int = 128,
        dropout: float = 0.2,
        train_srm: bool = False,
    ) -> None:
        super().__init__()
        assert img_size == 256
        self.img_size = img_size
        self.use_minmax = use_minmax
        self.pool_alpha = float(pool_alpha)
        self.out_dim = out_dim
        self.train_srm = bool(train_srm)

        if kernels is None:
            kernels = _default_kernels()
        kernels = [k if isinstance(k, torch.Tensor) else torch.tensor(k, dtype=torch.float32) for k in kernels]
        self.K = len(kernels)
        self.kernel_size = int(kernels[0].shape[0])

        self.register_buffer("srm_kernels_count", torch.tensor(self.K, dtype=torch.long))

        self._kernels_prototype = kernels

        in_ch = self.K * 3 + (3 if use_minmax else 0)

        self.conv1x1 = nn.Conv2d(in_ch, compress_channels, kernel_size=1, bias=False)
        self.bn0 = nn.BatchNorm2d(compress_channels)

        c1 = cnn_channels[0]
        c2 = cnn_channels[1]
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(compress_channels, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.GELU(),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.GELU(),
        )

        self.projector = nn.Sequential(
            nn.Linear(c2, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
        )

        self._srm_conv: Optional[nn.Conv2d] = None
        self._srm_kernels = kernels

    def _build_srm_conv(self, device: torch.device):
        K = self.K
        k = self.kernel_size
        in_ch = 3
        out_ch = K * 3

        W = torch.zeros((out_ch, in_ch, k, k), dtype=torch.float32, device=device)

        for i, kernel in enumerate(self._srm_kernels):
            kern = kernel.to(device=device, dtype=torch.float32)
            for c in range(3):
                W[i * 3 + c, c, :, :] = kern

        conv = nn.Conv2d(in_channels=3, out_channels=out_ch, kernel_size=k, padding=k // 2, bias=False)
        conv.weight.data.copy_(W)
        conv.weight.requires_grad = bool(self.train_srm)
        if not self.train_srm:
            conv.eval()
        conv.to(device)
        self._srm_conv = conv
        return conv

    @staticmethod
    def _minmax_per_channel(x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        dil = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        ero = -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)
        mm = dil - ero
        return mm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C == 3 and H == self.img_size and W == self.img_size

        device = x.device
        if self._srm_conv is None or next(self._srm_conv.parameters()).device != device:
            self._build_srm_conv(device=device)

        srm_out = self._srm_conv(x)

        if self.use_minmax:
            mm = self._minmax_per_channel(x)
            cat = torch.cat([srm_out, mm], dim=1)
        else:
            cat = srm_out

        z = self.conv1x1(cat)
        z = self.bn0(z)
        z = F.gelu(z)

        z = self.conv_block1(z)
        z = self.conv_block2(z)

        gmp = F.adaptive_max_pool2d(z, (1, 1)).reshape(B, -1)
        gap = F.adaptive_avg_pool2d(z, (1, 1)).reshape(B, -1)
        pooled = self.pool_alpha * gmp + (1.0 - self.pool_alpha) * gap

        out = self.projector(pooled)
        return out