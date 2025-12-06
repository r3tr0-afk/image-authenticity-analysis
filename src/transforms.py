from __future__ import annotations

import io
import random
from typing import List

import torch
from PIL import Image, ImageFilter
from torchvision import transforms as T
import torchvision.transforms.functional as F


class EnsureMinSize:
    def __init__(self, min_size: int = 256, padding_mode: str = "reflect"):
        self.min_size = min_size
        self.padding_mode = padding_mode

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if min(w, h) >= self.min_size:
            return img

        pad_w = max(0, self.min_size - w)
        pad_h = max(0, self.min_size - h)

        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top

        if self.padding_mode == "reflect":
            pad = T.Pad((left, top, right, bottom), padding_mode="reflect")
        else:
            pad = T.Pad((left, top, right, bottom), fill=0, padding_mode="constant")

        return pad(img)


class RandomJPEG:
    def __init__(self, min_quality: int = 70, max_quality: int = 100):
        assert 0 < min_quality <= max_quality <= 100
        self.min_q = min_quality
        self.max_q = max_quality

    def __call__(self, img: Image.Image) -> Image.Image:
        q = random.randint(self.min_q, self.max_q)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class FixedJPEG:
    def __init__(self, min_quality: int = 90, max_quality: int = 100):
        assert 0 < min_quality <= max_quality <= 100
        self.min_q = min_quality
        self.max_q = max_quality

    def __call__(self, img: Image.Image) -> Image.Image:
        q = random.randint(self.min_q, self.max_q)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class OptionalGaussianBlur:
    def __init__(self, p: float = 0.1, radius: float = 0.5):
        self.p = p
        self.radius = radius

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return img.filter(ImageFilter.GaussianBlur(self.radius))
        return img


class OptionalGaussianNoise:
    def __init__(self, p: float = 0.1, std: float = 0.02):
        self.p = p
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            noise = torch.randn_like(x) * self.std
            x = x + noise
            x = torch.clamp(x, 0.0, 1.0)
        return x

