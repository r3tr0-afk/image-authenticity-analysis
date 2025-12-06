from __future__ import annotations

from .transforms import EnsureMinSize, RandomJPEG, FixedJPEG, OptionalGaussianBlur, OptionalGaussianNoise

import random
from typing import List

import torch
from PIL import Image, ImageFilter
from torchvision import transforms as T
import torchvision.transforms.functional as F



def preprocess_train(img: Image.Image, crop_size: int = 256) -> torch.Tensor:
    # checking size : min side >= crop_size and padding if needed
    img = EnsureMinSize(min_size=crop_size, padding_mode="reflect")(img)

    # random crop
    img = T.RandomCrop(crop_size)(img)

    # random horizontal flip
    if random.random() < 0.5:
        img = F.hflip(img)

    # random jpeg recompression
    img = RandomJPEG(70, 100)(img)

    # random gaussian blur
    img = OptionalGaussianBlur(p=0.1, radius=0.5)(img)

    # to tensor
    x = F.to_tensor(img)  # [0,1]

    # gaussian noise on tensor
    x = OptionalGaussianNoise(p=0.1, std=0.02)(x)

    return x


def _five_fixed_crops(img: Image.Image, crop_size: int = 256) -> List[Image.Image]:
    w, h = img.size
    if w < crop_size or h < crop_size:
        raise ValueError(f"Image too small for crops even after padding: {img.size}")

    # Top-left
    tl = img.crop((0, 0, crop_size, crop_size))
    # Top-right
    tr = img.crop((w - crop_size, 0, w, crop_size))
    # Bottom-left
    bl = img.crop((0, h - crop_size, crop_size, h))
    # Bottom-right
    br = img.crop((w - crop_size, h - crop_size, w, h))
    # Center
    cx0 = (w - crop_size) // 2
    cy0 = (h - crop_size) // 2
    center = img.crop((cx0, cy0, cx0 + crop_size, cy0 + crop_size))

    return [tl, tr, bl, br, center]


def preprocess_eval_five_crops(
    img: Image.Image,
    crop_size: int = 256,
    jpeg_quality: int = 100,
) -> List[torch.Tensor]:
    # min side >= crop_size
    img = EnsureMinSize(min_size=crop_size, padding_mode="reflect")(img)

    # 5 fixed crops
    crops = _five_fixed_crops(img, crop_size=crop_size)

    # jpeg recompression per crop, then to tensor
    out_tensors: List[torch.Tensor] = []
    jpeg = FixedJPEG(min_quality=jpeg_quality, max_quality=jpeg_quality)
    for c in crops:
        c_jpeg = jpeg(c)
        x = F.to_tensor(c_jpeg)
        out_tensors.append(x)

    return out_tensors

