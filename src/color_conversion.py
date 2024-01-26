import torch
import numpy as np
from src.constants import RGB_FROM_HED, HED_FROM_RGB
from torchvision.transforms.transforms import (
    ColorJitter,
    RandomApply,
    GaussianBlur,
    Normalize,
)


def torch_rgb2hed(img, hed_t, e):
    img = img.movedim(-3, -1)

    img = torch.clamp(img, min=e)
    img = torch.log(img) / torch.log(e)
    img = torch.matmul(img, hed_t)
    return img.movedim(-1, -3)


def torch_hed2rgb(img, rgb_t, e):
    e = -torch.log(e)
    img = img.movedim(-3, -1)
    img = torch.matmul(-(img * e), rgb_t)
    img = torch.exp(img)
    img = torch.clamp(img, 0, 1)
    return img.movedim(-1, -3)


class Hed2Rgb(torch.nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.e = torch.tensor(1e-6).to(rank)
        self.rgb_t = torch.from_numpy(RGB_FROM_HED).to(rank)
        self.rank = rank

    def forward(self, img):
        return torch_hed2rgb(img, self.rgb_t, self.e)


class Rgb2Hed(torch.nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.e = torch.tensor(1e-6).to(rank)
        self.hed_t = torch.from_numpy(HED_FROM_RGB).to(rank)
        self.rank = rank

    def forward(self, img):
        return torch_rgb2hed(img, self.hed_t, self.e)


class HED_normalize_torch(torch.nn.Module):
    def __init__(self, sigma, bias, rank=0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sigma = sigma
        self.bias = bias
        self.rank = rank
        self.rgb2hed = Rgb2Hed(rank=rank)
        self.hed2rgb = Hed2Rgb(rank=rank)

    def rng(self, val, batch_size):
        return torch.empty(batch_size, 3).uniform_(-val, val).to(self.rank)

    def color_norm_hed(self, img):
        B = img.shape[0]
        sigmas = self.rng(self.sigma, B)
        biases = self.rng(self.bias, B)
        return (img * (1 + sigmas.view(*sigmas.shape, 1, 1))) + biases.view(
            *biases.shape, 1, 1
        )

    def forward(self, img):
        if img.dim() == 3:
            img = img.view(1, *img.shape)
        hed = self.rgb2hed(img)
        hed = self.color_norm_hed(hed)
        return self.hed2rgb(hed)


class GaussianNoise(torch.nn.Module):
    def __init__(self, sigma, rank):
        super().__init__()
        self.sigma = sigma
        self.rank = rank

    def forward(self, img):
        noise = torch.empty(img.shape).uniform_(-self.sigma, self.sigma).to(self.rank)
        return img + noise


def color_augmentations(train, sigma=0.05, bias=0.03, s=0.2, rank=0):
    if train:
        color_jitter = ColorJitter(
            # 0.8 * s, 0.0 * s, 0.8 * s, 0.2 * s
            0.8 * s,
            0.8 * s,
            0.5 * s,
            0.2 * s,
        )  # brightness, contrast, saturation, hue

        data_transforms = torch.nn.Sequential(
            RandomApply([HED_normalize_torch(sigma, bias, rank=rank)], p=0.75),
            RandomApply([color_jitter], p=0.3),
            RandomApply([GaussianNoise(0.05, rank)], p=0.3),
            RandomApply([GaussianBlur(kernel_size=15, sigma=(0.1, 2.0))], p=0.2),
        )
    else:
        data_transforms = torch.nn.Sequential(
            HED_normalize_torch(sigma, bias, rank=rank)
        )
    return data_transforms


def get_normalize(use_norm=True):
    if use_norm:
        return Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        return lambda x: x
