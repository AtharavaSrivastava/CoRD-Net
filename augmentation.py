"""
augmentation.py
===============
Data augmentation pipelines for CoRD-Net (mirrors data-aug.py).

Provides CLAHE preprocessing, grayscale-to-RGB conversion, Gaussian
noise injection, and composed train/val transform pipelines consistent
with config.yaml.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from config import ModelConfig


class CLAHE:
    """Contrast-Limited Adaptive Histogram Equalisation (OpenCV)."""

    def __init__(self, clip_limit: float = 2.0,
                 tile_grid_size: tuple[int, int] = (8, 8)) -> None:
        self.clip_limit     = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img: Image.Image) -> Image.Image:
        arr   = np.array(img)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit,
                                 tileGridSize=self.tile_grid_size)
        return Image.fromarray(clahe.apply(arr))


class GrayToRGB:
    """Stack a grayscale image into 3 identical channels."""

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img)
        return Image.fromarray(np.stack([arr, arr, arr], axis=-1))


class AddGaussianNoise:
    """Add uniform-sigma Gaussian noise to a float tensor."""

    def __init__(self, sigma_min: float = 0.01, sigma_max: float = 0.03) -> None:
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        return torch.clamp(tensor + torch.randn_like(tensor) * sigma, 0.0, 1.0)


def get_mild_train_transforms(cfg: ModelConfig) -> transforms.Compose:
    """Return mild training augmentation pipeline."""
    return transforms.Compose([
        CLAHE(clip_limit=2.0),
        transforms.RandomRotation(degrees=5),
        transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.95, 1.05)),
        GrayToRGB(),
        transforms.ColorJitter(brightness=0.08, contrast=0.08),
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        AddGaussianNoise(sigma_min=0.005, sigma_max=0.015),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_none_train_transforms(cfg: ModelConfig) -> transforms.Compose:
    """Return clean / no-augmentation pipeline for training."""
    return transforms.Compose([
        CLAHE(clip_limit=2.0),
        GrayToRGB(),
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_train_transforms(cfg: ModelConfig, mode: str = "standard") -> transforms.Compose:
    """Return the training augmentation pipeline from cfg based on mode."""
    if mode == "mild":
        return get_mild_train_transforms(cfg)
    if mode == "none":
        return get_none_train_transforms(cfg)
    return transforms.Compose([
        CLAHE(clip_limit=2.0),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05),
                                scale=(0.9, 1.1)),
        GrayToRGB(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        AddGaussianNoise(sigma_min=0.01, sigma_max=0.03),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms(cfg: ModelConfig) -> transforms.Compose:
    """Return the validation / test transform pipeline from cfg."""
    return transforms.Compose([
        CLAHE(clip_limit=2.0),
        GrayToRGB(),
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
