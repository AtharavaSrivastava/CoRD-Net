"""
augmentation.py
===============
Data augmentation pipelines for CoRD-Net.

IMPROVEMENTS
------------
1. Added RandomHorizontalFlip (p=0.5)
   Knee X-rays from OAI include both left and right knees.  The medial
   compartment is always on the inner side, so a horizontal flip of a
   left knee produces a plausible right-knee appearance.  This is valid
   domain augmentation that doubles effective training data for free.

2. Added RandomResizedCrop (scale 0.85–1.0)
   Simulates the variation in patient positioning and X-ray zoom level
   that occurs across OAI acquisition sites.  Much more realistic than
   the ±10% scale jitter inside RandomAffine alone.

3. Increased rotation to ±15° (was ±10°)
   OAI knee X-rays routinely exhibit 10–15° rotation from patient
   positioning.  The tighter ±10° under-represented this variation.

4. Increased ColorJitter to brightness=0.25, contrast=0.25 (was 0.15)
   Exposure variation across OAI scanners spans a wider range than 0.15.
   Higher jitter forces the backbone to learn bone/cartilage structure
   rather than scanner-specific intensity levels.

5. Increased Gaussian noise sigma_max to 0.05 (was 0.03)
   X-ray sensor noise is higher than photographic noise.  Slightly more
   aggressive noise regularises the backbone against image artefacts.

6. CLAHE clip_limit raised to 3.0 (was 2.0)
   More aggressive CLAHE improves joint-space visibility in darker scans.

7. Kept all fixes from previous session (torch.rand worker seeding).
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

    def __init__(self, clip_limit: float = 3.0,
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
    """Add uniform-sigma Gaussian noise to a float tensor.

    Uses torch.rand() (worker-safe) instead of np.random.uniform().
    """

    def __init__(self, sigma_min: float = 0.01, sigma_max: float = 0.05) -> None:
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        sigma = self.sigma_min + torch.rand(1).item() * (self.sigma_max - self.sigma_min)
        return torch.clamp(tensor + torch.randn_like(tensor) * sigma, 0.0, 1.0)


def get_train_transforms(cfg: ModelConfig) -> transforms.Compose:
    """Return the training augmentation pipeline."""
    return transforms.Compose([
        CLAHE(clip_limit=3.0),
        # IMPROVEMENT 1: horizontal flip — valid for bilateral knee dataset
        transforms.RandomHorizontalFlip(p=0.5),
        # IMPROVEMENT 3: wider rotation range
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05),
                                scale=(0.9, 1.1)),
        # IMPROVEMENT 2: crop-based zoom variation
        transforms.RandomResizedCrop(
            size=(cfg.image_size, cfg.image_size),
            scale=(0.85, 1.0),
            ratio=(0.95, 1.05),   # near-square — knee X-rays are portrait
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        GrayToRGB(),
        # IMPROVEMENT 4: stronger colour jitter
        transforms.ColorJitter(brightness=0.25, contrast=0.25),
        transforms.ToTensor(),
        # IMPROVEMENT 5: stronger noise
        AddGaussianNoise(sigma_min=0.01, sigma_max=0.05),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms(cfg: ModelConfig) -> transforms.Compose:
    """Return the validation / test transform pipeline (no augmentation)."""
    return transforms.Compose([
        CLAHE(clip_limit=3.0),
        GrayToRGB(),
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
