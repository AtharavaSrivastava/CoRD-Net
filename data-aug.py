import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# ==========================================
# CLAHE
# ==========================================
class CLAHE:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        img = np.array(img)

        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size
        )

        img = clahe.apply(img)

        return Image.fromarray(img)


# ==========================================
# Convert Grayscale → RGB
# ==========================================
class GrayToRGB:
    def __call__(self, img):
        img = np.array(img)

        img = np.stack([img, img, img], axis=-1)

        return Image.fromarray(img)


# ==========================================
# Gaussian Noise
# ==========================================
class AddGaussianNoise:
    def __init__(self, sigma_min=0.01, sigma_max=0.03):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, tensor):
        sigma = np.random.uniform(
            self.sigma_min,
            self.sigma_max
        )

        noise = torch.randn_like(tensor) * sigma

        return torch.clamp(
            tensor + noise,
            0.0,
            1.0
        )


# ==========================================
# TRAIN TRANSFORMS
# ==========================================
train_transform = transforms.Compose([

    # CLAHE
    CLAHE(clip_limit=2.0),

    # Rotation ±10°
    transforms.RandomRotation(
        degrees=10
    ),

    # Translation ±5%
    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        scale=(0.9, 1.1)
    ),

    # Brightness ±15%
    # Contrast ±15%
    GrayToRGB(),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15
    ),

    transforms.ToTensor(),

    # Gaussian Noise
    AddGaussianNoise(
        sigma_min=0.01,
        sigma_max=0.03
    ),

    # ImageNet normalization
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# VALIDATION / TEST
# ==========================================
val_transform = transforms.Compose([

    CLAHE(clip_limit=2.0),

    GrayToRGB(),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
