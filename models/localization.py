"""
models/localization.py
======================
E2 — Spatial Transformer Network (STN) for automatic knee localization.

Reference: Jaderberg et al. (2015) "Spatial Transformer Networks"

The KneeLocalizer predicts a 2×3 affine matrix θ that crops and aligns
the knee joint from a full-field radiograph.  It is initialised as the
identity transform so training starts from a no-op warp.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class KneeLocalizer(nn.Module):
    """
    STN-based knee joint auto-localization.

    Architecture
    ------------
    4-layer CNN (localization network) → FC → 6 affine parameters θ →
    affine_grid → grid_sample

    Parameters
    ----------
    img_size:
        Expected spatial resolution of the input radiograph.
    """

    def __init__(self, img_size: int = 512) -> None:
        super().__init__()
        self.img_size = img_size

        self.loc_net = nn.Sequential(
            # Layer 1: 1→32, 7×7 stride-2 → pool → (H/4, W/4)
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            # Layer 2: 32→64, 5×5 stride-2 → pool
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            # Layer 3: 64→128, 3×3 → pool
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            # Layer 4: 128→256, 3×3 → GAP to 4×4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 6),
        )

        # Identity initialisation: no-op warp at the start of training
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x:
            Grayscale radiograph tensor of shape (B, 1, H, W).

        Returns
        -------
        localized:
            Spatially transformed image, shape (B, 1, H, W).
        theta:
            Predicted affine parameters, shape (B, 2, 3).
        """
        xs = self.loc_net(x).flatten(1)
        theta = self.fc_loc(xs).view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=False)

        localized = F.grid_sample(
            x,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False
        )
        return localized, theta
