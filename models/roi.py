"""
models/roi.py
=============
E5 — Soft ROI Mask / Dense ROI Pooling (DRP Block).

FIX APPLIED
-----------
Removed 'from utils.visualizer import ModelVisualizer'.
utils is a .py file, not a package, so utils.visualizer does not exist.
This import crashed the entire codebase on every startup even for E1–E4,
which don't use DRPBlock at all, because models/__init__.py eagerly
imports DRPNet which imports roi.py unconditionally.

All visualizer calls have been removed from the forward path. They belong
in a dedicated offline analysis script, not in the training hot-path.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ROIAttentionMask(nn.Module):
    """
    Soft spatial ROI mask generator.

    (B, C, H, W) → (B, 1, H, W) weights in [0, 1].
    Architecture: 1×1 Conv → BN → ReLU → 1×1 Conv → Sigmoid
    """

    def __init__(self, in_channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(in_channels // reduction, 8)
        self.mask_net = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mask_net(x)


class FeatureReweighting(nn.Module):
    """
    Learnable blend between ROI-masked and original feature maps.

    output = alpha · (mask * features) + (1 − alpha) · features
    alpha initialised to 0.5 via sigmoid(0).
    """

    def __init__(self) -> None:
        super().__init__()
        self._alpha_raw = nn.Parameter(torch.zeros(1))

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self._alpha_raw)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked = mask * features
        return self.alpha * masked + (1.0 - self.alpha) * features


class DRPBlock(nn.Module):
    """
    Dense ROI Pooling Block — E5 module.

    (B, C, H, W) → ROI mask → reweight → pool → project → (B, out_dim)

    self.last_mask stores the most recent mask for offline visualisation.
    """

    def __init__(self, in_channels: int, out_dim: int = 256, reduction: int = 4) -> None:
        super().__init__()
        self.roi_mask = ROIAttentionMask(in_channels, reduction)
        self.reweight = FeatureReweighting()
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.proj     = nn.Sequential(
            nn.Linear(in_channels, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )
        self.last_mask: torch.Tensor | None = None

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        mask           = self.roi_mask(feature_map)
        self.last_mask = mask.detach()
        weighted       = self.reweight(feature_map, mask)
        pooled         = self.pool(weighted).flatten(1)
        return self.proj(pooled)
