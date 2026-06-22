"""
models/roi.py
=============
E5 — Soft ROI Mask / Dense ROI Pooling (DRP Block).

ROIAttentionMask generates a per-location soft weight in [0,1] from the
backbone spatial feature map.  FeatureReweighting blends masked and
original features via a learnable alpha.  DRPBlock composes both and
projects the pooled result to the shared embedding dimension.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ROIAttentionMask(nn.Module):
    """
    Soft spatial ROI mask generator.

    Maps (B, C, H, W) spatial features to a (B, 1, H, W) weight map
    whose values lie in [0, 1], indicating ROI membership at each spatial
    location.

    Architecture: 1×1 Conv → BN → ReLU → 1×1 Conv → Sigmoid

    Parameters
    ----------
    in_channels:
        Input feature map channel count.
    reduction:
        Channel reduction factor for the intermediate projection.
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
        """(B, C, H, W) → (B, 1, H, W) soft ROI mask in [0, 1]."""
        return self.mask_net(x)


class FeatureReweighting(nn.Module):
    """
    Learnable blend between masked and original feature maps.

    ``output = alpha · (mask * features) + (1 - alpha) · features``

    alpha is initialised to 0.5 (sigmoid(0)) and jointly optimised,
    letting the network learn how aggressively to suppress off-ROI
    activations without destroying gradient flow.
    """

    def __init__(self) -> None:
        super().__init__()
        self._alpha_raw = nn.Parameter(torch.zeros(1))  # sigmoid(0) = 0.5

    @property
    def alpha(self) -> torch.Tensor:
        """Constrained blend coefficient in (0, 1)."""
        return torch.sigmoid(self._alpha_raw)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        features: (B, C, H, W)
        mask:     (B, 1, H, W)

        Returns
        -------
        reweighted: (B, C, H, W)
        """
        masked = mask * features
        return self.alpha * masked + (1.0 - self.alpha) * features


class DRPBlock(nn.Module):
    """
    Dense ROI Pooling Block — E5 top-level module.

    Pipeline::

        (B, C, H, W)
            → ROIAttentionMask → (B, 1, H, W) soft mask
            → FeatureReweighting → (B, C, H, W) reweighted
            → AdaptiveAvgPool2d(1) → (B, C)
            → Linear + LayerNorm + GELU → (B, embedding_dim)

    The last ROI mask tensor is cached in ``self.last_mask`` for
    visualisation (e.g. Grad-CAM overlays).

    Parameters
    ----------
    in_channels:
        Backbone spatial feature channel count (768 for convnext_tiny).
    out_dim:
        Output embedding dimension.
    reduction:
        Channel reduction factor inside ROIAttentionMask.
    """

    def __init__(
        self, in_channels: int, out_dim: int = 256, reduction: int = 4
    ) -> None:
        super().__init__()
        self.roi_mask  = ROIAttentionMask(in_channels, reduction)
        self.reweight  = FeatureReweighting()
        self.pool      = nn.AdaptiveAvgPool2d(1)
        self.proj      = nn.Sequential(
            nn.Linear(in_channels, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )
        self.last_mask: torch.Tensor | None = None

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        feature_map: (B, C, H, W) — backbone spatial features

        Returns
        -------
        embedding: (B, out_dim)
        """
        mask = self.roi_mask(feature_map)
        self.last_mask = mask.detach()
        pooled = self.pool(self.reweight(feature_map, mask)).flatten(1)
        return self.proj(pooled)
