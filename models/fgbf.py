"""
models/fgbf.py
==============
Fine-Grained Boundary-Aware Low-Grade Feature Module (FGBF).

Auxiliary experimental module for CoRD-Net designed to enhance KL0/KL1/KL2
discrimination without duplicate backbone passes or E4 compartment coupling.
Reuses the shared ConvNeXt spatial feature map (B, 768, H, W).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FineGrainedBoundaryFeatureModule(nn.Module):
    """
    Lightweight Boundary-Aware Low-Grade Feature Module (FGBF).

    Maps backbone spatial feature maps (B, 768, H, W) to:
      1. boundary_feature (B, 256): Fused global + local boundary representation.
      2. low_grade_logits (B, 3): Refinement classification logits for [KL0, KL1, KL2].

    Parameters
    ----------
    in_channels:
        Backbone spatial feature map channels (default: 768).
    reduced_dim:
        Reduced intermediate feature dimension (default: 256).
    dropout:
        Dropout probability in projection and classifier heads (default: 0.1).
    eps:
        Epsilon for numerical stability during weighted spatial pooling normalization (default: 1e-6).
    """

    def __init__(
        self,
        in_channels: int = 768,
        reduced_dim: int = 256,
        feature_dim: int | None = None,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if feature_dim is not None:
            reduced_dim = feature_dim
        self.in_channels = in_channels
        self.reduced_dim = reduced_dim
        self.eps = eps

        # 1. Spatial channel reduction
        self.channel_reduce = nn.Sequential(
            nn.Conv2d(in_channels, reduced_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduced_dim),
            nn.GELU(),
        )

        # 2. Lightweight attention conv network
        # Input has (reduced_dim + 1) channels due to appended horizontal grid
        attn_mid_channels = 16
        self.attn_net = nn.Sequential(
            nn.Conv2d(reduced_dim + 1, attn_mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(attn_mid_channels),
            nn.GELU(),
            nn.Conv2d(attn_mid_channels, 2, kernel_size=1),
        )

        # Learnable spatial prior scale
        self.prior_scale = nn.Parameter(torch.tensor(1.0))

        # 3. Fusion projection layer: concat(global(256), medial(256), lateral(256)) = 768 -> 256
        self.projection = nn.Sequential(
            nn.Linear(reduced_dim * 3, reduced_dim),
            nn.LayerNorm(reduced_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )

        # 4. KL0/KL1/KL2 refinement classifier head
        self.low_grade_head = nn.Sequential(
            nn.Linear(reduced_dim, 128),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, 3),
        )

        # Cached attention maps for optional debugging/visualization
        self.last_medial_attention: torch.Tensor | None = None
        self.last_lateral_attention: torch.Tensor | None = None

    def _create_horizontal_grid(self, batch_size: int, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Creates a normalized horizontal coordinate map in [-1.0, 1.0] of shape (B, 1, H, W).
        """
        x_coords = torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=dtype)
        x_grid = x_coords.view(1, 1, 1, width).expand(batch_size, 1, height, width)
        return x_grid

    def get_last_attention_maps(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Exposes cached medial and lateral attention maps from the most recent forward pass.

        Returns
        -------
        (medial_attention, lateral_attention): tuple of (B, 1, H, W) tensors or None.
        """
        return self.last_medial_attention, self.last_lateral_attention

    def forward(self, feature_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for FineGrainedBoundaryFeatureModule.

        Parameters
        ----------
        feature_map:
            Tensor of shape (B, 768, H, W).

        Returns
        -------
        boundary_feature:
            Tensor of shape (B, 256).
        low_grade_logits:
            Tensor of shape (B, 3) representing logits for [KL0, KL1, KL2].
        """
        # Defensive input shape checks
        if feature_map.ndim != 4:
            raise ValueError(
                f"Expected 4D input feature map (B, C, H, W), got shape {tuple(feature_map.shape)}"
            )
        B, C, H, W = feature_map.shape
        if C != self.in_channels:
            raise ValueError(
                f"Expected feature map with {self.in_channels} channels, got {C} channels"
            )
        if H <= 0 or W <= 0:
            raise ValueError(f"Invalid spatial dimensions H={H}, W={W}")

        # Step 1: Channel reduction
        reduced_features = self.channel_reduce(feature_map)  # (B, 256, H, W)

        # Step 2: Create normalized horizontal coordinate information
        x_grid = self._create_horizontal_grid(B, H, W, device=feature_map.device, dtype=feature_map.dtype)  # (B, 1, H, W)

        # Predict raw attention logits from reduced features + horizontal coordinate channel
        attn_input = torch.cat([reduced_features, x_grid], dim=1)  # (B, 257, H, W)
        attn_logits = self.attn_net(attn_input)  # (B, 2, H, W)

        # Apply spatial priors: left-side bias (-x_grid) for medial, right-side bias (x_grid) for lateral
        medial_prior = -x_grid  # High on left (x < 0)
        lateral_prior = x_grid  # High on right (x > 0)
        spatial_priors = torch.cat([medial_prior, lateral_prior], dim=1)  # (B, 2, H, W)

        biased_logits = attn_logits + self.prior_scale * spatial_priors

        # Deriving soft, differentiable attention maps in [0, 1]
        medial_attention = torch.sigmoid(biased_logits[:, 0:1, :, :])   # (B, 1, H, W)
        lateral_attention = torch.sigmoid(biased_logits[:, 1:2, :, :])  # (B, 1, H, W)

        # Cache for optional debugging (detached to prevent memory leak)
        self.last_medial_attention = medial_attention.detach()
        self.last_lateral_attention = lateral_attention.detach()

        # Step 3: Weighted spatial pooling normalized by attention sum + eps
        medial_sum = medial_attention.sum(dim=(-2, -1), keepdim=True) + self.eps   # (B, 1, 1, 1)
        lateral_sum = lateral_attention.sum(dim=(-2, -1), keepdim=True) + self.eps # (B, 1, 1, 1)

        medial_feature = (reduced_features * medial_attention).sum(dim=(-2, -1)) / medial_sum.squeeze(-1).squeeze(-1)   # (B, 256)
        lateral_feature = (reduced_features * lateral_attention).sum(dim=(-2, -1)) / lateral_sum.squeeze(-1).squeeze(-1) # (B, 256)

        # Step 4: Fusion of global average pool + medial + lateral features
        global_feature = F.adaptive_avg_pool2d(reduced_features, (1, 1)).flatten(1)  # (B, 256)
        local_feature = torch.cat([medial_feature, lateral_feature], dim=1)           # (B, 512)
        fused = torch.cat([global_feature, local_feature], dim=1)                     # (B, 768)

        boundary_feature = self.projection(fused)  # (B, 256)

        # Step 5: Refinement head for [KL0, KL1, KL2]
        low_grade_logits = self.low_grade_head(boundary_feature)  # (B, 3)

        # Defensive output shape checks
        if boundary_feature.shape != (B, self.reduced_dim):
            raise RuntimeError(
                f"Expected boundary_feature shape ({B}, {self.reduced_dim}), got {tuple(boundary_feature.shape)}"
            )
        if low_grade_logits.shape != (B, 3):
            raise RuntimeError(
                f"Expected low_grade_logits shape ({B}, 3), got {tuple(low_grade_logits.shape)}"
            )

        return boundary_feature, low_grade_logits
