"""
models/compartment.py
=====================
E4 — Compartment Branches.

Three crops (global knee, medial, lateral) share a single ConvNeXt-tiny
backbone — medial and lateral branches additionally share weights with
each other.  An Edge-Gated Residual Block (EGRB) preserves fine joint
margin activations before pooling.  A gated CompartmentFusion produces
one 768-d fused descriptor while also exposing the per-branch pooled
vectors for downstream RTC (E7).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dual_intensity import DualIntensityStem


class EdgeGatedResidualBlock(nn.Module):
    """
    Edge-Gated Residual Block (EGRB).

    A directional Sobel gate (4-direction, grayscale-averaged) is computed
    from the input feature map and applied multiplicatively to the main
    depthwise convolution branch, preserving fine structural activations
    at joint margins.

    ``output = main_branch(x) ⊗ sobel_gate(x) + x``

    Parameters
    ----------
    channels:
        Number of input (= output) channels.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw_conv   = nn.Conv2d(channels, channels, 3, padding=1,
                                   groups=channels, bias=False)
        self.pw_conv   = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn        = nn.BatchNorm2d(channels)
        self.act       = nn.GELU()
        self.edge_proj = nn.Sequential(
            nn.Conv2d(4, channels, 1, bias=False), nn.Sigmoid()
        )
        _sobel = {
            "kH":  [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            "kV":  [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            "kD1": [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
            "kD2": [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]],
        }
        for name, k in _sobel.items():
            self.register_buffer(
                name, torch.tensor(k, dtype=torch.float32).view(1, 1, 3, 3)
            )

    def _sobel_gate(self, x: torch.Tensor) -> torch.Tensor:
        """Mean Sobel response across channels → (B, 4, H, W)."""
        g = x.mean(dim=1, keepdim=True)
        return torch.cat([
            F.conv2d(g, self.kH,  padding=1),
            F.conv2d(g, self.kV,  padding=1),
            F.conv2d(g, self.kD1, padding=1),
            F.conv2d(g, self.kD2, padding=1),
        ], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, C, H, W)."""
        main = self.act(self.bn(self.pw_conv(self.dw_conv(x))))
        gate = self.edge_proj(self._sobel_gate(x))
        return main * gate + x


class _EncoderBranch(nn.Module):
    """
    Single-crop encoder: stem → shared backbone features → EGRB → pool.

    Both the stem and backbone_features references are injected, not
    owned, so the caller controls weight sharing between branches.

    Parameters
    ----------
    stem:
        DualIntensityStem (or nn.Identity if E3 is disabled).
    backbone_features:
        Shared ConvNeXt-tiny feature extractor (nn.Sequential).
    feature_dim:
        Backbone output channels (768 for convnext_tiny).
    """

    def __init__(
        self,
        stem: nn.Module,
        backbone_features: nn.Module,
        feature_dim: int = 768,
    ) -> None:
        super().__init__()
        self.stem     = stem
        self.features = backbone_features
        self.egrb     = EdgeGatedResidualBlock(feature_dim)
        self.pool     = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) → (B, feature_dim)."""
        spatial = self.features(self.stem(x))   # (B, C, H, W)
        return self.pool(self.egrb(spatial)).flatten(1)


class CompartmentFusion(nn.Module):
    """
    Soft-gated fusion of global, medial, and lateral branch features.

    A 3-way softmax gate learns how much each branch contributes to the
    fused representation, preventing any single branch from dominating.

    ``fused = w_g · global + w_m · medial + w_l · lateral``

    Parameters
    ----------
    feat_dim:
        Per-branch feature dimension (768).
    """

    def __init__(self, feat_dim: int = 768) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feat_dim * 3, 3), nn.Softmax(dim=-1)
        )
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.LayerNorm(feat_dim), nn.GELU()
        )

    def forward(
        self,
        global_feat: torch.Tensor,
        medial_feat: torch.Tensor,
        lateral_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Three (B, feat_dim) tensors → one (B, feat_dim) tensor."""
        concat = torch.cat([global_feat, medial_feat, lateral_feat], dim=1)
        w = self.gate(concat)
        fused = w[:, 0:1] * global_feat + w[:, 1:2] * medial_feat + w[:, 2:3] * lateral_feat
        return self.proj(fused)


class CompartmentBranchModule(nn.Module):
    """
    E4 top-level module — shared-weight three-crop encoder.

    The global, medial, and lateral crops all pass through the **same**
    DualIntensityStem and ConvNeXt-tiny backbone_features weights.
    Medial and lateral branches additionally share their EGRB weights.

    Parameters
    ----------
    stem:
        Injected DualIntensityStem (or nn.Identity).
    backbone_features:
        Injected shared ConvNeXt-tiny feature extractor.
    feature_dim:
        Backbone output channel count.

    Returns (from forward)
    ----------------------
    fused_feat:  (B, feature_dim) — gated fusion of all three branches
    global_feat: (B, feature_dim) — global branch (→ E7 RTC, E5 DRP)
    medial_feat: (B, feature_dim) — medial branch (→ E7 RTC)
    lateral_feat:(B, feature_dim) — lateral branch (→ E7 RTC)
    """

    def __init__(
        self,
        stem: nn.Module,
        backbone_features: nn.Module,
        feature_dim: int = 768,
    ) -> None:
        super().__init__()
        self.global_branch      = _EncoderBranch(stem, backbone_features, feature_dim)
        self.compartment_branch = _EncoderBranch(stem, backbone_features, feature_dim)
        self.fusion             = CompartmentFusion(feature_dim)

    def forward(
        self,
        global_crop: torch.Tensor,
        medial_crop: torch.Tensor,
        lateral_crop: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Three (B, 3, H, W) crops → (fused, global, medial, lateral)."""
        g = self.global_branch(global_crop)
        m = self.compartment_branch(medial_crop)
        l = self.compartment_branch(lateral_crop)
        return self.fusion(g, m, l), g, m, l
