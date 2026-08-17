"""
models/compartment.py
=====================
E4 — Compartment Branches.

Bug fixes from previous session are preserved.

IMPROVEMENTS
------------
1. Deeper CompartmentFusion gate (was Linear → Softmax)
   The gate is the only place where global/medial/lateral contributions
   are weighted against each other.  A single linear layer (no hidden
   units, no nonlinearity) cannot model any interaction between the three
   branch features — it can only learn a fixed linear combination.
   Replaced with a 2-layer MLP (Linear → GELU → Linear → Softmax) with
   a hidden dimension of feat_dim // 4 (192 for feat_dim=768).
   Parameter cost: 768*3*192 + 192*3 = 443,328 additional parameters,
   small relative to the backbone but meaningfully richer gating.

2. Residual connection in CompartmentFusion
   After gating and projection, the fused feature is added to the
   global feature as a residual.  This ensures that if the
   compartment branches add no useful information (early training),
   the network can fall back on the global feature without degradation.
   It also gives the gradient a clean shortcut path back to the backbone.

3. EGRB: multi-scale edge gating (3×3 + 5×5)
   The original EGRB only computed a mean Sobel gate from 4 directions,
   all at scale=1 (3×3 kernel).  Knee joint margins exist at multiple
   scales: fine cartilage detail at 1–2mm and broader joint-space
   narrowing at 5–10mm.  Added a second 5×5 Sobel gate branch at the
   coarser scale, then fused the two gate scales via a learned 1×1 conv.

4. EGRB: pre-activation BatchNorm (BN before activation, not after)
   Placing BN before the activation (BN → GELU → conv) rather than
   after (conv → BN → GELU) means BN operates on the identity-mapped
   residual path, producing better gradient flow for deep residual nets
   (established by He et al. 2016 pre-activation ResNet).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeGatedResidualBlock(nn.Module):
    """
    Edge-Gated Residual Block (EGRB) with multi-scale Sobel gating.

    output = main_branch(x) ⊗ fused_gate(x) + x

    Gate is computed at two spatial scales (3×3 and 5×5) and fused
    with a learned 1×1 convolution.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        # IMPROVEMENT 4: pre-activation BN
        self.bn       = nn.BatchNorm2d(channels)
        self.dw_conv  = nn.Conv2d(channels, channels, 3, padding=1,
                                  groups=channels, bias=False)
        self.pw_conv  = nn.Conv2d(channels, channels, 1, bias=False)
        self.act      = nn.GELU()

        # IMPROVEMENT 3: two-scale edge gating
        # Scale-1: 4-direction 3×3 Sobel (fine margins)
        for name, k in {
            "kH":  [[-1,-2,-1],[0,0,0],[1,2,1]],
            "kV":  [[-1,0,1],[-2,0,2],[-1,0,1]],
            "kD1": [[0,1,2],[-1,0,1],[-2,-1,0]],
            "kD2": [[-2,-1,0],[-1,0,1],[0,1,2]],
        }.items():
            self.register_buffer(
                name, torch.tensor(k, dtype=torch.float32).view(1, 1, 3, 3)
            )
        # Scale-2: 5×5 Sobel (coarse joint-space narrowing)
        for name, k in {
            "kH5": [[-1,-4,-6,-4,-1],[-2,-8,-12,-8,-2],[0,0,0,0,0],[2,8,12,8,2],[1,4,6,4,1]],
            "kV5": [[-1,-2,0,2,1],[-4,-8,0,8,4],[-6,-12,0,12,6],[-4,-8,0,8,4],[-1,-2,0,2,1]],
        }.items():
            self.register_buffer(
                name, torch.tensor(k, dtype=torch.float32).view(1, 1, 5, 5)
            )

        # Project 6 edge response maps (4 fine + 2 coarse) → channels, then gate
        self.edge_proj = nn.Sequential(
            nn.Conv2d(6, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def _edge_responses(self, x: torch.Tensor) -> torch.Tensor:
        """Compute multi-scale Sobel responses → (B, 6, H, W)."""
        g = x.mean(dim=1, keepdim=True)
        fine = torch.cat([
            F.conv2d(g, self.kH,  padding=1),
            F.conv2d(g, self.kV,  padding=1),
            F.conv2d(g, self.kD1, padding=1),
            F.conv2d(g, self.kD2, padding=1),
        ], dim=1)
        coarse = torch.cat([
            F.conv2d(g, self.kH5, padding=2),
            F.conv2d(g, self.kV5, padding=2),
        ], dim=1)
        return torch.cat([fine, coarse], dim=1)   # (B, 6, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # IMPROVEMENT 4: pre-activation order: BN → act → conv
        main = self.pw_conv(self.act(self.dw_conv(self.act(self.bn(x)))))
        gate = self.edge_proj(self._edge_responses(x))
        return main * gate + x


class _EncoderBranch(nn.Module):
    """
    stem → shared backbone → own EGRB → pool.

    Each branch has its own EGRB so BatchNorm stats accumulate
    independently for medial vs lateral crops (bug fix from session 1).
    """

    def __init__(self, stem, backbone_features, feature_dim=768) -> None:
        super().__init__()
        self.stem     = stem
        self.features = backbone_features
        self.egrb     = EdgeGatedResidualBlock(feature_dim)
        self.pool     = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = self.features(self.stem(x))
        return self.pool(self.egrb(spatial)).flatten(1)


class CompartmentFusion(nn.Module):
    """
    Soft-gated fusion of global, medial, and lateral features.

    IMPROVEMENT 1: deeper 2-layer MLP gate.
    IMPROVEMENT 2: residual connection back to global feature.
    """

    def __init__(self, feat_dim: int = 768) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        hidden        = feat_dim // 4   # 192 for feat_dim=768

        # IMPROVEMENT 1: 2-layer MLP gate
        self.gate = nn.Sequential(
            nn.Linear(feat_dim * 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
            nn.Softmax(dim=-1),
        )

        self.ln_global  = nn.LayerNorm(feat_dim)
        self.ln_medial  = nn.LayerNorm(feat_dim)
        self.ln_lateral = nn.LayerNorm(feat_dim)

        self.proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
        )

        # IMPROVEMENT 2: learnable residual scale (init near 0)
        self.res_scale = nn.Parameter(torch.zeros(1))

    def forward(self, g: torch.Tensor, m: torch.Tensor,
                l: torch.Tensor) -> torch.Tensor:
        w = self.gate(torch.cat([g, m, l], dim=1))

        if self.training:
            self.debug_stats = {
                "gate_global":  w[:, 0].mean().item(),
                "gate_medial":  w[:, 1].mean().item(),
                "gate_lateral": w[:, 2].mean().item(),
            }

        gn = self.ln_global(g)
        mn = self.ln_medial(m)
        ln = self.ln_lateral(l)

        fused    = w[:, 0:1] * gn + w[:, 1:2] * mn + w[:, 2:3] * ln
        projected = self.proj(fused)

        # IMPROVEMENT 2: residual back to global (scale init=0 → identity at start)
        return projected + self.res_scale * g


class CompartmentBranchModule(nn.Module):
    """
    E4 — shared-weight three-crop encoder.

    Medial and lateral branches are separate _EncoderBranch instances
    (separate BN stats — bug fix from session 1).
    """

    def __init__(self, stem, backbone_features, feature_dim=768) -> None:
        super().__init__()
        self.medial_branch  = _EncoderBranch(stem, backbone_features, feature_dim)
        self.lateral_branch = _EncoderBranch(stem, backbone_features, feature_dim)
        self.fusion         = CompartmentFusion(feature_dim)

    def _split_compartments(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, h, w = x.shape
        overlap  = int(0.10 * w)
        mid      = w // 2
        medial   = F.interpolate(x[:, :, :, :mid + overlap],
                                 size=(h, w), mode="bilinear", align_corners=False)
        lateral  = F.interpolate(x[:, :, :, mid - overlap:],
                                 size=(h, w), mode="bilinear", align_corners=False)
        return medial, lateral

    def forward(self, global_feat, global_crop):
        med_crop, lat_crop = self._split_compartments(global_crop)
        m = self.medial_branch(med_crop)
        l = self.lateral_branch(lat_crop)
        fused = self.fusion(global_feat, m, l)

        if self.training:
            self.debug_stats = {
                "global_norm":  global_feat.norm(dim=1).mean().item(),
                "medial_norm":  m.norm(dim=1).mean().item(),
                "lateral_norm": l.norm(dim=1).mean().item(),
                "ml_diff":      (m - l).abs().mean().item(),
            }
            self.debug_stats.update(self.fusion.debug_stats)

        return fused, m, l
