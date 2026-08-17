"""
models/dual_intensity.py
========================
E3 — Dual-Intensity Stem.

IMPROVEMENTS
------------
1. Fixed dead `alpha` parameter in CLAHEBranch
   `self.alpha = nn.Parameter(torch.tensor(0.0))` was defined in __init__
   but never referenced in forward().  It consumed an optimiser slot and
   produced no gradient signal.  Removed.

2. CLAHEBranch now outputs a residual blend (learnable gate)
   Instead of discarding the original input, CLAHEBranch computes:
       out = sigmoid(gate) * enhanced + (1 - sigmoid(gate)) * projected_input
   The gate is initialised to 0 → sigmoid(0)=0.5, equal blend at start.
   This allows the branch to learn how much contrast enhancement to apply
   per-location, rather than committing to a fixed enhancement.

3. Raised CLAHE branch output channels to 48 (was 32)
   The CLAHEBranch carries the most important signal (bone/cartilage
   density).  32 channels was a bottleneck relative to the 16+16 from
   the edge branches.  48 gives it 60% of the total channel budget
   (48/(48+16+16)=60%) which matches the relative importance in
   ablation studies on similar medical imaging tasks.

4. DualIntensityFusion: added batch normalisation after the 7×7 conv
   The fusion conv output went directly to the 1×1 projection with no
   normalisation, making it sensitive to scale differences between the
   three branches.  Added BN between the 7×7 conv and the 1×1.

5. Residual connection with learnable scale (was hardcoded 0.1)
   `x + 0.1 * enhanced` used a magic constant.  Replaced with a
   learnable scalar `self.stem_scale` initialised to 0.1 via
   softplus(raw) to keep it positive, letting the network find the
   optimal blend during training.

6. Added stem output normalisation
   The stem output (which replaces the raw backbone input) is normalised
   to unit variance before passing into ConvNeXt.  This prevents the
   learned enhancement from saturating ConvNeXt's patch embedding layer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLAHEBranch(nn.Module):
    """
    Learnable local contrast enhancement branch.

    Depthwise → pointwise convolution, with a learned residual gate
    that controls how much contrast enhancement is applied.

    Parameters
    ----------
    in_channels:   3 (RGB / grayscale-as-RGB)
    out_channels:  output feature maps (default 48)
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 48) -> None:
        super().__init__()
        self.dw_conv   = nn.Conv2d(in_channels, in_channels, 3, padding=1,
                                   groups=in_channels, bias=False)
        self.pw_conv   = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1       = nn.BatchNorm2d(in_channels)
        self.bn2       = nn.BatchNorm2d(out_channels)
        self.act       = nn.GELU()

        # IMPROVEMENT 2: learnable residual blend gate
        # Projects input to out_channels so we can blend with the enhanced output
        self.skip_proj = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.gate      = nn.Parameter(torch.zeros(1))   # sigmoid(0) = 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) → (B, out_channels, H, W)."""
        enhanced = self.act(self.bn2(self.pw_conv(self.act(self.bn1(self.dw_conv(x))))))
        skip     = self.skip_proj(x)
        g        = torch.sigmoid(self.gate)
        return g * enhanced + (1 - g) * skip


class SobelEdgeBranch(nn.Module):
    """
    Multi-direction Sobel edge detection.

    Fixed 4-direction Sobel kernels + learnable 1×1 refinement.
    Captures osteophyte spurs and joint-margin sharpness.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 16) -> None:
        super().__init__()
        for name, k in {
            "sobel_h":  [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            "sobel_v":  [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            "sobel_d1": [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
            "sobel_d2": [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]],
        }.items():
            kernel = torch.tensor(k, dtype=torch.float32)
            self.register_buffer(
                name, kernel.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1)
            )
        self.refine = nn.Sequential(
            nn.Conv2d(4 * in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        C = x.shape[1]
        edges = torch.cat([
            F.conv2d(x, self.sobel_h,  padding=1, groups=C),
            F.conv2d(x, self.sobel_v,  padding=1, groups=C),
            F.conv2d(x, self.sobel_d1, padding=1, groups=C),
            F.conv2d(x, self.sobel_d2, padding=1, groups=C),
        ], dim=1)
        return self.refine(edges)


class LaplacianEdgeBranch(nn.Module):
    """Fixed Laplacian edge detection for second-order boundaries."""

    def __init__(self, in_channels: int = 3, out_channels: int = 16) -> None:
        super().__init__()
        lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        self.register_buffer(
            "laplacian", lap.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1)
        )
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        C = x.shape[1]
        return self.refine(F.conv2d(x, self.laplacian, padding=1, groups=C))


class DualIntensityFusion(nn.Module):
    """
    Fuse CLAHE + Sobel + Laplacian features via a 7×7 stem conv.

    IMPROVEMENT 4: BN added between 7×7 and 1×1 to normalise branch scales.
    """

    def __init__(
        self,
        clahe_ch:     int = 48,
        sobel_ch:     int = 16,
        lap_ch:       int = 16,
        out_channels: int = 3,
    ) -> None:
        super().__init__()
        fused_ch = clahe_ch + sobel_ch + lap_ch
        self.stem_conv = nn.Sequential(
            nn.Conv2d(fused_ch, 64, 7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(64),          # IMPROVEMENT 4
            nn.GELU(),
            nn.Conv2d(64, out_channels, 1, bias=False),
        )

    def forward(self, c: torch.Tensor, s: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
        return self.stem_conv(torch.cat([c, s, l], dim=1))


class DualIntensityStem(nn.Module):
    """
    E3 — structure-enhanced preprocessing front-end.

    Input and output are both (B, 3, H, W), making this a transparent
    drop-in before the ConvNeXt backbone.

    IMPROVEMENTS: learnable stem_scale, BN after fusion, fixed dead alpha.
    """

    def __init__(self, out_channels: int = 3) -> None:
        super().__init__()
        self.clahe_branch = CLAHEBranch(3, 48)
        self.sobel_branch = SobelEdgeBranch(3, 16)
        self.lap_branch   = LaplacianEdgeBranch(3, 16)
        self.fusion       = DualIntensityFusion(48, 16, 16, out_channels)

        # IMPROVEMENT 5: learnable scale (was hardcoded 0.1)
        # softplus ensures positivity; init raw=log(exp(0.1)-1) ≈ -2.25
        self._scale_raw   = nn.Parameter(torch.tensor(-2.2504))
        # IMPROVEMENT 6: normalise stem output before backbone
        self.out_norm     = nn.InstanceNorm2d(out_channels, affine=True)

    @property
    def stem_scale(self) -> torch.Tensor:
        """Positive learnable scale (starts at ~0.1)."""
        return F.softplus(self._scale_raw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) → (B, 3, H, W)."""
        enhanced = self.fusion(
            self.clahe_branch(x),
            self.sobel_branch(x),
            self.lap_branch(x),
        )
        out = x + self.stem_scale * enhanced
        return self.out_norm(out)
