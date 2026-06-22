"""
models/dual_intensity.py
========================
E3 — Dual-Intensity Stem.

Three complementary front-ends process the same input image in parallel:

* CLAHEBranch       — depthwise conv simulating local contrast enhancement
* SobelEdgeBranch   — 4-direction fixed Sobel filters for joint margin edges
* LaplacianEdgeBranch — fixed Laplacian for second-order boundary responses

DualIntensityFusion combines all three via a 7×7 stem convolution and
projects back to 3 channels, making the output a drop-in replacement for
the raw RGB input into ConvNeXt-tiny.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLAHEBranch(nn.Module):
    """
    Learnable CLAHE-style local contrast branch.

    Uses depthwise + pointwise convolutions to learn to extract
    subchondral bone density and joint-space darkness features.

    Parameters
    ----------
    in_channels:
        Input channel count (3 for RGB).
    out_channels:
        Number of output feature maps (default 32).
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 32) -> None:
        super().__init__()
        self.dw_conv = nn.Conv2d(
            in_channels, in_channels, 3, padding=1,
            groups=in_channels, bias=False
        )
        self.pw_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) → (B, out_channels, H, W)."""
        x = self.act(self.bn1(self.dw_conv(x)))
        return self.act(self.bn2(self.pw_conv(x)))


class SobelEdgeBranch(nn.Module):
    """
    Multi-direction Sobel edge detection branch.

    Fixed (non-learnable) Sobel kernels in four directions capture
    osteophyte spurs regardless of orientation.  A learnable 1×1 conv
    then refines the concatenated edge responses.

    Parameters
    ----------
    in_channels:
        Input channel count (3 for RGB).
    out_channels:
        Refined output channels (default 16).
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 16) -> None:
        super().__init__()
        _kernels = {
            "sobel_h":  [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            "sobel_v":  [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            "sobel_d1": [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
            "sobel_d2": [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]],
        }
        for name, k in _kernels.items():
            kernel = torch.tensor(k, dtype=torch.float32)
            self.register_buffer(
                name, kernel.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1)
            )
        self.refine = nn.Sequential(
            nn.Conv2d(4 * in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self._in_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) → (B, out_channels, H, W)."""
        C = x.shape[1]
        edges = torch.cat([
            F.conv2d(x, self.sobel_h,  padding=1, groups=C),
            F.conv2d(x, self.sobel_v,  padding=1, groups=C),
            F.conv2d(x, self.sobel_d1, padding=1, groups=C),
            F.conv2d(x, self.sobel_d2, padding=1, groups=C),
        ], dim=1)
        return self.refine(edges)


class LaplacianEdgeBranch(nn.Module):
    """
    Fixed Laplacian edge detection branch.

    Detects second-order intensity changes (blob/circular structures) in
    osteophyte and tibial plateau regions, complementing the Sobel branch.

    Parameters
    ----------
    in_channels:
        Input channel count (3 for RGB).
    out_channels:
        Refined output channels (default 16).
    """

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
        """(B, 3, H, W) → (B, out_channels, H, W)."""
        C = x.shape[1]
        return self.refine(F.conv2d(x, self.laplacian, padding=1, groups=C))


class DualIntensityFusion(nn.Module):
    """
    Fuses outputs of all three intensity branches via a 7×7 stem conv.

    The 7×7 kernel matches the ConvNeXt patch-embedding receptive field,
    so the fused output is a compatible drop-in for the backbone input.
    Output spatial size equals input spatial size (stride-1 conv).

    Parameters
    ----------
    clahe_ch / sobel_ch / lap_ch:
        Output channels from each upstream branch.
    out_channels:
        Final output channels (3 to remain backbone-compatible).
    """

    def __init__(
        self,
        clahe_ch: int = 32,
        sobel_ch: int = 16,
        lap_ch: int = 16,
        out_channels: int = 3,
    ) -> None:
        super().__init__()
        fused_ch = clahe_ch + sobel_ch + lap_ch
        self.stem_conv = nn.Sequential(
            nn.Conv2d(fused_ch, 64, 7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, out_channels, 1, bias=False),
        )

    def forward(
        self,
        clahe_feat: torch.Tensor,
        sobel_feat: torch.Tensor,
        lap_feat: torch.Tensor,
    ) -> torch.Tensor:
        """(B, C_c, H, W), (B, C_s, H, W), (B, C_l, H, W) → (B, out_ch, H, W)."""
        return self.stem_conv(torch.cat([clahe_feat, sobel_feat, lap_feat], dim=1))


class DualIntensityStem(nn.Module):
    """
    E3 top-level module — structure-enhanced front-end.

    Runs CLAHE, Sobel, and Laplacian branches in parallel, then fuses
    their outputs back to 3 channels.  Input and output spatial dimensions
    are identical, so this is a transparent drop-in before the backbone.

    Parameters
    ----------
    out_channels:
        Fused output channels; must be 3 to be backbone-compatible.
    """

    def __init__(self, out_channels: int = 3) -> None:
        super().__init__()
        self.clahe_branch = CLAHEBranch(in_channels=3, out_channels=32)
        self.sobel_branch = SobelEdgeBranch(in_channels=3, out_channels=16)
        self.lap_branch   = LaplacianEdgeBranch(in_channels=3, out_channels=16)
        self.fusion       = DualIntensityFusion(32, 16, 16, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) → (B, 3, H, W) — structure-enhanced."""
        return self.fusion(
            self.clahe_branch(x),
            self.sobel_branch(x),
            self.lap_branch(x),
        )
