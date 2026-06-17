"""
blocks_34.py  —  E3: Dual-Intensity Stem  |  E4: Compartment Branches


  - Each experiment block is self-contained
  - Each block ends with a standalone test function
  - Integration test ties both blocks + ConvNeXt-tiny backbone together
  - Run everything:  python blocks_34.py
  - Run one block:   python blocks_34.py --block e3
                     python blocks_34.py --block e4
                     python blocks_34.py --block integration
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG  (mirrors config.yaml — identical to blocks_56.py)
# ──────────────────────────────────────────────────────────────────────────────
CONFIG = {
    "dataset": {
        "name": "OAI",
        "image_size": 229,
        "num_classes": 5,
    },
    "model": {
        "backbone": "convnext_tiny",
        "pretrained": False,          # set True when real weights are available
        "in_channels": 3,
        "num_classes": 5,
    },
    "training": {
        "optimizer": "adamw",
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "batch_size": 16,
        "epochs": 100,
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# DATA AUGMENTATION  (shared across all blocks)
# ──────────────────────────────────────────────────────────────────────────────
def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ══════════════════════════════════════════════════════════════════════════════
#  E3 — DUAL-INTENSITY STEM
# ══════════════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────────────────
# E3-A: CLAHE Preprocessing Branch
# ──────────────────────────────────────────────────────────────────────────────
class CLAHEBranch(nn.Module):
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization) preprocessing branch.

    Simulates CLAHE-style local contrast enhancement using learnable
    depthwise convolutions. In deployment, actual CLAHE preprocessing
    is applied before the network. This branch learns to extract
    subchondral bone density and joint space darkness features.

    Architecture:
        DW Conv 3x3  →  BN  →  GELU  →  PW Conv 1x1  →  BN  →  GELU
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 32):
        super().__init__()
        self.dw_conv = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3, padding=1,
            groups=in_channels, bias=False
        )
        self.pw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)  — normalized RGB radiograph
        Returns:
            features: (B, out_channels, H, W)
        """
        x = self.act(self.bn1(self.dw_conv(x)))
        x = self.act(self.bn2(self.pw_conv(x)))
        return x


# ──────────────────────────────────────────────────────────────────────────────
# E3-B: Sobel Edge Branch
# ──────────────────────────────────────────────────────────────────────────────
class SobelEdgeBranch(nn.Module):
    """
    Sobel edge detection branch for osteophyte margin and structural boundary features.

    Applies fixed Sobel kernels in 4 directions (H, V, 45°, 135°) as a
    non-learnable front-end, then refines with a learnable conv layer.
    This captures osteophyte spurs regardless of their orientation.

    Architecture:
        Fixed Sobel filters (4 directions)  →  Concat  →  Conv 1x1  →  BN  →  GELU
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 16):
        super().__init__()

        # Fixed Sobel kernels — not updated by gradient
        self.register_buffer('sobel_h', self._make_sobel_kernel('horizontal', in_channels))
        self.register_buffer('sobel_v', self._make_sobel_kernel('vertical', in_channels))
        self.register_buffer('sobel_d1', self._make_sobel_kernel('diag45', in_channels))
        self.register_buffer('sobel_d2', self._make_sobel_kernel('diag135', in_channels))

        # Learnable refinement after edge extraction
        self.refine = nn.Sequential(
            nn.Conv2d(4 * in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    @staticmethod
    def _make_sobel_kernel(direction: str, in_channels: int) -> torch.Tensor:
        """Creates a grouped Sobel conv kernel for the given direction."""
        if direction == 'horizontal':
            k = torch.tensor([[-1, -2, -1],
                               [ 0,  0,  0],
                               [ 1,  2,  1]], dtype=torch.float32)
        elif direction == 'vertical':
            k = torch.tensor([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]], dtype=torch.float32)
        elif direction == 'diag45':
            k = torch.tensor([[ 0,  1,  2],
                               [-1,  0,  1],
                               [-2, -1,  0]], dtype=torch.float32)
        else:  # diag135
            k = torch.tensor([[-2, -1,  0],
                               [-1,  0,  1],
                               [ 0,  1,  2]], dtype=torch.float32)
        # Shape: (in_channels, 1, 3, 3) for depthwise grouped conv
        return k.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            edge_features: (B, out_channels, H, W)
        """
        C = x.shape[1]
        h  = F.conv2d(x, self.sobel_h,  padding=1, groups=C)
        v  = F.conv2d(x, self.sobel_v,  padding=1, groups=C)
        d1 = F.conv2d(x, self.sobel_d1, padding=1, groups=C)
        d2 = F.conv2d(x, self.sobel_d2, padding=1, groups=C)
        edges = torch.cat([h, v, d1, d2], dim=1)   # (B, 4*C, H, W)
        return self.refine(edges)


# ──────────────────────────────────────────────────────────────────────────────
# E3-C: Laplacian Edge Branch
# ──────────────────────────────────────────────────────────────────────────────
class LaplacianEdgeBranch(nn.Module):
    """
    Laplacian edge detection branch for tibial plateau margin and fine boundary features.

    Applies a fixed Laplacian kernel as a non-learnable front-end to
    capture second-order intensity changes, then refines with a
    learnable conv layer. Complements Sobel by detecting blob-like
    and circular edge structures in osteophyte regions.

    Architecture:
        Fixed Laplacian filter  →  Conv 1x1  →  BN  →  GELU
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 16):
        super().__init__()

        lap_kernel = torch.tensor([[0,  1, 0],
                                   [1, -4, 1],
                                   [0,  1, 0]], dtype=torch.float32)
        self.register_buffer(
            'laplacian',
            lap_kernel.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1)
        )

        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            lap_features: (B, out_channels, H, W)
        """
        C = x.shape[1]
        lap = F.conv2d(x, self.laplacian, padding=1, groups=C)  # (B, C, H, W)
        return self.refine(lap)


# ──────────────────────────────────────────────────────────────────────────────
# E3-D: Feature Fusion Module
# ──────────────────────────────────────────────────────────────────────────────
class DualIntensityFusion(nn.Module):
    """
    Feature fusion module that combines CLAHE, Sobel, and Laplacian branches.

    Concatenates all three branch outputs and projects to the target
    output channel count via a 7x7 stem conv (matching the ConvNeXt
    patch embedding stem). The output is a fused 3-channel map
    suitable as a direct drop-in for the ConvNeXt input.

    Architecture:
        Concat(CLAHE, Sobel, Laplacian)  →  7x7 Conv  →  BN  →  GELU  →  1x1 Conv
    """

    def __init__(
        self,
        clahe_ch: int = 32,
        sobel_ch: int = 16,
        lap_ch: int = 16,
        out_channels: int = 3,
    ):
        super().__init__()
        fused_ch = clahe_ch + sobel_ch + lap_ch  # 64

        self.stem_conv = nn.Sequential(
            nn.Conv2d(fused_ch, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, out_channels, kernel_size=1, bias=False),
        )

    def forward(
        self,
        clahe_feat: torch.Tensor,
        sobel_feat: torch.Tensor,
        lap_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            clahe_feat: (B, 32, H, W)
            sobel_feat: (B, 16, H, W)
            lap_feat:   (B, 16, H, W)
        Returns:
            fused: (B, out_channels, H, W)  — same spatial dims as input
        """
        fused = torch.cat([clahe_feat, sobel_feat, lap_feat], dim=1)
        return self.stem_conv(fused)


# ──────────────────────────────────────────────────────────────────────────────
# E3: DualIntensityStem  (top-level E3 module)
# ──────────────────────────────────────────────────────────────────────────────
class DualIntensityStem(nn.Module):
    """
    Dual-Intensity Stem  —  E3 top-level module.

    Wraps CLAHEBranch, SobelEdgeBranch, LaplacianEdgeBranch, and
    DualIntensityFusion into a single forward call.

    The output is a 3-channel tensor of the same spatial size as the
    input, suitable for direct use as the input to ConvNeXt-Tiny.
    The stem does not downsample, preserving fine structural detail
    for subsequent backbone processing.

    Input:  (B, 3, H, W)  — raw normalized radiograph
    Output: (B, 3, H, W)  — structure-enhanced representation
    """

    def __init__(self, out_channels: int = 3):
        super().__init__()
        self.clahe_branch  = CLAHEBranch(in_channels=3, out_channels=32)
        self.sobel_branch  = SobelEdgeBranch(in_channels=3, out_channels=16)
        self.lap_branch    = LaplacianEdgeBranch(in_channels=3, out_channels=16)
        self.fusion        = DualIntensityFusion(
            clahe_ch=32, sobel_ch=16, lap_ch=16,
            out_channels=out_channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            enhanced: (B, 3, H, W)  — dual-intensity fused output
        """
        clahe_feat = self.clahe_branch(x)
        sobel_feat = self.sobel_branch(x)
        lap_feat   = self.lap_branch(x)
        return self.fusion(clahe_feat, sobel_feat, lap_feat)


# ──────────────────────────────────────────────────────────────────────────────
# E3 Unit Test
# ──────────────────────────────────────────────────────────────────────────────
def test_e3_units():
    print("=" * 60)
    print("E3 — Unit Tests")
    print("=" * 60)
    B, C, H, W = 4, 3, 224, 224
    x = torch.randn(B, C, H, W)

    # CLAHEBranch
    clahe = CLAHEBranch(in_channels=3, out_channels=32)
    out_clahe = clahe(x)
    assert out_clahe.shape == (B, 32, H, W), f"CLAHE shape wrong: {out_clahe.shape}"
    print(f"  CLAHEBranch output      : {out_clahe.shape}  ✓")

    # SobelEdgeBranch
    sobel = SobelEdgeBranch(in_channels=3, out_channels=16)
    out_sobel = sobel(x)
    assert out_sobel.shape == (B, 16, H, W), f"Sobel shape wrong: {out_sobel.shape}"
    print(f"  SobelEdgeBranch output  : {out_sobel.shape}  ✓")

    # LaplacianEdgeBranch
    lap = LaplacianEdgeBranch(in_channels=3, out_channels=16)
    out_lap = lap(x)
    assert out_lap.shape == (B, 16, H, W), f"Laplacian shape wrong: {out_lap.shape}"
    print(f"  LaplacianEdgeBranch out : {out_lap.shape}  ✓")

    # DualIntensityFusion
    fusion = DualIntensityFusion(clahe_ch=32, sobel_ch=16, lap_ch=16, out_channels=3)
    out_fused = fusion(out_clahe, out_sobel, out_lap)
    assert out_fused.shape == (B, 3, H, W), f"Fusion shape wrong: {out_fused.shape}"
    print(f"  DualIntensityFusion out : {out_fused.shape}  ✓")

    # Full DualIntensityStem
    stem = DualIntensityStem(out_channels=3)
    out_stem = stem(x)
    assert out_stem.shape == (B, 3, H, W), f"Stem shape wrong: {out_stem.shape}"
    print(f"  DualIntensityStem out   : {out_stem.shape}  ✓")
    print()


def validate_e3_with_backbone():
    print("=" * 60)
    print("E3 — Validation with ConvNeXt-Tiny Backbone")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4

    stem = DualIntensityStem(out_channels=3).to(device)

    backbone = models.convnext_tiny(weights=None).to(device)
    backbone.classifier = nn.Identity()
    backbone.eval()

    dummy = torch.randn(batch_size, 3, 224, 224, device=device)

    enhanced = stem(dummy)                        # (B, 3, 224, 224)
    with torch.no_grad():
        feat = backbone(enhanced).flatten(1)      # (B, 768)

    assert enhanced.shape == (batch_size, 3, 224, 224), \
        f"Enhanced shape: {enhanced.shape}"
    assert feat.shape == (batch_size, 768), \
        f"Backbone feat shape: {feat.shape}"

    print(f"  Input image             : {dummy.shape}  ✓")
    print(f"  Dual-intensity enhanced : {enhanced.shape}  ✓")
    print(f"  ConvNeXt-Tiny feature   : {feat.shape}  ✓")
    print("  E3 VALIDATION PASS  ✓\n")


# ══════════════════════════════════════════════════════════════════════════════
#  E4 — COMPARTMENT BRANCHES
# ══════════════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────────────────
# E4-A: Edge-Gated Residual Block (EGRB)
# ──────────────────────────────────────────────────────────────────────────────
class EdgeGatedResidualBlock(nn.Module):
    """
    Edge-Gated Residual Block (EGRB).

    Applies a directional Sobel gate multiplicatively to the main
    depthwise conv branch. This preserves fine structural activations
    at joint margins and osteophyte spurs that standard convolutions
    suppress at deeper feature levels.

    Architecture:
        Main: DW Conv 3x3  →  GELU  →  PW Conv 1x1
        Gate: Sobel(H+V+45+135)  →  1x1 proj  →  Sigmoid
        Output = Main ⊗ Gate + Input  (residual)
    """

    def __init__(self, channels: int):
        super().__init__()

        # Main branch
        self.dw_conv = nn.Conv2d(channels, channels, 3, padding=1,
                                 groups=channels, bias=False)
        self.pw_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.GELU()

        # Edge gate: project 4-direction Sobel response to channel gate
        self.edge_proj = nn.Sequential(
            nn.Conv2d(4, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        # Fixed Sobel kernels (single-channel, applied per-channel via loop)
        self.register_buffer('kH',  torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],
                                                   dtype=torch.float32).view(1,1,3,3))
        self.register_buffer('kV',  torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],
                                                   dtype=torch.float32).view(1,1,3,3))
        self.register_buffer('kD1', torch.tensor([[0,1,2],[-1,0,1],[-2,-1,0]],
                                                   dtype=torch.float32).view(1,1,3,3))
        self.register_buffer('kD2', torch.tensor([[-2,-1,0],[-1,0,1],[0,1,2]],
                                                   dtype=torch.float32).view(1,1,3,3))

    def _sobel_gate(self, x: torch.Tensor) -> torch.Tensor:
        """Compute mean Sobel response across channels → (B, 4, H, W)."""
        # Average across channels for a single-channel edge map per direction
        x_gray = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        h  = F.conv2d(x_gray, self.kH,  padding=1)
        v  = F.conv2d(x_gray, self.kV,  padding=1)
        d1 = F.conv2d(x_gray, self.kD1, padding=1)
        d2 = F.conv2d(x_gray, self.kD2, padding=1)
        return torch.cat([h, v, d1, d2], dim=1)  # (B, 4, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            out: (B, C, H, W)  — edge-gated residual output
        """
        # Main branch
        main = self.act(self.bn(self.pw_conv(self.dw_conv(x))))

        # Edge gate
        edge_response = self._sobel_gate(x)          # (B, 4, H, W)
        gate = self.edge_proj(edge_response)          # (B, C, H, W)  in [0, 1]

        return main * gate + x                        # gated + residual


# ──────────────────────────────────────────────────────────────────────────────
# E4-B: Single Compartment Branch
# ──────────────────────────────────────────────────────────────────────────────
class CompartmentBranch(nn.Module):
    """
    Single compartment encoding branch (medial or lateral).

    Takes a compartment crop (same spatial size as the global crop),
    passes it through the shared DualIntensityStem, then through
    ConvNeXt-Tiny features, and finally through an EGRB to
    preserve fine joint margin activations.

    The branch uses shared weights with its counterpart branch,
    ensuring symmetric treatment of medial and lateral compartments.

    Input:  (B, 3, H, W)  — compartment patch
    Output: (B, 768)       — compartment feature vector
    """

    def __init__(self, stem: DualIntensityStem, backbone_features: nn.Module):
        super().__init__()
        self.stem = stem                         # shared DualIntensityStem
        self.features = backbone_features        # shared ConvNeXt feature extractor
        self.egrb = EdgeGatedResidualBlock(768)  # applied after spatial features
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)  — compartment crop
        Returns:
            feat: (B, 768)
        """
        enhanced = self.stem(x)                  # (B, 3, H, W)
        spatial = self.features(enhanced)         # (B, 768, 7, 7)
        gated = self.egrb(spatial)               # (B, 768, 7, 7)
        pooled = self.pool(gated).flatten(1)     # (B, 768)
        return pooled


# ──────────────────────────────────────────────────────────────────────────────
# E4-C: Global Branch
# ──────────────────────────────────────────────────────────────────────────────
class GlobalBranch(nn.Module):
    """
    Global whole-knee encoding branch.

    Processes the full-knee crop through the DualIntensityStem and
    ConvNeXt-Tiny backbone, with an EGRB applied on spatial features
    before pooling. Provides the global context vector that is
    fused with compartment-specific features.

    Input:  (B, 3, H, W)  — whole-knee crop
    Output: (B, 768)       — global feature vector
    """

    def __init__(self, stem: DualIntensityStem, backbone_features: nn.Module):
        super().__init__()
        self.stem = stem
        self.features = backbone_features
        self.egrb = EdgeGatedResidualBlock(768)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)  — global whole-knee crop
        Returns:
            feat: (B, 768)
        """
        enhanced = self.stem(x)
        spatial = self.features(enhanced)
        gated = self.egrb(spatial)
        return self.pool(gated).flatten(1)


# ──────────────────────────────────────────────────────────────────────────────
# E4-D: Feature Fusion between Global and Compartment Features
# ──────────────────────────────────────────────────────────────────────────────
class CompartmentFusion(nn.Module):
    """
    Feature fusion between global and compartment branch outputs.

    Concatenates global (768), medial (768), and lateral (768) feature
    vectors and projects to a unified 768-d fused representation.

    A gating mechanism learns how much each branch contributes to
    the fused output, preventing any single branch from dominating.

    Architecture:
        Concat(global, medial, lateral) [B, 2304]
            → Gate [B, 3]  (softmax over branches)
            → Weighted sum [B, 768]
            → LayerNorm  →  GELU  →  Linear(768, 768)
    """

    def __init__(self, feat_dim: int = 768):
        super().__init__()
        self.feat_dim = feat_dim
        total = feat_dim * 3   # global + medial + lateral

        # Branch attention gate
        self.gate = nn.Sequential(
            nn.Linear(total, 3),
            nn.Softmax(dim=-1),
        )

        # Final projection
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
        )

    def forward(
        self,
        global_feat: torch.Tensor,   # (B, 768)
        medial_feat: torch.Tensor,   # (B, 768)
        lateral_feat: torch.Tensor,  # (B, 768)
    ) -> torch.Tensor:               # (B, 768)
        """
        Gated fusion of global, medial, and lateral features.
        """
        concat = torch.cat([global_feat, medial_feat, lateral_feat], dim=1)  # (B, 2304)

        # Gate weights over 3 branches
        weights = self.gate(concat)           # (B, 3)
        w_g = weights[:, 0:1]                 # (B, 1)
        w_m = weights[:, 1:2]                 # (B, 1)
        w_l = weights[:, 2:3]                 # (B, 1)

        fused = w_g * global_feat + w_m * medial_feat + w_l * lateral_feat  # (B, 768)
        return self.proj(fused)


# ──────────────────────────────────────────────────────────────────────────────
# E4: CompartmentBranchModule  (top-level E4 module)
# ──────────────────────────────────────────────────────────────────────────────
class CompartmentBranchModule(nn.Module):
    """
    Compartment Branch Module  —  E4 top-level module.

    Wraps the shared DualIntensityStem (E3), shared ConvNeXt-Tiny
    feature extractor, global branch, medial branch, lateral branch,
    and CompartmentFusion into a single forward call.

    Medial and lateral branches share weights — symmetric treatment
    ensures neither compartment is implicitly favoured.

    Inputs:
        global_crop:  (B, 3, H, W)  — whole-knee crop
        medial_crop:  (B, 3, H, W)  — medial compartment patch
        lateral_crop: (B, 3, H, W)  — lateral compartment patch

    Outputs:
        fused_feat:   (B, 768)      — gated fusion of all three branches
        global_feat:  (B, 768)      — global branch output (for E7 RTC)
        medial_feat:  (B, 768)      — medial branch output (for E7 RTC)
        lateral_feat: (B, 768)      — lateral branch output (for E7 RTC)
    """

    def __init__(self):
        super().__init__()

        # Shared DualIntensityStem (E3)
        self.stem = DualIntensityStem(out_channels=3)

        # Shared ConvNeXt-Tiny feature extractor (spatial output)
        _backbone = models.convnext_tiny(
            weights=None if not CONFIG["model"]["pretrained"] else "IMAGENET1K_V1"
        )
        # Use all feature stages; output is (B, 768, 7, 7) for 224px input
        self.backbone_features = nn.Sequential(*list(_backbone.features.children()))

        # Global branch (uses shared stem + backbone)
        self.global_branch = GlobalBranch(self.stem, self.backbone_features)

        # Compartment branches — shared weights
        self.compartment_branch = CompartmentBranch(self.stem, self.backbone_features)

        # Fusion
        self.fusion = CompartmentFusion(feat_dim=768)

    def forward(
        self,
        global_crop: torch.Tensor,
        medial_crop: torch.Tensor,
        lateral_crop: torch.Tensor,
    ):
        """
        Args:
            global_crop:  (B, 3, H, W)
            medial_crop:  (B, 3, H, W)
            lateral_crop: (B, 3, H, W)
        Returns:
            fused_feat:   (B, 768)
            global_feat:  (B, 768)
            medial_feat:  (B, 768)
            lateral_feat: (B, 768)
        """
        global_feat  = self.global_branch(global_crop)
        medial_feat  = self.compartment_branch(medial_crop)
        lateral_feat = self.compartment_branch(lateral_crop)
        fused_feat   = self.fusion(global_feat, medial_feat, lateral_feat)

        return fused_feat, global_feat, medial_feat, lateral_feat


# ──────────────────────────────────────────────────────────────────────────────
# E4 Unit Test
# ──────────────────────────────────────────────────────────────────────────────
def test_e4_units():
    print("=" * 60)
    print("E4 — Unit Tests")
    print("=" * 60)
    B, C, H, W = 4, 768, 7, 7

    # EdgeGatedResidualBlock
    feat = torch.randn(B, C, H, W)
    egrb = EdgeGatedResidualBlock(channels=C)
    out_egrb = egrb(feat)
    assert out_egrb.shape == feat.shape, f"EGRB shape wrong: {out_egrb.shape}"
    print(f"  EdgeGatedResidualBlock  : {out_egrb.shape}  ✓")

    # CompartmentFusion
    g = torch.randn(B, 768)
    m = torch.randn(B, 768)
    l = torch.randn(B, 768)
    fusion = CompartmentFusion(feat_dim=768)
    out_fused = fusion(g, m, l)
    assert out_fused.shape == (B, 768), f"Fusion shape wrong: {out_fused.shape}"
    print(f"  CompartmentFusion out   : {out_fused.shape}  ✓")
    print()


def validate_e4_with_backbone():
    print("=" * 60)
    print("E4 — Validation with ConvNeXt-Tiny Backbone")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4

    e4 = CompartmentBranchModule().to(device)

    global_crop  = torch.randn(batch_size, 3, 224, 224, device=device)
    medial_crop  = torch.randn(batch_size, 3, 224, 224, device=device)
    lateral_crop = torch.randn(batch_size, 3, 224, 224, device=device)

    fused, g_feat, m_feat, l_feat = e4(global_crop, medial_crop, lateral_crop)

    assert fused.shape   == (batch_size, 768), f"Fused shape: {fused.shape}"
    assert g_feat.shape  == (batch_size, 768), f"Global shape: {g_feat.shape}"
    assert m_feat.shape  == (batch_size, 768), f"Medial shape: {m_feat.shape}"
    assert l_feat.shape  == (batch_size, 768), f"Lateral shape: {l_feat.shape}"

    print(f"  Global feature          : {g_feat.shape}  ✓")
    print(f"  Medial feature          : {m_feat.shape}  ✓")
    print(f"  Lateral feature         : {l_feat.shape}  ✓")
    print(f"  Fused feature           : {fused.shape}  ✓")
    print("  E4 VALIDATION PASS  ✓\n")


# ══════════════════════════════════════════════════════════════════════════════
#  INTEGRATION TEST — E3 + E4 + ConvNeXt-Tiny  (end-to-end + gradient check)
# ══════════════════════════════════════════════════════════════════════════════
def run_e3_e4_integration_test():
    print("=" * 60)
    print("E3 + E4 Integration Test")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    num_classes = CONFIG["model"]["num_classes"]

    # ── 1. E3 + E4 CompartmentBranchModule ───────────────────────────────────
    e4 = CompartmentBranchModule().to(device)

    # ── 2. Thin classifier head on fused features ────────────────────────────
    classifier = nn.Linear(768, num_classes).to(device)

    # ── 3. Simulate 3-crop input (matches E7/E8 pipeline) ────────────────────
    global_crop  = torch.randn(batch_size, 3, 224, 224, device=device)
    medial_crop  = torch.randn(batch_size, 3, 224, 224, device=device)
    lateral_crop = torch.randn(batch_size, 3, 224, 224, device=device)

    # ── 4. Forward pass ───────────────────────────────────────────────────────
    fused_feat, g_feat, m_feat, l_feat = e4(
        global_crop, medial_crop, lateral_crop
    )
    print(f"  [E3+E4] Fused feature   : {fused_feat.shape}")
    print(f"  [E4]    Global feature  : {g_feat.shape}   → feeds E7 RTC")
    print(f"  [E4]    Medial feature  : {m_feat.shape}   → feeds E7 RTC")
    print(f"  [E4]    Lateral feature : {l_feat.shape}  → feeds E7 RTC")

    logits = classifier(fused_feat)                   # (B, 5)
    print(f"  [Cls]   Class logits    : {logits.shape}")

    # ── 5. Loss + backward ───────────────────────────────────────────────────
    labels = torch.randint(0, num_classes, (batch_size,), device=device)
    loss = F.cross_entropy(logits, labels)
    loss.backward()

    # ── 6. Gradient checks ───────────────────────────────────────────────────
    stem_grad_ok = e4.stem.clahe_branch.pw_conv.weight.grad is not None
    egrb_grad_ok = e4.global_branch.egrb.pw_conv.weight.grad is not None
    fusion_grad_ok = e4.fusion.proj[0].weight.grad is not None
    cls_grad_ok = classifier.weight.grad is not None

    assert stem_grad_ok,   "No gradient to DualIntensityStem"
    assert egrb_grad_ok,   "No gradient to EGRB"
    assert fusion_grad_ok, "No gradient to CompartmentFusion"
    assert cls_grad_ok,    "No gradient to classifier"

    print()
    print(f"  Loss                         : {loss.item():.4f}")
    print(f"  Gradient flow to Stem        : {stem_grad_ok}  ✓")
    print(f"  Gradient flow to EGRB        : {egrb_grad_ok}  ✓")
    print(f"  Gradient flow to Fusion      : {fusion_grad_ok}  ✓")
    print(f"  Gradient flow to Classifier  : {cls_grad_ok}  ✓")
    print()
    print("  E3 + E4 INTEGRATION PASS  ✓\n")


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING STUB
#  Shows how E3 / E4 slot into the shared training loop.
#  Outputs (g_feat, m_feat, l_feat) feed directly into E7 RTC.
# ══════════════════════════════════════════════════════════════════════════════
def training_stub():
    """
    Minimal training loop demonstrating:
      - E3 DualIntensityStem as the front-end
      - E4 CompartmentBranchModule as the encoder
      - Output (g_feat, m_feat, l_feat) ready for E7 RTC
      - Gradient flow through all components
    """
    print("=" * 60)
    print("Training Stub  (3 mini-batches)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = CONFIG["training"]["batch_size"]
    num_classes = CONFIG["model"]["num_classes"]

    e4 = CompartmentBranchModule().to(device)
    head = nn.Linear(768, num_classes).to(device)

    optimizer = torch.optim.AdamW(
        list(e4.parameters()) + list(head.parameters()),
        lr=CONFIG["training"]["learning_rate"],
        weight_decay=CONFIG["training"]["weight_decay"],
    )

    for step in range(3):
        optimizer.zero_grad()

        g_crop = torch.randn(B, 3, 224, 224, device=device)
        m_crop = torch.randn(B, 3, 224, 224, device=device)
        l_crop = torch.randn(B, 3, 224, 224, device=device)
        labels = torch.randint(0, num_classes, (B,), device=device)

        fused, g_feat, m_feat, l_feat = e4(g_crop, m_crop, l_crop)
        logits = head(fused)
        loss = F.cross_entropy(logits, labels)

        loss.backward()
        optimizer.step()

        print(f"  Step {step+1}/3  loss={loss.item():.4f}  "
              f"fused={fused.shape}  "
              f"g/m/l ready for E7 RTC: {g_feat.shape}")

    print("  Training stub complete  ✓\n")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
BLOCKS = {
    "e3": [test_e3_units, validate_e3_with_backbone],
    "e4": [test_e4_units, validate_e4_with_backbone],
    "integration": [run_e3_e4_integration_test, training_stub],
}


def main():
    parser = argparse.ArgumentParser(description="Run E3/E4 block tests")
    parser.add_argument(
        "--block",
        choices=["e3", "e4", "integration", "all"],
        default="all",
        help="Which block(s) to test  (default: all)",
    )
    args = parser.parse_args()

    targets = (
        BLOCKS["e3"] + BLOCKS["e4"] + BLOCKS["integration"]
        if args.block == "all"
        else BLOCKS[args.block]
    )

    for fn in targets:
        fn()

    print("All selected tests passed  ✓")


if __name__ == "__main__":
    main()
