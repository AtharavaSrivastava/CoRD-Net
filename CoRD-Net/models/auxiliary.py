"""
models/auxiliary.py
===================
E8 — Auxiliary prediction heads for multi-task grading.

Seven heads attach to the fused 512-d representation:

H1  PrimaryKLHead          — 5-class KL grade (log-softmax)
H2  CORALOrdinalHead       — 4 binary ordinal classifiers
H3  MetricEmbeddingHead    — L2-normalised 128-d embedding (SupCon)
H4  MedialJSNHead          — 4-class medial JSN grading (log-softmax)
H5  LateralJSNHead         — 4-class lateral JSN grading (log-softmax)
H6  OsteophyteHeads        — 4 × 3-class osteophyte heads (log-softmax)
H7  UncertaintyHead        — non-negative scalar uncertainty estimate
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrimaryKLHead(nn.Module):
    """
    H1 — Primary KL grade classifier.

    Parameters
    ----------
    in_dim:    Input feature dimension.
    num_classes: Number of output classes (5 for KL 0–4).
    """

    def __init__(self, in_dim: int = 512, num_classes: int = 5) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, in_dim) → (B, num_classes) log-probabilities."""
        return F.log_softmax(self.fc(x), dim=-1)


class CORALOrdinalHead(nn.Module):
    """
    H2 — CORAL ordinal regression head.

    Outputs raw logits for 4 binary classifiers representing cumulative
    ordinal thresholds (KL ≥ 1, ≥ 2, ≥ 3, ≥ 4).

    Parameters
    ----------
    in_dim: Input feature dimension.
    """

    def __init__(self, in_dim: int = 512) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, in_dim) → (B, 4) raw logits."""
        return self.fc(x)


class MetricEmbeddingHead(nn.Module):
    """
    H3 — L2-normalised metric embedding for Supervised Contrastive loss.

    Parameters
    ----------
    in_dim:     Input feature dimension.
    embed_dim:  Output embedding dimension (default 128).
    """

    def __init__(self, in_dim: int = 512, embed_dim: int = 128) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, in_dim) → (B, embed_dim) unit-norm embedding."""
        return F.normalize(self.fc(x), p=2, dim=1)


class MedialJSNHead(nn.Module):
    """
    H4 — Medial Joint Space Narrowing grading head (4 classes).

    Parameters
    ----------
    in_dim: Input feature dimension.
    """

    def __init__(self, in_dim: int = 512) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, in_dim) → (B, 4) log-probabilities."""
        return F.log_softmax(self.fc(x), dim=-1)


class LateralJSNHead(nn.Module):
    """
    H5 — Lateral Joint Space Narrowing grading head (4 classes).

    Parameters
    ----------
    in_dim: Input feature dimension.
    """

    def __init__(self, in_dim: int = 512) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, in_dim) → (B, 4) log-probabilities."""
        return F.log_softmax(self.fc(x), dim=-1)


class OsteophyteHeads(nn.Module):
    """
    H6 — Four compartment-specific osteophyte severity heads (3 classes each).

    The four sub-heads correspond to: medial femur, lateral femur,
    medial tibia, lateral tibia.

    Parameters
    ----------
    in_dim: Input feature dimension.
    """

    def __init__(self, in_dim: int = 512) -> None:
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(in_dim, 3) for _ in range(4)])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """(B, in_dim) → list of 4 × (B, 3) log-probability tensors."""
        return [F.log_softmax(h(x), dim=-1) for h in self.heads]


class UncertaintyHead(nn.Module):
    """
    H7 — Predictive uncertainty estimation head.

    Outputs a non-negative scalar via softplus, supervised against the
    entropy of the primary KL prediction.

    Parameters
    ----------
    in_dim: Input feature dimension.
    """

    def __init__(self, in_dim: int = 512) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, in_dim) → (B, 1) non-negative uncertainty scalar."""
        return F.softplus(self.fc(x))
