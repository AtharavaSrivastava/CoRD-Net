"""
models/pgr.py
=============
E6 — Prototype-Guided Refinement (PGR).

One learnable L2-normalised prototype per KL grade is maintained in
GradePrototypeBank.  PrototypeSimilarity converts embedding–prototype
distances to classification logits (auxiliary loss).
PrototypeRefinementModule refines the input embedding via cross-attention
over the prototype set.  PGRModule composes all three.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradePrototypeBank(nn.Module):
    """
    Learnable per-grade prototype memory bank.

    Prototypes are L2-normalised and updated via back-prop.  An EMA
    shadow buffer provides stable inference-time prototypes without
    gradient noise.

    Parameters
    ----------
    num_classes:
        Number of grade classes (5 for KL grades 0–4).
    embed_dim:
        Prototype / embedding dimensionality.
    ema_momentum:
        Momentum coefficient for the EMA shadow update (0 < m < 1).
    """

    def __init__(
        self,
        num_classes: int = 5,
        embed_dim: int = 256,
        ema_momentum: float = 0.99,
    ) -> None:
        super().__init__()
        self.ema_momentum = ema_momentum
        raw = torch.randn(num_classes, embed_dim)
        self.prototypes = nn.Parameter(F.normalize(raw, p=2, dim=1))
        self.register_buffer(
            "ema_prototypes", F.normalize(raw.clone().detach(), p=2, dim=1)
        )

    @torch.no_grad()
    def ema_update(self, class_idx: int, new_embedding: torch.Tensor) -> None:
        """
        Update the EMA shadow prototype for one class.

        Parameters
        ----------
        class_idx:
            Integer grade label (0 … num_classes-1).
        new_embedding:
            Mean embedding of this class in the current batch, shape (embed_dim,).
        """
        m = self.ema_momentum
        self.ema_prototypes[class_idx] = F.normalize(
            m * self.ema_prototypes[class_idx] + (1 - m) * new_embedding,
            p=2,
            dim=0,
        )

    def update_prototypes_from_batch(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> None:
        """
        Convenience method: EMA-update every class present in the batch.

        Parameters
        ----------
        embeddings: (B, embed_dim) detached embedding batch.
        labels:     (B,) integer class labels.
        """
        num_classes = self.prototypes.shape[0]
        for c in range(num_classes):
            mask = labels == c
            if mask.any():
                self.ema_update(c, embeddings[mask].mean(0))

    def forward(self, use_ema: bool = False) -> torch.Tensor:
        """Return (num_classes, embed_dim) prototype matrix."""
        if use_ema:
            return self.ema_prototypes
        return F.normalize(self.prototypes, p=2, dim=1)


class PrototypeSimilarity(nn.Module):
    """
    Temperature-scaled embedding–prototype similarity.

    Produces (B, num_classes) logits usable directly as classification
    scores or as input to an auxiliary cross-entropy loss.

    Parameters
    ----------
    mode:
        'cosine' (default) or 'dot'.
    temperature:
        Scaling factor; lower → sharper distribution.
    """

    def __init__(self, mode: str = "cosine", temperature: float = 0.07) -> None:
        super().__init__()
        if mode not in ("cosine", "dot"):
            raise ValueError(f"Unknown similarity mode: {mode!r}")
        self.mode = mode
        self.temperature = temperature

    def forward(
        self, embeddings: torch.Tensor, prototypes: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        embeddings:  (B, D)
        prototypes:  (K, D)

        Returns
        -------
        logits: (B, K)
        """
        if self.mode == "cosine":
            ne  = F.normalize(embeddings, p=2, dim=1)
            np_ = F.normalize(prototypes, p=2, dim=1)
            return torch.mm(ne, np_.t()) / self.temperature
        return torch.mm(embeddings, prototypes.t()) / self.temperature


class PrototypeRefinementModule(nn.Module):
    """
    Cross-attention refinement of an embedding over the prototype set.

    The input embedding is the query; prototypes are keys and values.
    The attended prototype context is residually added to the embedding,
    then passed through a two-layer FFN — matching the standard
    transformer post-attention block pattern.

    The attention weight tensor (B, 1, K) is stored in ``self.last_attn``
    and can be interpreted as a soft grade-assignment distribution.

    Parameters
    ----------
    embed_dim:   Query / prototype embedding dimension.
    num_classes: Number of prototypes (= K).
    num_heads:   Multi-head attention heads.
    dropout:     Dropout in attention and FFN.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_classes: int = 5,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm  = nn.LayerNorm(embed_dim)
        self.drop  = nn.Dropout(dropout)
        self.ffn   = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2     = nn.LayerNorm(embed_dim)
        self.last_attn: torch.Tensor | None = None

    def forward(
        self, embedding: torch.Tensor, prototypes: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        embedding:  (B, embed_dim)
        prototypes: (K, embed_dim)

        Returns
        -------
        refined: (B, embed_dim)
        """
        B, K = embedding.size(0), prototypes.size(0)
        q  = embedding.unsqueeze(1)                    # (B, 1, D)
        kv = prototypes.unsqueeze(0).expand(B, K, -1)  # (B, K, D)

        attn_out, attn_w = self.cross_attn(q, kv, kv)
        self.last_attn   = attn_w.detach()             # (B, 1, K)

        refined = self.norm(embedding + self.drop(attn_out.squeeze(1)))
        return self.norm2(refined + self.ffn(refined))


class PGRModule(nn.Module):
    """
    E6 top-level module — Prototype-Guided Refinement.

    Composes GradePrototypeBank, PrototypeSimilarity, and
    PrototypeRefinementModule into a single differentiable unit.

    Parameters
    ----------
    embed_dim:    Shared embedding dimension.
    num_classes:  KL grade classes.
    num_heads:    Attention heads in the refinement cross-attention.
    dropout:      Dropout probability.
    sim_mode:     Similarity mode ('cosine' | 'dot').
    temperature:  Prototype similarity temperature.
    ema_momentum: EMA decay for shadow prototypes.

    Returns (from forward)
    ----------------------
    refined_emb: (B, embed_dim) — prototype-refined embedding
    sim_logits:  (B, num_classes) — grade similarity scores (→ aux loss)
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_classes: int = 5,
        num_heads: int = 4,
        dropout: float = 0.1,
        sim_mode: str = "cosine",
        temperature: float = 0.07,
        ema_momentum: float = 0.99,
    ) -> None:
        super().__init__()
        self.bank       = GradePrototypeBank(num_classes, embed_dim, ema_momentum)
        self.similarity = PrototypeSimilarity(sim_mode, temperature)
        self.refine     = PrototypeRefinementModule(embed_dim, num_classes,
                                                    num_heads, dropout)

    def update_prototypes_from_batch(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> None:
        """Delegate EMA prototype update to the bank."""
        self.bank.update_prototypes_from_batch(embeddings.detach(), labels)

    def forward(
        self, embedding: torch.Tensor, use_ema: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        embedding: (B, embed_dim)
        use_ema:   Use EMA shadow prototypes (True during inference).

        Returns
        -------
        refined_emb: (B, embed_dim)
        sim_logits:  (B, num_classes)
        """
        protos     = self.bank(use_ema)
        sim_logits = self.similarity(embedding, protos)
        refined    = self.refine(embedding, protos)
        return refined, sim_logits
