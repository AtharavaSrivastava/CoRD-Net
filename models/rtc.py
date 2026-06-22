"""
models/rtc.py
=============
E7 — Relational Token Coupling (RTC).

Projects medial, lateral, and (optionally) global branch features into
a shared token space, adds learnable role embeddings, then applies
multi-head self-attention.  The attended medial and lateral tokens are
concatenated and projected into the shared embedding dimension.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RelationalTokenCoupling(nn.Module):
    """
    E7 — cross-compartment relational attention.

    Three feature vectors (medial, lateral, optional global) are
    projected into a shared token space and augmented with learnable
    role embeddings before self-attention couples them.

    The final coupled embedding is derived from the attended medial and
    lateral tokens only, keeping the output dimension independent of
    whether the global context token is used.

    Parameters
    ----------
    in_channels:
        Input feature dimension (backbone_feature_dim, 768).
    token_dim:
        Projected token / output dimension (embedding_dim, 256).
    num_heads:
        Self-attention heads.
    dropout:
        Dropout in attention.
    use_global_context:
        Include the global crop as a third context token.

    Attributes
    ----------
    last_attn_weights:
        Stored attention weight tensor from the last forward pass,
        shape (B, N_tokens, N_tokens).
    """

    def __init__(
        self,
        in_channels: int = 768,
        token_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_global_context: bool = True,
    ) -> None:
        super().__init__()
        self.use_global_context = use_global_context

        self.proj             = nn.Linear(in_channels, token_dim)
        self.medial_role_emb  = nn.Parameter(torch.randn(1, 1, token_dim))
        self.lateral_role_emb = nn.Parameter(torch.randn(1, 1, token_dim))
        if use_global_context:
            self.global_role_emb = nn.Parameter(torch.randn(1, 1, token_dim))

        self.mha      = nn.MultiheadAttention(
            token_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.out_proj = nn.Linear(token_dim * 2, token_dim)
        self.norm     = nn.LayerNorm(token_dim)
        self.dropout  = nn.Dropout(dropout)

        self.last_attn_weights: torch.Tensor | None = None

    def forward(
        self,
        medial_feat: torch.Tensor,
        lateral_feat: torch.Tensor,
        global_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        medial_feat:  (B, in_channels)
        lateral_feat: (B, in_channels)
        global_feat:  (B, in_channels) — optional global context

        Returns
        -------
        coupled: (B, token_dim)
        """
        m = self.proj(medial_feat).unsqueeze(1)  + self.medial_role_emb
        l = self.proj(lateral_feat).unsqueeze(1) + self.lateral_role_emb

        if self.use_global_context and global_feat is not None:
            g      = self.proj(global_feat).unsqueeze(1) + self.global_role_emb
            tokens = torch.cat([m, l, g], dim=1)
        else:
            tokens = torch.cat([m, l], dim=1)

        attn_out, attn_w         = self.mha(tokens, tokens, tokens)
        self.last_attn_weights   = attn_w
        attn_out                 = self.norm(attn_out + tokens)

        coupled = self.out_proj(
            torch.cat([attn_out[:, 0, :], attn_out[:, 1, :]], dim=-1)
        )
        return self.dropout(coupled)
