"""
losses.py
=========
Multi-task loss for CoRD-Net (E8).

MultiTaskLoss aggregates seven per-head losses with configurable weights
and an optional prototype alignment term:

  h1  CrossEntropy (label-smoothing)   primary KL grade
  h2  BCEWithLogits (CORAL ordinal)    ordinal cumulative thresholds
  h3  SupCon proxy (cosine sim)        metric embedding
  h4  CrossEntropy (ignore=-1)         medial JSN
  h5  CrossEntropy (ignore=-1)         lateral JSN
  h6  CrossEntropy mean (4 sub-heads)  osteophyte grading
  h7  MSE vs entropy                   uncertainty calibration
  proto  CrossEntropy                  prototype alignment (from PGR)

Loss functions are separated from model files per single-responsibility
principle.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import TrainingConfig


class MultiTaskLoss(nn.Module):
    """
    Weighted multi-task loss for E8 auxiliary heads.

    Parameters
    ----------
    cfg:
        TrainingConfig supplying loss_weights and active_heads.
        If None, default weights and all heads are used.
    """

    _DEFAULT_WEIGHTS: Dict[str, float] = {
        "h1": 1.0, "h2": 0.5, "h3": 0.3,
        "h4": 0.4, "h5": 0.4, "h6": 0.3,
        "h7": 0.2, "proto": 0.3,
    }

    _DEFAULT_ACTIVE: List[str] = ["h1", "h2", "h3", "h4", "h5", "h6", "h7"]

    def __init__(self, cfg: Optional[TrainingConfig] = None) -> None:
        super().__init__()
        weights = cfg.loss_weights if cfg else self._DEFAULT_WEIGHTS
        active  = cfg.active_heads if cfg else self._DEFAULT_ACTIVE

        self.weights = {k: weights.get(k, self._DEFAULT_WEIGHTS.get(k, 0.0))
                        for k in self._DEFAULT_WEIGHTS}
        self.active  = set(active)

        self.ce_kl  = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.ce_jsn = nn.CrossEntropyLoss(ignore_index=-1)
        self.bce    = nn.BCEWithLogitsLoss()
        self.mse    = nn.MSELoss()
    def _safe_ce(self, logits, targets):
        valid = targets != -1

        if not valid.any():
            return torch.zeros(
                (),
                device=logits.device,
                dtype=logits.dtype,
            )

        return self.ce_jsn(logits, targets)

    # ── Individual loss components ─────────────────────────────────────────

    def _coral_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """CORAL ordinal loss: converts integer grades to binary rank vectors."""
        B = targets.size(0)
        ordinal_targets = torch.zeros(B, 4, device=targets.device)
        for i in range(B):
            if targets[i] > 0:
                ordinal_targets[i, : targets[i].long()] = 1.0
        return self.bce(logits, ordinal_targets)

    def _supcon_proxy(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Supervised contrastive proxy loss via pairwise cosine similarity.

        Returns the mean dissimilarity between positive pairs, scaled to
        approximately match the range of cross-entropy losses.
        """
        sim  = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
        mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        mask.fill_diagonal_(0)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device)
        return (1.0 - (sim * mask).sum() / mask.sum()) * 0.5

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(
        self,
        preds: Dict[str, torch.Tensor | list],
        labels: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute weighted multi-task loss.

        Parameters
        ----------
        preds:
            Dict of head predictions keyed by 'h1' … 'h7'.
            'h6' maps to a list of 4 tensors (one per osteophyte sub-head).
        labels:
            Dict with keys: 'kl', 'jsn_med', 'jsn_lat', 'osteophyte'.
            'osteophyte' is (B, 4) with -1 for ignored positions.

        Returns
        -------
        loss_dict:
            Dict of individual losses plus 'total' (weighted sum).
        """
        device = labels["kl"].device
        total  = torch.tensor(0.0, device=device)
        ld: Dict[str, torch.Tensor] = {}

        if "h1" in self.active and "h1" in preds:
            ld["kl"]   = self.ce_kl(preds["h1"], labels["kl"])
            total     += self.weights["h1"] * ld["kl"]

        if "h2" in self.active and "h2" in preds:
            ld["coral"] = self._coral_loss(preds["h2"], labels["kl"])
            total      += self.weights["h2"] * ld["coral"]

        if "h3" in self.active and "h3" in preds:
            ld["supcon"] = self._supcon_proxy(preds["h3"], labels["kl"])
            total       += self.weights["h3"] * ld["supcon"]

        if "h4" in self.active and "h4" in preds:
            ld["jsn_med"] = self._safe_ce(
                preds["h4"],
                labels["jsn_med"]
            )
            total += self.weights["h4"] * ld["jsn_med"]

        if "h5" in self.active and "h5" in preds:
            ld["jsn_lat"] = self._safe_ce(
                preds["h5"],
                labels["jsn_lat"]
            )
            total += self.weights["h5"] * ld["jsn_lat"]

        if "h6" in self.active and "h6" in preds:
            osteo_losses = [
                self._safe_ce(h, labels["osteophyte"][:, i])
                for i, h in enumerate(preds["h6"])
            ]

            ld["osteo"] = torch.stack(osteo_losses).mean()
            total += self.weights["h6"] * ld["osteo"]
        
        if "h7" in self.active and "h7" in preds and "h1" in preds:
            probs   = F.softmax(preds["h1"], dim=-1)
            entropy = -(probs * torch.log(probs + 1e-6)).sum(dim=-1, keepdim=True)
            ld["uncert"] = self.mse(preds["h7"], entropy.detach())
            total       += self.weights["h7"] * ld["uncert"]

        ld["total"] = total
        return ld
