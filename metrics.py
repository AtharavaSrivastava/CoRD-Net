"""
metrics.py
==========
Evaluation metrics for CoRD-Net KL grading.

All metrics operate on integer tensors (predictions and labels).
"""

from __future__ import annotations

from typing import Dict

import torch


def accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """Top-1 accuracy."""
    return (preds == labels).float().mean().item()


def quadratic_kappa(preds: torch.Tensor, labels: torch.Tensor,
                    num_classes: int = 5) -> float:
    """
    Quadratic-weighted Cohen's kappa — the primary metric for KL grading.

    Parameters
    ----------
    preds:  (B,) integer predictions.
    labels: (B,) integer ground-truth labels.
    num_classes: Number of ordinal classes.

    Returns
    -------
    kappa: float in [-1, 1], higher is better.
    """
    preds  = preds.cpu()
    labels = labels.cpu()
    K = num_classes

    conf = torch.zeros(K, K, dtype=torch.float)
    for p, l in zip(preds.tolist(), labels.tolist()):
        conf[int(l), int(p)] += 1

    w = torch.zeros(K, K)
    for i in range(K):
        for j in range(K):
            w[i, j] = ((i - j) ** 2) / ((K - 1) ** 2)

    hist_p = conf.sum(dim=0)
    hist_l = conf.sum(dim=1)
    E      = torch.outer(hist_l, hist_p) / conf.sum()

    num = (w * conf).sum()
    den = (w * E).sum()
    return (1.0 - num / den).item() if den != 0 else 1.0


def mean_absolute_error(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """Mean absolute error between predicted and true KL grades."""
    return (preds.float() - labels.float()).abs().mean().item()


def evaluate(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 5,
) -> Dict[str, float]:
    """
    Compute all grading metrics from raw logits.

    Parameters
    ----------
    logits: (B, num_classes) raw prediction scores.
    labels: (B,) integer ground truth.

    Returns
    -------
    Dict with keys: 'accuracy', 'kappa', 'mae'.
    """
    preds = logits.argmax(dim=1)
    return {
        "accuracy": accuracy(preds, labels),
        "kappa":    quadratic_kappa(preds, labels, num_classes),
        "mae":      mean_absolute_error(preds, labels),
    }
