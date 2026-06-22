"""
metrics.py
==========
Full evaluation metric suite for CoRD-Net KL grading.

All public functions accept numpy arrays or torch tensors.
sklearn is used for precision / recall / F1.
The quadratic_kappa implementation is kept pure-torch for speed during
the training loop; sklearn's cohen_kappa_score is used for final reports.

Public API
----------
compute_all_metrics(logits, labels, num_classes) -> Dict[str, float]
    Full 9-metric suite used in evaluate.py and reporting.py.

evaluate(logits, labels, num_classes) -> Dict[str, float]
    Lightweight 3-metric version used inside the training loop
    (accuracy, kappa, mae) — kept for backward compatibility with trainer.py.

get_predictions(logits) -> Tuple[np.ndarray, np.ndarray]
    Return (predicted_classes, max_probabilities) from raw logits.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    mean_absolute_error as sklearn_mae,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def get_predictions(
    logits: torch.Tensor | np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert raw logits to predicted class indices and confidence scores.

    Parameters
    ----------
    logits: (N, K) raw scores or log-probabilities.

    Returns
    -------
    preds: (N,) integer class predictions.
    probs: (N,) probability of the predicted class (softmax).
    """
    logits_np = _to_numpy(logits).astype(np.float32)
    # softmax
    e = np.exp(logits_np - logits_np.max(axis=1, keepdims=True))
    probs_all = e / e.sum(axis=1, keepdims=True)
    preds = probs_all.argmax(axis=1)
    probs = probs_all[np.arange(len(preds)), preds]
    return preds, probs


# ──────────────────────────────────────────────────────────────────────────────
# Individual metrics (kept for backward compatibility)
# ──────────────────────────────────────────────────────────────────────────────

def accuracy(
    preds: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
) -> float:
    """Top-1 accuracy."""
    return float(accuracy_score(_to_numpy(labels), _to_numpy(preds)))


def quadratic_kappa(
    preds: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    num_classes: int = 5,
) -> float:
    """
    Quadratic-weighted Cohen's kappa via sklearn.

    sklearn's implementation handles edge cases (e.g. single-class
    batches) more robustly than a manual implementation.
    """
    p = _to_numpy(preds).astype(int)
    l = _to_numpy(labels).astype(int)
    try:
        return float(cohen_kappa_score(l, p, weights="quadratic",
                                       labels=list(range(num_classes))))
    except Exception:
        return 0.0


def mean_absolute_error(
    preds: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
) -> float:
    """Mean absolute error between predicted and true KL grades."""
    return float(sklearn_mae(_to_numpy(labels), _to_numpy(preds)))


# ──────────────────────────────────────────────────────────────────────────────
# Full metric suite
# ──────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(
    logits: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    num_classes: int = 5,
) -> Dict[str, float]:
    """
    Compute the full 9-metric suite from raw logits and integer labels.

    Metrics
    -------
    accuracy
    macro_precision
    macro_recall
    macro_f1
    weighted_precision
    weighted_recall
    weighted_f1
    mae
    qwk

    Parameters
    ----------
    logits: (N, num_classes) raw model output scores.
    labels: (N,) integer ground-truth KL grades.

    Returns
    -------
    Dict[str, float] — all keys listed above.
    """
    preds, _ = get_predictions(logits)
    l        = _to_numpy(labels).astype(int)
    kl_labels = list(range(num_classes))
    kw        = dict(labels=kl_labels, zero_division=0)

    return {
        "accuracy":           float(accuracy_score(l, preds)),
        "macro_precision":    float(precision_score(l, preds, average="macro",  **kw)),
        "macro_recall":       float(recall_score(   l, preds, average="macro",  **kw)),
        "macro_f1":           float(f1_score(       l, preds, average="macro",  **kw)),
        "weighted_precision": float(precision_score(l, preds, average="weighted", **kw)),
        "weighted_recall":    float(recall_score(   l, preds, average="weighted", **kw)),
        "weighted_f1":        float(f1_score(       l, preds, average="weighted", **kw)),
        "mae":                float(sklearn_mae(l, preds)),
        "qwk":                quadratic_kappa(preds, l, num_classes),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Lightweight version used inside the training loop (backward compat)
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(
    logits: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    num_classes: int = 5,
) -> Dict[str, float]:
    """
    Compute the three metrics used during training-loop validation.

    Returns accuracy, kappa, mae — exactly as before, so trainer.py
    needs no changes.
    """
    preds, _ = get_predictions(logits)
    l        = _to_numpy(labels).astype(int)
    return {
        "accuracy": float(accuracy_score(l, preds)),
        "kappa":    quadratic_kappa(preds, l, num_classes),
        "mae":      float(sklearn_mae(l, preds)),
    }
