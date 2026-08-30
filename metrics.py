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

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
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

def compute_fgbf_metrics(
    fgbf_logits: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
) -> Dict[str, float]:
    """
    Compute FGBF auxiliary diagnostic metrics over low-grade samples (KL0, KL1, KL2).

    Parameters
    ----------
    fgbf_logits: (N, 3) raw logits for KL0, KL1, KL2.
    labels: (N,) ground-truth KL grades.

    Returns
    -------
    Dict containing:
      fgbf_low_grade_accuracy
      fgbf_kl0_precision, fgbf_kl0_recall, fgbf_kl0_f1
      fgbf_kl1_precision, fgbf_kl1_recall, fgbf_kl1_f1
      fgbf_kl2_precision, fgbf_kl2_recall, fgbf_kl2_f1
    """
    logits_np = _to_numpy(fgbf_logits)
    labels_np = _to_numpy(labels).astype(int)

    mask = (labels_np <= 2)
    if not mask.any():
        return {
            "fgbf_low_grade_accuracy": 0.0,
            "fgbf_kl0_precision": 0.0, "fgbf_kl0_recall": 0.0, "fgbf_kl0_f1": 0.0,
            "fgbf_kl1_precision": 0.0, "fgbf_kl1_recall": 0.0, "fgbf_kl1_f1": 0.0,
            "fgbf_kl2_precision": 0.0, "fgbf_kl2_recall": 0.0, "fgbf_kl2_f1": 0.0,
        }

    sub_logits = logits_np[mask]
    sub_labels = labels_np[mask]
    sub_preds = sub_logits.argmax(axis=1)

    kw = dict(labels=[0, 1, 2], zero_division=0)
    prec = precision_score(sub_labels, sub_preds, average=None, **kw)
    rec = recall_score(sub_labels, sub_preds, average=None, **kw)
    f1 = f1_score(sub_labels, sub_preds, average=None, **kw)
    cm = confusion_matrix(sub_labels, sub_preds, labels=[0, 1, 2])

    return {
        "fgbf_low_grade_accuracy": float(accuracy_score(sub_labels, sub_preds)),
        "fgbf_kl0_precision": float(prec[0]),
        "fgbf_kl0_recall": float(rec[0]),
        "fgbf_kl0_f1": float(f1[0]),
        "fgbf_kl1_precision": float(prec[1]),
        "fgbf_kl1_recall": float(rec[1]),
        "fgbf_kl1_f1": float(f1[1]),
        "fgbf_kl2_precision": float(prec[2]),
        "fgbf_kl2_recall": float(rec[2]),
        "fgbf_kl2_f1": float(f1[2]),
        "fgbf_boundary_KL1_to_KL0": int(cm[1, 0]),
        "fgbf_boundary_KL1_to_KL2": int(cm[1, 2]),
    }


def compute_all_metrics(
    logits: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    num_classes: int = 5,
    fgbf_logits: Optional[torch.Tensor | np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute the full metric suite from raw logits and integer labels.

    Parameters
    ----------
    logits: (N, num_classes) raw model output scores.
    labels: (N,) integer ground-truth KL grades.
    fgbf_logits: Optional (N, 3) FGBF auxiliary logits.

    Returns
    -------
    Dict[str, float] containing all metrics.
    """
    preds, _ = get_predictions(logits)
    l        = _to_numpy(labels).astype(int)
    kl_labels = list(range(num_classes))
    kw        = dict(labels=kl_labels, zero_division=0)

    rec_per_class  = recall_score(l, preds, average=None, **kw)
    f1_per_class   = f1_score(l, preds, average=None, **kw)

    cm = confusion_matrix(l, preds, labels=kl_labels)

    metrics: Dict[str, float] = {
        "accuracy":           float(accuracy_score(l, preds)),
        "macro_precision":    float(precision_score(l, preds, average="macro",  **kw)),
        "macro_recall":       float(recall_score(   l, preds, average="macro",  **kw)),
        "macro_f1":           float(f1_score(       l, preds, average="macro",  **kw)),
        "weighted_precision": float(precision_score(l, preds, average="weighted", **kw)),
        "weighted_recall":    float(recall_score(   l, preds, average="weighted", **kw)),
        "weighted_f1":        float(f1_score(       l, preds, average="weighted", **kw)),
        "mae":                float(sklearn_mae(l, preds)),
        "qwk":                quadratic_kappa(preds, l, num_classes),
        # Class-specific metrics for low-grade boundary error analysis
        "kl0_recall":         float(rec_per_class[0]) if num_classes > 0 else 0.0,
        "kl1_recall":         float(rec_per_class[1]) if num_classes > 1 else 0.0,
        "kl1_f1":             float(f1_per_class[1])  if num_classes > 1 else 0.0,
        "kl2_recall":         float(rec_per_class[2]) if num_classes > 2 else 0.0,
        # Key boundary error counts from 5x5 confusion matrix
        "boundary_KL1_to_KL0": int(cm[1, 0]) if num_classes > 1 else 0,
        "boundary_KL1_to_KL2": int(cm[1, 2]) if num_classes > 2 else 0,
        "boundary_KL0_to_KL1": int(cm[0, 1]) if num_classes > 1 else 0,
        "boundary_KL2_to_KL1": int(cm[2, 1]) if num_classes > 2 else 0,
    }

    if fgbf_logits is not None:
        fgbf_m = compute_fgbf_metrics(fgbf_logits, labels)
        metrics.update(fgbf_m)

    return metrics


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


def compute_per_class_metrics(y_true, y_pred, num_classes=5):
    labels = list(range(num_classes))

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=[f"KL{i}" for i in labels],
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    per_class = {}

    for i in labels:
        name = f"KL{i}"

        per_class[name] = {
            "precision": float(report[name]["precision"]),
            "recall": float(report[name]["recall"]),
            "f1": float(report[name]["f1-score"]),
            "support": int(report[name]["support"]),
        }

    boundary_errors = {
        "KL0_to_KL1": int(cm[0, 1]),
        "KL1_to_KL0": int(cm[1, 0]),
        "KL1_to_KL2": int(cm[1, 2]),
        "KL2_to_KL1": int(cm[2, 1]),
        "KL2_to_KL3": int(cm[2, 3]),
        "KL3_to_KL2": int(cm[3, 2]),
    }

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "boundary_errors": boundary_errors,
    }
