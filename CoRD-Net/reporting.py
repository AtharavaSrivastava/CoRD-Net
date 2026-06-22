"""
reporting.py
============
Publication-ready evaluation, plotting, and result-saving pipeline for CoRD-Net.

This module is the single place for all post-training output.
It is called by trainer.py at the end of fit() and by evaluate.py.
Nothing here touches the model, optimizer, or training logic.

Public API
----------
ResultsWriter(experiment, results_dir)
    Context object that owns the output directory and all save methods.

generate_all_reports(writer, history, train_logits, train_labels,
                     val_logits, val_labels, test_logits, test_labels,
                     num_classes, params)
    One call that produces every file listed in the specification.

Output layout
-------------
results/<experiment>/
    metrics.json
    metrics.csv
    predictions.csv                (val + test rows)
    classification_report_val.txt
    classification_report_test.txt
    val_confusion_matrix.png
    val_confusion_matrix.csv
    test_confusion_matrix.png
    test_confusion_matrix.csv
    plots/
        loss_curve.png
        accuracy_curve.png
        f1_curve.png
        qwk_curve.png
        mae_curve.png
        learning_rate_curve.png

results/ablation_summary.csv       (one row appended per completed experiment)
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe on servers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from metrics import compute_all_metrics, get_predictions, _to_numpy

logger = logging.getLogger(__name__)

# KL grade class labels used on every axis / report
KL_LABELS = ["KL0", "KL1", "KL2", "KL3", "KL4"]

# Plot style constants
_DPI        = 300
_FIGSIZE    = (7, 5)
_TRAIN_CLR  = "#2196F3"   # blue
_VAL_CLR    = "#F44336"   # red
_SINGLE_CLR = "#4CAF50"   # green


# ══════════════════════════════════════════════════════════════════════════════
#  ResultsWriter — owns the output directory tree
# ══════════════════════════════════════════════════════════════════════════════

class ResultsWriter:
    """
    Manages the results/<experiment>/ directory tree.

    Parameters
    ----------
    experiment:  Experiment tag, e.g. 'e8'.
    results_dir: Root results directory (default 'results').
    """

    def __init__(self, experiment: str, results_dir: str = "results") -> None:
        self.experiment  = experiment
        self.root        = Path(results_dir) / experiment
        self.plots_dir   = self.root / "plots"
        self.root.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Results directory: %s", self.root)

    def path(self, filename: str) -> Path:
        return self.root / filename

    def plot_path(self, filename: str) -> Path:
        return self.plots_dir / filename


# ══════════════════════════════════════════════════════════════════════════════
#  Confusion matrix
# ══════════════════════════════════════════════════════════════════════════════

def _save_confusion_matrix(
    labels: np.ndarray,
    preds:  np.ndarray,
    num_classes: int,
    split: str,
    writer: ResultsWriter,
) -> None:
    """
    Save confusion matrix as both PNG (300 dpi) and CSV.

    Every cell is annotated with its count.  Rows = true, columns = predicted.
    """
    kl = KL_LABELS[:num_classes]
    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))

    # ── PNG ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(kl, fontsize=10)
    ax.set_yticklabels(kl, fontsize=10)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(f"Confusion Matrix — {split.capitalize()} Split", fontsize=12)

    thresh = cm.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    png_path = writer.path(f"{split}_confusion_matrix.png")
    fig.savefig(png_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", png_path)

    # ── CSV ────────────────────────────────────────────────────────────────
    csv_path = writer.path(f"{split}_confusion_matrix.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["true\\pred"] + kl)
        for i, row in enumerate(cm):
            w.writerow([kl[i]] + row.tolist())
    logger.info("Saved %s", csv_path)


# ══════════════════════════════════════════════════════════════════════════════
#  Classification report
# ══════════════════════════════════════════════════════════════════════════════

def _save_classification_report(
    labels:      np.ndarray,
    preds:       np.ndarray,
    num_classes: int,
    split:       str,
    writer:      ResultsWriter,
) -> None:
    kl = KL_LABELS[:num_classes]
    report = classification_report(
        labels, preds,
        labels      = list(range(num_classes)),
        target_names = kl,
        zero_division = 0,
    )
    path = writer.path(f"classification_report_{split}.txt")
    path.write_text(report)
    logger.info("Saved %s", path)


# ══════════════════════════════════════════════════════════════════════════════
#  Training-history plots
# ══════════════════════════════════════════════════════════════════════════════

def _plot_one(
    ax,
    epochs: List[int],
    series: Dict[str, List[float]],
    title:  str,
    ylabel: str,
    colors: Optional[List[str]] = None,
) -> None:
    """Helper: plot one or more series on a single axes."""
    default_colors = [_TRAIN_CLR, _VAL_CLR, _SINGLE_CLR, "#FF9800", "#9C27B0"]
    for i, (label, values) in enumerate(series.items()):
        if not values:
            continue
        clr = (colors[i] if colors and i < len(colors) else default_colors[i])
        ax.plot(epochs, values, label=label, color=clr, linewidth=1.8)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def _save_plots(
    history: Dict[str, List],
    writer:  ResultsWriter,
) -> None:
    """
    Generate and save all 6 training-history plots at 300 dpi.

    history keys expected:
        epoch, train_loss, val_loss, train_accuracy, val_accuracy,
        val_macro_f1, val_qwk, val_mae, learning_rate
    """
    epochs = history.get("epoch", [])
    if not epochs:
        logger.warning("Training history is empty — skipping plots.")
        return

    plots = [
        # (filename, title, ylabel, {label: values}, colors)
        (
            "loss_curve.png",
            "Training & Validation Loss",
            "Loss",
            {"Train Loss": history.get("train_loss", []),
             "Val Loss":   history.get("val_loss",   [])},
            [_TRAIN_CLR, _VAL_CLR],
        ),
        (
            "accuracy_curve.png",
            "Training & Validation Accuracy",
            "Accuracy",
            {"Train Accuracy": history.get("train_accuracy", []),
             "Val Accuracy":   history.get("val_accuracy",   [])},
            [_TRAIN_CLR, _VAL_CLR],
        ),
        (
            "f1_curve.png",
            "Validation Macro F1",
            "Macro F1",
            {"Val Macro F1": history.get("val_macro_f1", [])},
            [_SINGLE_CLR],
        ),
        (
            "qwk_curve.png",
            "Validation Quadratic Weighted Kappa",
            "QWK",
            {"Val QWK": history.get("val_qwk", [])},
            [_SINGLE_CLR],
        ),
        (
            "mae_curve.png",
            "Validation Mean Absolute Error",
            "MAE",
            {"Val MAE": history.get("val_mae", [])},
            ["#FF9800"],
        ),
        (
            "learning_rate_curve.png",
            "Learning Rate Schedule",
            "Learning Rate",
            {"LR": history.get("learning_rate", [])},
            ["#9C27B0"],
        ),
    ]

    for fname, title, ylabel, series, colors in plots:
        fig, ax = plt.subplots(figsize=_FIGSIZE)
        _plot_one(ax, epochs, series, title, ylabel, colors)
        plt.tight_layout()
        path = writer.plot_path(fname)
        fig.savefig(path, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", path)


# ══════════════════════════════════════════════════════════════════════════════
#  Predictions CSV
# ══════════════════════════════════════════════════════════════════════════════

def _save_predictions_csv(
    val_labels:   np.ndarray,
    val_preds:    np.ndarray,
    val_probs:    np.ndarray,
    test_labels:  Optional[np.ndarray],
    test_preds:   Optional[np.ndarray],
    test_probs:   Optional[np.ndarray],
    writer:       ResultsWriter,
) -> None:
    path = writer.path("predictions.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "index", "ground_truth", "prediction",
                    "correct", "confidence"])
        for i, (gt, pred, prob) in enumerate(
                zip(val_labels, val_preds, val_probs)):
            w.writerow(["val", i, int(gt), int(pred),
                        int(gt) == int(pred), f"{prob:.4f}"])
        if test_labels is not None:
            for i, (gt, pred, prob) in enumerate(
                    zip(test_labels, test_preds, test_probs)):
                w.writerow(["test", i, int(gt), int(pred),
                            int(gt) == int(pred), f"{prob:.4f}"])
    logger.info("Saved %s", path)


# ══════════════════════════════════════════════════════════════════════════════
#  metrics.json + metrics.csv
# ══════════════════════════════════════════════════════════════════════════════

def _save_metrics(
    metrics_dict: Dict[str, Dict[str, float]],
    writer: ResultsWriter,
) -> None:
    """
    Save metrics.json and metrics.csv.

    metrics_dict has top-level keys 'train', 'val', 'test'.
    """
    # JSON
    json_path = writer.path("metrics.json")
    with open(json_path, "w") as fh:
        json.dump(metrics_dict, fh, indent=2)
    logger.info("Saved %s", json_path)

    # CSV — flat rows: split, metric, value
    csv_path = writer.path("metrics.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "metric", "value"])
        for split, mdict in metrics_dict.items():
            for metric, value in mdict.items():
                w.writerow([split, metric, f"{value:.6f}"])
    logger.info("Saved %s", csv_path)


# ══════════════════════════════════════════════════════════════════════════════
#  Ablation summary
# ══════════════════════════════════════════════════════════════════════════════

_ABLATION_COLUMNS = [
    "experiment", "parameters",
    "train_accuracy",
    "val_accuracy",   "val_macro_f1",   "val_qwk",   "val_mae",
    "test_accuracy",  "test_macro_f1",  "test_qwk",  "test_mae",
]

def update_ablation_summary(
    experiment:   str,
    parameters:   int,
    train_metrics: Dict[str, float],
    val_metrics:   Dict[str, float],
    test_metrics:  Dict[str, float],
    results_dir:   str = "results",
) -> None:
    """
    Append one row to results/ablation_summary.csv.

    Creates the file (with header) if it doesn't exist yet.
    Updates an existing row for the same experiment if already present.
    """
    summary_path = Path(results_dir) / "ablation_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "experiment":    experiment,
        "parameters":    parameters,
        "train_accuracy": train_metrics.get("accuracy", float("nan")),
        "val_accuracy":   val_metrics.get("accuracy",  float("nan")),
        "val_macro_f1":   val_metrics.get("macro_f1",  float("nan")),
        "val_qwk":        val_metrics.get("qwk",       float("nan")),
        "val_mae":        val_metrics.get("mae",        float("nan")),
        "test_accuracy":  test_metrics.get("accuracy", float("nan")),
        "test_macro_f1":  test_metrics.get("macro_f1", float("nan")),
        "test_qwk":       test_metrics.get("qwk",      float("nan")),
        "test_mae":       test_metrics.get("mae",       float("nan")),
    }

    # Read existing rows
    existing: List[Dict] = []
    if summary_path.exists():
        with open(summary_path, newline="") as fh:
            existing = list(csv.DictReader(fh))

    # Replace row if experiment already present, else append
    updated = False
    for i, r in enumerate(existing):
        if r.get("experiment") == experiment:
            existing[i] = row
            updated = True
            break
    if not updated:
        existing.append(row)

    with open(summary_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_ABLATION_COLUMNS)
        w.writeheader()
        w.writerows(existing)

    logger.info("Ablation summary updated → %s", summary_path)


# ══════════════════════════════════════════════════════════════════════════════
#  Final results console print
# ══════════════════════════════════════════════════════════════════════════════

def print_final_results(
    train_metrics: Dict[str, float],
    val_metrics:   Dict[str, float],
    test_metrics:  Dict[str, float],
) -> None:
    """Pretty-print the final results table to the logger."""

    def _block(split: str, m: Dict[str, float]) -> None:
        logger.info("%s", split.upper())
        logger.info("  Accuracy           : %.4f", m.get("accuracy",           float("nan")))
        logger.info("  Macro Precision    : %.4f", m.get("macro_precision",    float("nan")))
        logger.info("  Macro Recall       : %.4f", m.get("macro_recall",       float("nan")))
        logger.info("  Macro F1           : %.4f", m.get("macro_f1",           float("nan")))
        logger.info("  Weighted Precision : %.4f", m.get("weighted_precision", float("nan")))
        logger.info("  Weighted Recall    : %.4f", m.get("weighted_recall",    float("nan")))
        logger.info("  Weighted F1        : %.4f", m.get("weighted_f1",        float("nan")))
        if "mae" in m:
            logger.info("  MAE                : %.4f", m["mae"])
        if "qwk" in m:
            logger.info("  QWK                : %.4f", m["qwk"])

    sep = "=" * 52
    logger.info(sep)
    logger.info("  FINAL RESULTS")
    logger.info(sep)
    _block("train", train_metrics)
    logger.info("  " + "-" * 48)
    _block("validation", val_metrics)
    logger.info("  " + "-" * 48)
    _block("test", test_metrics)
    logger.info(sep)


# ══════════════════════════════════════════════════════════════════════════════
#  Master entry point
# ══════════════════════════════════════════════════════════════════════════════

def generate_all_reports(
    writer:       ResultsWriter,
    history:      Dict[str, List],
    train_logits: Optional[np.ndarray],
    train_labels: Optional[np.ndarray],
    val_logits:   np.ndarray,
    val_labels:   np.ndarray,
    test_logits:  Optional[np.ndarray],
    test_labels:  Optional[np.ndarray],
    num_classes:  int,
    parameters:   int,
    results_dir:  str = "results",
) -> Dict[str, Dict[str, float]]:
    """
    Generate every output file for one experiment.

    Parameters
    ----------
    writer:        ResultsWriter for this experiment.
    history:       Dict of per-epoch lists from Trainer.history.
    train_logits:  (N, K) logits from final training-set pass, or None.
    train_labels:  (N,)   integer labels for training set, or None.
    val_logits:    (N, K) logits from validation set.
    val_labels:    (N,)   integer labels for validation set.
    test_logits:   (N, K) logits from test set, or None.
    test_labels:   (N,)   integer labels for test set, or None.
    num_classes:   Number of KL grades (5).
    parameters:    Total trainable parameter count.
    results_dir:   Root results directory for ablation summary.

    Returns
    -------
    Dict {"train": {...}, "val": {...}, "test": {...}} of metric dicts.
    """
    logger.info("Generating reports for experiment '%s' …",
                writer.experiment)

    # ── Compute all metrics ───────────────────────────────────────────────
    val_preds,   val_probs   = get_predictions(val_logits)
    val_labels_np            = _to_numpy(val_labels).astype(int)

    val_metrics   = compute_all_metrics(val_logits,   val_labels, num_classes)

    train_metrics: Dict[str, float] = {}
    if train_logits is not None and train_labels is not None:
        train_metrics = compute_all_metrics(train_logits, train_labels, num_classes)

    test_metrics:  Dict[str, float] = {}
    test_preds  = test_probs = test_labels_np = None
    if test_logits is not None and test_labels is not None:
        test_metrics      = compute_all_metrics(test_logits, test_labels, num_classes)
        test_preds, test_probs = get_predictions(test_logits)
        test_labels_np         = _to_numpy(test_labels).astype(int)

    metrics_all = {
        "train": train_metrics,
        "val":   val_metrics,
        "test":  test_metrics,
    }

    # ── Confusion matrices ────────────────────────────────────────────────
    _save_confusion_matrix(
        val_labels_np, val_preds, num_classes, "val", writer
    )
    if test_labels_np is not None:
        _save_confusion_matrix(
            test_labels_np, test_preds, num_classes, "test", writer
        )

    # ── Classification reports ────────────────────────────────────────────
    _save_classification_report(
        val_labels_np, val_preds, num_classes, "val", writer
    )
    if test_labels_np is not None:
        _save_classification_report(
            test_labels_np, test_preds, num_classes, "test", writer
        )

    # ── Predictions CSV ───────────────────────────────────────────────────
    _save_predictions_csv(
        val_labels_np, val_preds, val_probs,
        test_labels_np, test_preds, test_probs,
        writer,
    )

    # ── metrics.json / metrics.csv ────────────────────────────────────────
    _save_metrics(metrics_all, writer)

    # ── Training-history plots ────────────────────────────────────────────
    _save_plots(history, writer)

    # ── Final results console print ───────────────────────────────────────
    print_final_results(train_metrics, val_metrics, test_metrics)

    # ── Ablation summary ──────────────────────────────────────────────────
    update_ablation_summary(
        experiment    = writer.experiment,
        parameters    = parameters,
        train_metrics = train_metrics,
        val_metrics   = val_metrics,
        test_metrics  = test_metrics,
        results_dir   = results_dir,
    )

    return metrics_all
