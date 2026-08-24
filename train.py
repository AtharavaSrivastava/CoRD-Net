"""
train.py
========
Main training entry point for CoRD-Net.

    python train.py --exp e1 --data-root /data/OAI
    python train.py --exp e8 --data-root /data/OAI --metadata-csv /data/OAI/labels.csv
    python train.py --exp e5 --data-root /data/OAI --pretrained --device cuda
    python train.py --exp e6 --data-root /data/OAI --resume checkpoints/e6_best.pt

After training completes, results are automatically written to
results/<experiment>/ (confusion matrices, classification reports,
metric JSON/CSV, training-history plots, predictions.csv, and the
cumulative ablation_summary.csv).
"""

from __future__ import annotations

import argparse
import logging

import json
from pathlib import Path
import numpy as np

from config import get_config, EXPERIMENT_NAMES
from dataset import build_loaders, build_test_loader
from metrics import compute_per_class_metrics
from models.drpnet import DRPNet
from trainer import Trainer
from utils import get_device, seed_everything, setup_logging

logger = logging.getLogger(__name__)


def save_class_metrics(y_true, y_pred, output_dir, split):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = compute_per_class_metrics(
        y_true,
        y_pred,
        num_classes=5,
    )

    with open(
        output_dir / f"per_class_metrics_{split}.json",
        "w",
    ) as f:
        json.dump(metrics, f, indent=2)

    np.savetxt(
        output_dir / f"confusion_matrix_{split}.csv",
        np.asarray(metrics["confusion_matrix"]),
        delimiter=",",
        fmt="%d",
    )

    return metrics


def print_per_class_summary(metrics, split):
    print(f"\n===== {split.upper()} PER-CLASS RESULTS =====")

    print(
        f"{'Class':<8}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'Support':>12}"
    )

    for cls in ["KL0", "KL1", "KL2", "KL3", "KL4"]:
        m = metrics["per_class"][cls]

        print(
            f"{cls:<8}"
            f"{m['precision']:>12.4f}"
            f"{m['recall']:>12.4f}"
            f"{m['f1']:>12.4f}"
            f"{m['support']:>12}"
        )

    print("\nKL BOUNDARY ERRORS:")

    for name, count in metrics["boundary_errors"].items():
        print(f"{name:<15}: {count}")

    print("\nCONFUSION MATRIX:")
    print(np.asarray(metrics["confusion_matrix"]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--exp",            required=True,
                   choices=list(EXPERIMENT_NAMES.keys()),
                   help="Ablation experiment (e1 … e8)")
    p.add_argument("--data-root",      type=str, default=None,
                   help="Root directory of the OAI dataset")
    p.add_argument("--metadata-csv",   type=str, default=None,
                   help="Path to OAI metadata CSV (Layout B)")
    p.add_argument("--train-csv",      type=str, default=None,
                   help="Mode 3: CSV for training split only")
    p.add_argument("--val-csv",        type=str, default=None,
                   help="Mode 3: CSV for validation split only")
    p.add_argument("--test-csv",       type=str, default=None,
                   help="Mode 3: CSV for test split only")
    p.add_argument("--epochs",         type=int, default=None)
    p.add_argument("--batch-size",     type=int, default=None)
    p.add_argument("--lr",             type=float, default=None)
    p.add_argument("--device",         type=str, default=None)
    p.add_argument("--pretrained",     action="store_true")
    p.add_argument("--resume",         type=str, default=None)
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--log-dir",        type=str, default="logs")
    p.add_argument("--checkpoint-dir", type=str, default=None)
    p.add_argument("--results-dir",    type=str, default="results",
                   help="Root directory for evaluation output (default: results)")
    p.add_argument("--num-workers",    type=int, default=None)
    p.add_argument("--amp",            action="store_true")
    p.add_argument(
        "--loss-type",
        type=str,
        default="ce",
        choices=["ce", "weighted_ce", "focal", "soft_qwk"],
        help="Primary-head loss"
    )
    p.add_argument(
        "--sampler",
        type=str,
        default="none",
        choices=["none", "weighted"],
        help="Training sampler: none or weighted"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    seed_everything(args.seed)
    setup_logging(args.log_dir, args.exp)

    cfg = get_config(
        args.exp,
        pretrained    = args.pretrained,
        device        = args.device,
        batch_size    = args.batch_size,
        epochs        = args.epochs,
        learning_rate = args.lr,
        data_root     = args.data_root,
        metadata_csv  = args.metadata_csv,
    )

    cfg.training.sampler = args.sampler

    if args.train_csv:
        cfg.training.train_csv = args.train_csv
    if args.val_csv:
        cfg.training.val_csv = args.val_csv
    if args.test_csv:
        cfg.training.test_csv = args.test_csv
    if args.checkpoint_dir:
        cfg.training.checkpoint_dir = args.checkpoint_dir
    if args.num_workers is not None:
        cfg.training.num_workers = args.num_workers
    if args.amp:
        cfg.training.amp = True

    logger.info("═" * 62)
    logger.info("  CoRD-Net — %s", cfg.description)
    logger.info("  Experiment  : %s", args.exp.upper())
    logger.info("  Device      : %s", cfg.training.device or "auto")
    logger.info("  Epochs      : %d", cfg.training.epochs)
    logger.info("  Batch size  : %d", cfg.training.batch_size)
    logger.info("  Data root   : %s", cfg.training.data_root or "not set")
    logger.info("  Pretrained  : %s", cfg.model.pretrained)
    logger.info("  Results dir : %s", args.results_dir)
    logger.info("═" * 62)

    from losses import MultiTaskLoss, build_primary_loss

    train_loader, val_loader = build_loaders(cfg)
    test_loader              = build_test_loader(cfg)

    model   = DRPNet(cfg.model)
    loss_fn = (
        MultiTaskLoss(cfg.training)
        if cfg.model.use_aux_heads
        else build_primary_loss(
            args.loss_type,
            train_loader.dataset.samples,
            cfg.model.num_classes,
            cfg.training.device or "cuda",
        )
    )

    trainer = Trainer(model, loss_fn, cfg)
    trainer.fit(
        train_loader = train_loader,
        val_loader   = val_loader,
        test_loader  = test_loader,
        resume       = args.resume,
        results_dir  = args.results_dir,
    )

    output_dir = Path(args.results_dir) / args.exp
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment_config = {
        "seed": args.seed,
        "loss_type": args.loss_type,
        "sampler": args.sampler,
        "num_classes": cfg.model.num_classes,
        "use_aux_heads": cfg.model.use_aux_heads,
    }

    with open(output_dir / "experiment_config.json", "w") as f:
        json.dump(experiment_config, f, indent=2)

    if val_loader is not None:
        val_logits, val_labels = trainer.collect_logits(val_loader)
        if len(val_labels) > 0:
            val_preds = val_logits.argmax(axis=1)
            val_class_metrics = save_class_metrics(val_labels, val_preds, output_dir, "val")
            print_per_class_summary(val_class_metrics, "val")

    if test_loader is not None:
        test_logits, test_labels = trainer.collect_logits(test_loader)
        if len(test_labels) > 0:
            test_preds = test_logits.argmax(axis=1)
            test_class_metrics = save_class_metrics(test_labels, test_preds, output_dir, "test")
            print_per_class_summary(test_class_metrics, "test")


if __name__ == "__main__":
    main()
