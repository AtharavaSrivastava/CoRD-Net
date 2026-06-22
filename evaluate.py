"""
evaluate.py
===========
Evaluation script for CoRD-Net.

Loads a saved checkpoint, runs inference over the validation or test
split, and reports Accuracy, Quadratic Weighted Kappa, and MAE.

Usage
-----
    python evaluate.py --checkpoint checkpoints/e8_best.pt \\
                       --exp e8 \\
                       --data-root /data/OAI \\
                       --split val

    python evaluate.py --checkpoint checkpoints/e5_best.pt \\
                       --exp e5 \\
                       --data-root /data/OAI \\
                       --split test \\
                       --metadata-csv /data/OAI/labels.csv
"""

from __future__ import annotations

import argparse
import logging

import torch

from config import get_config, EXPERIMENT_NAMES
from dataset import build_loaders, build_test_loader
from metrics import evaluate, quadratic_kappa, accuracy, mean_absolute_error
from models.drpnet import DRPNet
from utils import get_device, load_checkpoint, setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint",   required=True,
                   help="Path to .pt checkpoint file")
    p.add_argument("--exp",          required=True,
                   choices=list(EXPERIMENT_NAMES.keys()),
                   help="Experiment tag matching the checkpoint")
    p.add_argument("--data-root",    required=True,
                   help="Root directory of the OAI dataset")
    p.add_argument("--split",        default="val",
                   choices=["val", "test"],
                   help="Which data split to evaluate on (default: val)")
    p.add_argument("--metadata-csv", type=str, default=None)
    p.add_argument("--batch-size",   type=int, default=16)
    p.add_argument("--device",       type=str, default=None)
    p.add_argument("--num-workers",  type=int, default=4)
    p.add_argument("--log-dir",      type=str, default="logs")
    return p.parse_args()


@torch.no_grad()
def run_evaluation(
    model: DRPNet,
    loader,
    device: torch.device,
    num_classes: int,
) -> dict[str, float]:
    """
    Collect all logits and labels, then compute metrics.

    Returns
    -------
    Dict with keys: accuracy, kappa, mae
    """
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for batch in loader:
        crops, labels = batch
        global_crop  = crops[0].to(device)
        medial_crop  = crops[1].to(device) if len(crops) > 1 else None
        lateral_crop = crops[2].to(device) if len(crops) > 2 else None

        preds = model(global_crop, medial_crop, lateral_crop)
        all_logits.append(preds["logits"].cpu())
        all_labels.append(labels["kl"])

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return evaluate(logits, labels, num_classes)


def main() -> None:
    args   = parse_args()
    device = get_device(args.device)
    setup_logging(args.log_dir, f"eval_{args.exp}")

    cfg = get_config(
        args.exp,
        device       = str(device),
        batch_size   = args.batch_size,
        data_root    = args.data_root,
        metadata_csv = args.metadata_csv,
    )
    cfg.training.num_workers = args.num_workers

    # Build loader for the requested split
    if args.split == "test":
        loader = build_test_loader(cfg)
    else:
        _, loader = build_loaders(cfg)

    # Build model and load checkpoint
    model = DRPNet(cfg.model).to(device)
    load_checkpoint(args.checkpoint, model, device=device)

    logger.info("Evaluating %s on %s split …", args.exp.upper(), args.split)
    metrics = run_evaluation(model, loader, device, cfg.model.num_classes)

    logger.info("═" * 50)
    logger.info("  Results — %s (%s split)", args.exp.upper(), args.split)
    logger.info("  Accuracy : %.4f", metrics["accuracy"])
    logger.info("  Kappa    : %.4f", metrics["kappa"])
    logger.info("  MAE      : %.4f", metrics["mae"])
    logger.info("═" * 50)


if __name__ == "__main__":
    main()
