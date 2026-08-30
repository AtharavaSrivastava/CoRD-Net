"""
evaluate.py
===========
Standalone evaluation script for CoRD-Net.

Loads a checkpoint, runs inference over the validation and/or test split,
computes the full 9-metric suite, and writes all reports to
results/<experiment>/.

When the model has FGBF enabled (cfg.model.use_fgbf == True) the auxiliary
3-class head logits are also collected and routed to generate_all_reports,
which produces the 3x3 FGBF confusion matrix, per-class FGBF precision /
recall / F1, and boundary-error counts.  The primary 5-class predictions and
all existing metrics are never altered.

Usage
-----
    # Evaluate both val and test, write full reports
    python evaluate.py --checkpoint checkpoints/e8_best.pt \\
                       --exp e8 \\
                       --data-root /data/OAI

    # Evaluate test split only
    python evaluate.py --checkpoint checkpoints/e5_best.pt \\
                       --exp e5 \\
                       --data-root /data/OAI \\
                       --split test \\
                       --metadata-csv /data/OAI/labels.csv
"""

from __future__ import annotations

import argparse
import logging
from typing import Optional, Tuple

import numpy as np
import torch

from config import get_config, EXPERIMENT_NAMES
from dataset import build_loaders, build_test_loader
from metrics import compute_all_metrics, get_predictions, _to_numpy
from models.drpnet import DRPNet
from reporting import ResultsWriter, generate_all_reports
from utils import get_device, load_checkpoint, setup_logging, count_parameters

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint",    required=True,
                   help="Path to .pt checkpoint file")
    p.add_argument("--exp",           required=True,
                   choices=list(EXPERIMENT_NAMES.keys()),
                   help="Experiment tag matching the checkpoint")
    p.add_argument("--data-root",     required=True)
    p.add_argument("--split",         default="both",
                   choices=["val", "test", "both"],
                   help="Which split(s) to evaluate (default: both)")
    p.add_argument("--metadata-csv",  type=str, default=None)
    p.add_argument("--batch-size",    type=int, default=16)
    p.add_argument("--device",        type=str, default=None)
    p.add_argument("--num-workers",   type=int, default=4)
    p.add_argument("--log-dir",       type=str, default="logs")
    p.add_argument("--results-dir",   type=str, default="results")
    p.add_argument("--visualize",     action="store_true",
                   help="Generate and save Grad-CAM heatmaps and STN/compartment crop visualizations")
    return p.parse_args()


@torch.no_grad()
def _collect(
    model,
    loader,
    device,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Collect (logits, labels, fgbf_logits) over an entire loader.

    Returns
    -------
    logits:      (N, num_classes) numpy array of 5-class raw scores.
    labels:      (N,) integer KL grade ground truth.
    fgbf_logits: (N, 3) numpy array of FGBF auxiliary head scores, or
                 None when the model does not output 'fgbf_logits'.
    """
    model.eval()
    all_logits: list      = []
    all_labels: list      = []
    all_fgbf:   list      = []

    for batch in loader:
        crops, labels = batch
        g = crops[0].to(device)
        # medial / lateral crops kept for models that expect them via model(g)
        # (DRPNet forward only uses global_crop; kept for API compatibility)
        preds = model(g)

        all_logits.append(preds["logits"].cpu())
        all_labels.append(labels["kl"])

        if "fgbf_logits" in preds:
            all_fgbf.append(preds["fgbf_logits"].cpu())

    logits_np = _to_numpy(torch.cat(all_logits, dim=0))
    labels_np = _to_numpy(torch.cat(all_labels, dim=0)).astype(int)
    fgbf_np   = _to_numpy(torch.cat(all_fgbf, dim=0)) if all_fgbf else None

    return logits_np, labels_np, fgbf_np


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

    model = DRPNet(cfg.model).to(device)
    load_checkpoint(args.checkpoint, model, device=device)

    _, val_loader = build_loaders(cfg)
    test_loader   = build_test_loader(cfg)

    logger.info("Collecting logits …")
    val_logits,  val_labels,  val_fgbf_logits  = _collect(model, val_loader,  device)
    test_logits, test_labels, test_fgbf_logits = _collect(model, test_loader, device)

    # Skip splits the user didn't request
    if args.split == "val":
        test_logits      = test_labels      = None
        test_fgbf_logits = None
    elif args.split == "test":
        # still pass val (required by generate_all_reports), but only
        # report test prominently
        pass

    writer = ResultsWriter(args.exp, args.results_dir)

    generate_all_reports(
        writer            = writer,
        history           = {},           # no history in standalone eval
        train_logits      = None,
        train_labels      = None,
        val_logits        = val_logits,
        val_labels        = val_labels,
        test_logits       = test_logits,
        test_labels       = test_labels,
        num_classes       = cfg.model.num_classes,
        parameters        = count_parameters(model),
        results_dir       = args.results_dir,
        val_fgbf_logits   = val_fgbf_logits,
        test_fgbf_logits  = test_fgbf_logits,
        train_fgbf_logits = None,
    )

    if args.visualize:
        from visualization import run_visualizations
        vis_loader = test_loader if test_loader is not None else val_loader
        if vis_loader is not None:
            run_visualizations(
                model=model,
                loader=vis_loader,
                device=device,
                output_dir=writer.root,
                max_samples=15,
            )


if __name__ == "__main__":
    main()
