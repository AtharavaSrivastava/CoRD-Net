"""
run_experiment.py
=================
CoRD-Net ablation runner — thin dispatch layer.

This file's only job is to parse arguments, build DRPNet via config,
and call the Trainer stub.  All architecture and training logic lives
in models/ and trainer.py.

Usage
-----
    python run_experiment.py --exp e1
    python run_experiment.py --exp e8 --steps 5 --batch-size 2
    python run_experiment.py --exp all
"""

from __future__ import annotations

import argparse
import logging
import sys

import torch
import torch.nn as nn

from config import get_config, EXPERIMENT_NAMES
from losses import MultiTaskLoss
from models.drpnet import DRPNet
from trainer import Trainer
from utils import get_device, seed_everything, setup_logging

logger = logging.getLogger(__name__)

# Experiments that need >1 crop forward (E4+): DRPNet handles this internally,
# but the stub uses 112 px to stay within CPU memory limits.
_HEAVY_EXPS = {"e4", "e5", "e6", "e7", "e8"}


def run_one(exp: str, args: argparse.Namespace) -> None:
    """Build DRPNet for *exp* and run a training stub."""
    device = get_device(args.device)

    cfg = get_config(
        exp,
        pretrained=False,
        device=str(device),
        batch_size=args.batch_size,
    )

    logger.info("")
    logger.info("=" * 62)
    logger.info("  %s — %s", exp.upper(), cfg.description)
    logger.info("=" * 62)

    model = DRPNet(cfg.model).to(device)

    # Use MultiTaskLoss for E8 (aux heads), CrossEntropy for E1–E7
    if cfg.model.use_aux_heads:
        loss_fn = MultiTaskLoss(cfg.training)
    else:
        loss_fn = nn.CrossEntropyLoss()

    trainer = Trainer(model, loss_fn, cfg)

    # Reduce image size for multi-crop experiments on CPU to avoid OOM
    image_size = 112 if (exp in _HEAVY_EXPS and str(device) == "cpu") else 224

    last = trainer.stub_fit(steps=args.steps, image_size=image_size)
    logger.info("  %s PASS ✓  final total=%.4f", exp.upper(), last.get("total", 0.0))


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "--exp",
        choices=[*EXPERIMENT_NAMES.keys(), "all"],
        default="all",
        metavar="EXP",
        help="Experiment: e1 … e8  or  all  (default: all)",
    )
    parser.add_argument("--batch-size", type=int, default=4, metavar="B")
    parser.add_argument("--steps",      type=int, default=3, metavar="N",
                        help="Synthetic training steps per experiment (default: 3)")
    parser.add_argument("--device",     type=str, default=None,
                        help="cpu | cuda  (default: auto)")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--log-dir",    type=str, default="logs")
    args = parser.parse_args()

    seed_everything(args.seed)
    setup_logging(args.log_dir, args.exp or "all")

    targets = list(EXPERIMENT_NAMES.keys()) if args.exp == "all" else [args.exp]

    logger.info("CoRD-Net Ablation Suite | device=%s | batch=%d | steps=%d",
                args.device or "auto", args.batch_size, args.steps)

    for exp in targets:
        try:
            run_one(exp, args)
        except Exception:
            logger.exception("  %s FAILED", exp.upper())
            sys.exit(1)

    logger.info("")
    logger.info("=" * 62)
    logger.info("  Done — %d experiment(s) passed ✓", len(targets))
    logger.info("=" * 62)


if __name__ == "__main__":
    main()
