"""
train.py
========
Main training entry point for CoRD-Net.

    python train.py --exp e1 --data-root /data/OAI
    python train.py --exp e8 --data-root /data/OAI --metadata-csv /data/OAI/labels.csv
    python train.py --exp e5 --data-root /data/OAI --pretrained --device cuda
    python train.py --exp e6 --data-root /data/OAI --resume checkpoints/e6_best.pt
"""

from __future__ import annotations

import argparse
import logging

from config import get_config, EXPERIMENT_NAMES
from dataset import build_loaders
from losses import MultiTaskLoss
from models.drpnet import DRPNet
from trainer import Trainer
from utils import get_device, seed_everything, setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--exp",           required=True,
                   choices=list(EXPERIMENT_NAMES.keys()),
                   help="Ablation experiment to run (e1 … e8)")
    p.add_argument("--data-root",     type=str, default=None,
                   help="Root directory of the OAI dataset")
    p.add_argument("--metadata-csv",  type=str, default=None,
                   help="Path to OAI metadata CSV with KL/JSN/osteophyte labels")
    p.add_argument("--epochs",        type=int, default=None)
    p.add_argument("--batch-size",    type=int, default=None)
    p.add_argument("--lr",            type=float, default=None,
                   help="Learning rate (default: 1e-4)")
    p.add_argument("--device",        type=str, default=None,
                   help="cpu | cuda  (default: auto-detect)")
    p.add_argument("--pretrained",    action="store_true",
                   help="Use ImageNet-pretrained ConvNeXt-tiny weights")
    p.add_argument("--resume",        type=str, default=None,
                   help="Path to checkpoint to resume training from")
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--log-dir",       type=str, default="logs")
    p.add_argument("--checkpoint-dir",type=str, default=None,
                   help="Override checkpoint directory")
    p.add_argument("--num-workers",   type=int, default=None)
    p.add_argument("--amp",           action="store_true",
                   help="Enable automatic mixed precision (CUDA only)")
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

    # CLI overrides not in get_config signature
    if args.checkpoint_dir:
        cfg.training.checkpoint_dir = args.checkpoint_dir
    if args.num_workers is not None:
        cfg.training.num_workers = args.num_workers
    if args.amp:
        cfg.training.amp = True

    logger.info("═" * 62)
    logger.info("  CoRD-Net — %s", cfg.description)
    logger.info("  Experiment : %s", args.exp.upper())
    logger.info("  Device     : %s", cfg.training.device or "auto")
    logger.info("  Epochs     : %d", cfg.training.epochs)
    logger.info("  Batch size : %d", cfg.training.batch_size)
    logger.info("  Data root  : %s", cfg.training.data_root or "not set")
    logger.info("  Pretrained : %s", cfg.model.pretrained)
    logger.info("═" * 62)

    # Build data loaders
    train_loader, val_loader = build_loaders(cfg)

    # Build model + loss
    model   = DRPNet(cfg.model)
    loss_fn = (
        MultiTaskLoss(cfg.training)
        if cfg.model.use_aux_heads
        else __import__("torch").nn.CrossEntropyLoss()
    )

    # Build trainer and run
    trainer = Trainer(model, loss_fn, cfg)
    trainer.fit(train_loader, val_loader, resume=args.resume)


if __name__ == "__main__":
    main()
