"""
utils.py
========
Shared utilities for CoRD-Net: device resolution, reproducibility,
checkpoint I/O, parameter reporting, module summary, and logging.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Device / Seed
# ──────────────────────────────────────────────────────────────────────────────

def get_device(override: Optional[str] = None) -> torch.device:
    """Return a torch device, auto-detecting CUDA when override is None."""
    if override:
        return torch.device(override)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int = 42) -> None:
    """Fix random seeds for reproducibility across Python, NumPy, and PyTorch."""
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


# ──────────────────────────────────────────────────────────────────────────────
# Parameter reporting
# ──────────────────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_parameters(model: nn.Module) -> int:
    """Return the total number of parameters (trainable + frozen)."""
    return sum(p.numel() for p in model.parameters())


def log_parameter_summary(model: nn.Module, experiment: str) -> None:
    """
    Log total and trainable parameter counts at startup.

    Output example::

        ┌─ E5 parameter summary ─────────────────────────────┐
        │  Total parameters    : 33,218,209                  │
        │  Trainable parameters: 33,218,209                  │
        └────────────────────────────────────────────────────┘
    """
    total     = count_all_parameters(model)
    trainable = count_parameters(model)
    width = 52
    bar   = "─" * width
    logger.info("┌─ %s parameter summary %s┐", experiment.upper(),
                bar[len(experiment) + 20:])
    logger.info("│  Total parameters    : %-28s│", f"{total:,}")
    logger.info("│  Trainable parameters: %-28s│", f"{trainable:,}")
    logger.info("└%s┘", bar)


# ──────────────────────────────────────────────────────────────────────────────
# Module / ablation summary
# ──────────────────────────────────────────────────────────────────────────────

_MODULE_LABELS = [
    ("use_stn",           "STN (Auto-Localization)"),
    ("use_dual_intensity","Dual-Intensity Stem"),
    ("use_compartment",   "Compartment Branches"),
    ("use_drp",           "Soft ROI Mask (DRP Block)"),
    ("use_pgr",           "Prototype-Guided Refinement (PGR)"),
    ("use_rtc",           "Relational Token Coupling (RTC)"),
    ("use_aux_heads",     "Auxiliary Heads"),
]


def log_model_summary(model: nn.Module, experiment: str) -> None:
    """
    Print the active/inactive module flags for an experiment.

    Output example::

        ┌─ E6 active modules ────────────────────────────────┐
        │  ✓  STN (Auto-Localization)                        │
        │  ✓  Dual-Intensity Stem                            │
        │  ✓  Compartment Branches                           │
        │  ✓  Soft ROI Mask (DRP Block)                      │
        │  ✓  Prototype-Guided Refinement (PGR)              │
        │  ✗  Relational Token Coupling (RTC)                │
        │  ✗  Auxiliary Heads                                │
        └────────────────────────────────────────────────────┘
    """
    cfg   = model.cfg          # ModelConfig stored on DRPNet
    width = 52
    bar   = "─" * width
    logger.info("┌─ %s active modules %s┐", experiment.upper(),
                bar[len(experiment) + 16:])
    for attr, label in _MODULE_LABELS:
        tick = "✓" if getattr(cfg, attr, False) else "✗"
        line = f"  {tick}  {label}"
        logger.info("│%-*s│", width, line)
    logger.info("└%s┘", bar)


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint I/O
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: str,
    filename: str = "checkpoint.pt",
) -> Path:
    """
    Save *state* to *checkpoint_dir/filename*.

    Creates the directory if it doesn't exist.
    """
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    fpath = path / filename
    torch.save(state, fpath)
    logger.info("Checkpoint saved → %s", fpath)
    return fpath


def load_checkpoint(
    fpath: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load a checkpoint into *model* (and optionally *optimizer*, *scheduler*).

    Parameters
    ----------
    fpath:      Path to the .pt checkpoint file.
    model:      DRPNet instance (weights are loaded in-place).
    optimizer:  If provided, optimizer state is restored.
    scheduler:  If provided, scheduler state is restored.
    device:     Map location; defaults to CPU.

    Returns
    -------
    The full checkpoint dict (contains epoch, losses, experiment, etc.).

    Raises
    ------
    FileNotFoundError: if fpath does not exist.
    """
    fpath = Path(fpath)
    if not fpath.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {fpath}\n"
            "Check your --resume path."
        )
    checkpoint = torch.load(fpath, map_location=device or "cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    logger.info(
        "Checkpoint loaded ← %s  (experiment=%s  epoch=%s)",
        fpath,
        checkpoint.get("experiment", "?"),
        checkpoint.get("epoch", "?"),
    )
    return checkpoint


# ──────────────────────────────────────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────────────────────────────────────

def setup_logging(log_dir: str, experiment: str, level: int = logging.INFO) -> None:
    """
    Configure root logger to write to stdout and a timestamped log file.

    Parameters
    ----------
    log_dir:     Directory where the log file is created.
    experiment:  Experiment tag used in the log filename.
    level:       Logging verbosity level.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / f"{experiment}.log"

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)

    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    logger.info("Logging to %s", log_file)


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic-data stub (run_experiment.py only — not used in real training)
# ──────────────────────────────────────────────────────────────────────────────

def make_labels_stub(
    batch_size: int,
    num_classes: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Generate synthetic labels for run_experiment.py integration tests.

    NOT used in real training — the DataLoader supplies real labels.
    """
    b = batch_size
    return {
        "kl":         torch.randint(0, num_classes, (b,), device=device),
        "jsn_med":    torch.tensor([0, 1, -1, 2, 0][:b], device=device),
        "jsn_lat":    torch.tensor([1, -1, 0, 3, 1][:b], device=device),
        "osteophyte": torch.randint(-1, 3, (b, 4), device=device),
    }
