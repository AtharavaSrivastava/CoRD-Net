"""
dataset.py
==========
Production-ready OAI knee X-ray dataset loader for CoRD-Net.

Supported layouts
-----------------
Layout A — Grade subdirectories (no metadata CSV):

    root/
      0/  img001.png  img002.png  ...
      1/  img101.png  ...
      ...
      4/  ...

    KL grade is taken from the subdirectory name.
    Auxiliary labels (JSN, osteophyte) default to -1 (ignored by loss).

Layout B — Flat directory + metadata CSV:

    root/
      img001.png
      img002.png
      ...

    metadata.csv columns (at minimum):
        filename   — image filename (basename, e.g. "img001.png")
        kl         — integer KL grade 0-4
        jsn_med    — medial JSN 0-3, or -1 for missing
        jsn_lat    — lateral JSN 0-3, or -1 for missing
        osteo_mf   — medial femur osteophyte 0-2, or -1
        osteo_lf   — lateral femur osteophyte 0-2, or -1
        osteo_mt   — medial tibia osteophyte 0-2, or -1
        osteo_lt   — lateral tibia osteophyte 0-2, or -1

Compartment crops (E4+)
-----------------------
For E4–E8, the DataLoader returns three crops: global, medial, lateral.
The dataset looks for sibling files built from the global filename using
cfg.medial_suffix and cfg.lateral_suffix (default: "_MED", "_LAT").

Example:
    global  : root/img001.png
    medial  : root/img001_MED.png
    lateral : root/img001_LAT.png

If compartment files are absent and use_compartment=True, the global
crop is replicated for both compartments (acceptable for quick tests;
for real training ensure proper crops exist).

Usage
-----
    from config import get_config
    from dataset import build_loaders

    cfg = get_config("e8", data_root="/data/OAI",
                     metadata_csv="/data/OAI/metadata.csv")
    train_loader, val_loader = build_loaders(cfg)
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split

from augmentation import get_train_transforms, get_val_transforms
from config import Config, ModelConfig, TrainingConfig

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Label record
# ──────────────────────────────────────────────────────────────────────────────

class _Sample:
    """Lightweight container for one dataset sample."""
    __slots__ = ("path", "kl", "jsn_med", "jsn_lat", "osteophyte")

    def __init__(
        self,
        path: Path,
        kl: int,
        jsn_med: int = -1,
        jsn_lat: int = -1,
        osteophyte: Tuple[int, int, int, int] = (-1, -1, -1, -1),
    ) -> None:
        self.path       = path
        self.kl         = kl
        self.jsn_med    = jsn_med
        self.jsn_lat    = jsn_lat
        self.osteophyte = osteophyte   # (mf, lf, mt, lt)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class OAIDataset(Dataset):
    """
    OAI knee X-ray dataset supporting Layouts A and B (see module docstring).

    Parameters
    ----------
    samples:
        Pre-built list of _Sample objects (built by build_loaders).
    cfg_model:
        ModelConfig for image size, compartment flags, etc.
    cfg_train:
        TrainingConfig for suffix conventions.
    split:
        'train' | 'val' | 'test' — selects augmentation pipeline.
    transform:
        Override transform; None → use default pipeline for split.
    """

    def __init__(
        self,
        samples: List[_Sample],
        cfg_model: ModelConfig,
        cfg_train: TrainingConfig,
        split: str = "train",
        transform: Optional[Callable] = None,
    ) -> None:
        self.samples      = samples
        self.cfg_model    = cfg_model
        self.cfg_train    = cfg_train
        self.split        = split
        self.use_3crop    = cfg_model.use_compartment
        self.med_suffix   = cfg_train.medial_suffix
        self.lat_suffix   = cfg_train.lateral_suffix
        self.transform    = transform or (
            get_train_transforms(cfg_model) if split == "train"
            else get_val_transforms(cfg_model)
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self, path: Path) -> torch.Tensor:
        """Open an image as grayscale and apply the transform pipeline."""
        if not path.exists():
            raise FileNotFoundError(
                f"Image not found: {path}\n"
                "Check that your dataset root is correct and the file exists."
            )
        img = Image.open(path).convert("L")
        return self.transform(img)   # → (3, H, W)

    def _compartment_path(self, base: Path, suffix: str) -> Path:
        """Derive medial/lateral path from global path using suffix convention."""
        return base.parent / (base.stem + suffix + base.suffix)

    def __getitem__(self, idx: int) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:
        s = self.samples[idx]

        global_img = self._load(s.path)

        if self.use_3crop:
            med_path = self._compartment_path(s.path, self.med_suffix)
            lat_path = self._compartment_path(s.path, self.lat_suffix)

            # Gracefully fall back to global crop if compartment files are missing
            if not med_path.exists():
                logger.debug("Medial crop missing for %s — using global crop", s.path.name)
                med_img = global_img.clone()
            else:
                med_img = self._load(med_path)

            if not lat_path.exists():
                logger.debug("Lateral crop missing for %s — using global crop", s.path.name)
                lat_img = global_img.clone()
            else:
                lat_img = self._load(lat_path)

            crops = [global_img, med_img, lat_img]
        else:
            crops = [global_img]

        labels: Dict[str, torch.Tensor] = {
            "kl":         torch.tensor(s.kl,         dtype=torch.long),
            "jsn_med":    torch.tensor(s.jsn_med,    dtype=torch.long),
            "jsn_lat":    torch.tensor(s.jsn_lat,    dtype=torch.long),
            "osteophyte": torch.tensor(list(s.osteophyte), dtype=torch.long),
        }
        return crops, labels


# ──────────────────────────────────────────────────────────────────────────────
# Sample discovery helpers
# ──────────────────────────────────────────────────────────────────────────────

def _discover_layout_a(root: Path, num_classes: int) -> List[_Sample]:
    """
    Layout A: root/0/*.png, root/1/*.png, …

    KL grade = subdirectory name.  Auxiliary labels all -1.
    """
    samples: List[_Sample] = []
    for grade in range(num_classes):
        grade_dir = root / str(grade)
        if not grade_dir.exists():
            logger.warning("Grade directory not found: %s", grade_dir)
            continue
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"):
            for p in sorted(grade_dir.glob(ext)):
                samples.append(_Sample(path=p, kl=grade))
    return samples


def _discover_layout_b(root: Path, csv_path: Path) -> List[_Sample]:
    """
    Layout B: flat root/ directory + metadata CSV.

    Required CSV columns: filename, kl
    Optional columns: jsn_med, jsn_lat, osteo_mf, osteo_lf, osteo_mt, osteo_lt
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Metadata CSV not found: {csv_path}\n"
            "Provide --metadata-csv or use the grade-subdirectory layout."
        )

    samples: List[_Sample] = []
    missing_files: List[str] = []

    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"filename", "kl"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"Metadata CSV must contain columns: {required}.\n"
                f"Found: {reader.fieldnames}"
            )

        for row in reader:
            path = root / row["filename"]
            if not path.exists():
                missing_files.append(row["filename"])
                continue

            def _int(col: str, default: int = -1) -> int:
                val = row.get(col, "").strip()
                return int(val) if val not in ("", "nan", "NA", "None") else default

            osteophyte = (
                _int("osteo_mf"), _int("osteo_lf"),
                _int("osteo_mt"), _int("osteo_lt"),
            )
            samples.append(_Sample(
                path       = path,
                kl         = _int("kl"),
                jsn_med    = _int("jsn_med"),
                jsn_lat    = _int("jsn_lat"),
                osteophyte = osteophyte,
            ))

    if missing_files:
        logger.warning(
            "%d files listed in CSV but not found on disk (first 5: %s)",
            len(missing_files), missing_files[:5]
        )

    return samples


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def build_loaders(cfg: Config) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders from the OAI dataset.

    Parameters
    ----------
    cfg:
        Top-level Config.  cfg.training.data_root must be set.

    Returns
    -------
    train_loader, val_loader

    Raises
    ------
    ValueError:  data_root not specified, or no samples found.
    FileNotFoundError: data_root or metadata_csv doesn't exist.
    """
    tcfg  = cfg.training
    mcfg  = cfg.model

    if not tcfg.data_root:
        raise ValueError(
            "Dataset root not specified. "
            "Pass --data-root /path/to/OAI to train.py, "
            "or set cfg.training.data_root in your config."
        )

    root = Path(tcfg.data_root)
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset root does not exist: {root}\n"
            "Check your --data-root argument."
        )

    # Discover samples
    if tcfg.metadata_csv:
        logger.info("Loading dataset from CSV: %s", tcfg.metadata_csv)
        all_samples = _discover_layout_b(root, Path(tcfg.metadata_csv))
    else:
        logger.info("Loading dataset from grade subdirectories: %s", root)
        all_samples = _discover_layout_a(root, mcfg.num_classes)

    if not all_samples:
        raise ValueError(
            f"No samples found in {root}.\n"
            "Expected either grade subdirectories (0/,1/,…) or a --metadata-csv."
        )

    logger.info("Total samples discovered: %d", len(all_samples))

    # Patient-agnostic random split (replace with patient-level split for OAI)
    n       = len(all_samples)
    n_train = int(n * tcfg.train_ratio)
    n_val   = int(n * tcfg.val_ratio)
    n_test  = n - n_train - n_val

    generator = torch.Generator().manual_seed(tcfg.seed)

    # Build full dataset with train transforms; val/test get their own wrapper
    full_ds = OAIDataset(all_samples, mcfg, tcfg, split="train")
    train_indices, val_indices, test_indices = _split_indices(
        n, n_train, n_val, n_test, generator
    )

    train_samples = [all_samples[i] for i in train_indices]
    val_samples   = [all_samples[i] for i in val_indices]

    train_ds = OAIDataset(train_samples, mcfg, tcfg, split="train")
    val_ds   = OAIDataset(val_samples,   mcfg, tcfg, split="val")

    logger.info(
        "Split: train=%d  val=%d  test=%d",
        len(train_samples), len(val_samples), len(test_indices)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size  = tcfg.batch_size,
        shuffle     = True,
        num_workers = tcfg.num_workers,
        pin_memory  = tcfg.pin_memory,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = tcfg.batch_size,
        shuffle     = False,
        num_workers = tcfg.num_workers,
        pin_memory  = tcfg.pin_memory,
        drop_last   = False,
    )
    return train_loader, val_loader


def build_test_loader(cfg: Config) -> DataLoader:
    """
    Build a test DataLoader (held-out split, no shuffling, val transforms).

    Parameters
    ----------
    cfg: Top-level Config with data_root set.
    """
    tcfg = cfg.training
    mcfg = cfg.model

    if not tcfg.data_root:
        raise ValueError("data_root not set in config.")

    root = Path(tcfg.data_root)
    if tcfg.metadata_csv:
        all_samples = _discover_layout_b(root, Path(tcfg.metadata_csv))
    else:
        all_samples = _discover_layout_a(root, mcfg.num_classes)

    n       = len(all_samples)
    n_train = int(n * tcfg.train_ratio)
    n_val   = int(n * tcfg.val_ratio)
    n_test  = n - n_train - n_val

    generator = torch.Generator().manual_seed(tcfg.seed)
    _, _, test_indices = _split_indices(n, n_train, n_val, n_test, generator)
    test_samples = [all_samples[i] for i in test_indices]

    test_ds = OAIDataset(test_samples, mcfg, tcfg, split="test")
    return DataLoader(
        test_ds,
        batch_size  = tcfg.batch_size,
        shuffle     = False,
        num_workers = tcfg.num_workers,
        pin_memory  = tcfg.pin_memory,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _split_indices(
    n: int, n_train: int, n_val: int, n_test: int,
    generator: torch.Generator,
) -> Tuple[List[int], List[int], List[int]]:
    """Return three disjoint index lists via seeded random permutation."""
    perm   = torch.randperm(n, generator=generator).tolist()
    train  = perm[:n_train]
    val    = perm[n_train:n_train + n_val]
    test   = perm[n_train + n_val:]
    return train, val, test
