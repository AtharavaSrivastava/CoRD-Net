"""
dataset.py
==========
Production-ready OAI knee X-ray dataset loader for CoRD-Net.

Split modes  (selected automatically — no flag required)
------------------------------------------------------------
MODE 0 — Pre-split directories  ← YOUR LAYOUT, highest priority
    dataset/
        train/  0/  1/  2/  3/  4/
        val/    0/  1/  2/  3/  4/
        test/   0/  1/  2/  3/  4/

    Detected when --data-root contains subdirectories named
    "train", "val", and "test", each containing numbered grade folders.
    No CSV required.  No random splitting.  Fully reproducible.

MODE 1 — Random split
    dataset/
        0/  1/  2/  3/  4/

    Used when the root contains grade folders directly (no train/val/test
    subdirs).  Splits by train_ratio / val_ratio / test_ratio with a fixed
    seed.

MODE 2 — Split column in CSV
    metadata.csv has a "split" column → train / val / test per row.

MODE 3 — Separate CSV files
    --train-csv  --val-csv  --test-csv  each list their own images.

Compartment crops (E4–E8)
--------------------------
The loader looks for medial/lateral sibling files using the suffix
convention from config (default _MED / _LAT).  If absent, the global
crop is replicated with a warning.  This works identically across all
four split modes.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from augmentation import get_train_transforms, get_val_transforms
from config import Config, ModelConfig, TrainingConfig

logger = logging.getLogger(__name__)

_IMAGE_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff")


# ──────────────────────────────────────────────────────────────────────────────
# Label record
# ──────────────────────────────────────────────────────────────────────────────

class _Sample:
    __slots__ = ("path", "kl", "jsn_med", "jsn_lat", "osteophyte")

    def __init__(
        self,
        path:       Path,
        kl:         int,
        jsn_med:    int = -1,
        jsn_lat:    int = -1,
        osteophyte: Tuple[int, int, int, int] = (-1, -1, -1, -1),
    ) -> None:
        self.path       = path
        self.kl         = kl
        self.jsn_med    = jsn_med
        self.jsn_lat    = jsn_lat
        self.osteophyte = osteophyte


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class OAIDataset(Dataset):
    """
    OAI knee X-ray dataset.

    Parameters
    ----------
    samples:    Pre-built list of _Sample objects.
    cfg_model:  ModelConfig (image size, compartment flags).
    cfg_train:  TrainingConfig (suffix conventions).
    split:      'train' | 'val' | 'test' — picks augmentation pipeline.
    transform:  Override transform; None → default pipeline for split.
    """

    def __init__(
        self,
        samples:   List[_Sample],
        cfg_model: ModelConfig,
        cfg_train: TrainingConfig,
        split:     str = "train",
        transform: Optional[Callable] = None,
    ) -> None:
        self.samples    = samples
        self.cfg_model  = cfg_model
        self.cfg_train  = cfg_train
        self.split      = split
        self.transform  = transform or (
            get_train_transforms(cfg_model) if split == "train"
            else get_val_transforms(cfg_model)
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self, path: Path) -> torch.Tensor:
        if not path.exists():
            raise FileNotFoundError(
                f"Image not found: {path}\n"
                "Check that your --data-root is correct."
            )
        return self.transform(Image.open(path).convert("L"))

    def __getitem__(
        self, idx: int
    ) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:
        s          = self.samples[idx]
        global_img = self._load(s.path)

        labels: Dict[str, torch.Tensor] = {
            "kl":         torch.tensor(s.kl,         dtype=torch.long),
            "jsn_med":    torch.tensor(s.jsn_med,    dtype=torch.long),
            "jsn_lat":    torch.tensor(s.jsn_lat,    dtype=torch.long),
            "osteophyte": torch.tensor(list(s.osteophyte), dtype=torch.long),
        }
        if idx == 3166:
            print(
                f"idx={idx}",
                global_img.mean().item(),
                global_img.std().item()
            )
        return idx, [global_img], labels


# ──────────────────────────────────────────────────────────────────────────────
# Sample discovery helpers
# ──────────────────────────────────────────────────────────────────────────────

def _parse_int(val: str, default: int = -1) -> int:
    v = val.strip() if val else ""
    return int(v) if v not in ("", "nan", "NA", "None") else default


def _row_to_sample(root: Path, row: Dict[str, str]) -> Optional[_Sample]:
    path = root / row["filename"]
    if not path.exists():
        return None
    return _Sample(
        path       = path,
        kl         = _parse_int(row.get("kl", ""), 0),
        jsn_med    = _parse_int(row.get("jsn_med",  "")),
        jsn_lat    = _parse_int(row.get("jsn_lat",  "")),
        osteophyte = (
            _parse_int(row.get("osteo_mf", "")),
            _parse_int(row.get("osteo_lf", "")),
            _parse_int(row.get("osteo_mt", "")),
            _parse_int(row.get("osteo_lt", "")),
        ),
    )


def _discover_grade_dirs(root: Path, num_classes: int) -> List[_Sample]:
    """
    Load images from root/0/, root/1/, …, root/4/.

    KL grade is inferred from the subdirectory name.
    Auxiliary labels default to -1 (ignored by loss).
    """
    samples: List[_Sample] = []
    for grade in range(num_classes):
        grade_dir = root / str(grade)
        if not grade_dir.exists():
            logger.warning("Grade directory not found: %s", grade_dir)
            continue
        for ext in _IMAGE_EXTS:
            for p in sorted(grade_dir.glob(ext)):
                samples.append(_Sample(path=p, kl=grade))
    return samples


# ──────────────────────────────────────────────────────────────────────────────
# MODE 0 — Pre-split directory layout  (NEW — highest priority)
# ──────────────────────────────────────────────────────────────────────────────

def _is_presplit(root: Path) -> bool:
    """
    Return True when root contains train/, val/, and test/ subdirectories,
    each of which contains at least one numbered grade subdirectory.

    This is the definitive check for the pre-split layout.  It never
    triggers for a flat grade-subdir layout (where the grade folders sit
    directly under root).
    """
    for split_name in ("train", "val", "test"):
        split_dir = root / split_name
        if not split_dir.is_dir():
            return False
        # At least one numbered grade subdir must exist inside
        has_grade = any(
            (split_dir / str(g)).is_dir() for g in range(10)
        )
        if not has_grade:
            return False
    return True


def _load_presplit(
    root: Path, num_classes: int
) -> Tuple[List[_Sample], List[_Sample], List[_Sample]]:
    """
    Load samples directly from root/train/, root/val/, root/test/.

    Each subdirectory has the structure:
        <split>/
            0/  image001.png  ...
            1/  ...
            ...
            4/  ...

    No shuffling.  No random seed.  Completely deterministic.

    Returns
    -------
    (train_samples, val_samples, test_samples)
    """
    train_s = _discover_grade_dirs(root / "train", num_classes)
    val_s   = _discover_grade_dirs(root / "val",   num_classes)
    test_s  = _discover_grade_dirs(root / "test",  num_classes)

    # Sanity: assert no path appears in more than one split
    train_paths = {s.path for s in train_s}
    val_paths   = {s.path for s in val_s}
    test_paths  = {s.path for s in test_s}

    tv = train_paths & val_paths
    tt = train_paths & test_paths
    vt = val_paths   & test_paths
    if tv or tt or vt:
        overlaps = tv | tt | vt
        raise RuntimeError(
            f"{len(overlaps)} image(s) appear in more than one split. "
            "This should never happen with a pre-split directory layout. "
            f"First offender: {next(iter(overlaps))}"
        )

    logger.info(
        "Pre-split layout detected: train=%d  val=%d  test=%d",
        len(train_s), len(val_s), len(test_s),
    )
    return train_s, val_s, test_s


# ──────────────────────────────────────────────────────────────────────────────
# MODE 2 — Split column in CSV
# ──────────────────────────────────────────────────────────────────────────────

def _split_by_column(
    root: Path, csv_path: Path
) -> Tuple[List[_Sample], List[_Sample], List[_Sample]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

    buckets: Dict[str, List[_Sample]] = {"train": [], "val": [], "test": []}
    missing: List[str] = []

    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        if "split" not in set(reader.fieldnames or []):
            raise ValueError(
                "CSV does not have a 'split' column. "
                "Add 'split' with values train/val/test, "
                "or use --train-csv / --val-csv / --test-csv instead."
            )
        for row in reader:
            s = _row_to_sample(root, row)
            if s is None:
                missing.append(row["filename"])
                continue
            key = row.get("split", "train").strip().lower()
            buckets[key if key in buckets else "train"].append(s)

    if missing:
        logger.warning("%d CSV rows missing on disk (first 5: %s)",
                       len(missing), missing[:5])
    logger.info("Split column: train=%d  val=%d  test=%d",
                len(buckets["train"]), len(buckets["val"]), len(buckets["test"]))
    return buckets["train"], buckets["val"], buckets["test"]


# ──────────────────────────────────────────────────────────────────────────────
# MODE 3 — Separate CSV per split
# ──────────────────────────────────────────────────────────────────────────────

def _load_split_csv(root: Path, csv_path: str) -> List[_Sample]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Split CSV not found: {path}\n"
            "Check --train-csv / --val-csv / --test-csv."
        )
    samples, missing = [], []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            s = _row_to_sample(root, row)
            if s is None:
                missing.append(row.get("filename", "?"))
            else:
                samples.append(s)
    if missing:
        logger.warning("%d files missing (first 5: %s)", len(missing), missing[:5])
    return samples


# ──────────────────────────────────────────────────────────────────────────────
# MODE 1 — Random split
# ──────────────────────────────────────────────────────────────────────────────

def _random_split(
    all_samples: List[_Sample],
    train_ratio: float,
    val_ratio:   float,
    seed:        int,
) -> Tuple[List[_Sample], List[_Sample], List[_Sample]]:
    n       = len(all_samples)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    gen     = torch.Generator().manual_seed(seed)
    perm    = torch.randperm(n, generator=gen).tolist()
    train   = [all_samples[i] for i in perm[:n_train]]
    val     = [all_samples[i] for i in perm[n_train:n_train + n_val]]
    test    = [all_samples[i] for i in perm[n_train + n_val:]]
    return train, val, test


# ──────────────────────────────────────────────────────────────────────────────
# Split resolution  (mode priority order)
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_splits(
    cfg: Config,
) -> Tuple[List[_Sample], List[_Sample], List[_Sample]]:
    """
    Select the split mode and return (train, val, test) sample lists.

    Priority order (first match wins):
      0. Pre-split dirs   root/train/0..4, root/val/0..4, root/test/0..4
      1. Separate CSVs    --train-csv + --val-csv [+ --test-csv]
      2. Split column     --metadata-csv with a 'split' column
      3. Random split     everything else
    """
    tcfg = cfg.training
    mcfg = cfg.model
    root = Path(tcfg.data_root)

    # ── Mode 0: pre-split directory layout ───────────────────────────────
    if _is_presplit(root):
        logger.info("Split mode: pre-split directories (train/ val/ test/)")
        return _load_presplit(root, mcfg.num_classes)

    # ── Mode 3: separate CSV files ────────────────────────────────────────
    if tcfg.train_csv and tcfg.val_csv:
        logger.info("Split mode: separate CSV files")
        train_s = _load_split_csv(root, tcfg.train_csv)
        val_s   = _load_split_csv(root, tcfg.val_csv)
        test_s  = _load_split_csv(root, tcfg.test_csv) if tcfg.test_csv else []
        logger.info("Separate CSVs: train=%d  val=%d  test=%d",
                    len(train_s), len(val_s), len(test_s))
        return train_s, val_s, test_s

    # ── Mode 2: split column in CSV ───────────────────────────────────────
    if tcfg.metadata_csv:
        with open(tcfg.metadata_csv, newline="") as fh:
            has_split_col = "split" in (csv.DictReader(fh).fieldnames or [])
        if has_split_col:
            logger.info("Split mode: 'split' column in CSV")
            return _split_by_column(root, Path(tcfg.metadata_csv))
        else:
            logger.info("Split mode: random (CSV present, no 'split' column)")
            all_s = _load_split_csv(root, tcfg.metadata_csv)
            return _random_split(all_s, tcfg.train_ratio, tcfg.val_ratio, tcfg.seed)

    # ── Mode 1: random split from flat grade subdirectories ───────────────
    logger.info("Split mode: random (flat grade subdirectories)")
    all_s = _discover_grade_dirs(root, mcfg.num_classes)
    return _random_split(all_s, tcfg.train_ratio, tcfg.val_ratio, tcfg.seed)

# ──────────────────────────────────────────────────────────────────────────────
# Public DataLoader builders
# ──────────────────────────────────────────────────────────────────────────────

def _worker_init_fn(worker_id):
    import random
    import numpy as np

    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def _make_loader(
    samples:   List[_Sample],
    cfg_model: ModelConfig,
    cfg_train: TrainingConfig,
    split:     str,
    shuffle:   bool,
) -> DataLoader:
    ds = OAIDataset(samples, cfg_model, cfg_train, split=split)
    generator = torch.Generator()
    generator.manual_seed(42)

    return DataLoader(
        ds,
        batch_size=cfg_train.batch_size,
        shuffle=shuffle,
        num_workers=cfg_train.num_workers,
        pin_memory=cfg_train.pin_memory,
        drop_last=(split == "train"),
        generator=generator,
        worker_init_fn=_worker_init_fn,
    )


def build_loaders(cfg: Config) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders.

    The split mode is selected automatically — see module docstring.

    Returns
    -------
    train_loader, val_loader
    """
    if not cfg.training.data_root:
        raise ValueError(
            "Dataset root not specified. "
            "Pass --data-root /path/to/dataset to train.py."
        )
    root = Path(cfg.training.data_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    train_s, val_s, _ = _resolve_splits(cfg)

    if not train_s:
        raise ValueError(
            f"No training images found under {root}.\n"
            "Expected one of:\n"
            "  • root/train/0/ root/train/1/ … root/train/4/   (pre-split)\n"
            "  • root/0/ root/1/ … root/4/                      (random split)\n"
            "  • --train-csv / --metadata-csv                   (CSV modes)"
        )
    if not val_s:
        raise ValueError(
            f"No validation images found under {root}.\n"
            "For pre-split layouts make sure root/val/0..4/ exist and contain images."
        )

    logger.info("Dataset ready — train: %d  val: %d", len(train_s), len(val_s))
    return (
        _make_loader(train_s, cfg.model, cfg.training, "train", shuffle=True),
        _make_loader(val_s,   cfg.model, cfg.training, "val",   shuffle=False),
    )


def build_test_loader(cfg: Config) -> DataLoader:
    """
    Build a test DataLoader (val transforms, no shuffling).

    Returns an empty DataLoader if no test images are found, so
    downstream code never needs to guard for None.
    """
    if not cfg.training.data_root:
        raise ValueError("data_root not set.")
    root = Path(cfg.training.data_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    _, _, test_s = _resolve_splits(cfg)

    if not test_s:
        logger.warning(
            "No test images found — build_test_loader() returning empty DataLoader."
        )

    return _make_loader(test_s, cfg.model, cfg.training, "test", shuffle=False)
