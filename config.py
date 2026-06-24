"""
config.py
=========
Typed configuration dataclasses for CoRD-Net.

All hyperparameters live here.  No magic numbers appear elsewhere in the
codebase — every module receives a config object via dependency injection.

Usage
-----
    from config import ModelConfig, TrainingConfig, get_config

    cfg = get_config("e8")
    model = DRPNet(cfg.model)
    trainer = Trainer(model, cfg.training)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Model Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """Complete specification of the DRPNet architecture."""

    # Backbone
    backbone: str = "convnext_tiny"
    pretrained: bool = False
    backbone_feature_dim: int = 768   # ConvNeXt-tiny pooled output
    spatial_feature_dim: int = 768    # ConvNeXt-tiny spatial (before pool)

    # Embedding dimensions
    embedding_dim: int = 256          # DRP / PGR / RTC shared dim
    fused_dim: int = 512              # after projecting concatenated feats
    metric_embed_dim: int = 128       # MetricEmbeddingHead (SupCon)

    # Dataset
    num_classes: int = 5
    image_size: int = 224             # backbone canonical size
    in_channels: int = 3

    # STN (E2)
    stn_img_size: int = 512

    # PGR (E6)
    prototype_temperature: float = 0.07
    prototype_ema_momentum: float = 0.99
    pgr_num_heads: int = 4
    pgr_dropout: float = 0.1

    # RTC (E7)
    rtc_num_heads: int = 4
    rtc_dropout: float = 0.1
    rtc_use_global_context: bool = True

    # Ablation flags — set by get_config(experiment)
    use_stn: bool = False
    use_dual_intensity: bool = False
    use_compartment: bool = False
    use_drp: bool = False
    use_pgr: bool = False
    use_rtc: bool = False
    use_aux_heads: bool = False


# ──────────────────────────────────────────────────────────────────────────────
# Training Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    """Training loop, optimiser, and scheduler settings."""

    optimizer: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    scheduler: str = "cosine"         # 'cosine' | 'step' | 'none'
    warmup_epochs: int = 5

    batch_size: int = 16
    epochs: int = 100

    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        "h1": 1.0, "h2": 0.5, "h3": 0.3,
        "h4": 0.4, "h5": 0.4, "h6": 0.3,
        "h7": 0.2, "proto": 0.3,
    })

    active_heads: List[str] = field(
        default_factory=lambda: ["h1", "h2", "h3", "h4", "h5", "h6", "h7"]
    )

    device: Optional[str] = None      # None = auto-detect
    seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True
    gradient_clip: float = 1.0
    amp: bool = False

    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    results_dir: str = "results"
    save_every: int = 10

    # ── Dataset paths (set via CLI; no hardcoded paths) ───────────────────
    data_root: Optional[str] = None
    """Root directory of the OAI dataset (required for real training)."""

    metadata_csv: Optional[str] = None
    """Path to OAI metadata CSV with KL/JSN/osteophyte labels.
    If None, KL grade is inferred from the subdirectory name and
    auxiliary labels default to -1 (ignored by loss)."""

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    medial_suffix: str = "_MED"
    lateral_suffix: str = "_LAT"
    """Filename suffixes used to derive compartment crop paths.
    E.g. image "001.png" → medial "001_MED.png", lateral "001_LAT.png".
    Override if your OAI layout uses different conventions."""

    # ── Split Mode 3: separate CSV per split ──────────────────────────────
    train_csv: Optional[str] = None
    """Path to a CSV containing only training samples (Mode 3)."""

    val_csv: Optional[str] = None
    """Path to a CSV containing only validation samples (Mode 3)."""

    test_csv: Optional[str] = None
    """Path to a CSV containing only test samples (Mode 3).
    Optional — if omitted, the test DataLoader will be empty."""


# ──────────────────────────────────────────────────────────────────────────────
# Experiment registry
# ──────────────────────────────────────────────────────────────────────────────

_EXPERIMENT_FLAGS: Dict[str, Tuple[str, Dict[str, bool]]] = {
    "e1": ("Baseline ConvNeXt", {}),
    "e2": ("ConvNeXt + Auto-Localization (STN)",
           {"use_stn": True}),
    "e3": ("E2 + Dual-Intensity Stem",
           {"use_stn": True, "use_dual_intensity": True}),
    "e4": ("E3 + Compartment Branches",
           {"use_stn": True, "use_dual_intensity": True, "use_compartment": True}),
    "e5": ("E4 + Soft ROI Mask (DRP Block)",
           {"use_stn": True, "use_dual_intensity": True,
            "use_compartment": True, "use_drp": True}),
    "e6": ("E5 + Prototype-Guided Refinement",
           {"use_stn": True, "use_dual_intensity": True,
            "use_compartment": True, "use_drp": True, "use_pgr": True}),
    "e7": ("E6 + Relational Token Coupling",
           {"use_stn": True, "use_dual_intensity": True,
            "use_compartment": True, "use_drp": True,
            "use_pgr": True, "use_rtc": True}),
    "e8": ("Full DRP + Auxiliary Heads",
           {"use_stn": True, "use_dual_intensity": True,
            "use_compartment": True, "use_drp": True,
            "use_pgr": True, "use_rtc": True, "use_aux_heads": True}),
}

EXPERIMENT_NAMES: Dict[str, str] = {k: v[0] for k, v in _EXPERIMENT_FLAGS.items()}


@dataclass
class Config:
    """Top-level config bundling model + training settings."""
    experiment: str
    model: ModelConfig
    training: TrainingConfig

    @property
    def description(self) -> str:
        return EXPERIMENT_NAMES.get(self.experiment, self.experiment)


def get_config(
    experiment: str,
    *,
    pretrained: bool = False,
    device: Optional[str] = None,
    batch_size: Optional[int] = None,
    epochs: Optional[int] = None,
    learning_rate: Optional[float] = None,
    data_root: Optional[str] = None,
    metadata_csv: Optional[str] = None,
) -> Config:
    """Return a fully-merged Config for *experiment* (e1 … e8)."""
    if experiment not in _EXPERIMENT_FLAGS:
        raise ValueError(
            f"Unknown experiment '{experiment}'. "
            f"Valid choices: {list(_EXPERIMENT_FLAGS.keys())}"
        )
    _, flags = _EXPERIMENT_FLAGS[experiment]
    model_cfg = ModelConfig(pretrained=pretrained, **flags)
    train_cfg = TrainingConfig()
    if device is not None:
        train_cfg.device = device
    if batch_size is not None:
        train_cfg.batch_size = batch_size
    if epochs is not None:
        train_cfg.epochs = epochs
    if learning_rate is not None:
        train_cfg.learning_rate = learning_rate
    if data_root is not None:
        train_cfg.data_root = data_root
    if metadata_csv is not None:
        train_cfg.metadata_csv = metadata_csv
    return Config(experiment=experiment, model=model_cfg, training=train_cfg)
