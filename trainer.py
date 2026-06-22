"""
trainer.py
==========
Trainer — encapsulates the full training loop for all CoRD-Net experiments.

The Trainer owns the optimizer, scheduler, AMP scaler, checkpoint saving
and loading, metric accumulation, and logging.  Each experiment supplies
only a DRPNet model and a loss function; the loop is identical for E1–E8.

Usage
-----
    from config import get_config
    from models.drpnet import DRPNet
    from losses import MultiTaskLoss
    from trainer import Trainer

    cfg     = get_config("e8", data_root="/data/OAI", device="cuda")
    model   = DRPNet(cfg.model)
    loss_fn = MultiTaskLoss(cfg.training)
    trainer = Trainer(model, loss_fn, cfg)
    trainer.fit(train_loader, val_loader)
    # or resume:
    trainer.fit(train_loader, val_loader, resume="checkpoints/e8_best.pt")
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from config import Config
from losses import MultiTaskLoss
from metrics import evaluate
from utils import (
    count_parameters, count_all_parameters,
    log_parameter_summary, log_model_summary,
    save_checkpoint, load_checkpoint,
)

logger = logging.getLogger(__name__)


class Trainer:
    """
    Generic training loop for CoRD-Net experiments (E1–E8).

    Parameters
    ----------
    model:
        DRPNet instance (already created, not yet moved to device).
    loss_fn:
        MultiTaskLoss for E8, nn.CrossEntropyLoss for E1–E7.
    cfg:
        Top-level Config (model + training settings bundled).
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        cfg: Config,
    ) -> None:
        self.model   = model
        self.loss_fn = loss_fn
        self.cfg     = cfg
        self.tcfg    = cfg.training
        self.device  = torch.device(
            self.tcfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model.to(self.device)

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler    = GradScaler() if self.tcfg.amp else None

        self.epoch     = 0
        self.best_val  = float("inf")

        # Log model summary and parameter counts at startup
        log_model_summary(self.model, cfg.experiment)
        log_parameter_summary(self.model, cfg.experiment)

    # ── Optimizer / Scheduler ─────────────────────────────────────────────────

    def _build_optimizer(self) -> torch.optim.Optimizer:
        name = self.tcfg.optimizer.lower()
        if name == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.tcfg.learning_rate,
                weight_decay=self.tcfg.weight_decay,
            )
        if name == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.tcfg.learning_rate,
            )
        raise ValueError(
            f"Unknown optimizer '{self.tcfg.optimizer}'. Choose: adamw | adam"
        )

    def _build_scheduler(self):
        name = self.tcfg.scheduler.lower()
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.tcfg.epochs
            )
        if name == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        if name == "none":
            return None
        raise ValueError(
            f"Unknown scheduler '{self.tcfg.scheduler}'. Choose: cosine | step | none"
        )

    # ── Batch unpacking ───────────────────────────────────────────────────────

    @staticmethod
    def _unpack_batch(batch):
        """
        Accept both 1-crop and 3-crop batch formats.

        Format A (E1–E3):  ([global],                labels_dict)
        Format B (E4–E8):  ([global, medial, lateral], labels_dict)
        """
        crops, labels = batch
        global_crop  = crops[0]
        medial_crop  = crops[1] if len(crops) > 1 else None
        lateral_crop = crops[2] if len(crops) > 2 else None
        return global_crop, medial_crop, lateral_crop, labels

    def _to_device(self, global_crop, medial_crop, lateral_crop, labels):
        global_crop = global_crop.to(self.device, non_blocking=True)
        if medial_crop  is not None:
            medial_crop  = medial_crop.to(self.device, non_blocking=True)
        if lateral_crop is not None:
            lateral_crop = lateral_crop.to(self.device, non_blocking=True)
        labels = {k: v.to(self.device, non_blocking=True) for k, v in labels.items()}
        return global_crop, medial_crop, lateral_crop, labels

    # ── Loss computation ──────────────────────────────────────────────────────

    def _compute_loss(
        self,
        preds: Dict,
        labels: Dict,
    ) -> Dict[str, torch.Tensor]:
        """
        Route to MultiTaskLoss (E8) or simple CrossEntropyLoss (E1–E7).

        Prototype alignment loss is added whenever sim_logits is present
        (i.e. whenever PGR is active, E6+).

        Gracefully handles missing outputs — never assumes a key exists.
        """
        if isinstance(self.loss_fn, MultiTaskLoss):
            loss_dict = self.loss_fn(preds, labels)
        else:
            # E1–E7: single CE loss on primary KL logits
            if "logits" not in preds:
                raise KeyError(
                    "Model output is missing 'logits'. "
                    "Check DRPNet.forward() for this experiment."
                )
            ce = self.loss_fn(preds["logits"], labels["kl"])
            loss_dict = {"kl": ce, "total": ce}

        # Prototype alignment (E6+) — additive, not replacing CE
        if "sim_logits" in preds:
            w          = self.tcfg.loss_weights.get("proto", 0.3)
            proto_loss = w * F.cross_entropy(preds["sim_logits"], labels["kl"])
            loss_dict["proto"] = proto_loss
            loss_dict["total"] = loss_dict["total"] + proto_loss

        return loss_dict

    # ── Single training step ──────────────────────────────────────────────────

    def _step(self, batch) -> Dict[str, float]:
        """
        One forward → loss → backward → optimizer step.

        EMA prototype update is deferred until AFTER backward() to avoid
        corrupting the gradient graph.

        Returns
        -------
        Dict of scalar loss values (detached from graph).
        """
        global_crop, medial_crop, lateral_crop, labels = self._unpack_batch(batch)
        global_crop, medial_crop, lateral_crop, labels = \
            self._to_device(global_crop, medial_crop, lateral_crop, labels)

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=self.tcfg.amp):
            preds    = self.model(global_crop, medial_crop, lateral_crop)
            loss_kv  = self._compute_loss(preds, labels)

        total = loss_kv["total"]

        if self.scaler:
            self.scaler.scale(total).backward()
            if self.tcfg.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.tcfg.gradient_clip
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total.backward()
            if self.tcfg.gradient_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.tcfg.gradient_clip
                )
            self.optimizer.step()

        # ── EMA prototype update AFTER backward ───────────────────────────
        # drp_emb is stored on the model so the trainer never has to reach
        # inside PGRModule directly.
        if self.cfg.model.use_pgr and hasattr(self.model, "update_prototypes"):
            drp_emb = getattr(self.model, "_last_drp_emb", None)
            if drp_emb is not None:
                self.model.update_prototypes(drp_emb.detach(), labels["kl"])

        return {k: v.item() for k, v in loss_kv.items()}

    # ── Epoch loops ───────────────────────────────────────────────────────────

    def train_epoch(self, loader: Iterator) -> Dict[str, float]:
        """Run one full training epoch; return averaged loss dict."""
        self.model.train()
        totals: Dict[str, float] = {}
        n = 0
        for batch in loader:
            step_losses = self._step(batch)
            for k, v in step_losses.items():
                totals[k] = totals.get(k, 0.0) + v
            n += 1
        return {k: v / max(n, 1) for k, v in totals.items()}

    @torch.no_grad()
    def val_epoch(
        self, loader: Iterator
    ) -> Dict[str, float]:
        """
        Run one full validation epoch.

        Returns averaged losses AND classification metrics
        (accuracy, kappa, MAE) computed over the entire split.
        """
        self.model.eval()
        totals: Dict[str, float] = {}
        all_logits: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        n = 0

        for batch in loader:
            global_crop, medial_crop, lateral_crop, labels = self._unpack_batch(batch)
            global_crop, medial_crop, lateral_crop, labels = \
                self._to_device(global_crop, medial_crop, lateral_crop, labels)

            preds  = self.model(global_crop, medial_crop, lateral_crop)
            losses = self._compute_loss(preds, labels)

            for k, v in losses.items():
                totals[k] = totals.get(k, 0.0) + v.item()

            if "logits" in preds:
                all_logits.append(preds["logits"].cpu())
                all_labels.append(labels["kl"].cpu())
            n += 1

        result = {k: v / max(n, 1) for k, v in totals.items()}

        # Add classification metrics when logits are available
        if all_logits:
            logits_cat = torch.cat(all_logits, dim=0)
            labels_cat = torch.cat(all_labels, dim=0)
            metrics = evaluate(logits_cat, labels_cat, self.cfg.model.num_classes)
            result.update(metrics)

        return result

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def _save(
        self,
        epoch: int,
        train_losses: Dict,
        val_losses: Optional[Dict] = None,
        tag: str = "latest",
    ) -> None:
        state = {
            "experiment":           self.cfg.experiment,
            "epoch":                epoch,
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
                                    if self.scheduler else None,
            "best_val":             self.best_val,
            "train_losses":         train_losses,
            "val_losses":           val_losses or {},
        }
        save_checkpoint(
            state,
            self.tcfg.checkpoint_dir,
            filename=f"{self.cfg.experiment}_{tag}.pt",
        )

    def resume(self, checkpoint_path: str | Path) -> None:
        """
        Restore model, optimizer, scheduler, and epoch from a checkpoint.

        Call BEFORE trainer.fit() to continue interrupted training.

        Parameters
        ----------
        checkpoint_path: Path to a .pt file saved by this trainer.
        """
        ckpt = load_checkpoint(
            checkpoint_path,
            self.model,
            optimizer  = self.optimizer,
            scheduler  = self.scheduler,
            device     = self.device,
        )
        self.epoch    = ckpt.get("epoch", 0)
        self.best_val = ckpt.get("best_val", float("inf"))
        logger.info(
            "Resumed from epoch %d  (best_val=%.4f)",
            self.epoch, self.best_val
        )

    # ── Main fit loop ─────────────────────────────────────────────────────────

    def fit(
        self,
        train_loader: Iterator,
        val_loader: Optional[Iterator] = None,
        resume: Optional[str | Path] = None,
    ) -> None:
        """
        Run the full training loop for cfg.training.epochs epochs.

        Parameters
        ----------
        train_loader:
            Iterable of training batches from DataLoader.
        val_loader:
            Optional validation DataLoader.
        resume:
            Path to a checkpoint to resume from (restores epoch, optimizer,
            scheduler, and best_val before the loop starts).
        """
        if resume is not None:
            self.resume(resume)

        start_epoch = self.epoch + 1
        logger.info(
            "Training %s | epochs=%d→%d | device=%s",
            self.cfg.experiment.upper(),
            start_epoch,
            self.tcfg.epochs,
            self.device,
        )

        for epoch in range(start_epoch, self.tcfg.epochs + 1):
            self.epoch = epoch
            t0 = time.time()

            train_losses = self.train_epoch(train_loader)
            elapsed = time.time() - t0

            log_parts = [
                f"epoch={epoch}/{self.tcfg.epochs}",
                f"time={elapsed:.1f}s",
                f"lr={self.optimizer.param_groups[0]['lr']:.2e}",
            ]
            log_parts += [f"train/{k}={v:.4f}" for k, v in train_losses.items()]

            if val_loader is not None:
                val_metrics = self.val_epoch(val_loader)
                log_parts += [f"val/{k}={v:.4f}" for k, v in val_metrics.items()]
                val_total = val_metrics.get("total", float("inf"))
                if val_total < self.best_val:
                    self.best_val = val_total
                    self._save(epoch, train_losses, val_metrics, tag="best")

            logger.info(" | ".join(log_parts))

            if self.scheduler is not None:
                self.scheduler.step()

            if epoch % self.tcfg.save_every == 0:
                self._save(epoch, train_losses, tag=f"epoch{epoch:04d}")

        # Always save the final state
        self._save(self.epoch, train_losses, tag="final")
        logger.info("Training complete.")

    # ── Synthetic stub (run_experiment.py only) ───────────────────────────────

    def stub_fit(self, steps: int = 3, image_size: int = 112) -> Dict[str, float]:
        """
        Run *steps* synthetic mini-batch passes for integration testing.

        Not used in real training.  Returns the last step's loss dict.
        """
        from utils import make_labels_stub

        B        = self.tcfg.batch_size
        K        = self.cfg.model.num_classes
        use_3crop = self.cfg.model.use_compartment

        self.model.train()
        last_losses: Dict[str, float] = {}

        for step in range(1, steps + 1):
            self.optimizer.zero_grad(set_to_none=True)

            g = torch.randn(B, 3, image_size, image_size, device=self.device)
            m = torch.randn(B, 3, image_size, image_size, device=self.device) \
                if use_3crop else None
            l_ = torch.randn(B, 3, image_size, image_size, device=self.device) \
                if use_3crop else None
            labels = make_labels_stub(B, K, self.device)

            preds     = self.model(g, m, l_)
            loss_dict = self._compute_loss(preds, labels)
            loss_dict["total"].backward()

            if self.tcfg.gradient_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.tcfg.gradient_clip
                )
            self.optimizer.step()

            # EMA prototype update after backward
            if self.cfg.model.use_pgr and hasattr(self.model, "update_prototypes"):
                drp_emb = getattr(self.model, "_last_drp_emb", None)
                if drp_emb is not None:
                    self.model.update_prototypes(drp_emb.detach(), labels["kl"])

            last_losses = {k: v.item() for k, v in loss_dict.items()}
            parts = " | ".join(f"{k}={v:.4f}" for k, v in last_losses.items())
            logger.info("  step %d/%d  %s", step, steps, parts)

        return last_losses
