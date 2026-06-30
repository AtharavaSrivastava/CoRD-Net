"""
trainer.py
==========
Trainer — encapsulates the full training loop for all CoRD-Net experiments.

Changes from previous version
------------------------------
* Accumulates per-epoch training history (losses, accuracy, macro-F1,
  QWK, MAE, learning rate).
* Collects full logit tensors over val and (optionally) train sets.
* Calls reporting.generate_all_reports() after fit() completes.
* Adds collect_logits() for running a loader through the model without
  updating weights (used by train.py to get train-set metrics at the end).

Architecture, optimizer, scheduler, losses, and checkpoint logic are
unchanged.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from collections import defaultdict
import numpy as np

from config import Config
from losses import MultiTaskLoss
from metrics import evaluate, compute_all_metrics, get_predictions, _to_numpy
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
        DRPNet instance.
    loss_fn:
        MultiTaskLoss for E8, nn.CrossEntropyLoss for E1–E7.
    cfg:
        Top-level Config (model + training settings bundled).
    """

    def __init__(
        self,
        model:   nn.Module,
        loss_fn: nn.Module,
        cfg:     Config,
    ) -> None:
        self.model   = model
        print("\n========== CLASSIFIER INITIALIZATION ==========")

        for name, p in self.model.named_parameters():
            if "classifier" in name:
                print(name)
                print("mean:", p.mean().item())
                print("std :", p.std().item())
            if "classifier.bias" in name:
                print("Classifier bias:", p.detach().cpu())

        print("===============================================\n")
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

        self.epoch    = 0
        self.best_qwk = -1.0

        # ── Per-epoch history (populated during fit) ──────────────────────
        self.history: Dict[str, List] = {
            "epoch":         [],
            "train_loss":    [],
            "val_loss":      [],
            "train_accuracy": [],
            "val_accuracy":  [],
            "val_macro_f1":  [],
            "val_qwk":       [],
            "val_mae":       [],
            "learning_rate": [],
        }

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

        Format A (E1–E3):  ([global],                  labels_dict)
        Format B (E4–E8):  ([global, medial, lateral],  labels_dict)
        """
        crops, labels = batch
        global_crop = batch[0][0]
        labels = batch[1]
        return global_crop, labels

    def _to_device(self, global_crop, labels):
        global_crop = global_crop.to(self.device, non_blocking=True)
        labels = {k: v.to(self.device, non_blocking=True) for k, v in labels.items()}
        return global_crop, labels

    # ── Loss computation ──────────────────────────────────────────────────────

    def _compute_loss(
        self,
        preds:  Dict,
        labels: Dict,
    ) -> Dict[str, torch.Tensor]:
        """
        Route to MultiTaskLoss (E8) or CrossEntropyLoss (E1–E7).
        Prototype alignment is added whenever sim_logits is present.
        """
        if isinstance(self.loss_fn, MultiTaskLoss):
            loss_dict = self.loss_fn(preds, labels)
        else:
            if "logits" not in preds:
                raise KeyError(
                    "Model output missing 'logits'. "
                    "Check DRPNet.forward() for this experiment."
                )
            ce = self.loss_fn(preds["logits"], labels["kl"])
            loss_dict = {"kl": ce, "total": ce}

        if "sim_logits" in preds:
            w          = self.tcfg.loss_weights.get("proto", 0.3)
            proto_loss = w * F.cross_entropy(preds["sim_logits"], labels["kl"])
            loss_dict["proto"] = proto_loss
            loss_dict["total"] = loss_dict["total"] + proto_loss

        return loss_dict

    # ── Single training step ──────────────────────────────────────────────────

    def _step(self, batch) -> Dict[str, float]:
        """One forward → loss → backward → optimizer step."""
        global_crop, labels = self._unpack_batch(batch)
        global_crop, labels = \
            self._to_device(global_crop, labels)

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=self.tcfg.amp):
            preds   = self.model(global_crop)
            # ---------- DEBUG COLLECTION ----------
            if hasattr(self.model, "debug_stats"):
                for k, v in self.model.debug_stats.items():
                    self.debug[k].append(v)
            loss_kv = self._compute_loss(preds, labels)
            logits = preds["logits"]
            pred_cls = logits.argmax(dim=1)
            correct = (pred_cls == labels["kl"]).sum().item()
            count = labels["kl"].size(0)
        total = loss_kv["total"]

        if self.scaler:
            self.scaler.scale(total).backward()
            classifier = self.model.classifier
            if classifier is None:
                classifier = self.model.heads["h1"].classifier   # adjust if needed

            before = classifier.weight.detach().clone()
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

        # EMA prototype update AFTER backward
        if self.cfg.model.use_pgr and hasattr(self.model, "update_prototypes"):
            drp_emb = getattr(self.model, "_last_drp_emb", None)
            if drp_emb is not None:
                self.model.update_prototypes(drp_emb.detach(), labels["kl"])
        out = {k: v.item() for k, v in loss_kv.items()}
        out["correct"] = correct
        out["count"] = count
    
        # ADD THIS
        out["logits_mean"] = preds["logits"].detach().mean(dim=0).cpu()

        out["correct"] = correct
        out["count"] = count
        return out

    # ── Epoch loops ───────────────────────────────────────────────────────────
    def train_epoch(self, loader):
        """Run one full training epoch; return averaged losses + accuracy."""
        self.model.train()

        self.debug = defaultdict(list)

        totals = {}
        n = 0

        correct = 0
        count = 0

        # ADD HERE
        logit_sum = torch.zeros(self.cfg.model.num_classes)
        num_batches = 0

        for batch in loader:
            step = self._step(batch)

            correct += step.pop("correct")
            count += step.pop("count")

            # ADD THESE TWO LINES
            logit_sum += step.pop("logits_mean")
            num_batches += 1

            for k, v in step.items():
                totals[k] = totals.get(k, 0.0) + v

            n += 1

        result = {k: v / max(n, 1) for k, v in totals.items()}
        result["accuracy"] = correct / max(count, 1)

        print("\n" + "=" * 60)
        print("TRAIN EPOCH DEBUG")
        print("=" * 60)

        print("Average logits:", logit_sum / num_batches)

        for k, values in self.debug.items():
            print(f"{k:15s}: {np.mean(values):.4f} ± {np.std(values):.4f}")

        print("=" * 60 + "\n")

        return result


    @torch.no_grad()
    def val_epoch(self, loader: Iterator) -> Dict[str, float]:
        """
        Full validation pass.
        Returns averaged losses + accuracy / kappa / mae for the
        per-epoch log.  Does NOT store logits — that is done by
        collect_logits() when full metrics are needed.
        """
        self.model.eval()

        totals: Dict[str, float] = {}
        all_logits: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []

        n = 0
        pred_hist = torch.zeros(self.cfg.model.num_classes, dtype=torch.long)

        for batch in loader:
            global_crop, labels = \
                self._unpack_batch(batch)
            global_crop, labels = \
                self._to_device(global_crop, labels)

            preds = self.model(global_crop)

            logits = preds["logits"]          # <-- ADD THIS

            losses = self._compute_loss(preds, labels)

            for k, v in losses.items():
                totals[k] = totals.get(k, 0.0) + v.item()

            pred = logits.argmax(dim=1).cpu()
            pred_hist += torch.bincount(
                pred,
                minlength=self.cfg.model.num_classes,
            )

            if n == 0:
                print("First batch prediction counts:",
                    torch.bincount(
                        pred,
                        minlength=self.cfg.model.num_classes
                    ))

            if "logits" in preds:
                all_logits.append(logits.cpu())
                all_labels.append(labels["kl"].cpu())

            n += 1

        result = {k: v / max(n, 1) for k, v in totals.items()}

        if all_logits:
            logits_cat = torch.cat(all_logits, dim=0)
            labels_cat = torch.cat(all_labels, dim=0)

            m = evaluate(
                logits_cat,
                labels_cat,
                self.cfg.model.num_classes,
            )
            result.update(m)

            full = compute_all_metrics(
                logits_cat,
                labels_cat,
                self.cfg.model.num_classes,
            )
            result["macro_f1"] = full["macro_f1"]

        label_hist = torch.bincount(
            labels_cat,
            minlength=self.cfg.model.num_classes,
        )

        print("\n========== VALIDATION HISTOGRAM ==========")
        print("Pred :", pred_hist.tolist())
        print("True :", label_hist.tolist())
        print("==========================================\n")

        return result

    # ── Logit collection (used for final reporting) ───────────────────────────

    @torch.no_grad()
    def collect_logits(
        self, loader: Iterator
    ) -> tuple[np.ndarray, np.ndarray]:

        self.model.eval()

        all_logits = []
        all_labels = []

        for batch in loader:
            global_crop, labels = self._unpack_batch(batch)

            global_crop = global_crop.to(
                self.device,
                non_blocking=True,
            )

            preds = self.model(global_crop)

            if "logits" in preds:
                all_logits.append(preds["logits"].cpu())
                all_labels.append(labels["kl"])

        if not all_logits:
            empty = np.zeros((0, self.cfg.model.num_classes), dtype=np.float32)
            return empty, np.zeros(0, dtype=np.int64)

        return (
            _to_numpy(torch.cat(all_logits, dim=0)),
            _to_numpy(torch.cat(all_labels, dim=0)).astype(int),
        )

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def _save(
        self,
        epoch:        int,
        train_losses: Dict,
        val_losses:   Optional[Dict] = None,
        tag:          str = "latest",
    ) -> None:
        state = {
            "experiment":           self.cfg.experiment,
            "epoch":                epoch,
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
                                    if self.scheduler else None,
            "best_qwk":             self.best_qwk,
            "train_losses":         train_losses,
            "val_losses":           val_losses or {},
            "history":              self.history,
        }
        save_checkpoint(
            state,
            self.tcfg.checkpoint_dir,
            filename=f"{self.cfg.experiment}_{tag}.pt",
        )

    def resume(self, checkpoint_path: str | Path) -> None:
        """Restore model, optimizer, scheduler, epoch, and history."""
        ckpt = load_checkpoint(
            checkpoint_path,
            self.model,
            optimizer = self.optimizer,
            scheduler = self.scheduler,
            device    = self.device,
        )
        self.epoch    = ckpt.get("epoch", 0)
        self.best_qwk = ckpt.get("best_qwk", -1.0)
        if "history" in ckpt:
            self.history = ckpt["history"]
        logger.info(
            "Resumed from epoch %d  (best_val=%.4f)",
            self.epoch, self.best_val,
        )

    # ── Main fit loop ─────────────────────────────────────────────────────────

    def fit(
        self,
        train_loader:  Iterator,
        val_loader:    Optional[Iterator] = None,
        test_loader:   Optional[Iterator] = None,
        resume:        Optional[str | Path] = None,
        results_dir:   str = "results",
    ) -> None:
        """
        Run the full training loop, then generate all reports.

        Parameters
        ----------
        train_loader:  DataLoader for the training split.
        val_loader:    DataLoader for the validation split.
        test_loader:   DataLoader for the test split (used in final reporting).
        resume:        Path to checkpoint to resume from.
        results_dir:   Root results directory for reporting output.
        """
        if resume is not None:
            self.resume(resume)

        start_epoch = self.epoch + 1
        logger.info(
            "Training %s | epochs %d→%d | device=%s",
            self.cfg.experiment.upper(),
            start_epoch,
            self.tcfg.epochs,
            self.device,
        )

        train_losses: Dict[str, float] = {}

        for epoch in range(start_epoch, self.tcfg.epochs + 1):
            self.epoch = epoch
            t0 = time.time()

            train_losses = self.train_epoch(train_loader)
            elapsed      = time.time() - t0
            current_lr   = self.optimizer.param_groups[0]["lr"]

            log_parts = [
                f"epoch={epoch}/{self.tcfg.epochs}",
                f"time={elapsed:.1f}s",
                f"lr={current_lr:.2e}",
            ]
            log_parts += [f"train/{k}={v:.4f}" for k, v in train_losses.items()]

            val_metrics: Dict[str, float] = {}
            if val_loader is not None:
                val_metrics = self.val_epoch(val_loader)
                log_parts  += [f"val/{k}={v:.4f}" for k, v in val_metrics.items()]
                val_total   = val_metrics.get("total", float("inf"))
            if val_metrics.get("kappa",-1.0) > self.best_qwk:
                self.best_qwk = val_metrics["kappa"]
                self._save(epoch, train_losses, val_metrics, tag="best")

            logger.info(" | ".join(log_parts))

            # ── Record history ──────────────────────────────────────────
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_losses.get("total", float("nan")))
            self.history["val_loss"].append(val_metrics.get("total",    float("nan")))
            self.history["train_accuracy"].append(train_losses.get("accuracy", float("nan")))
            self.history["val_accuracy"].append(val_metrics.get("accuracy",  float("nan")))
            self.history["val_macro_f1"].append( val_metrics.get("macro_f1", float("nan")))
            self.history["val_qwk"].append(      val_metrics.get("kappa",    float("nan")))
            self.history["val_mae"].append(      val_metrics.get("mae",      float("nan")))
            self.history["learning_rate"].append(current_lr)

            if self.scheduler is not None:
                self.scheduler.step()

            if epoch % self.tcfg.save_every == 0:
                self._save(epoch, train_losses, tag=f"epoch{epoch:04d}")

        # Final checkpoint
        self._save(self.epoch, train_losses, tag="final")
        logger.info("Training complete. Generating reports …")

        # ── Post-training reporting ─────────────────────────────────────────
        self._run_reporting(
            train_loader = train_loader,
            val_loader   = val_loader,
            test_loader  = test_loader,
            results_dir  = results_dir,
        )

    def _run_reporting(
        self,
        train_loader: Optional[Iterator],
        val_loader:   Optional[Iterator],
        test_loader:  Optional[Iterator],
        results_dir:  str,
    ) -> None:
        """Collect final logits and call generate_all_reports()."""
        from reporting import ResultsWriter, generate_all_reports
        from utils import count_parameters

        if val_loader is None:
            logger.warning(
                "No val_loader provided — skipping reporting."
            )
            return

        writer = ResultsWriter(self.cfg.experiment, results_dir)

        logger.info("Collecting val logits …")
        val_logits, val_labels = self.collect_logits(val_loader)

        train_logits = train_labels = None
        if train_loader is not None:
            logger.info("Collecting train logits …")
            train_logits, train_labels = self.collect_logits(train_loader)


        test_logits = test_labels = None
        if test_loader is not None:
            logger.info("Collecting test logits …")
            test_logits, test_labels = self.collect_logits(test_loader)
        generate_all_reports(
            writer        = writer,
            history       = self.history,
            train_logits  = train_logits,
            train_labels  = train_labels,
            val_logits    = val_logits,
            val_labels    = val_labels,
            test_logits   = test_logits,
            test_labels   = test_labels,
            num_classes   = self.cfg.model.num_classes,
            parameters    = count_parameters(self.model),
            results_dir   = results_dir,
        )

    # ── Synthetic stub (run_experiment.py only) ───────────────────────────────

    def stub_fit(self, steps: int = 3, image_size: int = 112) -> Dict[str, float]:
        """
        Run *steps* synthetic mini-batch passes for integration testing.
        Not used in real training.
        """
        from utils import make_labels_stub

        B         = self.tcfg.batch_size
        K         = self.cfg.model.num_classes

        self.model.train()
        last_losses: Dict[str, float] = {}

        for step in range(1, steps + 1):
            self.optimizer.zero_grad(set_to_none=True)

            g  = torch.randn(B, 3, image_size, image_size, device=self.device)
            labels = make_labels_stub(B, K, self.device)
            preds     = self.model(g)
            loss_dict = self._compute_loss(preds, labels)
            loss_dict["total"].backward()

            if self.tcfg.gradient_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.tcfg.gradient_clip
                )
            self.optimizer.step()

            if self.cfg.model.use_pgr and hasattr(self.model, "update_prototypes"):
                drp_emb = getattr(self.model, "_last_drp_emb", None)
                if drp_emb is not None:
                    self.model.update_prototypes(drp_emb.detach(), labels["kl"])

            last_losses = {k: v.item() for k, v in loss_dict.items()}
            parts = " | ".join(f"{k}={v:.4f}" for k, v in last_losses.items())
            logger.info("  step %d/%d  %s", step, steps, parts)

        return last_losses
