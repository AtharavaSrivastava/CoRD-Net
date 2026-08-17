"""
trainer.py
==========
Trainer — training loop for all CoRD-Net experiments.

IMPROVEMENTS (beyond the bug fixes from the previous session)
-------------------------------------------------------------
1. Warmup + CosineAnnealingLR actually implemented
   The previous version had a `warmup_epochs` field in config that was
   silently ignored.  Now uses torch.optim.lr_scheduler.SequentialLR
   combining a LinearLR warmup and CosineAnnealingLR.

2. STN backbone freeze/unfreeze
   When E2 is active and KneeLocalizer has a pretrained backbone,
   the trainer calls localizer.freeze_backbone() at startup and
   localizer.unfreeze_backbone() after freeze_epochs.

3. STN translation regularisation
   When E2 is active, adds get_theta_reg_loss() * 0.01 to the total
   loss to prevent the STN from translating crops off-image.

4. gradient accumulation (configurable via cfg.training.grad_accum_steps)
   Allows effective batch sizes larger than GPU memory permits without
   changing the optimiser update frequency.

5. All bug fixes from the previous session are preserved.
"""

from __future__ import annotations

import os
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.amp import autocast

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
    def __init__(self, model: nn.Module, loss_fn: nn.Module, cfg: Config) -> None:
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
        self.grad_accum = getattr(self.tcfg, "grad_accum_steps", 1)

        self.epoch    = 0
        self.best_qwk = -1.0

        self.history: Dict[str, List] = {
            "epoch": [], "train_loss": [], "val_loss": [],
            "train_accuracy": [], "val_accuracy": [],
            "val_macro_f1": [], "val_qwk": [], "val_mae": [],
            "learning_rate": [],
        }

        # IMPROVEMENT 2: freeze STN backbone at startup if pretrained
        if cfg.model.use_stn:
            loc = getattr(model, "localizer", None)
            if loc is not None and hasattr(loc, "freeze_backbone"):
                loc.freeze_backbone()
                logger.info("STN backbone frozen for first %d epochs",
                            getattr(loc, "freeze_epochs", 5))

        log_model_summary(self.model, cfg.experiment)
        log_parameter_summary(self.model, cfg.experiment)

    # ── Optimizer ─────────────────────────────────────────────────────────────

    def _build_optimizer(self) -> torch.optim.Optimizer:
        name = self.tcfg.optimizer.lower()
        lr   = self.tcfg.learning_rate
        wd   = self.tcfg.weight_decay

        # Collect parameter groups; backbone gets a lower LR
        groups = []

        # STN: differential LR if the localizer supports it
        if self.cfg.model.use_stn:
            loc = getattr(self.model, "localizer", None)
            if loc is not None and hasattr(loc, "named_parameter_groups"):
                groups.extend(loc.named_parameter_groups(lr))

        # Everything else at full LR
        stn_param_ids = {id(p) for g in groups for p in g["params"]}
        rest = [p for p in self.model.parameters()
                if id(p) not in stn_param_ids and p.requires_grad]
        groups.append({"params": rest, "lr": lr, "name": "main"})

        if name == "adamw":
            return torch.optim.AdamW(groups, lr=lr, weight_decay=wd)
        if name == "adam":
            return torch.optim.Adam(groups, lr=lr)
        raise ValueError(f"Unknown optimizer '{self.tcfg.optimizer}'")

    # ── Scheduler (IMPROVEMENT 1: warmup actually implemented) ───────────────

    def _build_scheduler(self):
        name         = self.tcfg.scheduler.lower()
        warmup       = getattr(self.tcfg, "warmup_epochs", 0)
        total_epochs = self.tcfg.epochs

        if name == "none":
            return None

        if name == "cosine":
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max   = max(total_epochs - warmup, 1),
                eta_min = self.tcfg.learning_rate * 0.01,
            )
            if warmup > 0:
                linear = torch.optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor = 0.1,
                    end_factor   = 1.0,
                    total_iters  = warmup,
                )
                return torch.optim.lr_scheduler.SequentialLR(
                    self.optimizer,
                    schedulers  = [linear, cosine],
                    milestones  = [warmup],
                )
            return cosine

        if name == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )

        raise ValueError(f"Unknown scheduler '{self.tcfg.scheduler}'")

    # ── Batch helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _unpack_batch(batch):
        crops, labels = batch
        return crops[0], labels

    def _to_device(self, global_crop, labels):
        global_crop = global_crop.to(self.device, non_blocking=True)
        labels = {k: v.to(self.device, non_blocking=True) for k, v in labels.items()}
        return global_crop, labels

    # ── Loss ──────────────────────────────────────────────────────────────────

    def _compute_loss(self, preds, labels):
        if isinstance(self.loss_fn, MultiTaskLoss):
            loss_dict = self.loss_fn(preds, labels)
        else:
            if "logits" not in preds:
                raise KeyError("Model output missing 'logits'.")
            ce = self.loss_fn(preds["logits"], labels["kl"])
            loss_dict = {"kl": ce, "total": ce}

        if "sim_logits" in preds:
            w          = self.tcfg.loss_weights.get("proto", 0.3)
            proto_loss = w * F.cross_entropy(preds["sim_logits"], labels["kl"])
            loss_dict["proto"] = proto_loss
            loss_dict["total"] = loss_dict["total"] + proto_loss

        # IMPROVEMENT 3: STN translation regularisation
        if "theta" in preds:
            loc = getattr(self.model, "localizer", None)
            if loc is not None and hasattr(loc, "get_theta_reg_loss"):
                theta_reg = 0.01 * loc.get_theta_reg_loss(preds["theta"])
                loss_dict["theta_reg"] = theta_reg
                loss_dict["total"]     = loss_dict["total"] + theta_reg

        return loss_dict

    # ── Single training step ──────────────────────────────────────────────────

    def _step(self, batch, accumulate: bool = False) -> Dict:
        global_crop, labels = self._unpack_batch(batch)
        global_crop, labels = self._to_device(global_crop, labels)

        device_type = "cuda" if self.device.type == "cuda" else "cpu"

        with autocast(device_type, enabled=self.tcfg.amp):
            preds   = self.model(global_crop)
            if (hasattr(self.model, "compartment") and
                    self.model.compartment is not None and
                    hasattr(self.model.compartment, "debug_stats")):
                for k, v in self.model.compartment.debug_stats.items():
                    self.debug[k].append(v)
            loss_kv = self._compute_loss(preds, labels)

        total   = loss_kv["total"] / self.grad_accum
        correct = (preds["logits"].argmax(1) == labels["kl"]).sum().item()
        count   = labels["kl"].size(0)

        if self.scaler:
            self.scaler.scale(total).backward()
        else:
            total.backward()

        if not accumulate:
            if self.scaler:
                if self.tcfg.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(),
                                             self.tcfg.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if self.tcfg.gradient_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(),
                                             self.tcfg.gradient_clip)
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        if self.cfg.model.use_pgr and hasattr(self.model, "update_prototypes"):
            drp_emb = getattr(self.model, "_last_drp_emb", None)
            if drp_emb is not None:
                self.model.update_prototypes(drp_emb.detach(), labels["kl"])

        out = {k: v.item() * self.grad_accum for k, v in loss_kv.items()}
        out["correct"] = correct
        out["count"]   = count
        return out

    # ── Epoch loops ───────────────────────────────────────────────────────────

    def train_epoch(self, loader) -> Dict[str, float]:
        self.model.train()
        self.debug  = defaultdict(list)
        totals: Dict[str, float] = {}
        correct = count = n = 0

        self.optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(loader):
            accumulate = ((i + 1) % self.grad_accum != 0)
            step = self._step(batch, accumulate=accumulate)
            correct += step.pop("correct")
            count   += step.pop("count")
            for k, v in step.items():
                totals[k] = totals.get(k, 0.0) + v
            n += 1

        result = {k: v / max(n, 1) for k, v in totals.items()}
        result["accuracy"] = correct / max(count, 1)

        if self.debug:
            for k, values in self.debug.items():
                logger.debug("  %s: %.4f ± %.4f", k, np.mean(values), np.std(values))

        return result

    @torch.no_grad()
    def val_epoch(self, loader) -> Dict[str, float]:
        self.model.eval()
        totals: Dict[str, float] = {}
        all_logits: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []
        pred_hist = torch.zeros(self.cfg.model.num_classes, dtype=torch.long)
        n = 0

        for batch in loader:
            global_crop, labels = self._unpack_batch(batch)
            global_crop, labels = self._to_device(global_crop, labels)

            preds  = self.model(global_crop)
            logits = preds["logits"]
            losses = self._compute_loss(preds, labels)

            for k, v in losses.items():
                totals[k] = totals.get(k, 0.0) + v.item()

            pred = logits.argmax(dim=1).cpu()
            pred_hist += torch.bincount(pred, minlength=self.cfg.model.num_classes)
            all_logits.append(logits.cpu())
            all_labels.append(labels["kl"].cpu())
            n += 1

        result = {k: v / max(n, 1) for k, v in totals.items()}

        if all_logits:
            logits_cat = torch.cat(all_logits, dim=0)
            labels_cat = torch.cat(all_labels, dim=0)
            result.update(evaluate(logits_cat, labels_cat, self.cfg.model.num_classes))
            result["macro_f1"] = compute_all_metrics(
                logits_cat, labels_cat, self.cfg.model.num_classes
            )["macro_f1"]

            label_hist = torch.bincount(labels_cat, minlength=self.cfg.model.num_classes)
            logger.info("Val pred dist : %s", pred_hist.tolist())
            logger.info("Val true dist : %s", label_hist.tolist())

        return result

    @torch.no_grad()
    def collect_logits(self, loader) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        all_logits, all_labels = [], []
        for batch in loader:
            global_crop, labels = self._unpack_batch(batch)
            global_crop = global_crop.to(self.device, non_blocking=True)
            preds = self.model(global_crop)
            if "logits" in preds:
                all_logits.append(preds["logits"].cpu())
                all_labels.append(labels["kl"])
        if not all_logits:
            return (np.zeros((0, self.cfg.model.num_classes), dtype=np.float32),
                    np.zeros(0, dtype=np.int64))
        return (_to_numpy(torch.cat(all_logits)),
                _to_numpy(torch.cat(all_labels)).astype(int))

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def _save(self, epoch, train_losses, val_losses=None, tag="latest") -> None:
        sched_state = None
        if self.scheduler is not None:
            sched_state = self.scheduler.state_dict()
        save_checkpoint({
            "experiment":           self.cfg.experiment,
            "epoch":                epoch,
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": sched_state,
            "best_qwk":             self.best_qwk,
            "train_losses":         train_losses,
            "val_losses":           val_losses or {},
            "history":              self.history,
        }, self.tcfg.checkpoint_dir, filename=f"{self.cfg.experiment}_{tag}.pt")

    def resume(self, checkpoint_path) -> None:
        ckpt = load_checkpoint(checkpoint_path, self.model,
                               optimizer=self.optimizer, scheduler=self.scheduler,
                               device=self.device)
        self.epoch    = ckpt.get("epoch", 0)
        self.best_qwk = ckpt.get("best_qwk", -1.0)
        if "history" in ckpt:
            self.history = ckpt["history"]
        logger.info("Resumed from epoch %d  (best_qwk=%.4f)", self.epoch, self.best_qwk)

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, train_loader, val_loader=None, test_loader=None,
            resume=None, results_dir="results") -> None:
        if resume is not None:
            self.resume(resume)

        start_epoch      = self.epoch + 1
        epochs_no_improve = 0
        train_losses: Dict[str, float] = {}

        logger.info("Training %s | epochs %d→%d | device=%s",
                    self.cfg.experiment.upper(), start_epoch,
                    self.tcfg.epochs, self.device)

        for epoch in range(start_epoch, self.tcfg.epochs + 1):
            self.epoch = epoch

            # IMPROVEMENT 2: unfreeze STN backbone after freeze_epochs
            if self.cfg.model.use_stn:
                loc = getattr(self.model, "localizer", None)
                if loc is not None and hasattr(loc, "freeze_epochs"):
                    if epoch == loc.freeze_epochs + 1:
                        loc.unfreeze_backbone()
                        logger.info("Epoch %d: STN backbone unfrozen", epoch)

            t0           = time.time()
            train_losses = self.train_epoch(train_loader)
            elapsed      = time.time() - t0
            current_lr   = self.optimizer.param_groups[-1]["lr"]

            log_parts  = [f"epoch={epoch}/{self.tcfg.epochs}",
                          f"time={elapsed:.1f}s", f"lr={current_lr:.2e}"]
            log_parts += [f"train/{k}={v:.4f}" for k, v in train_losses.items()]

            val_metrics: Dict[str, float] = {}
            if val_loader is not None:
                val_metrics = self.val_epoch(val_loader)
                log_parts  += [f"val/{k}={v:.4f}" for k, v in val_metrics.items()]

            if val_metrics.get("kappa", -1.0) > self.best_qwk:
                self.best_qwk     = val_metrics["kappa"]
                epochs_no_improve = 0
                self._save(epoch, train_losses, val_metrics, tag="best")
            else:
                epochs_no_improve += 1

            logger.info(" | ".join(log_parts))

            if (self.tcfg.patience is not None
                    and epochs_no_improve >= self.tcfg.patience):
                logger.info("Early stopping at epoch %d", epoch)
                break

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

        self._save(self.epoch, train_losses, tag="final")
        logger.info("Training complete.")

        best_ckpt = os.path.join(self.tcfg.checkpoint_dir,
                                 f"{self.cfg.experiment}_best.pt")
        if os.path.exists(best_ckpt):
            load_checkpoint(best_ckpt, self.model, device=self.device)

        self._run_reporting(train_loader, val_loader, test_loader, results_dir)

    def _run_reporting(self, train_loader, val_loader, test_loader, results_dir):
        from reporting import ResultsWriter, generate_all_reports
        if val_loader is None:
            return
        writer = ResultsWriter(self.cfg.experiment, results_dir)
        val_logits,   val_labels   = self.collect_logits(val_loader)
        train_logits = train_labels = None
        if train_loader is not None:
            train_logits, train_labels = self.collect_logits(train_loader)
        test_logits = test_labels = None
        if test_loader is not None:
            test_logits, test_labels = self.collect_logits(test_loader)
        generate_all_reports(
            writer=writer, history=self.history,
            train_logits=train_logits, train_labels=train_labels,
            val_logits=val_logits,     val_labels=val_labels,
            test_logits=test_logits,   test_labels=test_labels,
            num_classes=self.cfg.model.num_classes,
            parameters=count_parameters(self.model),
            results_dir=results_dir,
        )

    # ── Stub fit ──────────────────────────────────────────────────────────────

    def stub_fit(self, steps=3, image_size=112) -> Dict[str, float]:
        from utils import make_labels_stub
        B = self.tcfg.batch_size; K = self.cfg.model.num_classes
        self.model.train(); self.debug = defaultdict(list)
        last: Dict[str, float] = {}
        self.optimizer.zero_grad(set_to_none=True)
        for step in range(1, steps + 1):
            g = torch.randn(B, 3, image_size, image_size, device=self.device)
            labels = make_labels_stub(B, K, self.device)
            preds  = self.model(g)
            ld     = self._compute_loss(preds, labels)
            ld["total"].backward()
            if self.tcfg.gradient_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.tcfg.gradient_clip)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            if self.cfg.model.use_pgr and hasattr(self.model, "update_prototypes"):
                drp_emb = getattr(self.model, "_last_drp_emb", None)
                if drp_emb is not None:
                    self.model.update_prototypes(drp_emb.detach(), labels["kl"])
            last = {k: v.item() for k, v in ld.items()}
            logger.info("  step %d/%d  %s", step, steps,
                        " | ".join(f"{k}={v:.4f}" for k, v in last.items()))
        return last
