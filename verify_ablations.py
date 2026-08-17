"""
verify_ablations.py
===================
Ablation integrity checker for CoRD-Net (Requirement 16).

For each experiment E1–E8, this script:
  1. Instantiates DRPNet with the experiment's config
  2. Runs one synthetic forward pass
  3. Verifies exactly which modules are active
  4. Counts parameters
  5. Checks output dictionary keys
  6. Checks expected loss keys
  7. Verifies gradient flow through every active module

Prints a summary table at the end.

Usage
-----
    python verify_ablations.py           # all experiments
    python verify_ablations.py --exp e5  # single experiment
    python verify_ablations.py --verbose # show per-param grad status
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import get_config, EXPERIMENT_NAMES
from losses import MultiTaskLoss
from models.drpnet import DRPNet
from utils import count_parameters, count_all_parameters, make_labels_stub

logging.basicConfig(level=logging.WARNING)   # suppress trainer logs during checks
logger = logging.getLogger("verify")


# ──────────────────────────────────────────────────────────────────────────────
# Expected spec per experiment
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ExpectedSpec:
    active_flags: Set[str]
    output_keys_always: Set[str]          # always in out dict
    output_keys_conditional: Set[str]     # present only when flag active
    loss_keys: Set[str]                   # expected in loss dict
    grad_modules: List[str]               # attribute paths that must have grad


_SPECS: Dict[str, ExpectedSpec] = {
    "e1": ExpectedSpec(
        active_flags            = set(),
        output_keys_always      = {"logits"},
        output_keys_conditional = set(),
        loss_keys               = {"kl", "total"},
        grad_modules            = ["backbone_features", "classifier"],
    ),
    "e2": ExpectedSpec(
        active_flags            = {"use_stn"},
        output_keys_always      = {"logits", "theta"},
        output_keys_conditional = set(),
        loss_keys               = {"kl", "total"},
        grad_modules            = ["localizer", "backbone_features", "classifier"],
    ),
    "e3": ExpectedSpec(
        active_flags            = {"use_stn", "use_dual_intensity"},
        output_keys_always      = {"logits", "theta"},
        output_keys_conditional = set(),
        loss_keys               = {"kl", "total"},
        grad_modules            = ["localizer", "stem", "backbone_features", "classifier"],
    ),
    "e4": ExpectedSpec(
        active_flags            = {"use_stn", "use_dual_intensity", "use_compartment"},
        output_keys_always      = {"logits", "theta"},
        output_keys_conditional = set(),
        loss_keys               = {"kl", "total"},
        grad_modules            = ["localizer", "stem", "backbone_features",
                                   "compartment", "classifier"],
    ),
    "e5": ExpectedSpec(
        active_flags            = {"use_stn", "use_dual_intensity",
                                   "use_compartment", "use_drp"},
        output_keys_always      = {"logits", "theta"},
        output_keys_conditional = set(),
        loss_keys               = {"kl", "total"},
        grad_modules            = ["localizer", "stem", "backbone_features",
                                   "compartment", "drp", "projector", "classifier"],
    ),
    "e6": ExpectedSpec(
        active_flags            = {"use_stn", "use_dual_intensity", "use_compartment",
                                   "use_drp", "use_pgr"},
        output_keys_always      = {"logits", "theta", "sim_logits"},
        output_keys_conditional = set(),
        loss_keys               = {"kl", "total", "proto"},
        grad_modules            = ["localizer", "stem", "backbone_features",
                                   "compartment", "drp", "pgr", "projector", "classifier"],
    ),
    "e7": ExpectedSpec(
        active_flags            = {"use_stn", "use_dual_intensity", "use_compartment",
                                   "use_drp", "use_pgr", "use_rtc"},
        output_keys_always      = {"logits", "theta", "sim_logits"},
        output_keys_conditional = set(),
        loss_keys               = {"kl", "total", "proto"},
        grad_modules            = ["localizer", "stem", "backbone_features",
                                   "compartment", "drp", "pgr", "rtc",
                                   "projector", "classifier"],
    ),
    "e8": ExpectedSpec(
        active_flags            = {"use_stn", "use_dual_intensity", "use_compartment",
                                   "use_drp", "use_pgr", "use_rtc", "use_aux_heads"},
        output_keys_always      = {"logits", "theta", "sim_logits",
                                   "h1", "h2", "h3", "h4", "h5", "h6", "h7"},
        output_keys_conditional = set(),
        loss_keys               = {"kl", "coral", "supcon", "jsn_med", "jsn_lat",
                                   "osteo", "uncert", "total", "proto"},
        grad_modules            = ["localizer", "stem", "backbone_features",
                                   "compartment", "drp", "pgr", "rtc",
                                   "projector", "heads"],
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# Verification logic
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class VerifyResult:
    exp: str
    param_total: int
    param_trainable: int
    active_flags: List[str]
    output_keys: List[str]
    loss_keys: List[str]
    grad_ok: bool
    flag_ok: bool
    output_ok: bool
    loss_ok: bool
    errors: List[str]


def _has_grad(module: nn.Module) -> bool:
    """Return True if any leaf parameter in *module* received a gradient."""
    return any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in module.parameters()
        if p.requires_grad
    )


def _get_nested(model: nn.Module, attr_path: str) -> Optional[nn.Module]:
    """Resolve a dotted attribute path like 'compartment.fusion'."""
    obj = model
    for part in attr_path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def verify_experiment(exp: str, device: torch.device, verbose: bool) -> VerifyResult:
    spec   = _SPECS[exp]
    errors: List[str] = []

    cfg   = get_config(exp)
    model = DRPNet(cfg.model).to(device)
    loss_fn = (
        MultiTaskLoss(cfg.training)
        if cfg.model.use_aux_heads
        else nn.CrossEntropyLoss()
    )

    # ── 1. Module flag check ──────────────────────────────────────────────
    all_flags = ["use_stn", "use_dual_intensity", "use_compartment",
                 "use_drp", "use_pgr", "use_rtc", "use_aux_heads"]
    active_flags = [f for f in all_flags if getattr(cfg.model, f)]
    expected_active = sorted(spec.active_flags)
    actual_active   = sorted(active_flags)
    flag_ok = (actual_active == expected_active)
    if not flag_ok:
        errors.append(
            f"FLAG MISMATCH: expected {expected_active}, got {actual_active}"
        )

    # ── 2. Parameter count ────────────────────────────────────────────────
    total_p     = count_all_parameters(model)
    trainable_p = count_parameters(model)

    # ── 3. Forward pass ───────────────────────────────────────────────────
    B = 2
    H = 64   # small for speed
    K = cfg.model.num_classes
    use_3crop = cfg.model.use_compartment

    model.train()
    g = torch.randn(B, 3, H, H, device=device)
    m = torch.randn(B, 3, H, H, device=device) if use_3crop else None
    l_= torch.randn(B, 3, H, H, device=device) if use_3crop else None

    try:
        preds = model(g)
    except Exception as e:
        errors.append(f"FORWARD ERROR: {e}")
        return VerifyResult(
            exp=exp, param_total=total_p, param_trainable=trainable_p,
            active_flags=active_flags, output_keys=[], loss_keys=[],
            grad_ok=False, flag_ok=flag_ok, output_ok=False, loss_ok=False,
            errors=errors,
        )

    output_keys = list(preds.keys())

    # ── 4. Output key check ───────────────────────────────────────────────
    missing_out = spec.output_keys_always - set(output_keys)
    output_ok   = len(missing_out) == 0
    if not output_ok:
        errors.append(f"MISSING OUTPUTS: {missing_out}")

    # ── 5. Loss computation ───────────────────────────────────────────────
    labels = make_labels_stub(B, K, device)
    try:
        if isinstance(loss_fn, MultiTaskLoss):
            loss_dict = loss_fn(preds, labels)
        else:
            ce = loss_fn(preds["logits"], labels["kl"])
            loss_dict = {"kl": ce, "total": ce}

        if "sim_logits" in preds:
            proto = 0.3 * F.cross_entropy(preds["sim_logits"], labels["kl"])
            loss_dict["proto"] = proto
            loss_dict["total"] = loss_dict["total"] + proto

        loss_keys = list(loss_dict.keys())
        missing_loss = spec.loss_keys - set(loss_keys)
        loss_ok = len(missing_loss) == 0
        if not loss_ok:
            errors.append(f"MISSING LOSS KEYS: {missing_loss}")
    except Exception as e:
        errors.append(f"LOSS ERROR: {e}")
        loss_keys = []
        loss_ok   = False
        loss_dict = {"total": torch.tensor(0.0, requires_grad=True)}

    # ── 6. Gradient flow check ────────────────────────────────────────────
    try:
        loss_dict["total"].backward()
    except Exception as e:
        errors.append(f"BACKWARD ERROR: {e}")
        return VerifyResult(
            exp=exp, param_total=total_p, param_trainable=trainable_p,
            active_flags=active_flags, output_keys=output_keys,
            loss_keys=loss_keys, grad_ok=False,
            flag_ok=flag_ok, output_ok=output_ok, loss_ok=loss_ok,
            errors=errors,
        )

    grad_ok = True
    for attr in spec.grad_modules:
        module = _get_nested(model, attr)
        if module is None:
            errors.append(f"GRAD CHECK: module '{attr}' not found on model")
            grad_ok = False
            continue
        if not _has_grad(module):
            errors.append(f"GRAD CHECK: no gradient in '{attr}'")
            grad_ok = False
        elif verbose:
            logger.warning("  [%s] %s — grad ✓", exp.upper(), attr)

    return VerifyResult(
        exp             = exp,
        param_total     = total_p,
        param_trainable = trainable_p,
        active_flags    = active_flags,
        output_keys     = output_keys,
        loss_keys       = loss_keys,
        grad_ok         = grad_ok,
        flag_ok         = flag_ok,
        output_ok       = output_ok,
        loss_ok         = loss_ok,
        errors          = errors,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Table formatter
# ──────────────────────────────────────────────────────────────────────────────

_FLAG_SHORT = {
    "use_stn":            "STN",
    "use_dual_intensity": "DIS",
    "use_compartment":    "CBM",
    "use_drp":            "DRP",
    "use_pgr":            "PGR",
    "use_rtc":            "RTC",
    "use_aux_heads":      "AUX",
}

def _flags_str(flags: List[str]) -> str:
    return " ".join(_FLAG_SHORT.get(f, f) for f in flags) or "—"


def print_table(results: List[VerifyResult]) -> None:
    header = (
        f"{'Exp':<4} │ {'Active Modules':<28} │ "
        f"{'Params (M)':<12} │ {'Outputs':<30} │ "
        f"{'Flag':>4} {'Out':>3} {'Loss':>4} {'Grad':>4} │ Status"
    )
    sep = "─" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)

    all_pass = True
    for r in results:
        flags_s = _flags_str(r.active_flags)
        params_m = f"{r.param_trainable / 1e6:.2f}M"
        out_s   = " ".join(sorted(
            k for k in r.output_keys
            if k not in {"h1","h2","h3","h4","h5","h6","h7"}
        ))

        f_ok = "✓" if r.flag_ok   else "✗"
        o_ok = "✓" if r.output_ok else "✗"
        l_ok = "✓" if r.loss_ok   else "✗"
        g_ok = "✓" if r.grad_ok   else "✗"
        status = "PASS" if all([r.flag_ok, r.output_ok, r.loss_ok, r.grad_ok]) else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(
            f"{r.exp.upper():<4} │ {flags_s:<28} │ "
            f"{params_m:<12} │ {out_s:<30} │ "
            f"{f_ok:>4} {o_ok:>3} {l_ok:>4} {g_ok:>4} │ {status}"
        )

        for err in r.errors:
            print(f"       ⚠  {err}")

    print(sep)
    print(f"  {'All experiments PASS ✓' if all_pass else 'Some experiments FAILED ✗'}")
    print(sep + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp",     choices=[*EXPERIMENT_NAMES.keys(), "all"],
                        default="all")
    parser.add_argument("--device",  type=str, default=None)
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-module gradient status")
    args = parser.parse_args()

    device  = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    targets = list(EXPERIMENT_NAMES.keys()) if args.exp == "all" else [args.exp]

    print(f"\nCoRD-Net Ablation Integrity Check — device={device}")
    print(f"Columns: Flag=module flags, Out=outputs, Loss=loss keys, Grad=gradient flow\n")

    results: List[VerifyResult] = []
    for exp in targets:
        print(f"  Checking {exp.upper()} …", end=" ", flush=True)
        r = verify_experiment(exp, device, args.verbose)
        ok = all([r.flag_ok, r.output_ok, r.loss_ok, r.grad_ok])
        print("✓" if ok else "✗")
        results.append(r)

    print_table(results)

    failed = [r for r in results if r.errors]
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
