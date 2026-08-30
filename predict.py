"""
predict.py
==========
Inference script for CoRD-Net.

Loads a checkpoint and predicts KL grade for a single image or every
image in a directory.  Works with any experiment (E1–E8).

Usage
-----
Single image:
    python predict.py --checkpoint checkpoints/e8_best.pt \\
                      --exp e8 \\
                      --input knee.png

Directory:
    python predict.py --checkpoint checkpoints/e8_best.pt \\
                      --exp e8 \\
                      --input /data/OAI/images/ \\
                      --output predictions.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import torch
from PIL import Image

from augmentation import get_val_transforms
from config import get_config, EXPERIMENT_NAMES
from models.drpnet import DRPNet
from utils import get_device, load_checkpoint, setup_logging

logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
KL_DESCRIPTIONS = {
    0: "Grade 0 — Normal",
    1: "Grade 1 — Doubtful",
    2: "Grade 2 — Minimal",
    3: "Grade 3 — Moderate",
    4: "Grade 4 — Severe",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to .pt checkpoint file")
    p.add_argument("--exp",        required=True,
                   choices=list(EXPERIMENT_NAMES.keys()),
                   help="Experiment tag matching the checkpoint")
    p.add_argument("--input",      required=True,
                   help="Single image file or directory of images")
    p.add_argument("--output",     type=str, default=None,
                   help="CSV output path (default: print to stdout)")
    p.add_argument("--device",     type=str, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--log-dir",    type=str, default="logs")
    return p.parse_args()


def collect_images(input_path: str) -> list[Path]:
    """Return sorted list of image paths from a file or directory."""
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"Input path not found: {p}")
    if p.is_file():
        if p.suffix.lower() not in _IMAGE_EXTS:
            raise ValueError(f"Unsupported image format: {p.suffix}")
        return [p]
    # directory
    images = sorted(
        f for f in p.rglob("*") if f.suffix.lower() in _IMAGE_EXTS
    )
    if not images:
        raise ValueError(f"No images found in directory: {p}")
    return images


@torch.no_grad()
def predict_batch(
    model: DRPNet,
    paths: list[Path],
    transform,
    device: torch.device,
    use_compartment: bool,
) -> list[dict]:
    """
    Run inference on a list of image paths.

    For compartment experiments (E4–E8), medial/lateral crops are derived
    using the same suffix convention as the DataLoader.  If compartment
    files are absent, the global crop is replicated.

    Returns
    -------
    List of dicts: {filename, kl_grade, kl_label, confidence, probabilities}
    """
    model.eval()
    results = []

    for path in paths:
        img = Image.open(path).convert("L")
        tensor = transform(img).unsqueeze(0).to(device)   # (1, 3, H, W)

        preds = model(tensor)

        logits = preds["logits"]                          # (1, K)
        probs  = torch.softmax(logits, dim=1)[0]         # (K,)
        grade  = probs.argmax().item()
        conf   = probs[grade].item()

        results.append({
            "filename":     path.name,
            "kl_grade":     int(grade),
            "kl_label":     KL_DESCRIPTIONS[int(grade)],
            "confidence":   f"{conf:.4f}",
            "probabilities": " ".join(f"{p:.4f}" for p in probs.tolist()),
        })

    return results


def main() -> None:
    args   = parse_args()
    device = get_device(args.device)
    setup_logging(args.log_dir, f"predict_{args.exp}")

    cfg = get_config(args.exp, device=str(device))
    transform = get_val_transforms(cfg.model)

    # Load model
    model = DRPNet(cfg.model).to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()

    logger.info("Model loaded: %s (%s)", args.exp.upper(), cfg.description)

    images  = collect_images(args.input)
    logger.info("Found %d image(s) to predict", len(images))

    results = predict_batch(
        model, images, transform, device,
        use_compartment=cfg.model.use_compartment,
    )

    # Output
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        logger.info("Predictions saved → %s", out_path)
    else:
        # Print to stdout
        header = f"{'Filename':<35} {'KL':>3}  {'Confidence':>10}  Label"
        print(header)
        print("-" * len(header))
        for r in results:
            print(
                f"{r['filename']:<35} {r['kl_grade']:>3}  "
                f"{r['confidence']:>10}  {r['kl_label']}"
            )


if __name__ == "__main__":
    main()
