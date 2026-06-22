# CoRD-Net — Differential Relational Prototype Network

Research codebase for automated knee osteoarthritis grading (KL scale 0–4)
using the OAI dataset.  Eight cumulative ablation experiments (E1–E8) share
a single `DRPNet` model whose stages are activated by config flags.

---

## Project Structure

```
CoRD-Net/
├── train.py              # Main training entry point
├── evaluate.py           # Evaluation (accuracy / kappa / MAE)
├── predict.py            # Inference on single image or directory
├── run_experiment.py     # Ablation runner (synthetic stub, no dataset needed)
├── verify_ablations.py   # Integrity checker — flags, outputs, grads, params
│
├── config.py             # Typed dataclasses + experiment registry
├── dataset.py            # Production OAI DataLoader (Layout A + B)
├── augmentation.py       # CLAHE, Sobel noise, train/val transforms
├── losses.py             # MultiTaskLoss (7 heads + prototype alignment)
├── metrics.py            # Quadratic kappa, accuracy, MAE
├── trainer.py            # Training loop, optimizer, scheduler, checkpointing
├── utils.py              # Device, seed, checkpoint I/O, param/module summary
│
├── models/
│   ├── drpnet.py         # DRPNet — unified model, ONE shared backbone
│   ├── localization.py   # E2: KneeLocalizer (STN)
│   ├── dual_intensity.py # E3: DualIntensityStem (CLAHE + Sobel + Laplacian)
│   ├── compartment.py    # E4: CompartmentBranchModule + EGRB
│   ├── roi.py            # E5: DRPBlock (soft ROI mask + feature reweighting)
│   ├── pgr.py            # E6: PGRModule (prototype bank + cross-attention)
│   ├── rtc.py            # E7: RelationalTokenCoupling
│   ├── auxiliary.py      # E8: 7 auxiliary prediction heads
│   └── __init__.py
│
├── checkpoints/          # Saved model checkpoints
├── logs/                 # Training logs (one .log file per experiment)
└── results/              # Evaluation outputs
```

---

## Ablation Experiments

| Exp | Description                          | New Component              | Params  |
|-----|--------------------------------------|----------------------------|---------|
| E1  | Baseline ConvNeXt                    | —                          | ~28.2 M |
| E2  | + Auto-Localization                  | STN (KneeLocalizer)        | ~30.7 M |
| E3  | + Dual-Intensity Stem                | CLAHE + Sobel + Laplacian  | ~30.9 M |
| E4  | + Compartment Branches               | 3-crop EGRB encoder        | ~32.7 M |
| E5  | + Soft ROI Mask                      | DRPBlock                   | ~33.2 M |
| E6  | + Prototype-Guided Refinement        | PGRModule                  | ~33.8 M |
| E7  | + Relational Token Coupling          | RTC cross-attention        | ~34.5 M |
| E8  | Full model + Auxiliary Heads         | 7-head MultiTaskLoss       | ~34.6 M |

---

## Dataset Setup

### Layout A — Grade subdirectories (no CSV required)

```
/data/OAI/
  0/  img001.png  img002.png  ...
  1/  img101.png  ...
  2/  ...
  3/  ...
  4/  ...
```

KL grade is inferred from the subdirectory name.
Auxiliary labels (JSN, osteophyte) default to −1 (ignored by loss).

### Layout B — Flat directory + metadata CSV

```
/data/OAI/
  img001.png
  img002.png
  ...

metadata.csv:
  filename,  kl, jsn_med, jsn_lat, osteo_mf, osteo_lf, osteo_mt, osteo_lt
  img001.png, 2,       1,       0,        1,        0,        1,        0
  img002.png, 0,      -1,      -1,       -1,       -1,       -1,       -1
  ...
```

Required columns: `filename`, `kl`.
Optional columns: `jsn_med`, `jsn_lat`, `osteo_mf`, `osteo_lf`, `osteo_mt`, `osteo_lt`.
Missing / empty values → −1 (ignored).

### Compartment crops (E4–E8)

For experiments E4–E8, the DataLoader expects medial and lateral crops derived
from the global filename using configurable suffixes (default: `_MED`, `_LAT`):

```
/data/OAI/
  img001.png        ← global
  img001_MED.png    ← medial compartment
  img001_LAT.png    ← lateral compartment
```

If compartment files are absent, the global crop is replicated with a warning
(acceptable for quick tests; real training requires proper crops).

---

## Training

```bash
# Baseline (E1) — Layout A
python train.py --exp e1 --data-root /data/OAI

# Full model (E8) — Layout B, pretrained backbone, GPU
python train.py --exp e8 \
    --data-root /data/OAI \
    --metadata-csv /data/OAI/metadata.csv \
    --pretrained \
    --device cuda \
    --epochs 100 \
    --batch-size 16

# Resume interrupted training
python train.py --exp e6 \
    --data-root /data/OAI \
    --resume checkpoints/e6_epoch0050.pt
```

All flags:

| Flag              | Default  | Description                              |
|-------------------|----------|------------------------------------------|
| `--exp`           | required | e1 … e8                                  |
| `--data-root`     | required | OAI dataset root directory               |
| `--metadata-csv`  | None     | CSV with auxiliary labels (Layout B)     |
| `--epochs`        | 100      | Training epochs                          |
| `--batch-size`    | 16       | Mini-batch size                          |
| `--lr`            | 1e-4     | Learning rate                            |
| `--device`        | auto     | cpu / cuda                               |
| `--pretrained`    | off      | ImageNet ConvNeXt-tiny weights           |
| `--resume`        | None     | Checkpoint path to resume from           |
| `--amp`           | off      | Automatic mixed precision (CUDA only)    |
| `--num-workers`   | 4        | DataLoader worker processes              |
| `--checkpoint-dir`| checkpoints | Override checkpoint directory         |
| `--seed`          | 42       | Random seed                              |

---

## Evaluation

```bash
python evaluate.py \
    --checkpoint checkpoints/e8_best.pt \
    --exp e8 \
    --data-root /data/OAI \
    --split val                # or: test

# With metadata CSV
python evaluate.py \
    --checkpoint checkpoints/e8_best.pt \
    --exp e8 \
    --data-root /data/OAI \
    --metadata-csv /data/OAI/metadata.csv \
    --split test
```

Reports: **Accuracy**, **Quadratic Weighted Kappa**, **MAE**.

---

## Prediction / Inference

```bash
# Single image → print to stdout
python predict.py \
    --checkpoint checkpoints/e8_best.pt \
    --exp e8 \
    --input knee.png

# Directory → save CSV
python predict.py \
    --checkpoint checkpoints/e8_best.pt \
    --exp e8 \
    --input /data/OAI/test_images/ \
    --output predictions.csv
```

Output columns: `filename`, `kl_grade`, `kl_label`, `confidence`, `probabilities`.

---

## Ablation Integrity Check

Verifies module flags, output keys, loss keys, gradient flow, and parameter
counts for every experiment without a dataset:

```bash
python verify_ablations.py          # all experiments
python verify_ablations.py --exp e5 # single experiment
```

Expected output:

```
Exp  │ Active Modules               │ Params (M)   │ Outputs          │ Flag Out Loss Grad │ Status
─────────────────────────────────────────────────────────────────────────────────────────────────────
E1   │ —                            │ 28.21M       │ logits           │    ✓   ✓    ✓    ✓ │ PASS
E2   │ STN                          │ 30.74M       │ logits theta     │    ✓   ✓    ✓    ✓ │ PASS
...
E8   │ STN DIS CBM DRP PGR RTC AUX  │ 34.55M       │ logits sim_logits│    ✓   ✓    ✓    ✓ │ PASS
```

---

## Quick Test (no dataset needed)

```bash
# Run synthetic training stub for all experiments
python run_experiment.py --exp all --steps 3 --batch-size 4

# Single experiment
python run_experiment.py --exp e8 --steps 5
```

---

## Requirements

```
torch >= 2.0
torchvision >= 0.15
opencv-python
Pillow
numpy
```

Install:
```bash
pip install torch torchvision opencv-python Pillow numpy
```

---

## Design Notes

- **One backbone**: `DRPNet` holds exactly one `ConvNeXt-tiny` instance.
  Compartment branches, DRP, PGR, and RTC all receive feature maps from it —
  no duplicate weights, no second forward pass.
- **Config-driven**: every hyperparameter lives in `config.py` dataclasses
- **Single responsibility**: losses in `losses.py`, training logic in
  `trainer.py`, model modules each in their own file.
- **Prototype encapsulation**: EMA updates go through `model.update_prototypes()`
  — the trainer never accesses `PGRModule.bank` directly.
- **Checkpoint resume**: `--resume` restores model, optimizer, scheduler, epoch,
  and best validation loss; training continues from the interrupted epoch.
