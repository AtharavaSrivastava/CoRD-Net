# CoRD-Net — How to Run: A Complete Guide

This document tells you exactly what to do, step by step, from a fresh
machine to a fully trained model.  Every command is copy-pasteable.
Every path you need to change is clearly marked.

---

## 1. Installation

```bash
# Clone or unzip the project
cd /path/to/CoRD-Net          # ← change this to wherever you put the folder

# Install Python dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python Pillow numpy
```

> **Python version**: 3.10 or 3.11 recommended.
> **CUDA**: if you don't have a GPU, drop the `--index-url` flag and torch
> will install the CPU version automatically.

---

## 2. Quick Sanity Check (no dataset needed)

Before touching any real data, confirm the code runs:

```bash
# Runs a 3-step synthetic training loop for all 8 experiments
python run_experiment.py --exp all --steps 3 --batch-size 4
```

Expected output (last lines):
```
  E8 PASS ✓  final total=3.xxxx
  Done — 8 experiment(s) passed ✓
```

If every experiment says PASS, your installation is correct.

---

## 3. Dataset Setup

You have two options depending on how your OAI data is organised.

---

### Option A — Grade Subdirectories (simplest)

Organise your images like this:

```
/home/yourname/data/OAI/        ← this is your --data-root
    0/
        9000798_20060601_SAG_3D_DESS_LEFT_016610798103.png
        9001270_20060601_SAG_3D_DESS_LEFT_016610800101.png
        ...
    1/
        9000798_20060601_SAG_3D_DESS_RIGHT_016610798106.png
        ...
    2/
        ...
    3/
        ...
    4/
        ...
```

**Rules:**
- Subfolder name must be the integer KL grade (0, 1, 2, 3, 4).
- Any image format works: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`.
- You do **not** need a CSV file.
- Auxiliary labels (JSN, osteophytes) will be set to −1, meaning they are
  silently ignored by the loss function — this is fine for E1–E7.
  For E8 with full multi-task loss, use Option B.

---

### Option B — Flat Directory + CSV (full labels)

Put all images in one folder and provide a CSV:

```
/home/yourname/data/OAI/        ← --data-root
    9000798_LEFT.png
    9000798_RIGHT.png
    9001270_LEFT.png
    ...
```

Create `metadata.csv` anywhere you like (e.g. `/home/yourname/data/OAI/metadata.csv`):

```csv
filename,kl,jsn_med,jsn_lat,osteo_mf,osteo_lf,osteo_mt,osteo_lt
9000798_LEFT.png,2,1,0,1,0,1,0
9000798_RIGHT.png,0,-1,-1,-1,-1,-1,-1
9001270_LEFT.png,3,2,1,2,1,2,1
```

**Column reference:**

| Column     | Meaning                             | Values       |
|------------|-------------------------------------|--------------|
| `filename` | Image filename (basename only)      | e.g. `img.png` |
| `kl`       | KL grade                            | 0, 1, 2, 3, 4 |
| `jsn_med`  | Medial joint space narrowing        | 0–3, or −1   |
| `jsn_lat`  | Lateral joint space narrowing       | 0–3, or −1   |
| `osteo_mf` | Medial femur osteophyte severity    | 0–2, or −1   |
| `osteo_lf` | Lateral femur osteophyte severity   | 0–2, or −1   |
| `osteo_mt` | Medial tibia osteophyte severity    | 0–2, or −1   |
| `osteo_lt` | Lateral tibia osteophyte severity   | 0–2, or −1   |

**Rules:**
- Use −1 for any label you don't have. That sample's contribution to that
  specific loss term will be zero (the loss uses `ignore_index=-1`).
- Only `filename` and `kl` are required. All other columns are optional.
- Empty cells, "NA", "nan", "None" are all treated as −1 automatically.

---

### Compartment Crops for E4–E8

Experiments E4, E5, E6, E7, E8 use three crops per knee:
global (full joint), medial compartment, lateral compartment.

The code looks for medial/lateral files automatically by appending `_MED`
and `_LAT` to the global filename's stem:

```
/home/yourname/data/OAI/
    9000798_LEFT.png           ← global crop
    9000798_LEFT_MED.png       ← medial crop
    9000798_LEFT_LAT.png       ← lateral crop
```

**If these files don't exist:**
- The code will log a warning and silently use the global crop for both
  compartments.
- This is fine for E1–E3 (which don't use compartment branches).
- For proper E4–E8 training, you need real compartment crops. These are
  typically extracted by cropping the medial and lateral halves of the
  knee X-ray image using a bounding box from a localisation model or
  manual annotation.

**Custom suffix:**
If your files use different names (e.g. `_medial`, `_lateral`), you can
change the suffix in `config.py`:

```python
# config.py  →  TrainingConfig
medial_suffix: str = "_medial"    # change from "_MED"
lateral_suffix: str = "_lateral"  # change from "_LAT"
```

---

## 4. Training

### Minimal working command (Option A, E1 baseline):

```bash
python train.py \
    --exp e1 \
    --data-root /home/yourname/data/OAI
```

### Full E8 training on GPU with all labels:

```bash
python train.py \
    --exp e8 \
    --data-root /home/yourname/data/OAI \
    --metadata-csv /home/yourname/data/OAI/metadata.csv \
    --pretrained \
    --device cuda \
    --epochs 100 \
    --batch-size 16
```

### Every flag explained:

| Flag              | Required? | What to put                                |
|-------------------|-----------|--------------------------------------------|
| `--exp`           | **Yes**   | `e1` through `e8`                          |
| `--data-root`     | **Yes**   | Full path to your OAI image folder         |
| `--metadata-csv`  | No        | Full path to your CSV (skip for Option A)  |
| `--pretrained`    | No        | Add this flag to use ImageNet weights      |
| `--device`        | No        | `cuda` for GPU, `cpu` for CPU, or omit for auto |
| `--epochs`        | No        | Default: 100                               |
| `--batch-size`    | No        | Default: 16. Reduce to 8 or 4 if you run out of GPU memory |
| `--lr`            | No        | Default: 0.0001                            |
| `--num-workers`   | No        | Default: 4. Set to 0 if you get DataLoader errors on Windows |
| `--amp`           | No        | Add for automatic mixed precision (faster on modern GPUs) |
| `--checkpoint-dir`| No        | Where to save .pt files. Default: `checkpoints/` |
| `--log-dir`       | No        | Where to save log files. Default: `logs/`  |
| `--seed`          | No        | Default: 42                                |
| `--resume`        | No        | See Section 6 (resuming)                   |

### Where outputs go:

```
CoRD-Net/
    checkpoints/
        e8_best.pt          ← saved whenever val loss improves
        e8_epoch0010.pt     ← saved every 10 epochs
        e8_epoch0020.pt
        ...
        e8_final.pt         ← saved at the end of training
    logs/
        e8.log              ← full training log with timestamps
```

### Training on Windows:

Add `--num-workers 0` to every command. Windows doesn't support
multi-process DataLoading the same way Linux does:

```bash
python train.py --exp e1 --data-root C:\data\OAI --num-workers 0
```

---

## 5. Evaluation

After training, evaluate on the validation or test split:

```bash
# Validate (same 15% held out during training)
python evaluate.py \
    --checkpoint checkpoints/e8_best.pt \
    --exp e8 \
    --data-root /home/yourname/data/OAI \
    --split val

# Test split (final held-out 15%)
python evaluate.py \
    --checkpoint checkpoints/e8_best.pt \
    --exp e8 \
    --data-root /home/yourname/data/OAI \
    --split test

# With metadata CSV
python evaluate.py \
    --checkpoint checkpoints/e8_best.pt \
    --exp e8 \
    --data-root /home/yourname/data/OAI \
    --metadata-csv /home/yourname/data/OAI/metadata.csv \
    --split test
```

**Output:**
```
  Results — E8 (test split)
  Accuracy : 0.7234
  Kappa    : 0.8103
  MAE      : 0.3421
```

> **Important:** the train/val/test split uses the same random seed (42) every
> time, so `evaluate.py` will always see the same held-out images as `train.py`
> used for validation — as long as you don't change `--seed`.

---

## 6. Resuming Interrupted Training

If training stops (power cut, time limit, etc.):

```bash
python train.py \
    --exp e8 \
    --data-root /home/yourname/data/OAI \
    --resume checkpoints/e8_epoch0050.pt
```

This restores:
- Model weights
- Optimizer state (momentum, adaptive rates)
- Scheduler state (learning rate position in the cosine curve)
- The epoch counter (training continues from epoch 51)
- The best validation loss (so best-model saving still works correctly)

You don't need to pass `--epochs` again — it reads the original value from
the config. But you can override it if you want to train for longer:

```bash
python train.py \
    --exp e8 \
    --data-root /home/yourname/data/OAI \
    --resume checkpoints/e8_epoch0050.pt \
    --epochs 150
```

---

## 7. Inference / Prediction

### Single image:

```bash
python predict.py \
    --checkpoint checkpoints/e8_best.pt \
    --exp e8 \
    --input /home/yourname/data/knee.png
```

Output printed to terminal:
```
Filename                             KL    Confidence  Label
---------------------------------------------------------------------
knee.png                              2        0.8341  Grade 2 — Minimal
```

### Entire folder of images:

```bash
python predict.py \
    --checkpoint checkpoints/e8_best.pt \
    --exp e8 \
    --input /home/yourname/data/test_images/ \
    --output /home/yourname/results/predictions.csv
```

`predictions.csv` will contain:
```csv
filename,kl_grade,kl_label,confidence,probabilities
knee001.png,2,Grade 2 — Minimal,0.8341,0.0123 0.0456 0.8341 0.0921 0.0159
knee002.png,0,Grade 0 — Normal,0.9102,0.9102 0.0612 0.0201 0.0071 0.0014
```

> **Note for E4–E8:** if you want proper compartment-aware predictions,
> your input folder must also contain `_MED` and `_LAT` images alongside
> each global image. If they're missing, the global crop is used for all
> three — predictions will still work but won't use compartment features.

---

## 8. Ablation Integrity Check

Before running any real training, verify the architecture is correct:

```bash
python verify_ablations.py
```

This takes about 2–3 minutes on CPU, runs a full forward+backward pass for
each of the 8 experiments with synthetic data, and prints a table:

```
Exp  │ Active Modules               │ Params (M)   │ Outputs        │ Flag Out Loss Grad │ Status
─────────────────────────────────────────────────────────────────────────────────────────────────
E1   │ —                            │ 28.21M       │ logits         │    ✓   ✓    ✓    ✓ │ PASS
E2   │ STN                          │ 30.74M       │ logits theta   │    ✓   ✓    ✓    ✓ │ PASS
...
E8   │ STN DIS CBM DRP PGR RTC AUX  │ 34.55M       │ logits sim_... │    ✓   ✓    ✓    ✓ │ PASS
```

Every row must say PASS before you run real training.

---

## 9. Experiment Selection Guide

Which experiment should you run?

| You want to...                                  | Run    |
|------------------------------------------------|--------|
| Just test the baseline                          | `e1`   |
| Add knee localization                           | `e2`   |
| Add structure-enhanced preprocessing            | `e3`   |
| Add medial/lateral compartment awareness        | `e4`   |
| Add soft ROI attention on spatial features      | `e5`   |
| Add KL grade prototype memory                   | `e6`   |
| Add relational coupling between compartments    | `e7`   |
| Full model — everything on, all aux losses      | `e8`   |
| Reproduce ablation table from the paper         | `all`  |

For the ablation table, train each experiment separately and evaluate each
checkpoint:

```bash
for exp in e1 e2 e3 e4 e5 e6 e7 e8; do
    python train.py --exp $exp --data-root /home/yourname/data/OAI --pretrained
    python evaluate.py --checkpoint checkpoints/${exp}_best.pt \
                       --exp $exp \
                       --data-root /home/yourname/data/OAI \
                       --split test
done
```

---

## 10. Changing Hyperparameters

All defaults live in `config.py`. You can change them there permanently,
or override most of them from the CLI.

### Common things to change:

**Batch size** (if you run out of GPU memory):
```bash
python train.py --exp e8 --data-root ... --batch-size 8
```

**Learning rate:**
```bash
python train.py --exp e8 --data-root ... --lr 0.00005
```

**Image size** — only change this in `config.py` because it affects the
model architecture:
```python
# config.py → ModelConfig
image_size: int = 224    # change to 229 to match config.yaml exactly
```

**Loss weights** — e.g. to weight KL loss more heavily:
```python
# config.py → TrainingConfig
loss_weights: Dict[str, float] = field(default_factory=lambda: {
    "h1": 2.0,   # ← increase KL weight (was 1.0)
    "h2": 0.5,
    ...
})
```

**Prototype temperature** (E6+):
```python
# config.py → ModelConfig
prototype_temperature: float = 0.05   # sharper (was 0.07)
```

**Scheduler** — switch from cosine to step decay:
```python
# config.py → TrainingConfig
scheduler: str = "step"   # was "cosine"
```

---

## 11. GPU Memory Guide

If you hit CUDA out-of-memory errors:

| Experiment | GPU RAM needed (batch 16, 224px) |
|-----------|----------------------------------|
| E1–E3     | ~6 GB                            |
| E4–E8     | ~10–12 GB (3 crops per image)    |

**To reduce memory:**
```bash
# Reduce batch size
--batch-size 8

# Or use mixed precision (cuts memory ~40%)
--batch-size 16 --amp

# Or reduce image size in config.py:
image_size: int = 160
```

---

## 12. File Paths — Summary of Everything You Need to Change

This is the complete list of paths you interact with. Nothing is hardcoded
in the source files — everything comes from CLI arguments.

| What                     | Flag                 | Example value                          |
|--------------------------|----------------------|----------------------------------------|
| OAI image folder         | `--data-root`        | `/home/yourname/data/OAI`             |
| Label CSV                | `--metadata-csv`     | `/home/yourname/data/OAI/metadata.csv`|
| Checkpoint to resume     | `--resume`           | `checkpoints/e8_epoch0050.pt`         |
| Checkpoint to evaluate   | `--checkpoint`       | `checkpoints/e8_best.pt`              |
| Image to predict         | `--input`            | `knee.png` or `/data/test_images/`    |
| Prediction output CSV    | `--output`           | `/home/yourname/results/preds.csv`    |
| Where checkpoints save   | `--checkpoint-dir`   | `checkpoints` (default)               |
| Where logs save          | `--log-dir`          | `logs` (default)                      |

**The only things you change in the source code** (`config.py`) are
hyperparameters (batch size, LR, loss weights, image size) that you want to
make permanent, or the compartment crop suffixes if your files use different
naming conventions.

---

## 13. Typical Workflow From Scratch

```bash
# 1. Confirm installation works
python run_experiment.py --exp all --steps 2 --batch-size 2

# 2. Verify all ablations are architecturally correct
python verify_ablations.py

# 3. Start with the baseline to make sure your dataset loads
python train.py --exp e1 --data-root /home/yourname/data/OAI --epochs 5

# 4. Check it learned something
python evaluate.py \
    --checkpoint checkpoints/e1_best.pt \
    --exp e1 \
    --data-root /home/yourname/data/OAI \
    --split val

# 5. Train the full model
python train.py \
    --exp e8 \
    --data-root /home/yourname/data/OAI \
    --metadata-csv /home/yourname/data/OAI/metadata.csv \
    --pretrained --device cuda --epochs 100

# 6. Evaluate on test set
python evaluate.py \
    --checkpoint checkpoints/e8_best.pt \
    --exp e8 \
    --data-root /home/yourname/data/OAI \
    --split test

# 7. Run inference on new images
python predict.py \
    --checkpoint checkpoints/e8_best.pt \
    --exp e8 \
    --input /home/yourname/new_images/ \
    --output results.csv
```
