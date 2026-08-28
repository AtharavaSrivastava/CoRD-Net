"""
visualization.py
================
Evaluation-only visualization module for CoRD-Net.

Generates and saves Grad-CAM / attention heatmaps and intermediate spatial
transformations (STN crops, compartment crops) for representative test images.

Requirements satisfied:
1. Evaluation-only, enabled via CLI `--visualize`.
2. Generates heatmaps for E1, E2, E4 (and all ablations).
3. Saves:
   - Original image
   - STN crop (if STN active)
   - Compartment crops (if compartment active)
   - Heatmap & Heatmap overlaid on image
4. Selects a small representative set (10-20 images) with mix of correct and incorrect samples.
5. Does not modify model architecture or training behavior.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config
from models.drpnet import DRPNet

logger = logging.getLogger(__name__)

# Standard ImageNet normalization constants used in dataset
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a normalized tensor (C, H, W) or (1, C, H, W) to RGB image (H, W, 3) in range [0, 1].
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    img_np = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img_np = img_np * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img_np, 0.0, 1.0)


class GradCAM:
    """
    Grad-CAM implementation for ConvNeXt backbone features.
    Attaches hooks to capture activation and gradients from the last stage of self.backbone_features.
    """

    def __init__(self, model: DRPNet) -> None:
        self.model = model
        self.target_layer = list(model.backbone_features.children())[-1]
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.h_forward = self.target_layer.register_forward_hook(forward_hook)
        self.h_backward = self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, target_class: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Run forward & backward pass for input_tensor (1, 3, H, W) and return Grad-CAM heatmap (H, W)
        along with any intermediate visualization outputs (STN crop, compartment crops, logits).
        """
        self.model.zero_grad()
        self.activations = None
        self.gradients = None

        # Enable gradient computation locally for visualization even during eval
        input_tensor = input_tensor.clone().requires_grad_(True)
        with torch.enable_grad():
            preds = self.model(input_tensor, return_debug_crops=True)
            logits = preds["logits"]
            score = logits[0, target_class]
            score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks failed to capture activations or gradients.")

        # Compute Grad-CAM
        grads = self.gradients.detach()        # (1, C, H', W')
        acts = self.activations.detach()       # (1, C, H', W')
        weights = grads.mean(dim=(2, 3), keepdim=True) # (1, C, 1, 1)

        cam = (weights * acts).sum(dim=1, keepdim=True) # (1, 1, H', W')
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(input_tensor.shape[2], input_tensor.shape[3]), mode="bilinear", align_corners=False)
        cam_np = cam.squeeze().cpu().numpy()

        if cam_np.max() > 0:
            cam_np = cam_np / cam_np.max()

        extra_outputs = {
            "logits": logits.detach().cpu(),
            "pred_cls": int(logits.argmax(dim=1).item()),
            "stn_crop": preds.get("_debug_stn_crop"),
            "medial_crop": preds.get("_debug_medial_crop"),
            "lateral_crop": preds.get("_debug_lateral_crop"),
        }

        return cam_np, extra_outputs

    def remove_hooks(self) -> None:
        self.h_forward.remove()
        self.h_backward.remove()


def run_visualizations(
    model: DRPNet,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: Path,
    max_samples: int = 15,
) -> None:
    """
    Generate and save Grad-CAM heatmaps and crop visualizations for representative samples.
    """
    output_dir = Path(output_dir) / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating evaluation visualizations in %s ...", output_dir)

    model.eval()

    # Step 1: Collect predictions across test set to select representative correct & incorrect samples
    all_inputs = []
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            crops, labels = batch
            g = crops[0].to(device)
            targets = labels["kl"].cpu()
            preds = model(g)["logits"].argmax(dim=1).cpu()

            for i in range(g.shape[0]):
                all_inputs.append(g[i:i+1].cpu())
                all_targets.append(int(targets[i].item()))
                all_preds.append(int(preds[i].item()))

    correct_indices = [i for i, (t, p) in enumerate(zip(all_targets, all_preds)) if t == p]
    incorrect_indices = [i for i, (t, p) in enumerate(zip(all_targets, all_preds)) if t != p]

    # Select balanced subset of correct and incorrect samples
    num_correct = min(max_samples // 2, len(correct_indices))
    num_incorrect = min(max_samples - num_correct, len(incorrect_indices))

    np.random.seed(42)
    selected_correct = np.random.choice(correct_indices, size=num_correct, replace=False).tolist() if num_correct > 0 else []
    selected_incorrect = np.random.choice(incorrect_indices, size=num_incorrect, replace=False).tolist() if num_incorrect > 0 else []
    selected_indices = selected_correct + selected_incorrect

    grad_cam = GradCAM(model)

    try:
        for idx_count, sample_idx in enumerate(selected_indices):
            x_input = all_inputs[sample_idx].to(device)
            gt_kl = all_targets[sample_idx]
            pred_kl = all_preds[sample_idx]
            status_str = "correct" if gt_kl == pred_kl else "incorrect"

            # Generate Grad-CAM targeting the predicted class
            heatmap, extras = grad_cam.generate(x_input, target_class=pred_kl)

            orig_img = denormalize_image(x_input)

            stn_crop = extras.get("stn_crop")
            medial_crop = extras.get("medial_crop")
            lateral_crop = extras.get("lateral_crop")

            # Plot grid depending on available components
            has_stn = stn_crop is not None
            has_comp = medial_crop is not None and lateral_crop is not None

            num_cols = 3  # Original, Heatmap, Overlay
            if has_stn:
                num_cols += 1
            if has_comp:
                num_cols += 2

            fig, axes = plt.subplots(1, num_cols, figsize=(4 * num_cols, 4))
            if num_cols == 1:
                axes = [axes]

            col_idx = 0

            # 1. Original Image
            axes[col_idx].imshow(orig_img)
            axes[col_idx].set_title(f"Original Input\n(GT: KL{gt_kl}, Pred: KL{pred_kl})", fontsize=10)
            axes[col_idx].axis("off")
            col_idx += 1

            # 2. STN Transformed Crop (if available)
            if has_stn:
                stn_img = denormalize_image(stn_crop)
                axes[col_idx].imshow(stn_img)
                axes[col_idx].set_title("STN Auto-Localized", fontsize=10)
                axes[col_idx].axis("off")
                col_idx += 1

            # 3. Compartment Crops (if available)
            if has_comp:
                m_img = denormalize_image(medial_crop)
                l_img = denormalize_image(lateral_crop)

                axes[col_idx].imshow(m_img)
                axes[col_idx].set_title("Medial Compartment", fontsize=10)
                axes[col_idx].axis("off")
                col_idx += 1

                axes[col_idx].imshow(l_img)
                axes[col_idx].set_title("Lateral Compartment", fontsize=10)
                axes[col_idx].axis("off")
                col_idx += 1

            # 4. Grad-CAM Heatmap
            axes[col_idx].imshow(heatmap, cmap="jet")
            axes[col_idx].set_title(f"Grad-CAM Heatmap\n(Target: KL{pred_kl})", fontsize=10)
            axes[col_idx].axis("off")
            col_idx += 1

            # 5. Overlaid Heatmap
            axes[col_idx].imshow(orig_img)
            axes[col_idx].imshow(heatmap, cmap="jet", alpha=0.5)
            axes[col_idx].set_title("Overlay", fontsize=10)
            axes[col_idx].axis("off")

            plt.tight_layout()
            save_name = f"sample_{idx_count+1:02d}_{status_str}_gt{gt_kl}_pred{pred_kl}.png"
            plt.savefig(output_dir / save_name, dpi=200, bbox_inches="tight")
            plt.close(fig)

        logger.info("Saved %d evaluation visualization images to %s", len(selected_indices), output_dir)
    finally:
        grad_cam.remove_hooks()
