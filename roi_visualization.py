import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def save_roi_visualizations(
    drp_block,
    save_dir="roi_vis",
    prefix="sample",
):
    """
    Save all ROI visualizations.

    Required cached tensors
    -----------------------
    drp_block.last_input
    drp_block.last_mask
    drp_block.last_feature_map
    drp_block.last_masked_feature
    """

    os.makedirs(save_dir, exist_ok=True)

    # -------------------------------------------------
    # Retrieve tensors
    # -------------------------------------------------

    print("last_input shape:", drp_block.last_input.shape)
    image = drp_block.last_input[0].cpu()              # (3,H,W)
    plt.figure(figsize=(5,5))
    plt.imshow(image[0], cmap="gray")
    plt.colorbar()
    plt.title("RAW CHANNEL")
    plt.savefig("raw_channel.png", dpi=200)
    plt.close()

    print(
    "last_input:",
    image.shape,
    image.min().item(),
    image.max().item(),
    image.mean().item(),
    image.std().item(),
)
    print("single image:", image.shape)
    mask = drp_block.last_mask[0].cpu()                # (1,h,w)
    fmap = drp_block.last_feature_map[0].cpu()         # (C,h,w)
    masked = drp_block.last_masked_feature[0].cpu()    # (C,h,w)

    # -------------------------------------------------
    # Prepare image
    # -------------------------------------------------

    # Undo ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    image = image * std + mean
    image = image.clamp(0, 1)

    # Convert RGB -> grayscale
    image = image.mean(0)

    H, W = image.shape

    # -------------------------------------------------
    # Upsample mask
    # -------------------------------------------------

    mask_up = F.interpolate(
        mask.unsqueeze(0),
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    # -------------------------------------------------
    # Average feature maps
    # -------------------------------------------------

    fmap_avg = fmap.mean(0)
    masked_avg = masked.mean(0)

    # -------------------------------------------------
    # Original image
    # -------------------------------------------------

    plt.figure(figsize=(5,5))
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.title("Original X-ray")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_01_original.png"), dpi=300)
    plt.close()

    # -------------------------------------------------
    # Raw mask
    # -------------------------------------------------

    plt.figure(figsize=(5,5))
    plt.imshow(mask[0], cmap="gray")
    plt.colorbar()
    plt.title("ROI Mask (7×7)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_02_mask_raw.png"), dpi=300)
    plt.close()

    # -------------------------------------------------
    # Upsampled mask
    # -------------------------------------------------

    plt.figure(figsize=(5,5))
    plt.imshow(mask_up, cmap="jet")
    plt.colorbar()
    plt.axis("off")
    plt.title("Upsampled ROI Mask")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_03_mask_up.png"), dpi=300)
    plt.close()

    # -------------------------------------------------
    # Overlay
    # -------------------------------------------------

    plt.figure(figsize=(5,5))
    plt.imshow(image, cmap="gray")
    plt.imshow(mask_up, cmap="jet", alpha=0.45)
    plt.axis("off")
    plt.title("ROI Overlay")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_04_overlay.png"), dpi=300)
    plt.close()

    # -------------------------------------------------
    # Backbone feature map
    # -------------------------------------------------

    plt.figure(figsize=(5,5))
    plt.imshow(fmap_avg, cmap="gray")
    plt.colorbar()
    plt.title("Backbone Feature Map")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_05_feature_map.png"), dpi=300)
    plt.close()

    # -------------------------------------------------
    # Masked feature map
    # -------------------------------------------------

    plt.figure(figsize=(5,5))
    plt.imshow(masked_avg, cmap="gray")
    plt.colorbar()
    plt.title("Masked Feature Map")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_06_masked_feature_map.png"), dpi=300)
    plt.close()

    print(f"ROI visualizations saved to '{save_dir}'")