import os
import torch
import matplotlib.pyplot as plt
import numpy as np


class ModelVisualizer:
    def __init__(self, save_dir="results/visualization"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def _unnormalize(self, img):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)

        img = img.detach().cpu()

        if img.shape[0] == 3:
            img = img * std + mean

        return img.clamp(0,1)

    def save_image(self, tensor, filename, title=""):

        img = self._unnormalize(tensor)

        plt.figure(figsize=(5,5))
        plt.imshow(img.permute(1,2,0))
        plt.axis("off")
        plt.title(title)

        plt.savefig(
            os.path.join(self.save_dir, filename),
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    def save_featuremap(self, fmap, filename, title=""):

        fmap = fmap.detach().cpu()

        fmap = fmap.mean(0)
        fmap -= fmap.min()
        fmap /= (fmap.max() + 1e-8)
        plt.figure(figsize=(5,5))
        plt.imshow(fmap, cmap="jet")
        plt.colorbar()
        plt.axis("off")
        plt.title(title)

        plt.savefig(
            os.path.join(self.save_dir, filename),
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    def save_mask(self, mask, filename):

        mask = mask.detach().cpu().squeeze()

        plt.figure(figsize=(5,5))
        plt.imshow(mask, cmap="jet")
        plt.colorbar()

        plt.axis("off")

        plt.savefig(
            os.path.join(self.save_dir, filename),
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    def save_overlay(self, image, mask, filename):

        img = self._unnormalize(image)

        img = img.permute(1,2,0).numpy()
        mask = torch.nn.functional.interpolate(
            mask.detach().cpu().unsqueeze(0),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        ).squeeze().numpy()
        plt.figure(figsize=(5,5))

        plt.imshow(img)

        plt.imshow(
            mask,
            cmap="jet",
            alpha=0.45
        )

        plt.axis("off")

        plt.savefig(
            os.path.join(self.save_dir, filename),
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()