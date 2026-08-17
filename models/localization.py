"""
models/localization.py
======================
E2 — Spatial Transformer Network (STN) for automatic knee localization.

IMPROVEMENTS over original
---------------------------
1. Pretrained MobileNetV3-Small feature extractor replaces the scratch CNN.
   The first conv is adapted from 3-channel to 1-channel by averaging RGB
   weights, preserving pretrained knowledge.  Falls back to a scratch CNN
   automatically if pretrained weights are unavailable (offline/air-gapped).

2. Freeze/unfreeze support for the pretrained backbone.
   freeze_backbone() / unfreeze_backbone() are called by the Trainer at
   epoch boundaries so the random fc_loc head does not corrupt pretrained
   features in early training.

3. named_parameter_groups() for differential learning rates.
   STN backbone gets 0.1× base_lr; head gets full base_lr.

4. max_delta tightened to 0.25 (was 0.3 in the fixed version, 0.5 original).

5. Translation regularisation loss via get_theta_reg_loss().
   Prevents the STN from translating the crop off the image boundary.
   Add to total loss with weight ~0.01.

6. Identity init: weight=0, bias=0 so tanh(0)=0 → theta=identity at start.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

logger = logging.getLogger(__name__)


def _build_pretrained_loc_net() -> tuple[nn.Module, int]:
    """
    Try to build a pretrained MobileNetV3-Small loc_net.

    Returns (features_module, feature_dim) on success.
    Falls back to scratch CNN if weights cannot be downloaded.
    """
    try:
        mob = tv_models.mobilenet_v3_small(
            weights=tv_models.MobileNet_V3_Small_Weights.DEFAULT
        )
        # Adapt first conv: 3-ch → 1-ch, keep pretrained knowledge
        orig = mob.features[0][0]
        new_conv = nn.Conv2d(
            1, orig.out_channels,
            kernel_size=orig.kernel_size,
            stride=orig.stride,
            padding=orig.padding,
            bias=False,
        )
        with torch.no_grad():
            new_conv.weight.copy_(orig.weight.mean(dim=1, keepdim=True))
        mob.features[0][0] = new_conv
        logger.info("KneeLocalizer: using pretrained MobileNetV3-Small backbone")
        return mob.features, 576
    except Exception as e:
        logger.warning(
            "KneeLocalizer: could not load pretrained MobileNetV3 (%s). "
            "Falling back to scratch CNN.", e
        )
        return _build_scratch_loc_net(), 256


def _build_scratch_loc_net() -> nn.Sequential:
    """4-layer scratch CNN (original architecture, fixed init)."""
    return nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
        nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(4),
    )


class KneeLocalizer(nn.Module):
    """
    STN-based knee localization.

    Tries to use a pretrained MobileNetV3-Small backbone for the
    localization network; falls back to the original scratch CNN if
    weights are unavailable.

    Parameters
    ----------
    img_size:      Kept for API compatibility.
    max_delta:     Max affine deviation from identity (default 0.25).
    freeze_epochs: Epochs to freeze pretrained backbone (default 5).
    """

    def __init__(
        self,
        img_size:      int   = 224,
        max_delta:     float = 0.25,
        freeze_epochs: int   = 5,
    ) -> None:
        super().__init__()
        self.img_size      = img_size
        self.max_delta     = max_delta
        self.freeze_epochs = freeze_epochs

        self.loc_net,  feat_dim = _build_pretrained_loc_net()
        self.loc_pool           = nn.AdaptiveAvgPool2d(1)
        self._is_pretrained     = (feat_dim == 576)   # MobileNetV3 dim

        self.fc_loc = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.Hardswish(inplace=True) if self._is_pretrained else nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 6),
        )
        # Identity init: theta_raw=0 → tanh(0)=0 → theta=identity
        nn.init.zeros_(self.fc_loc[-1].weight)
        nn.init.zeros_(self.fc_loc[-1].bias)

    # ── Freeze / unfreeze (called by Trainer) ─────────────────────────────────

    def freeze_backbone(self) -> None:
        if self._is_pretrained:
            for p in self.loc_net.parameters():
                p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        if self._is_pretrained:
            for p in self.loc_net.parameters():
                p.requires_grad = True

    def named_parameter_groups(self, base_lr: float) -> list[dict]:
        """Differential LR: backbone at 0.1×, head at 1×."""
        return [
            {"params": list(self.loc_net.parameters()),
             "lr": base_lr * 0.1, "name": "stn_backbone"},
            {"params": list(self.fc_loc.parameters()),
             "lr": base_lr, "name": "stn_head"},
        ]

    def get_theta_reg_loss(self, theta: torch.Tensor) -> torch.Tensor:
        """L2 penalty on translation params to prevent off-image warps."""
        t = torch.stack([theta[:, 0, 2], theta[:, 1, 2]], dim=1)
        return (t ** 2).mean()

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x : (B, 1, H, W) grayscale

        Returns
        -------
        localized : (B, 1, H, W)
        theta     : (B, 2, 3)
        """
        feats     = self.loc_pool(self.loc_net(x)).flatten(1)
        theta_raw = self.fc_loc(feats).view(-1, 2, 3)

        identity = torch.zeros_like(theta_raw)
        identity[:, 0, 0] = 1.0
        identity[:, 1, 1] = 1.0

        theta = identity + torch.tanh(theta_raw) * self.max_delta

        grid      = F.affine_grid(theta, x.size(), align_corners=False)
        localized = F.grid_sample(
            x, grid, mode="bilinear",
            padding_mode="border", align_corners=False,
        )
        return localized, theta
