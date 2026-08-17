"""
models/drpnet.py
================
DRPNet — unified model for all CoRD-Net ablations.

FIXES APPLIED
-------------
1. Removed 'from utils.visualizer import ModelVisualizer' at module level.
   utils is a .py file, not a package, so utils.visualizer doesn't exist.
   This caused an ImportError on every import of DRPNet, crashing the
   entire training process before a single epoch ran.

2. Removed all self.visualizer calls from forward().  Visualisation
   belongs in a separate debug script, not in the forward path of a
   training model.  These hasattr guards with _vis_* flags also created
   stale state that persisted across calls and prevented repeated
   visualisation.

3. Removed dead 'before = classifier.weight.detach().clone()' line.
   It was computed and never used. For E8, self.model.heads["h1"].fc
   is the correct attribute (not ".classifier"), so the original line
   would also have thrown AttributeError on E8+AMP.

4. drpnet.forward now accepts optional medial_crop/lateral_crop for
   forward compatibility, but E4+ routes through compartment which
   derives crops internally from global_crop — so the signature
   remains (global_crop,) for the trainer.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from torchvision.models import ConvNeXt_Tiny_Weights

from config import ModelConfig
from models.localization import KneeLocalizer
from models.dual_intensity import DualIntensityStem
from models.compartment import CompartmentBranchModule
from models.roi import DRPBlock
from models.pgr import PGRModule
from models.rtc import RelationalTokenCoupling
from models.auxiliary import (
    PrimaryKLHead, CORALOrdinalHead, MetricEmbeddingHead,
    MedialJSNHead, LateralJSNHead, OsteophyteHeads, UncertaintyHead,
)


class DRPNet(nn.Module):
    """
    Differential Relational Prototype Network.

    Parameters
    ----------
    cfg : ModelConfig  (produced by get_config(experiment).model)

    Forward input
    -------------
    global_crop : (B, 3, H, W)  always required

    Forward outputs  (dict; keys depend on active stages)
    -----------------------------------------------------
    logits      (B, num_classes)   always present
    sim_logits  (B, num_classes)   when use_pgr=True
    h1…h7       various            when use_aux_heads=True
    theta       (B, 2, 3)          when use_stn=True
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        D  = cfg.backbone_feature_dim   # 768
        E  = cfg.embedding_dim          # 256
        Fd = cfg.fused_dim              # 512
        K  = cfg.num_classes            # 5

        # ── E2: STN ───────────────────────────────────────────────────────
        self.localizer: Optional[KneeLocalizer] = None
        if cfg.use_stn:
            self.localizer = KneeLocalizer(img_size=cfg.stn_img_size)

        # ── E3: Dual-Intensity Stem ───────────────────────────────────────
        if cfg.use_dual_intensity:
            self.stem: nn.Module = DualIntensityStem(out_channels=3)
        else:
            self.stem = nn.Identity()

        # ── Shared ConvNeXt-tiny backbone (ONE instance) ──────────────────
        weights = ConvNeXt_Tiny_Weights.DEFAULT if cfg.pretrained else None
        _bb = tv_models.convnext_tiny(weights=weights)
        self.backbone_features = nn.Sequential(*list(_bb.features.children()))
        self.backbone_pool     = nn.Sequential(_bb.avgpool, nn.Flatten(1))

        # ── E4: Compartment Branches ──────────────────────────────────────
        self.compartment: Optional[CompartmentBranchModule] = None
        if cfg.use_compartment:
            self.compartment = CompartmentBranchModule(
                stem              = self.stem,
                backbone_features = self.backbone_features,
                feature_dim       = D,
            )

        # ── E5: DRP Block ─────────────────────────────────────────────────
        self.drp: Optional[DRPBlock] = None
        if cfg.use_drp:
            self.drp = DRPBlock(in_channels=D, out_dim=E)

        # ── E6: PGR Module ────────────────────────────────────────────────
        self.pgr: Optional[PGRModule] = None
        if cfg.use_pgr:
            self.pgr = PGRModule(
                embed_dim    = E,
                num_classes  = K,
                num_heads    = cfg.pgr_num_heads,
                dropout      = cfg.pgr_dropout,
                temperature  = cfg.prototype_temperature,
                ema_momentum = cfg.prototype_ema_momentum,
            )

        # ── E7: RTC ───────────────────────────────────────────────────────
        self.rtc: Optional[RelationalTokenCoupling] = None
        if cfg.use_rtc:
            self.rtc = RelationalTokenCoupling(
                in_channels        = D,
                token_dim          = E,
                num_heads          = cfg.rtc_num_heads,
                dropout            = cfg.rtc_dropout,
                use_global_context = cfg.rtc_use_global_context,
            )

        # ── Fusion projector ──────────────────────────────────────────────
        concat_dim = D
        if cfg.use_compartment:
            concat_dim += D
        if cfg.use_drp:
            concat_dim += E
        if cfg.use_rtc:
            concat_dim += E

        self.projector: nn.Module = (
            nn.Linear(concat_dim, Fd) if concat_dim != Fd else nn.Identity()
        )
        self._fused_dim = Fd if concat_dim != Fd else concat_dim

        # ── E8: Auxiliary Heads or simple classifier ───────────────────────
        if cfg.use_aux_heads:
            self.heads: Optional[nn.ModuleDict] = nn.ModuleDict({
                "h1": PrimaryKLHead(self._fused_dim, K),
                "h2": CORALOrdinalHead(self._fused_dim),
                "h3": MetricEmbeddingHead(self._fused_dim, cfg.metric_embed_dim),
                "h4": MedialJSNHead(self._fused_dim),
                "h5": LateralJSNHead(self._fused_dim),
                "h6": OsteophyteHeads(self._fused_dim),
                "h7": UncertaintyHead(self._fused_dim),
            })
            self.classifier: Optional[nn.Linear] = None
        else:
            self.heads      = None
            self.classifier = nn.Linear(self._fused_dim, K)

        self._last_drp_emb: Optional[torch.Tensor] = None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _encode_single(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """stem → backbone.  Returns (spatial (B,768,H',W'), pooled (B,768))."""
        enhanced = self.stem(x)
        spatial  = self.backbone_features(enhanced)
        pooled   = self.backbone_pool(spatial)
        return spatial, pooled

    def update_prototypes(self, embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        if self.pgr is not None:
            self.pgr.update_prototypes_from_batch(embeddings, labels)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, global_crop: torch.Tensor) -> dict[str, torch.Tensor | list]:
        out: dict[str, torch.Tensor | list] = {}

        # ── E2: Localization ──────────────────────────────────────────────
        if self.localizer is not None:
            x_gray = (global_crop.mean(dim=1, keepdim=True)
                      if global_crop.shape[1] == 3 else global_crop)
            localized, theta = self.localizer(x_gray)
            out["theta"]  = theta
            global_crop   = localized.expand(-1, 3, -1, -1).clone()

        # ── Backbone ──────────────────────────────────────────────────────
        global_spatial, global_pooled = self._encode_single(global_crop)
        g_feat = global_pooled

        m_feat: Optional[torch.Tensor] = None
        l_feat: Optional[torch.Tensor] = None

        # ── E4: Compartment Branches ──────────────────────────────────────
        fused_feat: Optional[torch.Tensor] = None
        if self.compartment is not None:
            fused_feat, m_feat, l_feat = self.compartment(global_pooled, global_crop)

        # ── E5: DRP Block ─────────────────────────────────────────────────
        drp_emb: Optional[torch.Tensor] = None
        if self.drp is not None:
            drp_emb            = self.drp(global_spatial)
            self._last_drp_emb = drp_emb

        # ── E6: PGR ───────────────────────────────────────────────────────
        refined_emb = drp_emb
        if self.pgr is not None and drp_emb is not None:
            refined_emb, sim_logits = self.pgr(drp_emb)
            out["sim_logits"] = sim_logits

        # ── E7: RTC ───────────────────────────────────────────────────────
        rtc_emb: Optional[torch.Tensor] = None
        if self.rtc is not None:
            if m_feat is None or l_feat is None:
                raise RuntimeError(
                    "RTC requires compartment features. "
                    "Set use_compartment=True when use_rtc=True."
                )
            rtc_emb = self.rtc(m_feat, l_feat, g_feat)

        # ── Fusion ────────────────────────────────────────────────────────
        parts = [g_feat]
        if fused_feat is not None:
            parts.append(fused_feat)
        if refined_emb is not None:
            parts.append(refined_emb)
        if rtc_emb is not None:
            parts.append(rtc_emb)

        fused = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        feat  = self.projector(fused)

        # ── Classifier / Heads ────────────────────────────────────────────
        if self.heads is not None:
            for name, head in self.heads.items():
                out[name] = head(feat)
            out["logits"] = out["h1"]
        else:
            assert self.classifier is not None
            out["logits"] = self.classifier(feat)

        return out
