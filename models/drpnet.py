"""
models/drpnet.py
================
DRPNet — the single unified model class for all CoRD-Net ablations.

Instantiate with a ModelConfig; ablation flags in the config activate
or deactivate each stage.  The experiment runner and trainer never
construct individual modules directly.

Pipeline (all optional stages shown)::

    Input (B, 3, H, W)
        │
        ├─[E2] KneeLocalizer (STN)       → localized (B, 3, H, W)
        │
        ├─[E3] DualIntensityStem         → enhanced (B, 3, H, W)
        │
        ├─ Shared ConvNeXt-tiny          → spatial  (B, 768, H', W')
        │                                → pooled   (B, 768)
        │
        ├─[E4] CompartmentBranchModule   → g/m/l each (B, 768)
        │       (injected stem+backbone)
        │
        ├─[E5] DRPBlock                  → drp_emb  (B, 256)
        │
        ├─[E6] PGRModule                 → refined  (B, 256)
        │                                → sim_logits (B, K)
        │
        ├─[E7] RelationalTokenCoupling   → rtc_emb  (B, 256)
        │
        ├─ Projection Linear             → feat     (B, 512)
        │
        └─[E8] Auxiliary Heads           → {h1…h7}
               or Linear classifier      → logits   (B, K)

One backbone, zero duplicate forward passes.
"""

from __future__ import annotations

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
    cfg:
        ModelConfig produced by ``get_config(experiment).model``.

    Forward inputs
    --------------
    global_crop:  (B, 3, H, W) — always required
    medial_crop:  (B, 3, H, W) — required when use_compartment=True
    lateral_crop: (B, 3, H, W) — required when use_compartment=True

    Forward outputs (dict — keys depend on active stages)
    -----------------------------------------------------
    logits      (B, num_classes)    always present
    sim_logits  (B, num_classes)    when use_pgr=True
    h1…h7       various             when use_aux_heads=True
    theta       (B, 2, 3)           when use_stn=True
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

        # ── E4: Compartment Branches (injected backbone — no duplication) ─
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
        # cat[global(D), drp/refined(E), rtc(E)] → fused_dim(Fd)
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

        # Internal cache for EMA prototype update (set during forward)
        self._last_drp_emb: Optional[torch.Tensor] = None


    # ── Helpers ───────────────────────────────────────────────────────────────

    def _encode_single(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run stem → backbone on one crop.

        Returns
        -------
        spatial: (B, 768, H', W')
        pooled:  (B, 768)
        """
        enhanced = self.stem(x)
        spatial  = self.backbone_features(enhanced)
        pooled   = self.backbone_pool(spatial)
        return spatial, pooled

    def update_prototypes(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> None:
        """
        EMA-update grade prototypes (called by Trainer after backward).

        Parameters
        ----------
        embeddings: (B, embedding_dim) — detached DRP embeddings.
        labels:     (B,) integer KL grades.
        """
        if self.pgr is not None:
            self.pgr.update_prototypes_from_batch(embeddings, labels)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        global_crop: torch.Tensor,
        medial_crop:  Optional[torch.Tensor] = None,
        lateral_crop: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor | list]:
        """
        Parameters
        ----------
        global_crop:
            (B, 3, H, W) — RGB (or grayscale-as-RGB after augmentation).
        medial_crop, lateral_crop:
            (B, 3, H, W) each — required when use_compartment=True.
            If None and compartment is active, raises a clear RuntimeError.

        Returns
        -------
        out: dict of named outputs (see class docstring).
        """
        out: dict[str, torch.Tensor | list] = {}

        # ── E2: Localization ──────────────────────────────────────────────
        if self.localizer is not None:
            # STN expects (B, 1, H, W). Average channels (lossless post-GrayToRGB).
            x_gray = (global_crop.mean(dim=1, keepdim=True)
                      if global_crop.shape[1] == 3 else global_crop)
            localized, theta = self.localizer(x_gray)
            out["theta"] = theta
            global_crop = localized.expand(-1, 3, -1, -1).clone()
            


            # ^ .clone() ensures the view is not shared with the graph leaf

        # ── Backbone: encode global crop (one pass, reused by E4 and E5) ─
        global_spatial, global_pooled = self._encode_single(global_crop)
        # global_spatial: (B, 768, H', W')  used by DRP (E5)
        # global_pooled:  (B, 768)           used by fusion + RTC

        g_feat = global_pooled          # updated by compartment below if E4+
        m_feat: Optional[torch.Tensor] = None
        l_feat: Optional[torch.Tensor] = None

        # ── E4: Compartment Branches ──────────────────────────────────────
        if self.compartment is not None:
            g_feat = global_pooled
            fused_feat, m_feat, l_feat = self.compartment(
                global_pooled,
                global_crop,
            )


            #m_feat, l_feat: (B, 768) each
            # global_spatial is still the single-pass result from above

        # ── E5: DRP Block ─────────────────────────────────────────────────
        drp_emb: Optional[torch.Tensor] = None
        if self.drp is not None:
            drp_emb = self.drp(global_spatial)       # (B, 256)
            self._last_drp_emb = drp_emb             # cache for EMA update

        # ── E6: PGR ───────────────────────────────────────────────────────
        refined_emb = drp_emb   # pass-through when PGR is inactive
        if self.pgr is not None and drp_emb is not None:
            refined_emb, sim_logits = self.pgr(drp_emb)
            out["sim_logits"] = sim_logits

        # ── E7: RTC ───────────────────────────────────────────────────────
        rtc_emb: Optional[torch.Tensor] = None
        if self.rtc is not None:
            if m_feat is None or l_feat is None:
                raise RuntimeError(
                    "RTC (E7) requires compartment features. "
                    "Ensure use_compartment=True when use_rtc=True."
                )
            rtc_emb = self.rtc(m_feat, l_feat, g_feat)   # (B, 256)

        # ── Fusion + Projection ───────────────────────────────────────────
        parts = [g_feat, fused_feat]
        if refined_emb is not None:
            parts.append(refined_emb)
        if rtc_emb is not None:
            parts.append(rtc_emb)

        fused = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        feat  = self.projector(fused)                     # (B, fused_dim)

        # ── E8: Auxiliary Heads or simple classifier ──────────────────────
        if self.heads is not None:
            for name, head in self.heads.items():
                out[name] = head(feat)
            out["logits"] = out["h1"]   # alias: primary KL log-probs
        else:
            assert self.classifier is not None
            out["logits"] = self.classifier(feat)
            if self.training and not hasattr(self, "_feat_debug"):
                self._feat_debug = True

                print("Feature mean :", feat.mean().item())
                print("Feature std  :", feat.std().item())
                print("Feature norm :", feat.norm(dim=1).mean().item())

                print("Classifier weight mean:",
                    self.classifier.weight.mean().item())
                print("Classifier weight std :",
                    self.classifier.weight.std().item())

                print("Classifier bias:",
                    self.classifier.bias.detach().cpu())

        return out


# Resolve Optional forward reference (Python 3.9 compat)
from typing import Optional   # noqa: E402  (must be after class body)
