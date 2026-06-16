"""
blocks_56.py  —  E5: Soft ROI Mask  |  E6: Prototype-Guided Refinement (PGR)

Structure mirrors blocks_78 / the existing notebook exactly:
  - Each experiment block is self-contained
  - Each block ends with a standalone test function
  - Integration test ties both blocks + ConvNeXt-tiny backbone together
  - Run everything:  python blocks_56.py
  - Run one block:   python blocks_56.py --block e5
                     python blocks_56.py --block e6
                     python blocks_56.py --block integration
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG  (mirrors config.yaml)
# ──────────────────────────────────────────────────────────────────────────────
CONFIG = {
    "dataset": {
        "name": "OAI",
        "image_size": 229,
        "num_classes": 5,
    },
    "model": {
        "backbone": "convnext_tiny",
        "pretrained": False,          # set True when real weights are available
        "in_channels": 3,
        "num_classes": 5,
    },
    "training": {
        "optimizer": "adamw",
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "batch_size": 16,
        "epochs": 100,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  E5 — SOFT ROI MASK
# ══════════════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────────────────
# E5-A: ROI Attention Mask Generator
# ──────────────────────────────────────────────────────────────────────────────
class ROIAttentionMask(nn.Module):
    """
    Generates a soft spatial attention mask from a feature map.

    Takes a spatial feature map (B, C, H, W) and produces a
    per-location scalar weight in [0, 1], indicating how much
    each spatial position belongs to the region of interest.

    Architecture:
        1x1 conv  →  BN  →  ReLU  →  1x1 conv  →  Sigmoid
    """

    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        mid = max(in_channels // reduction, 8)
        self.mask_net = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feature_map: (B, C, H, W)
        Returns:
            mask: (B, 1, H, W)  — soft ROI weights in [0, 1]
        """
        return self.mask_net(feature_map)


# ──────────────────────────────────────────────────────────────────────────────
# E5-B: Feature Reweighting Mechanism
# ──────────────────────────────────────────────────────────────────────────────
class FeatureReweighting(nn.Module):
    """
    Applies the soft ROI mask to reweight spatial features.

    Combines element-wise masking (hard gating) with a learnable
    residual blend parameter alpha, so the network can learn how
    aggressively to suppress off-ROI features:

        out = alpha * (mask * features) + (1 - alpha) * features

    alpha is initialised to 0.5 and constrained to [0, 1] via Sigmoid.
    """

    def __init__(self):
        super().__init__()
        self._alpha_raw = nn.Parameter(torch.zeros(1))   # sigmoid(0) = 0.5

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self._alpha_raw)

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: (B, C, H, W)
            mask:     (B, 1, H, W)
        Returns:
            reweighted: (B, C, H, W)
        """
        masked = mask * features
        return self.alpha * masked + (1.0 - self.alpha) * features


# ──────────────────────────────────────────────────────────────────────────────
# E5-C: DRP Block  (Dense ROI Pooling with soft mask)
# ──────────────────────────────────────────────────────────────────────────────
class DRPBlock(nn.Module):
    """
    Dense ROI Pooling block  —  E5 integration point.

    Pipeline:
        spatial features  →  ROIAttentionMask  →  FeatureReweighting
                          →  masked average-pool  →  projected embedding

    The output is a compact, ROI-focused descriptor suitable as input
    to downstream heads or the PGR module (E6).
    """

    def __init__(
        self,
        in_channels: int,
        out_dim: int = 256,
        reduction: int = 4,
    ):
        super().__init__()
        self.roi_mask = ROIAttentionMask(in_channels, reduction=reduction)
        self.reweight = FeatureReweighting()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(in_channels, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )
        # store last mask for inspection / Grad-CAM style vis
        self.last_mask: torch.Tensor | None = None

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feature_map: (B, C, H, W)  — spatial features from backbone stage
        Returns:
            embedding: (B, out_dim)
        """
        mask = self.roi_mask(feature_map)           # (B, 1, H, W)
        self.last_mask = mask.detach()

        reweighted = self.reweight(feature_map, mask)   # (B, C, H, W)
        pooled = self.pool(reweighted).flatten(1)        # (B, C)
        return self.proj(pooled)                         # (B, out_dim)


# ──────────────────────────────────────────────────────────────────────────────
# E5 Unit Test
# ──────────────────────────────────────────────────────────────────────────────
def test_e5_units():
    print("=" * 60)
    print("E5 — Unit Tests")
    print("=" * 60)
    B, C, H, W = 4, 256, 7, 7

    feat = torch.randn(B, C, H, W)

    # ROIAttentionMask
    roi_mask_mod = ROIAttentionMask(in_channels=C, reduction=4)
    mask = roi_mask_mod(feat)
    assert mask.shape == (B, 1, H, W), f"Mask shape wrong: {mask.shape}"
    assert mask.min() >= 0.0 and mask.max() <= 1.0, "Mask not in [0,1]"
    print(f"  ROIAttentionMask output : {mask.shape}  ✓")

    # FeatureReweighting
    reweight_mod = FeatureReweighting()
    reweighted = reweight_mod(feat, mask)
    assert reweighted.shape == feat.shape, "Reweighted shape mismatch"
    print(f"  FeatureReweighting output: {reweighted.shape}  ✓")
    print(f"  Blend alpha (initial)    : {reweight_mod.alpha.item():.4f}  (expect ≈0.5)")

    # DRPBlock
    drp = DRPBlock(in_channels=C, out_dim=256)
    emb = drp(feat)
    assert emb.shape == (B, 256), f"DRP embedding shape wrong: {emb.shape}"
    assert drp.last_mask is not None, "last_mask not stored"
    print(f"  DRPBlock embedding       : {emb.shape}  ✓")
    print(f"  Stored mask shape        : {drp.last_mask.shape}  ✓")
    print()


def validate_e5_with_backbone():
    print("=" * 60)
    print("E5 — Validation with ConvNeXt-Tiny Backbone")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4

    # ConvNeXt-tiny: features[6] is the penultimate stage → (B, 384, 7, 7)
    # features[7] is the final stage                       → (B, 768, 7, 7)
    backbone = models.convnext_tiny(weights=None).to(device)
    spatial_extractor = nn.Sequential(*list(backbone.features.children())).to(device)
    # last spatial stage output: 768 channels, 7×7 for 224px input
    spatial_extractor.eval()

    drp = DRPBlock(in_channels=768, out_dim=256, reduction=4).to(device)

    dummy = torch.randn(batch_size, 3, 224, 224, device=device)
    with torch.no_grad():
        spatial_feats = spatial_extractor(dummy)   # (B, 768, 7, 7)
    emb = drp(spatial_feats)

    assert spatial_feats.shape == (batch_size, 768, 7, 7), \
        f"Spatial feats shape: {spatial_feats.shape}"
    assert emb.shape == (batch_size, 256), \
        f"DRP embedding shape: {emb.shape}"
    assert drp.last_mask.shape == (batch_size, 1, 7, 7)

    print(f"  Spatial feature map : {spatial_feats.shape}  ✓")
    print(f"  ROI mask            : {drp.last_mask.shape}  ✓")
    print(f"  DRP embedding       : {emb.shape}  ✓")
    print("  E5 VALIDATION PASS  ✓\n")


# ══════════════════════════════════════════════════════════════════════════════
#  E6 — PROTOTYPE-GUIDED REFINEMENT (PGR)
# ══════════════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────────────────
# E6-A: Grade Prototype Initialization
# ──────────────────────────────────────────────────────────────────────────────
class GradePrototypeBank(nn.Module):
    """
    Learnable prototype memory bank with one prototype per grade class.

    Each prototype is a dense vector in the embedding space.  They are
    initialised from a unit normal (then L2-normalised) and updated
    jointly with the rest of the model through back-prop, with an
    optional exponential moving-average (EMA) update path for inference.

    Args:
        num_classes:   number of grade classes  (5 for KL grades 0-4)
        embed_dim:     dimensionality of prototype / feature embeddings
        ema_momentum:  EMA coefficient for non-gradient prototype updates
    """

    def __init__(
        self,
        num_classes: int = 5,
        embed_dim: int = 256,
        ema_momentum: float = 0.99,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.ema_momentum = ema_momentum

        # Learnable prototypes (gradient path)
        raw = torch.randn(num_classes, embed_dim)
        self.prototypes = nn.Parameter(F.normalize(raw, p=2, dim=1))

        # EMA shadow buffer (no gradient)
        self.register_buffer(
            "ema_prototypes",
            F.normalize(raw.clone().detach(), p=2, dim=1),
        )

    @torch.no_grad()
    def ema_update(self, class_idx: int, new_embedding: torch.Tensor):
        """
        Update a single class prototype via EMA (call during training).

        Args:
            class_idx:     integer grade label
            new_embedding: (embed_dim,)  —  mean embedding of class samples in batch
        """
        m = self.ema_momentum
        self.ema_prototypes[class_idx] = F.normalize(
            m * self.ema_prototypes[class_idx] + (1 - m) * new_embedding,
            p=2,
            dim=0,
        )

    def get_prototypes(self, use_ema: bool = False) -> torch.Tensor:
        """Returns (num_classes, embed_dim) prototype matrix."""
        if use_ema:
            return self.ema_prototypes
        return F.normalize(self.prototypes, p=2, dim=1)

    def forward(self, use_ema: bool = False) -> torch.Tensor:
        return self.get_prototypes(use_ema)


# ──────────────────────────────────────────────────────────────────────────────
# E6-B: Prototype Similarity Computation
# ──────────────────────────────────────────────────────────────────────────────
class PrototypeSimilarity(nn.Module):
    """
    Computes similarity between input embeddings and all class prototypes.

    Two modes:
        'cosine'  — cosine similarity  (default, scale-invariant)
        'dot'     — scaled dot product (temperature-controlled)

    Returns a (B, num_classes) logit tensor usable directly as class scores.
    """

    def __init__(
        self,
        mode: str = "cosine",
        temperature: float = 0.07,
    ):
        super().__init__()
        assert mode in ("cosine", "dot"), f"Unknown mode: {mode}"
        self.mode = mode
        self.temperature = temperature

    def forward(
        self,
        embeddings: torch.Tensor,           # (B, D)
        prototypes: torch.Tensor,           # (num_classes, D)
    ) -> torch.Tensor:                      # (B, num_classes)
        if self.mode == "cosine":
            norm_emb = F.normalize(embeddings, p=2, dim=1)
            norm_proto = F.normalize(prototypes, p=2, dim=1)
            return torch.mm(norm_emb, norm_proto.t()) / self.temperature
        else:
            return torch.mm(embeddings, prototypes.t()) / self.temperature


# ──────────────────────────────────────────────────────────────────────────────
# E6-C: Prototype Refinement Module
# ──────────────────────────────────────────────────────────────────────────────
class PrototypeRefinementModule(nn.Module):
    """
    Refines an input embedding by attending to grade prototypes.

    Pipeline:
        embedding  →  cross-attention over prototypes
                   →  prototype-weighted context vector
                   →  residual add + LayerNorm
                   →  refined embedding (same dim as input)

    The attention weights can be interpreted as a soft grade-assignment
    distribution and are stored in self.last_attn for inspection.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_classes: int = 5,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Cross-attention: query = embedding, key/value = prototypes
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Lightweight feed-forward refinement
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        self.last_attn: torch.Tensor | None = None   # (B, 1, num_classes)

    def forward(
        self,
        embedding: torch.Tensor,        # (B, embed_dim)
        prototypes: torch.Tensor,       # (num_classes, embed_dim)
    ) -> torch.Tensor:                  # (B, embed_dim)
        B = embedding.size(0)
        K = prototypes.size(0)

        # Query: unsqueeze to (B, 1, D)
        query = embedding.unsqueeze(1)

        # Key / Value: expand prototypes to (B, K, D)
        kv = prototypes.unsqueeze(0).expand(B, K, -1)

        # Cross-attention
        attn_out, attn_weights = self.cross_attn(query, kv, kv)
        self.last_attn = attn_weights.detach()    # (B, 1, K)

        # Residual + norm
        refined = self.norm(embedding + self.dropout(attn_out.squeeze(1)))

        # FFN + residual
        refined = self.norm2(refined + self.ffn(refined))
        return refined


# ──────────────────────────────────────────────────────────────────────────────
# E6 Full PGR Module  (bank + similarity + refinement composed)
# ──────────────────────────────────────────────────────────────────────────────
class PGRModule(nn.Module):
    """
    Prototype-Guided Refinement  —  E6 top-level module.

    Wraps GradePrototypeBank, PrototypeSimilarity, and
    PrototypeRefinementModule into a single forward call.

    Returns:
        refined_emb   : (B, embed_dim)   — prototype-refined embedding
        sim_logits    : (B, num_classes) — grade similarity scores (for aux loss)
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_classes: int = 5,
        num_heads: int = 4,
        dropout: float = 0.1,
        sim_mode: str = "cosine",
        temperature: float = 0.07,
        ema_momentum: float = 0.99,
    ):
        super().__init__()
        self.bank = GradePrototypeBank(
            num_classes=num_classes,
            embed_dim=embed_dim,
            ema_momentum=ema_momentum,
        )
        self.similarity = PrototypeSimilarity(mode=sim_mode, temperature=temperature)
        self.refine = PrototypeRefinementModule(
            embed_dim=embed_dim,
            num_classes=num_classes,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(
        self,
        embedding: torch.Tensor,            # (B, embed_dim)
        use_ema_prototypes: bool = False,
    ):
        prototypes = self.bank(use_ema=use_ema_prototypes)  # (K, D)
        sim_logits = self.similarity(embedding, prototypes)  # (B, K)
        refined_emb = self.refine(embedding, prototypes)     # (B, D)
        return refined_emb, sim_logits


# ──────────────────────────────────────────────────────────────────────────────
# E6 Unit Test
# ──────────────────────────────────────────────────────────────────────────────
def test_e6_units():
    print("=" * 60)
    print("E6 — Unit Tests")
    print("=" * 60)
    B, D, K = 4, 256, 5

    embeddings = F.normalize(torch.randn(B, D), p=2, dim=1)

    # GradePrototypeBank
    bank = GradePrototypeBank(num_classes=K, embed_dim=D)
    protos = bank()
    assert protos.shape == (K, D), f"Proto shape wrong: {protos.shape}"
    norms = protos.norm(dim=1)
    assert torch.allclose(norms, torch.ones(K), atol=1e-5), "Prototypes not unit-norm"
    print(f"  GradePrototypeBank  : {protos.shape}  ✓  (unit-norm: ✓)")

    # EMA update
    bank.ema_update(class_idx=0, new_embedding=torch.randn(D))
    ema_protos = bank(use_ema=True)
    assert ema_protos.shape == (K, D)
    print(f"  EMA update          : shape {ema_protos.shape}  ✓")

    # PrototypeSimilarity
    sim = PrototypeSimilarity(mode="cosine", temperature=0.07)
    logits = sim(embeddings, protos)
    assert logits.shape == (B, K), f"Sim logits shape: {logits.shape}"
    print(f"  PrototypeSimilarity : {logits.shape}  ✓")

    # PrototypeRefinementModule
    prm = PrototypeRefinementModule(embed_dim=D, num_classes=K, num_heads=4)
    refined = prm(embeddings, protos)
    assert refined.shape == (B, D), f"Refined shape: {refined.shape}"
    assert prm.last_attn is not None
    assert prm.last_attn.shape == (B, 1, K)
    print(f"  PrototypeRefinement : {refined.shape}  ✓")
    print(f"  Cross-attn weights  : {prm.last_attn.shape}  ✓")

    # Full PGR
    pgr = PGRModule(embed_dim=D, num_classes=K)
    refined_emb, sim_logits = pgr(embeddings)
    assert refined_emb.shape == (B, D)
    assert sim_logits.shape == (B, K)
    print(f"  PGRModule refined   : {refined_emb.shape}  ✓")
    print(f"  PGRModule logits    : {sim_logits.shape}  ✓")
    print()


def validate_e6_with_backbone():
    print("=" * 60)
    print("E6 — Validation with ConvNeXt-Tiny Backbone")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    num_classes = CONFIG["model"]["num_classes"]

    backbone = models.convnext_tiny(weights=None).to(device)
    backbone.classifier = nn.Identity()
    backbone.eval()

    projector = nn.Linear(768, 256).to(device)
    pgr = PGRModule(embed_dim=256, num_classes=num_classes).to(device)

    dummy = torch.randn(batch_size, 3, 224, 224, device=device)
    with torch.no_grad():
        global_feat = backbone(dummy).flatten(1)   # (B, 768)

    emb_256 = projector(global_feat)               # (B, 256)
    refined_emb, sim_logits = pgr(emb_256)

    assert refined_emb.shape == (batch_size, 256)
    assert sim_logits.shape == (batch_size, num_classes)

    print(f"  Backbone feature    : {global_feat.shape}  ✓")
    print(f"  Projected embedding : {emb_256.shape}  ✓")
    print(f"  Refined embedding   : {refined_emb.shape}  ✓")
    print(f"  Similarity logits   : {sim_logits.shape}  ✓")
    print("  E6 VALIDATION PASS  ✓\n")


# ══════════════════════════════════════════════════════════════════════════════
#  INTEGRATION TEST — E5 + E6 + ConvNeXt-Tiny  (end-to-end + gradient check)
# ══════════════════════════════════════════════════════════════════════════════
def run_e5_e6_integration_test():
    print("=" * 60)
    print("E5 + E6 Integration Test")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    num_classes = CONFIG["model"]["num_classes"]

    # ── 1. Backbone ──────────────────────────────────────────────────────────
    backbone = models.convnext_tiny(weights=None).to(device)
    # keep spatial features; pool separately in DRPBlock
    spatial_extractor = nn.Sequential(
        *list(backbone.features.children())
    ).to(device)
    # final stage: (B, 768, 7, 7) for 224px input

    # ── 2. E5: DRP Block ─────────────────────────────────────────────────────
    drp = DRPBlock(in_channels=768, out_dim=256, reduction=4).to(device)

    # ── 3. E6: PGR Module ────────────────────────────────────────────────────
    pgr = PGRModule(
        embed_dim=256,
        num_classes=num_classes,
        num_heads=4,
        dropout=0.1,
        sim_mode="cosine",
        temperature=0.07,
    ).to(device)

    # ── 4. Thin classifier head on top of refined embedding ──────────────────
    classifier = nn.Linear(256, num_classes).to(device)

    # ── 5. Forward pass ──────────────────────────────────────────────────────
    dummy_imgs = torch.randn(batch_size, 3, 224, 224, device=device)

    spatial_feats = spatial_extractor(dummy_imgs)          # (B, 768, 7, 7)
    print(f"  [E5] Spatial feature map  : {spatial_feats.shape}")

    drp_emb = drp(spatial_feats)                           # (B, 256)
    print(f"  [E5] DRP embedding        : {drp_emb.shape}")
    print(f"  [E5] ROI mask             : {drp.last_mask.shape}")

    refined_emb, sim_logits = pgr(drp_emb)                # (B, 256), (B, 5)
    print(f"  [E6] Refined embedding    : {refined_emb.shape}")
    print(f"  [E6] Similarity logits    : {sim_logits.shape}")

    cls_logits = classifier(refined_emb)                   # (B, 5)
    print(f"  [Cls] Class logits        : {cls_logits.shape}")

    # ── 6. Loss + backward ───────────────────────────────────────────────────
    labels = torch.randint(0, num_classes, (batch_size,), device=device)

    ce_loss = F.cross_entropy(cls_logits, labels)
    proto_loss = F.cross_entropy(sim_logits, labels)    # prototype alignment
    total_loss = ce_loss + 0.3 * proto_loss

    total_loss.backward()

    # ── 7. Gradient checks ───────────────────────────────────────────────────
    drp_grad_ok = drp.proj[0].weight.grad is not None
    pgr_proto_grad_ok = pgr.bank.prototypes.grad is not None
    cls_grad_ok = classifier.weight.grad is not None

    assert drp_grad_ok, "No gradient to DRP projector"
    assert pgr_proto_grad_ok, "No gradient to PGR prototypes"
    assert cls_grad_ok, "No gradient to classifier"

    print()
    print("  Losses:")
    print(f"    CE loss      : {ce_loss.item():.4f}")
    print(f"    Proto loss   : {proto_loss.item():.4f}")
    print(f"    Total        : {total_loss.item():.4f}")
    print()
    print(f"  Gradient flow to DRP projector  : {drp_grad_ok}  ✓")
    print(f"  Gradient flow to PGR prototypes : {pgr_proto_grad_ok}  ✓")
    print(f"  Gradient flow to classifier     : {cls_grad_ok}  ✓")
    print()
    print("  E5 + E6 INTEGRATION PASS  ✓\n")


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING STUB  (shows how E5 / E6 slot into your training loop)
# ══════════════════════════════════════════════════════════════════════════════
def training_stub():
    """
    Minimal training loop demonstrating:
      - EMA prototype update after each batch
      - Prototype alignment auxiliary loss
      - Gradient flow through both E5 and E6
    """
    print("=" * 60)
    print("Training Stub  (3 mini-batches)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, num_classes = 4, CONFIG["model"]["num_classes"]

    backbone = models.convnext_tiny(weights=None).to(device)
    spatial_ext = nn.Sequential(*list(backbone.features.children())).to(device)
    drp  = DRPBlock(768, 256).to(device)
    pgr  = PGRModule(256, num_classes).to(device)
    head = nn.Linear(256, num_classes).to(device)

    params = (
        list(spatial_ext.parameters())
        + list(drp.parameters())
        + list(pgr.parameters())
        + list(head.parameters())
    )
    optimizer = torch.optim.AdamW(
        params,
        lr=CONFIG["training"]["learning_rate"],
        weight_decay=CONFIG["training"]["weight_decay"],
    )

    for step in range(3):
        optimizer.zero_grad()

        imgs   = torch.randn(B, 3, 224, 224, device=device)
        labels = torch.randint(0, num_classes, (B,), device=device)

        feats         = spatial_ext(imgs)
        drp_emb       = drp(feats)
        refined, sims = pgr(drp_emb)
        logits        = head(refined)

        ce_loss    = F.cross_entropy(logits, labels)
        proto_loss = F.cross_entropy(sims, labels)
        loss       = ce_loss + 0.3 * proto_loss

        loss.backward()
        optimizer.step()

        # EMA prototype update per class in this batch
        for c in range(num_classes):
            mask = labels == c
            if mask.any():
                class_mean = drp_emb[mask].detach().mean(0)
                pgr.bank.ema_update(c, class_mean)

        print(f"  Step {step+1}/3  loss={loss.item():.4f}  "
              f"(ce={ce_loss.item():.4f}, proto={proto_loss.item():.4f})")

    print("  Training stub complete  ✓\n")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
BLOCKS = {
    "e5": [test_e5_units, validate_e5_with_backbone],
    "e6": [test_e6_units, validate_e6_with_backbone],
    "integration": [run_e5_e6_integration_test, training_stub],
}


def main():
    parser = argparse.ArgumentParser(description="Run E5/E6 block tests")
    parser.add_argument(
        "--block",
        choices=["e5", "e6", "integration", "all"],
        default="all",
        help="Which block(s) to test  (default: all)",
    )
    args = parser.parse_args()

    targets = (
        BLOCKS["e5"] + BLOCKS["e6"] + BLOCKS["integration"]
        if args.block == "all"
        else BLOCKS[args.block]
    )

    for fn in targets:
        fn()

    print("All selected tests passed  ✓")


if __name__ == "__main__":
    main()
