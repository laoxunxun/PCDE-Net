"""AGD — Abundance-Guided feature Decoupling (research content 2).

Splits every fused feature map F into a TARGET stream and a BACKGROUND stream
with a gate whose SPATIAL factor is anchored to the physical quantities from
S²CPM (abundance a_t, coupling reliability κ, mixture residual r) and whose
CHANNEL factor stays free:

    s = σ(Conv([a_t, κ, r, d]))      (B,1,h,w)  physics-anchored spatial anchor
    c = σ(Conv_c(F))                 (B,C,1,1)  free channel selection
    g = s ⊙ c                        (broadcast product)
    F_t = F ⊙ g,   F_b = F ⊙ (1 − g)

Decoupling properties (enforced, not hoped-for):
  * COMPLETENESS by construction: F_t + F_b ≡ F — the split cannot lose or
    invent information (unlike erasing designs, PFNet-style).
  * ORTHOGONALITY by loss: E|cos(F_t, F_b)| → 0 (in losses.py).
  * PHYSICAL ANCHORING: the spatial layout of the split is driven by the
    latent physical model; finest-scale s is supervised with the mask (l_gate).

Control experiment for the paper (anchor="learned"): identical architecture and
parameter count, but the anchor's input is a 4-channel learned projection of F
itself instead of the physics maps — isolates the VALUE of the physical anchor.

This module is the tractable, constrained version of the proposal's "物理先验
引导的特征解耦机制": 解耦的方式(在哪分/怎么分)由物理模型决定, 网络只决定通道偏好.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AbundanceGuidedDecoupling(nn.Module):
    def __init__(self, c_in: int, anchor: str = "phys", mid: int = 16):
        super().__init__()
        assert anchor in ("phys", "learned")
        self.anchor = anchor
        # spatial anchor head — input is 4 channels either way (fair control)
        if anchor == "learned":
            self.proj = nn.Conv2d(c_in, 4, 1)               # learned 4-ch surrogate
        else:
            self.proj = None                                 # physics maps come in
        self.s_head = nn.Sequential(
            nn.Conv2d(4, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 1))                            # anchor logits
        # free channel-selection factor (bottleneck to keep params modest)
        self.c_head = nn.Sequential(
            nn.Conv2d(c_in, max(c_in // 4, 8), 1), nn.ReLU(inplace=True),
            nn.Conv2d(max(c_in // 4, 8), c_in, 1))

    def forward(self, f: torch.Tensor, phys: torch.Tensor | None
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """f: (B,C,h,w) fused feature; phys: (B,4,h,w) physics maps (anchor="phys").
        Returns (F_t, F_b, s_logits at this scale).

        The gate is SOFT-ROUTED: s = 0.5 + 0.25·tanh(head) ∈ [0.25, 0.75] —
        each stream always keeps ≥25% of the feature (v10 bounded decoupling;
        an unconstrained sigmoid gate let the pos_w=50 task gradient hollow out
        one stream, the failure behind the v1–v8 delayed collapses).
        Completeness F_t + F_b = F still holds exactly."""
        src = self.proj(f) if self.anchor == "learned" else phys
        assert src is not None, "anchor='phys' requires physics maps"
        s_logit = self.s_head(src)                           # (B,1,h,w)
        s = 0.5 + 0.25 * torch.tanh(s_logit)
        c_logit = self.c_head(f).mean(dim=(2, 3), keepdim=True)   # (B,C,1,1)
        g = s * torch.sigmoid(c_logit)                       # broadcast → (B,C,h,w)
        f_t = f * g
        f_b = f * (1.0 - g)
        return f_t, f_b, s_logit
