"""S²CPM — Spectral–Spatial Coupled Physics Model (differentiable front-end).

Research content 1 of the proposal: the multi-dimensional physical model.

Observation model (per band k):   X_k = g_k·ρ_k + s_k + n_k
Linear spectral mixture (LSMA):   ρ(p) ≈ Σ_k A_k(p)·E_k,  A≥0, ΣA=1, E≥0

This module makes the mixture model DIFFERENTIABLE and LEARNED (endmembers E are
latent physical variables, abundances A from a light head), then derives four
physics maps that drive the rest of the network:

  a_t    target-material abundance  A_1          — physical soft mask
  r      mixture residual ||ρ−ΣA·E||/√8          — where the SPECTRAL model FAILS
         (camouflage is spectrally tight exactly there)
  d      spectral separability: cos∠(ρ,E_t) − max_{k≥2} cos∠(ρ,E_k)
                                                     — spectral-contrast strength
  κ      coupling reliability gate σ(MLP([d, r])) — the EXECUTABLE form of the
         spectral–spatial coupling: κ high → spectral evidence trustworthy;
         κ low  → discrimination must come from spatial/frequency structure.

Physics enters as STRUCTURE, not as a regularizer: a_t anchors the decoupling
gate (AGD), κ arbitrates the tri-domain enhancement (TDE). A plain backbone
cannot reproduce these operators because they depend on the latent endmembers
constrained by reconstruction + abundance supervision (global consistency).

All computation happens on a pooled work-size cube in a float32 island (AMP-
safe); maps are later upsampled to feature scales by the caller.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .physics_common import to_reflectance, N_BANDS


class S2CPM(nn.Module):
    def __init__(self, n_bands: int = N_BANDS, K: int = 4, work_size: int = 96,
                 endmember_init: torch.Tensor | None = None, mid: int = 32):
        super().__init__()
        self.K, self.n_bands, self.work_size = K, n_bands, work_size
        if endmember_init is not None:
            assert endmember_init.shape == (K, n_bands), \
                f"endmember_init must be {K}x{n_bands}, got {endmember_init.shape}"
            init = endmember_init.detach().float().clamp(min=0.02, max=0.98)
        else:
            # neutral init: target slightly NIR-flat (8-band) / slightly
            # green-shifted (RGB), backgrounds spread
            init = torch.full((K, n_bands), 0.45)
            init[0] = torch.linspace(0.45, 0.38, n_bands)
            for k in range(1, K):
                init[k] = (init[k] + 0.15 * torch.cos(
                    torch.linspace(0, k * 2.5, n_bands))).clamp(0.05, 0.95)
        # positivity via clamp (identity-gradient region for values > 1e-3);
        # data-driven init keeps us strictly inside that region
        self.E = nn.Parameter(init)

        # abundance estimator: light conv head on the reflectance cube
        self.abund_head = nn.Sequential(
            nn.Conv2d(n_bands, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, K, 1))

        # coupling reliability gate κ = σ(MLP([d, r]))  (2 → mid → 1)
        # zero-init last layer → κ ≈ σ(0) = 0.5 at start (unbiased arbitration
        # while the physical quantities themselves calibrate)
        self.kappa_head = nn.Sequential(
            nn.Conv2d(2, mid, 1), nn.ReLU(inplace=True), nn.Conv2d(mid, 1, 1))
        nn.init.zeros_(self.kappa_head[-1].weight)
        nn.init.zeros_(self.kappa_head[-1].bias)

        # caches for the loss contract
        self.last_x: torch.Tensor | None = None      # pooled reflectance
        self.last_xhat: torch.Tensor | None = None   # mixture reconstruction
        self.last_A_logits: torch.Tensor | None = None
        self.last_maps: dict[str, torch.Tensor] = {}

    # ---------------------------------------------------------------- utils
    def _unmix(self, x01: torch.Tensor):
        """Float32 island: abundances + reconstruction + physics maps.

        x01: (B,8,H,W) reflectance in [0,1] at work size."""
        A_logits = self.abund_head(x01)                      # (B,K,h,w)
        A = F.softmax(A_logits, dim=1)                       # non-neg, sum-to-one
        E = self.E.clamp(min=1e-3)                           # (K,8) non-negative
        xhat = torch.einsum("bkhw,ke->behw", A, E)           # (B,8,h,w) mixture
        return A_logits, A, E, xhat

    def _separability(self, x01: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        """d = cos(ρ,E_t) − max_{k≥2} cos(ρ,E_k)  per pixel (B,1,h,w)."""
        xN = F.normalize(x01, dim=1)                         # (B,8,h,w)
        eN = F.normalize(E, dim=1)                           # (K,8)
        cs = torch.einsum("behw,ke->bkhw", xN, eN).clamp(-1 + 1e-6, 1 - 1e-6)
        d = cs[:, 0:1] - cs[:, 1:].max(dim=1, keepdim=True).values
        return d

    # -------------------------------------------------------------- forward
    def forward(self, x_z: torch.Tensor) -> dict[str, torch.Tensor]:
        """x_z: (B,8,H,W) z-scored cube. Returns physics maps at work_size."""
        with torch.autocast(device_type="cuda", enabled=False):
            x01 = to_reflectance(x_z).float()
            xs = F.adaptive_avg_pool2d(x01, self.work_size)  # (B,8,ws,ws)
            A_logits, A, E, xhat = self._unmix(xs)
            # residual: spectral failure of the mixture model, per pixel
            r = (xs - xhat).pow(2).sum(dim=1, keepdim=True).sqrt() / (self.n_bands ** 0.5)
            d = self._separability(xs, E)
            kappa = torch.sigmoid(self.kappa_head(torch.cat([d, r.clamp(0, 1)], dim=1)))
            a_t = A[:, 0:1]

        self.last_x, self.last_xhat = xs, xhat
        self.last_A_logits = A_logits
        self.last_maps = {"a_t": a_t, "r": r, "d": d, "kappa": kappa}
        return self.last_maps

    # --------------------------------------------------------------- losses
    def recon_loss(self) -> torch.Tensor:
        """Mixture-model self-consistency (physics consistency loss)."""
        if self.last_x is None:
            return torch.tensor(0.0, device=self.E.device)
        return F.mse_loss(self.last_xhat, self.last_x)

    def endmember_diversity(self) -> torch.Tensor:
        """Negative mean pairwise endmember distance (anti-collapse)."""
        e = self.E.clamp(min=1e-3)
        pd = torch.cdist(e, e)
        eye = torch.eye(self.K, device=e.device).bool()
        return -pd[~eye].mean()

    def phys_maps(self, size: tuple[int, int]) -> torch.Tensor:
        """Physics maps stacked (B,4,h',w') upsampled to `size`: [a_t, r, d, κ].

        Gradients flow back to the endmembers / abundance head — required for the
        physics losses to shape the latent physical model."""
        m = torch.cat([self.last_maps["a_t"], self.last_maps["r"],
                       self.last_maps["d"], self.last_maps["kappa"]], dim=1)
        return F.interpolate(m, size=size, mode="bilinear", align_corners=False)
