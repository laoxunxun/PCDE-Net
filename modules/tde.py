"""TDE — Tri-Domain Enhancement (research content 3).

Enhances the DECOUPLED target stream F_t with the background stream F_b as the
REFERENCE, in three domains, with per-pixel arbitration driven by the coupling
reliability κ from S²CPM:

  spectral domain   F_sp = F_t ⊙ m(E)
      m(E) = 2σ(MLP([E_t − Ē_bg, E_t ⊙ Ē_bg]))  → per-channel gains in (0,2).
      PHYSICS→FEATURE cross-domain operator: an 8-band physical contrast vector
      (which spectral signatures separate the endmembers) directly reweights
      feature channels. Zero-init ⇒ gain ≡ 1 (identity at start).

  spatial domain    F_sx = F_t − α ⊙ AvgPool(F_b)
      Background-referenced center–surround suppression: any target-stream
      response that resembles the LOCAL background statistics is subtracted —
      literally 背景噪声抑制. α zero-init ⇒ identity at start.

  frequency domain  F_ft = IRFFT( ReLU(|FFT(F_t)| − λ ⊙ |FFT(F_b)|) · e^{iφ_t} )
      Cross-stream amplitude subtraction with phase retention. Camouflage can
      match texture STATISTICS (amplitude spectrum) but cannot hide its own
      SHAPE (phase); removing the amplitude components shared with the actual
      background stream isolates the leaked structure. Distinct from FEDER's
      absolute frequency-band decomposition: a RELATIONAL operator between the
      two decoupled streams. λ zero-init ⇒ identity at start. fp32 island.

  arbitration       w = softmax(Conv([κ, d, r, a_t]))   (B,3,h,w)
      The coupling model decides, per pixel, which domain to trust: where
      spectral evidence is reliable (κ↑) the spectral branch dominates; where
      the mixture model fails (κ↓ — the tightest camouflage) the load shifts to
      spatial/frequency structure. This closes the loop between research
      contents 1 and 3: 耦合模型支配增强策略.

      F̂ = F_t + Fuse( Σ_b w_b · F_b )      Fuse zero-init ⇒ identity at start.

Every physics→feature entry point is zero-initialized, so the module starts as
an exact identity and LEARNS how much each domain contributes — protecting the
pretrained encoder calibration (lesson from the failed hard-replacement exp36).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralChannelGain(nn.Module):
    """m(E): endmember spectral contrast → feature channel gains in (0,2)."""

    def __init__(self, c_out: int, n_bands: int = 8, phys_dim: int = 16):
        super().__init__()
        self.n_bands = n_bands
        self.mlp = nn.Sequential(
            nn.Linear(2 * n_bands, phys_dim), nn.ReLU(inplace=True),
            nn.Linear(phys_dim, c_out))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)      # σ(0)·2 = 1 → identity at start

    def forward(self, endmembers: torch.Tensor) -> torch.Tensor:
        """endmembers: (K,8) with index 0 = target. Returns gains (1,C,1,1)."""
        e_t = endmembers[0]
        e_bg = endmembers[1:].mean(dim=0) if endmembers.shape[0] > 1 else torch.zeros_like(e_t)
        feats = torch.cat([e_t - e_bg, e_t * e_bg], dim=0)      # (2*8,)
        return (2.0 * torch.sigmoid(self.mlp(feats))).view(1, -1, 1, 1)


class TriDomainEnhancement(nn.Module):
    def __init__(self, c_in: int, use_spec: bool = True, use_spat: bool = True,
                 use_freq: bool = True, use_arbitrate: bool = True,
                 pool_k: int = 5, tau: float = 0.25, n_bands: int = 8, arb_mode: str = "learned",
                 fusion_mode: str = "avg"):
        super().__init__()
        self.use_spec, self.use_spat = use_spec, use_spat
        self.use_freq, self.use_arbitrate = use_freq, use_arbitrate
        self.fusion_mode = fusion_mode
        self.branches = [b for b, on in (("spec", use_spec), ("spat", use_spat),
                                         ("freq", use_freq)) if on]
        assert self.branches, "TDE needs at least one branch"
        self.pool_k = pool_k
        self.tau = tau
        # v10 BOUNDED enhancement: the residual is tanh-bounded to ±tau. In
        # v1–v8 the unbounded fuse let the wBCE(pos_w=50) task gradient slowly
        # turn the enhancement into a foreground amplifier (fuse norm grew
        # 0.38→1.05 across collapses). A bounded residual can only MODULATE,
        # never dominate — zero-init still gives exact identity at start.
        if use_spec:
            self.spec_gain = SpectralChannelGain(c_in, n_bands=n_bands)
        if use_spat:
            self.alpha = nn.Parameter(torch.zeros(c_in))       # zero → identity
        if use_freq:
            self.lam = nn.Parameter(torch.zeros(c_in))         # zero → identity
        self.arb_mode = arb_mode
        if arb_mode == "kappa":
            self.arb = None            # physics-parameterized fusion: zero learned params
        elif use_arbitrate and len(self.branches) > 1:
            # per-pixel domain weights from the 4 physics maps
            self.arb = nn.Conv2d(4, len(self.branches), 1)
            nn.init.zeros_(self.arb.weight)
            nn.init.zeros_(self.arb.bias)                      # uniform at start
        else:
            self.arb = None
        # SEPARATED fusion (mechanism control for the two-domain deficit):
        # each branch gets its OWN zero-init 1x1 fuse + tanh bound, branches are
        # combined AFTER the nonlinearity — no pre-fusion averaging, so the
        # multiplicative-spec × subtractive-spat dilution cannot occur.
        if fusion_mode == "sep":
            self.sep_fuse = nn.ModuleList(
                [nn.Conv2d(c_in, c_in, 1) for _ in self.branches])
            for m_ in self.sep_fuse:
                nn.init.zeros_(m_.weight); nn.init.zeros_(m_.bias)
        # residual fuse of the (weighted) branch sum — zero-init → identity
        self.fuse = nn.Conv2d(c_in, c_in, 1)
        nn.init.zeros_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)
        self.last_weights: torch.Tensor | None = None          # cached for vis

    # ------------------------------------------------------------------ freq
    def _freq_branch(self, f_t: torch.Tensor, f_b: torch.Tensor) -> torch.Tensor:
        """Cross-stream amplitude subtraction, phase retention. fp32 island."""
        with torch.autocast(device_type="cuda", enabled=False):
            t = f_t.float()
            b = f_b.float()
            T = torch.fft.rfft2(t, norm="ortho")
            Bf = torch.fft.rfft2(b, norm="ortho")
            lam = self.lam.view(1, -1, 1, 1).float()
            amp = F.relu(T.abs() - lam * Bf.abs())
            out = torch.fft.irfft2(torch.polar(amp, T.angle()),
                                   s=t.shape[-2:], norm="ortho")
            return out.to(f_t.dtype)

    # --------------------------------------------------------------- forward
    def forward(self, f_t: torch.Tensor, f_b: torch.Tensor,
                phys: torch.Tensor, endmembers: torch.Tensor) -> torch.Tensor:
        """f_t/f_b: (B,C,h,w); phys: (B,4,h,w) [a_t,r,d,κ]; endmembers (K,8)."""
        outs = []
        if self.use_spec:
            outs.append(f_t * self.spec_gain(endmembers).to(f_t.dtype))
        if self.use_spat:
            bg_local = F.avg_pool2d(f_b, self.pool_k, stride=1,
                                    padding=self.pool_k // 2, count_include_pad=False)
            outs.append(f_t - self.alpha.view(1, -1, 1, 1).to(f_t.dtype) * bg_local)
        if self.use_freq:
            outs.append(self._freq_branch(f_t, f_b))

        if self.fusion_mode == "sep":
            self.last_weights = None
            return f_t + self.tau * sum(
                torch.tanh(fm_(o)) for fm_, o in zip(self.sep_fuse, outs))
        if len(outs) == 1:
            enhanced = outs[0]
            self.last_weights = None
        elif self.arb_mode == "kappa" and len(outs) == 3:
            # PHYSICS-PARAMETERIZED TRI-DOMAIN FUSION: the coupling gate κ from
            # S²CPM decides, per pixel, whether spectral evidence (w=κ) or the
            # structural domains (spatial+frequency, w=(1-κ)/2 each) dominates.
            kap = phys[:, 3:4]                                # κ is the 4th physics map
            w = torch.cat([kap, (1 - kap) / 2, (1 - kap) / 2], dim=1)
            self.last_weights = w
            enhanced = sum(w[:, i:i+1].to(o.dtype) * o for i, o in enumerate(outs))
            return f_t + self.tau * torch.tanh(self.fuse(enhanced))
        else:
            if self.arb is not None:
                w = torch.softmax(self.arb(phys), dim=1)       # (B,nb,h,w)
            else:
                w = torch.full((f_t.shape[0], len(outs), *f_t.shape[-2:]),
                               1.0 / len(outs), device=f_t.device, dtype=f_t.dtype)
            self.last_weights = w
            enhanced = sum(w[:, i:i + 1].to(o.dtype) * o for i, o in enumerate(outs))
        return f_t + self.tau * torch.tanh(self.fuse(enhanced))
