"""PCE decoder — top-down pyramid over the ENHANCED target-stream features.

Follows the group's proven strong-decoder recipe (multi-scale fusion + residual
refinement + deep supervision + boundary branch — the capacity that lets Swin
features become accurate masks), with two PCDE-specific changes:

  * the pyramid consumes the TDE-enhanced TARGET stream per scale (the physics
    pipeline lives upstream, so the decoder stays clean);
  * the final head is a PHYSICS-GUIDED refinement: input = [coarse mask,
    a_t, r, d, κ, reflectance cube] — the latent physical maps tell the head
    where the mixture model separates material (a_t) and where spectral
    evidence failed (r, κ), which is exactly the information needed to resolve
    fuzzy camouflage boundaries.

Returns (main_logits, [aux5, aux4, aux3], edge_logits) — the same contract as
the shared infrastructure's deep-supervision / boundary losses expect.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RB(nn.Module):
    """Residual refine block (capacity)."""

    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)

    def forward(self, x):
        return F.relu(x + self.bn2(self.c2(F.relu(self.bn1(self.c1(x))))), inplace=True)


class PCEDecoder(nn.Module):
    def __init__(self, chans=(96, 192, 384, 768), c: int = 128, dual: bool = False):
        """chans: encoder channel contract, shallow→deep (Swin: 96/192/384/768).
        dual=True: every lateral consumes concat[target-stream, background-stream]
        (2C→C). Rationale (v1–v7 collapse forensics): decoding from the gated
        target stream alone systematically weakens the decoder's input and its
        background evidence, and the wBCE(pos_w=50) pressure eventually drives
        the head into an all-foreground basin. With dual laterals the decoder
        keeps FULL-strength, both-stream information (like the stable baseline)
        while all physics (decoupling + tri-domain enhancement) still happens
        on the target stream before the laterals."""
        super().__init__()
        c2, c3, c4, c5 = chans
        m = 2 if dual else 1
        self.dual = dual
        self.lat5 = nn.Conv2d(c5 * m, c, 1)
        self.lat4 = nn.Conv2d(c4 * m, c, 1)
        self.lat3 = nn.Conv2d(c3 * m, c, 1)
        self.lat2 = nn.Conv2d(c2 * m, c, 1)
        self.ref5, self.ref4, self.ref3, self.ref2 = RB(c), RB(c), RB(c), RB(c)
        self.aux5 = nn.Conv2d(c, 1, 1)
        self.aux4 = nn.Conv2d(c, 1, 1)
        self.aux3 = nn.Conv2d(c, 1, 1)
        self.bnd_feat = nn.Sequential(nn.Conv2d(c, c, 3, padding=1),
                                      nn.BatchNorm2d(c), nn.ReLU(inplace=True))
        self.bnd_head = nn.Conv2d(c, 1, 1)
        self.main = nn.Sequential(nn.Conv2d(2 * c, c, 3, padding=1),
                                  nn.BatchNorm2d(c), nn.ReLU(inplace=True), RB(c))
        self.head = nn.Conv2d(c, 1, 1)

    def forward(self, feats, feats_b=None):
        """feats: [F̂_2..F̂_5] enhanced target-stream (shallow→deep);
        feats_b: matching background-stream list (required iff built dual).
        Returns (main_logits, [aux5, aux4, aux3], edge_logits)."""
        def lat(mod, f, i):
            if self.dual:
                return mod(torch.cat([f, feats_b[i]], dim=1))
            return mod(f)

        f2, f3, f4, f5 = feats
        d5 = self.ref5(lat(self.lat5, f5, 3))
        d4 = self.ref4(lat(self.lat4, f4, 2) + F.interpolate(d5, size=f4.shape[-2:],
                                                     mode="bilinear", align_corners=False))
        d3 = self.ref3(lat(self.lat3, f3, 1) + F.interpolate(d4, size=f3.shape[-2:],
                                                     mode="bilinear", align_corners=False))
        p2 = lat(self.lat2, f2, 0) + F.interpolate(d3, size=f2.shape[-2:],
                                           mode="bilinear", align_corners=False)
        d2 = self.ref2(p2)
        bfeat = self.bnd_feat(d2)
        edge = self.bnd_head(bfeat)
        m = self.main(torch.cat([d2, bfeat], dim=1))
        return self.head(m), [self.aux5(d5), self.aux4(d4), self.aux3(d3)], edge


class PhysicsRefineHead(nn.Module):
    """Final residual refinement guided by the latent physical maps.

    input = [coarse mask, a_t, r, d, κ, z-scored cube(8)] (13 ch) or
            [coarse mask, z-scored cube(8)] (9 ch) for the physics-off baseline —
    the channel contract changes so the baseline head CANNOT quietly relearn a
    physics prior from the raw cube alone at the same capacity. The cube input
    is z-scored (not reflectance) to match the proven SDS-Net refine recipe."""

    def __init__(self, use_physics: bool = True, mid: int = 32, n_bands: int = 8):
        super().__init__()
        in_ch = 1 + (4 if use_physics else 0) + n_bands
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 1))
        # zero-init the output layer: exact identity at start (see v1 forensics)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, mask: torch.Tensor, x_z: torch.Tensor,
                phys_full: torch.Tensor | None) -> torch.Tensor:
        ins = [mask, x_z] if phys_full is None else [mask, phys_full, x_z]
        r = self.net(torch.cat(ins, dim=1))
        # Zero-mean the residual PER IMAGE: refinement is spatial STRUCTURE,
        # not global calibration (blocks the all-foreground bias drift).
        return r - r.mean(dim=(2, 3), keepdim=True)
