"""Shared physics constants & helpers for PCDE-Net (trible2one).

Single source of truth for band layout and the z-score -> reflectance inverse.
Stats come from shared.dataset (computed once over the full train set)."""
from __future__ import annotations

import torch

V_BANDS = [0, 1, 2, 3, 4]       # visible: Violet422 Blue487 Green550 Orange602 Red660
NV_BANDS = [5, 6, 7]            # non-visible: RedEdge725 NIR1_785 NIR2_887
N_BANDS = 8

from shared.dataset import BAND_MEAN, BAND_STD
_MEAN = torch.tensor(BAND_MEAN).view(1, 8, 1, 1)
_STD = torch.tensor(BAND_STD).view(1, 8, 1, 1)
# RGB-COD branch (COD10K-train stats; keep in sync with rgb_dataset)
_RGB_MEAN3 = torch.tensor([0.394, 0.423, 0.398]).view(1, 3, 1, 1)
_RGB_STD3 = torch.tensor([0.266, 0.259, 0.262]).view(1, 3, 1, 1)


def to_reflectance(x_z: torch.Tensor) -> torch.Tensor:
    """z-scored cube (B,8,H,W) or RGB image (B,3,H,W) -> [0,1] reflectance."""
    if x_z.shape[1] == 3:
        return (x_z * _RGB_STD3.to(x_z.device) + _RGB_MEAN3.to(x_z.device)).clamp(0.0, 1.0)
    return (x_z * _STD.to(x_z.device) + _MEAN.to(x_z.device)).clamp(0.0, 1.0)
