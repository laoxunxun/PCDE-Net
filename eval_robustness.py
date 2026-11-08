"""Spectral-shift robustness: physics vs no-physics under illumination/atmospheric
perturbation (inference-only).

Perturbation (physically plausible, wavelength-smooth — same family as PGSA):
  gain_k = 1 + a_k, shift_k = b_k, with a/b smooth across bands (3-tap averaged
  random draws, amplitude g/s). Evaluates Sm on the first N test images under
  K random perturbation seeds, for each checkpoint given.

If PCDE-Net degrades less than the no-physics baseline under spectral shift,
the physics front provides distribution-shift robustness — a claim RGB-style
models cannot make.

Usage: python trible2one/eval_robustness.py --exps exp01_v11_pvt512 exp00_baseline_512pvt [--n 100] [--k 3]
"""
from __future__ import annotations

import os
import sys
import json
import argparse

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "modules"))

import numpy as np
import torch
import torch.nn.functional as F


def smooth(v: torch.Tensor) -> torch.Tensor:
    xp = torch.cat([v[:1], v, v[-1:]])
    return (xp[:-2] + xp[1:-1] + xp[2:]) / 3.0


def perturb(cube01: torch.Tensor, gain_amp: float, shift_amp: float,
            gen: torch.Generator) -> torch.Tensor:
    g = smooth(1.0 + (torch.rand(8, generator=gen) * 2 - 1) * gain_amp)
    s = smooth((torch.rand(8, generator=gen) * 2 - 1) * shift_amp)
    return (cube01 * g.view(8, 1, 1) + s.view(8, 1, 1)).clamp(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="+",
                    default=["exp01_v11_pvt512", "exp00_baseline_512pvt"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--gain", type=float, default=0.15)
    ap.add_argument("--shift", type=float, default=0.03)
    a = ap.parse_args()

    import yaml
    from run_experiment import build_model
    from shared.dataset import MCODDataset, BAND_MEAN, BAND_STD
    from shared.metrics import smeasure
    from shared.utils import get_device

    dev = get_device()
    mean = torch.tensor(BAND_MEAN).view(8, 1, 1)
    std = torch.tensor(BAND_STD).view(8, 1, 1)

    results = {}
    for exp in a.exps:
        cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", f"{exp}.yaml"),
                                  encoding="utf-8"))
        model = build_model(cfg).to(dev)
        ck = torch.load(os.path.join(ROOT, "trible2one/experiments", exp, "best.pth"),
                        map_location=dev)
        model.load_state_dict(ck["model"])
        model.eval()
        ds = MCODDataset(cfg["data"]["root"], "test", cfg["data"]["size"],
                         augment=False, cache_dir=cfg["data"].get("cache"))
        idxs = np.linspace(0, len(ds) - 1, a.n).astype(int)

        def run_all(perturbed):
            sms = []
            with torch.no_grad():
                for j, i in enumerate(idxs):
                    cube, gt, _ = ds[i]
                    x01 = cube * std + mean                       # reflectance
                    if perturbed is not None:
                        x01 = perturbed[j]
                    xz = ((x01 - mean) / std).unsqueeze(0).to(dev)
                    p = torch.sigmoid(model(xz)[0, 0]).cpu()
                    sms.append(float(smeasure(p, gt[0])))
            return float(np.mean(sms))

        rows = {}
        rows["clean"] = run_all(None)
        for k in range(a.k):
            pert = []
            for j, i in enumerate(idxs):
                cube, _, _ = ds[i]
                gen = torch.Generator().manual_seed(int(1234 + 1000 * k + i))
                pert.append(perturb(cube * std + mean, a.gain, a.shift, gen))
            rows[f"perturb{k+1}"] = run_all(pert)
        rows["perturb_mean"] = float(np.mean([rows[f"perturb{k+1}"] for k in range(a.k)]))
        rows["drop"] = rows["clean"] - rows["perturb_mean"]
        results[exp] = rows
        print(f"{exp}: clean {rows['clean']:.4f} -> perturbed {rows['perturb_mean']:.4f} "
              f"(drop {rows['drop']:.4f})")

    out = os.path.join(ROOT, "trible2one/experiments/robustness_spectral.json")
    old = json.load(open(out)) if os.path.exists(out) else {}
    old.update(results)
    json.dump(old, open(out, "w"), indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()
