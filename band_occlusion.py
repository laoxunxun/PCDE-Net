"""Band-occlusion analysis — which spectral bands carry the discriminative signal.

For each of the 8 bands: replace that band with its mean (z-score 0 = average
reflectance) over the WHOLE val set, measure val-Sm. Drop vs clean baseline =
band importance. Physics-method-exclusive analysis (RGB methods have no bands
to occlude); also feeds the band-budget selection (which bands to keep at 5/3).

Inference-only, single-scale, val split (103 imgs). GPU-light (b1) — safe next
to a training run.

Usage: python trible2one/band_occlusion.py --exp exp01_v11_pvt512
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

BANDS = ["Violet422", "Blue487", "Green550", "Orange602", "Red660",
         "RedEdge725", "NIR785", "NIR887"]


def val_sm(model, ds, dev, occlude=None):
    sms, maes = [], []
    with torch.no_grad():
        for i in range(len(ds)):
            cube, gt, _ = ds[i]
            x = cube.unsqueeze(0).to(dev)
            if occlude is not None:
                x = x.clone()
                x[:, occlude] = 0.0                     # z-score 0 = band mean
            p = torch.sigmoid(model(x)[0, 0])
            from shared.metrics import smeasure
            sms.append(float(smeasure(p, gt[0].to(p.device))))
            maes.append(float((p - gt[0].to(p.device)).abs().mean()))
    return float(np.mean(sms)), float(np.mean(maes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    a = ap.parse_args()
    import yaml
    from run_experiment import build_model
    from shared.dataset import MCODDataset
    from shared.utils import get_device

    dev = get_device()
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", f"{a.exp}.yaml"),
                              encoding="utf-8"))
    model = build_model(cfg).to(dev)
    ck = torch.load(os.path.join(ROOT, "trible2one/experiments", a.exp, "best.pth"),
                    map_location=dev)
    model.load_state_dict(ck["model"])
    model.eval()
    ds = MCODDataset(cfg["data"]["root"], "val", cfg["data"]["size"], augment=False,
                     cache_dir=cfg["data"].get("cache"))

    base_sm, base_mae = val_sm(model, ds, dev)
    print(f"clean:            val-Sm {base_sm:.4f}  MAE {base_mae:.4f}")
    res = {"clean": {"sm": base_sm, "mae": base_mae}, "occluded": {}}
    for b in range(8):
        sm, mae = val_sm(model, ds, dev, occlude=b)
        drop = base_sm - sm
        res["occluded"][BANDS[b]] = {"sm": sm, "mae": mae, "drop": drop}
        print(f"occlude {BANDS[b]:>10}: val-Sm {sm:.4f}  (Δ {drop:+.4f})")

    out = os.path.join(ROOT, "trible2one/experiments", a.exp, "band_occlusion.json")
    json.dump(res, open(out, "w"), indent=2)
    ranked = sorted(res["occluded"].items(), key=lambda kv: -kv[1]["drop"])
    print("\nimportance ranking (drop when occluded):")
    print("  " + " > ".join(f"{k}({v['drop']:+.4f})" for k, v in ranked))
    print("saved", out)


if __name__ == "__main__":
    main()
