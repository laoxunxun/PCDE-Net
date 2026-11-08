"""Multi-scale TTA offline evaluation for a trained PCDE-Net checkpoint.

Standard COD reporting = single-scale 4-flip (what run_experiment already
produces). This script adds the ms-TTA secondary row: scales x 4 flips,
prediction averaged in probability space, evaluated with official pysodmetrics.

Usage:
  python trible2one/eval_ms_tta.py --exp exp01_v11_pvt512 \
      [--config configs/exp01_v11_pvt512.yaml] [--scales 0.75,1.0,1.25]
Writes metrics_ms_tta.json next to the experiment's metrics.json.
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
from PIL import Image
import py_sod_metrics as PSM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--scales", default="0.75,1.0,1.25")
    ap.add_argument("--ckpt", default=None)
    a = ap.parse_args()

    import yaml
    cfg_path = a.config or os.path.join(ROOT, "configs", f"{a.exp}.yaml")
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    _mcod = os.environ.get("MCOD_ROOT")
    if _mcod:
        cfg["data"]["root"] = _mcod

    from shared.dataset import MCODDataset, load_mask
    from shared.utils import get_device
    from run_experiment import build_model

    dev = get_device()
    model = build_model(cfg).to(dev)
    ck = torch.load(a.ckpt or os.path.join(
        ROOT, "trible2one/experiments", a.exp, "best.pth"), map_location=dev)
    model.load_state_dict(ck["model"])
    model.eval()

    ds = MCODDataset(cfg["data"]["root"], "test", cfg["data"]["size"], augment=False,
                     cache_dir=cfg["data"].get("cache"))
    gt_dir = os.path.join(cfg["data"]["root"], "TestDataset", "GT")
    scales = [float(s) for s in a.scales.split(",")]

    m_m = PSM.MAE(); sm = PSM.Smeasure(); em = PSM.Emeasure()
    wf = PSM.WeightedFmeasure(); fm = PSM.Fmeasure()
    with torch.no_grad():
        for i in range(len(ds)):
            cube, _, name = ds[i]
            base = cube.unsqueeze(0).to(dev)
            preds = []
            for sc in scales:
                if sc != 1.0:
                    h = w = int(round(cfg["data"]["size"] * sc))
                    x = F.interpolate(base, size=(h, w), mode="bilinear",
                                      align_corners=False)
                    if sc > 1.0:                      # center crop back to train size
                        top = (h - cfg["data"]["size"]) // 2
                        left = (w - cfg["data"]["size"]) // 2
                        x = x[..., top:top + cfg["data"]["size"],
                               left:left + cfg["data"]["size"]]
                else:
                    x = base
                for flip in [None, [2], [3], [2, 3]]:
                    xi = torch.flip(x, dims=flip) if flip else x
                    p = torch.sigmoid(model(xi)[0, 0]).cpu()
                    if flip:
                        p = torch.flip(p, dims=[d - 2 for d in flip])
                    if sc != 1.0:                      # resize prediction back
                        p = F.interpolate(p[None, None], size=(
                            cfg["data"]["size"], cfg["data"]["size"]),
                            mode="bilinear", align_corners=False)[0, 0]
                    preds.append(p)
            pred512 = torch.stack(preds).mean(0)
            gt = load_mask(os.path.join(gt_dir, name + ".jpg")); Hn, Wn = gt.shape
            pred = F.interpolate(pred512[None, None], size=(Hn, Wn), mode="bilinear",
                                 align_corners=False)[0, 0].numpy()
            pred_u = (pred * 255).clip(0, 255).astype(np.uint8)
            gt_u = ((gt > 0.5).astype(np.uint8) * 255)
            for M in (m_m, sm, em, wf, fm):
                M.step(pred=pred_u, gt=gt_u)
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(ds)}", flush=True)

    def cs(d):
        d = d if isinstance(d, dict) else {"curve": d}
        c = np.asarray(d["curve"]) if "curve" in d else np.asarray(d)
        mean = float(c.mean()) if c.ndim else float(c)
        mx = float(c.max()) if c.ndim else mean
        return mean, mx
    em_mean, em_max = cs(em.get_results()["em"])
    fb_mean, fb_max = cs(fm.get_results()["fm"])
    off = {"MAE": m_m.get_results()["mae"], "Sm": sm.get_results()["sm"],
           "Em": em_mean, "Em_max": em_max, "wF": wf.get_results()["wfm"],
           "Fb": fb_mean, "Fb_max": fb_max,
           "scales": scales, "flips": 4}
    out = os.path.join(ROOT, "trible2one/experiments", a.exp, "metrics_ms_tta.json")
    json.dump(off, open(out, "w"), indent=2, default=float)
    print("MS-TTA OFFICIAL:", " ".join(f"{k}={v:.4f}" for k, v in off.items()
                                       if isinstance(v, float)), flush=True)


if __name__ == "__main__":
    main()
