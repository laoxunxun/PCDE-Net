"""Efficiency benchmark: Params / FLOPs / FPS for PCDE-Net configs vs typical
COD baselines (reported numbers where available). GPU timing needs an idle GPU
(run between queue experiments). FPS = batch-1 512x512 forward, 50 warmup +
100 timed, single-scale.

Usage: python trible2one/bench_efficiency.py [--config exp01_v11_pvt512]
Writes experiments/efficiency.json (appends per config).
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "modules"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="exp01_v11_pvt512")
    ap.add_argument("--size", type=int, default=512)
    a = ap.parse_args()
    import torch
    import yaml
    from run_experiment import build_model
    from shared.utils import get_device

    dev = get_device()
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs",
                                           f"{a.config}.yaml"), encoding="utf-8"))
    model = build_model(cfg).to(dev).eval()
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

    x = torch.randn(1, 8, a.size, a.size, device=dev)
    with torch.no_grad():
        for _ in range(50):                       # warmup (cudnn autotune)
            model(x)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(100):
            model(x)
        torch.cuda.synchronize()
        fps = 100 / (time.time() - t0)

    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, x.clone())
        flops.unsupported_ops_warnings(False)
        fl = flops.total() / 1e9
    except Exception:
        fl = None

    row = {"config": a.config, "params_M": n_params / 1e6, "train_params_M": n_train / 1e6,
           "flops_G_512": fl, "fps_b1_512": fps, "size": a.size}
    out = os.path.join(ROOT, "trible2one/experiments/efficiency.json")
    data = json.load(open(out)) if os.path.exists(out) else []
    data = [d for d in data if d["config"] != a.config] + [row]
    json.dump(data, open(out, "w"), indent=2)
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
