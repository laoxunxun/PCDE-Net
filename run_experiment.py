"""PCDE-Net unified experiment driver (mirrors the MCOD clean protocol).

    build from config -> train (cosine LR, AMP, BN-guard) -> val-selected best
    -> single-pass official eval (pysodmetrics + 4-way TTA) -> append LOG.md.

Run:  python -u run_experiment.py --name exp01_pcde_full \
          --config configs/exp01_pcde_full.yaml
cloud: export MCOD_ROOT=/root/autodl-tmp/MCOD  (same command otherwise)
"""
from __future__ import annotations

import sys, os, json, math, argparse, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "modules"))

import yaml, numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

import py_sod_metrics as PSM

_v = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
if _v >= (2, 1):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

EXP_ROOT = os.path.join(ROOT, "experiments")


def build_model(cfg):
    cm = cfg["model"]
    if cm.get("type", "pcde") in ("pcde", "dual_baseline"):
        from pcde_net import PCDENet
        return PCDENet(
            v_arch=cm.get("v_arch", "swin_small_patch4_window7_224"),
            nv_arch=cm.get("nv_arch", "swin_tiny_patch4_window7_224"),
            backbone_type=cm.get("backbone_type", "swin"),
            pretrained=cm.get("pretrained", True),
            img_size=cfg.get("data", {}).get("size", 384),
            k_endmembers=cm.get("k_endmembers", 4),
            work_size=cm.get("work_size", 96),
            use_s2cpm=cm.get("use_s2cpm", cm.get("type", "pcde") == "pcde"),
            use_decouple=cm.get("use_decouple", True),
            anchor=cm.get("anchor", "phys"),
            use_spec=cm.get("use_spec", True),
            use_spat=cm.get("use_spat", True),
            use_freq=cm.get("use_freq", True),
            use_arbitrate=cm.get("use_arbitrate", True),
            tde_tau=cm.get("tde_tau", 0.25), arb_mode=cm.get("arb_mode", "learned"),
            fusion_mode=cm.get("fusion_mode", "avg"),
            rgb=cm.get("rgb", False),
            endmember_init=cm.get("endmember_init") or os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "assets", "endmember_init.npz"),
            grad_ckpt=cm.get("grad_ckpt", True),
            v_pretrained=cm.get("v_pretrained", None),
            nv_pretrained=cm.get("nv_pretrained", None))
    raise ValueError(f"unknown model type: {cm.get('type')}")


def evaluate_full(model, ds, gt_dir, preds_dir, dev):
    """Official single-pass eval with 4-way flip TTA (identical to mso-net)."""
    from shared.dataset import load_mask
    os.makedirs(preds_dir, exist_ok=True)
    n = len(ds)
    m_m = PSM.MAE(); sm = PSM.Smeasure(); em = PSM.Emeasure()
    wf = PSM.WeightedFmeasure(); fm = PSM.Fmeasure()
    model.eval()
    with torch.no_grad():
        for i in range(n):
            cube, _, name = ds[i]
            inp = cube.unsqueeze(0).to(dev)
            preds = []
            for flip_dims in [None, [2], [3], [2, 3]]:
                xi = torch.flip(inp, dims=flip_dims) if flip_dims else inp
                pi = torch.sigmoid(model(xi)[0, 0]).cpu()
                if flip_dims:
                    pi = torch.flip(pi, dims=[d - 2 for d in flip_dims])
                preds.append(pi)
            pred352 = sum(preds) / len(preds)
            gt = load_mask(os.path.join(gt_dir, name + ".jpg")); Hn, Wn = gt.shape
            pred = F.interpolate(pred352[None, None], size=(Hn, Wn), mode="bilinear",
                                 align_corners=False)[0, 0].numpy()
            Image.fromarray((pred * 255).clip(0, 255).astype(np.uint8)).save(
                os.path.join(preds_dir, name + ".png"))
            pred_u = (pred * 255).clip(0, 255).astype(np.uint8)
            gt_u = ((gt > 0.5).astype(np.uint8) * 255)
            for M in (m_m, sm, em, wf, fm):
                M.step(pred=pred_u, gt=gt_u)
            if (i + 1) % 100 == 0:
                print(f"  eval {i+1}/{n}", flush=True)

    def cs(d):
        d = d if isinstance(d, dict) else {"curve": d}
        c = np.asarray(d["curve"]) if "curve" in d else np.asarray(d)
        mean = float(c.mean()) if c.ndim else float(c)
        mx = float(c.max()) if c.ndim else mean
        return mean, mx
    em_mean, em_max = cs(em.get_results()["em"])
    fb_mean, fb_max = cs(fm.get_results()["fm"])
    return {"MAE": m_m.get_results()["mae"], "Sm": sm.get_results()["sm"],
            "Em": em_mean, "Em_max": em_max, "wF": wf.get_results()["wfm"],
            "Fb": fb_mean, "Fb_max": fb_max}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true",
                    help="continue from experiments/<name>/ckpt_last.pth if present")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _mcod = os.environ.get("MCOD_ROOT")                     # cloud portability
    if _mcod:
        cfg["data"]["root"] = _mcod
        if cfg["data"].get("cache"):
            cfg["data"]["cache"] = os.path.join(
                _mcod, os.path.basename(cfg["data"]["cache"].replace("\\", "/")))

    from shared.dataset import MCODDataset
    from shared.engine import train_one_epoch, evaluate
    from shared.metrics import evaluate_image
    from shared.utils import set_seed, get_device
    from losses import PCELoss

    dev = get_device()
    set_seed(cfg.get("seed", 42))
    out = os.path.join(EXP_ROOT, args.name)
    os.makedirs(out, exist_ok=True)
    cd = cfg["data"].get("cache")
    val_ds = None
    if cfg["data"].get("pgma") and cfg["data"].get("dataset_type") != "rgbcod":
        # physics-guided mixing augmentation (harder-camouflage synthesis)
        from pgma import PGMAMix, PCDEDataset
        import numpy as _np
        em_file = cfg["model"].get("endmember_init") or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "assets", "endmember_init.npz")
        mix = PGMAMix(em_file, p=cfg["data"].get("pgma_p", 0.5),
                      alpha_mean_lo=cfg["data"].get("pgma_mean_lo", 0.10),
                      alpha_mean_hi=cfg["data"].get("pgma_mean_hi", 0.35),
                      alpha_cap=cfg["data"].get("pgma_cap", 0.45),
                      mode=cfg["data"].get("pgma_mode", "mix"))
        train_ds = PCDEDataset(cfg["data"]["root"], "train", cfg["data"]["size"],
                               augment=True, multiscale=cfg["data"].get("multiscale", False),
                               physics_aug=cfg["data"].get("physics_aug", True),
                               physics_aug_mode=cfg["data"].get("physics_aug_mode", "full"),
                               pgma=mix)
        print("[pgma] physics-guided mixing augmentation ENABLED "
              f"(mode={mix.mode}, p={mix.p}, ā∈[{mix.a_lo},{mix.a_hi}], cap={mix.cap})",
              flush=True)
    elif cfg["data"].get("dataset_type") != "rgbcod":
        train_ds = MCODDataset(cfg["data"]["root"], "train", cfg["data"]["size"], augment=True,
                               cache_dir=cd, multiscale=cfg["data"].get("multiscale", False),
                               physics_aug=cfg["data"].get("physics_aug", False),
                               physics_aug_mode=cfg["data"].get("physics_aug_mode", "full"))
    if cfg["data"].get("dataset_type") == "rgbcod":
        from rgb_dataset import RGBCODTrain
        from pgma import PGMAMix as _PM
        _mix = None
        if cfg["data"].get("pgma"):
            _mix = _PM(None, p=0.5)
            print("[pgma] RGB physics-guided mixing augmentation ENABLED", flush=True)
        train_ds = RGBCODTrain(cfg["data"]["root"], cfg["data"]["size"],
                               multiscale=cfg["data"].get("multiscale", True),
                               pgma=_mix)
        print(f"[rgbcod] COD10K-train camo: {len(train_ds)} images", flush=True)
        # val = tail 5% for model selection
        import torch.utils.data as U
        from rgb_dataset import RGBCODTrain as _T
        n_val = max(1, int(len(train_ds) * 0.05))
        idx = list(range(len(train_ds)))
        val_full = _T(cfg["data"]["root"], cfg["data"]["size"],
                      multiscale=False, augment=False)
        train_ds, val_ds = U.Subset(train_ds, idx[:-n_val]), U.Subset(val_full, idx[-n_val:])
    sub = cfg["data"].get("train_subset")
    if sub:
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, list(range(int(sub))))
        print(f"[subset] training on first {sub} train images", flush=True)
    test_ds = None if cfg["data"].get("dataset_type") == "rgbcod" else MCODDataset(
        cfg["data"]["root"], "test", cfg["data"]["size"], augment=False, cache_dir=cd)
    if val_ds is None:
        val_ds = MCODDataset(cfg["data"]["root"], "val", cfg["data"]["size"],
                             augment=False, cache_dir=cd)

    nw = cfg["train"].get("workers", 4)
    train_dl = DataLoader(train_ds, batch_size=cfg["train"]["batch"], shuffle=True,
                          num_workers=nw, drop_last=True,
                          pin_memory=(dev.type == "cuda"), persistent_workers=nw > 0)
    val_dl = DataLoader(val_ds, batch_size=cfg["train"].get("val_batch", 8), shuffle=False,
                        num_workers=min(nw, 2), pin_memory=(dev.type == "cuda"))
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = build_model(cfg).to(dev)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[{args.name}] model built — {n_par:.2f}M params", flush=True)
    loss_fn = PCELoss(**cfg.get("loss", {})); loss_fn.bind(model)
    epochs = cfg["train"]["epochs"]; base_lr = cfg["train"]["lr"]
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler() if dev.type == "cuda" else None
    ei = cfg["train"].get("eval_interval", 15); best = -1.0; best_ep = 0
    start_ep = 0
    last_ckpt = os.path.join(out, "ckpt_last.pth")
    if args.resume and os.path.exists(last_ckpt):
        ck = torch.load(last_ckpt, map_location=dev)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        if ck.get("scaler") and scaler is not None:
            scaler.load_state_dict(ck["scaler"])
        start_ep = ck["epoch"] + 1
        best = ck.get("best", -1.0); best_ep = ck.get("best_ep", 0)
        print(f"[{args.name}] RESUMED from epoch {ck['epoch']} (best val-Sm so far "
              f"{best:.4f}@{best_ep})", flush=True)

    def _bn_modules(m):
        return [b for b in m.modules() if isinstance(b, torch.nn.modules.batchnorm._BatchNorm)]
    bn_snapshot: dict[int, tuple] = {}

    def snapshot_bn():
        for i, b in enumerate(_bn_modules(model)):
            bn_snapshot[i] = (b.running_mean.detach().clone(), b.running_var.detach().clone())

    def restore_bn_if_poisoned() -> bool:
        bad = any((not torch.isfinite(b.running_mean).all()) or
                  (not torch.isfinite(b.running_var).all()) for b in _bn_modules(model))
        if bad and bn_snapshot:
            for i, b in enumerate(_bn_modules(model)):
                if i in bn_snapshot:
                    b.running_mean.copy_(bn_snapshot[i][0])
                    b.running_var.copy_(bn_snapshot[i][1])
            print(f"[{args.name}] BN stats non-finite -> restored from last healthy epoch",
                  flush=True)
        return bad
    snapshot_bn()

    for ep in range(start_ep, epochs):
        lr = base_lr * 0.5 * (1 + math.cos(math.pi * ep / epochs))
        for g in opt.param_groups:
            g["lr"] = lr
        if hasattr(loss_fn, "set_epoch"):
            loss_fn.set_epoch(ep)                 # loss warmups (pos_warmup/phys_warmup)
        tr = train_one_epoch(model, train_dl, opt, loss_fn, dev, scaler, ep,
                             log_every=20, clip=cfg["train"].get("clip", 0.0))
        restore_bn_if_poisoned()
        snapshot_bn()
        # rolling full checkpoint for crash/stall resume (monitor relaunches with --resume)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "scaler": scaler.state_dict() if scaler is not None else None,
                    "epoch": ep, "best": best, "best_ep": best_ep}, last_ckpt)
        if (ep + 1) % ei == 0 or ep == epochs - 1:
            m = evaluate(model, val_dl, dev, evaluate_image)
            print(f"[{args.name} ep{ep}] loss={tr:.4f} val-MAE={m['mae']:.4f} "
                  f"val-Sm={m['sm']:.4f}", flush=True)
            if m["sm"] > best:
                best = m["sm"]; best_ep = ep
                torch.save({"model": model.state_dict(), "epoch": ep},
                           os.path.join(out, "best.pth"))
                snapshot_bn()

    print(f"[{args.name}] best VAL-Sm={best:.4f}@ep{best_ep}; final test eval "
          f"(single pass)...", flush=True)
    ck = torch.load(os.path.join(out, "best.pth"), map_location=dev)
    model.load_state_dict(ck["model"])
    if cfg["data"].get("dataset_type") == "rgbcod":
        from rgb_dataset import RGBTest
        from evaluate_full_rgb import evaluate_full_rgb
        sets = {"COD10K": (os.path.join(cfg["data"]["root"], "Test", "Image"),
                           os.path.join(cfg["data"]["root"], "Test", "GT_Object"),
                           os.path.join(cfg["data"]["root"], "Test",
                                        "CAM_Instance_Test.json")),
                "CAMO": ("./data/camo/Image", "./data/camo/GT", None),
                "CHAMELEON": ("./data/chameleon/Image",
                              "./data/chameleon/GT", None),
                "NC4K": ("./data/NC4K/Imgs", "./data/NC4K/GT", None)}
        off = {}
        for nm, (im, gt, js) in sets.items():
            if not os.path.isdir(im):
                print(f"  [skip] {nm} not found: {im}", flush=True); continue
            ds_t = RGBTest(im, gt, cfg["data"]["size"], list_json=js)
            off[nm] = evaluate_full_rgb(model, ds_t,
                                        os.path.join(out, f"preds_{nm}"), dev)
    else:
        gt_dir = os.path.join(cfg["data"]["root"], "TestDataset", "GT")
        off = evaluate_full(model, test_ds, gt_dir, os.path.join(out, "preds"), dev)
    json.dump({"name": args.name, "official": off, "best_val_Sm": float(best),
               "best_ep": best_ep, "params_M": float(n_par),
               "protocol": "val-selected (deterministic 103-img tail of train); "
                           "test touched once, 4-way flip TTA"},
              open(os.path.join(out, "metrics.json"), "w"), indent=2, default=float)
    if off and isinstance(next(iter(off.values()), None), dict):
        for set_name, m in off.items():                     # rgb multi-set results
            print(f"[{args.name}] OFFICIAL {set_name}: " +
                  " ".join(f"{k}={v:.4f}" for k, v in m.items()
                           if isinstance(v, float)), flush=True)
    else:
        print(f"[{args.name}] OFFICIAL: " + " ".join(f"{k}={v:.4f}" for k, v in off.items()),
              flush=True)
    cm = cfg["model"]
    tags = [cm.get("type", "pcde")]
    for k in ("anchor", "use_spec", "use_spat", "use_freq", "use_arbitrate",
              "use_decouple", "k_endmembers"):
        if k in cm:
            tags.append(f"{k}={cm[k]}")
    if off and isinstance(next(iter(off.values()), None), dict):
        cells = " | ".join(f"{sn}: Sm={m['Sm']:.4f} Em={m['Em']:.4f} Fb={m['Fb']:.4f} "
                           f"MAE={m['MAE']:.4f}" for sn, m in off.items())
        line = f"| {args.name} | {'/'.join(tags)} | {cells} | {n_par:.1f}M | val {best:.4f}@{best_ep} |\n"
    else:
        line = (f"| {args.name} | {'/'.join(tags)} | {off['MAE']:.4f} | {off['Sm']:.4f} | "
                f"{off['Em']:.4f}/{off['Em_max']:.4f} | {off['wF']:.4f} | "
                f"{off['Fb']:.4f}/{off['Fb_max']:.4f} | {n_par:.1f}M | "
                f"val {best:.4f}@{best_ep} |\n")
    with open(os.path.join(EXP_ROOT, "LOG.md"), "a", encoding="utf-8") as f:
        f.write(line)
    print(f"[{args.name}] appended LOG.md", flush=True)


if __name__ == "__main__":
    main()
