"""Official RGB-COD evaluation (pysodmetrics, 4-flip TTA) for one test set."""
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import py_sod_metrics as PSM


@torch.no_grad()
def evaluate_full_rgb(model, ds, preds_dir, dev):
    os.makedirs(preds_dir, exist_ok=True)
    m_m = PSM.MAE(); sm = PSM.Smeasure(); em = PSM.Emeasure()
    wf = PSM.WeightedFmeasure(); fm = PSM.Fmeasure()
    model.eval()
    for i in range(len(ds)):
        x, name = ds[i]
        inp = x.unsqueeze(0).to(dev)
        preds = []
        for flip in [None, [2], [3], [2, 3]]:
            xi = torch.flip(inp, dims=flip) if flip else inp
            p = torch.sigmoid(model(xi)[0, 0]).cpu()
            if flip:
                p = torch.flip(p, dims=[d - 2 for d in flip])
            preds.append(p)
        p = torch.stack(preds).mean(0)
        gt = ds.gt(name); Hn, Wn = gt.shape
        pr = F.interpolate(p[None, None], size=(Hn, Wn), mode="bilinear",
                           align_corners=False)[0, 0].numpy()
        Image.fromarray((pr * 255).clip(0, 255).astype(np.uint8)).save(
            os.path.join(preds_dir, name + ".png"))
        pu = (pr * 255).clip(0, 255).astype(np.uint8)
        gu = ((gt > 0.5).astype(np.uint8) * 255)
        for M in (m_m, sm, em, wf, fm):
            M.step(pred=pu, gt=gu)
    def cs(d):
        d = d if isinstance(d, dict) else {"curve": d}
        c = np.asarray(d["curve"]) if "curve" in d else np.asarray(d)
        return (float(c.mean()) if c.ndim else float(c),
                float(c.max()) if c.ndim else float(c))
    em_mean, em_max = cs(em.get_results()["em"])
    fb_mean, fb_max = cs(fm.get_results()["fm"])
    return {"MAE": m_m.get_results()["mae"], "Sm": sm.get_results()["sm"],
            "Em": em_mean, "Em_max": em_max, "wF": wf.get_results()["wfm"],
            "Fb": fb_mean, "Fb_max": fb_max}
