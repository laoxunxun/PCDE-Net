"""Offline: data-driven endmember initialization for S²CPM.

From the TRAIN-PROPER split only (the deterministic 103-image val tail is
excluded, same rule as shared.dataset, so model selection stays clean):
  E_0  target endmember  = mask-weighted mean spectrum of camouflaged pixels
  E_1.. background       = k-means (K−1) clusters of non-target spectra
                           (vegetation / soil / shadow variants emerge from data)

Run once (≈2 min on the cache):   python trible2one/build_endmember_init.py
Output: trible2one/assets/endmember_init.npz  (key 'E', shape (K,8))
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from shared.dataset import VAL_N  # noqa: E402

BANDS_NM = [422, 487, 550, 602, 660, 725, 785, 887]


def kmeans(x: np.ndarray, k: int, iters: int = 30, seed: int = 0) -> np.ndarray:
    """Plain k-means with farthest-point init (deterministic)."""
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    centers = [x[rng.integers(n)]]
    for _ in range(k - 1):
        d = np.min(np.linalg.norm(x[:, None] - np.asarray(centers)[None], axis=2), axis=1)
        centers.append(x[int(np.argmax(d))])
    c = np.asarray(centers)
    for _ in range(iters):
        assign = np.argmin(np.linalg.norm(x[:, None] - c[None], axis=2), axis=1)
        for j in range(k):
            if (assign == j).any():
                c[j] = x[assign == j].mean(axis=0)
    return c


def main(root: str = "./data/mcod", n_images: int = 300, k_bg: int = 3,
         per_img_bg: int = 1500):
    mat_dir = os.path.join(root, "TrainDataset", "Mat")
    gt_dir = os.path.join(root, "TrainDataset", "GT")
    cache_dir = os.path.join(root, "cache352", "train")
    names = sorted(os.path.splitext(p)[0] for p in os.listdir(mat_dir)
                   if p.endswith(".mat"))[:-VAL_N]        # train proper only
    step = max(1, len(names) // n_images)
    names = names[::step][:n_images]

    tgt_acc = np.zeros(8, dtype=np.float64); tgt_n = 0
    bg_pool = []
    for i, n in enumerate(names):
        cube_p = os.path.join(cache_dir, n + ".npy")
        if os.path.exists(cube_p):
            cube = np.load(cube_p).astype(np.float32)     # (8,352,352) [0,1]
        else:                                             # fallback: raw .mat
            import h5py
            with h5py.File(os.path.join(mat_dir, n + ".mat"), "r") as f:
                cube = np.transpose(np.asarray(f["gray"], dtype=np.float32), (0, 2, 1)) / 255.0
        from PIL import Image
        m = np.asarray(Image.open(os.path.join(gt_dir, n + ".jpg")).convert("L")) > 127
        if m.shape != cube.shape[1:]:                      # GT is native-res, cube cached at 352
            m = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize(
                (cube.shape[2], cube.shape[1]), Image.NEAREST)) > 127
        cube = cube.reshape(8, -1); m = m.reshape(-1)
        if m.sum() > 50:
            tgt_acc += cube[:, m].sum(axis=1); tgt_n += int(m.sum())
        # stratified background subsample (spread over the image, not the first rows)
        bg_idx = np.flatnonzero(~m)
        if len(bg_idx) > per_img_bg:
            pick = rng_choice(len(bg_idx), per_img_bg)
            bg_pool.append(cube[:, bg_idx[pick]].T)
        else:
            bg_pool.append(cube[:, bg_idx].T)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(names)} images", flush=True)

    e_t = (tgt_acc / max(tgt_n, 1)).astype(np.float32)
    bg = np.concatenate(bg_pool, axis=0)
    print(f"[stats] target pixels={tgt_n}, background pool={bg.shape[0]}")
    e_bg = kmeans(bg, k_bg).astype(np.float32)

    E = np.clip(np.concatenate([e_t[None], e_bg], axis=0), 0.02, 0.98).astype(np.float32)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "endmember_init.npz")
    np.savez(out, E=E, bands_nm=np.array(BANDS_NM))
    print(f"saved {out}  E shape={E.shape}")

    print("\nEndmember spectra (reflectance × 100):")
    hdr = "band(nm) " + "".join(f"{b:>7}" for b in BANDS_NM)
    print(hdr)
    for k, row in enumerate(E):
        tag = "TARGET" if k == 0 else f"BG-{k}"
        print(f"{tag:>8} " + "".join(f"{v * 100:>7.1f}" for v in row))
    ndvi = (E[:, 7] - E[:, 4]) / (E[:, 7] + E[:, 4] + 1e-6)
    print("NDVI-like (NIR2−Red)/(NIR2+Red): " +
          ", ".join(f"{v:+.3f}" for v in ndvi) +
          "   ← vegetation BG should be high, TARGET low")


def rng_choice(n: int, k: int, seed: int = 0):
    rng = np.random.default_rng(seed + n)                 # stable per image
    return rng.choice(n, size=k, replace=False)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./data/mcod")
    ap.add_argument("--n-images", type=int, default=300)
    ap.add_argument("--k-bg", type=int, default=3)
    a = ap.parse_args()
    main(a.root, a.n_images, a.k_bg)
