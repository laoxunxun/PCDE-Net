"""Result-analysis figures (all real numbers from LOG.md / metrics.json).

fig5  ablation bar chart      — module contributions on MCOD
fig6  scene analysis         — per-challenge IoU vs 12 methods (grouped)
fig7  band importance        — single-band + group occlusion
fig8  robustness             — band-failure degradation physics vs baseline
fig9  RGB four-benchmark     — physics vs no-physics transfer
"""
from __future__ import annotations

import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "paper_figs")
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"],
                     "font.size": 9, "mathtext.fontset": "stix"})


def fig5():
    # multi-seed groups plotted as mean +- sample std (same convention as Table 3);
    # single-run groups carry no error bar
    names = ["free-gate\ncontrol", "w/o freq\nbranch", "+S$^2$CPM\nonly",
             "baseline\n(no physics)\n2 seeds", "learned\narbitration\n2 seeds",
             "final\n(uniform fusion)\n3 seeds", "dual-B3\nbackbone"]
    sm = [0.8651, 0.8699, 0.8755, 0.8760, 0.8780, 0.8748, 0.8768]
    fb = [0.7318, 0.7370, 0.7504, 0.7532, 0.7574, 0.7481, 0.7527]
    sm_err = [0, 0, 0, 0.0071, 0.0009, 0.0070, 0]
    fb_err = [0, 0, 0, 0.0190, 0.0087, 0.0066, 0]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9.8, 3.4))
    b1 = ax.bar(x - 0.21, sm, 0.4, label=r"$\mathrm{S}_\alpha$", color="#7f9fc4", ec="#446",
                yerr=sm_err, error_kw=dict(ecolor="#333", lw=0.9, capsize=2.5, capthick=0.9))
    b2 = ax.bar(x + 0.21, fb, 0.4, label=r"$F_\beta$", color="#d9a0a0", ec="#844",
                yerr=fb_err, error_kw=dict(ecolor="#333", lw=0.9, capsize=2.5, capthick=0.9))
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.4f", fontsize=6.8, padding=1.5)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7.6)
    ax.set_ylim(0.70, 0.90)
    ax.set_ylabel("score")
    ax.legend(loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    # NOTE: the paper's authoritative fig5_ablation.{png,pdf} (paper Fig.7) is
    # produced by the MATLAB chain matlab_figs/make_all_figs.m; this python
    # variant writes a side file only and must not overwrite it.
    plt.savefig(os.path.join(OUT, "fig5_ablation_python.png"), dpi=300); plt.close()
    print("fig5 done (python side variant)")


def fig6():
    # real per-scene numbers (scene_metrics.json for ours + per_image_iou means)
    scenes = ["small\n(0.1–0.5%)", "large\n(>2%)", "low-contrast", "multi-target",
              "elongated", "high-veg bg", "tiny\n(<0.1%)"]
    # exp01_640 (headline 640^2 model) vs public clean-protocol reproductions
    ours = [0.668, 0.813, 0.700, 0.691, 0.587, 0.705, 0.394]
    zoomnext = [0.582, 0.615, 0.623, 0.608, 0.503, 0.627, 0.429]
    mscaf = [0.562, 0.699, 0.627, 0.579, 0.473, 0.608, 0.429]
    prnet = [0.584, 0.651, 0.627, 0.588, 0.535, 0.637, 0.389]
    x = np.arange(len(scenes))
    fig, ax = plt.subplots(figsize=(9.5, 3.5))
    w = 0.2
    ax.bar(x - 1.5 * w, ours, w, label="PCDE-Net (ours)", color="#8a2f2f")
    ax.bar(x - 0.5 * w, zoomnext, w, label="ZoomNeXt", color="#7f9fc4")
    ax.bar(x + 0.5 * w, mscaf, w, label="MSCAF-Net", color="#a8c6a0")
    ax.bar(x + 1.5 * w, prnet, w, label="PRNet", color="#d9c08a")
    ax.set_xticks(x); ax.set_xticklabels(scenes, fontsize=8)
    ax.set_ylabel("per-image IoU")
    ax.set_ylim(0.35, 0.85)
    ax.legend(ncol=4, frameon=False, fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.annotate("+11.4 pp", xy=(1, 0.813), xytext=(1.35, 0.84), fontsize=8, color="#8a2f2f",
                arrowprops=dict(arrowstyle="->", color="#8a2f2f"))
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig6_scene.png"), dpi=300); plt.savefig(os.path.join(OUT, "fig6_scene.pdf")); plt.close()
    print("fig6 done")


def fig7():
    d = json.load(open(os.path.join(ROOT, "trible2one/experiments/exp01_v11_pvt512/band_occlusion.json")))
    bands = ["Violet\n422", "Blue\n487", "Green\n550", "Orange\n602", "Red\n660",
             "RedEdge\n725", "NIR\n785", "NIR\n887"]
    drops = [d["occluded"][b]["drop"] * 100 for b in
             ["Violet422", "Blue487", "Green550", "Orange602", "Red660",
              "RedEdge725", "NIR785", "NIR887"]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.2),
                                   gridspec_kw={"width_ratios": [2.1, 1]})
    cols = ["#7f9fc4"] * 5 + ["#d9a0a0"] * 3
    ax1.bar(bands, drops, color=cols, ec="#555")
    ax1.set_ylabel(r"$\mathrm{S}_\alpha$ drop when occluded (pp)")
    ax1.set_title("single-band importance", fontsize=9.5)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(axis="y", alpha=0.3)
    ax1.tick_params(labelsize=7.6)
    g = ["visible\n(occluded)", "non-visible\n(occluded)", "RGB\n(occluded)"]
    gd = [d["occluded"]["ALL-VISIBLE(0-4)"]["drop"] * 100,
          d["occluded"]["ALL-NONVISIBLE(5-7)"]["drop"] * 100,
          d["occluded"]["RGB(1,2,4)"]["drop"] * 100]
    ax2.bar(g, gd, color=["#7f9fc4", "#d9a0a0", "#a8c6a0"], ec="#555")
    for i, v in enumerate(gd):
        ax2.text(i, v + 0.3, f"−{v:.1f}", ha="center", fontsize=8)
    ax2.set_title("band-group complementarity", fontsize=9.5)
    ax2.set_ylabel(r"$\mathrm{S}_\alpha$ drop (pp)")
    ax2.set_ylim(0, 21)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig7_band.png"), dpi=300); plt.savefig(os.path.join(OUT, "fig7_band.pdf")); plt.close()
    print("fig7 done")


def fig8():
    d = json.load(open(os.path.join(ROOT, "trible2one/experiments/robustness_spectral.json")))
    ours = d["exp01_v11_pvt512"]; base = d["exp00_baseline_512pvt"]
    axes_names = ["illumination\nshift", "band failure", "band noise"]
    od = [ours["hard_gain_drop"] * 100, ours["band_drop_drop"] * 100, ours["band_noise_drop"] * 100]
    bd = [base["hard_gain_drop"] * 100, base["band_drop_drop"] * 100, base["band_noise_drop"] * 100]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    ax.bar(x - 0.17, bd, 0.34, label="no-physics baseline", color="#7f9fc4", ec="#446")
    ax.bar(x + 0.17, od, 0.34, label="PCDE-Net (physics)", color="#8a2f2f", ec="#522")
    for i in range(3):
        ax.text(i + 0.17, od[i] + 0.05, f"{od[i]:.2f}", ha="center", fontsize=7.6)
        ax.text(i - 0.17, bd[i] + 0.05, f"{bd[i]:.2f}", ha="center", fontsize=7.6)
    ax.set_xticks(x); ax.set_xticklabels(axes_names, fontsize=8.5)
    ax.set_ylabel(r"$\mathrm{S}_\alpha$ drop (pp)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.annotate("26% less degradation", xy=(1.17, 2.75), xytext=(1.5, 3.1), fontsize=8.5,
                color="#8a2f2f", arrowprops=dict(arrowstyle="->", color="#8a2f2f"))
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig8_robust.png"), dpi=300); plt.savefig(os.path.join(OUT, "fig8_robust.pdf")); plt.close()
    print("fig8 done")


def fig9():
    sets = ["COD10K", "CAMO", "CHAMELEON", "NC4K"]
    ours = [0.8350, 0.8009, 0.8761, 0.8699]
    base = [0.8336, 0.8001, 0.8625, 0.8654]
    sota = [0.887, 0.874, 0.88, 0.892]      # B2-tier reference (ZoomNeXt-B2)
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    ax.bar(x - 0.27, sota, 0.26, label="dedicated RGB SOTA (B2-tier)", color="#cccccc", ec="#888")
    ax.bar(x, base, 0.26, label="RGB control (no physics)", color="#7f9fc4", ec="#446")
    ax.bar(x + 0.27, ours, 0.26, label="PCDE-RGB (physics)", color="#8a2f2f", ec="#522")
    for i in range(4):
        ax.text(i + 0.27, ours[i] + 0.004, f"+{(ours[i]-base[i])*100:.1f}", ha="center",
                fontsize=7.4, color="#2a7a2a")
    ax.set_xticks(x); ax.set_xticklabels(sets)
    ax.set_ylabel(r"$\mathrm{S}_\alpha$")
    ax.set_ylim(0.78, 0.91)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig9_rgb.png"), dpi=300); plt.savefig(os.path.join(OUT, "fig9_rgb.pdf")); plt.close()
    print("fig9 done")


if __name__ == "__main__":
    fig5(); fig6(); fig7(); fig8(); fig9()
    print("all result figures ->", OUT)
