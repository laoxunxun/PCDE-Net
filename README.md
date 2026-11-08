# PCDE-Net: Spectral Unmixing-Guided Feature Decoupling and Enhancement for Multispectral Camouflaged Object Detection

PCDE-Net turns spectral unmixing into a **differentiable front end** trained jointly with the detector: the learned endmembers and abundances drive a bounded soft-routing feature split (AGD) and a tri-domain enhancement (TDE) that counters material, texture, and statistical camouflage in the spectral, spatial, and frequency domains.

> Paper: *Spectral Unmixing-Guided Feature Decoupling and Enhancement for Multispectral Camouflaged Object Detection* (link to be added upon publication).

## Results (MCOD test set, 500 cubes)

| Model | Input | S-measure | E-measure | F-measure | MAE |
|---|---|---|---|---|---|
| PCDE-Net | 640² | **0.8833** | 0.9346 | 0.7694 | 0.0019 |
| PCDE-Net | 512² | 0.8824 | 0.9345 | 0.7547 | 0.0022 |
| No-physics baseline (same recipe) | 512² | 0.8822 / 0.8810* | 0.9267 | 0.7622 | 0.0020 |

\* single run at 640² / 512². Multi-seed repetition (3 random seeds each): PCDE-Net final configuration **0.8748 ± 0.0070**, no-physics baseline **0.8756 ± 0.0050** — the mean difference (0.09 pp) lies within seed noise and is **not** significant; the physical design pays off in band-failure robustness (26% less degradation), interpretability (abundance map localizes targets at 0.829 AUC), and zero-tuning RGB transfer (22/24 metric wins), at 1.2% extra parameters. All per-run numbers are in `results/`.

## Install

```bash
pip install -r requirements.txt   # Python >= 3.10, PyTorch >= 2.1 (CUDA build recommended)
```

## Data

1. Download the **MCOD** dataset (1,527 eight-band cubes, 422–887 nm) from the benchmark authors: <https://github.com/yl2900260-bit/MCOD>
2. Place it under `./data/mcod` (or point the `data.root` field of any config elsewhere, or `export MCOD_ROOT=/path/to/MCOD`).
3. RGB transfer experiments use the standard COD10K/CAMO/CHAMELEON/NC4K protocol; place those datasets under `./data/`.

## Train & evaluate

Every run trains (AdamW, cosine schedule, 300 epochs), selects the best checkpoint on a fixed validation tail, and evaluates **once** on the official test split with py-sod-metrics + 4-way flip TTA:

```bash
# main model (Table II, 640² row)
python run_experiment.py --name exp01_640 --config configs/exp01_640.yaml

# final configuration at 512² and its seed repetitions (Table III)
python run_experiment.py --name abl_noarbitrate_512 --config configs/abl_noarbitrate_512.yaml

# no-physics baseline (Table III)
python run_experiment.py --name exp00_baseline_512pvt --config configs/exp00_baseline_512pvt.yaml
```

Outputs land in `experiments/<name>/` (checkpoints, predictions, `metrics.json`).

## Pretrained weights

All 23 checkpoints referenced by the paper (5.6 GB total, `best.pth` only; `ckpt_last.pth` is not distributed). The download is organized as **one folder per run**, each containing that run's `best.pth` — the folder name equals the configuration name in `configs/` (a run name without a seed suffix uses the default seed 42):

| Folder | Take | Role in paper |
|---|---|---|
| `exp01_640` | `best.pth` | **Main model** (Table II, 640², S 0.8833) |
| `abl_noarbitrate_512` + `_s123` + `_s202` | `best.pth` ×3 | Final configuration, 3 seeds (Table III, 0.8748 ± 0.0070) |
| `exp00_baseline_512pvt` + `_s123` + `_s202` | `best.pth` ×3 | No-physics baseline, 3 seeds (Table III, 0.8756 ± 0.0050) |
| `exp00_baseline_640` | `best.pth` | 640² baseline |
| `exp01_v11_pvt512`, `exp01_v11_seed2` | `best.pth` ×2 | Learned arbitration, 2 seeds |
| `exp01_kappa_arb` | `best.pth` | κ-fusion (Table IV) |
| `abl_anchor_learned_512` | `best.pth` | Free-gate control |
| `abl_only_spec` / `abl_only_spat` / `abl_only_freq` | `best.pth` ×3 | Single-domain controls (Table IV) |
| `abl_sep2dom_512`, `abl_tde_nofreq_512` | `best.pth` ×2 | Separate fusion / no-frequency controls |
| `abl_k2`, `abl_k6`, `abl_dual_b3_512`, `exp01_nodc_512pvt` | `best.pth` ×4 | Design choices (Table III) |
| `rgb_pcde_cod10k`, `rgb_baseline` | `best.pth` ×2 | RGB transfer pair (Table VII) |

**Download (Baidu Netdisk)**: <https://pan.baidu.com/s/1ZaiT9TNuUUIqp_pvVIE6iw> (access code: `34uz`)
International mirror (Zenodo DOI) will be added upon publication.

Verify after download: `sha256sum -c weights_sha256.txt` from the parent folder of the run folders (each checksum refers to `<run>/best.pth`).

Per-image predictions of all 23 runs are published as a Release attachment (`preds.zip`).

## Reproduce key analyses

```bash
python band_occlusion.py      # Fig. 9  band importance
python eval_robustness.py     # Fig. 10 out-of-distribution stresses (band failure etc.)
python bench_efficiency.py --config exp01_v11_pvt512   # Table VIII params/GFLOPs/FPS
python make_result_figs.py    # data figures from saved metrics
```

## Configuration name ↔ paper table

| Config | Paper |
|---|---|
| `exp01_640` | Table II, PCDE-Net 640² (0.8833) |
| `abl_noarbitrate_512` (+`_s123`, `_s202`) | Table II 512² row / Table III uniform fusion (final), 3 seeds |
| `exp00_baseline_512pvt` (+`_s123`, `_s202`), `exp00_baseline_640` | Table III no-physics baseline |
| `exp01_v11_pvt512`, `exp01_v11_seed2` | Table III learned arbitration (2 seeds) |
| `exp01_kappa_arb` | Table IV κ-fusion |
| `abl_anchor_learned_512` | Table III free-gate control |
| `abl_only_spec/spat/freq`, `abl_sep2dom_512`, `abl_tde_nofreq_512` | Table IV tri-domain matrix |
| `abl_k2`, `abl_k6`, `abl_dual_b3_512`, `exp01_nodc_512pvt` | Table III design choices |
| `rgb_pcde_cod10k`, `rgb_baseline` | Table VII RGB transfer |

## License

MIT (this repository's own code). Comparison methods referenced or reproduced follow their original licenses and are not redistributed here.

## Citation

```
@article{pcdenet,
  title   = {Spectral Unmixing-Guided Feature Decoupling and Enhancement for Multispectral Camouflaged Object Detection},
  author  = {Sun, Haoran and Qu, Jue and Wang, Wei and Liu, Haiping and Zhou, Hengfei and Yang, Bo},
  journal = {IEEE Transactions on Circuits and Systems for Video Technology},
  year    = {2026},
  note    = {to appear}
}
```
