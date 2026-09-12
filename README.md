# Uncertainty-Aware Federated Learning for Multi-Band Radar Localization

A single-file benchmark that trains **18 radar-localization backbones** across
**4 uncertainty-quantification heads**, **3 self-supervised pre-training
regimes**, and **two training paradigms** (centralized vs. federated), reporting
per-band results at **77 GHz, 24 GHz, 10 GHz, and the fused multi-band view**.

Everything runs from one script — `RADAR_Localization_withFed.py`. There are no
extra steps, no shell scripts, and no build system: edit the `CONFIG` block at
the top of the file and run it.

---

## Table of contents

- [What this does](#what-this-does)
- [The model zoo](#the-model-zoo)
- [The ablation axes](#the-ablation-axes)
- [Federated learning protocol](#federated-learning-protocol)
- [Quick start](#quick-start)
- [Dataset](#dataset)
- [Configuration reference](#configuration-reference)
- [Outputs](#outputs)
- [Runtime and scaling](#runtime-and-scaling)
- [Reproducibility](#reproducibility)
- [Repository layout](#repository-layout)
- [Citation](#citation)
- [License](#license)

---

## What this does

The script casts indoor radar localization as a supervised regression problem
over micro-Doppler spectrograms and evaluates, under identical conditions, how
model family, uncertainty head, representation pre-training, and federation
interact. For every model it produces a position estimate `(x, y)`, a bearing
(angle-of-arrival), a time-of-flight estimate, an NLOS (non-line-of-sight)
detection flag, and a calibrated confidence.

Each configuration is trained **twice** — once centrally on pooled data and once
with **federated averaging** over non-IID clients — so the effect of federation
can be read off directly, holding everything else fixed.

## The model zoo

18 backbones in three families, all sharing one uncertainty-aware localization
head, one physics-informed loss, and one data pipeline:

| Family | Count | Models |
|---|---|---|
| **DRL** (deep reinforcement learning) | 5 | MAPPO, SAC, D3QN, DDPG, QMIX |
| **Hybrid** (convolutional / attention) | 7 | TransRAD, RadarGNN, PINN-CNN, PINN-LSTM, DecisionTR, ISAC-BiLSTM, RaDiT |
| **GenAI** (foundation-model components) | 6 | RadarCLIP, RadarVLM, RadarLLaMA, RadarPhi, LatentDiff, TimeGAN |

The **PINN-CNN** and **PINN-LSTM** backbones retain a physics-residual loss
(steering-vector consistency), so the physics-informed pathway is preserved and
ablated on the same footing as every other model.

## The ablation axes

Each model is evaluated over **12 combinations** = 4 UQ heads × 3 representation
regimes, each reported on 4 frequency views.

**Uncertainty quantification (UQ) head:**

1. **Softmax** — baseline predictive distribution.
2. **Softmax + MC-Dropout** — Monte-Carlo dropout (20 stochastic forward passes).
3. **Softmax + Temperature Scaling** — post-hoc calibration.
4. **Evidential Deep Learning** — Dirichlet-evidence uncertainty.

**Representation pre-training:**

- **None** — train from scratch.
- **Contrastive** — SimCLR-style NT-Xent pre-training of the encoder.
- **DAE** — signal denoising auto-encoder pre-training.

**Frequency views reported:** `combined` (fused multi-band), `77ghz`, `24ghz`,
`10ghz`.

## Federated learning protocol

Horizontal **FedAvg**, weighted by client data size — the standard scheme in the
federated-LLM literature. Each round the server samples a fraction of clients;
each client trains locally on its **own private shard**; the server then averages
the weight updates into a new global model. **Only model weights are shared,
never raw data.**

- **Clients:** the training set is partitioned among `N_CLIENTS` data owners.
- **Non-IID split:** Dirichlet partitioning over spatial zones
  (`DIRICHLET_ALPHA = 0.5`; smaller means more heterogeneous).
- **Round-level early stopping:** training halts if validation error has not
  improved for `FL_EARLY_STOP_ROUNDS` rounds.
- **Hyper-parameter search:** an overfitting-penalized random search selects the
  communication rounds × local epochs `(R, E)` on a cheap proxy model, scored on
  validation with a penalty on the (validation − train) gap. It never touches the
  test set.
- **Federated pre-training:** the contrastive and DAE representation arms can
  themselves be pre-trained under FedAvg before the supervised sweep.

## Quick start

```bash
# 1. Clone
git clone https://github.com/<your-username>/radar-localization-fed.git
cd radar-localization-fed

# 2. Install dependencies (a virtual environment is recommended)
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Smoke-test the entire matrix in a few minutes (synthetic data, tiny budget)
#    Open RADAR_Localization_withFed.py and set  FAST_DEV = True
python RADAR_Localization_withFed.py

# 4. Full run: set FAST_DEV = False, point DATASET_ROOT at your data, and run
python RADAR_Localization_withFed.py
```

In **VS Code** you can simply press **Run Python File** — the script is designed
to run with no arguments.

> **First time?** Leave `ALLOW_SYNTHETIC_FALLBACK = True` and set
> `FAST_DEV = True`. The script will generate a synthetic stand-in dataset and
> exercise the full pipeline (all models, both training modes) in minutes, so you
> can confirm your environment works before committing to a real run.

## Dataset

The script targets the **CI4R multi-frequency human-activity** micro-Doppler
dataset (77 / 24 / 10 GHz). Point `DATASET_ROOT` at your local copy.

- **Formats:** MATLAB `.mat` (v5 via SciPy, v7.3 / HDF5 via h5py) and raw `.dat`.
  The confirmed primary variable in the CI4R `.mat` files is `sx1`
  (a frequency × time micro-Doppler spectrogram).
- **Preprocessing:** spectrograms are log-normalized, the Doppler axis is
  centre-cropped to drop outer noise bands, and each map is resized to
  `SPEC_H × SPEC_W` (224 × 224 by default; 384 × 384 is the higher-quality
  option).
- **Splits:** record-level 70 / 15 / 15 train / validation / test, with training
  statistics shared to validation and test so there is **no normalization
  leakage**. A leakage audit runs automatically.

If the dataset path is missing and `ALLOW_SYNTHETIC_FALLBACK = True`, a synthetic
stand-in is generated so the pipeline runs end-to-end — clearly flagged in the
log, and intended for pipeline testing only. Set it to `False` to force
real-data-only (the program exits if the path is absent).

> **Please do not commit raw radar data.** `Data/`, `*.mat`, and `*.dat` are
> already listed in `.gitignore`.

## Configuration reference

Everything below lives in the `CONFIG` block at the top of the script. The most
useful knobs:

| Setting | Default | Purpose |
|---|---|---|
| `DATASET_ROOT` | `Data` | Path to the dataset. |
| `OUTPUT_DIR` | `Radar_Localization_withFed` | Where results are written. |
| `ALLOW_SYNTHETIC_FALLBACK` | `True` | Use synthetic data if the real set is missing. |
| `SUITES_TO_RUN` | `["hybrid", "genai"]` | Any subset of `drl`, `hybrid`, `genai`. |
| `MODELS_TO_RUN` | `None` | `None` = all models in the chosen suites, or an explicit list of keys. |
| `UQ_METHODS` | all 4 | Subset of the UQ heads. |
| `REP_METHODS` | all 3 | Subset of the pre-training regimes. |
| `BANDS` | all 4 | Frequency views to report. |
| `TRAINING_MODES` | `["centralized", "federated"]` | Keep both for the with/without-FL comparison. |
| `EPOCHS` | `100` | Supervised epochs (early stopping usually cuts it short). |
| `BATCH_SIZE` | `128` | |
| `SEED` | `42` | Global seed for deterministic runs. |
| `N_CLIENTS` | `6` | Federated data owners. |
| `CLIENT_FRACTION` | `0.5` | Fraction of clients sampled per round. |
| `DIRICHLET_ALPHA` | `0.5` | Non-IID heterogeneity (smaller = more skewed). |
| `FL_TUNE` | `True` | Search `(rounds × local-epochs)` and pick the best. |
| `FL_EARLY_STOP_ROUNDS` | `10` | Round-level early stopping patience. |
| `FAST_DEV` | `False` | Set `True` for a fast smoke test of the whole matrix. |

**Scaling a run down.** To go from the full 216-training sweep to something
quick: narrow `SUITES_TO_RUN` / `MODELS_TO_RUN`, shorten `UQ_METHODS` and
`REP_METHODS`, lower `EPOCHS`, or set `FAST_DEV = True`. Nothing else needs to
change.

## Outputs

All written to `OUTPUT_DIR`, timestamped (`{ts}`) so runs never overwrite each
other:

| File | Contents |
|---|---|
| `radar_ablation_master_{ts}.csv` | One row per `(suite, model, uq, rep, band)` — every metric. |
| `radar_ablation_master_{ts}.json` | The same data, nested. |
| `summary_MAE_{suite}_{ts}.png` | MAE heatmap: models × 12 combos, per band. |
| `summary_ECE_{suite}_{ts}.png` | Calibration (ECE) heatmap, combined band. |
| `leaderboard_{ts}.txt` | Best combination per model, plus the global ranking. |
| `ablation_{ts}.out` | The full run log. |

**Metrics per row:** localization MAE, RMSE, MSE, error percentiles (P50/P80/P90/P95),
per-axis MAE, angle-of-arrival MAE/RMSE, time-of-flight MAE/RMSE, NLOS accuracy,
F1, false-alarm and missed-detection rates, zone and floor accuracy, expected
calibration error (ECE), NLL, Brier score, mean predictive uncertainty,
inference latency per sample, parameter count, and training time.

## Runtime and scaling

The **full** sweep is 18 models × 12 combinations = **216 trainings**, plus
contrastive/DAE pre-trainings for 8 of every 12 combinations — and each is run
under both centralized and federated paradigms. That is a substantial compute
budget; a single high-end GPU will take a long time on the complete matrix.

Use the levers in [Configuration reference](#configuration-reference) to size the
run to your hardware. A GPU is strongly recommended; the code runs on CPU but is
much slower. The reference experiments used a single NVIDIA RTX 5090.

## Reproducibility

A global `SEED` (default 42) seeds Python, NumPy, and PyTorch, including CUDA.
Data splits, client partitions, and augmentation are all seeded, and training
statistics are shared from the train split to validation and test to prevent
leakage (audited automatically at run time). The federated hyper-parameter
search scores trials on validation only and never touches the test set.

## Repository layout

```
.
├── RADAR_Localization_withFed.py   # the entire benchmark — one file
├── README.md                       # this file
├── requirements.txt                # Python dependencies
├── CITATION.cff                    # citation metadata
├── LICENSE                         # MIT
└── .gitignore                      # ignores datasets and run outputs
```

## Citation

If you use this code, please cite the associated paper (see `CITATION.cff` for
machine-readable metadata):

```bibtex
@article{mahabub2026radarfed,
  title   = {Uncertainty-Aware Federated Learning for Multi-Band Radar
             Localization: Benchmarking Deep Reinforcement Learning and
             Generative-AI Backbones on 10/24/77 GHz Micro-Doppler Data},
  author  = {Mahabub, Atik and Vakili, Shervin},
  journal = {IEEE Open Journal of Signal Processing},
  year    = {2026},
  note    = {Institut national de la recherche scientifique (INRS-EMT)}
}
```

## License

Released under the [MIT License](LICENSE).

The dataset is **not** covered by this license — obtain the CI4R micro-Doppler
data from its original source under its own terms.
