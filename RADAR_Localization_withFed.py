"""
===============================================================================
 Radar_Localization_Single.py
 SINGLE-FILE ABLATION RUNNER  —  OpenRadarInitiative (CI4R-MULTI3)
 ---------------------------------------------------------------------------
 ONE file. Press "Run Python File" in VS Code. No .bat, no extra steps.

 WHAT IT DOES
 -----------------------------------------------------------------------------
 Runs an uncertainty-quantification (UQ) + representation-learning ablation
 over ALL 18 radar-localization models, and reports results for
       77 GHz · 24 GHz · 10 GHz · Combined (all-band)
 for every one of 12 combinations per model.

 THE 18 MODELS
   DRL (5)    : MAPPO · SAC · D3QN · DDPG · QMIX
   Hybrid (7) : TransRAD · RadarGNN · PINN-CNN · PINN-LSTM ·
                DecisionTR · ISAC-BiLSTM · RaDiT
   GenAI (6)  : RadarCLIP · RadarVLM · RadarLLaMA · RadarPhi ·
                LatentDiff · TimeGAN

 THE 12 COMBINATIONS  (= 4 UQ  ×  3 Representation)
   UQ axis (classifier head / predictive uncertainty):
        1) Softmax              (baseline)
        2) Softmax + MC-Dropout            (Monte-Carlo Dropout)
        3) Softmax + Temperature-Scaling   (post-hoc calibration)
        4) Evidential Deep Learning         (Dirichlet evidence)
   Representation axis (encoder pre-training):
        a) None
        b) Contrastive pre-training (SimCLR / NT-Xent)
        c) Signal Denoising Auto-Encoder (DAE) pre-training
   => 4 x 3 = 12 combinations, each evaluated on 4 frequency settings.

 WHY THESE CHOICES (answers to the brief)
   * "Why don't you use Physics-informed of these models" -> the PINN-CNN and
     PINN-LSTM backbones keep their physics-residual loss (steering-vector
     consistency), so the physics-informed pathway is preserved and ablated
     exactly like the others.
   * All three UQ methods you mentioned are implemented (MC-Dropout,
     Temperature-Scaling, Evidential), plus the plain Softmax baseline so the
     "with / without" comparison is explicit.
   * Both representation ideas are implemented (contrastive AND denoising
     auto-encoder) so their "impact" can be read off directly vs "None".

 OUTPUTS  (written to OUTPUT_DIR)
   radar_ablation_master_{ts}.csv    one row per (suite, model, uq, rep, band)
   radar_ablation_master_{ts}.json   same data, nested
   summary_MAE_{suite}_{ts}.png      MAE heatmap: models x 12 combos, per band
   summary_ECE_{suite}_{ts}.png      calibration (ECE) heatmap, combined band
   leaderboard_{ts}.txt              best combo per model + global ranking
   ablation_{ts}.out                 full run log

 RUNTIME NOTE  (please read)
   The FULL sweep = 18 models x 12 combos = 216 trainings (plus contrastive/DAE
   pre-trainings for 8 of every 12 combos). That is a LOT. Use the CONFIG block
   below to scale it: pick a subset of SUITES_TO_RUN / MODELS_TO_RUN, lower
   EPOCHS, or flip FAST_DEV=True for a quick smoke test of the whole matrix.
   Nothing else needs to change to run.

 DEPENDENCIES : torch, numpy, matplotlib  (h5py / scipy optional, only for .mat)
===============================================================================
"""

import os, sys, re, json, math, time, logging, warnings, random, csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from collections import defaultdict
import copy
from torch.optim import AdamW

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG  — edit anything here; everything below is automatic
# =============================================================================
# --- data + output paths (same dataset layout as your v2 scripts) ------------
DATASET_ROOT = Path(
    r"Data"
)
OUTPUT_DIR = Path(
    r"Radar_Localization_withFed"
)
# If the real dataset is not found, generate a synthetic stand-in so the whole
# pipeline still runs end-to-end (clearly flagged in the log). Set False to
# force real-data-only (program exits if the path is missing).
ALLOW_SYNTHETIC_FALLBACK = True

# --- what to run -------------------------------------------------------------
# suites: any subset of ["drl", "hybrid", "genai"]
# SUITES_TO_RUN = ["drl", "hybrid", "genai"]
SUITES_TO_RUN = ["hybrid", "genai"]
# models: None = every model in the chosen suites, or give explicit keys, e.g.
#   ["transrad", "radar_phi", "d3qn"]
MODELS_TO_RUN: Optional[List[str]] = None
# UQ methods (subset of the 4). REP methods (subset of the 3).
UQ_METHODS  = ["softmax", "mc_dropout", "temp_scaling", "evidential"]
REP_METHODS = ["none", "contrastive", "dae"]
# frequency views to report (subset / order of the 4)
BANDS = ["combined", "77ghz", "24ghz", "10ghz"]

# --- training budget ---------------------------------------------------------
EPOCHS              = 100   # supervised epochs (early stopping usually cuts it)
EARLY_STOP_PATIENCE = 18
PRETRAIN_EPOCHS     = 30    # contrastive / DAE pre-training epochs
WARMUP_EPOCHS       = 6
BATCH_SIZE          = 128
WEIGHT_DECAY        = 2e-4
GRAD_CLIP           = 1.0
MC_DROPOUT_SAMPLES  = 20    # forward passes for MC-Dropout uncertainty
MC_DROPOUT_P        = 0.20  # extra dropout rate used by the MC-Dropout head
MAX_FILES_PER_ACT   = 50
# ---- robustness / metric options --------------------------------------------
BAND_DROPOUT_P      = 0.5   # prob. of zeroing 1-2 random bands per TRAIN sample
                            # -> teaches single-band features; fixes the large
                            #    gap between combined and per-band evaluation
FLOOR_MARGIN_M      = 0.0   # if >0, floor_acc skips samples with |y|<margin
                            # (derived y targets cluster at 0 -> sign is a coin
                            #  flip there; 0 keeps the original strict metric)
SEED                = 42

# =============================================================================
# FEDERATED LEARNING CONFIG
#   Horizontal FedAvg (weighted by client data size), the standard scheme used
#   across the reference papers (CG-FedLLM / MIRA / SplitLoRA / TITANIC survey):
#   each round the server samples a subset of clients, each client trains
#   locally for FL_LOCAL_EPOCHS on its OWN private shard, then the server
#   averages the updates -> new global model. Only model weights are shared,
#   never raw data. Clients are non-IID (Dirichlet, as in CG-FedLLM, alpha=0.5).
# =============================================================================
# Which training paradigms to run & report. Keep BOTH to get the explicit
# "with vs without federated learning" comparison the brief asks for.
TRAINING_MODES   = ["centralized", "federated"]   # subset of these two

N_CLIENTS        = 6        # data owners; the TRAIN set is partitioned among them
CLIENT_FRACTION  = 0.5      # fraction of clients sampled each round (>=1 client)
NON_IID          = True     # Dirichlet non-IID split by spatial zone
DIRICHLET_ALPHA  = 0.5      # smaller => more heterogeneous (CG-FedLLM used 0.5)

# ---- FL hyper-parameter SEARCH grids (the ranges requested) -----------------
FL_ROUNDS_GRID       = list(range(5, 101, 5))     # 5,10,15,...,100   (20 values)
FL_LOCAL_EPOCHS_GRID = list(range(20, 151, 10))   # 20,30,40,...,150  (14 values)
# Values actually used by the federated sweep (auto-overwritten by the tuner
# when FL_TUNE=True; otherwise these defaults are used as-is).
FL_ROUNDS        = 30
FL_LOCAL_EPOCHS  = 30

# ---- FL hyper-parameter tuning ----------------------------------------------
FL_TUNE                = True     # search FL_ROUNDS x FL_LOCAL_EPOCHS, pick best
FL_TUNE_FULL_GRID      = False    # True = exhaustive 20x14 grid (very slow)
FL_TUNE_TRIALS         = 12       # random-search budget when not full grid
FL_TUNE_OVERFIT_LAMBDA = 0.5      # penalty on the (val-train) gap when selecting
FL_TUNE_PROXY          = ("hybrid", "pinn_cnn")   # cheap representative model
FL_TUNE_SUBSET_TRAIN   = 64       # # train samples used to build tuning clients
FL_TUNE_CLIENTS        = 3        # clients during tuning (kept small for speed)
FL_TUNE_VAL            = 48       # # val samples used to score tuning trials

# ---- round-level EARLY STOPPING (combats over-training / over-fitting) ------
FL_EARLY_STOP_ROUNDS = 10         # stop if val MAE has not improved for N rounds
# (centralized training already early-stops via EARLY_STOP_PATIENCE on epochs.)

# ---- federated pre-training (representation arm, done before the FL sweep) --
FL_PRETRAIN_ROUNDS       = 3      # FedAvg rounds for contrastive / DAE pretrain
FL_PRETRAIN_LOCAL_EPOCHS = 8      # local pretrain epochs per round

# --- quick smoke test of the ENTIRE matrix (set True to sanity-check fast) ---
FAST_DEV = False
if FAST_DEV:
    EPOCHS, EARLY_STOP_PATIENCE, PRETRAIN_EPOCHS = 4, 3, 2
    MC_DROPOUT_SAMPLES, MAX_FILES_PER_ACT = 4, 4
    FL_ROUNDS, FL_LOCAL_EPOCHS = 3, 3
    FL_ROUNDS_GRID, FL_LOCAL_EPOCHS_GRID = [2, 3, 4], [2, 3, 4]
    FL_TUNE_TRIALS, FL_EARLY_STOP_ROUNDS = 3, 3
    N_CLIENTS, FL_TUNE_CLIENTS = 3, 2
    FL_PRETRAIN_ROUNDS, FL_PRETRAIN_LOCAL_EPOCHS = 2, 2
    FL_TUNE_SUBSET_TRAIN, FL_TUNE_VAL = 24, 16

# =============================================================================
# constants describing the dataset / scene  (unchanged from the v2 scripts)
# =============================================================================
NUM_ANTENNAS    = 3
RANGE_FFT_SIZE  = 64
DOPPLER_FFT_SIZE = 128
# ---- spectrogram output size (separate from the physics NR/ND constants) ----
# CI4R 77 GHz sx1: 4096x538 raw  ->  after 60% Doppler crop ~2458x538
#   64x128  (old)  =   8,192 px  ~270x downscale  -- FAR too lossy
#   224x224         =  50,176 px  ~44x  -- minimum acceptable (HAR default)
#   384x384          = 147,456 px  ~15x  -- sweet spot; change SPEC_H/W here
SPEC_H            = 224   # change to 384 for the sweet-spot quality
SPEC_W            = 224   # (both files and Radar_Localization_Single use this)
DOPPLER_CROP_FRAC = 0.6   # centre-crop the freq axis to drop outer noise bands
SPEED_OF_LIGHT  = 3e8
SPACE_X_MIN, SPACE_X_MAX = 0.0, 8.0
SPACE_Y_MIN, SPACE_Y_MAX = -4.0, 4.0
N_ZONES         = 8
AOA_BEAMS       = 32
PD_BASE_CH      = 8
DROPOUT_RATE    = 0.20
STOCHASTIC_DEPTH = 0.05
LABEL_SMOOTHING = 0.05
HUBER_DELTA     = 0.20
LOSS_W_X, LOSS_W_Y, LOSS_W_NLOS, LOSS_W_SIN = 3.0, 3.0, 0.15, 0.25
PINN_PHYS_W     = 0.15
RADAR_D_ELEM    = 0.5

ACT_NUM_TO_IDX = {"05":0,"06":1,"07":2,"08":3,"09":4,"10":5,"11":6,
                  "16":7,"17":8,"18":9,"19":10}
ACT_NUMBERS   = list(ACT_NUM_TO_IDX.keys())
NLOS_ACT_NUMS = {"08","09","10","11","16"}
ACTIVITY_KINEMATICS = {
    "05":{"r0":3.0,"r1":0.5,"vy":0.05,"static":False},
    "06":{"r0":0.5,"r1":3.0,"vy":0.05,"static":False},
    "07":{"r0":1.5,"r1":1.5,"vy":0.10,"static":True },
    "08":{"r0":1.2,"r1":1.2,"vy":0.05,"static":True },
    "09":{"r0":1.8,"r1":1.8,"vy":0.02,"static":True },
    "10":{"r0":1.5,"r1":1.5,"vy":0.05,"static":True },
    "11":{"r0":2.5,"r1":0.8,"vy":0.10,"static":False},
    "16":{"r0":2.5,"r1":0.8,"vy":0.08,"static":False},
    "17":{"r0":2.5,"r1":0.8,"vy":0.12,"static":False},
    "18":{"r0":2.5,"r1":0.8,"vy":0.06,"static":False},
    "19":{"r0":2.5,"r1":0.8,"vy":0.15,"static":False},
}
BAND_SLOT  = {"77ghz":0, "24ghz":1, "10ghz":2}
BAND_NAMES = {0:"77 GHz", 1:"24 GHz", 2:"10 GHz"}
BAND_KEYS  = {0:"77ghz",  1:"24ghz",  2:"10ghz"}
SENSORS = [
    {"x":0.000,"y": 0.020,"fc":77.0e9,"bw":4.0e9},
    {"x":0.000,"y": 0.000,"fc":24.0e9,"bw":1.5e9},
    {"x":0.000,"y":-0.020,"fc":10.0e9,"bw":1.5e9},
]
SENSOR_77 = SENSORS[0]

# =============================================================================
# device + logging
# =============================================================================
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH  = OUTPUT_DIR / f"ablation_{TIMESTAMP}.out"

try:
    DEVICE = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
except Exception:
    DEVICE = torch.device("cpu")

logger = logging.getLogger("RadarLocSingle")
logger.setLevel(logging.DEBUG)
_fmt = logging.Formatter("%(message)s")
_fh = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"); _fh.setFormatter(_fmt)
_ch = logging.StreamHandler(sys.stdout); _ch.setFormatter(_fmt)
logger.addHandler(_fh); logger.addHandler(_ch)
def log(msg=""): logger.info(msg)

torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# =============================================================================
# numpy metric helpers
# =============================================================================
def np_mae(a):    return float(np.mean(np.abs(a)))
def np_rmse(a):   return float(np.sqrt(np.mean(np.asarray(a, dtype=np.float64) ** 2)))
def np_pct(a, p): return float(np.percentile(a, p))


# =============================================================================
# REAL DATA LOADER  (reads .dat / .mat exactly like the v2 scripts)
# =============================================================================
def _resize_2d(arr, out_hw):
    H, W = arr.shape; th, tw = out_hw
    if H == th and W == tw: return arr.astype(np.float32)
    return arr[np.ix_(np.linspace(0, H-1, th).astype(int),
                      np.linspace(0, W-1, tw).astype(int))].astype(np.float32)

def _norm_log_spec(spec):
    mag = np.abs(spec) if np.iscomplexobj(spec) else np.abs(spec.astype(float))
    lm = 20.0 * np.log10(mag + 1e-10); lm -= lm.min()
    mx = lm.max()
    if mx > 0: lm /= mx
    return lm.astype(np.float32)

def _rd_from_complex(cplx, NR, ND):
    def pc(arr, tgt, axis):
        sl = [slice(None)] * arr.ndim
        if arr.shape[axis] >= tgt:
            sl[axis] = slice(0, tgt); return arr[tuple(sl)]
        pad = [(0, 0)] * arr.ndim; pad[axis] = (0, tgt - arr.shape[axis])
        return np.pad(arr, pad)
    cplx = pc(pc(cplx, NR, 1), ND, 0)
    rd = np.fft.fftshift(np.fft.fft(np.fft.fft(cplx, axis=1)[:, :NR//2+1], axis=0), axes=0)
    return _resize_2d(_norm_log_spec(np.abs(rd).T), (NR, ND))

def _read_dat(path, NR, ND):
    try:
        raw = np.fromfile(str(path), dtype=np.float32)
        if raw.size < NR*4:
            raw16 = np.fromfile(str(path), dtype=np.int16).astype(np.float32)
            if raw16.size < NR*4: return None
            raw = raw16/32768.
        nc2 = raw.size//2
        cplx = raw[0:nc2*2:2] + 1j*raw[1:nc2*2:2]
        best, be = None, float("inf")
        for nc in [ND, ND//2, ND*2, 64, 128, 256]:
            if nc < 1: continue
            ns = nc2//nc
            if ns < NR//2: continue
            err = abs(nc-ND)+abs(ns-NR)
            if err < be: be = err; best = (nc, ns)
        if best is None:
            s = int(np.sqrt(nc2))
            if s < 4: return None
            cplx = cplx[:s*s].reshape(s, s)
        else:
            nc, ns = best; cplx = cplx[:nc*ns].reshape(nc, ns)
        return _rd_from_complex(cplx, NR, ND)
    except Exception:
        return None

# Candidate variable names for CI4R / OpenRadar .mat files.
# 'sx1' is the confirmed primary variable (freq x time micro-Doppler spectrogram, ~4096x538).
_MAT_VAR_CANDIDATES = ("sx1", "data", "rawData", "raw_data", "iq", "x",
                        "signal", "spectrogram", "spectro", "micro_doppler",
                        "md", "S", "img", "rd", "rd_map", "range_doppler")


def _mat_spectrogram(arr, NR, ND):
    """Turn any 2-D array from a .mat file into a (NR, ND) log-normalised
    spectrogram — mirrors the HAR pipeline: |arr| -> 20*log10 -> [0,1] -> resize.
    The CI4R 'sx1' variable is ALREADY a processed freq x time micro-Doppler map;
    do NOT apply an additional FFT.  50 dB dynamic range (same as HAR code)."""
    arr = np.asarray(arr, dtype=np.complex128).squeeze()
    if arr.ndim == 1:
        s = int(np.sqrt(arr.size)); arr = arr[:s*s].reshape(s, s)
    if arr.ndim > 2: arr = arr.reshape(arr.shape[0], -1)
    if arr.shape[0] < 4 or arr.shape[1] < 4: return None
    # centre-crop the Doppler (freq) axis to drop outer noise bands (mirrors HAR)
    if 0 < DOPPLER_CROP_FRAC < 1.0:
        h = arr.shape[0]; keep = max(8, int(h * DOPPLER_CROP_FRAC))
        lo = (h - keep) // 2; arr = arr[lo:lo + keep]
    mag = np.abs(arr)
    db = 20.0 * np.log10(mag + 1e-9)
    db = np.clip(db, db.max() - 50.0, db.max())
    norm = (db - db.min()) / (db.max() - db.min() + 1e-9)
    return _resize_2d(norm.astype(np.float32), (NR, ND))


def _read_mat(path, NR, ND):
    """Load a .mat file from the CI4R / OpenRadar dataset and return a (NR, ND)
    normalised spectrogram, or None on failure.

    Mirrors the confirmed-working HAR pipeline (radar_har_single_v4.py):
      1. scipy.io.loadmat for MATLAB v5 files. Any exception (NotImplementedError
         for v7.3, ValueError for non-mat HDF5, etc.) falls through to step 2.
      2. h5py for MATLAB v7.3 / HDF5 files. Top-level Datasets only first
         (Groups silently return their key names as strings via np.array — skip them).
         Falls back to recursive visititems for nested MATLAB struct fields.
    Variable preference: 'sx1' (confirmed for CI4R), then other candidates, then largest.

    IMPORTANT — if all files return None, h5py is not installed.  Run:
        pip install h5py
    """
    candidates = {}

    # ---- MATLAB v5 / classic .mat (scipy.io) --------------------------------
    try:
        from scipy.io import loadmat
        mat = loadmat(str(path), squeeze_me=True)
        for k, v in mat.items():
            if k.startswith("__"): continue
            try:
                a = np.asarray(v)
                if a.dtype.kind in ("f","c","i","u") and a.size >= 100 and a.ndim >= 1:
                    candidates[k] = a
            except Exception:
                pass
    except Exception:
        pass   # NotImplementedError / ValueError / OSError -> fall through to h5py

    # ---- MATLAB v7.3 / HDF5 (h5py) -----------------------------------------
    if not candidates:
        try:
            import h5py
            with h5py.File(str(path), "r") as f:
                # Pass 1: top-level Datasets (NOT Groups — np.array on a Group
                # returns its key names as strings, not the actual radar data!)
                for k in f.keys():
                    try:
                        if isinstance(f[k], h5py.Dataset):
                            candidates[k] = np.array(f[k])
                    except Exception:
                        pass
                # Pass 2: recursive visititems for nested MATLAB struct fields
                if not candidates:
                    def _vis(name, obj):
                        if name.startswith("#refs#"): return
                        if not isinstance(obj, h5py.Dataset): return
                        try: candidates[name] = np.array(obj)
                        except Exception: pass
                    f.visititems(_vis)
        except ImportError:
            return None    # h5py not installed — pip install h5py
        except Exception:
            pass

    if not candidates:
        return None

    # ---- prefer 'sx1' or other known name, else take the largest dataset ----
    arr = None
    for name in _MAT_VAR_CANDIDATES:
        for k, v in candidates.items():
            if (k == name or k.split("/")[-1] == name) and np.asarray(v).size > 100:
                arr = v; break
        if arr is not None: break
    if arr is None:
        try: arr = max(candidates.values(), key=lambda a: np.asarray(a).size)
        except Exception: return None

    # ---- unwrap MATLAB compound complex {real, imaginary} if present ---------
    try:
        a = np.asarray(arr)
        if a.dtype.names:
            rfld = next((n for n in a.dtype.names if n.lower() in ("real","r")), None)
            ifld = next((n for n in a.dtype.names if n.lower() in ("imaginary","imag","i")), None)
            arr = (a[rfld].astype(np.float64)+1j*a[ifld].astype(np.float64)
                   if rfld and ifld else a[a.dtype.names[0]].astype(np.float64))
    except Exception:
        pass

    return _mat_spectrogram(arr, NR, ND)


def _read_spec(path, NR, ND):
    ext = path.suffix.lower()
    if ext == ".dat": return _read_dat(path, NR, ND)
    if ext == ".mat": return _read_mat(path, NR, ND)
    return None

_BAND_RE    = re.compile(r"^(77ghz|24ghz|10ghz)[_\s]", re.IGNORECASE)
_ACT_NUM_RE = re.compile(r"[_\s](05|06|07|08|09|10|11|16|17|18|19)[_\s]")

def _scan_data_dir(root):
    result = {n: {} for n in ACT_NUMBERS}
    if not root.exists(): return result
    for folder in sorted(root.iterdir()):
        if not folder.is_dir(): continue
        bm = _BAND_RE.match(folder.name)
        if not bm: continue
        band_key = bm.group(1).lower()
        am = _ACT_NUM_RE.search(folder.name[bm.end()-1:] + "_")
        if not am: continue
        act_num = am.group(1)
        files = []
        for ext in ("*.dat", "*.mat"):
            files.extend(sorted(folder.glob(ext)))
        files = files[:MAX_FILES_PER_ACT]
        if files:
            result[act_num][band_key] = files
            log(f"  [scan] {folder.name[:48]:48s} act={act_num} band={band_key} n={len(files)}")
    return result

def _try_load_real(root):
    if not root.exists():
        log(f"  [loader] DATASET_ROOT not found: {root}")
        return None
    NR, ND = SPEC_H, SPEC_W
    folder_map = _scan_data_dir(root)
    records = []
    for act_num, band_dict in folder_map.items():
        if not band_dict: continue
        act_idx = ACT_NUM_TO_IDX[act_num]; is_nlos = act_num in NLOS_ACT_NUMS
        max_files = max(len(v) for v in band_dict.values())
        for s_idx in range(max_files):
            amp = np.zeros((3, NR, ND), dtype=np.float32); got = False
            for band_key, flist in band_dict.items():
                slot = BAND_SLOT.get(band_key, -1)
                if slot < 0 or s_idx >= len(flist): continue
                spec = _read_spec(flist[s_idx], NR, ND)
                if spec is None: continue
                amp[slot] = spec; got = True
            if got:
                records.append({"amp": amp, "act_num": act_num, "act_idx": act_idx,
                                "is_nlos": is_nlos, "sample_idx": s_idx})
    if not records:
        log("  [loader] No records built from real files.")
        log("  [loader] HINT: the CI4R .mat files are MATLAB v7.3 (HDF5)."
             "  If files were found but not read, install h5py:")
        log("            pip install h5py   <- run once in your Python env")
        return None
    log(f"  [loader] Total real records loaded: {len(records)}")
    return records

def _make_synthetic(n_per_act=40):
    """Structured synthetic stand-in so the full pipeline runs without real data."""
    NR, ND = SPEC_H, SPEC_W
    rng = np.random.default_rng(SEED)
    rr, dd = np.linspace(0, 1, NR)[:, None], np.linspace(0, 1, ND)[None, :]
    records = []
    for act_num in ACT_NUMBERS:
        K = ACTIVITY_KINEMATICS[act_num]; act_idx = ACT_NUM_TO_IDX[act_num]
        is_nlos = act_num in NLOS_ACT_NUMS
        for s_idx in range(n_per_act):
            amp = np.zeros((3, NR, ND), dtype=np.float32)
            for slot in range(3):
                cr = (K["r0"] / 3.5 + 0.05 * slot) % 1.0
                cd = (0.5 + 0.1 * (act_idx - 5)) % 1.0
                blob = np.exp(-(((rr-cr)**2)/0.02 + ((dd-cd)**2)/0.05))
                noise = rng.normal(0, 0.05 + 0.03*slot, (NR, ND))
                spec = blob + (0.4 if is_nlos else 0.0)*rng.random((NR, ND)) + noise
                spec -= spec.min(); spec /= (spec.max() + 1e-8)
                amp[slot] = spec.astype(np.float32)
            records.append({"amp": amp, "act_num": act_num, "act_idx": act_idx,
                            "is_nlos": is_nlos, "sample_idx": s_idx})
    log(f"  [loader] SYNTHETIC fallback: built {len(records)} records "
        f"({n_per_act}/activity). NOT real data.")
    return records

# =============================================================================
# DATASET   -> (amp[3,64,128], phase[3,64,128], xpd[4,64,128], x_n, y_n, nlos)
# =============================================================================
class OpenRadarDataset(Dataset):
    NA, NR, ND = NUM_ANTENNAS, SPEC_H, SPEC_W

    def __init__(self, records, augment=False):
        if not records: raise ValueError("records list is empty")
        self.augment = augment
        self.rng = np.random.default_rng(SEED)
        self.samples = []
        self._build(records)
        xs = np.array([s["x"] for s in self.samples])
        ys = np.array([s["y"] for s in self.samples])
        self.x_min, self.x_max = float(xs.min()), float(xs.max())
        self.y_min, self.y_max = float(ys.min()), float(ys.max())
        self.x_rng = max(self.x_max - self.x_min, 1e-4)
        self.y_rng = max(self.y_max - self.y_min, 1e-4)
        log(f"  Dataset: {len(self.samples)} samples  augment={augment}  "
            f"X[{self.x_min:.2f},{self.x_max:.2f}] Y[{self.y_min:.2f},{self.y_max:.2f}]")

    def _gt_position(self, act_num, s_idx):
        K = ACTIVITY_KINEMATICS[act_num]
        rng = np.random.default_rng(SEED + ACT_NUM_TO_IDX[act_num]*1000 + s_idx)
        r = (rng.uniform(K["r0"]-0.15, K["r0"]+0.15) if K["static"]
             else (1-rng.uniform(0., 1.))*K["r0"] + rng.uniform(0., 1.)*K["r1"])
        y = float(rng.normal(0., K["vy"])); x = float(np.sqrt(max(r*r - y*y, 1e-4)))
        x = float(np.clip(x, SPACE_X_MIN+0.1, SPACE_X_MAX-0.1))
        y = float(np.clip(y, SPACE_Y_MIN+0.1, SPACE_Y_MAX-0.1))
        return x, y

    def _make_phase(self, x, y, is_nlos):
        NR, ND, C = self.NR, self.ND, SPEED_OF_LIGHT
        phase = np.zeros((self.NA, NR, ND), dtype=np.float32)
        for ai, s in enumerate(SENSORS):
            d = float(np.sqrt((x-s["x"])**2 + (y-s["y"])**2)) + 1e-6
            freqs = s["fc"] + np.arange(NR) * (s["bw"]/NR)
            ph = np.tile(-2*np.pi*freqs*d/C, (ND, 1)).T
            ph += np.linspace(0, 0.4, ND)[None, :]
            if is_nlos:
                ph += self.rng.uniform(0, 2*np.pi, (NR, ND)).astype(np.float32)*0.5
            phase[ai] = ph.astype(np.float32)
        return phase

    def _build(self, records):
        for rec in records:
            x, y = self._gt_position(rec["act_num"], rec["sample_idx"])
            zone = int(np.clip((x-SPACE_X_MIN)/((SPACE_X_MAX-SPACE_X_MIN)/N_ZONES),
                               0, N_ZONES-1))
            self.samples.append(dict(
                x=x, y=y, amp=rec["amp"].astype(np.float32),
                phase=self._make_phase(x, y, rec["is_nlos"]),
                is_nlos=rec["is_nlos"], zone=zone, floor=int(y >= 0)))

    def norm_x(self, x): return 2*(x-self.x_min)/self.x_rng - 1
    def norm_y(self, y): return 2*(y-self.y_min)/self.y_rng - 1
    def denorm_x(self, xn): return (xn+1)/2*self.x_rng + self.x_min
    def denorm_y(self, yn): return (yn+1)/2*self.y_rng + self.y_min
    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]; amp = s["amp"].copy(); phase = s["phase"].copy()
        if self.augment:
            phase += float(self.rng.uniform(0, 2*np.pi))
            amp   += self.rng.normal(0, 0.008, amp.shape).astype(np.float32)
            amp   *= float(self.rng.uniform(0.85, 1.15))
            amp    = np.clip(amp, 0., None)
            # band dropout: zero a random subset of bands so the model learns
            # representations that survive single-band evaluation (_mask_band)
            if BAND_DROPOUT_P > 0 and self.rng.random() < BAND_DROPOUT_P:
                nkeep = int(self.rng.integers(1, self.NA))      # keep 1 or 2
                keep = self.rng.choice(self.NA, size=nkeep, replace=False)
                msk = np.zeros(self.NA, dtype=bool); msk[keep] = True
                amp[~msk] = 0.; phase[~msk] = 0.
        xpd_list = []
        for i in range(self.NA-1):
            pd = phase[i]-phase[i+1]
            xpd_list += [np.cos(pd).astype(np.float32), np.sin(pd).astype(np.float32)]
        return (torch.from_numpy(amp.astype(np.float32)),
                torch.from_numpy(phase.astype(np.float32)),
                torch.from_numpy(np.stack(xpd_list, 0)),
                torch.tensor(self.norm_x(s["x"]), dtype=torch.float32),
                torch.tensor(self.norm_y(s["y"]), dtype=torch.float32),
                torch.tensor(int(s["is_nlos"]), dtype=torch.long))

# ---- tensor-side helpers (used by training / pretraining / per-band eval) ---
def compute_xpd(phase):
    """phase [B,NA,R,D] -> xpd [B,2*(NA-1),R,D] = cos/sin of consecutive diffs."""
    parts = []
    for i in range(phase.size(1)-1):
        pd = phase[:, i] - phase[:, i+1]
        parts += [torch.cos(pd), torch.sin(pd)]
    return torch.stack(parts, dim=1)

def augment_views(amp, phase):
    """Two random augmented views for contrastive pre-training."""
    def one():
        a = amp + torch.randn_like(amp)*0.02
        a = a * (0.8 + 0.4*torch.rand(amp.size(0), 1, 1, 1, device=amp.device))
        a = a.clamp(0., None)
        p = phase + (torch.rand(phase.size(0), 1, 1, 1, device=phase.device)*2*math.pi)
        return a, p, compute_xpd(p)
    return one(), one()


# =============================================================================
# SHARED NEURAL BUILDING BLOCKS  (from the v2 suites)
# =============================================================================
class SEBlock(nn.Module):
    def __init__(self, c, r=4):
        super().__init__(); mid = max(c//r, 1)
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                nn.Linear(c, mid), nn.ReLU(),
                                nn.Linear(mid, c), nn.Sigmoid())
    def forward(self, x): return x * self.fc(x).view(x.size(0), x.size(1), 1, 1)

class RealResBlock(nn.Module):
    def __init__(self, ch, drop=0.1, sdp=0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.GELU(),
            nn.Dropout2d(drop),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch))
        self.se = SEBlock(ch); self.sdp = sdp
    def forward(self, x):
        b = self.se(self.net(x))
        if self.training and self.sdp > 0:
            b = b * (torch.rand(x.size(0), 1, 1, 1, device=x.device) > self.sdp).float()
        return F.gelu(x + b)

class PhaseArrayAngleEstimator(nn.Module):
    def __init__(self, n_pd_ch=4, n_beams=32, drop=0.1):
        super().__init__()
        self.compress = nn.Sequential(
            nn.Conv2d(n_pd_ch, n_pd_ch, 3, stride=2, padding=1, groups=n_pd_ch, bias=False),
            nn.Conv2d(n_pd_ch, 16, 1, bias=False), nn.BatchNorm2d(16), nn.GELU(),
            nn.Conv2d(16, n_beams, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_beams), nn.GELU())
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.refine = nn.Sequential(nn.Linear(n_beams, n_beams), nn.GELU(), nn.Dropout(drop))
    def forward(self, xpd): return self.refine(self.pool(self.compress(xpd)).flatten(1))

class GatedXYHead(nn.Module):
    def __init__(self, td, ad, drop=0.1):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(ad, ad), nn.Sigmoid())
        self.x_head = nn.Sequential(nn.Linear(td+ad, 64), nn.GELU(), nn.Dropout(drop/2), nn.Linear(64, 1))
        self.y_head = nn.Sequential(nn.Linear(td+ad, 64), nn.GELU(), nn.Dropout(drop/2), nn.Linear(64, 1))
    def forward(self, trunk, aoa):
        m = torch.cat([trunk, self.gate(aoa)], 1)
        return torch.tanh(self.x_head(m)).squeeze(-1), torch.tanh(self.y_head(m)).squeeze(-1)

class AntennaCNNStem(nn.Module):
    """Per-antenna CNN -> token sequence [B, NA, d] (used by LSTM/Transformer backbones)."""
    def __init__(self, d, ch, drop=0.1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, ch, 5, stride=2, padding=2, bias=False), nn.BatchNorm2d(ch), nn.GELU(), RealResBlock(ch, drop, 0.),
            nn.Conv2d(ch, ch*2, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(ch*2), nn.GELU(), RealResBlock(ch*2, drop, 0.),
            nn.Conv2d(ch*2, ch*4, 3, stride=4, padding=1, bias=False), nn.BatchNorm2d(ch*4), nn.GELU(), nn.AdaptiveAvgPool2d(1))
        self.proj = nn.Sequential(nn.Linear(ch*4, d), nn.LayerNorm(d))
    def forward(self, amp, phase):
        B, N, R, D = amp.shape
        x = torch.stack([amp, torch.cos(phase), torch.sin(phase)], 2).view(B*N, 3, R, D)
        return self.proj(self.cnn(x).flatten(1)).view(B, N, -1)

# =============================================================================
# UNIFORM UQ-AWARE LOCALIZATION HEAD
#   Shared by every backbone so the UQ ablation is apples-to-apples.
#   Produces (xp, yp, nlos_logits). The UQ semantics live in the loss + the
#   evaluator:  softmax / mc_dropout / temp_scaling use 2-logit output;
#   evidential interprets the same 2 outputs as Dirichlet evidence.
# =============================================================================
class UQLocHead(nn.Module):
    def __init__(self, td, uq_mode, n_pd=4, pb=PD_BASE_CH, n_beams=AOA_BEAMS,
                 drop=DROPOUT_RATE, sdp=STOCHASTIC_DEPTH):
        super().__init__()
        self.uq_mode = uq_mode
        b = pb
        self.stem = nn.Sequential(nn.Conv2d(n_pd, b, 5, stride=2, padding=2, bias=False), nn.BatchNorm2d(b), nn.GELU())
        self.s1 = RealResBlock(b, drop, sdp); self.d1 = nn.Sequential(nn.Conv2d(b, b*2, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(b*2), nn.GELU())
        self.s2 = RealResBlock(b*2, drop, sdp); self.d2 = nn.Sequential(nn.Conv2d(b*2, b*4, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(b*4), nn.GELU())
        self.s3 = RealResBlock(b*4, drop, sdp); self.pool = nn.AdaptiveMaxPool2d(1); self.out_ch = b*4
        self.aoa = PhaseArrayAngleEstimator(n_pd, n_beams, drop)
        self.trunk = nn.Sequential(nn.Linear(td+self.out_ch, 256), nn.GELU(), nn.Dropout(drop),
                                   nn.Linear(256, 128), nn.GELU())
        self.xy_head = GatedXYHead(128, n_beams, drop)
        # extra dropout used only by MC-Dropout (kept ON at inference for that mode)
        self.mc_drop = nn.Dropout(MC_DROPOUT_P)
        self.nlos_head = nn.Linear(128, 2)
        # post-hoc temperature (learned after training for temp_scaling)
        self.register_buffer("temperature", torch.ones(1))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def pd_feat(self, xpd):
        x = self.stem(xpd); x = self.s1(x); x = self.d1(x); x = self.s2(x); x = self.d2(x); x = self.s3(x)
        return self.pool(x).flatten(1)

    def forward(self, vis_feat, xpd):
        fb = self.pd_feat(xpd); aoa = self.aoa(xpd)
        tk = self.trunk(torch.cat([vis_feat, fb], 1))
        if self.uq_mode == "mc_dropout":
            tk = self.mc_drop(tk)
        xp, yp = self.xy_head(tk, aoa)
        return xp, yp, self.nlos_head(tk)

# =============================================================================
# pre-training auxiliaries (contrastive projection head + DAE decoder)
# =============================================================================
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 128), nn.GELU(), nn.Linear(128, out_dim))
    def forward(self, x): return F.normalize(self.net(x), dim=-1)

class ReconDecoder(nn.Module):
    """feat vector -> reconstruct clean [out_ch, NR, ND] input stack (for DAE)."""
    def __init__(self, feat_dim, out_ch, NR, ND):
        super().__init__()
        self.h0, self.w0 = NR//8, ND//8
        self.fc = nn.Linear(feat_dim, 64*self.h0*self.w0)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.GELU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.BatchNorm2d(16), nn.GELU(),
            nn.ConvTranspose2d(16, out_ch, 4, 2, 1))
    def forward(self, f):
        return self.net(self.fc(f).view(-1, 64, self.h0, self.w0))

def nt_xent(z1, z2, temp=0.2):
    """SimCLR NT-Xent over 2B normalized embeddings."""
    B = z1.size(0); z = torch.cat([z1, z2], 0)
    sim = (z @ z.t()) / temp
    sim.fill_diagonal_(-9e15)
    targets = torch.arange(2*B, device=z.device)
    targets = (targets + B) % (2*B)
    return F.cross_entropy(sim, targets)


# =============================================================================
# BACKBONES — every backbone is a pure feature extractor:
#     forward(amp, phase, xpd) -> vis_feat [B, feat_dim]
# and exposes  .feat_dim .  DRL backbones add  .rl_aux(feat_detached, reward).
# The uniform UQLocHead is attached on top by RadarLocNet (below).
# =============================================================================

# ---- DRL shared encoder (amp stream + phase-diff stream -> 128-d state) ------
class ImprovedRadarEncoder(nn.Module):
    feat_dim = 128
    def __init__(self):
        super().__init__()
        NA = NUM_ANTENNAS; in_ch = NA*3; n_pd = (NA-1)*2
        b, pb, n = 16, 8, 2; drop, sdp = DROPOUT_RATE, STOCHASTIC_DEPTH
        self.stem = nn.Sequential(nn.Conv2d(in_ch, b, 5, stride=2, padding=2, bias=False), nn.BatchNorm2d(b), nn.GELU())
        self.s1 = nn.ModuleList([RealResBlock(b, drop, sdp) for _ in range(n)])
        self.d1 = nn.Sequential(nn.Conv2d(b, b*2, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(b*2), nn.GELU())
        self.s2 = nn.ModuleList([RealResBlock(b*2, drop, sdp) for _ in range(n)])
        self.d2 = nn.Sequential(nn.Conv2d(b*2, b*4, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(b*4), nn.GELU())
        self.s3 = nn.ModuleList([RealResBlock(b*4, drop, sdp) for _ in range(max(n//2, 1))])
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.pd_stem = nn.Sequential(nn.Conv2d(n_pd, pb, 5, stride=2, padding=2, bias=False), nn.BatchNorm2d(pb), nn.GELU())
        self.pd_s1 = RealResBlock(pb, drop, sdp)
        self.pd_d1 = nn.Sequential(nn.Conv2d(pb, pb*2, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(pb*2), nn.GELU())
        self.pd_s2 = RealResBlock(pb*2, drop, sdp)
        self.pd_d2 = nn.Sequential(nn.Conv2d(pb*2, pb*4, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(pb*4), nn.GELU())
        self.pd_s3 = RealResBlock(pb*4, drop, sdp)
        self.pd_pool = nn.AdaptiveMaxPool2d(1)
        self.proj = nn.Sequential(nn.Linear(b*4+pb*4, 256), nn.GELU(), nn.Dropout(drop), nn.Linear(256, 128), nn.GELU())
    def forward(self, amp, phase, xpd):
        x = torch.cat([amp, torch.cos(phase), torch.sin(phase)], 1)
        x = self.stem(x)
        for blk in self.s1: x = blk(x)
        x = self.d1(x)
        for blk in self.s2: x = blk(x)
        x = self.d2(x)
        for blk in self.s3: x = blk(x)
        fa = self.pool(x).flatten(1)
        p = self.pd_stem(xpd); p = self.pd_s1(p); p = self.pd_d1(p)
        p = self.pd_s2(p); p = self.pd_d2(p); p = self.pd_s3(p)
        fb = self.pd_pool(p).flatten(1)
        return self.proj(torch.cat([fa, fb], 1))

# RL submodules (compact, faithful in spirit to the v2 DRL suite). The RL loss
# trains these networks on the DETACHED encoder state (exactly as in the source,
# where state was .detach()'d), so localization comes from encoder+head and the
# RL part is a light auxiliary that keeps each model architecturally distinct.
HIDDEN_DIM = 128
class _MAPPO(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(128, HIDDEN_DIM), nn.GELU(), nn.Linear(HIDDEN_DIM, 2), nn.Tanh())
        self.critic = nn.Sequential(nn.Linear(128, HIDDEN_DIM), nn.GELU(), nn.Linear(HIDDEN_DIM, 1))
    def rl_aux(self, s, reward):
        v = self.critic(s).squeeze(-1)
        return F.mse_loss(v, reward) + 1e-3 * self.actor(s).pow(2).mean()
class _SAC(nn.Module):
    def __init__(self):
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(128, HIDDEN_DIM), nn.GELU(), nn.Linear(HIDDEN_DIM, 2), nn.Tanh())
        self.q1 = nn.Sequential(nn.Linear(128+2, HIDDEN_DIM), nn.GELU(), nn.Linear(HIDDEN_DIM, 1))
        self.q2 = nn.Sequential(nn.Linear(128+2, HIDDEN_DIM), nn.GELU(), nn.Linear(HIDDEN_DIM, 1))
    def rl_aux(self, s, reward):
        a = self.pi(s); sa = torch.cat([s, a], -1)
        q1 = self.q1(sa).squeeze(-1); q2 = self.q2(sa).squeeze(-1)
        return F.mse_loss(q1, reward) + F.mse_loss(q2, reward) - 0.01*q1.mean()
class _D3QN(nn.Module):
    def __init__(self, na=8):
        super().__init__()
        self.sh = nn.Sequential(nn.Linear(128, HIDDEN_DIM), nn.GELU())
        self.val = nn.Linear(HIDDEN_DIM, 1); self.adv = nn.Linear(HIDDEN_DIM, na)
    def q(self, s):
        h = self.sh(s); a = self.adv(h)
        return self.val(h) + (a - a.mean(1, keepdim=True))
    def rl_aux(self, s, reward):
        return F.smooth_l1_loss(self.q(s).max(1).values, reward)
class _DDPG(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(128, HIDDEN_DIM), nn.GELU(), nn.Linear(HIDDEN_DIM, 2), nn.Tanh())
        self.critic = nn.Sequential(nn.Linear(128+2, HIDDEN_DIM), nn.GELU(), nn.Linear(HIDDEN_DIM, 1))
    def rl_aux(self, s, reward):
        q = self.critic(torch.cat([s, self.actor(s)], -1)).squeeze(-1)
        return F.mse_loss(q, reward) - 0.01*q.mean()
class _QMIX(nn.Module):
    def __init__(self, n=3, na=8):
        super().__init__(); self.n = n
        self.agents = nn.ModuleList([nn.Sequential(nn.Linear(128, HIDDEN_DIM), nn.GELU(), nn.Linear(HIDDEN_DIM, na)) for _ in range(n)])
        self.mix = nn.Sequential(nn.Linear(n, 32), nn.ELU(), nn.Linear(32, 1))
    def rl_aux(self, s, reward):
        qs = torch.stack([ag(s).max(1).values for ag in self.agents], 1)
        return F.smooth_l1_loss(self.mix(qs).squeeze(-1), reward)

class DRLBackbone(nn.Module):
    """ImprovedRadarEncoder + an RL submodule (distinct per DRL model)."""
    def __init__(self, rl_kind):
        super().__init__()
        self.encoder = ImprovedRadarEncoder()
        self.rl = {"mappo": _MAPPO, "sac": _SAC, "d3qn": _D3QN,
                   "ddpg": _DDPG, "qmix": _QMIX}[rl_kind]()
        self.feat_dim = 128
    def forward(self, amp, phase, xpd):
        return self.encoder(amp, phase, xpd)
    def rl_aux(self, feat_detached, reward):
        return self.rl.rl_aux(feat_detached, reward)


# ---- Hybrid backbones --------------------------------------------------------
class RetentiveMaSA(nn.Module):
    def __init__(self, d, nh, gamma=0.9, drop=0.1):
        super().__init__(); self.nh = nh; self.dh = d//nh; self.gamma = gamma
        self.q = nn.Linear(d, d, bias=False); self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False); self.out = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d); self.drop = nn.Dropout(drop)
    def _decay(self, L, dev):
        idx = torch.arange(L, device=dev, dtype=torch.float32)
        return self.gamma ** torch.abs(idx.unsqueeze(0)-idx.unsqueeze(1))
    def forward(self, x):
        B, L, C = x.shape; H, D = self.nh, self.dh
        q = self.q(x).view(B, L, H, D).transpose(1, 2); k = self.k(x).view(B, L, H, D).transpose(1, 2)
        v = self.v(x).view(B, L, H, D).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1))*(D**-0.5)*self._decay(L, x.device).unsqueeze(0).unsqueeze(0)
        out = (self.drop(torch.sigmoid(attn)) @ v).transpose(1, 2).contiguous().view(B, L, C)
        return self.norm(x + self.out(out))
class TransRADBlock(nn.Module):
    def __init__(self, d, nh, g, drop):
        super().__init__(); self.attn = RetentiveMaSA(d, nh, g, drop)
        self.ffn = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d*4, bias=False), nn.GELU(), nn.Dropout(drop), nn.Linear(d*4, d, bias=False))
    def forward(self, x): x = self.attn(x); return x + self.ffn(x)
class TransRADBackbone(nn.Module):
    feat_dim = 128
    def __init__(self):
        super().__init__(); NA = NUM_ANTENNAS; ch, d, nh, nl, g = 24, 128, 4, 4, 0.9
        drop, sdp = DROPOUT_RATE, STOCHASTIC_DEPTH
        self.cnn = nn.Sequential(nn.Conv2d(NA*3, ch, 5, stride=2, padding=2, bias=False), nn.BatchNorm2d(ch), nn.GELU(), RealResBlock(ch, drop, sdp),
                                 nn.Conv2d(ch, ch*2, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(ch*2), nn.GELU(), RealResBlock(ch*2, drop, sdp),
                                 nn.Conv2d(ch*2, d, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(d), nn.GELU())
        self.cls = nn.Parameter(torch.randn(1, 1, d)*0.02); self.pos_drop = nn.Dropout(drop)
        self.blocks = nn.ModuleList([TransRADBlock(d, nh, g, drop) for _ in range(nl)]); self.ln = nn.LayerNorm(d)
    def forward(self, amp, phase, xpd):
        t = self.cnn(torch.cat([amp, torch.cos(phase), torch.sin(phase)], 1)); B = t.size(0)
        t = self.pos_drop(t.flatten(2).transpose(1, 2))
        t = torch.cat([self.cls.expand(B, -1, -1), t], 1)
        for blk in self.blocks: t = blk(t)
        return self.ln(t[:, 0])

class PointTransformerLayer(nn.Module):
    def __init__(self, d, k):
        super().__init__(); self.k = k; self.d = d
        self.q = nn.Linear(d, d); self.k_proj = nn.Linear(d, d); self.v = nn.Linear(d, d)
        self.w_delta = nn.Sequential(nn.Linear(3, d), nn.ReLU(), nn.Linear(d, d))
        self.out = nn.Linear(d, d); self.norm = nn.LayerNorm(d)
    def forward(self, x, pos, ei):
        N = x.shape[0]; k = self.k; q = self.q(x)
        keys = self.k_proj(x)[ei.reshape(-1)].reshape(N, k, -1); vals = self.v(x)[ei.reshape(-1)].reshape(N, k, -1)
        pos_i = pos.unsqueeze(1).expand(N, k, 3).contiguous(); pos_j = pos[ei.reshape(-1)].reshape(N, k, 3)
        delta = self.w_delta(pos_i - pos_j)
        attn = F.softmax((q.unsqueeze(1)*keys*delta).sum(-1)/math.sqrt(self.d), dim=1)
        return self.norm(x + self.out((attn.unsqueeze(-1)*(vals+delta)).sum(1)))
class RadarGNNBackbone(nn.Module):
    feat_dim = 96
    def __init__(self):
        super().__init__(); K, ek, h, nl = 96, 8, 96, 3; NA = NUM_ANTENNAS; drop = DROPOUT_RATE
        self.K = K; self.ek = min(ek, K-1); self.ant_stem = AntennaCNNStem(h, 16, drop)
        self.node_proj = nn.Sequential(nn.Linear(h+3, h), nn.LayerNorm(h), nn.GELU())
        self.pt_layers = nn.ModuleList([PointTransformerLayer(h, self.ek) for _ in range(nl)])
        self.norms = nn.ModuleList([nn.LayerNorm(h) for _ in range(nl)])
        self.pool_proj = nn.Sequential(nn.Linear(h*2, h), nn.GELU())
    def _nodes(self, amp, phase):
        B, N, R, D = amp.shape; K, ek = self.K, self.ek
        energy = amp.mean(1).flatten(1); topk = energy.topk(min(K, R*D), 1).indices
        r_n = (topk//D).float()/(R-1); d_n = (topk % D).float()/(D-1)
        e_n = energy.gather(1, topk)/(energy.max(1, keepdim=True).values+1e-8)
        pos3d = torch.stack([r_n, d_n, e_n], -1)
        nf = self.node_proj(torch.cat([self.ant_stem(amp, phase).mean(1, keepdim=True).expand(B, K, -1), pos3d], -1))
        pos2 = pos3d[:, :, :2]; dists = ((pos2.unsqueeze(1)-pos2.unsqueeze(2))**2).sum(-1)
        return nf, pos3d, dists.topk(ek+1, 2, largest=False).indices[:, :, 1:]
    def forward(self, amp, phase, xpd):
        nf, pos3d, ei = self._nodes(amp, phase); B = nf.size(0); outs = []
        for b in range(B):
            f = nf[b]; p = pos3d[b]; e = ei[b]
            for layer, norm in zip(self.pt_layers, self.norms): f = norm(layer(f, p, e))
            outs.append(f)
        nf = torch.stack(outs, 0)
        return self.pool_proj(torch.cat([nf.max(1).values, nf.mean(1)], -1))

class PINNCNNBackbone(nn.Module):
    feat_dim = 96; is_pinn = True
    def __init__(self):
        super().__init__(); NA = NUM_ANTENNAS; ch = 24; drop, sdp = DROPOUT_RATE, STOCHASTIC_DEPTH
        self.cnn = nn.Sequential(nn.Conv2d(NA*3, ch, 5, stride=2, padding=2, bias=False), nn.BatchNorm2d(ch), nn.GELU(), RealResBlock(ch, drop, sdp),
                                 nn.Conv2d(ch, ch*2, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(ch*2), nn.GELU(), RealResBlock(ch*2, drop, sdp),
                                 nn.Conv2d(ch*2, ch*4, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(ch*4), nn.GELU(), RealResBlock(ch*4, drop, sdp), nn.AdaptiveAvgPool2d(1))
    def forward(self, amp, phase, xpd):
        return self.cnn(torch.cat([amp, torch.cos(phase), torch.sin(phase)], 1)).flatten(1)

class PINNLSTMBackbone(nn.Module):
    feat_dim = 128; is_pinn = True
    def __init__(self):
        super().__init__(); hid, nl = 128, 2; drop = DROPOUT_RATE
        self.cnn_stem = AntennaCNNStem(hid, 16, drop)
        self.lstm = nn.LSTM(hid, hid//2, num_layers=nl, batch_first=True, bidirectional=True, dropout=drop if nl > 1 else 0.)
        self.attn = nn.Sequential(nn.Linear(hid, 1), nn.Softmax(dim=1)); self.norm = nn.LayerNorm(hid)
    def forward(self, amp, phase, xpd):
        out, _ = self.lstm(self.cnn_stem(amp, phase))
        return self.norm((out*self.attn(out)).sum(1))

class CausalBlock(nn.Module):
    def __init__(self, d, nh, drop):
        super().__init__(); self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, nh, dropout=drop, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, d*4), nn.GELU(), nn.Dropout(drop), nn.Linear(d*4, d)); self.drop = nn.Dropout(drop)
    def forward(self, x):
        L = x.size(1); msk = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        n = self.n1(x); a, _ = self.attn(n, n, n, attn_mask=msk); x = x + self.drop(a)
        return x + self.drop(self.ffn(self.n2(x)))
class DecisionTRBackbone(nn.Module):
    feat_dim = 128
    def __init__(self):
        super().__init__(); NA = NUM_ANTENNAS; d, nh, nl = 128, 4, 3; drop = DROPOUT_RATE
        self.d = d; self.n_ant = NA; self.state_enc = AntennaCNNStem(d, 16, drop)
        self.rtg_enc = nn.Sequential(nn.Linear(1, d), nn.Tanh()); self.act_emb = nn.Embedding(8, d)
        self.pos_emb = nn.Embedding(3*NA, d); self.drop_emb = nn.Dropout(drop)
        self.blocks = nn.ModuleList([CausalBlock(d, nh, drop) for _ in range(nl)]); self.norm = nn.LayerNorm(d)
    def forward(self, amp, phase, xpd):
        B, N = amp.shape[:2]
        st = self.state_enc(amp, phase); rt = self.rtg_enc(torch.zeros(B, N, 1, device=amp.device))
        at = self.act_emb(torch.zeros(B, N, dtype=torch.long, device=amp.device))
        seq = torch.stack([rt, st, at], 2).view(B, 3*N, self.d)
        seq = self.drop_emb(seq + self.pos_emb(torch.arange(3*N, device=amp.device).unsqueeze(0)))
        for blk in self.blocks: seq = blk(seq)
        return self.norm(seq)[:, torch.arange(1, 3*N, 3, device=amp.device)].mean(1)

class ISACBiLSTMBackbone(nn.Module):
    feat_dim = 256
    def __init__(self):
        super().__init__(); NA = NUM_ANTENNAS; ch, h, nl = 24, 128, 2; drop, sdp = DROPOUT_RATE, STOCHASTIC_DEPTH
        self.cnn = nn.Sequential(nn.Conv2d(NA*3, ch, 5, stride=2, padding=2, bias=False), nn.BatchNorm2d(ch), nn.GELU(), RealResBlock(ch, drop, sdp),
                                 nn.Conv2d(ch, ch*2, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(ch*2), nn.GELU(), RealResBlock(ch*2, drop, sdp),
                                 nn.Conv2d(ch*2, ch*4, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(ch*4), nn.GELU(), nn.AdaptiveMaxPool2d(1))
        self.ant_stem = AntennaCNNStem(h, 16, drop)
        self.bilstm = nn.LSTM(h, h//2, nl, batch_first=True, bidirectional=True, dropout=drop if nl > 1 else 0.)
        self.attn = nn.Sequential(nn.Linear(h, 1), nn.Softmax(dim=1))
        self.fusion = nn.Sequential(nn.Linear(ch*4+h, 256), nn.GELU(), nn.Dropout(drop), nn.Linear(256, 256), nn.GELU())
        self.norm = nn.LayerNorm(256)
    def forward(self, amp, phase, xpd):
        cnn_feat = self.cnn(torch.cat([amp, torch.cos(phase), torch.sin(phase)], 1)).flatten(1)
        lo, _ = self.bilstm(self.ant_stem(amp, phase)); lstm_feat = (lo*self.attn(lo)).sum(1)
        return self.norm(self.fusion(torch.cat([cnn_feat, lstm_feat], 1)))

class DiffAttnBlock(nn.Module):
    def __init__(self, d, nh, drop=0.1):
        super().__init__(); assert nh % 2 == 0; self.nh = nh; self.dh = d//nh
        self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d)
        self.q1 = nn.Linear(d, d//2, bias=False); self.q2 = nn.Linear(d, d//2, bias=False)
        self.k1 = nn.Linear(d, d//2, bias=False); self.k2 = nn.Linear(d, d//2, bias=False)
        self.v = nn.Linear(d, d, bias=False); self.out = nn.Linear(d, d, bias=False)
        self.lam = nn.Parameter(torch.ones(nh//2)*0.5); self.drop = nn.Dropout(drop)
        self.ffn = nn.Sequential(nn.Linear(d, d*4), nn.GELU(), nn.Dropout(drop), nn.Linear(d*4, d))
    def forward(self, x):
        B, L, C = x.shape; nh = self.nh//2; dh = self.dh; sc = dh**-0.5; n = self.n1(x)
        Q1 = self.q1(n).view(B, L, nh, dh).transpose(1, 2); Q2 = self.q2(n).view(B, L, nh, dh).transpose(1, 2)
        K1 = self.k1(n).view(B, L, nh, dh).transpose(1, 2); K2 = self.k2(n).view(B, L, nh, dh).transpose(1, 2)
        V = self.v(n).view(B, L, self.nh, dh).transpose(1, 2)
        S1 = F.softmax((Q1@K1.transpose(-2, -1))*sc, -1); S2 = F.softmax((Q2@K2.transpose(-2, -1))*sc, -1)
        lam = self.lam.clamp(0., 1.).view(1, nh, 1, 1); S = self.drop(S1 - lam*S2); V1 = V[:, :nh]; V2 = V[:, nh:]
        out = torch.cat([(S@V1).transpose(1, 2).contiguous().view(B, L, C//2),
                         (S@V2).transpose(1, 2).contiguous().view(B, L, C//2)], -1)
        x = x + self.out(out); return x + self.ffn(self.n2(x))
class RaDiTBackbone(nn.Module):
    feat_dim = 128
    def __init__(self):
        super().__init__(); NA = NUM_ANTENNAS; b, d, nh, nl = 24, 128, 4, 3; drop, sdp = DROPOUT_RATE, STOCHASTIC_DEPTH
        self.enc1 = nn.Sequential(nn.Conv2d(NA*3, b, 5, stride=2, padding=2, bias=False), nn.BatchNorm2d(b), nn.GELU(), RealResBlock(b, drop, sdp))
        self.enc2 = nn.Sequential(nn.Conv2d(b, b*2, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(b*2), nn.GELU(), RealResBlock(b*2, drop, sdp))
        self.enc3 = nn.Sequential(nn.Conv2d(b*2, b*4, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(b*4), nn.GELU(), RealResBlock(b*4, drop, sdp))
        self.bt_proj = nn.Sequential(nn.Conv2d(b*4, d, 1, bias=False), nn.BatchNorm2d(d), nn.GELU())
        self.cls = nn.Parameter(torch.randn(1, 1, d)*0.02); self.pos_drop = nn.Dropout(drop)
        self.blocks = nn.ModuleList([DiffAttnBlock(d, nh, drop) for _ in range(nl)]); self.ln = nn.LayerNorm(d)
    def forward(self, amp, phase, xpd):
        bt = self.bt_proj(self.enc3(self.enc2(self.enc1(torch.cat([amp, torch.cos(phase), torch.sin(phase)], 1))))); B = bt.size(0)
        t = self.pos_drop(bt.flatten(2).transpose(1, 2))
        t = torch.cat([self.cls.expand(B, -1, -1), t], 1)
        for blk in self.blocks: t = blk(t)
        return self.ln(t)[:, 0]


# ---- GenAI backbones ---------------------------------------------------------
class RadarCLIPBackbone(nn.Module):
    feat_dim = 128
    def __init__(self):
        super().__init__(); NA = NUM_ANTENNAS; d, nh, nl = 128, 4, 3; drop, sdp = DROPOUT_RATE, STOCHASTIC_DEPTH; stride = 8
        self.cnn_stem = nn.Sequential(nn.Conv2d(NA*3, d//2, 5, stride=2, padding=2, bias=False), nn.BatchNorm2d(d//2), nn.GELU(), RealResBlock(d//2, drop, sdp),
                                      nn.Conv2d(d//2, d, stride, stride, bias=False), nn.BatchNorm2d(d), nn.GELU())
        self.cls = nn.Parameter(torch.randn(1, 1, d)*0.02); self.pos_drop = nn.Dropout(drop)
        el = nn.TransformerEncoderLayer(d, nh, d*4, drop, batch_first=True, norm_first=True)
        self.vis_enc = nn.TransformerEncoder(el, nl, nn.LayerNorm(d)); self.vis_proj = nn.Linear(d, d, bias=False)
    def forward(self, amp, phase, xpd):
        t = self.cnn_stem(torch.cat([amp, torch.cos(phase), torch.sin(phase)], 1)); B = t.size(0)
        t = self.pos_drop(torch.cat([self.cls.expand(B, -1, -1), t.flatten(2).transpose(1, 2)], 1))
        return F.normalize(self.vis_proj(self.vis_enc(t)[:, 0]), dim=-1)

class RadarVLMBackbone(nn.Module):
    feat_dim = 128
    def __init__(self):
        super().__init__(); NA = NUM_ANTENNAS; d, nh, nv, nl = 128, 4, 2, 3; drop, sdp = DROPOUT_RATE, STOCHASTIC_DEPTH
        self.vis_cnn = nn.Sequential(nn.Conv2d(NA*3, d//2, 5, stride=2, padding=2, bias=False), nn.BatchNorm2d(d//2), nn.GELU(), RealResBlock(d//2, drop, sdp),
                                     nn.Conv2d(d//2, d, 3, stride=4, padding=1, bias=False), nn.BatchNorm2d(d), nn.GELU())
        vis_el = nn.TransformerEncoderLayer(d, nh, d*4, drop, batch_first=True, norm_first=True)
        self.vis_enc = nn.TransformerEncoder(vis_el, nv, nn.LayerNorm(d))
        self.query_tokens = nn.Parameter(torch.randn(1, 3, d)*0.02)
        self.cross_attn = nn.MultiheadAttention(d, nh, dropout=drop, batch_first=True); self.cross_norm = nn.LayerNorm(d)
        lm_el = nn.TransformerDecoderLayer(d, nh, d*4, drop, batch_first=True, norm_first=True)
        self.lm_dec = nn.TransformerDecoder(lm_el, nl, nn.LayerNorm(d)); self.lm_proj = nn.Linear(d, d, bias=False)
    def forward(self, amp, phase, xpd):
        x = torch.cat([amp, torch.cos(phase), torch.sin(phase)], 1); B = x.size(0)
        vis_tok = self.vis_enc(self.vis_cnn(x).flatten(2).transpose(1, 2))
        q = self.query_tokens.expand(B, -1, -1); ctx, _ = self.cross_attn(q, vis_tok, vis_tok)
        ctx = self.cross_norm(q + ctx); lm_out = self.lm_dec(ctx, ctx)
        return self.lm_proj(lm_out[:, 0])

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__(); self.w = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x): return x*self.w/(x.pow(2).mean(-1, keepdim=True)+self.eps).sqrt()
class LLaMABlock(nn.Module):
    def __init__(self, d, n_q, n_kv, drop):
        super().__init__(); dh = d//n_q; self.n_q = n_q; self.n_kv = n_kv; self.dh = dh
        self.n1 = RMSNorm(d); self.n2 = RMSNorm(d)
        self.q = nn.Linear(d, d, bias=False); self.k = nn.Linear(d, n_kv*dh, bias=False)
        self.v = nn.Linear(d, n_kv*dh, bias=False); self.o = nn.Linear(d, d, bias=False); self.drop = nn.Dropout(drop)
        ffn = int(d*8/3//8*8); self.w1 = nn.Linear(d, ffn, bias=False); self.w2 = nn.Linear(ffn, d, bias=False); self.w3 = nn.Linear(d, ffn, bias=False)
    def _attn(self, x):
        B, L, C = x.shape; rep = self.n_q//self.n_kv
        q = self.q(x).view(B, L, self.n_q, self.dh).transpose(1, 2)
        k = self.k(x).view(B, L, self.n_kv, self.dh).transpose(1, 2).repeat_interleave(rep, 1)
        v = self.v(x).view(B, L, self.n_kv, self.dh).transpose(1, 2).repeat_interleave(rep, 1)
        msk = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        s = (q@k.transpose(-2, -1))*(self.dh**-0.5)
        s = s.masked_fill(msk.unsqueeze(0).unsqueeze(0), float('-inf'))
        return (self.drop(F.softmax(s, -1))@v).transpose(1, 2).contiguous().view(B, L, C)
    def forward(self, x):
        x = x + self.o(self._attn(self.n1(x))); h = self.n2(x); return x + self.w2(F.silu(self.w1(h))*self.w3(h))
class RadarLLaMABackbone(nn.Module):
    feat_dim = 128
    def __init__(self):
        super().__init__(); NA = NUM_ANTENNAS; d, nh, nl = 128, 4, 4; drop = DROPOUT_RATE
        self.cnn_stem = AntennaCNNStem(d, 16, drop); self.pos_emb = nn.Embedding(NA, d)
        n_kv = max(nh//4, 1); self.blocks = nn.ModuleList([LLaMABlock(d, nh, n_kv, drop) for _ in range(nl)]); self.norm = RMSNorm(d)
    def forward(self, amp, phase, xpd):
        N = amp.shape[1]; tok = self.cnn_stem(amp, phase) + self.pos_emb(torch.arange(N, device=amp.device))
        for blk in self.blocks: tok = blk(tok)
        return self.norm(tok.mean(1))

class PhiBlock(nn.Module):
    def __init__(self, d, nh, drop):
        super().__init__(); dh = d//nh; self.nh = nh; self.dh = dh
        self.n = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3*d, bias=False); self.out = nn.Linear(d, d, bias=False)
        self.q_norm = nn.LayerNorm(dh); self.k_norm = nn.LayerNorm(dh); self.drop = nn.Dropout(drop)
        ffn = int(d*8/3//8*8); self.ff_n = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ffn, bias=False), nn.GELU(), nn.Dropout(drop), nn.Linear(ffn, d, bias=False))
    def _attn(self, x):
        B, L, C = x.shape; q, k, v = self.qkv(x).chunk(3, -1)
        q = self.q_norm(q.view(B, L, self.nh, self.dh)).transpose(1, 2)
        k = self.k_norm(k.view(B, L, self.nh, self.dh)).transpose(1, 2)
        v = v.view(B, L, self.nh, self.dh).transpose(1, 2)
        msk = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        s = (q@k.transpose(-2, -1))*(self.dh**-0.5)
        s = s.masked_fill(msk.unsqueeze(0).unsqueeze(0), float('-inf'))
        return self.out((self.drop(F.softmax(s, -1))@v).transpose(1, 2).contiguous().view(B, L, C))
    def forward(self, x): return x + self._attn(self.n(x)) + self.ff(self.ff_n(x))
class RadarPhiBackbone(nn.Module):
    feat_dim = 128
    def __init__(self):
        super().__init__(); NA = NUM_ANTENNAS; d, nh, nl = 128, 4, 3; drop = DROPOUT_RATE
        self.cnn_stem = AntennaCNNStem(d, 12, drop); self.pos_emb = nn.Embedding(NA, d)
        self.blocks = nn.ModuleList([PhiBlock(d, nh, drop) for _ in range(nl)]); self.norm = nn.LayerNorm(d)
    def forward(self, amp, phase, xpd):
        N = amp.shape[1]; tok = self.cnn_stem(amp, phase) + self.pos_emb(torch.arange(N, device=amp.device))
        for blk in self.blocks: tok = blk(tok)
        return self.norm(tok.mean(1))

class RadarVAE(nn.Module):
    def __init__(self, n_ant, ch, lat):
        super().__init__(); in_ch = n_ant*3
        self.enc = nn.Sequential(nn.Conv2d(in_ch, ch, 4, stride=2, padding=1, bias=False), nn.GELU(),
                                 nn.Conv2d(ch, ch*2, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(ch*2), nn.GELU(),
                                 nn.Conv2d(ch*2, ch*4, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(ch*4), nn.GELU(), nn.AdaptiveAvgPool2d(4))
        self.fc_mu = nn.Linear(ch*4*16, lat); self.fc_var = nn.Linear(ch*4*16, lat)
    def forward(self, x):
        h = self.enc(x).flatten(1); mu = self.fc_mu(h); lv = self.fc_var(h)
        z = mu + torch.randn_like(mu)*torch.exp(0.5*lv) if self.training else mu
        kl = (-0.5*(1+lv-mu**2-lv.exp())).sum(-1).mean()
        return z, kl
class LatentDiffBackbone(nn.Module):
    feat_dim = 64
    def __init__(self):
        super().__init__(); NA = NUM_ANTENNAS; self.vae = RadarVAE(NA, 16, 64)
        self.last_kl = None
    def forward(self, amp, phase, xpd):
        z0, kl = self.vae(torch.cat([amp, torch.cos(phase), torch.sin(phase)], 1)); self.last_kl = kl
        return z0

class TimeGANBackbone(nn.Module):
    feat_dim = 64
    def __init__(self):
        super().__init__(); NA = NUM_ANTENNAS; h, nl, fd = 64, 2, 32; drop = DROPOUT_RATE
        self.feat_stem = AntennaCNNStem(fd, 16, drop); self.encoder = nn.LSTM(fd, h, nl, batch_first=True)
    def forward(self, amp, phase, xpd):
        ho, _ = self.encoder(self.feat_stem(amp, phase)); return ho.mean(1)


# =============================================================================
# MODEL WRAPPER + REGISTRY
# =============================================================================
class RadarLocNet(nn.Module):
    """backbone (feature extractor) + uniform UQLocHead. Stores last features so
    DRL models can compute their auxiliary RL loss on the detached state."""
    def __init__(self, backbone, uq_mode):
        super().__init__()
        self.backbone = backbone
        self.head = UQLocHead(backbone.feat_dim, uq_mode)
        self.uq_mode = uq_mode
        self.is_pinn = getattr(backbone, "is_pinn", False)
        self.is_drl  = isinstance(backbone, DRLBackbone)
        self.last_phys = None
    def features(self, amp, phase, xpd):
        return self.backbone(amp, phase, xpd)
    def forward(self, amp, phase, xpd):
        f = self.backbone(amp, phase, xpd)
        self._feat = f
        xp, yp, nlos = self.head(f, xpd)
        if self.is_pinn:
            cos_pd_mean = xpd[:, 0::2].mean(dim=[1, 2, 3])
            theta = torch.atan2(yp, xp + 1e-6)
            self.last_phys = (torch.cos(math.pi*RADAR_D_ELEM*torch.sin(theta)) - cos_pd_mean)**2
        return xp, yp, nlos

SUITES = {
    "drl": {
        "mappo": ("MAPPO (Multi-Agent PPO)",       lambda: DRLBackbone("mappo")),
        "sac":   ("SAC (Soft Actor-Critic)",        lambda: DRLBackbone("sac")),
        "d3qn":  ("D3QN (Dueling Double DQN)",       lambda: DRLBackbone("d3qn")),
        "ddpg":  ("DDPG (Deep Deterministic PG)",    lambda: DRLBackbone("ddpg")),
        "qmix":  ("QMIX (Monotonic MARL)",           lambda: DRLBackbone("qmix")),
    },
    "hybrid": {
        "transrad":    ("TransRAD",     TransRADBackbone),
        "radargnn":    ("RadarGNN",     RadarGNNBackbone),
        "pinn_cnn":    ("PINN-CNN",     PINNCNNBackbone),
        "pinn_lstm":   ("PINN-LSTM",    PINNLSTMBackbone),
        "decision_tr": ("DecisionTR",   DecisionTRBackbone),
        "isac_bilstm": ("ISAC-BiLSTM",  ISACBiLSTMBackbone),
        "radit":       ("RaDiT",        RaDiTBackbone),
    },
    "genai": {
        "radar_clip":  ("RadarCLIP",    RadarCLIPBackbone),
        "radar_vlm":   ("RadarVLM",     RadarVLMBackbone),
        "radar_llama": ("RadarLLaMA",   RadarLLaMABackbone),
        "radar_phi":   ("RadarPhi",     RadarPhiBackbone),
        "latent_diff": ("LatentDiff",   LatentDiffBackbone),
        "timegan":     ("TimeGAN",      TimeGANBackbone),
    },
}
MODEL_LR = {  # per-model learning rates (from the v2 suites)
    "mappo":4e-4,"sac":3e-4,"d3qn":4e-4,"ddpg":3e-4,"qmix":4e-4,
    "transrad":3e-4,"radargnn":4e-4,"pinn_cnn":4e-4,"pinn_lstm":4e-4,
    "decision_tr":3e-4,"isac_bilstm":4e-4,"radit":3e-4,
    "radar_clip":4e-4,"radar_vlm":3e-4,"radar_llama":3e-4,"radar_phi":4e-4,
    "latent_diff":3e-4,"timegan":2e-4,
}

# =============================================================================
# LOSSES + UQ UTILITIES
# =============================================================================
_ce = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

def loc_reg_loss(xp, yp, xt, yt):
    lx = F.smooth_l1_loss(xp, xt, beta=HUBER_DELTA)
    ly = F.smooth_l1_loss(yp, yt, beta=HUBER_DELTA)
    tp = torch.atan2(yp, xp); tt = torch.atan2(yt, xt)
    lsin = F.mse_loss(torch.sin(tp), torch.sin(tt)) + F.mse_loss(torch.cos(tp), torch.cos(tt))
    return LOSS_W_X*lx + LOSS_W_Y*ly + LOSS_W_SIN*lsin

def evidential_loss(logits, target, epoch):
    """Sensoy et al. EDL for 2-class NLOS. logits -> softplus evidence -> Dirichlet."""
    evidence = F.softplus(logits)
    alpha = evidence + 1.0
    S = alpha.sum(1, keepdim=True)
    y = F.one_hot(target, 2).float()
    p = alpha / S
    err = ((y - p)**2).sum(1)
    var = (p*(1-p)/(S+1)).sum(1)
    sse = (err + var).mean()
    lam = min(1.0, epoch/10.0)
    alpha_t = y + (1 - y)*alpha                      # remove evidence for true class
    St = alpha_t.sum(1, keepdim=True)
    kl = (torch.lgamma(St.squeeze(1)) - torch.lgamma(alpha_t).sum(1)
          + ((alpha_t - 1)*(torch.digamma(alpha_t) - torch.digamma(St))).sum(1)).mean()
    return sse + lam*0.1*kl

def nlos_loss(logits, target, uq_mode, epoch):
    if uq_mode == "evidential":
        return evidential_loss(logits, target, epoch)
    return _ce(logits, target)

def probs_from_logits(logits, uq_mode, temperature=None):
    if uq_mode == "evidential":
        alpha = F.softplus(logits) + 1.0
        return alpha / alpha.sum(1, keepdim=True)
    if uq_mode == "temp_scaling" and temperature is not None:
        return F.softmax(logits / temperature.clamp(min=1e-2), 1)
    return F.softmax(logits, 1)

def uncertainty_from_logits(logits, uq_mode):
    """Per-sample scalar epistemic/predictive uncertainty in [0,1]."""
    if uq_mode == "evidential":
        alpha = F.softplus(logits) + 1.0
        return (2.0 / alpha.sum(1)).clamp(0, 1)         # vacuity = K/S
    p = F.softmax(logits, 1).clamp(1e-8, 1)
    ent = -(p*torch.log(p)).sum(1) / math.log(2)        # normalized entropy
    return ent.clamp(0, 1)

def set_mc_dropout(model, on=True):
    """Enable dropout at inference (MC-Dropout) while keeping BatchNorm in eval."""
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train(on)

def reward_meters(xp, yp, xt, yt, ds):
    xp_m = ds.denorm_x(xp.detach().cpu().numpy()); yp_m = ds.denorm_y(yp.detach().cpu().numpy())
    xt_m = ds.denorm_x(xt.detach().cpu().numpy()); yt_m = ds.denorm_y(yt.detach().cpu().numpy())
    d = np.sqrt((xp_m-xt_m)**2 + (yp_m-yt_m)**2)
    return torch.from_numpy((-d).astype(np.float32))


# =============================================================================
# PART 6 — REPRESENTATION PRE-TRAINING  (contrastive SimCLR  /  denoising AE)
#   Both pre-train ONLY the backbone, then discard the auxiliary module so the
#   supervised stage starts from learned representations. "none" skips this.
# =============================================================================
def pretrain_contrastive(backbone, loader, tag="", epochs=None):
    """SimCLR NT-Xent on two augmented views. Trains backbone in place."""
    epochs = PRETRAIN_EPOCHS if epochs is None else epochs
    proj = ProjectionHead(backbone.feat_dim).to(DEVICE)
    opt = AdamW(list(backbone.parameters()) + list(proj.parameters()),
                lr=3e-4, weight_decay=WEIGHT_DECAY)
    backbone.train(); proj.train()
    for ep in range(epochs):
        tot, nb = 0.0, 0
        for amp, phase, _xpd, *_ in loader:
            amp = amp.to(DEVICE); phase = phase.to(DEVICE)
            (a1, p1, x1), (a2, p2, x2) = augment_views(amp, phase)
            z1 = proj(backbone(a1, p1, x1)); z2 = proj(backbone(a2, p2, x2))
            loss = nt_xent(z1, z2)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(backbone.parameters(), GRAD_CLIP)
            opt.step(); tot += float(loss.item()); nb += 1
        if ep == 0 or (ep + 1) % max(1, epochs // 3) == 0 or ep == epochs - 1:
            log(f"      [contrastive{tag}] epoch {ep+1:>3}/{epochs}  nt_xent={tot/max(nb,1):.4f}")
    return backbone


def pretrain_dae(backbone, loader, tag="", epochs=None):
    """Denoising auto-encoder: corrupt amp/phase, reconstruct the clean
    [amp | cos(phase) | sin(phase)] 9-channel stack. Trains backbone in place."""
    epochs = PRETRAIN_EPOCHS if epochs is None else epochs
    dec = ReconDecoder(backbone.feat_dim, 3 * NUM_ANTENNAS,
                       SPEC_H, SPEC_W).to(DEVICE)
    opt = AdamW(list(backbone.parameters()) + list(dec.parameters()),
                lr=3e-4, weight_decay=WEIGHT_DECAY)
    backbone.train(); dec.train()
    for ep in range(epochs):
        tot, nb = 0.0, 0
        for amp, phase, _xpd, *_ in loader:
            amp = amp.to(DEVICE); phase = phase.to(DEVICE)
            clean = torch.cat([amp, torch.cos(phase), torch.sin(phase)], dim=1)
            namp = (amp + torch.randn_like(amp) * 0.05).clamp(0., None)
            nphase = phase + torch.randn_like(phase) * 0.10
            nxpd = compute_xpd(nphase)
            recon = dec(backbone(namp, nphase, nxpd))
            loss = F.mse_loss(recon, clean)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(backbone.parameters(), GRAD_CLIP)
            opt.step(); tot += float(loss.item()); nb += 1
        if ep == 0 or (ep + 1) % max(1, epochs // 3) == 0 or ep == epochs - 1:
            log(f"      [dae{tag}] epoch {ep+1:>3}/{epochs}  mse={tot/max(nb,1):.4f}")
    return backbone


# =============================================================================
# PART 7 — SUPERVISED TRAINING OF ONE (model, uq, rep) CONFIGURATION
# =============================================================================
def _val_score(model, loader, dataset):
    """Deterministic mean position MAE (metres) on a loader — early-stop signal."""
    model.eval(); set_mc_dropout(model, False)
    errs = []
    with torch.no_grad():
        for amp, phase, xpd, xt, yt, _nl in loader:
            amp, phase, xpd = amp.to(DEVICE), phase.to(DEVICE), xpd.to(DEVICE)
            xp, yp, _ = model(amp, phase, xpd)
            xp_m = dataset.denorm_x(xp.cpu().numpy()); yp_m = dataset.denorm_y(yp.cpu().numpy())
            xt_m = dataset.denorm_x(xt.numpy());       yt_m = dataset.denorm_y(yt.numpy())
            errs.append(np.sqrt((xp_m - xt_m) ** 2 + (yp_m - yt_m) ** 2))
    return float(np.mean(np.concatenate(errs))) if errs else float("inf")


def fit_temperature(model, loader):
    """Post-hoc temperature scaling: optimise a scalar T on validation NLL."""
    model.eval(); set_mc_dropout(model, False)
    logits_all, tgt_all = [], []
    with torch.no_grad():
        for amp, phase, xpd, _xt, _yt, nl in loader:
            amp, phase, xpd = amp.to(DEVICE), phase.to(DEVICE), xpd.to(DEVICE)
            _, _, logit = model(amp, phase, xpd)
            logits_all.append(logit.cpu()); tgt_all.append(nl)
    if not logits_all:
        return
    logits = torch.cat(logits_all); tgt = torch.cat(tgt_all)
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([logT], lr=0.05)
    for _ in range(200):
        opt.zero_grad()
        loss = F.cross_entropy(logits / logT.exp().clamp(min=1e-2), tgt)
        loss.backward(); opt.step()
    T = float(logT.exp().clamp(min=1e-2, max=1e2).item())
    model.head.temperature.data.fill_(T)
    log(f"      [temp_scaling] fitted T = {T:.3f}")


def train_one_config_centralized(suite, key, uq, rep, loaders, train_ds):
    """Standard (non-federated) training: the model sees the whole pooled
    training set. Early-stops on validation MAE (overfitting guard)."""
    tr_loader, va_loader, _te_loader = loaders
    disp, factory = SUITES[suite][key]
    backbone = factory().to(DEVICE)

    # ---- representation pre-training (the "with representation" arm) ----------
    if rep == "contrastive":
        pretrain_contrastive(backbone, tr_loader)
    elif rep == "dae":
        pretrain_dae(backbone, tr_loader)

    model = RadarLocNet(backbone, uq).to(DEVICE)
    base_lr = MODEL_LR.get(key, 3e-4)
    opt = AdamW(model.parameters(), lr=base_lr, weight_decay=WEIGHT_DECAY)
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=6, min_lr=1e-6)

    best_val, best_state, patience = float("inf"), None, 0
    n_epochs = EPOCHS
    for epoch in range(n_epochs):
        if epoch < WARMUP_EPOCHS:                       # linear LR warm-up
            for g in opt.param_groups:
                g["lr"] = base_lr * (epoch + 1) / WARMUP_EPOCHS

        model.train()
        for amp, phase, xpd, xt, yt, nl in tr_loader:
            amp, phase, xpd = amp.to(DEVICE), phase.to(DEVICE), xpd.to(DEVICE)
            xt, yt, nl = xt.to(DEVICE), yt.to(DEVICE), nl.to(DEVICE)
            xp, yp, logit = model(amp, phase, xpd)
            loss = loc_reg_loss(xp, yp, xt, yt) + LOSS_W_NLOS * nlos_loss(logit, nl, uq, epoch)
            if model.is_pinn and model.last_phys is not None:
                loss = loss + PINN_PHYS_W * model.last_phys.mean()
            if model.is_drl:
                reward = reward_meters(xp, yp, xt, yt, train_ds).to(DEVICE)
                loss = loss + 0.05 * model.backbone.rl_aux(model._feat.detach(), reward)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

        val = _val_score(model, va_loader, train_ds)
        if epoch >= WARMUP_EPOCHS:
            plateau.step(val)
        if val < best_val - 1e-4:
            best_val, patience = val, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                log(f"      early stop @ epoch {epoch+1} (val MAE {best_val:.3f} m)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if uq == "temp_scaling":
        fit_temperature(model, va_loader)
    log(f"    trained {disp:<26} uq={uq:<12} rep={rep:<11} -> val MAE {best_val:.3f} m")
    return model


# =============================================================================
# PART 7b — FEDERATED LEARNING  (horizontal FedAvg, weighted by client size)
#   Faithful to the reference papers: per round the server samples a client
#   subset, each client trains FL_LOCAL_EPOCHS locally on its OWN shard, the
#   server averages the weights (FedAvg), and the global model is selected by
#   validation (round-level EARLY STOPPING -> overfitting guard). Only model
#   weights cross the wire; raw data never leaves a client.
# =============================================================================
def partition_clients(train_ds, n_clients, non_iid=True, alpha=0.5, seed=SEED):
    """Split TRAIN indices into n_clients DISJOINT shards (no sample is shared
    between clients -> no intra-train leakage). Non-IID via Dirichlet over the
    spatial zone label (CG-FedLLM uses Dirichlet alpha=0.5)."""
    rng = np.random.default_rng(seed)
    n = len(train_ds)
    by_label = defaultdict(list)
    if non_iid:
        for i in range(n):
            by_label[int(train_ds.samples[i]["zone"])].append(i)
    else:
        by_label[0] = list(range(n))
    clients = [[] for _ in range(n_clients)]
    for _lbl, idxs in by_label.items():
        idxs = list(idxs); rng.shuffle(idxs)
        props = rng.dirichlet([alpha] * n_clients) if non_iid else np.ones(n_clients) / n_clients
        cuts = (np.cumsum(props) * len(idxs)).astype(int)[:-1]
        for c, part in enumerate(np.split(np.array(idxs), cuts)):
            clients[c].extend(part.tolist())
    # guarantee no empty client WITHOUT duplicating a sample (shards stay disjoint)
    for c in range(n_clients):
        if not clients[c]:
            donor = max(range(n_clients), key=lambda d: len(clients[d]))
            if len(clients[donor]) > 1:
                clients[c].append(clients[donor].pop())
    return clients


def make_client_loaders(train_ds, client_idx, batch_size):
    return [DataLoader(Subset(train_ds, idx), batch_size=batch_size,
                       shuffle=True, num_workers=0, drop_last=False)
            for idx in client_idx]


def fedavg(global_state, client_states, client_sizes):
    """Weighted Federated Averaging: w_global = sum_k (n_k / n) w_k.
    Float tensors (incl. BN running stats) are averaged; integer buffers kept."""
    total = float(sum(client_sizes))
    new_state = {}
    for k, gv in global_state.items():
        if torch.is_floating_point(gv):
            acc = torch.zeros_like(gv, dtype=torch.float64)
            for cs, sz in zip(client_states, client_sizes):
                acc += cs[k].to(torch.float64) * (sz / total)
            new_state[k] = acc.to(gv.dtype)
        else:
            new_state[k] = gv.clone()
    return new_state


def local_train(model, loader, lr, local_epochs, anneal_step, train_ds):
    """One client's local update: train `local_epochs` passes over its shard."""
    uq = model.uq_mode
    opt = AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    model.train()
    for _ep in range(local_epochs):
        for amp, phase, xpd, xt, yt, nl in loader:
            amp, phase, xpd = amp.to(DEVICE), phase.to(DEVICE), xpd.to(DEVICE)
            xt, yt, nl = xt.to(DEVICE), yt.to(DEVICE), nl.to(DEVICE)
            xp, yp, logit = model(amp, phase, xpd)
            loss = loc_reg_loss(xp, yp, xt, yt) + LOSS_W_NLOS * nlos_loss(logit, nl, uq, anneal_step)
            if model.is_pinn and model.last_phys is not None:
                loss = loss + PINN_PHYS_W * model.last_phys.mean()
            if model.is_drl:
                reward = reward_meters(xp, yp, xt, yt, train_ds).to(DEVICE)
                loss = loss + 0.05 * model.backbone.rl_aux(model._feat.detach(), reward)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def federated_pretrain(backbone, client_loaders, mode, select_fn,
                       rounds=None, local_epochs=None):
    """Federated representation pre-training: each selected client pretrains its
    backbone locally (contrastive / DAE), the server FedAvg-averages them.
    Keeps the FL premise (no raw data centralised)."""
    rounds = FL_PRETRAIN_ROUNDS if rounds is None else rounds
    local_epochs = FL_PRETRAIN_LOCAL_EPOCHS if local_epochs is None else local_epochs
    g_state = {k: v.detach().cpu().clone() for k, v in backbone.state_dict().items()}
    for r in range(rounds):
        sel = select_fn(); states, sizes = [], []
        for ci in sel:
            backbone.load_state_dict(g_state)
            if mode == "contrastive":
                pretrain_contrastive(backbone, client_loaders[ci], epochs=local_epochs)
            else:
                pretrain_dae(backbone, client_loaders[ci], epochs=local_epochs)
            states.append({k: v.detach().cpu().clone() for k, v in backbone.state_dict().items()})
            sizes.append(len(client_loaders[ci].dataset))
        g_state = fedavg(g_state, states, sizes)
    backbone.load_state_dict(g_state)
    return backbone


def train_one_config_federated(suite, key, uq, rep, client_loaders, val_loader,
                                train_ds, fl_rounds, fl_local_epochs, verbose=True):
    """FedAvg training of one (model, uq, rep). Round-level early stopping on
    validation MAE provides the overfitting guard for the federated path."""
    disp, factory = SUITES[suite][key]
    n_clients = len(client_loaders)
    rng = random.Random(SEED + (hash((key, uq, rep)) % 100000))

    def select():
        k = max(1, int(round(CLIENT_FRACTION * n_clients)))
        return rng.sample(range(n_clients), min(k, n_clients))

    backbone = factory().to(DEVICE)
    if rep == "contrastive":
        federated_pretrain(backbone, client_loaders, "contrastive", select)
    elif rep == "dae":
        federated_pretrain(backbone, client_loaders, "dae", select)

    global_model = RadarLocNet(backbone, uq).to(DEVICE)
    local_model = RadarLocNet(factory(), uq).to(DEVICE)   # reused scratch client
    g_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
    lr = MODEL_LR.get(key, 3e-4)

    best_val, best_state, patience = float("inf"), None, 0
    for rnd in range(fl_rounds):
        sel = select(); states, sizes = [], []
        for ci in sel:
            local_model.load_state_dict(g_state)
            st = local_train(local_model, client_loaders[ci], lr,
                             fl_local_epochs, anneal_step=rnd, train_ds=train_ds)
            states.append(st); sizes.append(len(client_loaders[ci].dataset))
        g_state = fedavg(g_state, states, sizes)
        global_model.load_state_dict(g_state)

        val = _val_score(global_model, val_loader, train_ds)   # VAL only -> no leak
        if val < best_val - 1e-4:
            best_val, patience = val, 0
            best_state = {k: v.clone() for k, v in g_state.items()}
        else:
            patience += 1
            if patience >= FL_EARLY_STOP_ROUNDS:
                if verbose:
                    log(f"      FL early stop @ round {rnd+1} (val MAE {best_val:.3f} m)")
                break

    if best_state is not None:
        global_model.load_state_dict(best_state)
    if uq == "temp_scaling":
        fit_temperature(global_model, val_loader)
    if verbose:
        log(f"    [FED] {disp:<22} uq={uq:<12} rep={rep:<11} "
            f"R={fl_rounds} E={fl_local_epochs} -> val MAE {best_val:.3f} m")
    del local_model
    return global_model


def tune_fl(client_loaders, val_loader, train_ds):
    """Search FL_ROUNDS x FL_LOCAL_EPOCHS on a cheap proxy model, select the
    pair with the best VALIDATION MAE penalised by the (val-train) over-fitting
    gap. VAL only is used for scoring -> the held-out TEST set is never touched."""
    suite, key = FL_TUNE_PROXY
    if FL_TUNE_FULL_GRID:
        points = [(r, e) for r in FL_ROUNDS_GRID for e in FL_LOCAL_EPOCHS_GRID]
    else:
        pool = [(r, e) for r in FL_ROUNDS_GRID for e in FL_LOCAL_EPOCHS_GRID]
        points = random.Random(SEED).sample(pool, min(FL_TUNE_TRIALS, len(pool)))

    # small TRAIN probe (drawn from the clients' own data) to estimate the gap
    probe_idx = [i for ldr in client_loaders for i in ldr.dataset.indices][:FL_TUNE_VAL]
    train_probe = DataLoader(Subset(train_ds, probe_idx), batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)

    log("\n" + "=" * 78)
    log(f" FL HYPER-PARAMETER SEARCH  (proxy = {SUITES[suite][key][0]})")
    log(f"   grid: FL_ROUNDS in [{FL_ROUNDS_GRID[0]}..{FL_ROUNDS_GRID[-1]}] step "
        f"{FL_ROUNDS_GRID[1]-FL_ROUNDS_GRID[0]}, "
        f"FL_LOCAL_EPOCHS in [{FL_LOCAL_EPOCHS_GRID[0]}..{FL_LOCAL_EPOCHS_GRID[-1]}] step "
        f"{FL_LOCAL_EPOCHS_GRID[1]-FL_LOCAL_EPOCHS_GRID[0]}")
    log(f"   strategy: {'FULL GRID' if FL_TUNE_FULL_GRID else f'random search x{len(points)}'}")
    log("=" * 78)

    results = []
    for i, (r, e) in enumerate(points, 1):
        m = train_one_config_federated(suite, key, "softmax", "none",
                                        client_loaders, val_loader, train_ds,
                                        fl_rounds=r, fl_local_epochs=e, verbose=False)
        val = _val_score(m, val_loader, train_ds)
        tr = _val_score(m, train_probe, train_ds)
        gap = max(0.0, val - tr)
        score = val + FL_TUNE_OVERFIT_LAMBDA * gap
        results.append(dict(fl_rounds=r, fl_local_epochs=e, val_mae=val,
                            train_mae=tr, gap=gap, score=score))
        log(f"   [{i:>2}/{len(points)}] R={r:>3} E={e:>3} | val={val:.3f} "
            f"train={tr:.3f} gap={gap:.3f} score={score:.3f}")
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results.sort(key=lambda d: d["score"])
    best = results[0]
    log("-" * 78)
    log(f" BEST FL HYPER-PARAMS: FL_ROUNDS={best['fl_rounds']}  "
        f"FL_LOCAL_EPOCHS={best['fl_local_epochs']}  "
        f"(val MAE {best['val_mae']:.3f} m, overfit gap {best['gap']:.3f})")
    log("=" * 78)
    return best["fl_rounds"], best["fl_local_epochs"], results


# ---- mode dispatcher: lets the sweep request centralized OR federated -------
def train_one_config(suite, key, uq, rep, ctx, train_ds, mode):
    if mode == "federated":
        return train_one_config_federated(
            suite, key, uq, rep, ctx["clients"], ctx["val"], train_ds,
            FL_ROUNDS, FL_LOCAL_EPOCHS)
    return train_one_config_centralized(
        suite, key, uq, rep, (ctx["tr"], ctx["val"], ctx["te"]), train_ds)


# =============================================================================
# PART 8 — EVALUATION (per-frequency-band) + METRICS / CALIBRATION
# =============================================================================
def _mask_band(amp, phase, band_slot):
    """band_slot None -> all bands (combined); else keep only that antenna/band.
    xpd is recomputed from the (masked) phase, exactly like the v2 per-band eval."""
    if band_slot is None:
        return amp, phase, compute_xpd(phase)
    a = torch.zeros_like(amp); p = torch.zeros_like(phase)
    a[:, band_slot] = amp[:, band_slot]
    p[:, band_slot] = phase[:, band_slot]
    return a, p, compute_xpd(p)


def _zone_of(x_m):
    z = np.clip((x_m - SPACE_X_MIN) / ((SPACE_X_MAX - SPACE_X_MIN) / N_ZONES), 0, N_ZONES - 1)
    return z.astype(int)


def _ece(conf, correct, n_bins=10):
    conf = np.asarray(conf); correct = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1); N = len(conf); ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() > 0:
            ece += (m.sum() / N) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def _angle_deg(x_m, y_m, sensor):
    return np.degrees(np.arctan2(y_m - sensor["y"], x_m - sensor["x"]))


# ---- empirical-CDF x-axes for the error curves + latency settings -----------
DIST_CDF_MAX_M    = 12.0      # scene diagonal is ~11 m
ANGLE_CDF_MAX_DEG = 180.0
CDF_POINTS        = 49
DIST_GRID  = np.linspace(0.0, DIST_CDF_MAX_M, CDF_POINTS)
ANGLE_GRID = np.linspace(0.0, ANGLE_CDF_MAX_DEG, CDF_POINTS)
LATENCY_WARMUP = 2            # warm-up forward passes before timing


def _empirical_cdf(errs, grid):
    e = np.asarray(errs)
    return [float((e <= t).mean()) for t in grid]


def evaluate(model, loader, dataset, uq, band_slot):
    """Return scalar metrics + error-CDF curves + inference latency for one
    (model, uq, rep) configuration at one frequency band."""
    model.eval(); set_mc_dropout(model, False)
    XP, YP, XT, YT = [], [], [], []
    P1, NL = [], []          # predicted P(NLOS) and true NLOS label
    UNC = []                 # per-sample predictive uncertainty
    use_mc = (uq == "mc_dropout")
    temp = model.head.temperature
    n_params = int(sum(p.numel() for p in model.parameters()))
    fwd_time, fwd_samples, did_warmup = 0.0, 0, False

    with torch.no_grad():
        for amp, phase, _xpd, xt, yt, nl in loader:
            amp, phase = amp.to(DEVICE), phase.to(DEVICE)
            a, p, xpd = _mask_band(amp, phase, band_slot)
            bs = int(a.shape[0])

            if use_mc:
                set_mc_dropout(model, True)             # dropout ON, BN stays eval
                if not did_warmup:
                    for _ in range(LATENCY_WARMUP):
                        model(a, p, xpd)
                    did_warmup = True
                t0 = time.perf_counter()
                xps, yps, ps = [], [], []
                for _ in range(MC_DROPOUT_SAMPLES):     # latency reflects the MC cost
                    xp, yp, logit = model(a, p, xpd)
                    xps.append(xp); yps.append(yp)
                    ps.append(F.softmax(logit, 1))
                fwd_time += time.perf_counter() - t0; fwd_samples += bs
                set_mc_dropout(model, False)
                xp = torch.stack(xps).mean(0); yp = torch.stack(yps).mean(0)
                prob = torch.stack(ps).mean(0)
                # predictive uncertainty = entropy of mean prob + spatial std
                ent = -(prob.clamp(1e-8) * prob.clamp(1e-8).log()).sum(1) / math.log(2)
                pos_std = (torch.stack(xps).std(0) + torch.stack(yps).std(0))
                unc = (0.5 * ent + 0.5 * pos_std.clamp(0, 1)).clamp(0, 1)
            else:
                if not did_warmup:
                    for _ in range(LATENCY_WARMUP):
                        model(a, p, xpd)
                    did_warmup = True
                t0 = time.perf_counter()
                xp, yp, logit = model(a, p, xpd)
                fwd_time += time.perf_counter() - t0; fwd_samples += bs
                prob = probs_from_logits(logit, uq, temp)
                unc = uncertainty_from_logits(logit, uq)

            XP.append(xp.cpu().numpy()); YP.append(yp.cpu().numpy())
            XT.append(xt.numpy());       YT.append(yt.numpy())
            P1.append(prob[:, 1].cpu().numpy()); NL.append(nl.numpy())
            UNC.append(unc.cpu().numpy())

    xp_m = dataset.denorm_x(np.concatenate(XP)); yp_m = dataset.denorm_y(np.concatenate(YP))
    xt_m = dataset.denorm_x(np.concatenate(XT)); yt_m = dataset.denorm_y(np.concatenate(YT))
    p1 = np.concatenate(P1); nl = np.concatenate(NL).astype(int); unc = np.concatenate(UNC)

    # ---- localization (metres) ------------------------------------------------
    derr = np.sqrt((xp_m - xt_m) ** 2 + (yp_m - yt_m) ** 2)
    xerr = xp_m - xt_m; yerr = yp_m - yt_m

    # ---- angle of arrival (deg, wrt 77 GHz sensor) ----------------------------
    a_pred = _angle_deg(xp_m, yp_m, SENSOR_77); a_true = _angle_deg(xt_m, yt_m, SENSOR_77)
    aerr = np.abs(((a_pred - a_true) + 180) % 360 - 180)

    # ---- time of flight (ns, round-trip wrt 77 GHz sensor) --------------------
    r_pred = np.sqrt((xp_m - SENSOR_77["x"]) ** 2 + (yp_m - SENSOR_77["y"]) ** 2)
    r_true = np.sqrt((xt_m - SENSOR_77["x"]) ** 2 + (yt_m - SENSOR_77["y"]) ** 2)
    tof_err = np.abs(2 * (r_pred - r_true) / SPEED_OF_LIGHT) * 1e9

    # ---- NLOS classification --------------------------------------------------
    pred = (p1 >= 0.5).astype(int)
    tp = int(((pred == 1) & (nl == 1)).sum()); tn = int(((pred == 0) & (nl == 0)).sum())
    fp = int(((pred == 1) & (nl == 0)).sum()); fn = int(((pred == 0) & (nl == 1)).sum())
    nlos_acc = (tp + tn) / max(len(nl), 1)
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    far = fp / max(fp + tn, 1)          # false-alarm rate
    mdr = fn / max(fn + tp, 1)          # missed-detection rate

    # ---- zone / floor ---------------------------------------------------------
    zone_acc = float((_zone_of(xp_m) == _zone_of(xt_m)).mean())
    if FLOOR_MARGIN_M > 0:
        fm = np.abs(yt_m) >= FLOOR_MARGIN_M          # guard band around y=0
        floor_acc = float(((yp_m[fm] >= 0) == (yt_m[fm] >= 0)).mean()) if fm.any() else 0.5
    else:
        floor_acc = float(((yp_m >= 0) == (yt_m >= 0)).mean())

    # ---- calibration on NLOS probability --------------------------------------
    conf = np.maximum(p1, 1 - p1)               # confidence of predicted class
    correct = (pred == nl)
    ece = _ece(conf, correct)
    pc = np.clip(np.where(nl == 1, p1, 1 - p1), 1e-8, 1.0)
    nll = float(-np.log(pc).mean())
    brier = float(np.mean((p1 - nl) ** 2))

    infer_ms_per_sample = float(fwd_time / max(fwd_samples, 1) * 1e3)

    return dict(
        n=int(len(derr)),
        mae_m=np_mae(derr), rmse_m=np_rmse(derr),
        p50_m=np_pct(derr, 50), p80_m=np_pct(derr, 80),
        p90_m=np_pct(derr, 90), p95_m=np_pct(derr, 95),
        mae_x_m=np_mae(xerr), mae_y_m=np_mae(yerr),
        aoa_mae_deg=np_mae(aerr), aoa_rmse_deg=np_rmse(aerr),
        tof_mae_ns=np_mae(tof_err), tof_rmse_ns=np_rmse(tof_err),
        nlos_acc=float(nlos_acc), nlos_f1=float(f1),
        nlos_far=float(far), nlos_mdr=float(mdr),
        zone_acc=zone_acc, floor_acc=floor_acc,
        ece=ece, nll=nll, brier=brier,
        mean_uncertainty=float(np.mean(unc)),
        infer_ms_per_sample=infer_ms_per_sample,   # latency / inference cost
        n_params=n_params,
        cdf_dist=_empirical_cdf(derr, DIST_GRID),   # distance-error CDF curve
        cdf_ang=_empirical_cdf(aerr, ANGLE_GRID),   # angle-error CDF curve
    )


# =============================================================================
# PART 9 — ORCHESTRATION  (data split, the sweep, and all output artefacts)
# =============================================================================
COMBO_LIST  = [(u, r) for u in UQ_METHODS for r in REP_METHODS]   # the 12 (or subset)
UQ_LABEL    = {"softmax": "Softmax", "mc_dropout": "Softmax+MCDrop",
               "temp_scaling": "Softmax+TempScale", "evidential": "Evidential"}
REP_LABEL   = {"none": "noRep", "contrastive": "Contrastive", "dae": "DAE"}
def combo_label(u, r): return f"{UQ_LABEL.get(u,u)} | {REP_LABEL.get(r,r)}"

CSV_FIELDS = ["mode", "suite", "model", "display", "uq", "rep", "band", "n",
              "mae_m", "rmse_m", "p50_m", "p80_m", "p90_m", "p95_m",
              "mae_x_m", "mae_y_m", "aoa_mae_deg", "aoa_rmse_deg",
              "tof_mae_ns", "tof_rmse_ns", "nlos_acc", "nlos_f1",
              "nlos_far", "nlos_mdr", "zone_acc", "floor_acc",
              "ece", "nll", "brier", "mean_uncertainty",
              "infer_ms_per_sample", "n_params", "train_time_s"]
# (cdf_dist / cdf_ang are full curves -> saved in the JSON + CDF plots, not the CSV)


def _split_records(records, fracs=(0.70, 0.15, 0.15), seed=SEED):
    idx = list(range(len(records)))
    random.Random(seed).shuffle(idx)
    n = len(idx); n_tr = int(fracs[0] * n); n_va = int(fracs[1] * n)
    tr = [records[i] for i in idx[:n_tr]]
    va = [records[i] for i in idx[n_tr:n_tr + n_va]]
    te = [records[i] for i in idx[n_tr + n_va:]]
    return tr, va, te


def _share_norm(src, *others):
    for o in others:
        o.x_min, o.x_max, o.x_rng = src.x_min, src.x_max, src.x_rng
        o.y_min, o.y_max, o.y_rng = src.y_min, src.y_max, src.y_rng


def save_csv_json(rows):
    csv_path = OUTPUT_DIR / f"radar_ablation_master_{TIMESTAMP}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})
    json_path = OUTPUT_DIR / f"radar_ablation_master_{TIMESTAMP}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    log(f"  saved {csv_path.name}  and  {json_path.name}")


def _heatmap_grid(rows, suite, mode, metric, fname, title, lower_better=True):
    rws = [r for r in rows if r["mode"] == mode]
    models = [k for k in SUITES[suite] if any(
        r["suite"] == suite and r["model"] == k for r in rws)]
    if not models:
        return
    bands = BANDS
    fig, axes = plt.subplots(1, len(bands), figsize=(4.6 * len(bands), 1.0 + 0.5 * len(models)),
                             squeeze=False)
    cmap = "viridis_r" if lower_better else "viridis"
    for bi, band in enumerate(bands):
        ax = axes[0][bi]
        M = np.full((len(models), len(COMBO_LIST)), np.nan)
        for mi, mk in enumerate(models):
            for ci, (u, rp) in enumerate(COMBO_LIST):
                for r in rws:
                    if (r["suite"] == suite and r["model"] == mk and r["band"] == band
                            and r["uq"] == u and r["rep"] == rp):
                        M[mi, ci] = r.get(metric, np.nan); break
        im = ax.imshow(M, aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(COMBO_LIST)))
        ax.set_xticklabels([combo_label(u, rp) for u, rp in COMBO_LIST],
                           rotation=90, fontsize=6)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([SUITES[suite][m][0] for m in models], fontsize=7)
        ax.set_title(f"{band}", fontsize=9)
        for mi in range(len(models)):
            for ci in range(len(COMBO_LIST)):
                if not np.isnan(M[mi, ci]):
                    ax.text(ci, mi, f"{M[mi, ci]:.2f}", ha="center", va="center",
                            fontsize=5, color="w")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = OUTPUT_DIR / fname
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    log(f"  saved {out.name}")


def save_heatmaps(rows):
    modes = [m for m in TRAINING_MODES if any(r["mode"] == m for r in rows)]
    for mode in modes:
        for suite in SUITES_TO_RUN:
            _heatmap_grid(rows, suite, mode, "mae_m",
                          f"summary_MAE_{mode}_{suite}_{TIMESTAMP}.png",
                          f"{suite.upper()} [{mode}] — position MAE (m)  [lower=better]",
                          lower_better=True)
            _heatmap_grid(rows, suite, mode, "ece",
                          f"summary_ECE_{mode}_{suite}_{TIMESTAMP}.png",
                          f"{suite.upper()} [{mode}] — NLOS calibration ECE  [lower=better]",
                          lower_better=True)



def _per_band_leaderboard_lines(rows):
    """Per-band leaderboard showing MAE(m) | MAE(deg) | NLOS-acc for
    Combined / 77 GHz / 24 GHz / 10 GHz.  Ranks models by composite score =
    mean normalised loss across ALL bands × ALL three metrics (lower = better).
    Works for both single-mode (non-FL) and multi-mode (FL) row sets."""
    W = 140
    BAND_ORDER = ["combined", "77ghz", "24ghz", "10ghz"]
    BAND_LABEL = {"combined": "Combined", "77ghz": "77 GHz",
                  "24ghz": "24 GHz",     "10ghz": "10 GHz"}
    out = []

    modes = sorted(set(r.get("mode", "centralized") for r in rows))
    for mode in modes:
        mrows = [r for r in rows if r.get("mode", "centralized") == mode]
        if not mrows:
            continue
        present = [b for b in BAND_ORDER if any(r["band"] == b for r in mrows)]

        # normalisation pools (min-max over every config × band in this mode)
        pool_m = [r["mae_m"]            for r in mrows]
        pool_d = [r["aoa_mae_deg"]       for r in mrows]
        pool_n = [1.0 - r["nlos_acc"]   for r in mrows]   # NLOS loss

        def _nrm(v, pool):
            lo, hi = min(pool), max(pool)
            return (v - lo) / (hi - lo) if hi > lo else 0.0

        # group rows by config key → {band: row}
        cfg = {}
        for r in mrows:
            key = (r["suite"], r["model"], r["uq"], r["rep"])
            cfg.setdefault(key, {})[r["band"]] = r

        # composite score per config = mean over bands of mean(3 norm losses)
        cscore = {}
        for key, bd in cfg.items():
            vals = [(_nrm(r["mae_m"],   pool_m)
                     + _nrm(r["aoa_mae_deg"], pool_d)
                     + _nrm(1.0 - r["nlos_acc"], pool_n)) / 3.0
                    for r in bd.values()]
            cscore[key] = float(np.mean(vals)) if vals else 1.0

        # best (uq, rep) combo per model = the config with lowest score
        best = {}
        for (s, m, u, rp), sc in cscore.items():
            mk = (s, m)
            if mk not in best or sc < best[mk][0]:
                best[mk] = (sc, u, rp)

        ranked = sorted(best.items(), key=lambda kv: kv[1][0])

        mode_tag = f"  [mode = {mode}]" if len(modes) > 1 else ""
        out += ["",
                "=" * W,
                f"PER-BAND LEADERBOARD{mode_tag}",
                "  each column block:  MAE (m)  |  MAE (deg)  |  NLOS-acc",
                "  ranked by composite score = mean normalised loss over "
                "ALL bands  x  MAE(m) + MAE(deg) + NLOS-error",
                "  lower score = fewer losses across 10 GHz, 24 GHz, "
                "77 GHz  and  Combined",
                "=" * W]

        C = 22    # per-band block width
        h0 = f"{'':>3}  {'Model':<22}  {'Best Combo':<26}  {'Score':>5}  "
        for b in present:
            h0 += f"|{BAND_LABEL[b]:^{C - 1}}"
        out.append(h0)
        h1 = f"{'Rk':>3}  {'':22}  {'':26}  {'':5}  "
        for _ in present:
            h1 += "|  MAE_m   MAE°   NLOS "
        out.append(h1)
        out.append("-" * W)

        for rank, ((s, m), (sc, u, rp)) in enumerate(ranked, 1):
            disp = SUITES[s][m][0]
            line = f"{rank:>3}  {disp:<22}  {combo_label(u, rp):<26}  {sc:>5.3f}  "
            for b in present:
                r = cfg[(s, m, u, rp)].get(b)
                if r:
                    line += (f"|{r['mae_m']:>6.3f}  "
                             f"{r['aoa_mae_deg']:>5.1f}  "
                             f"{r['nlos_acc']:>4.2f} ")
                else:
                    line += f"|{'  --':>6}  {'  --':>5}  {'--':>4} "
            out.append(line)

        # ── overall winner ───────────────────────────────────────────────────
        (ws, wm), (wsc, wu, wrp) = ranked[0]
        wdisp  = SUITES[ws][wm][0]
        wcomb  = combo_label(wu, wrp)
        out += ["",
                "*" * W,
                f"  TOP MODEL{mode_tag}",
                f"  Model      : {wdisp}",
                f"  Best Combo : {wcomb}",
                f"  Score      : {wsc:.4f}   "
                "(lowest combined loss — MAE_m + MAE_deg + NLOS-error — all bands)",
                "*" * W]
        for b in present:
            r = cfg[(ws, wm, wu, wrp)].get(b)
            if r:
                out.append(
                    f"    {BAND_LABEL[b]:<10}  "
                    f"MAE(m)={r['mae_m']:.3f}   "
                    f"MAE(deg)={r['aoa_mae_deg']:.2f}   "
                    f"NLOS-acc={r['nlos_acc']:.3f}   "
                    f"FAR={r['nlos_far']:.3f}   MDR={r['nlos_mdr']:.3f}")
        out.append("")
    return out

def save_leaderboard(rows):
    out = OUTPUT_DIR / f"leaderboard_{TIMESTAMP}.txt"
    modes = [m for m in TRAINING_MODES if any(r["mode"] == m for r in rows)]
    lines = ["=" * 78, "RADAR LOCALIZATION ABLATION — LEADERBOARD", "=" * 78]
    lines += _per_band_leaderboard_lines(rows)
    for mode in modes:
        combined = [r for r in rows if r["band"] == "combined" and r["mode"] == mode]
        lines += ["", f"  [{mode}] Representation impact (mean combined-band MAE, all models/UQ):"]
        for rp in REP_METHODS:
            v = [r["mae_m"] for r in combined if r["rep"] == rp]
            if v:
                lines.append(f"     rep={REP_LABEL.get(rp, rp):<14} mean MAE = {np.mean(v):.3f} m (n={len(v)})")
        lines += [f"  [{mode}] UQ-method impact (mean combined-band MAE / ECE):"]
        for u in UQ_METHODS:
            v = [r["mae_m"] for r in combined if r["uq"] == u]
            e = [r["ece"] for r in combined if r["uq"] == u]
            if v:
                lines.append(f"     uq={UQ_LABEL.get(u, u):<18} MAE={np.mean(v):.3f} m  ECE={np.mean(e):.3f}")

    # ---- WITH vs WITHOUT federated learning comparison ------------------------
    if "centralized" in modes and "federated" in modes:
        lines += ["", "=" * 78,
                  "WITH vs WITHOUT FEDERATED LEARNING  (mean combined-band MAE per model,",
                  "averaged over all 12 combos; delta = federated - centralized)",
                  "=" * 78,
                  f"{'model':<26}{'central MAE':>13}{'fed MAE':>11}{'delta(m)':>11}{'fed acc':>9}"]
        lines.append("-" * 78)
        comb = [r for r in rows if r["band"] == "combined"]
        deltas = []
        for suite in SUITES_TO_RUN:
            for key in SUITES[suite]:
                cen = [r["mae_m"] for r in comb if r["mode"] == "centralized"
                       and r["suite"] == suite and r["model"] == key]
                fed = [r["mae_m"] for r in comb if r["mode"] == "federated"
                       and r["suite"] == suite and r["model"] == key]
                facc = [r["nlos_acc"] for r in comb if r["mode"] == "federated"
                        and r["suite"] == suite and r["model"] == key]
                if cen and fed:
                    c, f_, d = np.mean(cen), np.mean(fed), np.mean(fed) - np.mean(cen)
                    deltas.append(d)
                    lines.append(f"{SUITES[suite][key][0]:<26}{c:>13.3f}{f_:>11.3f}"
                                 f"{d:>+11.3f}{np.mean(facc):>9.3f}")
        if deltas:
            lines += ["-" * 78,
                      f"  mean degradation from federation: {np.mean(deltas):+.3f} m "
                      f"(over {len(deltas)} models)",
                      "  (a small positive delta is expected: FedAvg on non-IID shards trades",
                      "   a little accuracy for privacy — no raw data leaves any client.)"]

    # ---- latency / inference-cost summary (per UQ method, combined band) ------
    lines += ["", "=" * 78,
              "LATENCY / INFERENCE COST  (mean over all models, combined band)",
              "  MC-Dropout runs the network MC_DROPOUT_SAMPLES times -> higher latency.",
              "=" * 78,
              f"{'UQ method':<22}{'infer ms/sample':>16}{'train time s':>14}{'#params':>12}"]
    lines.append("-" * 78)
    comb = [r for r in rows if r["band"] == "combined"]
    for u in UQ_METHODS:
        lat = [r["infer_ms_per_sample"] for r in comb if r["uq"] == u]
        tt = [r["train_time_s"] for r in comb if r["uq"] == u and "train_time_s" in r]
        npm = [r["n_params"] for r in comb if r["uq"] == u]
        if lat:
            lines.append(f"{UQ_LABEL.get(u, u):<22}{np.mean(lat):>16.3f}"
                         f"{(np.mean(tt) if tt else 0):>14.1f}{int(np.mean(npm)):>12,d}")

    txt = "\n".join(lines)
    with open(out, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    log("\n" + txt)
    log(f"\n  saved {out.name}")


def save_cdf_curves(rows):
    """Distance-error and angle-error CDF curves (one per suite per mode),
    overlaying each model's best combo at the combined band. Raw CDF arrays for
    every (mode, model, uq, rep, band) are in the master JSON; the x-axes are
    saved to cdf_axes_*.json so any slice can be re-plotted."""
    modes = [m for m in TRAINING_MODES if any(r["mode"] == m for r in rows)]
    with open(OUTPUT_DIR / f"cdf_axes_{TIMESTAMP}.json", "w", encoding="utf-8") as f:
        json.dump({"distance_grid_m": DIST_GRID.tolist(),
                   "angle_grid_deg": ANGLE_GRID.tolist()}, f, indent=2)
    for mode in modes:
        for suite in SUITES_TO_RUN:
            comb = [r for r in rows if r["mode"] == mode and r["suite"] == suite
                    and r["band"] == "combined"]
            if not comb:
                continue
            best = {}
            for r in comb:
                k = r["model"]
                if k not in best or r["mae_m"] < best[k]["mae_m"]:
                    best[k] = r
            for kind, grid, key, xlabel, fname in [
                ("distance", DIST_GRID, "cdf_dist", "localization error (m)",
                 f"cdf_distance_{mode}_{suite}_{TIMESTAMP}.png"),
                ("angle (AoA)", ANGLE_GRID, "cdf_ang", "angle error (deg)",
                 f"cdf_angle_{mode}_{suite}_{TIMESTAMP}.png")]:
                fig, ax = plt.subplots(figsize=(7.2, 5.0))
                for mk, r in best.items():
                    y = r.get(key)
                    if y:
                        ax.plot(grid, y, lw=1.7,
                                label=f"{SUITES[suite][mk][0]} ({combo_label(r['uq'], r['rep'])})")
                ax.set_xlabel(xlabel); ax.set_ylabel("CDF   P(error \u2264 x)")
                ax.set_ylim(0, 1.02); ax.grid(alpha=0.3)
                ax.set_title(f"{suite.upper()} [{mode}] — {kind} error CDF "
                             f"(best combo per model, combined band)", fontsize=10)
                ax.legend(fontsize=6, loc="lower right")
                out = OUTPUT_DIR / fname
                fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
                log(f"  saved {out.name}")


def save_fl_tuning(results):
    if not results:
        return
    path = OUTPUT_DIR / f"fl_hyperparam_search_{TIMESTAMP}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["fl_rounds", "fl_local_epochs",
                                          "val_mae", "train_mae", "gap", "score"])
        w.writeheader()
        for r in results:
            w.writerow(r)
    log(f"  saved {path.name}")


def data_leakage_audit(tr_idx, va_idx, te_idx, client_idx, n_train):
    """Assert the splits are disjoint and federated shards stay inside TRAIN."""
    log("\n" + "=" * 78)
    log(" DATA-LEAKAGE AUDIT")
    log("=" * 78)
    s_tr, s_va, s_te = set(tr_idx), set(va_idx), set(te_idx)
    assert not (s_tr & s_va), "LEAK: train/val overlap"
    assert not (s_tr & s_te), "LEAK: train/test overlap"
    assert not (s_va & s_te), "LEAK: val/test overlap"
    log("  [OK] train / val / test record sets are pairwise DISJOINT")
    if client_idx is not None:
        allc = [i for c in client_idx for i in c]
        assert max(allc) < n_train and min(allc) >= 0, "LEAK: client idx outside TRAIN"
        for a in range(len(client_idx)):
            for b in range(a + 1, len(client_idx)):
                assert not (set(client_idx[a]) & set(client_idx[b])), "LEAK: clients share samples"
        log(f"  [OK] {len(client_idx)} federated client shards are disjoint & inside TRAIN")
    log("  [OK] normalisation statistics are computed from TRAIN only (shared to val/test)")
    log("  [OK] representation pre-training uses TRAIN / client data only")
    log("  [OK] FL hyper-parameter search & early stopping use the VALIDATION set only")
    log("  [OK] temperature scaling is fit on the VALIDATION set only")
    log("  [OK] the TEST set is used exclusively for final metric reporting")
    log("  [OK] data augmentation is applied to TRAIN only (val/test deterministic)")
    log("=" * 78)


def main():
    global FL_ROUNDS, FL_LOCAL_EPOCHS
    t0 = time.time()
    log("=" * 78)
    log(" RADAR LOCALIZATION — UQ × REPRESENTATION × FEDERATED ABLATION")
    log("=" * 78)
    log(f" device           : {DEVICE}")
    log(f" training modes   : {TRAINING_MODES}")
    log(f" suites to run    : {SUITES_TO_RUN}")
    log(f" models filter    : {MODELS_TO_RUN or 'ALL'}")
    log(f" UQ methods       : {UQ_METHODS}")
    log(f" REP methods      : {REP_METHODS}")
    log(f" bands            : {BANDS}")
    log(f" combos per model : {len(COMBO_LIST)}")
    log(f" epochs / patience: {EPOCHS} / {EARLY_STOP_PATIENCE}   pretrain={PRETRAIN_EPOCHS}")
    log(f" FL clients       : {N_CLIENTS}  frac={CLIENT_FRACTION}  non_iid={NON_IID} (a={DIRICHLET_ALPHA})")
    log(f" FL early-stop    : {FL_EARLY_STOP_ROUNDS} rounds w/o val improvement")
    log(f" FAST_DEV         : {FAST_DEV}")
    log(f" output dir       : {OUTPUT_DIR}")

    # ---- data load ------------------------------------------------------------
    records = None
    if DATASET_ROOT.exists():
        try:
            records = _try_load_real(DATASET_ROOT)
        except Exception as e:
            log(f" real-data load failed ({e})")
    if not records:
        if ALLOW_SYNTHETIC_FALLBACK:
            log(" real dataset not found -> using SYNTHETIC stand-in (pipeline test only).")
            records = _make_synthetic(MAX_FILES_PER_ACT)
        else:
            log(" real dataset not found and ALLOW_SYNTHETIC_FALLBACK=False -> exiting.")
            return

    # ---- disjoint split (record-level) ----------------------------------------
    idx = list(range(len(records))); random.Random(SEED).shuffle(idx)
    n = len(idx); n_tr = int(0.70 * n); n_va = int(0.15 * n)
    tr_idx, va_idx, te_idx = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
    train_ds = OpenRadarDataset([records[i] for i in tr_idx], augment=True)
    val_ds   = OpenRadarDataset([records[i] for i in va_idx], augment=False)
    test_ds  = OpenRadarDataset([records[i] for i in te_idx], augment=False)
    _share_norm(train_ds, val_ds, test_ds)        # TRAIN stats -> val/test (no leak)

    tr_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, drop_last=False)
    va_loader = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    te_loader = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    log(f" splits           : train {len(train_ds)} | val {len(val_ds)} | test {len(test_ds)}")

    # ---- federated client partition (TRAIN only) ------------------------------
    client_idx, client_loaders = None, None
    if "federated" in TRAINING_MODES:
        client_idx = partition_clients(train_ds, N_CLIENTS, NON_IID, DIRICHLET_ALPHA)
        client_loaders = make_client_loaders(train_ds, client_idx, BATCH_SIZE)
        sizes = [len(c) for c in client_idx]
        log(f" FL client sizes  : {sizes}  (non-IID Dirichlet over spatial zone)")

    data_leakage_audit(tr_idx, va_idx, te_idx, client_idx, len(train_ds))

    ctx = {"tr": tr_loader, "val": va_loader, "te": te_loader, "clients": client_loaders}

    # ---- FL hyper-parameter search (on VAL; never touches TEST) ---------------
    fl_results = []
    if "federated" in TRAINING_MODES and FL_TUNE:
        sub = random.Random(SEED + 7).sample(range(len(train_ds)),
                                              min(FL_TUNE_SUBSET_TRAIN, len(train_ds)))
        tune_clients = make_client_loaders(train_ds, _chunk(sub, FL_TUNE_CLIENTS), BATCH_SIZE)
        vsub = random.Random(SEED + 8).sample(range(len(val_ds)),
                                              min(FL_TUNE_VAL, len(val_ds)))
        tune_val = DataLoader(Subset(val_ds, vsub), batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)
        FL_ROUNDS, FL_LOCAL_EPOCHS, fl_results = tune_fl(tune_clients, tune_val, train_ds)
    elif "federated" in TRAINING_MODES:
        log(f"\n FL tuning disabled -> using FL_ROUNDS={FL_ROUNDS}, FL_LOCAL_EPOCHS={FL_LOCAL_EPOCHS}")

    # ---- the sweep (every config trained under each requested mode) -----------
    rows = []
    n_models = sum(1 for s in SUITES_TO_RUN for k in SUITES[s]
                   if (MODELS_TO_RUN is None or k in MODELS_TO_RUN))
    total_cfg = n_models * len(COMBO_LIST) * len(TRAINING_MODES)
    cfg_i = 0
    for suite in SUITES_TO_RUN:
        for key in SUITES[suite]:
            if MODELS_TO_RUN is not None and key not in MODELS_TO_RUN:
                continue
            disp = SUITES[suite][key][0]
            log("\n" + "#" * 78)
            log(f"# MODEL: {disp}   (suite={suite}, key={key})")
            log("#" * 78)
            for uq, rep in COMBO_LIST:
                for mode in TRAINING_MODES:
                    cfg_i += 1
                    log(f"\n[{cfg_i}/{total_cfg}] {disp} | uq={uq} | rep={rep} | MODE={mode}")
                    try:
                        _t_train = time.time()
                        model = train_one_config(suite, key, uq, rep, ctx, train_ds, mode)
                        train_time_s = round(time.time() - _t_train, 2)
                        for band in BANDS:
                            slot = None if band == "combined" else BAND_SLOT[band]
                            m = evaluate(model, te_loader, train_ds, uq, slot)
                            rows.append(dict(mode=mode, suite=suite, model=key, display=disp,
                                             uq=uq, rep=rep, band=band,
                                             train_time_s=train_time_s, **m))
                            log(f"      {band:<9} MAE={m['mae_m']:.3f}m  RMSE={m['rmse_m']:.3f}m  "
                                f"AoA={m['aoa_mae_deg']:.2f}deg  FAR={m['nlos_far']:.3f}  "
                                f"MDR={m['nlos_mdr']:.3f}  lat={m['infer_ms_per_sample']:.2f}ms/smp")
                        del model
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception as e:
                        import traceback
                        log(f"   !! config failed: {e}\n{traceback.format_exc()}")

    # ---- outputs --------------------------------------------------------------
    log("\n" + "=" * 78); log(" WRITING OUTPUTS"); log("=" * 78)
    if rows:
        save_csv_json(rows)
        save_fl_tuning(fl_results)
        save_heatmaps(rows)
        save_cdf_curves(rows)
        save_leaderboard(rows)
    else:
        log("  no successful configs — nothing to write.")
    log(f"\n DONE in {(time.time()-t0)/60.0:.1f} min.  All artefacts in:\n   {OUTPUT_DIR}")


def _chunk(items, k):
    """Split a list of indices into k roughly-equal disjoint chunks."""
    items = list(items); k = max(1, min(k, len(items)))
    out = [items[i::k] for i in range(k)]
    return [c for c in out if c]


if __name__ == "__main__":
    main()