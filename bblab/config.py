"""Single source of truth for paths, window/horizon, weather channels, and
training hyperparameters. Every other bblab module and the notebook import
from here instead of re-declaring constants -- this is what fixes the
PERBUILDING_REAL_DIR NameError class of bug in the old notebook (paths were
declared ad hoc in multiple cells and could silently go out of scope).
"""
import os
import torch

# ---------------------------------------------------------------------------
# Window / horizon -- matches BuildingsBench's context_len=168, pred_len=24
# ---------------------------------------------------------------------------
L = 168          # history length (hours)
H = 24           # forecast horizon (hours)
WIN = L + H      # 192
HOURS = 8760     # one year, fixed-length synthetic series

# ---------------------------------------------------------------------------
# Weather channels -- confirmed order from buildings_bench/data/buildings900K.py
# ---------------------------------------------------------------------------
WEATHER_COLS = [
    "temperature", "humidity", "wind_speed", "wind_direction",
    "global_horizontal_radiation", "direct_normal_radiation",
    "diffuse_horizontal_radiation",
]
N_WEATHER = len(WEATHER_COLS)          # 7
N_TIME = 3                             # day-of-year, day-of-week, hour-of-day (linear [-1,1], not sin/cos)

CONDITIONS = {"A": False, "B": True}   # A = no weather, B = +weather

# ---------------------------------------------------------------------------
# Model registry (populated in models.py; listed here for the notebook to
# import a stable, single ordering used everywhere -- training, results,
# significance tests).
# ---------------------------------------------------------------------------
PERSISTENCE_MODELS = ["persistence_avg", "persistence_last_day", "persistence_last_week"]
TRAINED_NEURAL_MODELS = [
    "lstm", "gru", "patchtst", "itransformer", "timexer",
    "dlinear", "informer", "autoformer", "crossformer",
]
# The paper's own pretrained model family (buildings_bench/models/transformers.py,
# continuous_loads=True, continuous_head='gaussian_nll') -- not a canonical
# baseline from elsewhere, the actual architecture BuildingsBench itself uses.
# transformer_l matches their largest config (12+12 layers, d_model=768) and is
# meaningfully more compute-heavy per epoch than everything else in this
# registry; resumable checkpointing (see train.run_training_sweep) means it's
# fine to let it span multiple Colab sessions.
PAPER_MODELS = ["transformer_s", "transformer_m", "transformer_l"]
TREE_MODELS = ["xgboost", "lightgbm"]
# Novel architectures adapted from post-BuildingsBench literature (see study.ipynb
# intro for citations): TFT-style variable selection (tftlite), xLSTM's sLSTM
# exponential-gated cell (xlstm), TSMixer+FITS frequency extrapolation with a
# degree-day correction term (spectramix), and Mamba/S6 selective-state-space
# scan (mamba) -- all operate over 24h patches (7 patches per 168h window)
# instead of raw hourly steps to keep the sequential-recurrence ones (xlstm,
# mamba) cheap enough for the Colab training budget.
NOVEL_MODELS = ["tftlite", "xlstm", "spectramix", "mamba"]
ALL_MODELS = PERSISTENCE_MODELS + TRAINED_NEURAL_MODELS + PAPER_MODELS + TREE_MODELS + NOVEL_MODELS
BASELINE_MODEL = "persistence_avg"    # reference point for significance tests

RNN_MODELS = {"lstm", "gru", "xlstm"}  # trained fp32 -- bf16 destabilizes exponential/gated recurrence

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
EPOCHS, PATIENCE = 200, 20
STEPS, BS, LR = 300, 512, 4e-4
SEED = 0
USE_AMP, AMP_DTYPE = True, torch.bfloat16
N_VAL_WINDOWS_PER_BUILDING = 4         # resampled positions per val building per epoch
# Matches the official val_timerange (2018-12-17 to 2018-12-31 = 14 days) from
# scripts/data_generation/create_index_files.py -- validation is a temporal
# holdout on the amy2018-release buildings already in the training cache, not
# a separate building pool or a separate S3 fetch. See train.train().
VAL_HOLDOUT_HOURS = 336

XGB_WINDOWS, XGB_TREES, XGB_DEPTH = 80000, 500, 6
LGB_WINDOWS, LGB_TREES, LGB_DEPTH = 80000, 500, 6

N_TRAIN_BUILDINGS = 20000              # compute-bounded subsample of Buildings-900K

# ---------------------------------------------------------------------------
# Weather-value analyses (bblab/analysis.py)
# ---------------------------------------------------------------------------
ROLLOUT_STEPS = 3                      # autoregressive rollout: 3 x 24h = 72h total horizon
EXTREME_PCT = 10                       # extreme day = top/bottom decile of window future temperature

# ---------------------------------------------------------------------------
# Paths -- resolved once, imported everywhere (no re-declaration in notebook cells)
# ---------------------------------------------------------------------------
def resolve_bench_root(default="/content/drive/MyDrive/quick/bench"):
    import glob
    if os.path.isdir(default):
        return default
    hits = glob.glob("/content/drive/MyDrive/**/bench", recursive=True)
    return hits[0] if hits else default


class Paths:
    def __init__(self, bench_root=None):
        self.BENCH = bench_root or resolve_bench_root()
        self.RAW_CACHE_DIR = f"{self.BENCH}/raw_cache"
        self.TRAIN_DIR = f"{self.BENCH}/train_20k"
        self.SIM_TEST_DIR = f"{self.BENCH}/sim_test"
        self.TRANSFORMS_DIR = f"{self.BENCH}/transforms"

        self.RESULTS_DIR = f"{self.BENCH}/results"
        self.WEIGHTS_DIR = f"{self.RESULTS_DIR}/weights"
        # v2: paper-global metrics + paper CRPS approximation + compute
        # accounting. A new filename on purpose -- rows in the old
        # sim_results.csv were computed under different metric definitions, and
        # pointing at a fresh file makes the sweep re-EVALUATE everything while
        # still reusing every trained checkpoint in WEIGHTS_DIR (no retraining).
        self.SIM_CSV = f"{self.RESULTS_DIR}/sim_results_v2.csv"
        self.REAL_CSV = f"{self.RESULTS_DIR}/real_results.csv"
        self.REAL_WEATHER_TEMP_CSV = f"{self.RESULTS_DIR}/real_weather_temp_results.csv"

        self.PERBUILDING_SIM_DIR = f"{self.RESULTS_DIR}/perbuilding_sim"
        self.PERBUILDING_REAL_DIR = f"{self.RESULTS_DIR}/perbuilding_real"
        self.PERBUILDING_REAL_WEATHER_TEMP_DIR = f"{self.RESULTS_DIR}/perbuilding_real_weather_temp"

        self.FIGURES_DIR = f"{self.RESULTS_DIR}/figures"        # EDA + analysis plots (Drive)
        self.ANALYSIS_DIR = f"{self.RESULTS_DIR}/analysis"      # weather-value analysis CSVs (Drive)

    def makedirs(self):
        for d in (self.RAW_CACHE_DIR, self.TRANSFORMS_DIR, self.RESULTS_DIR, self.WEIGHTS_DIR,
                  self.PERBUILDING_SIM_DIR, self.PERBUILDING_REAL_DIR,
                  self.PERBUILDING_REAL_WEATHER_TEMP_DIR, self.FIGURES_DIR, self.ANALYSIS_DIR):
            os.makedirs(d, exist_ok=True)
        return self
