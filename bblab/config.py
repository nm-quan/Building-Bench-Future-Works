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
TREE_MODELS = ["xgboost", "lightgbm"]
ALL_MODELS = PERSISTENCE_MODELS + TRAINED_NEURAL_MODELS + TREE_MODELS
BASELINE_MODEL = "persistence_avg"    # reference point for significance tests

RNN_MODELS = {"lstm", "gru"}          # trained fp32 -- bf16 destabilizes RNNs

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
EPOCHS, PATIENCE = 200, 20
STEPS, BS, LR = 300, 512, 4e-4
SEED = 0
USE_AMP, AMP_DTYPE = True, torch.bfloat16
N_VAL_WINDOWS_PER_BUILDING = 4         # fixes the single-fixed-window validation bug
VFRAC = 0.05

XGB_WINDOWS, XGB_TREES, XGB_DEPTH = 80000, 500, 6
LGB_WINDOWS, LGB_TREES, LGB_DEPTH = 80000, 500, 6

N_TRAIN_BUILDINGS = 20000              # compute-bounded subsample of Buildings-900K

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
        self.SIM_CSV = f"{self.RESULTS_DIR}/sim_results.csv"
        self.REAL_CSV = f"{self.RESULTS_DIR}/real_results.csv"
        self.REAL_WEATHER_TEMP_CSV = f"{self.RESULTS_DIR}/real_weather_temp_results.csv"

        self.PERBUILDING_SIM_DIR = f"{self.RESULTS_DIR}/perbuilding_sim"
        self.PERBUILDING_REAL_DIR = f"{self.RESULTS_DIR}/perbuilding_real"
        self.PERBUILDING_REAL_WEATHER_TEMP_DIR = f"{self.RESULTS_DIR}/perbuilding_real_weather_temp"

    def makedirs(self):
        for d in (self.RAW_CACHE_DIR, self.TRANSFORMS_DIR, self.RESULTS_DIR, self.WEIGHTS_DIR,
                  self.PERBUILDING_SIM_DIR, self.PERBUILDING_REAL_DIR,
                  self.PERBUILDING_REAL_WEATHER_TEMP_DIR):
            os.makedirs(d, exist_ok=True)
        return self
