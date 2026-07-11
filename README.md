# Building-Bench Future Works

A BuildingsBench-faithful short-term load forecasting (STLF) study: **21 models
× 2 weather conditions**, trained on a compute-bounded subsample of the official
Buildings-900K corpus and evaluated on both the official simulated test set and
7 real-building datasets — plus weather-value analyses (extreme days, longer
horizons, per-channel sensitivity) and dataset EDA figures.

Everything runs from **one notebook** (`study.ipynb`, on Google Colab) backed by
a small reviewable package (`bblab/`).

## Repository layout

```
study.ipynb          # the ONE notebook to run (Colab, A100 recommended)
bblab/
├── config.py        # single source of truth: window/horizon, model lists, paths, hyperparams
├── data.py          # S3 sourcing, official transforms, chronological sort, cache building
├── bb_transforms.py # vendored official BuildingsBench transforms (BSD-3-Clause, attributed)
├── models.py        # 21-model registry (persistence x3, LSTM/GRU, trees x2,
│                    #   7 canonical transformer/linear baselines >=7M params,
│                    #   the paper's own Transformer-S/M/L (Gaussian), 4 novel models)
├── train.py         # shared train/eval loop, resumable sweep, compute-budget tracking
├── metrics.py       # paper-exact metrics + bootstrap CI, Wilcoxon+Holm-Bonferroni, Friedman+Nemenyi
├── analysis.py      # weather-value analyses: extreme days, 48/72h rollout, channel knockout
├── eda.py           # dataset EDA figures (load heatmaps, profiles, weather correlation)
└── weather_real.py  # published era5 temp/humidity track for real-building +weather eval
legacy/              # superseded original notebooks (kept for provenance -- see legacy/README.md)
```

## Reproduction

1. Open `study.ipynb` in Google Colab (GPU runtime; A100 recommended).
2. Run cells top to bottom. Cell 1 clones/pulls this repo and installs
   `requirements.txt`; everything else persists to Google Drive under
   `quick/bench/` (change `bblab.config.resolve_bench_root` to relocate).
3. Every long-running step is **resumable**: data caches, model checkpoints
   (`results/weights_v2/`), per-(model, condition) result rows, and analysis
   CSVs are all skipped if already present on Drive. A Colab disconnect costs
   only the model currently training.
4. Compute: the full 21-model × 2-condition sweep is several hours on an A100
   (Transformer-L dominates; `config.ALL_MODELS.remove("transformer_l")` to
   skip it). Per-model epochs/wall-time/peak-memory are recorded in
   `results/sim_results_v4.csv`.

### Sanity anchors (check these before trusting a run)

Persistence baselines are closed-form, so their medians are a pure data check.
On the official Buildings-900K-Test they should reproduce the paper's table:

| Baseline | Com NRMSE | Res NRMSE |
|---|---|---|
| Average Persistence | ~33.1 | ~54.8 |
| Previous-Day Persistence | ~34.9 | ~59.4 |

On the real-building benchmark, the official leaderboard sits at Persistence
Ensemble 16.68 (Com) / 77.88 (Res); the paper's pretrained Transformer-L
(Gaussian) reaches 13.31 / 79.34.

## Methodology notes (paper fidelity)

- **Preprocessing is the paper's, not a reimplementation**: the official
  already-fit Box-Cox load transform and per-channel weather StandardScalers
  are downloaded from the public `oedi-data-lake` S3 bucket (and the Box-Cox
  pickle is validated by a numeric round-trip at load time); calendar features
  are linearly scaled to [-1, 1] (day-of-year / day-of-week / hour-of-day),
  matching `TimestampTransform`.
- **Chronological sort**: Buildings-900K parquet rows are *not* stored in time
  order (per the official loader's own docstring); `bblab/data.py` sorts and
  asserts monotonic timestamps. Skipping this scrambles every series —
  discovered the hard way; see `legacy/README.md`.
- **Validation split** matches the official one: a temporal holdout of the last
  2 weeks of the year, amy2018-release buildings only.
- **Metrics**: headline NRMSE/NMAE/CRPS use the paper's published aggregation —
  the **median across per-building values** (their `return_aggregate_median`),
  with per-building CV(RMSE)-style normalization and CRPS in kWh via the
  paper's inverse-Box-Cox Gaussian approximation. A pooled/global view is kept
  as `*_pooled` columns (their `pretrain.py` validation monitor) — diagnostic
  only, **not** comparable to the paper's tables.
- **Statistics**: bootstrap median CIs, paired Wilcoxon vs. persistence with
  Holm-Bonferroni correction, and Friedman + Nemenyi across the full model set.
- **Weather conditions**: A = no weather, B = + the 7 official weather
  channels (county-joined, official scalers). Real-building +weather uses only
  the era5 temperature/humidity actually published per dataset — no external
  fetching, no filled-in channels.

## License

BSD 3-Clause (see `LICENSE`). `bblab/bb_transforms.py` is vendored from
[BuildingsBench](https://github.com/NatLabRockies/BuildingsBench)
(BSD-3-Clause, Alliance for Sustainable Energy, LLC) with attribution.
If you use this study, please also cite the BuildingsBench paper
(Emami, Sahu, Graf, *NeurIPS 2023 Datasets & Benchmarks*).
