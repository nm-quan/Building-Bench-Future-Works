# Legacy notebooks (superseded — kept for provenance)

These are the original study notebooks, replaced by the `bblab/` package +
`study.ipynb`. They are **not maintained** and should not be run, but they are
kept in the repo deliberately:

- `data_prep_study_(2).ipynb` — the original data-preparation notebook
  (per-building PowerTransformer/StandardScaler pipeline, its own ~1,480-building
  simulated test sample).
- `train_eval_study.ipynb` — the original train/eval notebook (~20 models
  including several homegrown architectures), **with stored outputs**.

## Why they were replaced

The current pipeline fixes issues the audit found here: sin/cos calendar
encoding (the paper uses linear [-1, 1]), calendar features left unnormalized in
the real-building eval path, a crashed real-building significance cell
(`PERBUILDING_REAL_DIR` NameError), a real-weather ablation that zero-filled
5 of 7 channels, a single fixed validation window, a silently zero-filled
corrupt training building, no multiple-comparison correction across ~20
significance tests, and a non-citable custom model zoo.

## Why they are worth keeping

`train_eval_study.ipynb`'s stored outputs match the BuildingsBench paper's
published Buildings-900K-Test table to within 0.5% on four persistence
baselines (Persistence Ensemble 33.03/54.60 vs the paper's 33.10/54.77;
Previous-Day 34.83/59.22 vs 34.91/59.38). These outputs are the reference
that exposed the temporal-scrambling bug in an early version of `bblab/data.py`
(Buildings-900K parquet rows are not stored chronologically and must be sorted
— this notebook's prep did that correctly). They remain the study's external
sanity anchor: a fresh `bblab` run's persistence medians should land on these
same values.
