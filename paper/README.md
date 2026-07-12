# Paper draft

LaTeX source for the write-up of the BuildingsBench re-benchmark study.

- `main.tex` -- full draft following the journal-article format of the
  reference papers; section structure per the study outline (Abstract,
  Introduction, Related Work, Methodology, Results, Conclusion).
- `refs.bib` -- bibliography. **Entries are drafted from recall and MUST be
  verified against the real publications before submission** (see the note at
  the top of the file).
- `figures/loss_curves.png` -- generated from the actual validation-NLL logs.

## Build
```
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Before this is submittable
Every `[PENDING]` (orange) and `[TODO]` (red) marker in `main.tex` flags
either missing data or a placeholder:
- Author names/affiliation.
- Copy the EDA PNGs from Drive `results/figures/` (load_heatmap, daily_profiles,
  weather_load_corr, sensitivity_heatmap) into `figures/`.
- Complete runs: +weather real-building track; XGBoost/LightGBM real-eval;
  TFTLite/B and the original Transformer-S/M/L checkpoints (for a controlled
  same-pipeline comparison replacing the published-value one in Table 5).
- Resolve the two flagged anomalies before writing them as findings: the
  Crossformer/TimeXer weather-sensitivity spike and the real-world RPS scale
  discrepancy.
- Verify every `refs.bib` entry.

All numeric values currently in the tables were transcribed directly from the
committed notebook outputs and cross-checked programmatically.
