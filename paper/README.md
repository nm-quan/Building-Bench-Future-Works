# Paper draft

LaTeX source for the write-up of the BuildingsBench re-benchmark study.

- `main.tex` -- full draft following the journal-article format of the
  reference papers; section structure Abstract, Introduction, Related Work,
  Methodology (incl. real-building weather acquisition + validation), Results
  (incl. the simulation-to-real weather-transfer study), Limitations and Future
  Work, Conclusion.
- `refs.bib` -- bibliography. Core entries (BuildingsBench, ERA5, Open-Meteo,
  and the architecture papers) were cross-checked against the published sources;
  please re-confirm any entry before camera-ready. Citations are numbered
  (`natbib` `[numbers,sort&compress]`).
- `figures/loss_curves.png` -- generated from the actual validation-NLL logs.

## Build
```
pdflatex main && bibtex main && pdflatex main && pdflatex main
```
The document compiles **with or without** the external figure PNGs: each
`\includegraphics` is wrapped in `\figorbox`, which renders the real plot when
the file is present and a labelled placeholder otherwise.

## Before this is submittable
- Author names/affiliation (the `[TODO]` markers in `main.tex`).
- Copy the EDA/visualization PNGs from Drive `results/figures/` into `figures/`
  to replace the placeholders: `load_heatmap`, `daily_profiles`,
  `weather_load_corr`, `actual_versus_predicted` (and optionally
  `sensitivity_heatmap`).
- Remaining runs for a fully controlled comparison: TFTLite/B and the original
  Transformer-S/M/L checkpoints (to replace the published-value comparison in
  the "Comparison with the Original Benchmark Model" table with a same-pipeline
  head-to-head). The +weather real-building track and XGBoost/LightGBM real-eval
  are now complete and in the tables.
- The Crossformer/TimeXer weather-sensitivity spike is reported descriptively;
  the simulation-to-real weather-transfer section now provides the broader
  context for it.

## What is in the tables
All numeric values were transcribed directly from the committed notebook
(`study.ipynb`) outputs and cross-checked programmatically. New since the last
draft:
- **Real-building weather acquisition + validation gate** (Methodology): full
  seven-channel ERA5 weather fetched from Open-Meteo for each building's true
  location, gated against the benchmark's published temperature (Pearson
  r > 0.95 at zero lag, best lag 0 h, mean offset < 3 C). All 10 locations pass.
- **Simulation-to-real weather transfer** (Results): the large simulated
  +weather gains (e.g. Crossformer -21% commercial NRMSE) do not transfer to
  real buildings (mean |Delta| ~ 0.1 commercial NRMSE points, mixed sign).
- **Limitations and Future Work** section.
