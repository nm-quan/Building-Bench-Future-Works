"""Dataset EDA / visualization for the paper: load heatmaps, daily profiles,
weather-load correlation, and a real-dataset overview. Every figure is saved
as a 200-dpi PNG under paths.FIGURES_DIR (on Drive) so the artifacts survive
the Colab session and can go straight into the manuscript.

Pure matplotlib (no seaborn dependency); each function returns the saved path
(and the overview also returns its summary DataFrame).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import config


def _norm_loads(ds: dict, mask: np.ndarray, max_buildings: int, seed: int = 0):
    """Per-building mean-normalized loads (N', T) for a building-type mask --
    normalizing per building keeps one big commercial site from dominating
    the group average."""
    idx = np.where(mask)[0]
    if len(idx) > max_buildings:
        idx = np.random.default_rng(seed).choice(idx, max_buildings, replace=False)
    loads = ds["loads"][idx]
    m = loads.mean(axis=1, keepdims=True)
    m[m <= 1e-9] = 1.0
    return loads / m


def load_heatmaps(ds: dict, paths: config.Paths, max_buildings: int = 2000) -> str:
    """Mean normalized load, day-of-year x hour-of-day, Com vs Res -- the
    'shape of the dataset' figure (seasonal + diurnal structure in one map)."""
    days = ds["T"] // 24
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for ax, (label, mask) in zip(axes, [("Commercial", ~ds["is_res"]), ("Residential", ds["is_res"])]):
        g = _norm_loads(ds, mask, max_buildings).mean(0)[: days * 24].reshape(days, 24)
        im = ax.imshow(g.T, aspect="auto", origin="lower", cmap="viridis", interpolation="nearest")
        ax.set(title=f"{label} (n={int(mask.sum())})", xlabel="day of year", ylabel="hour of day")
        fig.colorbar(im, ax=ax, label="load / building mean")
    fig.suptitle("Buildings-900K subsample: mean normalized load")
    out = f"{paths.FIGURES_DIR}/load_heatmap.png"
    fig.savefig(out, dpi=200); plt.show(); plt.close(fig)
    return out


def daily_profiles(ds: dict, paths: config.Paths, max_buildings: int = 2000) -> str:
    """Median + IQR daily profile (hour-of-day), Com vs Res."""
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    hours = np.arange(24)
    for label, mask, color in [("Commercial", ~ds["is_res"], "tab:blue"),
                                ("Residential", ds["is_res"], "tab:orange")]:
        ln = _norm_loads(ds, mask, max_buildings)
        prof = ln[:, : (ds["T"] // 24) * 24].reshape(len(ln), -1, 24).mean(1)  # (N', 24) per-building mean profile
        med = np.median(prof, 0); lo, hi = np.percentile(prof, [25, 75], axis=0)
        ax.plot(hours, med, color=color, label=label)
        ax.fill_between(hours, lo, hi, color=color, alpha=0.2)
    ax.set(xlabel="hour of day", ylabel="load / building mean", title="Mean daily profile (median, IQR)")
    ax.legend()
    out = f"{paths.FIGURES_DIR}/daily_profiles.png"
    fig.savefig(out, dpi=200); plt.show(); plt.close(fig)
    return out


def weather_load_correlation(ds: dict, paths: config.Paths, max_buildings: int = 1500) -> str:
    """Heatmap of the mean per-building Pearson correlation between hourly
    load and each (normalized) weather channel, Com vs Res. The
    'is there weather signal in this data at all' figure."""
    rows = []
    for label, mask in [("Commercial", ~ds["is_res"]), ("Residential", ds["is_res"])]:
        idx = np.where(mask & (ds["b2w"] >= 0))[0]
        if len(idx) > max_buildings:
            idx = np.random.default_rng(0).choice(idx, max_buildings, replace=False)
        cors = np.full((len(idx), config.N_WEATHER), np.nan)
        for r, bi in enumerate(idx):
            y = ds["loads"][bi]
            ys = y.std()
            if ys <= 1e-9:
                continue
            w = ds["wuniq"][ds["b2w"][bi]]                       # (T, n_weather), already standardized
            yc = (y - y.mean()) / ys
            ws = w.std(0); ws[ws <= 1e-9] = 1.0
            cors[r] = ((w - w.mean(0)) / ws * yc[:, None]).mean(0)
        rows.append(np.nanmean(cors, axis=0))
    mat = np.stack(rows)

    fig, ax = plt.subplots(figsize=(9, 2.8), constrained_layout=True)
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    ax.set_xticks(range(config.N_WEATHER), [c.replace("_", "\n") for c in config.WEATHER_COLS], fontsize=8)
    ax.set_yticks([0, 1], ["Com", "Res"])
    for i in range(2):
        for j in range(config.N_WEATHER):
            ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="mean Pearson r (load vs channel)")
    ax.set_title("Weather-load correlation")
    out = f"{paths.FIGURES_DIR}/weather_load_corr.png"
    fig.savefig(out, dpi=200); plt.show(); plt.close(fig)
    return out


def real_dataset_overview(buildings: list, paths: config.Paths):
    """Per-dataset summary of the real-building eval set (counts, com/res,
    length, load scale) -- table saved as CSV + bar chart PNG. Returns
    (DataFrame, png_path)."""
    rows = {}
    for b in buildings:
        ds_name = b["building_id"].split(":", 1)[0]
        r = rows.setdefault(ds_name, {"buildings": 0, "com": 0, "res": 0, "hours": [], "mean_kw": []})
        r["buildings"] += 1
        r["com" if b["building_type"] > 0 else "res"] += 1
        r["hours"].append(len(b["series"]))
        r["mean_kw"].append(float(b["series"].mean()))
    df = pd.DataFrame([{
        "dataset": k, "buildings": v["buildings"], "com": v["com"], "res": v["res"],
        "median_hours": int(np.median(v["hours"])), "median_mean_kw": round(float(np.median(v["mean_kw"])), 3),
    } for k, v in sorted(rows.items())])
    df.to_csv(f"{paths.FIGURES_DIR}/real_overview.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 3.5), constrained_layout=True)
    x = np.arange(len(df))
    ax.bar(x, df["com"], label="commercial", color="tab:blue")
    ax.bar(x, df["res"], bottom=df["com"], label="residential", color="tab:orange")
    ax.set_xticks(x, df["dataset"], rotation=20)
    ax.set(ylabel="# building-years", title="Real-building evaluation set")
    ax.set_yscale("log")
    ax.legend()
    out = f"{paths.FIGURES_DIR}/real_overview.png"
    fig.savefig(out, dpi=200); plt.show(); plt.close(fig)
    return df, out
