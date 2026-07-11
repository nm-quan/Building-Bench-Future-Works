"""Metrics matching buildings_bench.evaluation exactly (NRMSE/NMAE via
per-building CV(RMSE)-style normalization, closed-form Gaussian CRPS), plus
the statistical-rigor layer the earlier draft of this study was missing:
bootstrap CIs, paired significance, AND multiple-comparison correction
(Holm-Bonferroni across pairwise tests, Friedman + Nemenyi across the full
model set).

Formulas cross-checked against buildings_bench/evaluation/metrics.py and
scoring_rules.py (BSD-3-Clause, NatLabRockies/BuildingsBench):
  - normalize=True divides by the mean of the true values accumulated for
    that same metric instance -- i.e. CV(RMSE) when applied per-building.
  - CRPS: sigma * (z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi)), z=(y-mu)/sigma.
"""
import math

import numpy as np
from scipy.special import erf
from scipy.stats import friedmanchisquare, studentized_range, wilcoxon


# ---------------------------------------------------------------------------
# Per-building accumulator -- call `update` once per evaluated window, then
# `finalize` once per building. Mirrors buildings_bench.evaluation.metrics.Metric
# with normalize=True, sqrt=True for NRMSE; normalize=True for NMAE.
# ---------------------------------------------------------------------------
class BuildingAccumulator:
    def __init__(self, n_buildings: int):
        self.se = np.zeros(n_buildings)
        self.sae = np.zeros(n_buildings)
        self.sy = np.zeros(n_buildings)
        self.sy2 = np.zeros(n_buildings)
        self.cr = np.zeros(n_buildings)
        self.cnt = np.zeros(n_buildings)
        self.has_sigma = False  # tracks whether ANY update() call provided sigma

    def update(self, building_idx: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
               sigma: np.ndarray = None) -> None:
        """building_idx: (B,) int array. y_true, y_pred: (B, H). sigma: (B, H) or None
        (e.g. for CopyLastDay/CopyLastWeekPersistence, which have no
        probabilistic forecast -- matches the official model's predict() ->
        (forecast, None). CRPS is left as NaN, not silently reported as 0,
        for models that never provide a sigma.)"""
        err = y_pred - y_true
        np.add.at(self.se, building_idx, (err ** 2).sum(1))
        np.add.at(self.sae, building_idx, np.abs(err).sum(1))
        np.add.at(self.sy, building_idx, y_true.sum(1))
        np.add.at(self.sy2, building_idx, (y_true ** 2).sum(1))
        np.add.at(self.cnt, building_idx, y_true.shape[1])
        if sigma is not None:
            self.has_sigma = True
            np.add.at(self.cr, building_idx, gaussian_crps(y_true, y_pred, sigma).sum(1))

    def finalize(self) -> dict:
        valid = self.cnt > 0
        rmse = np.full_like(self.se, np.nan)
        mae = np.full_like(self.sae, np.nan)
        meany = np.full_like(self.sy, np.nan)
        crps = np.full_like(self.cr, np.nan)
        rmse[valid] = np.sqrt(self.se[valid] / self.cnt[valid])
        mae[valid] = self.sae[valid] / self.cnt[valid]
        meany[valid] = self.sy[valid] / self.cnt[valid]
        if self.has_sigma:
            crps[valid] = self.cr[valid] / self.cnt[valid]
        ok = valid & (meany > 1e-6)
        nrmse = np.full_like(self.se, np.nan)
        nmae = np.full_like(self.sae, np.nan)
        nrmse[ok] = 100 * rmse[ok] / meany[ok]
        nmae[ok] = 100 * mae[ok] / meany[ok]
        return {"nrmse": nrmse, "nmae": nmae, "crps": crps, "valid": ok}


def gaussian_crps(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Closed-form Gaussian CRPS, elementwise. Matches
    ContinuousRankedProbabilityScore.crps exactly."""
    z = (y_true - mu) / sigma
    cdf = 0.5 * (1 + erf(z / math.sqrt(2)))
    pdf = np.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
    return sigma * (z * (2 * cdf - 1) + 2 * pdf - 1 / math.sqrt(math.pi))


def summarize_by_group(result: dict, is_res: np.ndarray) -> dict:
    """result: output of BuildingAccumulator.finalize(). Returns {"Com NRMSE":
    median, "Res NRMSE": median, ...} matching the paper's Com/Res reporting."""
    out = {}
    for label, mask in [("Com", ~is_res), ("Res", is_res)]:
        m = mask & result["valid"]
        if m.any():
            for metric in ("nrmse", "nmae", "crps"):
                vals = result[metric][m]
                # crps is legitimately all-NaN for models with no probabilistic
                # forecast (CopyLastDay/CopyLastWeekPersistence) -- nanmedian's
                # "All-NaN slice" warning is expected there, not a bug.
                out[f"{label} {metric.upper()}"] = float(np.nanmedian(vals)) if np.isfinite(vals).any() else float("nan")
    return out


# ---------------------------------------------------------------------------
# Bootstrap CI + paired significance (unchanged formulas from the earlier
# draft -- these were already correct)
# ---------------------------------------------------------------------------
def bootstrap_median_ci(values: np.ndarray, n_boot: int = 2000, ci: int = 95, seed: int = 0):
    values = np.asarray(values)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boots = np.array([np.median(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return (float(np.median(values)), float(lo), float(hi))


def paired_significance(values_a: np.ndarray, values_b: np.ndarray, alpha: float = 0.05):
    """Paired Wilcoxon signed-rank test on matched per-building values.
    Returns (p_value, a_better: bool or None)."""
    a, b = np.asarray(values_a), np.asarray(values_b)
    if a.shape != b.shape:
        return (float("nan"), None)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) < 10 or np.allclose(a, b):
        return (float("nan"), None)
    try:
        _, p = wilcoxon(a, b)
    except ValueError:
        return (float("nan"), None)
    a_better = bool(p < alpha and np.median(a) < np.median(b))
    return (float(p), a_better)


# ---------------------------------------------------------------------------
# Multiple-comparison correction -- MISSING from the earlier draft, which ran
# ~12-20 pairwise Wilcoxon tests at alpha=0.05 uncorrected (spurious
# "significant" wins expected by chance at that count).
# ---------------------------------------------------------------------------
def holm_bonferroni(p_values: dict, alpha: float = 0.05) -> dict:
    """p_values: {model_name: p_value}. Returns {model_name: (p_value,
    significant_after_correction: bool)}, using the Holm-Bonferroni
    step-down procedure (uniformly more powerful than plain Bonferroni,
    still controls family-wise error rate)."""
    items = sorted(((k, v) for k, v in p_values.items() if v == v), key=lambda kv: kv[1])
    m = len(items)
    out = {k: (v, False) for k, v in p_values.items()}
    for i, (k, p) in enumerate(items):
        threshold = alpha / (m - i)
        if p <= threshold:
            out[k] = (p, True)
        else:
            break  # Holm-Bonferroni: stop at the first non-rejection
    return out


def friedman_nemenyi(rank_data: np.ndarray, model_names: list, alpha: float = 0.05) -> dict:
    """rank_data: (n_buildings, n_models) array of a per-building metric
    (e.g. NRMSE) for every model, same buildings across columns. Runs a
    Friedman test (is there any difference among the models at all?) and, if
    significant, computes the Nemenyi critical difference for pairwise
    average-rank comparisons -- the standard way to report "which models
    differ" across a full model set without re-running N^2 uncorrected
    pairwise tests.
    """
    n, k = rank_data.shape
    valid_rows = ~np.isnan(rank_data).any(axis=1)
    data = rank_data[valid_rows]
    n = data.shape[0]

    stat, p = friedmanchisquare(*[data[:, j] for j in range(k)])

    ranks = np.apply_along_axis(lambda row: pd_rank(row), 1, data)
    avg_ranks = ranks.mean(axis=0)

    q_alpha = studentized_range.ppf(1 - alpha, k, np.inf) / math.sqrt(2)
    cd = q_alpha * math.sqrt(k * (k + 1) / (6 * n))

    return {
        "friedman_stat": float(stat), "friedman_p": float(p), "n_buildings": int(n),
        "avg_ranks": dict(zip(model_names, avg_ranks.tolist())),
        "nemenyi_critical_difference": float(cd),
        "significant_pairs": _nemenyi_pairs(model_names, avg_ranks, cd),
    }


def pd_rank(row: np.ndarray) -> np.ndarray:
    """Ranks a 1D array (1 = smallest/best), averaging ties -- avoids adding
    a pandas dependency just for `Series.rank`."""
    order = np.argsort(row, kind="mergesort")
    ranks = np.empty(len(row), dtype=float)
    sorted_vals = row[order]
    i = 0
    while i < len(row):
        j = i
        while j + 1 < len(row) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    return ranks


def _nemenyi_pairs(model_names: list, avg_ranks: np.ndarray, cd: float) -> list:
    pairs = []
    k = len(model_names)
    for i in range(k):
        for j in range(i + 1, k):
            diff = abs(avg_ranks[i] - avg_ranks[j])
            if diff > cd:
                pairs.append((model_names[i], model_names[j], float(diff)))
    return pairs
