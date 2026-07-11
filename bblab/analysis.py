"""Weather-value analyses -- the "finding" layer on top of the benchmark:

1. Extreme-day evaluation (`window_errors` + `extreme_day_summary`): error
   conditional on the window's future temperature being in the coldest /
   hottest decile of that building's year vs. mild days. Comparing condition
   A (no weather) vs B (+weather) per bucket answers *when* weather inputs
   actually pay off -- averages over a whole year can hide value that's
   concentrated in exactly the extreme days grid operators care about.
2. Autoregressive horizon rollout (`rollout_eval`): every model is trained
   for H=24; feeding its own 24h forecast back as history extends the
   horizon to 48h/72h. Weather (known exogenous future) plausibly matters
   *more* at longer horizons, where persistence-style copying decays.
3. Weather-sensitivity gating (`weather_sensitivity`): knock out one weather
   channel at a time (set to the training mean, i.e. 0 in normalized space)
   and measure how much predictions move and how much error degrades --
   a per-model, per-channel measure of how much each input is actually used.

All functions work through a `predict_fn(yh_bc_dev, exo_dev) -> mu_bc numpy`
closure (see `neural_predict_fn` / `tree_predict_fn`), so persistence, neural
(incl. the teacher-forcing paper models, which greedy-decode when called
without yf), and tree models all go through identical analysis code.
"""
import numpy as np
import torch

from . import config, models
from .train import gather


# ---------------------------------------------------------------------------
# predict_fn constructors
# ---------------------------------------------------------------------------
def neural_predict_fn(model, dev: str, amp: bool = True, amp_dtype=torch.bfloat16):
    model.eval()

    def fn(yh_bc, exo):
        with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype, enabled=(amp and dev == "cuda")):
            mu_n, rs, mean, sd = model(yh_bc, exo)
        return (mu_n.float() * sd.float() + mean.float()).cpu().numpy()

    return fn


def tree_predict_fn(mdl, L: int = None):
    L = L or config.L

    def fn(yh_bc, exo):
        X, mu, sd = models._tree_xy(yh_bc, exo, L)
        return models.tree_predict(mdl, X) * sd[:, None] + mu[:, None]

    return fn


def load_predict_fn(name: str, cond: str, use_w: bool, Ft: int, Fw: int,
                    paths: config.Paths, dev: str):
    """Builds a predict_fn for a model from its SAVED checkpoint on Drive
    (never trains). Returns None if that (model, condition) hasn't finished
    training yet -- analysis cells simply skip it and pick it up on re-run."""
    import os
    import pickle

    if name in config.TREE_MODELS:
        ckpt = f"{paths.WEIGHTS_DIR}/{name}_{cond}.pkl"
        if not os.path.exists(ckpt):
            return None
        mdl, _sigma = pickle.load(open(ckpt, "rb"))
        return tree_predict_fn(mdl)
    base = models.build(name, L=config.L, H=config.H, n_time=Ft, n_weather=Fw,
                        use_weather=use_w, **models.MODEL_KW.get(name, {})).to(dev)
    if models.count_params(base) > 0:
        ckpt = f"{paths.WEIGHTS_DIR}/{name}_{cond}.pt"
        if not os.path.exists(ckpt):
            return None
        try:
            base.load_state_dict(torch.load(ckpt, map_location=dev))
        except RuntimeError:
            return None  # stale checkpoint from an older architecture size -- skip until retrained
    amp = config.USE_AMP and (name not in config.RNN_MODELS)
    return neural_predict_fn(base, dev, amp=amp)


# ---------------------------------------------------------------------------
# Shared pooled-metric helper (paper-global aggregation, matches
# metrics.summarize_global but over per-WINDOW sums)
# ---------------------------------------------------------------------------
def _pooled_nrmse(se, ysum, cnt, mask):
    n = cnt[mask].sum()
    if n == 0:
        return float("nan")
    meany = ysum[mask].sum() / n
    if meany <= 1e-6:
        return float("nan")
    return float(100 * np.sqrt(se[mask].sum() / n) / meany)


# ---------------------------------------------------------------------------
# 1) Per-window errors + future-temperature tag -> extreme-day buckets
# ---------------------------------------------------------------------------
@torch.no_grad()
def window_errors(predict_fn, ds: dict, dev: str, use_w: bool, load_transform,
                  stride: int = 24, chunk: int = 1024) -> dict:
    """Evaluates every stride-spaced window and returns PER-WINDOW arrays:
    b (building idx), s (start hour), se/ae/ysum (summed over the 24h
    horizon, kWh space), cnt (=H), temp (mean normalized future temperature
    of the window; NaN for buildings with no weather match)."""
    smax = ds["T"] - config.WIN
    pairs = [(bi, s) for bi in range(ds["N"]) for s in range(0, smax, stride)]

    out_b, out_s, out_se, out_ae, out_y, out_t = [], [], [], [], [], []
    for i in range(0, len(pairs), chunk):
        ch = pairs[i:i + chunk]
        b = torch.tensor([p[0] for p in ch], device=dev)
        s = torch.tensor([p[1] for p in ch], device=dev)
        yh, _, exo, yf_kw = gather(ds, b, s, use_w, load_transform, dev)
        mu_kw = load_transform.undo_transform(predict_fn(yh, exo))
        err = mu_kw - yf_kw.cpu().numpy()
        out_b.append(b.cpu().numpy()); out_s.append(s.cpu().numpy())
        out_se.append((err ** 2).sum(1)); out_ae.append(np.abs(err).sum(1))
        out_y.append(yf_kw.cpu().numpy().sum(1))
        # future-temperature tag (channel 0 of the OFFICIAL weather order),
        # taken from the cache directly so it exists for condition-A models too
        b2w = ds["b2w_t"][b]
        idx = s[:, None] + torch.arange(config.L, config.WIN, device=dev)
        temp = ds["wuniq_t"][b2w.clamp_min(0), :, 0].gather(1, idx).mean(1)
        temp = torch.where(b2w >= 0, temp, torch.full_like(temp, float("nan")))
        out_t.append(temp.cpu().numpy())

    H = float(config.H)
    return {"b": np.concatenate(out_b), "s": np.concatenate(out_s),
            "se": np.concatenate(out_se), "ae": np.concatenate(out_ae),
            "ysum": np.concatenate(out_y), "temp": np.concatenate(out_t),
            "cnt": np.full(sum(len(x) for x in out_b), H)}


def extreme_day_summary(win: dict, ds: dict, pct: int = None) -> dict:
    """Buckets windows into cold/mild/hot by each BUILDING'S OWN future-temp
    percentiles (so 'extreme' is relative to local climate, not absolute),
    then reports pooled paper-global NRMSE per bucket x Com/Res. Windows from
    buildings without weather are excluded (no temperature to bucket on)."""
    pct = pct or config.EXTREME_PCT
    has_t = np.isfinite(win["temp"])
    bucket = np.full(len(win["b"]), -1, dtype=np.int8)  # 0=cold 1=mild 2=hot
    for bi in np.unique(win["b"][has_t]):
        m = has_t & (win["b"] == bi)
        lo, hi = np.percentile(win["temp"][m], [pct, 100 - pct])
        bucket[m] = 1
        bucket[m & (win["temp"] <= lo)] = 0
        bucket[m & (win["temp"] >= hi)] = 2

    is_res_w = ds["is_res"][win["b"]]
    out = {}
    for bname, bval in [("cold", 0), ("mild", 1), ("hot", 2)]:
        for label, gmask in [("Com", ~is_res_w), ("Res", is_res_w)]:
            m = (bucket == bval) & gmask
            out[f"{label} {bname}"] = _pooled_nrmse(win["se"], win["ysum"], win["cnt"], m)
        out[f"n {bname}"] = int((bucket == bval).sum())
    return out


# ---------------------------------------------------------------------------
# 2) Autoregressive horizon rollout: 24h -> 48h -> 72h
# ---------------------------------------------------------------------------
@torch.no_grad()
def rollout_eval(predict_fn, ds: dict, dev: str, use_w: bool, load_transform,
                 steps: int = None, stride: int = 48, chunk: int = 512) -> dict:
    """Feeds the model's own 24h forecast back into the history window to
    forecast 24h more, `steps` times. Exogenous features for the shifted
    windows come from the cache (calendar/weather ARE known in advance --
    that's the point of the analysis: known-future weather should help more
    where copied-history information has decayed). Returns pooled NRMSE per
    horizon segment: {"h1-24": {"Com": ..., "Res": ...}, "h25-48": ..., ...}.
    Segment h1-24 reproduces the standard evaluation (sanity anchor)."""
    steps = steps or config.ROLLOUT_STEPS
    smax = ds["T"] - (config.L + 24 * steps)
    pairs = [(bi, s) for bi in range(ds["N"]) for s in range(0, smax, stride)]

    se = np.zeros((steps, ds["N"])); ysum = np.zeros((steps, ds["N"])); cnt = np.zeros((steps, ds["N"]))
    for i in range(0, len(pairs), chunk):
        ch = pairs[i:i + chunk]
        b = torch.tensor([p[0] for p in ch], device=dev)
        s0 = torch.tensor([p[1] for p in ch], device=dev)
        hist_kw = None
        for k in range(steps):
            yh_true, _, exo, yf_kw = gather(ds, b, s0 + 24 * k, use_w, load_transform, dev)
            if hist_kw is None:
                yh_bc = yh_true  # first step: true history (already Box-Cox from gather)
            else:
                yh_bc = torch.from_numpy(
                    load_transform.transform(np.clip(hist_kw, 1e-3, None))).to(dev).float()
            mu_kw = load_transform.undo_transform(predict_fn(yh_bc, exo))
            err = mu_kw - yf_kw.cpu().numpy()
            bnp = b.cpu().numpy()
            np.add.at(se[k], bnp, (err ** 2).sum(1))
            np.add.at(ysum[k], bnp, yf_kw.cpu().numpy().sum(1))
            np.add.at(cnt[k], bnp, config.H)
            # slide history forward 24h, replacing the newest day with our forecast
            prev_kw = (load_transform.undo_transform(yh_bc.cpu().numpy())
                       if hist_kw is None else hist_kw)
            hist_kw = np.concatenate([prev_kw[:, 24:], np.clip(mu_kw, 1e-3, None)], axis=1)

    out = {}
    for k in range(steps):
        seg = {}
        for label, gmask in [("Com", ~ds["is_res"]), ("Res", ds["is_res"])]:
            seg[label] = _pooled_nrmse(se[k], ysum[k], cnt[k], gmask & (cnt[k] > 0))
        out[f"h{24 * k + 1}-{24 * (k + 1)}"] = seg
    return out


# ---------------------------------------------------------------------------
# 3) Weather-sensitivity gating: per-channel knockout
# ---------------------------------------------------------------------------
@torch.no_grad()
def weather_sensitivity(predict_fn, ds: dict, dev: str, load_transform,
                        n_windows: int = 4096, seed: int = 0,
                        segment: str = "future") -> dict:
    """For a condition-B (weather-using) model: zero one weather channel at a
    time (0 = training mean in normalized space) over the `segment` ('future'
    = forecast-horizon hours only, the forecast-value question; 'all' = whole
    window) and measure (a) mean relative prediction movement %, (b) pooled
    NRMSE degradation vs the unperturbed baseline. Only buildings with a
    weather match are sampled."""
    rng = np.random.default_rng(seed)
    with_w = np.where(ds["b2w"] >= 0)[0]
    if len(with_w) == 0:
        return {"_base_nrmse": float("nan")}
    smax = ds["T"] - config.WIN
    b = torch.tensor(rng.choice(with_w, size=n_windows, replace=True), device=dev)
    s = torch.tensor(rng.integers(0, smax, size=len(b)), device=dev)

    yh, _, exo, yf_kw = gather(ds, b, s, True, load_transform, dev)
    yf_np = yf_kw.cpu().numpy()
    mu0 = load_transform.undo_transform(predict_fn(yh, exo))
    base_scale = np.abs(mu0).mean()
    cnt = np.full(len(yf_np), float(config.H)); ones = np.ones(len(yf_np), bool)
    err0 = mu0 - yf_np
    base_nrmse = _pooled_nrmse((err0 ** 2).sum(1), yf_np.sum(1), cnt, ones)

    t0 = config.L if segment == "future" else 0
    out = {"_base_nrmse": base_nrmse}
    for ci, col in enumerate(config.WEATHER_COLS):
        exo_p = exo.clone()
        exo_p[:, t0:, config.N_TIME + ci] = 0.0
        mu_c = load_transform.undo_transform(predict_fn(yh, exo_p))
        err_c = mu_c - yf_np
        out[col] = {
            "pred_delta_pct": float(100 * np.abs(mu_c - mu0).mean() / max(base_scale, 1e-9)),
            "delta_nrmse": float(_pooled_nrmse((err_c ** 2).sum(1), yf_np.sum(1), cnt, ones) - base_nrmse),
        }
    return out
