"""Shared train/eval loop: windowing (`gather`), resumable checkpointed
training, and evaluation on the simulated + real test sets. All result/
checkpoint paths come from a single `config.Paths` instance passed in by the
caller -- this is what fixes the old notebook's `PERBUILDING_REAL_DIR`
NameError class of bug (paths were re-declared ad hoc across multiple
notebook cells and could silently go out of scope between them).

Preprocessing note: the load history/target fed to every neural model is
Box-Cox-transformed (via the official, already-fit transform) before RevIN,
matching the official Buildings900K.__getitem__ exactly (`apply_scaler_
transform='boxcox'`). Point-forecast metrics (NRMSE/NMAE) are computed after
`undo_transform` back to physical kWh, matching how the paper always reports
them. CRPS uses the paper's own approximation for Box-Cox-trained Gaussian
heads (scripts/pretrain.py, apply_scaler_transform=='boxcox' branch): push
mu, mu+sigma, and mu-sigma through the inverse transform and rebuild an
approximate kWh-space Gaussian whose sigma is the average of the two
half-widths -- see `sigma_to_kw`. This makes CRPS directly comparable to the
paper's published numbers (in kWh).
"""
import math
import os
import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import config, metrics, models


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------
def gather(ds: dict, b: torch.Tensor, s: torch.Tensor, use_w: bool, load_transform, dev: str):
    """b, s: (nb,) int tensors -- building index and window-start hour.
    Returns (yh_bc, yf_bc, exo, yf_kw):
      yh_bc, yf_bc : (nb, L)/(nb, H) Box-Cox-transformed load (history/target)
      exo          : (nb, L+H, c_exo) = [calendar | weather? | building_type]
      yf_kw        : (nb, H) physical-kWh target (for point-metric evaluation)
    """
    L, H, WIN = config.L, config.H, config.WIN
    idx = s[:, None] + torch.arange(WIN, device=b.device)
    loads = ds["loads_t"]                      # (N, T) torch tensor, physical kWh
    win_kw = loads[b[:, None].expand(-1, WIN), idx]           # (nb, WIN)

    # Box-Cox requires strictly positive input; clip the rare non-positive
    # reading (e.g. real smart-meter net-metering export) rather than letting
    # the transform blow up to NaN/inf.
    win_kw_np = np.clip(win_kw.detach().cpu().numpy(), 1e-3, None)
    win_bc_np = load_transform.transform(win_kw_np)
    win_bc = torch.from_numpy(win_bc_np).to(dev).float()

    g_tf = ds["g_tf_t"]                                        # (n_release, T, n_time)
    xm = g_tf[ds["b2g_t"][b]].gather(1, idx.unsqueeze(-1).expand(-1, -1, ds["Ft"]))
    parts = [xm]
    if use_w:
        wuniq = ds["wuniq_t"]                                  # (n_wgroup, T, n_weather)
        b2w = ds["b2w_t"][b]
        valid = b2w >= 0
        wall = wuniq[b2w.clamp_min(0)]
        wgat = wall.gather(1, idx.unsqueeze(-1).expand(-1, -1, ds["Fw"]))
        wgat = wgat * valid.view(-1, 1, 1).float()  # buildings without a weather match get 0 (=train-mean, documented)
        parts.append(wgat)
    parts.append(ds["btype_t"][b].view(-1, 1, 1).expand(-1, WIN, 1))
    exo = torch.cat(parts, -1)

    return win_bc[:, :L], win_bc[:, L:], exo, win_kw[:, L:]


def to_device_cache(ds: dict, dev: str) -> dict:
    """Moves a data.py cache dict to torch tensors on `dev`, once, so
    `gather` doesn't repeatedly convert numpy->torch every call."""
    ds = dict(ds)
    ds["loads_t"] = torch.tensor(ds["loads"], dtype=torch.float32, device=dev)
    ds["g_tf_t"] = torch.tensor(ds["g_tf"], dtype=torch.float32, device=dev)
    ds["wuniq_t"] = torch.tensor(ds["wuniq"], dtype=torch.float32, device=dev) if ds["wuniq"].size else \
        torch.zeros(1, ds["T"], config.N_WEATHER, dtype=torch.float32, device=dev)
    ds["b2g_t"] = torch.tensor(ds["b2g"], dtype=torch.long, device=dev)
    ds["b2w_t"] = torch.tensor(ds["b2w"], dtype=torch.long, device=dev)
    ds["btype_t"] = torch.tensor(ds["btype"], dtype=torch.float32, device=dev)
    if "is_amy" in ds:
        ds["is_amy_t"] = torch.tensor(ds["is_amy"], dtype=torch.bool, device=dev)
    return ds


def gaussian_nll(mu, raw, target):
    sigma = F.softplus(raw) + 1e-3
    return (0.5 * math.log(2 * math.pi) + torch.log(sigma) + 0.5 * ((target - mu) / sigma) ** 2).mean()


def sigma_to_kw(load_transform, mu_bc: np.ndarray, sigma_bc: np.ndarray):
    """The paper's approximate kWh-space Gaussian for a Box-Cox-space Gaussian
    (ported from scripts/pretrain.py, apply_scaler_transform=='boxcox'):
        mu_kw       = undo(mu)
        sigma_upper = undo(mu + sigma) - mu_kw
        sigma_lower = mu_kw - undo(mu - sigma)
        sigma_kw    = (sigma_upper + sigma_lower) / 2
    Returns (mu_kw, sigma_kw), both numpy. Non-finite sigmas (inverse Box-Cox
    leaving its domain in the far tail) degrade to ~0, where Gaussian CRPS
    smoothly reduces to absolute error rather than poisoning the accumulator."""
    mu_kw = load_transform.undo_transform(mu_bc)
    upper = load_transform.undo_transform(mu_bc + sigma_bc)
    lower = load_transform.undo_transform(mu_bc - sigma_bc)
    sigma_kw = ((upper - mu_kw) + (mu_kw - lower)) / 2
    sigma_kw = np.clip(np.nan_to_num(sigma_kw, nan=0.0, posinf=0.0, neginf=0.0), 1e-6, None)
    return mu_kw, sigma_kw


def _forward(model, yh, exo, yf):
    """A handful of models (the paper's own Transformer-S/M/L) are trained
    with teacher forcing -- they need the ground-truth target during
    training/validation, unlike every other model in this registry, which
    only ever sees (yh, exo). Flagged via a `USES_TEACHER_FORCING` class
    attribute so this is the only place that needs to know about it."""
    if getattr(model, "USES_TEACHER_FORCING", False):
        return model(yh, exo, yf=yf)
    return model(yh, exo)


# ---------------------------------------------------------------------------
# Training -- validation matches the official BuildingsBench split: a
# temporal holdout of the last config.VAL_HOLDOUT_HOURS (336h = 2 weeks) of
# the year, on the amy2018-release buildings only (a real chronological year,
# unlike the tmy3 releases' synthetic "typical year" splice -- see
# create_index_files.py's val_timerange/train_amy2018_timerange). tmy3-release
# buildings use the full year for training and never contribute to
# validation, matching the paper exactly. This also fixes the earlier
# single-fixed-validation-window bug: within the holdout range, samples
# config.N_VAL_WINDOWS_PER_BUILDING fresh window positions per building every
# epoch instead of one fixed window.
# ---------------------------------------------------------------------------
def train(model, ds: dict, dev: str, use_w: bool, load_transform, epochs=None, patience=None,
          steps=None, bs=None, lr=None, amp=True, amp_dtype=torch.bfloat16,
          clip=1.0, warmup_frac=0.03, log=False):
    epochs = epochs or config.EPOCHS
    patience = patience if patience is not None else config.PATIENCE
    steps = steps or config.STEPS
    bs = bs or config.BS
    lr = lr or config.LR

    try:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, fused=(dev == "cuda"))
    except TypeError:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    T, hold = ds["T"], config.VAL_HOLDOUT_HOURS
    is_amy = ds["is_amy_t"]
    val_b = torch.nonzero(is_amy, as_tuple=True)[0]
    assert val_b.numel() > 0, "no amy2018-release buildings in this cache -- can't build the official validation split"
    # per-building training-window-start upper bound: amy2018 buildings stop
    # short of the holdout range so no training window ever overlaps it;
    # tmy3 buildings use the full year (matches train_tmy_timerange).
    smax_train = torch.where(is_amy, T - hold - config.WIN, T - config.WIN).float()
    val_lo, val_hi = T - hold, T - config.WIN  # shared window-start range within the holdout

    tot = epochs * steps
    warm = max(50, int(tot * warmup_frac))
    gstep = 0

    def setlr(g):
        f = g / warm if g < warm else 0.5 * (1 + math.cos(math.pi * (g - warm) / max(1, tot - warm)))
        for pg in opt.param_groups:
            pg["lr"] = lr * f

    ac = lambda: torch.autocast("cuda", dtype=amp_dtype, enabled=(amp and dev == "cuda"))
    best, bstate, bad = 1e9, None, 0
    t_start, ep = time.time(), -1

    for ep in range(epochs):
        model.train()
        for _ in range(steps):
            setlr(gstep)
            gstep += 1
            b = torch.randint(0, ds["N"], (bs,), device=dev)
            s = (torch.rand(bs, device=dev) * smax_train[b]).long()
            yh, yf, exo, _ = gather(ds, b, s, use_w, load_transform, dev)
            with ac():
                mu_n, rs, mean, sd = _forward(model, yh, exo, yf)
            tn = (yf - mean.float()) / sd.float()
            loss = gaussian_nll(mu_n.float(), rs.float(), tn)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()

        model.eval()
        vs = []
        with torch.no_grad():
            for _ in range(config.N_VAL_WINDOWS_PER_BUILDING):
                s = torch.randint(val_lo, val_hi + 1, (len(val_b),), device=dev)
                for v0 in range(0, len(val_b), 128):
                    b = val_b[v0:v0 + 128]
                    sv = s[v0:v0 + 128]
                    yh, yf, exo, _ = gather(ds, b, sv, use_w, load_transform, dev)
                    with ac():
                        mu_n, rs, mean, sd = _forward(model, yh, exo, yf)
                    vs.append(gaussian_nll(mu_n.float(), rs.float(), (yf - mean.float()) / sd.float()).item())
        v = float(np.mean(vs))
        if v < best - 1e-4:
            best, bstate, bad = v, {k: t.detach().clone() for k, t in model.state_dict().items()}, 0
        else:
            bad += 1
        if log and ep % 10 == 0:
            print(f"    ep{ep:03d} val={v:.4f} best={best:.4f} bad={bad}")
        if bad >= patience:
            if log:
                print(f"    early stop @ep{ep}")
            break

    if bstate:
        model.load_state_dict(bstate)
    # compute-budget accounting, persisted by run_training_sweep to the
    # results CSV on Drive (epochs actually run incl. early stop, wall time)
    stats = {"epochs": ep + 1, "train_sec": round(time.time() - t_start, 1)}
    return best, stats


# ---------------------------------------------------------------------------
# Simulated-test evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, ds: dict, dev: str, use_w: bool, load_transform, stride=24, chunk=1024,
             amp=True, amp_dtype=torch.bfloat16):
    model.eval()
    ac = lambda: torch.autocast("cuda", dtype=amp_dtype, enabled=(amp and dev == "cuda"))
    smax = ds["T"] - config.WIN
    N = ds["N"]
    starts = list(range(0, smax, stride))
    pairs = [(bi, s) for bi in range(N) for s in starts]

    acc = metrics.BuildingAccumulator(N)
    for i in range(0, len(pairs), chunk):
        ch = pairs[i:i + chunk]
        b = torch.tensor([p[0] for p in ch], device=dev)
        s = torch.tensor([p[1] for p in ch], device=dev)
        yh, yf_bc, exo, yf_kw = gather(ds, b, s, use_w, load_transform, dev)
        with ac():
            mu_n, rs, mean, sd = model(yh, exo)
        mu_n, mean, sd = mu_n.float(), mean.float(), sd.float()
        mu_bc = (mu_n * sd + mean).cpu().numpy()
        # rs (raw pre-softplus scale) is None for CopyLastDay/CopyLastWeekPersistence,
        # matching the official model's predict() -> (forecast, None) -- no CRPS for those.
        if rs is not None:
            sigma_bc = ((F.softplus(rs.float()) + 1e-3) * sd).cpu().numpy()
            mu_kw, sigma_np = sigma_to_kw(load_transform, mu_bc, sigma_bc)  # paper's kWh approximation
        else:
            mu_kw, sigma_np = load_transform.undo_transform(mu_bc), None
        acc.update(b.cpu().numpy(), yf_kw.cpu().numpy(), mu_kw, sigma=sigma_np)
    res = acc.finalize()
    out = metrics.summarize(res, ds["is_res"])
    out.update({"_nrmse": res["nrmse"], "_nmae": res["nmae"], "_crps": res["crps"], "_is_res": ds["is_res"]})
    return out


# ---------------------------------------------------------------------------
# Real-building evaluation (variable-length windowing; no fixed 8760 grid)
# ---------------------------------------------------------------------------
def _real_window_batches(buildings: list, weather_cache, weather_transforms, n_time, n_weather,
                          stride, max_windows_per_building, batch):
    """Shared window-assembly generator for both neural (`evaluate_real`) and
    tree (`evaluate_real_tree`) real-building evaluation. Yields
    (bidx: list[int], yh_kw, exo, yf_kw) numpy arrays per batch. If
    `weather_cache` is None, exo = [calendar | building_type] (no-weather
    condition -- the headline real-building table, covers every real
    building). If provided (a dict {building_id: DataFrame indexed by
    timestamp, columns a subset of config.WEATHER_COLS}), exo includes a
    weather block: available channels are reindexed to each window's exact
    timestamps and normalized with the OFFICIAL per-channel scaler
    (`weather_transforms`, required whenever `weather_cache` is given);
    channels absent from the cache stay 0.0 in normalized space (=
    training-set average -- the same, explicitly documented approximation
    the paper's own real-eval makes for channels it doesn't have). Buildings
    with no entry in `weather_cache` are skipped entirely in the +weather
    condition (they still appear in the no-weather condition)."""
    import pandas as pd
    from . import data as data_mod

    use_w = weather_cache is not None
    if use_w:
        assert weather_transforms is not None, "weather_transforms required when weather_cache is given"
    L, WIN = config.L, config.WIN

    jobs = []
    for bi, bdg in enumerate(buildings):
        n = len(bdg["series"])
        if n < WIN:
            continue
        starts = list(range(0, n - WIN, stride))
        if len(starts) > max_windows_per_building:
            rng = np.random.default_rng(bi)
            starts = sorted(rng.choice(starts, max_windows_per_building, replace=False).tolist())
        jobs += [(bi, s) for s in starts]

    for i in range(0, len(jobs), batch):
        chunk = jobs[i:i + batch]
        yh_list, exo_list, yf_list, bidx = [], [], [], []
        for bi, s in chunk:
            bdg = buildings[bi]
            ser = bdg["series"]
            win_kw = ser.values[s:s + WIN].astype(np.float32)
            idx = ser.index[s:s + WIN]
            cal = data_mod.calendar_features(idx)
            btype_col = np.full((WIN, 1), bdg["building_type"], dtype=np.float32)
            parts = [cal]
            if use_w:
                wdf = weather_cache.get(bdg["building_id"])
                if wdf is None:
                    continue
                wsub = wdf.reindex(idx, method="nearest", tolerance=pd.Timedelta("2h")).interpolate().bfill().ffill()
                wblock = np.zeros((WIN, n_weather), dtype=np.float32)
                for ci, col in enumerate(config.WEATHER_COLS):
                    if col in wsub.columns:
                        raw_col = wsub[col].values.astype(np.float32)
                        wblock[:, ci] = weather_transforms[col].transform(raw_col.reshape(-1, 1)).numpy().reshape(-1)
                    # else: stays 0.0 = training-set mean in normalized space (documented above)
                parts.append(wblock)
            parts.append(btype_col)
            exo = np.concatenate(parts, 1)
            yh_list.append(win_kw[:L])
            yf_list.append(win_kw[L:])
            exo_list.append(exo)
            bidx.append(bi)
        if yh_list:
            yield bidx, np.stack(yh_list), np.stack(exo_list), np.stack(yf_list)


def _real_summary(acc, buildings, matched_bidx=None):
    res = acc.finalize()
    is_res = np.array([b["building_type"] < 0 for b in buildings])
    out = metrics.summarize(res, is_res)
    out["n_buildings"] = int(res["valid"].sum())
    if matched_bidx is not None:
        out["n_weather_matched"] = len(matched_bidx)
    out.update({"_nrmse": res["nrmse"], "_nmae": res["nmae"], "_crps": res["crps"], "_is_res": is_res})
    return out


@torch.no_grad()
def evaluate_real(model, buildings: list, dev: str, load_transform, weather_cache=None,
                   weather_transforms=None, n_time=None, n_weather=None, stride=24,
                   max_windows_per_building=90, batch=2048, amp=True, amp_dtype=torch.bfloat16):
    n_time = n_time or config.N_TIME
    n_weather = n_weather or config.N_WEATHER
    model.eval()
    ac = lambda: torch.autocast("cuda", dtype=amp_dtype, enabled=(amp and dev == "cuda"))

    acc = metrics.BuildingAccumulator(len(buildings))
    matched = set()
    for bidx, yh_kw_np, exo_np, yf_kw_np in _real_window_batches(
            buildings, weather_cache, weather_transforms, n_time, n_weather, stride, max_windows_per_building, batch):
        matched.update(bidx)
        yh_kw = torch.tensor(yh_kw_np, device=dev)
        yf_kw = torch.tensor(yf_kw_np, device=dev)
        exo = torch.tensor(exo_np, device=dev)
        yh_bc = torch.from_numpy(load_transform.transform(np.clip(yh_kw_np, 1e-3, None))).to(dev).float()

        with ac():
            mu_n, rs, mean, sd = model(yh_bc, exo)
        mu_n, mean, sd = mu_n.float(), mean.float(), sd.float()
        mu_bc = (mu_n * sd + mean).cpu().numpy()
        if rs is not None:
            sigma_bc = ((F.softplus(rs.float()) + 1e-3) * sd).cpu().numpy()
            mu_kw, sigma_np = sigma_to_kw(load_transform, mu_bc, sigma_bc)
        else:
            mu_kw, sigma_np = load_transform.undo_transform(mu_bc), None

        acc.update(np.array(bidx), yf_kw.cpu().numpy(), mu_kw, sigma=sigma_np)

    return _real_summary(acc, buildings, matched if weather_cache is not None else None)


def evaluate_real_tree(mdl, sigma: np.ndarray, buildings: list, load_transform, weather_cache=None,
                        weather_transforms=None, n_time=None, n_weather=None, stride=24,
                        max_windows_per_building=90, batch=2048):
    """XGBoost/LightGBM counterpart to `evaluate_real` (trees aren't
    nn.Module, so they don't go through the (yh_bc, exo) -> model(...) path)."""
    n_time = n_time or config.N_TIME
    n_weather = n_weather or config.N_WEATHER

    acc = metrics.BuildingAccumulator(len(buildings))
    matched = set()
    for bidx, yh_kw_np, exo_np, yf_kw_np in _real_window_batches(
            buildings, weather_cache, weather_transforms, n_time, n_weather, stride, max_windows_per_building, batch):
        matched.update(bidx)
        yh_bc_np = load_transform.transform(np.clip(yh_kw_np, 1e-3, None))
        yh_bc = torch.from_numpy(yh_bc_np).float()
        exo = torch.from_numpy(exo_np).float()
        X, mu, sd = models._tree_xy(yh_bc, exo, config.L)
        pred_bc = models.tree_predict(mdl, X) * sd[:, None] + mu[:, None]
        sigma_bc = np.broadcast_to(sigma[None, :], pred_bc.shape) * sd[:, None]
        pred_kw, sigma_kw = sigma_to_kw(load_transform, pred_bc, sigma_bc)
        acc.update(np.array(bidx), yf_kw_np, pred_kw, sigma=sigma_kw)

    return _real_summary(acc, buildings, matched if weather_cache is not None else None)


# ---------------------------------------------------------------------------
# Run orchestration -- ALL paths come from `paths` (config.Paths), constructed
# once by the caller and passed through everywhere. This is what fixes the
# old notebook's PERBUILDING_REAL_DIR NameError: there is exactly one place
# these paths are defined, and every consumer receives it as an argument
# instead of re-declaring module-level globals across cells.
# ---------------------------------------------------------------------------
def run_training_sweep(model_names: list, conditions: dict, ds_train: dict, ds_sim: dict,
                        dev: str, load_transform, paths: config.Paths, reset: bool = False,
                        log: bool = True):
    import csv
    import pandas as pd

    # v2 schema: paper-global headline metrics + per-building medians (＿med)
    # + compute-budget accounting (epochs, train/eval wall time, GPU, peak
    # memory). Written to paths.SIM_CSV (sim_results_v2.csv): a NEW file, so
    # pairs evaluated under the old metric definitions re-evaluate -- but
    # training is NEVER repeated: any existing checkpoint on Drive
    # ({model}_{cond}.pt / .pkl) is loaded instead of retraining, so
    # already-trained models only pay the ~15s re-scoring cost.
    fields = ["model", "condition", "weather",
              "com_nrmse", "res_nrmse", "com_nmae", "res_nmae", "com_crps", "res_crps",
              "com_nrmse_med", "res_nrmse_med",
              "val_nll", "params", "epochs", "train_sec", "eval_sec", "sec",
              "gpu", "peak_mem_gb", "reused_ckpt"]
    if reset and os.path.exists(paths.SIM_CSV):
        os.remove(paths.SIM_CSV)
    if not os.path.exists(paths.SIM_CSV):
        csv.DictWriter(open(paths.SIM_CSV, "w", newline=""), fields).writeheader()
    done = set()
    if os.path.getsize(paths.SIM_CSV) > 50:
        d = pd.read_csv(paths.SIM_CSV)
        # migration guard: rows written before the >=7M transformer floor
        # carry the old (smaller) architecture's results -- drop them so those
        # models re-run at the current size instead of being skipped forever
        stale = (d["model"].isin(models.TRANSFORMER_FAMILY)
                 & (d["params"] > 0) & (d["params"] < models.MIN_TRANSFORMER_PARAMS))
        if stale.any():
            if log:
                print(f"dropping {int(stale.sum())} stale rows (pre-7M-floor transformer sizes) -- will re-run")
            d = d[~stale]
            d.to_csv(paths.SIM_CSV, index=False)
        done = {(r.model, r.condition) for _, r in d.iterrows()}

    gpu_name = torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"

    run_list = [(m, c) for m in model_names for c in conditions]
    for name, cond in run_list:
        if (name, cond) in done:
            if log:
                print(f"skip {name}/{cond}")
            continue
        use_w = conditions[cond]
        t0 = time.time()
        tstats = {"epochs": 0, "train_sec": 0.0}  # stays zero when a checkpoint is reused
        reused = False
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        try:
            if name in ("xgboost", "lightgbm"):
                ckpt = f"{paths.WEIGHTS_DIR}/{name}_{cond}.pkl"
                if os.path.exists(ckpt):
                    mdl, sigma = pickle.load(open(ckpt, "rb"))
                    reused = True
                else:
                    if log:
                        print(f"fit {name}/{cond}...", flush=True)
                    t_fit = time.time()
                    gather_fn = lambda ds, b, s, use_w: gather(ds, b, s, use_w, load_transform, dev)[:3]
                    mdl, sigma = models.tree_fit(name, gather_fn, ds_train, dev, use_w, config.L, config.H,
                                                  n_windows=config.XGB_WINDOWS if name == "xgboost" else config.LGB_WINDOWS,
                                                  n_estimators=config.XGB_TREES if name == "xgboost" else config.LGB_TREES,
                                                  max_depth=config.XGB_DEPTH if name == "xgboost" else config.LGB_DEPTH,
                                                  hours=ds_train["T"])
                    pickle.dump((mdl, sigma), open(ckpt, "wb"))
                    tstats["train_sec"] = round(time.time() - t_fit, 1)
                params = 0
                t_eval = time.time()
                res = _eval_tree(mdl, sigma, ds_sim, dev, use_w, load_transform)
                val = float("nan")
            else:
                amp_this = config.USE_AMP and (name not in config.RNN_MODELS)
                torch.manual_seed(config.SEED)
                base = models.build(name, L=config.L, H=config.H, n_time=ds_train["Ft"], n_weather=ds_train["Fw"],
                                     use_weather=use_w, **models.MODEL_KW.get(name, {})).to(dev)
                params = models.count_params(base)
                ckpt = f"{paths.WEIGHTS_DIR}/{name}_{cond}.pt"
                if params > 0 and os.path.exists(ckpt):
                    # A checkpoint saved under an older hyperparameter config
                    # can't be loaded into a resized architecture -- detect the
                    # shape mismatch and retrain instead of crashing (the new
                    # weights then overwrite the stale checkpoint).
                    try:
                        base.load_state_dict(torch.load(ckpt, map_location=dev))
                        reused = True
                    except RuntimeError:
                        if log:
                            print(f"  {name}/{cond}: checkpoint is from an older architecture size -- retraining", flush=True)
                if reused:
                    val = float("nan")
                    if log:
                        print(f"reuse checkpoint {name}/{cond} (no retraining)", flush=True)
                elif params == 0:
                    val = float("nan")
                else:
                    if log:
                        print(f"train {name}/{cond} ({params:,}p, amp={amp_this})...", flush=True)
                    val, tstats = train(base, ds_train, dev, use_w, load_transform, amp=amp_this, log=log)
                    torch.save(base.state_dict(), ckpt)
                t_eval = time.time()
                res = evaluate(base, ds_sim, dev, use_w, load_transform, amp=amp_this)

            os.makedirs(paths.PERBUILDING_SIM_DIR, exist_ok=True)
            pickle.dump({k: res[k] for k in ("_nrmse", "_nmae", "_crps", "_is_res")},
                        open(f"{paths.PERBUILDING_SIM_DIR}/{name}_{cond}.pkl", "wb"))
            peak_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2) if dev == "cuda" else 0.0
            row = {"model": name, "condition": cond, "weather": use_w,
                   "com_nrmse": round(res.get("Com NRMSE", float("nan")), 3),
                   "res_nrmse": round(res.get("Res NRMSE", float("nan")), 3),
                   "com_nmae": round(res.get("Com NMAE", float("nan")), 3),
                   "res_nmae": round(res.get("Res NMAE", float("nan")), 3),
                   "com_crps": round(res.get("Com CRPS", float("nan")), 3),
                   "res_crps": round(res.get("Res CRPS", float("nan")), 3),
                   "com_nrmse_med": round(res.get("Com NRMSE med", float("nan")), 3),
                   "res_nrmse_med": round(res.get("Res NRMSE med", float("nan")), 3),
                   "val_nll": (round(val, 4) if val == val else ""), "params": params,
                   "epochs": tstats["epochs"], "train_sec": tstats["train_sec"],
                   "eval_sec": round(time.time() - t_eval, 1), "sec": round(time.time() - t0),
                   "gpu": gpu_name, "peak_mem_gb": peak_gb, "reused_ckpt": reused}
            csv.DictWriter(open(paths.SIM_CSV, "a", newline=""), fields).writerow(row)
            if log:
                print(f"  done {row['sec']}s  Com NRMSE={row['com_nrmse']}  Res NRMSE={row['res_nrmse']}"
                      f"  (med {row['com_nrmse_med']}/{row['res_nrmse_med']})\n")
        except Exception as e:
            print(f"  FAILED {name}/{cond}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    print("DONE ->", paths.SIM_CSV)


def _eval_tree(mdl, sigma, ds: dict, dev: str, use_w: bool, load_transform, stride=24, chunk=8192):
    smax = ds["T"] - config.WIN
    N = ds["N"]
    starts = list(range(0, smax, stride))
    pairs = [(bi, s) for bi in range(N) for s in starts]
    acc = metrics.BuildingAccumulator(N)
    for i in range(0, len(pairs), chunk):
        ch = pairs[i:i + chunk]
        b = torch.tensor([p[0] for p in ch], device=dev)
        s = torch.tensor([p[1] for p in ch], device=dev)
        yh, yf_bc, exo, yf_kw = gather(ds, b, s, use_w, load_transform, dev)
        X, mu, sd = models._tree_xy(yh, exo, config.L)
        pred_bc = models.tree_predict(mdl, X) * sd[:, None] + mu[:, None]
        sigma_bc = np.broadcast_to(sigma[None, :], pred_bc.shape) * sd[:, None]
        pred_kw, sigma_kw = sigma_to_kw(load_transform, pred_bc, sigma_bc)
        acc.update(b.cpu().numpy(), yf_kw.cpu().numpy(), pred_kw, sigma=sigma_kw)
    res = acc.finalize()
    out = metrics.summarize(res, ds["is_res"])
    out.update({"_nrmse": res["nrmse"], "_nmae": res["nmae"], "_crps": res["crps"], "_is_res": ds["is_res"]})
    return out
