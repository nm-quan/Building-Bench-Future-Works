"""BuildingsBench-faithful data pipeline: sourcing, transforms, and windowing
for (1) a compute-bounded subsample of the official Buildings-900K pretraining
corpus, (2) the official Buildings-900K-test simulated test set, and (3) the
7 real-building datasets used for zero-shot real-world evaluation.

Everything here reads directly from the public, unauthenticated
`s3://oedi-data-lake/buildings-bench` bucket (the same bucket BuildingsBench's
own README points to) and applies the vendored official transforms
(`bb_transforms.py`) -- so preprocessing is provably identical to the paper's,
not a reimplementation of unknown fidelity.

This module also fixes two bugs from the earlier draft of this study:
  - calendar features are computed by ONE function (`calendar_features`)
    called identically by every training/eval path, so the "normalized in
    training, forgotten in real-eval" bug class cannot recur.
  - buildings whose load series is fully non-finite after the Box-Cox
    undo/redo round trip are dropped (logged), not silently zero-filled.
"""
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .bb_transforms import BoxCoxTransform, StandardScalerTransform, TimestampTransform

S3_BUCKET = "oedi-data-lake"
S3_PREFIX = "buildings-bench/v2.0.0/BuildingsBench"
S3_BASE = f"s3://{S3_BUCKET}/{S3_PREFIX}"

RELEASES = [
    "comstock_tmy3_release_1", "resstock_tmy3_release_1",
    "comstock_amy2018_release_1", "resstock_amy2018_release_1",
]
REGIONS = ["by_puma_midwest", "by_puma_south", "by_puma_northeast", "by_puma_west"]

REAL_DATASETS = ["BDG-2", "Borealis", "Electricity", "IDEAL", "LCL", "SMART", "Sceaux"]
BDG2_SITES = ["Bear", "Fox", "Panther", "Rat"]


# ---------------------------------------------------------------------------
# Low-level S3 helpers (public bucket, --no-sign-request, no credentials needed)
# ---------------------------------------------------------------------------
def _s3_cp(key: str, local_path: str, quiet: bool = True) -> None:
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if os.path.exists(local_path):
        return
    cmd = ["aws", "s3", "cp", f"s3://{S3_BUCKET}/{key}", local_path, "--no-sign-request"]
    subprocess.run(cmd, check=True, capture_output=quiet)


def _s3_ls(key_prefix: str) -> list:
    cmd = ["aws", "s3", "ls", f"s3://{S3_BUCKET}/{key_prefix}", "--no-sign-request"]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    return [ln.split()[-1] for ln in out.splitlines() if ln.strip()]


def _s3_ls_recursive(key_prefix: str) -> list:
    cmd = ["aws", "s3", "ls", f"s3://{S3_BUCKET}/{key_prefix}", "--no-sign-request", "--recursive"]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    return [ln.split()[-1] for ln in out.splitlines() if ln.strip()]


def _first_parquet_key(dir_key: str) -> str:
    """dir_key: e.g. '.../puma=G17000104/' (trailing slash). Returns the full
    key of the single data part-file inside it (ignores _SUCCESS/.crc)."""
    names = _s3_ls(dir_key)
    hits = [n for n in names if n.endswith(".parquet")]
    if not hits:
        raise FileNotFoundError(f"no parquet file under {dir_key}")
    return dir_key + hits[0]


# ---------------------------------------------------------------------------
# Official transforms (downloaded once, cached locally)
# ---------------------------------------------------------------------------
def _validate_boxcox(load_transform) -> None:
    """The official boxcox.pkl is a pickled sklearn PowerTransformer fit under
    sklearn 1.5.x; unpickling under a different sklearn version emits an
    InconsistentVersionWarning and COULD silently misbehave. Rather than
    pinning sklearn (which would force a Colab downgrade + runtime restart),
    validate numerically: the fitted lambda must exist and be finite, and a
    kWh probe spanning the realistic load range must round-trip
    transform -> undo_transform back to itself."""
    pt = load_transform.boxcox
    lambdas = getattr(pt, "lambdas_", None)
    assert lambdas is not None and np.isfinite(lambdas).all(), \
        f"official boxcox.pkl loaded without valid lambdas_ ({lambdas}) -- sklearn unpickle failure"
    probe = np.array([0.01, 0.5, 1.0, 5.0, 25.0, 100.0, 1000.0], dtype=np.float64)
    rt = load_transform.undo_transform(load_transform.transform(probe))
    assert np.allclose(rt, probe, rtol=1e-4, atol=1e-6), \
        f"Box-Cox round-trip failed: {probe} -> {rt} -- sklearn version incompatibility"


def download_official_transforms(paths: config.Paths) -> dict:
    """Downloads the paper's own already-fit Box-Cox (load) and StandardScaler
    (per weather channel) transform parameters, and validates the Box-Cox
    pickle numerically (see _validate_boxcox). Returns
    {'load': BoxCoxTransform, 'weather': {col: StandardScalerTransform}}.
    """
    tdir = Path(paths.TRANSFORMS_DIR)
    tdir.mkdir(parents=True, exist_ok=True)

    boxcox_local = tdir / "boxcox.pkl"
    _s3_cp(f"{S3_PREFIX}/metadata/transforms/boxcox.pkl", str(boxcox_local))
    load_transform = BoxCoxTransform()
    load_transform.load(boxcox_local)
    _validate_boxcox(load_transform)

    weather_transforms = {}
    for col in config.WEATHER_COLS:
        local = tdir / "weather" / col / "standard_scaler.npy"
        _s3_cp(f"{S3_PREFIX}/metadata/transforms/weather/{col}/standard_scaler.npy", str(local))
        t = StandardScalerTransform()
        t.load(local.parent)
        weather_transforms[col] = t

    return {"load": load_transform, "weather": weather_transforms}


# ---------------------------------------------------------------------------
# Calendar features -- ONE function, used by every code path (fixes the
# unnormalized-real-eval-calendar bug: there is no second implementation to
# drift out of sync with this one).
# ---------------------------------------------------------------------------
def calendar_features(index: pd.DatetimeIndex) -> np.ndarray:
    """[day_of_year, day_of_week, hour_of_day] each linearly scaled to
    [-1, 1], matching buildings_bench.transforms.TimestampTransform exactly.
    Computed per-row by actual leap-year-ness (a faithful generalization of
    the official single-bool-per-file design, needed because real datasets
    can span multiple/leap years within one series)."""
    index = pd.DatetimeIndex(index)
    out = np.empty((len(index), 3), dtype=np.float32)
    leap_mask = index.is_leap_year
    for is_leap in (True, False):
        m = leap_mask == is_leap
        if not m.any():
            continue
        out[m] = TimestampTransform(is_leap_year=is_leap).transform(index[m])
    return out


# ---------------------------------------------------------------------------
# Official train/val index files (weekly-sampled (building, hour_ptr) pairs)
# ---------------------------------------------------------------------------
_INDEX_SIZES = [1000, 10000, 100000]  # official convenience subsample sizes; None = full


def _index_file_name(split: str, size: int = None) -> str:
    base = f"{split}_weekly"
    return f"{base}_{size}.idx" if size else f"{base}.idx"


def download_index_file(split: str, size: int, paths: config.Paths) -> Path:
    """split: 'train' or 'val'. size: smallest official size >= what's needed."""
    name = _index_file_name(split, size)
    local = Path(paths.RAW_CACHE_DIR) / "metadata" / name
    _s3_cp(f"{S3_PREFIX}/metadata/{name}", str(local), quiet=False)
    return local


def parse_index_file(path: Path) -> pd.DataFrame:
    """Index file columns: release, region, puma, building_id, hour_ptr
    (tab-separated, one line per weekly-sampled window). `release` and
    `region` are stored as small integers indexing into
    Buildings900K.building_type_and_year / .census_regions (confirmed by
    inspecting buildings_bench/data/buildings900K.py) -- decoded here via
    RELEASES/REGIONS, which use the identical order."""
    df = pd.read_csv(path, sep="\t", header=None,
                      names=["release", "region", "puma", "building_id", "hour_ptr"],
                      dtype={"release": int, "region": int, "puma": str,
                             "building_id": str, "hour_ptr": int})
    df["release"] = df["release"].map(lambda i: RELEASES[i])
    df["region"] = df["region"].map(lambda i: REGIONS[i])
    return df


def sample_buildings(index_df: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    """Deterministically subsample n unique (release, region, puma, building_id)
    buildings from a parsed index file. Returns one row per unique building
    (hour_ptr dropped -- windows are sampled fresh at train time)."""
    uniq = index_df.drop_duplicates(subset=["release", "region", "puma", "building_id"])
    uniq = uniq.sample(n=min(n, len(uniq)), random_state=seed).reset_index(drop=True)
    return uniq[["release", "region", "puma", "building_id"]]


def smallest_index_size(n_needed: int) -> int:
    for s in _INDEX_SIZES:
        if s >= n_needed:
            return s
    return None  # fall back to the full index file


# ---------------------------------------------------------------------------
# PUMA -> county lookup (for joining buildings to their weather file)
# ---------------------------------------------------------------------------
def load_puma_county_lookup(paths: config.Paths) -> pd.Series:
    local = Path(paths.RAW_CACHE_DIR) / "metadata" / "puma_county_lookup_weather_only.csv"
    _s3_cp(f"{S3_PREFIX}/metadata/puma_county_lookup_weather_only.csv", str(local))
    df = pd.read_csv(local)
    return df.set_index("nhgis_2010_puma_gisjoin")["nhgis_2010_county_gisjoin"]


# ---------------------------------------------------------------------------
# Buildings-900K (and Buildings-900K-test): fetch parquet + county weather,
# apply official transforms, assemble a windows-ready cache.
# ---------------------------------------------------------------------------
_WEATHER_CSV_COLS = {
    "date_time": "timestamp",
    "Dry Bulb Temperature [°C]": "temperature",
    "Relative Humidity [%]": "humidity",
    "Wind Speed [m/s]": "wind_speed",
    "Wind Direction [Deg]": "wind_direction",
    "Global Horizontal Radiation [W/m2]": "global_horizontal_radiation",
    "Direct Normal Radiation [W/m2]": "direct_normal_radiation",
    "Diffuse Horizontal Radiation [W/m2]": "diffuse_horizontal_radiation",
}


def _root_s3_path(root_prefix: str) -> str:
    """`Buildings-900K` (pretraining) and `Buildings-900K-test` use different
    S3 sub-paths -- confirmed by direct bucket listing: the pretraining root
    has an extra `end-use-load-profiles-for-us-building-stock` segment (this
    matches buildings_bench/data/buildings900K.py's own hardcoded dataset_path)
    that the test root does not."""
    if root_prefix == "Buildings-900K":
        return f"{S3_PREFIX}/Buildings-900K/end-use-load-profiles-for-us-building-stock/2021"
    return f"{S3_PREFIX}/{root_prefix}/2021"


def _fetch_puma_parquet(root_prefix: str, release: str, region: str, puma: str, paths: config.Paths) -> pd.DataFrame:
    local = Path(paths.RAW_CACHE_DIR) / root_prefix / release / region / f"{puma}.parquet"
    if not local.exists():
        dir_key = f"{_root_s3_path(root_prefix)}/{release}/timeseries_individual_buildings/{region}/upgrade=0/puma={puma}/"
        key = _first_parquet_key(dir_key)
        _s3_cp(key, str(local))
    df = pd.read_parquet(local)
    # CRITICAL: Buildings-900K parquet rows are NOT stored chronologically --
    # the official loader warns "The time series are not stored chronologically
    # and must be sorted by timestamp after loading" (buildings900K.py) and
    # sorts in its __getitem__. Verified empirically: raw row order is fully
    # shuffled (0.01% of consecutive rows are consecutive hours); sorted order
    # is a clean hourly series. Skipping this sort scrambles every series.
    df = df.sort_values("timestamp").reset_index(drop=True)
    ts = pd.to_datetime(df["timestamp"])
    assert ts.is_monotonic_increasing, f"parquet {puma} not chronological after sort"
    return df


def _fetch_county_weather(root_prefix: str, release: str, county: str, paths: config.Paths) -> pd.DataFrame:
    local = Path(paths.RAW_CACHE_DIR) / root_prefix / release / "weather" / f"{county}.csv"
    if not local.exists():
        _s3_cp(f"{_root_s3_path(root_prefix)}/{release}/weather/{county}.csv", str(local))
    df = pd.read_csv(local)
    df = df.rename(columns=_WEATHER_CSV_COLS)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.set_index("timestamp")[config.WEATHER_COLS].sort_index()


def build_dataset_cache(building_keys: pd.DataFrame, transforms: dict, paths: config.Paths,
                         cache_path: str, root_prefix: str = "Buildings-900K",
                         verbose: bool = True) -> dict:
    """building_keys: DataFrame with columns [release, region, puma, building_id]
    (one row per building, e.g. from `sample_buildings` or an enumeration of
    every building under Buildings-900K-test). Fetches/caches raw parquet +
    weather, applies the official StandardScaler transform to weather (Box-Cox
    for load is applied at windowing time in `train.py`, matching the official
    per-sample `__getitem__` behavior), computes calendar features, and writes
    a single .npz cache. Buildings whose raw load series is non-finite are
    dropped (logged), not silently zero-filled.

    Returns the same dict `load_dataset_cache` returns (loaded back in).
    """
    if os.path.exists(cache_path):
        if verbose:
            print(f"  [cache] found existing {cache_path}, loading")
        return load_dataset_cache(cache_path)

    puma_county = load_puma_county_lookup(paths)
    weather_t = transforms["weather"]  # load Box-Cox is applied later, at windowing time (train.py)

    # group buildings by (release, region, puma) to fetch each parquet once
    groups = building_keys.groupby(["release", "region", "puma"])

    # Pass 1: collect everything at its NATIVE length (building-year files can
    # differ by a few hours across releases/pumas -- e.g. leap years). We only
    # know the safe common length (T_ref) after seeing every group, so nothing
    # is truncated/stacked until pass 2.
    loads_native, keys_list = [], []          # per-building: native-length np.ndarray, key string
    release_of, county_of = [], []            # per-building
    release_tf_native = {}                    # release -> (native_T, 3)
    county_weather_native = {}                # (release, county) -> (native_T, n_weather) raw

    n_dropped_nonfinite = 0

    # Progress logging: this loop is dominated by reading (and on the first
    # run, downloading) one parquet per (release, region, puma) group --
    # through Colab's slow Drive FUSE mount this can take tens of minutes with
    # no other output, which looks like a hang without a heartbeat.
    import time as _time
    n_groups = groups.ngroups
    t0 = _time.time()
    for gi, ((release, region, puma), g) in enumerate(groups, 1):
        if verbose and (gi % 50 == 0 or gi == n_groups):
            el = _time.time() - t0
            eta = el / gi * (n_groups - gi)
            print(f"  [{gi}/{n_groups} pumas] {len(loads_native)} buildings so far, "
                  f"{el/60:.1f}min elapsed, ~{eta/60:.1f}min left", flush=True)
        try:
            pq_df = _fetch_puma_parquet(root_prefix, release, region, puma, paths)
        except Exception as e:
            if verbose:
                print(f"  [warn] skipping puma={puma} ({release}/{region}): {type(e).__name__}: {e}")
            continue

        ts = pd.to_datetime(pq_df["timestamp"])
        if release not in release_tf_native:
            release_tf_native[release] = calendar_features(ts)

        county = puma_county.get(puma, None)
        if county is not None and (release, county) not in county_weather_native:
            try:
                wdf = _fetch_county_weather(root_prefix, release, county, paths)
                wdf = wdf.reindex(ts.values, method="nearest")
                county_weather_native[(release, county)] = wdf[config.WEATHER_COLS].values.astype(np.float32)
            except Exception as e:
                if verbose:
                    print(f"  [warn] no weather for county={county} ({release}): {type(e).__name__}")
                county_weather_native[(release, county)] = None

        # The index file zero-pads building_id (e.g. "023646") but the parquet
        # column names do not (e.g. "23646") -- confirmed by direct inspection.
        # Match on the integer value, not the raw string.
        col_lookup = {str(int(c)): c for c in pq_df.columns if c != "timestamp"}

        for _, row in g.iterrows():
            bid_raw = row["building_id"]
            bid = col_lookup.get(str(int(bid_raw)))
            if bid is None:
                continue
            raw = pq_df[bid].values.astype(np.float32)
            # Buildings-900K parquet columns are raw physical kWh (confirmed by
            # inspection -- plausible kWh magnitudes, not standardized Box-Cox
            # output). The official dataloader applies load_transform.transform()
            # per-sample at __getitem__ time, not baked into the file; `train.py`
            # mirrors that by applying the Box-Cox transform at windowing time,
            # so this cache stores physical kWh, matching how the paper always
            # reports metrics in physical units (via undo_transform).
            if not np.isfinite(raw).all():
                n_dropped_nonfinite += 1
                continue
            loads_native.append(raw)
            keys_list.append(f"{release}__{region}__{puma}__{bid}")
            release_of.append(release)
            county_of.append(county)

    if not loads_native:
        raise RuntimeError("no buildings successfully loaded -- check S3 connectivity/keys")

    # Pass 2: T_ref = the minimum length across every array we're about to
    # stack (loads, per-release calendar features, per-county weather) --
    # guarantees np.stack never sees a length mismatch.
    all_lengths = ([len(a) for a in loads_native]
                    + [len(a) for a in release_tf_native.values()]
                    + [len(a) for a in county_weather_native.values() if a is not None])
    T_ref = min(all_lengths)

    loads = np.stack([a[:T_ref] for a in loads_native])
    N = loads.shape[0]
    is_res = np.array(["resstock" in r for r in release_of])
    btype = np.where(is_res, -1.0, 1.0).astype(np.float32)
    # amy2018 releases are a real chronological year (so a genuine last-2-weeks
    # holdout makes sense for validation); tmy3 releases are a synthetic
    # "typical meteorological year" splice with no true chronological tail --
    # confirmed from scripts/data_generation/create_index_files.py: val_idx_file
    # entries are only ever written for the 2 amy2018 releases, using
    # val_timerange=(Dec17,Dec31) i.e. the last 336h of the year; tmy3 releases
    # use the full year for train_idx_file and contribute no val entries at all.
    is_amy = np.array(["amy2018" in r for r in release_of])

    # release -> integer group id (shared calendar features per release)
    uniq_releases = sorted(release_tf_native.keys())
    rel2gid = {r: i for i, r in enumerate(uniq_releases)}
    b2g = np.array([rel2gid[r] for r in release_of], dtype=np.int64)
    g_tf = np.stack([release_tf_native[r][:T_ref] for r in uniq_releases]).astype(np.float32)

    # (release, county) -> weather group id
    uniq_wkeys = sorted(set((r, c) for r, c in zip(release_of, county_of) if county_weather_native.get((r, c)) is not None))
    w2gid = {k: i for i, k in enumerate(uniq_wkeys)}
    b2w = np.array([w2gid.get((r, c), -1) for r, c in zip(release_of, county_of)], dtype=np.int64)
    if len(uniq_wkeys):
        wuniq_raw = np.stack([county_weather_native[k][:T_ref] for k in uniq_wkeys]).astype(np.float32)
        # apply the OFFICIAL, globally-fit weather StandardScaler per channel
        wuniq = np.empty_like(wuniq_raw)
        for ci, col in enumerate(config.WEATHER_COLS):
            wuniq[..., ci] = weather_t[col].transform(wuniq_raw[..., ci]).numpy().reshape(wuniq_raw.shape[:-1])
    else:
        wuniq = np.zeros((0, T_ref, config.N_WEATHER), dtype=np.float32)

    if verbose:
        print(f"  [cache] {N} buildings, T={T_ref}h, dropped {n_dropped_nonfinite} non-finite series")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, loads=loads, g_tf=g_tf, wuniq=wuniq,
                         b2g=b2g, b2w=b2w, btype=btype, is_res=is_res, is_amy=is_amy, T=T_ref,
                         keys=np.array(keys_list))
    return load_dataset_cache(cache_path)


def load_dataset_cache(cache_path: str) -> dict:
    d = np.load(cache_path, allow_pickle=False)
    # is_amy was added after some caches were already written; backfill it from
    # the stored per-building keys ("{release}__{region}__{puma}__{bid}") so an
    # existing cache from an earlier run doesn't have to be rebuilt from S3.
    if "is_amy" in d.files:
        is_amy = d["is_amy"]
    else:
        is_amy = np.array(["amy2018" in str(k) for k in d["keys"]])
    return {
        "loads": d["loads"], "g_tf": d["g_tf"], "wuniq": d["wuniq"],
        "b2g": d["b2g"], "b2w": d["b2w"], "btype": d["btype"], "is_res": d["is_res"],
        "is_amy": is_amy,
        "T": int(d["T"]), "N": d["loads"].shape[0], "Ft": config.N_TIME, "Fw": config.N_WEATHER,
    }


def build_train_cache(paths: config.Paths, transforms: dict, n_buildings: int = None,
                       seed: int = None, verbose: bool = True) -> dict:
    n_buildings = n_buildings or config.N_TRAIN_BUILDINGS
    seed = seed if seed is not None else config.SEED
    size = smallest_index_size(n_buildings)
    idx_path = download_index_file("train", size, paths)
    index_df = parse_index_file(idx_path)
    building_keys = sample_buildings(index_df, n_buildings, seed=seed)
    return build_dataset_cache(building_keys, transforms, paths,
                                cache_path=f"{paths.TRAIN_DIR}.npz", root_prefix="Buildings-900K",
                                verbose=verbose)


def build_sim_test_cache(paths: config.Paths, transforms: dict, verbose: bool = True) -> dict:
    """Enumerate every building under Buildings-900K-test (the official,
    small, held-out simulated test set) -- no subsampling."""
    rows = []
    for release in RELEASES:
        for region in REGIONS:
            prefix = f"{_root_s3_path('Buildings-900K-test')}/{release}/timeseries_individual_buildings/{region}/upgrade=0/"
            try:
                names = _s3_ls(prefix)
            except Exception:
                continue
            for n in names:
                if not n.startswith("puma="):
                    continue
                puma = n[len("puma="):].rstrip("/")
                try:
                    pq_df = _fetch_puma_parquet("Buildings-900K-test", release, region, puma, paths)
                except Exception:
                    continue
                for bid in pq_df.columns:
                    if bid == "timestamp":
                        continue
                    rows.append((release, region, puma, bid))
    building_keys = pd.DataFrame(rows, columns=["release", "region", "puma", "building_id"])
    return build_dataset_cache(building_keys, transforms, paths,
                                cache_path=f"{paths.SIM_TEST_DIR}.npz", root_prefix="Buildings-900K-test",
                                verbose=verbose)


# ---------------------------------------------------------------------------
# Real-building datasets (7 sources) -- building_type comes from the
# official metadata/benchmark.toml (per-dataset, per-BDG-2-site), not guessed.
# ---------------------------------------------------------------------------
def load_benchmark_toml(paths: config.Paths) -> dict:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # py<3.11 fallback
    local = Path(paths.RAW_CACHE_DIR) / "metadata" / "benchmark.toml"
    _s3_cp(f"{S3_PREFIX}/metadata/benchmark.toml", str(local))
    with open(local, "rb") as f:
        return tomllib.load(f)["buildings_bench"]


def _building_type_for(dataset: str, site: str, meta: dict) -> float:
    key = dataset.lower()
    entry = meta.get(key, {})
    if "building_type" in entry:
        bt = entry["building_type"]
    else:
        bt = entry.get(site.lower(), {}).get("building_type", "residential")
    return 1.0 if bt == "commercial" else -1.0


# Published real-building CSVs are inconsistent about the timestamp column
# name (confirmed by direct header inspection): most use 'timestamp', LCL uses
# 'DateTime', and IDEAL leaves it unnamed as the leading index column. Detect
# it robustly instead of assuming 'timestamp'.
_TIME_COL_CANDIDATES = ("timestamp", "time", "datetime", "date_time", "date")


def _real_timestamp_col(df: pd.DataFrame):
    """Returns the name of the timestamp column in a real-building CSV, or None
    if none is identifiable."""
    for c in df.columns:
        if str(c).strip().lower() in _TIME_COL_CANDIDATES:
            return c
    first = df.columns[0]  # IDEAL: header ",power" -> leading column is 'Unnamed: 0'
    if str(first).startswith("Unnamed") or str(first).strip() == "":
        return first
    return None


def load_real_buildings(paths: config.Paths, datasets: list = None, verbose: bool = True) -> list:
    """Returns a list of dicts: {building_id, building_type (+1/-1), series
    (pd.Series, kW, DatetimeIndex)}. One entry per data column per CSV: most
    datasets are one-building-per-file, but BDG-2 and Electricity pack many
    buildings/meters as separate columns in a single file, so those expand to
    one entry each (taking only the first column silently dropped almost all
    of their buildings)."""
    datasets = datasets or REAL_DATASETS
    meta = load_benchmark_toml(paths)
    out = []
    for ds in datasets:
        try:
            names = _s3_ls(f"{S3_PREFIX}/{ds}/")
        except Exception as e:
            if verbose:
                print(f"  [warn] cannot list {ds}: {e}")
            continue
        csvs = [n for n in names if n.endswith(".csv") and "_clean=" in n]
        n_dropped_empty, n_added = 0, 0
        for name in csvs:
            local = Path(paths.RAW_CACHE_DIR) / "real" / ds / name
            _s3_cp(f"{S3_PREFIX}/{ds}/{name}", str(local))
            df = pd.read_csv(local)
            ts_col = _real_timestamp_col(df)
            data_cols = [c for c in df.columns if c != ts_col] if ts_col is not None else []
            # Some published building-year files have no usable time column or
            # no data column at all (a building-year with zero readings that
            # still got a file written). Skip rather than crash the whole load.
            if ts_col is None or not data_cols:
                n_dropped_empty += 1
                continue
            idx = pd.to_datetime(df[ts_col], errors="coerce")
            site = name.split("_clean=")[0]
            bt = _building_type_for(ds, site, meta)
            stem = name.split(".csv")[0]
            multi = len(data_cols) > 1
            for col in data_cols:
                vals = pd.to_numeric(df[col], errors="coerce").values
                series = pd.Series(vals, index=idx).sort_index()
                series = series[series.index.notna()]
                series = series[np.isfinite(series.values)]
                if series.empty:
                    continue
                bid = f"{ds}:{stem}:{col}" if multi else f"{ds}:{stem}"
                out.append({"building_id": bid, "building_type": bt, "series": series})
                n_added += 1
        if verbose:
            dropped_note = f", dropped {n_dropped_empty} empty/unparseable files" if n_dropped_empty else ""
            print(f"  [real] {ds}: {n_added} buildings from {len(csvs)} files{dropped_note}")
    return out


def _parse_era5_weather(local: Path):
    """Reads an era5 weather CSV into a DataFrame indexed by timestamp with
    ['temperature', 'humidity'] columns. Published era5 files are inconsistent
    about the time column name (some use 'timestamp', some 'time' -- confirmed
    by direct header inspection: Electricity/LCL use 'time', the rest use
    'timestamp'), so normalize it here. Returns None if the file lacks any
    usable time column or both weather channels."""
    df = pd.read_csv(local)
    ts_col = next((c for c in ("timestamp", "time") if c in df.columns), None)
    if ts_col is None:
        return None
    keep = [c for c in ("temperature", "humidity") if c in df.columns]
    if not keep:
        return None
    return df.set_index(pd.to_datetime(df[ts_col]))[keep].sort_index()


def load_real_weather_temp_humidity(paths: config.Paths, datasets: list = None, verbose: bool = True) -> dict:
    """Paper-faithful real-weather source: era5 temperature + humidity only,
    per dataset (per BDG-2 site). Returns {dataset: {"default": df}} or
    {"BDG-2": {site: df, ...}}."""
    datasets = datasets or REAL_DATASETS
    out = {}
    for ds in datasets:
        if ds == "BDG-2":
            out[ds] = {}
            for site in BDG2_SITES:
                local = Path(paths.RAW_CACHE_DIR) / "real" / ds / f"weather_{site}_era5.csv"
                try:
                    _s3_cp(f"{S3_PREFIX}/{ds}/weather_{site}_era5.csv", str(local))
                except Exception:
                    continue
                wdf = _parse_era5_weather(local)
                if wdf is not None:
                    out[ds][site] = wdf
                elif verbose:
                    print(f"  [warn] unusable weather columns in {ds}/weather_{site}_era5.csv")
        else:
            local = Path(paths.RAW_CACHE_DIR) / "real" / ds / "weather_era5.csv"
            try:
                _s3_cp(f"{S3_PREFIX}/{ds}/weather_era5.csv", str(local))
            except Exception:
                if verbose:
                    print(f"  [warn] no weather_era5.csv for {ds}")
                continue
            wdf = _parse_era5_weather(local)
            if wdf is not None:
                out[ds] = {"default": wdf}
            elif verbose:
                print(f"  [warn] unusable weather columns in {ds}/weather_era5.csv")
    return out


def weather_for_building(building_id: str, wcache: dict):
    """building_id: 'DatasetName:filename' (e.g. 'BDG-2:Bear_clean=2016' or
    'Sceaux:Sceaux_clean=2007'). wcache: output of
    load_real_weather_temp_humidity. Returns the matching DataFrame, or None
    if that building's dataset/site has no weather available -- callers
    (weather_real.py) simply omit such buildings from the +weather track;
    they still appear in the no-weather real-eval track, which covers every
    real building regardless of weather availability."""
    dataset, rest = building_id.split(":", 1)
    if dataset == "BDG-2":
        site = rest.split("_")[0]
        return wcache.get(dataset, {}).get(site)
    return wcache.get(dataset, {}).get("default")
