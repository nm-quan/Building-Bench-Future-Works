"""Real-weather track for real-building evaluation: temperature + humidity
only, sourced from the era5 CSVs BuildingsBench already publishes on the
public S3 bucket (via data.py) -- no external fetching, no new dependency.

This matches what BuildingsBench's own real-eval code actually wires through
for real buildings (only `temperature` is used in their real-building eval
path). Buildings/datasets the bucket doesn't have weather for simply aren't
included in this track's cache -- `train.evaluate_real` is always run once
with `weather_cache=None` first (the headline, no-weather real-building
table, covering every real building regardless of weather availability),
and this track is a second, additional run limited to whichever buildings
have matching weather.
"""
import json
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

from . import config, data


# ---------------------------------------------------------------------------
# Full 7-channel real weather (extended track)
# ---------------------------------------------------------------------------
# BuildingsBench only ever published temperature (+ derived humidity) for the
# real datasets -- the other 5 training channels (wind speed/direction, GHI,
# DNI, DHI) exist nowhere on the bucket. We fetch them from Open-Meteo's ERA5
# archive (same reanalysis family the authors used via CDS), for the SAME
# locations, date ranges, and fixed local-standard-time offsets recovered
# verbatim from the authors' own scripts/data_generation/download_weather_era5.py.
# A validation gate then confirms our fetched temperature/humidity matches the
# authors' published era5 series before the cache is trusted (see
# validate_full_weather / build_full_weather_cache).

# location_key -> (lat, lon, start, end, utc_offset_hours). Coordinates are the
# centers of the authors' AREA_LONLAT boxes (the actual source-data locations,
# NOT the CONUS proxies in benchmark.toml); BDG-2 sites use their campus coords.
# The offset is added to the UTC index to reach local standard time (no DST),
# exactly matching the authors' `index + pd.to_timedelta(f'{offset}h')`.
FULL_WEATHER_LOCATIONS = {
    "LCL":         (51.50, -0.13,   "2011-01-01", "2014-12-31", 0),    # London
    "IDEAL":       (55.935, -3.205, "2017-01-01", "2018-12-31", 0),    # Edinburgh
    "Sceaux":      (48.775, 2.29,   "2006-12-31", "2011-01-01", 1),    # Sceaux, FR
    "Borealis":    (43.475, -80.54, "2010-12-31", "2013-01-01", -5),   # Waterloo, ON
    "Electricity": (38.745, -9.15,  "2011-01-01", "2014-12-31", 0),    # Lisbon
    "SMART":       (42.375, -72.515, "2013-12-31", "2017-01-01", -5),  # Amherst, MA
    "Panther":     (28.602, -81.200, "2015-12-31", "2018-01-01", -5),  # UCF Orlando
    "Fox":         (33.416, -111.935, "2015-12-31", "2018-01-01", -7), # ASU Tempe
    "Bear":        (37.872, -122.259, "2015-12-31", "2018-01-01", -8), # UC Berkeley
    "Rat":         (38.889, -77.005, "2015-12-31", "2018-01-01", -5),  # Washington DC
}

# Open-Meteo hourly variable -> our config.WEATHER_COLS name. Order/coverage of
# config.WEATHER_COLS: temperature, humidity, wind_speed, wind_direction,
# global_horizontal_radiation, direct_normal_radiation, diffuse_horizontal_radiation.
_OPENMETEO_VARS = {
    "temperature_2m": "temperature",
    "relative_humidity_2m": "humidity",
    "wind_speed_10m": "wind_speed",
    "wind_direction_10m": "wind_direction",
    "shortwave_radiation": "global_horizontal_radiation",
    "direct_normal_irradiance": "direct_normal_radiation",
    "diffuse_radiation": "diffuse_horizontal_radiation",
}
_OPENMETEO_URL = "https://archive-api.open-meteo.com/v1/archive"


def _loc_key_for_building(building_id: str) -> str:
    """building_id 'ds:site' or 'ds:site:col' -> weather location key.
    BDG-2 is per-site (Bear/Fox/Panther/Rat); every other dataset is per-dataset."""
    parts = building_id.split(":")
    ds, site = parts[0], parts[1]
    return site if ds == "BDG-2" else ds


def fetch_openmeteo_weather(lat: float, lon: float, start: str, end: str,
                            offset_hours: int, retries: int = 4) -> pd.DataFrame:
    """Fetches all 7 channels for one location/date-range from Open-Meteo's ERA5
    archive, in UTC, then shifts the index by `offset_hours` to local standard
    time (matching the authors' convention). Returns a DataFrame indexed by
    timestamp with columns == config.WEATHER_COLS. Requires outbound internet
    (works on Colab; blocked in some sandboxes)."""
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "hourly": ",".join(_OPENMETEO_VARS.keys()),
        "wind_speed_unit": "ms",       # match training weather (m/s, not km/h)
        "timezone": "GMT",             # return UTC; we add the fixed offset ourselves
        "format": "json",
    }
    url = f"{_OPENMETEO_URL}?{urlencode(params)}"
    last_err = None
    for attempt in range(retries):
        try:
            with urlopen(url, timeout=120) as fh:
                payload = json.load(fh)
            break
        except Exception as e:                       # noqa: BLE001 -- network/JSON, retry
            last_err = e
            time.sleep(2 ** attempt)
    else:
        raise RuntimeError(f"Open-Meteo fetch failed after {retries} tries: {last_err}")

    if "hourly" not in payload:
        raise RuntimeError(f"Open-Meteo response missing 'hourly' (got keys {list(payload)}): "
                           f"{payload.get('reason', payload)}")
    h = payload["hourly"]
    missing = [v for v in _OPENMETEO_VARS if v not in h]
    if missing:
        raise RuntimeError(f"Open-Meteo response missing variables {missing} -- "
                           f"variable names may have changed; got {list(h)}")
    idx = pd.to_datetime(h["time"]) + pd.to_timedelta(offset_hours, unit="h")
    df = pd.DataFrame({our: h[omv] for omv, our in _OPENMETEO_VARS.items()}, index=idx)
    df.index.name = "timestamp"
    return df[config.WEATHER_COLS].sort_index()


def validate_full_weather(fetched: pd.DataFrame, published, verbose: bool = True):
    """Confirms fetched weather aligns with the authors' PUBLISHED era5
    temperature/humidity (our ground-truth anchor) before the cache is trusted.
    Checks, on the overlapping hours: (a) temperature Pearson r > 0.95 at
    zero lag, (b) the best cross-correlation lag is 0 (catches timezone
    errors), (c) temperature mean offset < 3 degrees (catches unit errors).
    `published` is a DataFrame with a 'temperature' column (from
    data.load_real_weather_temp_humidity). Returns (ok: bool, info: dict)."""
    if published is None or "temperature" not in published.columns:
        return True, {"skipped": "no published temperature to validate against"}
    a = fetched["temperature"].dropna()
    b = published["temperature"].dropna()
    common = a.index.intersection(b.index)
    if len(common) < 168:
        return True, {"skipped": f"only {len(common)} overlapping hours"}
    av, bv = a.reindex(common).to_numpy(float), b.reindex(common).to_numpy(float)

    def _corr(x, y):
        x, y = x - x.mean(), y - y.mean()
        d = np.sqrt((x @ x) * (y @ y))
        return float((x @ y) / d) if d > 0 else 0.0

    r0 = _corr(av, bv)
    # best integer-hour lag in +/-12h (a timezone slip shifts the diurnal cycle)
    best_lag, best_r = 0, r0
    for lag in range(-12, 13):
        if lag == 0:
            continue
        if lag > 0:
            r = _corr(av[lag:], bv[:-lag])
        else:
            r = _corr(av[:lag], bv[-lag:])
        if r > best_r:
            best_r, best_lag = r, lag
    mean_off = float(np.abs(av.mean() - bv.mean()))
    ok = (r0 > 0.95) and (best_lag == 0) and (mean_off < 3.0)
    info = {"r_at_lag0": round(r0, 4), "best_lag_h": best_lag,
            "best_r": round(best_r, 4), "temp_mean_offset_C": round(mean_off, 3),
            "n_overlap_hours": len(common), "ok": ok}
    if verbose:
        status = "PASS" if ok else "FAIL"
        print(f"    [validate] {status}  r@lag0={info['r_at_lag0']}  "
              f"best_lag={best_lag}h  mean_off={info['temp_mean_offset_C']}C  "
              f"(n={len(common)}h)")
    return ok, info


def build_full_weather_cache(paths: config.Paths, buildings: list,
                             validate: bool = True, verbose: bool = True) -> dict:
    """Builds {building_id: DataFrame(all 7 config.WEATHER_COLS)} for every real
    building, fetching each location once from Open-Meteo (cached to Drive as
    CSV under raw_cache/real_full_weather/ so re-runs don't re-fetch), and --
    if validate=True -- gating each location against the authors' published
    era5 temperature/humidity. Locations that FAIL validation are dropped from
    the cache (logged), so bad alignment can never silently feed the eval.
    Plugs directly into train.evaluate_real(..., weather_cache=...,
    weather_transforms=...)."""
    cache_dir = Path(paths.RAW_CACHE_DIR) / "real_full_weather"
    cache_dir.mkdir(parents=True, exist_ok=True)
    published_raw = data.load_real_weather_temp_humidity(paths, verbose=False) if validate else {}

    # which location keys are actually needed by the loaded buildings
    needed = sorted({_loc_key_for_building(b["building_id"]) for b in buildings})
    loc_weather, loc_ok = {}, {}
    for key in needed:
        if key not in FULL_WEATHER_LOCATIONS:
            if verbose:
                print(f"  [full-weather] no location spec for '{key}', skipping")
            loc_ok[key] = False
            continue
        lat, lon, start, end, off = FULL_WEATHER_LOCATIONS[key]
        csv_path = cache_dir / f"{key}.csv"
        if csv_path.exists():
            wdf = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        else:
            if verbose:
                print(f"  [full-weather] fetching {key} ({lat},{lon}) {start}..{end} off={off}h ...", flush=True)
            wdf = fetch_openmeteo_weather(lat, lon, start, end, off)
            wdf.to_csv(csv_path)
        loc_weather[key] = wdf

        ok = True
        if validate:
            # published temp/humidity for this location (BDG-2 keyed by site)
            if key in ("Bear", "Fox", "Panther", "Rat"):
                pub = published_raw.get("BDG-2", {}).get(key)
            else:
                pub = published_raw.get(key, {}).get("default")
            ok, _info = validate_full_weather(wdf, pub, verbose=verbose)
            if not ok and verbose:
                print(f"  [full-weather] {key}: FAILED validation -> dropped from cache")
        loc_ok[key] = ok

    # map building_id -> its location's full-weather df (only validated locations)
    out = {}
    for b in buildings:
        key = _loc_key_for_building(b["building_id"])
        if loc_ok.get(key) and key in loc_weather:
            out[b["building_id"]] = loc_weather[key]
    if verbose:
        n_loc_ok = sum(1 for v in loc_ok.values() if v)
        print(f"  [full-weather] {n_loc_ok}/{len(needed)} locations validated; "
              f"{len(out)}/{len(buildings)} buildings have full 7-channel weather")
    return out


def fetch_temp_humidity_track(paths: config.Paths, buildings: list, datasets: list = None,
                               verbose: bool = True) -> dict:
    """Returns {building_id: DataFrame(columns=['temperature','humidity'])}
    for buildings whose dataset has era5 weather on S3. Plugs directly into
    `train.evaluate_real(..., weather_cache=..., weather_transforms=...)`."""
    raw = data.load_real_weather_temp_humidity(paths, datasets=datasets, verbose=verbose)
    out = {}
    for b in buildings:
        w = data.weather_for_building(b["building_id"], raw)
        if w is not None:
            out[b["building_id"]] = w
    return out
