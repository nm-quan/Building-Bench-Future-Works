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
from . import config, data


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
