"""
Stage 1: download daily rainfall, soil moisture and river discharge for the Bagmati basin.

Sources (free, no API key):
  - Open-Meteo Historical API (ERA5 / ERA5-Land reanalysis): rainfall + soil moisture
  - Open-Meteo Flood API (GloFAS v4): modelled river discharge

Outputs:
  data/raw/weather_daily.csv        basin-average weather per day
  data/raw/glofas_daily.csv         discharge at Khokana + 4 nearby grid points
  data/processed/stage1_daily.csv   both merged on date

Run:  python stage1_fetch.py
"""
import time

import pandas as pd
import requests

from floodwatch.core.config import (BASIN_POINTS, DATA_PROCESSED, DATA_RAW, END_DATE,
                    GLOFAS_POINTS, START_DATE, TIMEZONE)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FLOOD_URL = "https://flood-api.open-meteo.com/v1/flood"

SOIL_VARS = ["soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm", "soil_moisture_28_to_100cm"]


def get_json(url, params, retries=4):
    """GET with simple backoff for rate limits (HTTP 429) and timeouts."""
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=120)
        except requests.exceptions.RequestException as e:
            print(f"  network error ({e}); retry {attempt}/{retries}")
            time.sleep(10 * attempt)
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            wait = 60 * attempt
            print(f"  rate limited; waiting {wait}s (retry {attempt}/{retries})")
            time.sleep(wait)
            continue
        raise RuntimeError(f"HTTP {r.status_code} from {url}: {r.text[:300]}")
    raise RuntimeError(f"Gave up on {url} after {retries} attempts")


def fetch_weather_year(year):
    """One year of weather for all 9 basin points, averaged to one basin value per day."""
    start = max(f"{year}-01-01", START_DATE)
    end = min(f"{year}-12-31", END_DATE)
    params = {
        "latitude": ",".join(str(p[0]) for p in BASIN_POINTS),
        "longitude": ",".join(str(p[1]) for p in BASIN_POINTS),
        "start_date": start,
        "end_date": end,
        "daily": "precipitation_sum",
        "hourly": ",".join(SOIL_VARS),
        "timezone": TIMEZONE,
    }
    data = get_json(ARCHIVE_URL, params)
    if isinstance(data, dict):  # single location returns a dict, many return a list
        data = [data]

    frames = []
    for i, loc in enumerate(data):
        daily = pd.DataFrame({
            "date": pd.to_datetime(loc["daily"]["time"]),
            "precip_mm": loc["daily"]["precipitation_sum"],
        }).set_index("date")

        hourly = pd.DataFrame({v: loc["hourly"][v] for v in SOIL_VARS})
        hourly.index = pd.to_datetime(loc["hourly"]["time"])
        soil_daily = hourly.resample("D").mean()
        soil_daily.columns = ["sm_0_7", "sm_7_28", "sm_28_100"]

        df = daily.join(soil_daily, how="left")
        df["point"] = i
        frames.append(df)

    all_points = pd.concat(frames)
    basin = all_points.groupby(level=0)[["precip_mm", "sm_0_7", "sm_7_28", "sm_28_100"]].mean()
    basin["precip_max_point_mm"] = all_points.groupby(level=0)["precip_mm"].max()
    return basin


def fetch_weather():
    years = range(int(START_DATE[:4]), int(END_DATE[:4]) + 1)
    parts = []
    for y in years:
        print(f"Weather {y} ...")
        parts.append(fetch_weather_year(y))
        time.sleep(2)  # stay polite with the free API
    weather = pd.concat(parts).sort_index()
    weather.index.name = "date"
    out = DATA_RAW / "weather_daily.csv"
    weather.to_csv(out)
    print(f"Saved {out} ({len(weather)} rows)")
    return weather


def fetch_glofas():
    series = {}
    for name, (lat, lon) in GLOFAS_POINTS.items():
        print(f"GloFAS {name} ({lat:.3f}, {lon:.3f}) ...")
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "river_discharge",
            "start_date": START_DATE,
            "end_date": END_DATE,
        }
        data = get_json(FLOOD_URL, params)
        s = pd.Series(data["daily"]["river_discharge"],
                      index=pd.to_datetime(data["daily"]["time"]), name=name)
        series[name] = s
        time.sleep(2)
    glofas = pd.DataFrame(series)
    glofas.index.name = "date"
    out = DATA_RAW / "glofas_daily.csv"
    glofas.to_csv(out)
    print(f"Saved {out} ({len(glofas)} rows)")
    return glofas


def main():
    weather = fetch_weather()
    glofas = fetch_glofas()
    merged = weather.join(glofas, how="outer")
    out = DATA_PROCESSED / "stage1_daily.csv"
    merged.to_csv(out)
    print(f"\nSaved {out} ({len(merged)} rows, {merged.shape[1]} columns)")
    print("Next: python stage1_check.py")


if __name__ == "__main__":
    main()