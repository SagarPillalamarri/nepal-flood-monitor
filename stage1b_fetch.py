"""Fetch approved atmospheric data, stopping on any missing input. Run with Python."""
from io import StringIO
import json
import sys
import time

import numpy as np
import pandas as pd
import requests
import config as c


def read_daily(path, allow_tcwv_gap=False):
    df = pd.read_csv(path, parse_dates=['date']).set_index('date')
    if df.index.has_duplicates or not df.index.is_monotonic_increasing:
        raise ValueError(f"Invalid or missing daily data: {path}")
    validate_values(df, allow_tcwv_gap)
    return df


def validate_values(df, allow_tcwv_gap=False):
    for col in df.columns:
        bad = ~np.isfinite(df[col].to_numpy(dtype=float))
        allowed = pd.Series(False, index=df.index)
        if allow_tcwv_gap and col == "tcwv_mm":
            allowed = (df.index >= pd.Timestamp(c.TCWV_GAP_START).normalize()) & (df.index <= pd.Timestamp(c.TCWV_GAP_END).normalize()) & df[col].isna()
        if (bad & ~np.asarray(allowed)).any():
            raise ValueError(f"Unexpected missing or non-finite values in {col}")


def require_dates(df, start, end, allow_tcwv_gap=False):
    expected = pd.date_range(start, end, freq='D')
    if not df.index.equals(expected):
        raise ValueError(f"Incomplete daily coverage: {start} to {end}")
    validate_values(df, allow_tcwv_gap)


def request_json(url, params, cache):
    if cache.exists():
        print(f'  Reusing cache: {cache.name}', flush=True)
        return json.loads(cache.read_text())
    response = requests.get(url, params=params, timeout=c.API_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    # Validate timestamps, hourly counts, values and all points before caching.
    hourly_basin(data, params['hourly'].split(','), params['start_date'], params['end_date'], allow_tcwv_gap=params['models'] == c.ERA5_MODEL)
    temporary = cache.with_suffix('.tmp')
    temporary.write_text(json.dumps(data))
    temporary.replace(cache)
    time.sleep(c.API_PAUSE_SECONDS)
    return data


def hourly_basin(data, variables, start, end, rainfall=False, allow_tcwv_gap=False):
    if not isinstance(data, list) or len(data) != len(c.BASIN_POINTS):
        raise ValueError('Expected all nine basin point responses')
    selected = c.BASIN_POINTS
    response_points = c.BASIN_POINTS
    expected = pd.date_range(start, pd.Timestamp(end) + pd.Timedelta(days=1), freq='h', inclusive='left')
    points = []
    for point, loc in zip(response_points, data):
        hourly = loc['hourly']
        index = pd.DatetimeIndex(pd.to_datetime(hourly['time']))
        frame = pd.DataFrame({v: hourly[v] for v in variables}, index=index)
        if not index.equals(expected):
            raise ValueError(f"Unexpected hourly timestamps for {start} to {end}")
        bad = ~np.isfinite(frame.to_numpy(dtype=float))
        allowed = (index >= pd.Timestamp(c.TCWV_GAP_START)) & (index <= pd.Timestamp(c.TCWV_GAP_END))
        accepted = allow_tcwv_gap and variables == [c.TCWV_VARIABLE]
        if (bad & ~(allowed[:, None] & frame.isna().to_numpy() & accepted)).any():
            bad_times = index[bad.any(axis=1)]
            raise ValueError(f"Missing {bad.sum()} hourly values at basin point "
                             f"{point}: "
                             f"{bad_times.min()} to {bad_times.max()}; no averaging over gaps")
        units = loc['hourly_units']
        if any(units[v] not in ('mm', 'kg/m²', 'kg/m2') for v in variables):
            raise ValueError(f"Unexpected units: {units}")
        if (frame < 0).any().any():
            raise ValueError("Negative rainfall or water vapour")
        daily = frame.resample('D').sum() if rainfall else frame.resample('D').mean()
        # Any incomplete point-day stays NaN; never average a partial day.
        daily = daily.where(frame.resample('D').count() == c.HOURS_PER_DAY)
        points.append(daily)
    if len(points) != len(selected):
        raise ValueError('Incomplete fixed TCWV point set')
    basin = sum(points) / len(points)
    basin.index.name = 'date'
    if allow_tcwv_gap:
        basin.columns = ["tcwv_mm"]
    require_dates(basin, start, end, allow_tcwv_gap)
    return basin


def fetch_series(name, url, model, variables, start, end, rainfall=False):
    parts = []
    for year in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        lo = max(pd.Timestamp(start), pd.Timestamp(year=year, month=1, day=1))
        hi = min(pd.Timestamp(end), pd.Timestamp(year=year, month=12, day=31))
        print(f"{name}: {lo.date()} to {hi.date()}", flush=True)
        selected = c.BASIN_POINTS
        params = dict(latitude=','.join(str(p[0]) for p in selected),
                      longitude=','.join(str(p[1]) for p in selected),
                      start_date=str(lo.date()), end_date=str(hi.date()),
                      hourly=','.join(variables), timezone=c.TIMEZONE, models=model)
        cache = c.STAGE1B_CACHE / f'{name}_{lo.date()}_{hi.date()}.json'
        if name == 'era5_tcwv' and year == c.ERA5_REPAIR_YEAR and not cache.exists() and c.ERA5_REPAIR_CACHE.exists():
            print('  Reusing preserved response; repair disabled', flush=True)
            data = json.loads(c.ERA5_REPAIR_CACHE.read_text())
        else:
            data = request_json(url, params, cache)
        parts.append(hourly_basin(data, variables, lo, hi, rainfall, allow_tcwv_gap=model == c.ERA5_MODEL))
    result = pd.concat(parts)
    require_dates(result, start, end, allow_tcwv_gap=model == c.ERA5_MODEL)
    return result


def load_oni():
    path = c.ONI_RAW_FILE
    if path.exists():
        text = path.read_text()
        print('Reusing cached NOAA ONI', flush=True)
    else:
        response = requests.get(c.ONI_URL, timeout=c.API_TIMEOUT)
        response.raise_for_status()
        text = response.text
    table = pd.read_csv(StringIO(text), sep=r'\s+')
    if not {'SEAS', 'YR', 'ANOM'}.issubset(table.columns):
        raise ValueError('Unexpected NOAA ONI schema')
    if table.duplicated(['YR', 'SEAS']).any() or not np.isfinite(table['ANOM']).all():
        raise ValueError('Duplicate or missing ONI values')
    path.write_text(text)
    return table


def oni_for_days(index, table):
    # NOAA labels seasons by the centre month's year, including DJF and NDJ.
    lookup = table.set_index(['YR', 'SEAS'])['ANOM']
    values = []
    for date in index:
        ending = date.to_period('M') - c.ONI_LAG_MONTHS
        centre = ending - 1
        key = (centre.year, c.ONI_SEASONS[centre.month - 1])
        if key not in lookup.index:
            raise ValueError(f'Missing ONI season {key} for {date.date()}')
        values.append(lookup.loc[key])
    return pd.Series(values, index=index, name='oni')


def main():
    c.STAGE1B_CACHE.mkdir(parents=True, exist_ok=True)
    base = read_daily(c.STAGE1_FILE)
    require_dates(base, c.START_DATE, c.END_DATE)
    oni = oni_for_days(base.index, load_oni())
    oni.to_csv(c.ONI_DAILY_FILE)
    print(f'ONI downloaded/validated correctly: {len(oni)} daily assignments', flush=True)
    print('TCWV uses all nine basin points; confirmed source gap remains NaN', flush=True)
    era = fetch_series('era5_tcwv', c.ERA5_URL, c.ERA5_MODEL, [c.TCWV_VARIABLE], c.START_DATE, c.END_DATE)
    era.columns = ['tcwv_mm']
    merged = base.join(era).join(oni)
    require_dates(merged, c.START_DATE, c.END_DATE, allow_tcwv_gap=True)
    merged.to_csv(c.STAGE1B_FILE)
    print(f'Saved {c.STAGE1B_FILE} ({len(merged)} rows). Run stage1b_check.py')


if __name__ == '__main__':
    try:
        main()
    except (requests.RequestException, ValueError, KeyError, OSError) as error:
        print(f'[FAIL] Stopped: {error}', file=sys.stderr)
        sys.exit(1)
