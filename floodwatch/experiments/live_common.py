"""Live adapters. Frozen feature engineering and model artifacts are read only."""
from pathlib import Path
import hashlib
from floodwatch.layout import legacy_digest, original_source
import json
import numpy as np
import pandas as pd
import requests
from floodwatch.core import config as c
from floodwatch.core.features import build_features

LIVE = c.ROOT / 'data/live'
MAPS = LIVE / 'quantile_maps'
REPORT = c.ROOT / 'outputs/live'
VARIABLES = {
    'precip_mm': 'precipitation',
    'sm_0_7': 'soil_moisture_0_to_7cm',
    'sm_7_28': 'soil_moisture_7_to_28cm',
    'sm_28_100': 'soil_moisture_28_to_100cm',
    'tcwv_mm': 'total_column_integrated_water_vapour',
}
SOURCE = 'ECMWF IFS'
FORECAST_MODEL = 'ecmwf_ifs025'
BANNER = ('Live mode: inputs from ECMWF IFS forecasts, quantile-mapped to ERA5 '
          'using overlapping 2017–2025 monsoon days; models were trained on ERA5, '
          'so scores are indicative. Today includes forecast hours. '
          'Soil depths are 0–7, 7–28 and 28–100 cm; no depth approximation.')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_daily(path):
    return pd.read_csv(path, index_col=0, parse_dates=True)


def get_json(url, params):
    response = requests.get(url, params=params, timeout=180)
    response.raise_for_status()
    return response.json()


def weather_params(model, **dates):
    return dict(latitude=','.join(str(p[0]) for p in c.BASIN_POINTS),
                longitude=','.join(str(p[1]) for p in c.BASIN_POINTS),
                hourly=','.join(VARIABLES.values()), models=model,
                timezone=c.TIMEZONE, **dates)


def weather_daily(payload, start, end):
    """Require complete 24-hour days at all nine points; never fill nulls."""
    if not isinstance(payload, list) or len(payload) != 9:
        raise ValueError('Expected exactly nine weather point responses')
    expected = pd.date_range(start, end, freq='h')
    expected = pd.date_range(pd.Timestamp(start), pd.Timestamp(end)+pd.Timedelta(hours=23), freq='h')
    points = []
    for item in payload:
        hourly = item['hourly']
        times = pd.DatetimeIndex(pd.to_datetime(hourly['time']))
        if not times.equals(expected):
            raise ValueError(f'Truncated hourly response: {len(times)} rows, expected {len(expected)}')
        frame = pd.DataFrame({key: hourly[var] for key, var in VARIABLES.items()}, index=times, dtype=float)
        count = frame.resample('D').count()
        daily = frame.resample('D').mean()
        daily['precip_mm'] = frame.precip_mm.resample('D').sum(min_count=24)
        daily = daily.where(count == 24)
        points.append(daily)
    stacked = pd.concat(points, keys=range(9))
    result = stacked.groupby(level=1).mean().where(stacked.groupby(level=1).count() == 9)
    rains = pd.concat([p.precip_mm for p in points], axis=1)
    result['precip_max_point_mm'] = rains.max(axis=1).where(rains.count(axis=1) == 9)
    result.index.name = 'date'
    return result


def calendar_day(index):
    return pd.to_datetime('2000-' + index.strftime('%m-%d')).dayofyear.to_numpy()


def mapped_inputs(raw, mapping_dir=MAPS):
    out = raw.copy()
    for column in [*VARIABLES, 'precip_max_point_mm']:
        mapping = json.loads((mapping_dir / f'{column}.json').read_text())
        dates = pd.DatetimeIndex(mapping['dates'])
        ref_days = calendar_day(dates)
        x, y = np.array(mapping['source']), np.array(mapping['era5'])
        out[column] = np.nan
        for day in np.unique(calendar_day(raw.index)):
            delta = np.abs(ref_days - day)
            pool = np.minimum(delta, 366-delta) <= 15
            if pool.sum() < 30:
                continue
            sx, sy = np.sort(x[pool]), np.sort(y[pool])
            mask = calendar_day(raw.index) == day
            values = raw.loc[mask, column].to_numpy()
            rank = (np.searchsorted(sx, values, side='left') + np.searchsorted(sx, values, side='right')) / (2*len(sx))
            mapped = np.quantile(sy, rank)
            mapped[~np.isfinite(values)] = np.nan
            out.loc[mask, column] = mapped
    return out


def live_features(mapped):
    return build_features(mapped, read_daily(c.STAGE3_REFERENCE))


def score_features(features, day):
    import xgboost as xgb
    selected = json.loads(c.STAGE3_CALENDAR_SELECTED.read_text())
    result = {}
    for item in selected['horizons']:
        h = item['horizon']
        result[f'threshold_{h}d'] = item['threshold']
        result[f'variant_{h}d'] = item['selected_variant']
        result[f'risk_{h}d'] = None
        if day not in features.index:
            continue
        model = xgb.Booster()
        model.load_model(c.STAGE3_CALENDAR_DIR / f"{item['selected_variant']}_{h}d.json")
        matrix = xgb.DMatrix(features.loc[[day], item['columns']], feature_names=item['columns'])
        result[f'risk_{h}d'] = float(model.predict(matrix)[0])
    return result


def analogue(features, day):
    metadata = json.loads(c.STAGE4_METADATA.read_text())
    names, window = metadata['features'], metadata['window_days']
    values = features.reindex(pd.date_range(day-pd.Timedelta(days=window-1), day))[names]
    if not np.isfinite(values.to_numpy()).all():
        return None
    scaler = pd.read_csv(c.STAGE4_SCALERS, index_col='feature').loc[names]
    query = ((values-scaler.training_mean)/scaler.training_std).to_numpy().ravel()
    templates = pd.read_csv(c.STAGE4_TEMPLATES)
    templates = templates[pd.to_datetime(templates.event_onset) < day].copy()
    if day.year == 2024 and day.month == 9:
        templates = templates[pd.to_datetime(templates.event_onset) <= '2023-12-31']
    cols = [f'day_{d:02d}__{name}' for d in range(1, window+1) for name in names]
    distance = np.sqrt(((templates[cols].to_numpy()-query)**2).sum(axis=1))
    if not len(distance):
        return None
    i = int(np.argmin(distance))
    row = templates.iloc[i]
    return dict(event_id=int(row.event_id), onset=row.event_onset,
                days_before_flood=int(row.days_before_flood), distance=float(distance[i]), window_days=window)


def verify_frozen():
    baseline = json.loads((c.ROOT/'outputs/satellite/frozen_artifact_hashes.json').read_text())
    # The Today tab wrapper is explicitly authorized; replay logic is checked separately.
    changed = [name for name, sha in baseline.items() if name != 'app.py' and legacy_digest(c.ROOT/name) != sha]
    if changed:
        raise ValueError(f'Frozen files changed: {changed}')
    return len(baseline)-1
