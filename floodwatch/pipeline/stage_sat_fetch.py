"""Validation-only IMERG extraction; never alters model inputs or frozen artifacts."""
import argparse
import hashlib
from floodwatch.layout import legacy_digest, original_source
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from floodwatch.core import config as c

PROJECT = 'bdasem-7'
COLLECTION = 'NASA/GPM_L3/IMERG_V07'
BAND = 'precipitation'
OUT = c.ROOT / 'outputs/satellite'
CACHE = c.DATA_RAW / 'satellite'
DAILY_FILE = c.DATA_PROCESSED / 'imerg_daily.csv'
FROZEN = OUT / 'frozen_artifact_hashes.json'
TIMEBOX_SECONDS = 90 * 60
BBOX = [min(lon for lat, lon in c.BASIN_POINTS), min(lat for lat, lon in c.BASIN_POINTS),
        max(lon for lat, lon in c.BASIN_POINTS), max(lat for lat, lon in c.BASIN_POINTS)]


def verify_frozen():
    manifest = json.loads(FROZEN.read_text())
    changed = [name for name, wanted in manifest.items()
               if legacy_digest(c.ROOT/'outputs/spark/dashboard_original_app.py' if name == 'app.py' else c.ROOT/name) != wanted]
    if changed:
        raise ValueError(f'Frozen artifacts changed: {changed}')
    return len(manifest)


def month_dates(year, month):
    start = pd.Timestamp(year=year, month=month, day=1)
    return pd.date_range(start, start + pd.offsets.MonthEnd(0))


def fetch_month(ee, collection, geometry, projection, projection_info, year, month):
    dates = month_dates(year, month)
    start = ee.Date.parse('YYYY-MM-dd', dates[0].strftime('%Y-%m-%d'), c.TIMEZONE)
    end = start.advance(1, 'month', c.TIMEZONE)
    subset = collection.filterDate(start.advance(-15, 'minute'), end).sort('system:time_start')
    images = []
    for date in dates:
        day_start = ee.Date.parse('YYYY-MM-dd', date.strftime('%Y-%m-%d'), c.TIMEZONE)
        day_end = day_start.advance(1, 'day', c.TIMEZONE)
        frames = subset.filterDate(day_start.advance(-15, 'minute'), day_end)

        def integrate(image):
            stamp = ee.Date(image.get('system:time_start'))
            overlap_ms = stamp.advance(30, 'minute').millis().min(day_end.millis()).subtract(
                stamp.millis().max(day_start.millis())).max(0)
            return image.select(BAND).multiply(overlap_ms.divide(3600000))

        total = frames.map(integrate).sum().setDefaultProjection(projection)
        complete = frames.select(BAND).count().eq(49)
        area = ee.Image.pixelArea().updateMask(complete)
        image = total.updateMask(complete).multiply(area).rename('rain_area').addBands(area.rename('valid_area'))
        images.append(image.set('system:index', date.strftime('%Y%m%d')))
    stack = ee.ImageCollection.fromImages(images).toBands()
    kwargs = dict(geometry=geometry, crs=projection_info['crs'], crsTransform=projection_info['transform'], maxPixels=10000)
    result = ee.Dictionary({
        'sums': stack.reduceRegion(reducer=ee.Reducer.sum(), **kwargs),
        'total_area': ee.Image.pixelArea().reduceRegion(reducer=ee.Reducer.sum(), **kwargs),
        'timestamps': subset.aggregate_array('system:time_start'),
        'statuses': subset.aggregate_array('status'),
    }).getInfo()
    timestamps = pd.to_datetime(result['timestamps'], unit='ms', utc=True)
    expected_start = dates[0].tz_localize(c.TIMEZONE).tz_convert('UTC') - pd.Timedelta(minutes=15)
    expected = pd.date_range(expected_start, periods=len(dates)*48+1, freq='30min')
    if not timestamps.equals(expected):
        raise ValueError(f'{year}-{month:02d}: missing, duplicate or non-half-hour timestamps')
    if len(result['statuses']) != len(timestamps):
        raise ValueError('Source status metadata count differs from interval count')
    statuses = pd.Series(result['statuses'], index=timestamps)
    total_area = float(result['total_area']['area'])
    rows = []
    for date in dates:
        key = date.strftime('%Y%m%d')
        area = result['sums'].get(f'{key}_valid_area')
        weighted_rain = result['sums'].get(f'{key}_rain_area')
        if area is None or weighted_rain is None or not np.isclose(area, total_area, rtol=1e-6):
            raise ValueError(f'{date.date()}: missing IMERG pixel intervals; no partial-area average permitted')
        start_utc = date.tz_localize(c.TIMEZONE).tz_convert('UTC')
        selected = statuses[(statuses.index >= start_utc-pd.Timedelta(minutes=15))
                            & (statuses.index < start_utc+pd.Timedelta(days=1))]
        value = float(weighted_rain / area)
        if len(selected) != 49 or not np.isfinite(value) or value < 0:
            raise ValueError(f'{date.date()}: invalid daily rainfall or interval count')
        rows.append({'date': date.strftime('%Y-%m-%d'), 'imerg_mm': value,
                     'n_intervals': 49, 'provisional_intervals': int(selected.ne('permanent').sum())})
    return rows


def main(year=None, month=None):
    import ee
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    verify_frozen()
    ee.Initialize(project=PROJECT)
    ee.data.setDeadline(180000)
    collection = ee.ImageCollection(COLLECTION)
    geometry = ee.Geometry.Rectangle(BBOX, proj='EPSG:4326', geodesic=False)
    projection = collection.first().select(BAND).projection()
    projection_info = projection.getInfo()
    for y in ([year] if year else range(2001, 2026)):
        for m in ([month] if month else c.MONSOON_MONTHS):
            if time.monotonic()-started > TIMEBOX_SECONDS:
                raise TimeoutError('90-minute IMERG extraction time-box reached; use the approved fallback')
            path = CACHE / f'imerg_{y}_{m:02d}.json'
            if path.exists():
                payload = json.loads(path.read_text())
                if payload.get('project') != PROJECT or payload.get('bbox') != BBOX:
                    raise ValueError('Cached satellite extraction metadata differs')
                rows = payload['rows']
                if len(rows) != len(month_dates(y,m)):
                    raise ValueError('Truncated satellite cache')
                print(f'Reused {y}-{m:02d}: {len(rows)} days', flush=True)
                continue
            rows = fetch_month(ee, collection, geometry, projection, projection_info, y, m)
            payload = {'project': PROJECT, 'collection': COLLECTION, 'band': BAND, 'bbox': BBOX,
                       'timezone': c.TIMEZONE, 'rows': rows}
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(payload, indent=2))
            temporary.replace(path)
            print(f'Downloaded {y}-{m:02d}: {len(rows)} complete days', flush=True)
    all_rows = []
    for y in range(2001,2026):
        for m in c.MONSOON_MONTHS:
            path = CACHE / f'imerg_{y}_{m:02d}.json'
            if not path.exists():
                print('Probe complete; full 2001–2025 output awaits remaining months.')
                return
            all_rows.extend(json.loads(path.read_text())['rows'])
    daily = pd.DataFrame(all_rows).sort_values('date')
    expected = pd.date_range(c.START_DATE,c.END_DATE)
    expected = expected[expected.month.isin(c.MONSOON_MONTHS)]
    if not pd.DatetimeIndex(pd.to_datetime(daily.date)).equals(expected) or daily.imerg_mm.isna().any():
        raise ValueError('IMERG output date/value coverage is incomplete')
    daily.to_csv(DAILY_FILE,index=False)
    metadata = {'project': PROJECT, 'collection': COLLECTION, 'band': BAND, 'bbox': BBOX,
                'timezone': c.TIMEZONE, 'rows': len(daily),
                'integration': '49 overlapping half-hours: first/last 0.25 h, middle 47 at 0.5 h; total 24 h',
                'spatial_mean': 'native 0.1-degree pixel-area-weighted rectangle mean; full valid-area coverage required',
                'provisional_days': int(daily.provisional_intervals.gt(0).sum()),
                'frozen_artifacts_verified': verify_frozen()}
    (OUT/'metadata.json').write_text(json.dumps(metadata,indent=2))
    print(f'Saved {DAILY_FILE}: {len(daily)} monsoon days; frozen artifacts verified.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--year',type=int)
    parser.add_argument('--month',type=int,choices=c.MONSOON_MONTHS)
    args = parser.parse_args()
    main(args.year,args.month)
