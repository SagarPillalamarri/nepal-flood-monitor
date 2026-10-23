"""Calibrate GFS seasonal ranks before any operational training is permitted.

Run from the repository root: python -m floodwatch.operational.calibration
"""
import hashlib
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from floodwatch.core import config as c
from floodwatch.core.features import tcwv_percentile
from floodwatch.pipeline.stage_sat_fetch import verify_frozen

OUT = c.ROOT / 'outputs/operational'
CACHE = c.DATA_RAW / 'operational_gfs'
VARIABLES = {'precip_mm': 'precipitation', 'tcwv_mm': c.TCWV_VARIABLE}


def daily(payload, start, end):
    expected = pd.date_range(start, pd.Timestamp(end) + pd.Timedelta(hours=23), freq='h')
    if not isinstance(payload, list) or len(payload) != len(c.BASIN_POINTS):
        raise ValueError('Expected nine basin-point responses')
    points = []
    for loc in payload:
        h = loc['hourly']
        dates = pd.DatetimeIndex(pd.to_datetime(h['time']))
        if not dates.equals(expected):
            raise ValueError(f'Truncated hourly response: {len(dates)} vs {len(expected)} expected')
        series = {}
        for column, variable in VARIABLES.items():
            if len(h.get(variable, [])) != len(expected):
                raise ValueError(f'Missing/truncated {variable}')
            values = pd.Series(h[variable], index=dates, dtype=float)
            if column == 'precip_mm':
                values.index = values.index - pd.Timedelta(hours=1)
                series[column] = values.resample('D').sum(min_count=24)
            else:
                series[column] = values.resample('D').mean().where(values.resample('D').count().eq(24))
        points.append(pd.DataFrame(series))
    result = sum(points) / len(points)  # Null at any point remains null.
    result.index.name = 'date'
    return result.loc[start:pd.Timestamp(end)-pd.Timedelta(days=1)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    print(f'[PASS] {verify_frozen()} original frozen artifacts verified', flush=True)
    years = []
    for year in range(2021, 2026):
        start, end = f'{year}-05-18', f'{year}-11-01'
        params = dict(latitude=','.join(str(p[0]) for p in c.BASIN_POINTS),
                      longitude=','.join(str(p[1]) for p in c.BASIN_POINTS),
                      hourly=','.join(VARIABLES.values()), models='gfs_seamless',
                      timezone=c.TIMEZONE, start_date=start, end_date=end)
        path = CACHE / f'gfs_{start}_{end}.json'
        if path.exists():
            payload = json.loads(path.read_text())
            frame = daily(payload, start, end)
            print(f'[CACHE] {year}: validated hourly timestamps and array lengths', flush=True)
        else:
            response = requests.get(c.GFS_HISTORY_URL, params=params, timeout=180)
            metadata = dict(url=response.url, status=response.status_code,
                            fetched_at=datetime.now(timezone.utc).isoformat(), params=params)
            (CACHE/f'{year}_request.json').write_text(json.dumps(metadata, indent=2))
            if not response.ok:
                (CACHE/f'{year}_error.txt').write_text(response.text)
                response.raise_for_status()
            payload = response.json()
            frame = daily(payload, start, end)  # Validate BEFORE caching.
            counts = [{v:sum(x is not None for x in p['hourly'][v]) for v in VARIABLES.values()} for p in payload]
            metadata['non_null_hour_counts_by_point'] = counts
            (CACHE/f'{year}_request.json').write_text(json.dumps(metadata, indent=2))
            path.write_bytes(response.content)
            time.sleep(2)
        years.append(frame)
        print(f'[FETCH] {year}: complete monsoon basin days '+str(frame.loc[f'{year}-06-01':f'{year}-10-31'].notna().sum().to_dict()), flush=True)
    raw = pd.concat(years).sort_index()
    raw.to_csv(OUT/'gfs_daily.csv')
    # Preserve warm-up dates for operational rolling rainfall climatologies.
    signals = raw.reindex(pd.date_range(raw.index.min(), raw.index.max(), name='date'))
    ref = pd.DataFrame(index=signals.index)
    for n in c.RAINFALL_WINDOWS:
        ref[f'rain_{n}d'] = signals.precip_mm.rolling(n, min_periods=n).sum()
    ref['tcwv'] = signals.tcwv_mm
    ref = ref.loc[ref.index.month.isin(c.MONSOON_MONTHS)]
    ref.to_csv(OUT/'gfs_reference.csv')
    era = pd.read_csv(c.STAGE1B_FILE, index_col=0, parse_dates=True)
    scores, rows = [], []
    for variable in VARIABLES:
        g = raw.loc[raw.index.month.isin(c.MONSOON_MONTHS), variable]
        e = era[variable].reindex(g.index)
        era_reference = era.loc[c.TRAIN_START:c.TRAIN_END, variable]
        ep = tcwv_percentile(e, era_reference)
        # Keep the research minimum-sample rule. Unsupported seasonal windows
        # remain NaN and are excluded explicitly from diagnostic correlations.
        gp = pd.Series(np.nan, index=g.index, dtype=float)
        for _, group in g.groupby(g.index.strftime('%m-%d')):
            if not group.notna().any():
                continue
            try:
                gp.loc[group.index] = tcwv_percentile(group, g)
            except ValueError as error:
                if 'Too few climatology samples' not in str(error):
                    raise
        unsupported = int((g.notna() & gp.isna()).sum())
        joined = pd.DataFrame({'era5_value':e,'gfs_value':g,'era5_percentile':ep,'gfs_percentile':gp}).dropna()
        r = joined.era5_percentile.corr(joined.gfs_percentile)
        status = 'FAIL' if not np.isfinite(r) or r < .5 else 'WARN' if r < .7 else 'PASS'
        rows.append(dict(variable=variable, overlap_start=str(joined.index.min().date()),
                         overlap_end=str(joined.index.max().date()), n_days=len(joined),
                         percentile_correlation=r, status=status,
                         gfs_missing_days=int(g.isna().sum()),era5_missing_days=int(e.isna().sum()),
                         insufficient_climatology_days=unsupported))
        joined['variable'] = variable
        scores.append(joined.reset_index())
    table = pd.DataFrame(rows)
    table.to_csv(OUT/'calibration_correlations.csv', index=False)
    pd.concat(scores).to_csv(OUT/'calibration_daily_comparison.csv', index=False)
    count = verify_frozen()
    incomplete = table.gfs_missing_days.gt(0).any() or table.insufficient_climatology_days.gt(0).any()
    stopped = table.status.eq('FAIL').any() or incomplete
    summary = dict(model='gfs_seamless', monsoon_months=c.MONSOON_MONTHS,
                   era5_reference=['2001-01-01','2019-12-31'], gfs_reference=['2021-06-01','2025-10-31'],
                   local_timezone=c.TIMEZONE,rain_hours='01:00 through next-day 00:00',
                   tcwv_hours='00:00 through 23:00', n_basin_points=9,
                   frozen_entries_verified=count, operational_training_permitted=not bool(stopped))
    (OUT/'calibration_metadata.json').write_text(json.dumps(summary, indent=2))
    coverage = raw.loc[raw.index.month.isin(c.MONSOON_MONTHS)].groupby(raw.loc[raw.index.month.isin(c.MONSOON_MONTHS)].index.year).count()
    coverage.index.name = 'year'
    coverage.to_csv(OUT/'calibration_coverage.csv')
    lines=['Complete monsoon days per year (out of 153):', coverage.to_string(), table.to_string(index=False),f'[PASS] {count} original frozen artifacts unchanged',
           '[PASS] ERA5 ranks use only 2001–2019; GFS ranks use only 2021–2025 monsoon values',
           '[PASS] Nine-point means require complete hourly observations; missing values are not filled',
           '[STOP] Calibration gate failed: correlation below 0.5/undefined or incomplete requested GFS climatology; operational training and scoring not started' if stopped else '[PASS] Calibration gate permits operational model development']
    if incomplete:
        lines.append('[WARN] GFS TCWV climatology is incomplete; its reported correlation covers supported seasonal windows only')
    lines.append(f"Summary: {3 + int(table.status.eq('PASS').sum())} PASS, {int(table.status.eq('WARN').sum()) + int(incomplete)} WARN, {int(table.status.eq('FAIL').sum())} FAIL")
    report='\n'.join(lines)+'\n'
    (OUT/'calibration_check_output.txt').write_text(report)
    print(report,flush=True)
    return int(stopped)


if __name__ == '__main__':
    raise SystemExit(main())
