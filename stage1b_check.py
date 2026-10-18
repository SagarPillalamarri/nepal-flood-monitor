"""Stage 1b completeness, ONI timing, event sanity and feature leakage checks."""
import json
import sys
import numpy as np
import pandas as pd
import config as c
from features import tcwv_percentile
from stage1b_fetch import read_daily, require_dates, oni_for_days, hourly_basin

results = []


def report(status, message):
    results.append(dict(status=status, message=message))
    print(f'[{status}] {message}')


def cached_era(start, end):
    """Read complete requested hours from cached years without any download."""
    parts = []
    for year in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        year_start, year_end = f'{year}-01-01', f'{year}-12-31'
        lo = max(pd.Timestamp(start), pd.Timestamp(year_start))
        hi = min(pd.Timestamp(end), pd.Timestamp(year_end))
        path = c.STAGE1B_CACHE / f'era5_tcwv_{year_start}_{year_end}.json'
        if not path.exists() and year == c.ERA5_REPAIR_YEAR:
            path = c.ERA5_REPAIR_CACHE
        locations = json.loads(path.read_text())
        if lo == pd.Timestamp(year_start) and hi == pd.Timestamp(year_end):
            parts.append(hourly_basin(locations, [c.TCWV_VARIABLE], lo, hi, allow_tcwv_gap=True).iloc[:, 0])
            continue
        trimmed = []
        for loc in locations:
            times = pd.to_datetime(loc['hourly']['time'])
            mask = (times >= lo) & (times < hi + pd.Timedelta(days=1))
            values = np.array(loc['hourly'][c.TCWV_VARIABLE], dtype=object)[mask]
            trimmed.append(dict(hourly={'time': list(times[mask].strftime('%Y-%m-%dT%H:%M')),
                                       c.TCWV_VARIABLE: values.tolist()},
                                hourly_units=loc['hourly_units']))
        parts.append(hourly_basin(trimmed, [c.TCWV_VARIABLE], lo, hi, allow_tcwv_gap=True).iloc[:, 0])
    return pd.concat(parts).rename('tcwv_mm')


def check_oni():
    table = pd.read_csv(c.ONI_RAW_FILE, sep=r'\s+')
    daily = read_daily(c.ONI_DAILY_FILE)
    require_dates(daily, c.START_DATE, c.END_DATE)
    index = pd.date_range(c.START_DATE, c.END_DATE, name='date')
    expected = oni_for_days(index, table)
    pd.testing.assert_series_equal(daily.oni, expected, check_freq=False)
    report('PASS', f'NOAA ONI downloaded and validated; {len(daily)} daily assignments, no missing values')
    for date in index[index.is_month_start]:
        end = date.to_period('M') - c.ONI_LAG_MONTHS
        season_index = (end.month - 2) % len(c.ONI_SEASONS)
        year = end.year - int(end.month == 1)
        row = table[(table.YR == year) & (table.SEAS == c.ONI_SEASONS[season_index])]
        if len(row) != 1 or daily.loc[date, 'oni'] != row.ANOM.iloc[0]:
            raise ValueError(f'ONI lag mapping failed at {date}')
    report('PASS', 'ONI seasons end two months before scoring month; year boundaries checked')


def check_output():
    base = read_daily(c.STAGE1_FILE)
    expected_tcwv = cached_era(c.START_DATE, c.END_DATE)
    require_dates(expected_tcwv.to_frame(), c.START_DATE, c.END_DATE, allow_tcwv_gap=True)
    data = read_daily(c.STAGE1B_FILE, allow_tcwv_gap=True)
    require_dates(data, c.START_DATE, c.END_DATE, allow_tcwv_gap=True)
    pd.testing.assert_frame_equal(data[base.columns], base)
    if list(data.columns) != list(base.columns) + ['tcwv_mm', 'oni']:
        raise ValueError('Unexpected output columns')
    if (data.tcwv_mm < 0).any():
        raise ValueError('Negative TCWV')
    pd.testing.assert_series_equal(data.tcwv_mm, expected_tcwv, check_freq=False)
    missing = data.tcwv_mm.isna()
    monsoon_missing = missing & data.index.month.isin(c.MONSOON_MONTHS)
    counts = data.index[monsoon_missing].to_period('M').value_counts().sort_index()
    detail = ', '.join(f'{month}: {count}' for month, count in counts.items())
    if missing.any():
        report('WARN', f'Confirmed Open-Meteo ERA5 source gap: {missing.sum()} TCWV days remain NaN; {monsoon_missing.sum()} affected monsoon days ({detail}); no filling or interpolation')
    report('PASS', 'TCWV matches all-nine-point cached aggregation; incomplete days remain NaN')
    report('PASS', f'Complete merged output: {len(data)} days; Stage 1 unchanged, only tcwv_mm and oni added')


def check_event():
    # A separate cached diagnostic can run even if another year has missing hours.
    reference = cached_era(c.TRAIN_START, c.TRAIN_END)
    require_dates(reference.to_frame(), c.TRAIN_START, c.TRAIN_END)
    if reference.index.max() > pd.Timestamp(c.TRAIN_END):
        raise ValueError('Test-period data entered ERA5 climatology')
    case = cached_era(c.TCWV_EVENT_START, c.TCWV_EVENT_END)
    percentile = tcwv_percentile(case, reference)
    report('PASS', f'ERA5 seasonal reference restricted to {c.TRAIN_START} through {c.TRAIN_END}; monsoon days only')
    for date, value in percentile.items():
        report('PASS' if value >= c.TCWV_EVENT_PERCENTILE else 'WARN',
               f'{date.date()}: TCWV {case.loc[date]:.2f} mm; seasonal percentile {value:.2f}')
    # Every historical prefix must score identically with the frozen reference.
    for end in case.index:
        pd.testing.assert_series_equal(tcwv_percentile(case.loc[:end], reference), percentile.loc[:end])
    report('PASS', 'TCWV prefix invariance: future input days do not change earlier features')


def main():
    report('WARN', 'ERA5 vs GFS percentile correlation deferred until after Monday; GFS calibration not run')
    for check in (check_oni, check_output, check_event):
        try:
            check()
        except (OSError, ValueError, KeyError, AssertionError) as error:
            report('FAIL', str(error))
    pd.DataFrame(results).to_csv(c.STAGE1B_REPORT, index=False)
    statuses = [r['status'] for r in results]
    print(f"\nSummary: {statuses.count('PASS')} PASS, {statuses.count('WARN')} WARN, {statuses.count('FAIL')} FAIL")
    return int('FAIL' in statuses)


if __name__ == '__main__':
    sys.exit(main())
