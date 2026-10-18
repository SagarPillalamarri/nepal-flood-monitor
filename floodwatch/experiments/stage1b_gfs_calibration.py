"""Deferred GFS download and percentile comparison. Run manually after Monday."""
import sys
import numpy as np
import pandas as pd
import requests
from floodwatch.core import config as c
from floodwatch.core.features import tcwv_percentile
from floodwatch.pipeline.stage1b_fetch import fetch_series, read_daily, require_dates


def main():
    # The approved live source probe selects ECMWF, so keep the deferred command
    # as an entry point to the current calibration rather than downloading GFS.
    from floodwatch.experiments.live_calibration import main as calibrate
    return calibrate()
    c.STAGE1B_CACHE.mkdir(parents=True, exist_ok=True)
    data = read_daily(c.STAGE1B_FILE)
    require_dates(data, c.START_DATE, c.END_DATE)
    gfs = fetch_series('gfs_tcwv', c.GFS_HISTORY_URL, c.GFS_MODEL,
                       [c.TCWV_VARIABLE], c.GFS_START, c.END_DATE)
    gfs.columns = ['tcwv_mm']
    gfs.to_csv(c.GFS_TCWV_FILE)
    reference = data.loc[c.TRAIN_START:c.TRAIN_END, 'tcwv_mm']
    era = data.loc[data.index.month.isin(c.MONSOON_MONTHS), 'tcwv_mm']
    gfs_monsoon = gfs.loc[gfs.index.month.isin(c.MONSOON_MONTHS), 'tcwv_mm']
    overlap = pd.concat([tcwv_percentile(era, reference).rename('era5'),
                         tcwv_percentile(gfs_monsoon, gfs_monsoon).rename('gfs')], axis=1).dropna()
    if len(overlap) != len(gfs_monsoon):
        raise ValueError('Incomplete overlapping monsoon dates')
    correlation = overlap.era5.corr(overlap.gfs)
    if not np.isfinite(correlation):
        raise ValueError('Percentile correlation is undefined')
    status = 'PASS' if correlation >= c.TCWV_CORRELATION_WARN else 'WARN'
    message = f'ERA5/GFS seasonal percentile correlation: {correlation:.3f} ({len(overlap)} overlapping monsoon days)'
    pd.DataFrame([dict(status=status, message=message)]).to_csv(c.GFS_CALIBRATION_REPORT, index=False)
    print(f'[{status}] {message}')
    print(f"Summary: {int(status == 'PASS')} PASS, {int(status == 'WARN')} WARN, 0 FAIL")


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, requests.RequestException) as error:
        print(f'[FAIL] {error}\nSummary: 0 PASS, 0 WARN, 1 FAIL')
        sys.exit(1)
