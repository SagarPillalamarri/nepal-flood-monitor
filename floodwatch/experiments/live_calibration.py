"""Fit input quantile maps only; never train models or refit feature references."""
import json
import time
import pandas as pd
import numpy as np
from floodwatch.core import config as c
from floodwatch.experiments.live_common import LIVE, MAPS, REPORT, VARIABLES, get_json, weather_params, weather_daily, read_daily, mapped_inputs, verify_frozen, digest


def main():
    verify_frozen()
    MAPS.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    cache = LIVE/'calibration_cache'
    cache.mkdir(exist_ok=True)
    pieces = []
    for year in range(2017, 2026):
        start, end = f'{year}-05-01', f'{year}-10-31'
        path = cache/f'ecmwf_ifs_{year}.json'
        if path.exists():
            payload = json.loads(path.read_text())
        else:
            print(f'Downloading ECMWF IFS {year}, all nine points ...', flush=True)
            payload = get_json(c.ERA5_URL, weather_params('ecmwf_ifs', start_date=start, end_date=end))
            daily = weather_daily(payload, start, end)  # Validate before caching.
            if daily[list(VARIABLES)].count().min() == 0:
                raise ValueError(f'{year}: a requested variable is entirely null; response not cached')
            path.write_text(json.dumps(payload))
            time.sleep(2)
        pieces.append(weather_daily(payload, start, end))
    source = pd.concat(pieces).sort_index()
    source.to_csv(LIVE/'ecmwf_daily.csv')
    era = read_daily(c.STAGE1B_FILE)
    rows = []
    for column in [*VARIABLES, 'precip_max_point_mm']:
        pair = pd.concat([source[column].rename('source'), era[column].rename('era5')], axis=1).dropna()
        pair = pair.loc[pair.index.month.isin(c.MONSOON_MONTHS)]
        if len(pair) < 100:
            raise ValueError(f'Insufficient paired days for {column}: {len(pair)}')
        obj = dict(variable=column, source_model='ecmwf_ifs', forecast_model='ecmwf_ifs025',
                   method='seasonal +/-15 calendar days; empirical source midrank -> linear ERA5 inverse CDF',
                   dates=pair.index.strftime('%Y-%m-%d').tolist(), source=pair.source.tolist(), era5=pair.era5.tolist())
        (MAPS/f'{column}.json').write_text(json.dumps(obj))
    mapped = mapped_inputs(source)
    for column in [*VARIABLES, 'precip_max_point_mm']:
        pair = pd.concat([mapped[column].rename('mapped'), era[column].rename('era5')], axis=1).dropna()
        pair = pair.loc[pair.index.month.isin(c.MONSOON_MONTHS)]
        corr = pair.mapped.corr(pair.era5)
        rows.append(dict(variable=column, start=str(pair.index.min().date()), end=str(pair.index.max().date()), days=len(pair), correlation=corr,
                         status='PASS' if np.isfinite(corr) and corr >= .7 else 'WARN'))
    report = pd.DataFrame(rows)
    report.to_csv(REPORT/'calibration_correlations.csv', index=False)
    manifest = {p.name:digest(p) for p in MAPS.glob('*.json')}
    (REPORT/'mapping_hashes.json').write_text(json.dumps(manifest, indent=2))
    verify_frozen()
    print(report.to_string(index=False), flush=True)
    print('Mappings fitted. Correlations are descriptive in-sample input agreement, not model validation.', flush=True)


if __name__ == '__main__':
    main()
