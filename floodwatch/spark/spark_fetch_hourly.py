"""Optional nine-point Stage 1 hourly recovery; stop immediately on rate limiting."""
import json
import time
from pathlib import Path
import pandas as pd
import requests
from floodwatch.core import config as c


def main():
    out = c.DATA_RAW/'spark_weather'
    out.mkdir(exist_ok=True)
    status = []
    variables = ['precipitation', 'soil_moisture_0_to_7cm', 'soil_moisture_7_to_28cm', 'soil_moisture_28_to_100cm']
    for year in range(2001, 2026):
        path = out/f'stage1_best_match_{year}.json'
        if path.exists():
            print('Reuse', path.name, flush=True)
            continue
        params = dict(latitude=','.join(str(x[0]) for x in c.BASIN_POINTS),
                      longitude=','.join(str(x[1]) for x in c.BASIN_POINTS),
                      hourly=','.join(variables), timezone=c.TIMEZONE,
                      start_date=f'{year}-01-01', end_date=f'{year}-12-31')
        print('Request hourly Stage 1 weather', year, flush=True)
        r = requests.get(c.ERA5_URL, params=params, timeout=180)
        if r.status_code == 429:
            status.append(dict(year=year, status='RATE_LIMIT', reason=r.text[:500]))
            print('Rate limited: stopping all further downloads; cached data only.', flush=True)
            break
        r.raise_for_status()
        payload = r.json()
        expected = pd.date_range(f'{year}-01-01', f'{year+1}-01-01', inclusive='left', freq='h')
        if not isinstance(payload, list) or len(payload) != 9:
            raise ValueError('Expected nine point responses')
        for point in payload:
            h = point['hourly']
            if not pd.DatetimeIndex(pd.to_datetime(h['time'])).equals(expected):
                raise ValueError('Truncated response; not cached')
            if any(len(h[v]) != len(expected) for v in variables):
                raise ValueError('Truncated variable; not cached')
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(payload))
        tmp.replace(path)
        status.append(dict(year=year, status='DOWNLOADED', hours=len(expected),
                           non_null={v:sum(x is not None for p in payload for x in p['hourly'][v]) for v in variables}))
        time.sleep(3)
    report = c.ROOT/'outputs/spark'
    report.mkdir(parents=True, exist_ok=True)
    (report/'download_status.json').write_text(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
