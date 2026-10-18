"""Read-only GFS input compatibility probes before any frozen-model live work."""
from datetime import datetime, timezone
import json
import sys
import requests
import pandas as pd
import config as c
from stage_sat_fetch import verify_frozen

REQUIRED = ('precipitation', 'total_column_integrated_water_vapour',
            'soil_moisture_0_to_7cm', 'soil_moisture_7_to_28cm',
            'soil_moisture_28_to_100cm')
NATIVE_SOIL = ('soil_moisture_0_to_10cm','soil_moisture_10_to_40cm','soil_moisture_40_to_100cm')
OUT = c.ROOT/'outputs/live_preflight'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    frozen_count = verify_frozen()
    rows = []
    for name,url,extra in [
        ('historical','https://historical-forecast-api.open-meteo.com/v1/forecast',
         {'start_date':'2024-09-27','end_date':'2024-09-27'}),
        ('forecast','https://api.open-meteo.com/v1/forecast',{'forecast_days':1}),
    ]:
        params = {'latitude':c.BASIN_POINTS[0][0], 'longitude':c.BASIN_POINTS[0][1],
                  'models':c.GFS_MODEL,'timezone':c.TIMEZONE,
                  'hourly':','.join(REQUIRED+NATIVE_SOIL),**extra}
        response = requests.get(url,params=params,timeout=60)
        (OUT/f'{name}_response.json').write_text(response.text)
        (OUT/f'{name}_request.json').write_text(json.dumps({
            'url':url,'params':params,'http_status':response.status_code,
            'fetched_at_utc':datetime.now(timezone.utc).isoformat()},indent=2))
        response.raise_for_status()
        payload = response.json()
        hourly = payload['hourly']
        for variable in REQUIRED+NATIVE_SOIL:
            values = hourly.get(variable,[])
            non_null = sum(value is not None for value in values)
            rows.append({'endpoint':name,'variable':variable,'required':variable in REQUIRED,
                         'rows':len(values),'non_null':non_null,
                         'complete':len(values)==len(hourly['time']) and non_null==len(hourly['time'])})
        print(f'{name}: HTTP {response.status_code}, {len(hourly["time"])} hourly rows',flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT/'availability.csv',index=False)
    print(table.to_string(index=False))
    missing = table[table.required & ~table.complete]
    lines = [f'[PASS] {frozen_count} frozen model/scoring artifacts verified unchanged']
    for row in missing.itertuples():
        lines.append(f'[FAIL] {row.endpoint}: required {row.variable}: {row.non_null}/{row.rows} non-null hours')
    if len(missing):
        lines.append('[STOP] Exact required GFS inputs are unavailable; no substitution, calibration or live scoring attempted.')
    lines.append('Native GFS soil values above are availability evidence only; they were not used as substitute model inputs.')
    lines.append(f'Summary: 1 PASS, 0 WARN, {len(missing)} FAIL')
    verify_frozen()
    text='\n'.join(lines)+'\n'
    print(text)
    (OUT/'report.txt').write_text(text)
    return int(bool(len(missing)))


if __name__=='__main__':
    sys.exit(main())
