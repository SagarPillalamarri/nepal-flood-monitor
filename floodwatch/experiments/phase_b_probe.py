"""Bounded Phase B archive gate, run only after Phase A passes."""
import json
import pandas as pd
import requests
from floodwatch.core import config as c


def main():
    out=c.ROOT/'outputs/live/phase_b_probe';out.mkdir(parents=True,exist_ok=True)
    variables=['precipitation','total_column_integrated_water_vapour','soil_moisture_0_to_7cm','soil_moisture_7_to_28cm','soil_moisture_28_to_100cm']
    rows=[]
    for year in (2017,2021,2023,2025):
        date=f'{year}-09-27'
        params=dict(latitude=','.join(str(p[0]) for p in c.BASIN_POINTS),longitude=','.join(str(p[1]) for p in c.BASIN_POINTS),
                    hourly=','.join(variables),models='ecmwf_ifs',timezone=c.TIMEZONE,start_date=date,end_date=date)
        response=requests.get(c.ERA5_URL,params=params,timeout=120)
        (out/f'{date}.json').write_bytes(response.content)
        if response.status_code!=200:
            rows.append(dict(date=date,variable='all',status=f'HTTP {response.status_code}',detail=response.text[:300]))
            print(response.status_code,response.text[:300],flush=True)
            break
        payload=response.json()
        if not isinstance(payload,list) or len(payload)!=9:
            raise ValueError('Expected nine ECMWF points')
        for variable in variables:
            valid=[sum(x is not None for x in p['hourly'][variable]) for p in payload]
            row=dict(date=date,variable=variable,minimum_valid_hours=min(valid),total_valid_hours=sum(valid),expected_hours=216,status='PASS' if min(valid)==24 else 'WARN')
            rows.append(row)
            print(row,flush=True)
    pd.DataFrame(rows).to_csv(out/'coverage.csv',index=False)


if __name__=='__main__':
    main()
