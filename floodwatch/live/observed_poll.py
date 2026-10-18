"""Anonymous DHM observations only; no model inputs or authentication changes."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import pandas as pd
import requests
from floodwatch.core import config as c

PAGE='https://dhm.gov.np/hydrology/hms-Single/50'
ENDPOINT='https://dhm.gov.np/site/getRiverWatchBySeriesId_Single'
LATEST=c.ROOT/'data/live/observed_latest.json'


def decode_rows(payload):
    if payload.get('status')!='success':
        raise ValueError('DHM did not return successful public chart data')
    chart=payload['data']['chart']
    match=re.search(r"var river\s*=\s*'(.*?)';",chart,re.S)
    if not match:
        raise ValueError('DHM chart data format changed')
    # Decode only the site's hex-escaped JSON string. Never execute JavaScript.
    text=re.sub(r'\\x([0-9a-fA-F]{2})',lambda m:chr(int(m.group(1),16)),match.group(1))
    rows=json.loads(text)
    if not isinstance(rows,list) or not rows:
        raise ValueError('DHM observations are empty')
    raw=pd.DataFrame(rows)
    raw['datetime']=pd.to_datetime(raw.datetime,utc=True)
    raw['value']=pd.to_numeric(raw.value,errors='raise')
    if raw.datetime.duplicated().any() or raw.value.isna().any():
        raise ValueError('DHM duplicate timestamps or null observations')
    return raw.sort_values('datetime').set_index('datetime').rename(columns={'value':'water_level_m'})


def rebuild_observed(snapshot):
    path=Path(snapshot)
    meta=json.loads((path/'observed_manifest.json').read_text())
    station=meta['station']
    if station['stationIndex']!='550.05' or station['series_id']!=927:
        raise ValueError('DHM station identity changed')
    raw=decode_rows(json.loads((path/'dhm_response.json').read_text()))
    local=raw.tz_convert(c.TIMEZONE)
    start=local.index.max()-pd.Timedelta(days=7)
    local=local.loc[start:]
    hourly=local.resample('h').agg(water_level_m=('water_level_m','mean'),n_observations=('water_level_m','count'))
    hourly['partial_hour']=hourly.n_observations.between(1,5)
    hourly.index.name='hour'
    summary=dict(station='Bagmati at Khokana',station_index='550.05',series_id=927,
                 latest_time_utc=raw.index[-1].isoformat(),latest_time_nepal=local.index[-1].isoformat(),
                 latest_water_level_m=float(raw.water_level_m.iloc[-1]),
                 warning_level_m=float(station['warning_level']) if station.get('warning_level') else None,
                 danger_level_m=float(station['danger_level']) if station.get('danger_level') else None,
                 fetched_at=meta['fetched_at'],n_observations=len(local),n_hourly_rows=len(hourly))
    return local,hourly,summary


def poll_observed():
    now=datetime.now(timezone.utc)
    path=c.ROOT/'data/live/snapshots'/('observed_'+now.strftime('%Y%m%dT%H%M%S_%fZ'))
    path.mkdir(parents=True)
    session=requests.Session()
    try:
        page=session.get(PAGE,timeout=60)
        (path/'dhm_page.html').write_bytes(page.content)
        page.raise_for_status()  # Stop on blocks; no user-agent/proxy workaround.
        station=json.loads(re.search(r"var river = '(.*?)';",page.text,re.S).group(1))
        token=re.search(r'name="csrf_test_name" value="([^"]+)"',page.text).group(1)
        date=re.search(r'name="date"[^>]*value="([^"]+)"',page.text).group(1)
        if station['stationIndex']!='550.05' or station['series_id']!=927:
            raise ValueError('Unexpected DHM station')
        response=session.post(ENDPOINT,data=dict(csrf_test_name=token,date=date,period='4',seriesid='927'),timeout=60)
        (path/'dhm_response.json').write_bytes(response.content)
        response.raise_for_status()
        meta=dict(fetched_at=now.isoformat(),page_url=PAGE,endpoint=ENDPOINT,
                  request=dict(date=date,period='4',seriesid='927'),station=station,
                  anonymous=True,timezone=c.TIMEZONE,
                  raw_hashes={name:hashlib.sha256((path/name).read_bytes()).hexdigest() for name in ['dhm_page.html','dhm_response.json']})
        (path/'observed_manifest.json').write_text(json.dumps(meta,indent=2))
        raw,hourly,summary=rebuild_observed(path)
        raw.to_csv(path/'observations.csv');hourly.to_csv(path/'hourly.csv')
        (path/'observed_summary.json').write_text(json.dumps(summary,indent=2))
        temporary=LATEST.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(snapshot_path=str(path),fetched_at=now.isoformat())))
        temporary.replace(LATEST)
        print(json.dumps(summary,indent=2),flush=True)
        return path
    except Exception as error:
        (path/'observed_error.json').write_text(json.dumps(dict(error=str(error),fetched_at=now.isoformat()),indent=2))
        raise


def latest_observed():
    return Path(json.loads(LATEST.read_text())['snapshot_path']) if LATEST.exists() else None


if __name__=='__main__':
    print('Saved observed snapshot:',poll_observed())
