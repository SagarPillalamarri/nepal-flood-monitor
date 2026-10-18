"""Phase A manual poller: raw GFS/GloFAS inputs only, no model scoring."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import requests
import config as c

LIVE = c.ROOT/'data/live'
SNAPSHOTS = LIVE/'snapshots'
DB = LIVE/'live_scores.sqlite'
THRESHOLD = 445.191
BANNER = 'Live inputs only. Model risk scores need inputs on the ERA5 scale; see README.'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rebuild(snapshot):
    """Deterministic offline parsing shared by the dashboard and checks."""
    snapshot = Path(snapshot)
    meta = json.loads((snapshot/'manifest.json').read_text())
    today = pd.Timestamp(meta['today'])
    dates = pd.date_range(today-pd.Timedelta(days=14),today+pd.Timedelta(days=7),name='date')
    payload = json.loads((snapshot/'weather.json').read_text())
    if not isinstance(payload,list) or len(payload) != len(c.BASIN_POINTS):
        raise ValueError('Expected all nine weather responses')
    expected_hours = pd.date_range(meta['weather_start'],pd.Timestamp(meta['weather_end'])+pd.Timedelta(hours=23),freq='h')
    points = []
    for loc in payload:
        hourly = loc['hourly']
        index = pd.DatetimeIndex(pd.to_datetime(hourly['time']))
        if not index.equals(expected_hours):
            raise ValueError('Incomplete or duplicate weather timestamps')
        rain = pd.Series(hourly['precipitation'],index=index,dtype=float)
        rain.index = rain.index-pd.Timedelta(hours=1)
        tcwv = pd.Series(hourly['total_column_integrated_water_vapour'],index=index,dtype=float)
        frame = pd.DataFrame(index=dates)
        frame['precip_mm'] = rain.resample('D').sum(min_count=24).reindex(dates)
        frame['tcwv_mm'] = tcwv.resample('D').mean().where(tcwv.resample('D').count()==24).reindex(dates)
        points.append(frame)
    weather = sum(points)/len(points)  # Missing at any point propagates.
    flood = json.loads((snapshot/'flood.json').read_text())['daily']
    q = pd.Series(flood['river_discharge'],index=pd.to_datetime(flood['time']),dtype=float)
    if q.index.has_duplicates or not q.index.equals(dates):
        raise ValueError('Incomplete GloFAS date coverage')
    weather['discharge_m3_s'] = q.reindex(dates)
    weather['period'] = np.where(dates<=today,'Past / today (modelled)','Forecast')
    weather.index.name = 'date'
    forecast = weather.loc[today+pd.Timedelta(days=1):]
    exceed = forecast.index[forecast.discharge_m3_s > THRESHOLD].strftime('%Y-%m-%d').tolist()
    if forecast.discharge_m3_s.isna().any():
        statement = 'Forecast discharge is incomplete; threshold exceedance cannot be ruled out.'
    elif exceed:
        statement = 'Forecast exceeds the threshold on '+', '.join(exceed)
    else:
        statement = 'No forecast day exceeds the flood threshold'
    summary = dict(snapshot_id=meta['snapshot_id'],fetch_time=meta['fetch_time'],today=meta['today'],
                   threshold_m3_s=THRESHOLD,statement=statement,exceedance_dates=exceed,
                   forecast_max_discharge_m3_s=float(forecast.discharge_m3_s.max()) if forecast.discharge_m3_s.notna().any() else None,
                   weather_model='gfs_seamless',flood_model='seamless_v4',risk_1d=None,risk_3d=None,
                   status='inputs_only',missing_weather_values=int(weather[['precip_mm','tcwv_mm']].isna().sum().sum()))
    return weather,summary


def poll():
    now = datetime.now(timezone.utc)
    today = pd.Timestamp(now.astimezone(ZoneInfo(c.TIMEZONE)).date())
    snapshot_id = now.strftime('%Y%m%dT%H%M%S_%fZ')
    path = SNAPSHOTS/snapshot_id
    path.mkdir(parents=True)
    start = str((today-pd.Timedelta(days=14)).date())
    end = str((today+pd.Timedelta(days=7)).date())
    weather_end = str((today+pd.Timedelta(days=8)).date())
    weather_params = dict(latitude=','.join(str(x[0]) for x in c.BASIN_POINTS),
                          longitude=','.join(str(x[1]) for x in c.BASIN_POINTS),
                          hourly='precipitation,total_column_integrated_water_vapour',models='gfs_seamless',
                          timezone=c.TIMEZONE,start_date=start,end_date=weather_end)
    lat,lon = c.GLOFAS_POINTS[c.GLOFAS_CHOSEN]
    flood_params = dict(latitude=lat,longitude=lon,daily='river_discharge',models='seamless_v4',
                        start_date=start,end_date=end,timezone='GMT')
    meta = dict(snapshot_id=snapshot_id,fetch_time=now.isoformat(),today=str(today.date()),
                weather_start=start,weather_end=weather_end,display_end=end,
                timezone=c.TIMEZONE,glofas_day_timezone='UTC',glofas_chosen=c.GLOFAS_CHOSEN,
                basin_points=c.BASIN_POINTS,requests={
                    'weather':dict(url='https://api.open-meteo.com/v1/forecast',params=weather_params),
                    'flood':dict(url='https://flood-api.open-meteo.com/v1/flood',params=flood_params)})
    (path/'manifest.json').write_text(json.dumps(meta,indent=2))
    try:
        hashes = {}
        for name,request in meta['requests'].items():
            response = requests.get(request['url'],params=request['params'],timeout=120)
            # Preserve the exact response bytes, including unsuccessful responses.
            (path/f'{name}.json').write_bytes(response.content)
            response.raise_for_status()
            response.json()
            hashes[f'{name}.json'] = sha(path/f'{name}.json')
        meta['response_hashes'] = hashes
        (path/'manifest.json').write_text(json.dumps(meta,indent=2))
        frame,summary = rebuild(path)
        frame.to_csv(path/'daily.csv')
        (path/'summary.json').write_text(json.dumps(summary,indent=2))
        with sqlite3.connect(DB) as db:
            db.execute('CREATE TABLE IF NOT EXISTS runs (snapshot_id TEXT PRIMARY KEY, fetch_time TEXT NOT NULL, snapshot_path TEXT NOT NULL, summary_json TEXT NOT NULL)')
            db.execute('INSERT INTO runs VALUES (?,?,?,?)',(snapshot_id,summary['fetch_time'],str(path),json.dumps(summary)))
        print(json.dumps(summary,indent=2),flush=True)
        return path
    except Exception as error:
        (path/'error.json').write_text(json.dumps(dict(error=str(error),fetch_time=meta['fetch_time']),indent=2))
        raise


def latest():
    if not DB.exists():
        return None
    with sqlite3.connect(DB) as db:
        row = db.execute('SELECT snapshot_path FROM runs ORDER BY fetch_time DESC LIMIT 1').fetchone()
    return Path(row[0]) if row else None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print('Saved snapshot:',poll())
