"""Shared operational feature, model and analogue inference; never trains models."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from floodwatch.core import config as c
from floodwatch.core.features import tcwv_percentile

OUT=c.ROOT/'outputs/operational'
MAP=c.ROOT/'data/live/quantile_maps/tcwv_gfs_era5.json'
COLS=['rain_7d_pct','rain_14d_pct','tcwv_pct','discharge','discharge_rise_1d','discharge_rise_3d']
ANALOGUE_COLS=['tcwv_pct','rain_7d_pct','rain_14d_pct','discharge_rise_1d','discharge_rise_3d']
CAVEAT='Operational model: water vapour (GFS mapped to ERA5), 7/14-day rain percentiles, GloFAS discharge. 1-day: 3 of 6 test floods caught. 3-day: not reliable.'


def features(data,source):
    if source not in ('era5','gfs'):raise ValueError(source)
    if not data.index.equals(pd.date_range(data.index.min(),data.index.max(),name=data.index.name)):
        raise ValueError('Continuous daily calendar required')
    ref=pd.read_csv(OUT/('era5_reference.csv' if source=='era5' else 'gfs_reference.csv'),index_col=0,parse_dates=True)
    if source=='era5' and ref.index.max()>pd.Timestamp(c.TRAIN_END):raise ValueError('Historical reference leakage')
    days=data.index[data.index.month.isin(c.MONSOON_MONTHS)]
    result=pd.DataFrame(index=days)
    for n in (7,14):
        values=data.precip_mm.rolling(n,min_periods=n).sum().reindex(days)
        result[f'rain_{n}d_pct']=tcwv_percentile(values,ref[f'rain_{n}d'])
    values=data.tcwv_mm.reindex(days)
    era=pd.read_csv(OUT/'era5_reference.csv',index_col=0,parse_dates=True)
    if source=='gfs':
        mapping=json.loads(MAP.read_text())
        values=pd.Series(np.interp(values,mapping['gfs'],mapping['era5']),index=days).where(values.notna())
    result['tcwv_pct']=tcwv_percentile(values,era.tcwv)
    result['discharge']=data.discharge_m3_s.reindex(days)
    for n in (1,3):result[f'discharge_rise_{n}d']=data.discharge_m3_s.diff(n).div(n).reindex(days)
    result.index.name='date'
    return result[COLS]


def match(f,date,cutoff=None):
    date=pd.Timestamp(date)
    scaler=json.loads((OUT/'analogue_scaler.json').read_text())
    mean=pd.Series(scaler['mean']);std=pd.Series(scaler['std'])
    dates=pd.date_range(date-pd.Timedelta(days=13),date)
    query=((f.reindex(dates)[ANALOGUE_COLS]-mean)/std).to_numpy().ravel()
    if not np.isfinite(query).all():return {'status':'incomplete_window','event_onset':None,'lead_days':None}
    templates=json.loads((OUT/'analogue_templates.json').read_text())
    eligible=[t for t in templates if pd.Timestamp(t['event_onset'])<date and (cutoff is None or pd.Timestamp(t['event_onset'])<=pd.Timestamp(cutoff))]
    if not eligible:return {'status':'no_templates','event_onset':None,'lead_days':None}
    best=min(eligible,key=lambda t:(float(np.linalg.norm(query-np.array(t['vector']))),t['event_id'],t['lead_days']))
    return {k:best[k] for k in ('event_id','event_onset','lead_days')}|{'status':'matched','distance':float(np.linalg.norm(query-np.array(best['vector'])))}


def score(data,date,source='gfs'):
    date=pd.Timestamp(date);f=features(data,source)
    result={'date':str(date.date()),'source':source,'features':{},'scores':{},'analogue':None}
    if date not in f.index:
        result['status']='outside_monsoon';return result
    row=f.loc[date];result['features']={k:None if pd.isna(v) else float(v) for k,v in row.items()}
    blocked=pd.isna(row.discharge) or row.discharge>445.191
    for h in (1,3):
        meta=json.loads((OUT/f'metadata_{h}d.json').read_text())
        model=XGBClassifier();model.load_model(OUT/f'model_{h}d.json')
        value=None if blocked else float(model.predict_proba(f.loc[[date],meta['columns']])[:,1][0])
        result['scores'][f'{h}d']={'score':value,'threshold':meta['threshold'],'alert':None if value is None else value>=meta['threshold']}
    result['status']='ongoing_flood_or_missing_discharge' if blocked else 'scored'
    result['analogue']=match(f,date,'2023-12-31' if date.strftime('%Y-%m')=='2024-09' else None)
    from floodwatch.core.analogue_labels import label
    a=result['analogue'];a.update(label(a.get('distance'),a.get('event_onset'),a.get('lead_days'),'operational'))
    return result


def dependencies():
    paths=[c.ROOT/'floodwatch/core/analogue_config.py',MAP,OUT/'era5_reference.csv',OUT/'gfs_reference.csv',OUT/'analogue_scaler.json',OUT/'analogue_templates.json']
    paths += [OUT/f'{kind}_{h}d.json' for kind in ('model','metadata') for h in (1,3)]
    return {str(p.relative_to(c.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
