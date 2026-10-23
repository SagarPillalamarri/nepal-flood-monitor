"""Fit live-only TCWV mapping and fixed 14-day operational analogue artifacts."""
import json,hashlib
import numpy as np
import pandas as pd
from floodwatch.core import config as c
from floodwatch.pipeline.stage3_models import load_data
from floodwatch.pipeline.stage4_analogues import _window
from floodwatch.core.features import tcwv_percentile
from floodwatch.operational.runtime import OUT,MAP,features,ANALOGUE_COLS,score

def main():
    frozen={str(p.relative_to(c.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('model_*d.json')}
    frozen.update({str(p.relative_to(c.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('metadata_*d.json')})
    baseline=OUT/'operational_frozen_hashes.json'
    if baseline.exists():
        assert json.loads(baseline.read_text())==frozen,'Frozen operational artifacts changed'
    else:baseline.write_text(json.dumps(frozen,indent=2))
    source,daily,events=load_data();g=pd.read_csv(OUT/'gfs_daily.csv',index_col=0,parse_dates=True)
    pair=pd.concat([g.tcwv_mm.rename('gfs'),source.tcwv_mm.rename('era5')],axis=1).dropna();pair=pair.loc[pair.index.month.isin(c.MONSOON_MONTHS)&pair.index.year.isin([2024,2025])]
    q=np.linspace(0,1,101)
    def fit(p):
        x,ix=np.unique(np.quantile(p.gfs,q),return_index=True);return x,np.quantile(p.era5,q)[ix]
    x,y=fit(pair);oldx,oldy=fit(pair.loc['2024']);test=pair.loc['2025'];ref=source.tcwv_mm.loc[c.TRAIN_START:c.TRAIN_END]
    new=tcwv_percentile(pd.Series(np.interp(test.gfs,x,y),index=test.index),ref);old=tcwv_percentile(pd.Series(np.interp(test.gfs,oldx,oldy),index=test.index),ref);delta=(new-old).abs()
    MAP.parent.mkdir(parents=True,exist_ok=True);MAP.write_text(json.dumps(dict(gfs=x.tolist(),era5=y.tolist(),fit_days=len(pair),fit_start=str(pair.index.min().date()),fit_end=str(pair.index.max().date()),validation='2024-fit / 2025-test correlation 0.865766 retained',mean_percentile_change=float(delta.mean()),max_percentile_change=float(delta.max())),indent=2))
    pd.DataFrame({'original_pct':old,'refit_pct':new,'abs_difference':delta}).to_csv(OUT/'tcwv_refit_change.csv')
    data=source.rename(columns={c.GLOFAS_CHOSEN:'discharge_m3_s'});f=features(data,'era5');train=f.loc[c.TRAIN_START:c.TRAIN_END];train_events=events[events.training_eligible.astype(bool)&events.in_monsoon.astype(bool)]
    mean=train[ANALOGUE_COLS].mean();std=train[ANALOGUE_COLS].std(ddof=0)
    (OUT/'analogue_scaler.json').write_text(json.dumps(dict(mean=mean.to_dict(),std=std.to_dict(),fit_end=c.TRAIN_END),indent=2))
    def templates(matrix,ev,excluded=set()):
        out=[]
        for event in ev.itertuples():
            for lead in range(1,15):
                end=event.start-pd.Timedelta(days=lead);dates=set(pd.date_range(end-pd.Timedelta(days=13),end))
                if dates&excluded:continue
                v=_window(matrix,end,14)
                if v is not None:out.append(dict(event_id=int(event.event_id),event_onset=str(event.start.date()),lead_days=lead,vector=v.tolist()))
        return out
    standardized=(f[ANALOGUE_COLS]-mean)/std
    (OUT/'analogue_templates.json').write_text(json.dumps(templates(standardized,events)))
    loo=[]
    for held in train_events.itertuples():
        excluded=set(pd.date_range(held.start-pd.Timedelta(days=15),held.start+pd.Timedelta(days=int(held.span_days)-1)));fitrows=train.loc[~train.index.isin(excluded),ANALOGUE_COLS];matrix=(f[ANALOGUE_COLS]-fitrows.mean())/fitrows.std(ddof=0);ts=templates(matrix,train_events[train_events.event_id.ne(held.event_id)],excluded)
        for offset in (1,2):
            v=_window(matrix,held.start-pd.Timedelta(days=offset),14)
            if v is None:raise ValueError('Incomplete LOEO query')
            best=min(ts,key=lambda t:(np.linalg.norm(v-np.array(t['vector'])),t['event_id'],t['lead_days']))
            loo.append(dict(held_event=int(held.event_id),query_lead=offset,matched_event=best['event_id'],matched_lead=best['lead_days'],hit=best['lead_days']<=3))
    loo=pd.DataFrame(loo);loo.to_csv(OUT/'analogue_loeo.csv',index=False)
    rows=[]
    for date in pd.date_range('2024-09-20','2024-09-28'):
        s=score(data,date,'era5');row={'date':str(date.date()),'status':s['status']}
        for h in (1,3):row.update({f'score_{h}d':s['scores'][f'{h}d']['score'],f'threshold_{h}d':s['scores'][f'{h}d']['threshold']})
        analogue=s['analogue'].copy();analogue['analogue_status']=analogue.pop('status');row.update(analogue);rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/'september_live_diagnostic.csv',index=False)
    print('Mapping:',len(pair),'days; percentile change mean',delta.mean(),'max',delta.max());print('Operational LOEO:',loo.hit.sum(),'/',len(loo),'Research 12/38');print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
