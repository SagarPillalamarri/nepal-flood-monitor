"""Training-only discrimination using frozen scalers/templates; no fitting."""
import json
import numpy as np
import pandas as pd
from floodwatch.core import config as c
from floodwatch.core.analogue_config import RESEARCH_CLOSE_MATCH_CUTOFF,OPERATIONAL_CLOSE_MATCH_CUTOFF
from floodwatch.pipeline.stage3_models import load_data
from floodwatch.pipeline.stage4_analogues import _window
from floodwatch.operational.runtime import OUT,features,ANALOGUE_COLS


def main():
 source,daily,events=load_data();ev=events[events.training_eligible.astype(bool)&events.in_monsoon.astype(bool)];train_onsets=events[events.start.between(c.TRAIN_START,c.TRAIN_END)].start
 historical=pd.read_csv(c.STAGE3_FEATURES,index_col=0,parse_dates=True);op=features(source.rename(columns={c.GLOFAS_CHOSEN:'discharge_m3_s'}),'era5');records=[];summary=[]
 for kind,f in [('research',historical),('operational',op)]:
  if kind=='research':
   columns=list(c.STAGE4_MATCH_FEATURES);sc=pd.read_csv(c.STAGE4_SCALERS).set_index('feature');matrix=(f[columns]-sc.training_mean)/sc.training_std;t=pd.read_csv(c.STAGE4_TEMPLATES);t=t[pd.to_datetime(t.event_onset).between(c.TRAIN_START,c.TRAIN_END)];vectors=t[[f'day_{d:02d}__{col}' for d in range(1,15) for col in columns]].to_numpy();ids=t.event_id.to_numpy();leads=t.days_before_flood.to_numpy();old=RESEARCH_CLOSE_MATCH_CUTOFF
  else:
   columns=ANALOGUE_COLS;sc=json.loads((OUT/'analogue_scaler.json').read_text());matrix=(f[columns]-pd.Series(sc['mean']))/pd.Series(sc['std']);t=[x for x in json.loads((OUT/'analogue_templates.json').read_text()) if pd.Timestamp(c.TRAIN_START)<=pd.Timestamp(x['event_onset'])<=pd.Timestamp(c.TRAIN_END)];vectors=np.array([x['vector'] for x in t]);ids=np.array([x['event_id'] for x in t]);leads=np.array([x['lead_days'] for x in t]);old=OPERATIONAL_CLOSE_MATCH_CUTOFF
  # Ordinary days exclude windows around ANY training-period onset, including non-monsoon onsets.
  ordinary=matrix.loc[c.TRAIN_START:c.TRAIN_END].index;ordinary=ordinary[ordinary.month.isin(c.MONSOON_MONTHS)]
  forbidden=set()
  for onset in train_onsets:forbidden.update(pd.date_range(onset-pd.Timedelta(days=14),onset+pd.Timedelta(days=7)))
  queries=[('ordinary',d,None) for d in ordinary if d not in forbidden]
  queries += [('pre_flood',e.start-pd.Timedelta(days=lead),int(e.event_id)) for e in ev.itertuples() for lead in range(1,8) if (e.start-pd.Timedelta(days=lead)).month in c.MONSOON_MONTHS]
  for group,date,held in queries:
   v=_window(matrix,date,14);mask=ids!=held if held is not None else np.ones(len(ids),dtype=bool)
   if v is None:records.append(dict(analogue=kind,group=group,date=date,held_event=held,distance=np.nan,matched_lead=np.nan));continue
   eligible=np.flatnonzero(mask);dist=np.linalg.norm(vectors[mask]-v,axis=1);best=min(range(len(dist)),key=lambda i:(dist[i],ids[eligible[i]],leads[eligible[i]]));j=eligible[best];records.append(dict(analogue=kind,group=group,date=date,held_event=held,distance=dist[best],matched_lead=int(leads[j]),matched_event=int(ids[j])))
  old=float(pd.read_csv(OUT/'analogue_close_match_report_before_discrimination.csv').query('analogue == @kind').close_match_cutoff.iloc[0]);data=pd.DataFrame([x for x in records if x['analogue']==kind]);normal=data[data.group.eq('ordinary')].distance.dropna().sort_values().to_numpy();allowed=int(np.floor(.1*int(data.group.eq('ordinary').sum())));cutoff=float(np.nextafter(normal[allowed],-np.inf));pre=data[data.group.eq('pre_flood')].dropna(subset=['distance']);hits=(pre.distance.le(cutoff)&pre.matched_lead.le(3));rate=float(hits.mean())
  for group in ('ordinary','pre_flood'):
   d=data.loc[data.group.eq(group),'distance'];valid=d.dropna();summary.append(dict(analogue=kind,group=group,n_queries=len(d),n_valid=len(valid),n_missing=int(d.isna().sum()),median=valid.median(),p10=valid.quantile(.1),p90=valid.quantile(.9),old_cutoff=old,old_close_rate=float(valid.le(old).sum()/len(d)),old_close_rate_valid_only=float(valid.le(old).mean()),new_cutoff=cutoff,new_close_rate=float(valid.le(cutoff).sum()/len(d)),new_close_rate_valid_only=float(valid.le(cutoff).mean()),pre_flood_hits=int(hits.sum()),pre_flood_n=len(pre),pre_flood_hit_rate=rate,display_enabled=rate>=.25))
 pd.DataFrame(records).to_csv(OUT/'discrimination_distances.csv',index=False);pd.DataFrame(summary).to_csv(OUT/'discrimination_summary.csv',index=False);print(pd.DataFrame(summary).to_string(index=False))
if __name__=='__main__':main()
