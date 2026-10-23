"""Derive distance cutoffs from training LOEO hits only; no template/model fitting."""
from pathlib import Path
import pandas as pd,numpy as np,json,hashlib
from floodwatch.operational.runtime import features,ANALOGUE_COLS,OUT
from floodwatch.pipeline.stage3_models import load_data
from floodwatch.pipeline.stage4_analogues import _window
from floodwatch.core import config as c
source,daily,events=load_data();f=features(source.rename(columns={c.GLOFAS_CHOSEN:'discharge_m3_s'}),'era5');train=f.loc[c.TRAIN_START:c.TRAIN_END];loo=pd.read_csv(OUT/'analogue_loeo.csv');dist=[]
for r in loo.itertuples():
 held=events.loc[events.event_id.eq(r.held_event)].iloc[0];excluded=pd.date_range(held.start-pd.Timedelta(days=15),held.start+pd.Timedelta(days=int(held.span_days)-1));fit=train.loc[~train.index.isin(excluded),ANALOGUE_COLS];matrix=(f[ANALOGUE_COLS]-fit.mean())/fit.std(ddof=0);other=events.loc[events.event_id.eq(r.matched_event)].iloc[0];a=_window(matrix,held.start-pd.Timedelta(days=r.query_lead),14);b=_window(matrix,other.start-pd.Timedelta(days=r.matched_lead),14);dist.append(float(np.linalg.norm(a-b)))
loo['distance']=dist;loo.to_csv(OUT/'analogue_loeo_distances.csv',index=False)
research=pd.read_csv(c.STAGE4_DIR/'loo_evaluation.csv');research=research[research.window_days.eq(14)&research.hit_within_3_days];op=loo[loo.hit];rc=float(research.distance.quantile(.75));oc=float(op.distance.quantile(.75))
p=Path('floodwatch/core/analogue_config.py');p.write_text(f'''"""Presentation confidence gates; no model or analogue-template changes."""\n# 75th percentile (linear interpolation) of HIT distances in 2001–2019\n# leave-one-event-out training runs, selected 14-day windows only.\n# Research: {len(research)} hits; operational: {len(op)} hits. No test/case/live data.\nRESEARCH_CLOSE_MATCH_CUTOFF = {rc!r}\nOPERATIONAL_CLOSE_MATCH_CUTOFF = {oc!r}\n''')
print('Research',rc,'Operational',oc)
