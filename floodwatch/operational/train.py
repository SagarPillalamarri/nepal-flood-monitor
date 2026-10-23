"""Separate operational training with fixed transfer-selected inputs."""
import json
import pandas as pd
from floodwatch.core import config as c
from floodwatch.core.features import fit_reference,build_features
from floodwatch.core.targets import onset_series,onset_target,eligible_rows
from floodwatch.pipeline.stage3_models import load_data,make_model,choose_threshold,evaluate_predictions
from floodwatch.pipeline.stage_sat_fetch import verify_frozen
from floodwatch.operational.calibration import OUT

COLS=['rain_7d_pct','rain_14d_pct','tcwv_pct','discharge','discharge_rise_1d','discharge_rise_3d']
def main():
 verify_frozen();source,daily,events=load_data();reference=fit_reference(source);full=build_features(source,reference)[COLS];reference.to_csv(OUT/'era5_reference.csv');allmetrics=[]
 for h in (1,3):
  target=onset_target(onset_series(source.index,events),h);validation=[]
  for fold,(end,start,stop) in enumerate(c.VALIDATION_FOLDS,1):
   f=build_features(source.loc[:stop],fit_reference(source,end))[COLS];tr=eligible_rows(daily,f,target,c.TRAIN_START,end,h);va=eligible_rows(daily,f,target,start,stop,h);assert daily.loc[tr.union(va),'flood'].eq(0).all();m=make_model(target.loc[tr]);m.fit(f.loc[tr],target.loc[tr]);validation.append(pd.DataFrame({'date':va,'target':target.loc[va],'score':m.predict_proba(f.loc[va])[:,1],'fold':fold}))
  v=pd.concat(validation);threshold,tuning=choose_threshold(v.target,v.score,c.XGB_SCORE_THRESHOLDS);v.to_csv(OUT/f'validation_{h}d.csv',index=False);tuning.to_csv(OUT/f'threshold_tuning_{h}d.csv',index=False)
  tr=eligible_rows(daily,full,target,c.TRAIN_START,c.TRAIN_END,h);m=make_model(target.loc[tr]);m.fit(full.loc[tr],target.loc[tr]);m.save_model(OUT/f'model_{h}d.json');records=[]
  for split,start,end,exclude in [('test',c.TEST_START,c.TEST_END,True),('case',c.CASE_REPORT_START,c.CASE_REPORT_END,False)]:
   dates=eligible_rows(daily,full,target,start,end,h,exclude_case=exclude);assert daily.loc[dates,'flood'].eq(0).all();score=m.predict_proba(full.loc[dates])[:,1];records.append(pd.DataFrame({'date':dates,'variant':'operational','split':split,'target':target.loc[dates],'score':score,'alarm':score>=threshold,'threshold':threshold}))
  pred=pd.concat(records);pred.to_csv(OUT/f'predictions_{h}d.csv',index=False);metrics,leads=evaluate_predictions(pred,events,h);allmetrics.append(metrics);leads.to_csv(OUT/f'event_leads_{h}d.csv',index=False);(OUT/f'metadata_{h}d.json').write_text(json.dumps(dict(columns=COLS,threshold=threshold,training_end=c.TRAIN_END),indent=2))
 pd.concat(allmetrics).to_csv(OUT/'metrics.csv',index=False);print(pd.concat(allmetrics).to_string(index=False));verify_frozen();print('PASS: chronological folds, ongoing-flood exclusion, frozen research hashes')
if __name__=='__main__':main()
