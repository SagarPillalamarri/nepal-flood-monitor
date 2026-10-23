"""Offline checks of separate operational artifacts and fixed decision rules."""
import json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from floodwatch.core import config as c
from floodwatch.core.features import fit_reference,build_features
from floodwatch.pipeline.stage3_models import load_data
from floodwatch.pipeline.stage_sat_fetch import verify_frozen
from floodwatch.operational.calibration import OUT

def main():
 source,daily,events=load_data();lines=[]
 def ok(text):lines.append('[PASS] '+text)
 table=pd.read_csv(OUT/'transfer_correlations.csv');assert table.set_index('variable').loc['rain_3d','correlation']<.5;assert table.set_index('variable').loc[['rain_7d','rain_14d'],'correlation'].ge(.5).all();assert table.set_index('variable').loc['tcwv_2025_OOS','correlation']>=.7;ok('Fixed transfer decision rules select TCWV, rain 7d and 14d only')
 for h in (1,3):
  meta=json.loads((OUT/f'metadata_{h}d.json').read_text());pred=pd.read_csv(OUT/f'predictions_{h}d.csv',parse_dates=['date']);v=pd.read_csv(OUT/f'validation_{h}d.csv',parse_dates=['date']);assert v.date.max().year<=2019;assert meta['training_end']=='2019-12-31';assert daily.loc[pred.date,'flood'].eq(0).all();assert daily.loc[v.date,'flood'].eq(0).all();assert not ((pred.split=='test')&pred.date.between('2024-09-01','2024-09-30')).any();f=build_features(source,fit_reference(source));m=XGBClassifier();m.load_model(OUT/f'model_{h}d.json');np.testing.assert_allclose(m.predict_proba(f.loc[pred.date,meta['columns']])[:,1],pred.score,rtol=0,atol=1e-7);np.testing.assert_array_equal(pred.alarm,pred.score>=meta['threshold']);ok(f'{h}d: validation years, case separation, no ongoing floods, saved scores and thresholds verified')
 ok(f'{verify_frozen()} original research artifact hashes verified')
 lines.append('Summary: 4 PASS, 0 WARN, 0 FAIL');text='\n'.join(lines)+'\n';(OUT/'operational_check_output.txt').write_text(text);print(text)
if __name__=='__main__':main()
