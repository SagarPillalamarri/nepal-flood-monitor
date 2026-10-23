"""Read-only bridge to saved project outputs. Never fits features or scores."""
import json,sys
from pathlib import Path
from datetime import datetime,timezone
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

def csv(path):
    frame=pd.read_csv(path,float_precision='round_trip')
    return frame.astype(object).where(pd.notna(frame),None).to_dict(orient='records')
def read(path):return json.loads(path.read_text())
def payload(kind,start=None,end=None):
    sources=[]
    def table(name):sources.append(name);return csv(ROOT/name)
    def js(name):sources.append(name);return read(ROOT/name)
    if kind=='replay':
        rows=table('outputs/stage5/labelled_september_2024_replay.csv')
        data={'days':[r for r in rows if (not start or r['date']>=start) and (not end or r['date']<=end)]}
    elif kind=='pipeline':
        data={'lake':js('outputs/spark/lake_stats.json'),'parity':table('outputs/spark/column_parity.csv'),'events':table('outputs/spark/events_per_year.csv'),'conditional':table('outputs/spark/conditional_probabilities.csv'),'scaling':js('outputs/spark/scaling_trials.json'),'rainfall_shift':js('outputs/spark/rainfall_diagnosis.json'),'coverage':table('outputs/spark/variable_coverage.csv'),'plan':(ROOT/'outputs/spark/daily_physical_plan.txt').read_text()};sources.append('outputs/spark/daily_physical_plan.txt')
    elif kind=='live':
        from floodwatch.live.live_poll import latest
        from floodwatch.live.observed_poll import latest_observed
        from floodwatch.operational.live import load_score
        from floodwatch.dashboard.precursor import panel_data
        snapshot=latest();gauge=latest_observed();data={'snapshot_id':None,'days':[],'operational':None,'gauge':None,'precursor':None,'typical_october_flow':js('outputs/live/october_sanity.json')['historical']['q_north']['median']}
        if snapshot:
            relative=str(snapshot.relative_to(ROOT));data.update(snapshot_id=snapshot.name,days=table(relative+'/daily.csv'),summary=js(relative+'/summary.json'),operational=load_score(snapshot),precursor=panel_data(snapshot));sources.extend([relative+'/operational_score.json','data/live/live_scores.sqlite','outputs/stage2/conditional_probabilities.csv','data/processed/stage1_daily.csv'])
        if gauge:
            relative=str(gauge.relative_to(ROOT));data['gauge']={'summary':js(relative+'/observed_summary.json'),'hourly':table(relative+'/hourly.csv')}
    else:raise ValueError('Unknown endpoint')
    return {'source':sources,'generated_at':datetime.now(timezone.utc).isoformat(),**data}
if __name__=='__main__':print(json.dumps(payload(*sys.argv[1:]),allow_nan=False))
