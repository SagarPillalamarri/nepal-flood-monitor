"""Offline presentation provenance and chart-caption checks."""
import ast,hashlib,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from floodwatch.core import config as c
from floodwatch.dashboard.precursor import panel_data
from floodwatch.live.live_poll import latest,THRESHOLD
from floodwatch.operational.live import load_score


def main():
 lines=[]
 def check(name,fn):
  try:detail=fn();lines.append('[PASS] '+name+': '+str(detail))
  except Exception as e:lines.append('[FAIL] '+name+': '+str(e))
 def numbers():
  p=panel_data();saved=load_score(latest());f=saved['features'];assert p['water_pct']==f['tcwv_pct'];assert p['rain7_pct']==f['rain_7d_pct'];assert p['rain14_pct']==f['rain_14d_pct'];assert p['river_flow']==f['discharge'];assert p['distance_to_flood']==THRESHOLD-f['discharge']
  t=pd.read_csv(c.STAGE2_CONDITIONAL);t=t[t.population.eq('all_monsoon_days')&t.horizon_days.eq(3)].set_index('condition');b=t.loc['all_days'];a=t.loc['tcwv_pct_ge_90'];z=t.loc['tcwv_pct_ge_95'];w=f['tcwv_pct'];n=int(b.n_days-a.n_days) if w<90 else int(a.n_days-z.n_days) if w<95 else int(z.n_days);k=int(b.onset_within_horizon_days-a.onset_within_horizon_days) if w<90 else int(a.onset_within_horizon_days-z.onset_within_horizon_days) if w<95 else int(z.onset_within_horizon_days)
  assert p['n_days']==n and p['n_flood']==k and p['probability']==k/n and p['base_probability']==float(b.probability)
  assert p['high_probability']==float(z.probability) and p['high_n_days']==int(z.n_days) and p['high_n_flood']==int(z.onset_within_horizon_days)
  assert p['flow_percent_of_flood']==100*f['discharge']/THRESHOLD
  # Independent seasonal midrank of the training river series.
  q=pd.read_csv(c.STAGE1_FILE,index_col=0,parse_dates=True).loc['2001':'2019',c.GLOFAS_CHOSEN];q=q[q.index.month.isin(c.MONSOON_MONTHS)];day=pd.Timestamp(p['date']);rd=pd.to_datetime('2000-'+q.index.strftime('%m-%d')).dayofyear;d=pd.Timestamp('2000-'+day.strftime('%m-%d')).dayofyear;distance=np.abs(rd-d);pool=np.round(q[np.minimum(distance,366-distance)<=15].dropna().to_numpy(),8);value=round(f['discharge'],8);pct=100*((pool<value).sum()+.5*(pool==value).sum())/len(pool);assert p['river_percentile']==pct
  from floodwatch.live.observed_poll import latest_observed,rebuild_observed
  _,_,obs=rebuild_observed(latest_observed());assert p['gauge']['percent']==100*obs['latest_water_level_m']/obs['warning_level_m'];assert p['gauge']['time']==obs['latest_time_nepal']
  return 'All panel values reproduce raw saved features, disjoint training counts, seasonal flow ranks and timestamped gauge data exactly; only displayed rounding applied'
 check('Precursor numeric provenance',numbers)
 def freshness():
  p=panel_data();saved=load_score(latest());_,_,obs=__import__('floodwatch.live.observed_poll',fromlist=['rebuild_observed']).rebuild_observed(__import__('floodwatch.live.observed_poll',fromlist=['latest_observed']).latest_observed())
  time=pd.Timestamp(saved['fetch_time'])
  for hours in (6,6.001,30,96):
   synthetic=dict(obs,latest_time_nepal=(time-pd.Timedelta(hours=hours)).isoformat())
   with patch('floodwatch.dashboard.precursor.rebuild_observed',return_value=(None,None,synthetic)):
    result=panel_data(panel_time=time)
   assert result['gauge_stale']==(hours>6)
   assert result['summary'].endswith('(gauge reading out of date)')==(hours>6)
   assert ('Press Refresh gauge.' in result['text'])==(hours>6)
  from floodwatch.dashboard.live_monitor import refresh_sources
  with patch('floodwatch.dashboard.live_monitor.poll') as weather,patch('floodwatch.live.observed_poll.poll_observed') as gauge:
   assert refresh_sources()==[];assert weather.call_count==gauge.call_count==1
  with patch('floodwatch.dashboard.live_monitor.poll',side_effect=ValueError('weather unavailable')),patch('floodwatch.live.observed_poll.poll_observed') as gauge:
   assert len(refresh_sources())==1;assert gauge.call_count==1
  return 'Strict >6-hour boundary, stale summary suffix, comparison counts and dual-source refresh verified; rounding is presentation-only'
 check('Staleness and combined refresh',freshness)
 def captions():
  count=0
  for name in ['app.py','floodwatch/dashboard/live_monitor.py','floodwatch/dashboard/observed_monitor.py','floodwatch/dashboard/spark_dashboard.py']:
   tree=ast.parse((c.ROOT/name).read_text())
   for node in ast.walk(tree):
    if not isinstance(node,(ast.Module,ast.FunctionDef,ast.With,ast.If,ast.For,ast.Try)):continue
    for field in ('body','orelse','finalbody'):
     statements=getattr(node,field,[])
     for i,stmt in enumerate(statements):
      if isinstance(stmt,ast.Expr) and isinstance(stmt.value,ast.Call) and isinstance(stmt.value.func,ast.Attribute) and stmt.value.func.attr in ('line_chart','bar_chart','altair_chart'):
       count+=1;following=statements[i+1];assert isinstance(following,ast.Expr) and isinstance(following.value,ast.Call) and isinstance(following.value.func,ast.Name) and following.value.func.id=='caption',(name,stmt.lineno)
  assert count==8,count
  app=AppTest.from_file(str(c.ROOT/'app.py')).run(timeout=90);assert not app.exception
  for tab in app.tabs:
   assert any('What this page shows:' in x.value for x in tab.info)
   assert any(x.label=='Glossary' for x in tab.expander)
  metric_count=0
  def visit(node,parent=None):
   nonlocal metric_count
   if getattr(node,'type',None)=='metric':
    metric_count+=1;assert any(x.type=='caption' and x.value.startswith('What it shows:') and 'How to read it:' in x.value and 'Source:' in x.value for x in parent.children.values())
   for child in getattr(node,'children',{}).values():visit(child,node)
  visit(app._tree);assert metric_count>=15,metric_count
  return f'{count} chart call sites have explicit explanatory source captions; {metric_count} rendered metrics have explanatory captions; all tabs have introductions and glossaries'
 check('Charts, metrics and plain-language layout',captions)
 def frozen():
  baseline=json.loads((c.ROOT/'outputs/usability/frozen_data_hashes.json').read_text());mutable={'data/live/live_scores.sqlite','data/live/observed_latest.json'};baseline={p:w for p,w in baseline.items() if p not in mutable};changed=[p for p,w in baseline.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=w];assert not changed,changed
  return f'{len(baseline)} data, snapshot, model and template files byte-for-byte unchanged'
 check('Frozen data and calculations',frozen)
 text='\n'.join(lines)+f"\nSummary: {sum(x.startswith('[PASS]') for x in lines)} PASS, 0 WARN, {sum(x.startswith('[FAIL]') for x in lines)} FAIL\n";print(text);(c.ROOT/'outputs/usability/check_output.txt').write_text(text);return int(any(x.startswith('[FAIL]') for x in lines))
if __name__=='__main__':
 with patch('requests.sessions.Session.request',side_effect=AssertionError('Network forbidden')):raise SystemExit(main())
