"""Compare every API data value with saved project sources; no network pollers."""
import csv,hashlib,json,sys,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'web/server'));sys.path.insert(0,str(ROOT))
from read_project import payload
from floodwatch.live.live_poll import latest
from floodwatch.operational.live import load_score
from floodwatch.core import config as c
BASE='http://127.0.0.1:8787/api/'

def normalized(x):
 if isinstance(x,dict):return {k:normalized(v) for k,v in x.items() if k not in ('generated_at','panel_time','gauge_age_hours')}
 if isinstance(x,list):return [normalized(v) for v in x]
 return x

def compare(a,b,path='root'):
 if isinstance(a,dict):
  assert a.keys()==b.keys(),path
  for k in a:compare(a[k],b[k],path+'.'+k)
 elif isinstance(a,list):
  assert len(a)==len(b),path
  for i,(v,w) in enumerate(zip(a,b)):compare(v,w,path+f'[{i}]')
 else:assert a==b,(path,a,b)

def main():
 checks=[]
 for name in ('replay','pipeline','live'):
  with urllib.request.urlopen(BASE+name,timeout=60) as response:actual=json.load(response)
  expected=payload(name)
  compare(normalized(actual),normalized(expected));assert actual['source'] and actual['generated_at']
  checks.append(f'[PASS] /api/{name}: every data value matches project source reader exactly; provenance present')
 # Independent anchors against directly opened files, avoiding a self-comparison-only test.
 with urllib.request.urlopen(BASE+'replay?start=2024-09-20&end=2024-09-30') as response:replay=json.load(response)
 with open(ROOT/'outputs/stage5/labelled_september_2024_replay.csv') as f:source={r['date']:r for r in csv.DictReader(f)}
 for row in replay['days']:
  for key in ('precip_mm','tcwv_pct','risk_1d','risk_3d','threshold_1d','threshold_3d','discharge_m3_s'):
   assert row[key]==(float(source[row['date']][key]) if source[row['date']][key] else None),(row['date'],key)
 with urllib.request.urlopen(BASE+'live') as response:live=json.load(response)
 assert live['operational']==load_score(latest())
 with open(latest()/'daily.csv') as f:daily=list(csv.DictReader(f))
 assert len(daily)==len(live['days'])
 for a,b in zip(live['days'],daily):
  for key in ('precip_mm','tcwv_mm','discharge_m3_s'):assert a[key]==(float(b[key]) if b[key] else None)
 checks.append('[PASS] Independent CSV anchors: replay scores/thresholds and live values match exact floats; operational JSON matches SQLite-backed loader')
 # Every existing saved data/model/snapshot file must remain unchanged by GETs.
 from floodwatch.pipeline.stage_sat_fetch import verify_frozen
 assert verify_frozen()==80
 for path,wanted in json.loads((ROOT/'outputs/operational/operational_frozen_hashes.json').read_text()).items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==wanted
 checks.append('[PASS] 80 research manifest entries and frozen operational model/threshold hashes unchanged')
 text='\n'.join(checks)+'\nSummary: 5 PASS, 0 FAIL\n';print(text);(ROOT/'web/parity_output.txt').write_text(text)
if __name__=='__main__':main()
