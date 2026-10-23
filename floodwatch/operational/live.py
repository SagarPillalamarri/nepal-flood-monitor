"""Snapshot-bound operational inference and separate SQLite score table."""
import json
import sqlite3
from floodwatch.core import config as c
from floodwatch.operational.runtime import score,dependencies,OUT


def rescore(snapshot):
    from floodwatch.live.live_poll import rebuild
    frame,summary=rebuild(snapshot,full_history=True)
    result=score(frame,summary['today'],'gfs')
    return dict(snapshot_id=summary['snapshot_id'],fetch_time=summary['fetch_time'],**result)


def save_score(snapshot):
    from floodwatch.live.live_poll import DB,sha
    result=rescore(snapshot)
    manifest={'dependencies':dependencies(),'snapshot_hashes':{name:sha(snapshot/name) for name in ('manifest.json','weather.json','flood.json','weather_history.json') if (snapshot/name).exists()}}
    (snapshot/'operational_provenance.json').write_text(json.dumps(manifest,indent=2))
    (snapshot/'operational_score.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    with sqlite3.connect(DB) as db:
        db.execute('CREATE TABLE IF NOT EXISTS operational_scores (snapshot_id TEXT PRIMARY KEY, fetch_time TEXT NOT NULL, result_json TEXT NOT NULL)')
        db.execute('INSERT OR REPLACE INTO operational_scores VALUES (?,?,?)',(result['snapshot_id'],result['fetch_time'],json.dumps(result,allow_nan=False)))
    return result


def load_score(snapshot):
    from floodwatch.live.live_poll import DB
    path=snapshot/'operational_score.json'
    if not path.exists():return None
    result=json.loads(path.read_text())
    with sqlite3.connect(DB) as db:
        row=db.execute('SELECT result_json FROM operational_scores WHERE snapshot_id=?',(result['snapshot_id'],)).fetchone()
    if row is None or json.loads(row[0])!=result:raise ValueError('Operational snapshot/SQLite mismatch')
    return result
