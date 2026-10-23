"""One manual refresh of both dashboard sources, with a saved panel report."""
import json
from pathlib import Path
from floodwatch.dashboard.live_monitor import refresh_sources
from floodwatch.dashboard.precursor import panel_data
from floodwatch.core import config as c


def main():
    errors=refresh_sources()
    panel=panel_data()
    out=c.ROOT/'outputs/usability'
    (out/'precursor_today.txt').write_text(panel['text']+'\n')
    (out/'precursor_today.json').write_text(json.dumps(panel,indent=2))
    (out/'refresh_result.json').write_text(json.dumps({'errors':errors,'snapshot_id':panel.get('snapshot_id'),'panel_time':panel.get('panel_time')},indent=2))
    print('\nREFRESH ERRORS:',errors)
    print(panel['text'])
    return int(bool(errors))


if __name__=='__main__':raise SystemExit(main())
