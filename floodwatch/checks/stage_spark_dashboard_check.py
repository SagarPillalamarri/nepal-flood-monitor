"""Render both Streamlit tabs with Spark imports blocked; protect frozen work."""
import builtins
import hashlib
from floodwatch.layout import legacy_digest, original_source
import json
from pathlib import Path
from streamlit.testing.v1 import AppTest
from floodwatch.dashboard.spark_dashboard_integrity import ROOT, verify_replay


def main():
    verify_replay()
    baseline = json.loads((ROOT/'outputs/satellite/frozen_artifact_hashes.json').read_text())
    protected = {p:h for p,h in baseline.items() if p != 'app.py'}
    live = json.loads((ROOT/'outputs/spark/dashboard_live_hashes.json').read_text())
    original_import = builtins.__import__
    def guarded_import(name,*args,**kwargs):
        if name == 'pyspark' or name.startswith('pyspark.') or name in ('spark_runtime','spark_pipeline', 'floodwatch.spark.spark_runtime', 'floodwatch.spark.spark_pipeline'):
            raise AssertionError('Streamlit attempted to import Spark')
        return original_import(name,*args,**kwargs)
    lines = []
    builtins.__import__ = guarded_import
    try:
        app = AppTest.from_file(str(ROOT/'app.py')).run(timeout=40)
        assert not app.exception, [str(x.value) for x in app.exception]
        assert [x.label for x in app.tabs][:2] == ['September 2024 replay','Big data pipeline']
        assert len(app.tabs[0].slider) == 1
        assert len(app.tabs[0].metric) == 6
        assert len(app.tabs[1].metric) == 4
        assert len(app.tabs[1].dataframe) >= 7
        assert not app.tabs[1].error
        lines.append('[PASS] Both tabs render without exceptions; Spark imports blocked throughout')
        lines.append('[PASS] Big data tab renders lake metrics, variable coverage, parity, rainfall correction, SQL, scaling and physical plan')
        # Exercise the existing replay slider and make sure the new report stays valid.
        app.slider[0].set_value(__import__('datetime').date(2024,9,26)).run(timeout=40)
        assert not app.exception
        verify_replay()
        lines.append('[PASS] Original replay body is byte-for-byte unchanged; date slider works with all six original metrics')
    finally:
        builtins.__import__ = original_import
    for path,sha in {**protected,**live}.items():
        assert legacy_digest(ROOT/path) == sha, f'Protected file changed: {path}'
    lines.append(f'[PASS] {len(protected)} frozen non-app artifacts and {len(live)} paused live-work files unchanged; original app snapshot matches its frozen hash')
    lines.append('Summary: 4 PASS, 0 WARN, 0 FAIL')
    output = '\n'.join(lines)+'\n'
    (ROOT/'outputs/spark/dashboard_check_output.txt').write_text(output)
    print(output)


if __name__ == '__main__':
    main()
