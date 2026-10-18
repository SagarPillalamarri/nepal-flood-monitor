"""Verify the authorized tab wrapper without replacing the original freeze hash."""
import hashlib
import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def verify_replay():
    original = (ROOT/'outputs/spark/dashboard_original_app.py').read_text()
    baseline = json.loads((ROOT/'outputs/satellite/frozen_artifact_hashes.json').read_text())
    assert hashlib.sha256(original.encode()).hexdigest() == baseline['app.py'], 'Original app snapshot differs from the frozen baseline'
    current = (ROOT/'app.py').read_text()
    prefix,body = original.split("st.title(",1)
    body = 'st.title('+body
    declaration = "replay_tab, pipeline_tab = st.tabs(['September 2024 replay', 'Big data pipeline'])"
    if 'with monitor_tab:' in current:
        declaration = "replay_tab, pipeline_tab, monitor_tab = st.tabs(['September 2024 replay', 'Big data pipeline', 'Live monitor'])"
    expected_prefix = prefix+declaration+'\nwith replay_tab:\n'
    assert current.startswith(expected_prefix), 'Original imports/page configuration changed'
    replay,tail = current[len(expected_prefix):].split('\nwith pipeline_tab:\n',1)
    assert textwrap.dedent(replay) == body, 'September replay body changed'
    if '\nwith monitor_tab:\n' in tail:
        tail,monitor = tail.split('\nwith monitor_tab:\n',1)
        assert monitor == '    from live_monitor import render_monitor\n    render_monitor()\n', 'Unexpected live wrapper'
    assert tail == '    from spark_dashboard import render_pipeline\n    render_pipeline()\n', 'Unexpected code in pipeline wrapper'
    return True
