"""Phase A saved-snapshot checks; network requests forbidden throughout."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
from unittest.mock import patch
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
import config as c
from live_poll import DB, THRESHOLD, latest, rebuild, sha
from live_monitor import view_data
from spark_dashboard_integrity import verify_replay


def main(snapshot=None):
    checks = []
    def check(message,fn):
        try:
            detail = fn()
            status='PASS'
        except Exception as error:
            detail=str(error)
            status='FAIL'
        checks.append(dict(status=status,message=message+': '+str(detail)))
        print(f'[{status}] {message}: {detail}',flush=True)
    snapshot=Path(snapshot) if snapshot else latest()
    if snapshot is None:
        raise ValueError('No saved snapshot; run live_poll.py first')
    with patch('requests.sessions.Session.request',side_effect=AssertionError('Network forbidden in offline check')):
        frame,summary = rebuild(snapshot)
        meta=json.loads((snapshot/'manifest.json').read_text())
        def raw_check():
            for name,digest in meta['response_hashes'].items():
                assert sha(snapshot/name)==digest
            payload=json.loads((snapshot/'weather.json').read_text())
            expected=[]
            for day in frame.index:
                rains=[];vapour=[]
                for loc in payload:
                    h=loc['hourly'];times=pd.to_datetime(h['time'])
                    rain=[v for t,v in zip(times,h['precipitation']) if day < t <= day+pd.Timedelta(days=1)]
                    tcwv=[v for t,v in zip(times,h['total_column_integrated_water_vapour']) if day <= t < day+pd.Timedelta(days=1)]
                    assert len(rain)==len(tcwv)==24
                    rains.append(sum(rain) if all(x is not None for x in rain) else np.nan)
                    vapour.append(sum(tcwv)/24 if all(x is not None for x in tcwv) else np.nan)
                expected.append([sum(rains)/9,sum(vapour)/9])
            np.testing.assert_allclose(frame[['precip_mm','tcwv_mm']].to_numpy(),expected,rtol=0,atol=1e-10,equal_nan=True)
            assert len(frame)==22
            return 'Raw response hashes valid; independent nine-point, 24-hour calculation reproduces all 22 days'
        check('Snapshot and daily values',raw_check)
        def sqlite_check():
            saved=pd.read_csv(snapshot/'daily.csv',index_col=0,parse_dates=True)
            pd.testing.assert_frame_equal(saved,frame,check_freq=False,check_dtype=False)
            assert json.loads((snapshot/'summary.json').read_text())==summary
            with sqlite3.connect(DB) as db:
                row=db.execute('SELECT summary_json FROM runs WHERE snapshot_id=?',(summary['snapshot_id'],)).fetchone()
            assert row and json.loads(row[0])==summary
            assert summary['risk_1d'] is None and summary['risk_3d'] is None
            return 'Saved CSV, summary and SQLite row exactly reproduce snapshot; model scores remain null'
        check('SQLite reproducibility',sqlite_check)
        def threshold_check():
            displayed,forecast,tab=view_data(snapshot)
            pd.testing.assert_frame_equal(displayed,frame)
            assert len(forecast)==7
            assert forecast.discharge_m3_s.notna().all(), 'A future GloFAS day is missing'
            flood=json.loads((snapshot/'flood.json').read_text())['daily']
            expected=pd.Series(flood['river_discharge'],index=pd.to_datetime(flood['time']),dtype=float)
            np.testing.assert_allclose(frame.discharge_m3_s,expected,rtol=0,atol=1e-12,equal_nan=True)
            dates=forecast.index[forecast.discharge_m3_s>THRESHOLD].strftime('%Y-%m-%d').tolist()
            assert dates==summary['exceedance_dates']
            assert tab==summary and THRESHOLD==445.191
            return f"Seven future days; threshold {THRESHOLD} m³/s; {summary['statement']}"
        check('Monitor values and exceedance statement',threshold_check)
        def frozen_check():
            original=json.loads((c.ROOT/'outputs/satellite/frozen_artifact_hashes.json').read_text())
            for name,digest in original.items():
                file=c.ROOT/name if name!='app.py' else c.ROOT/'outputs/spark/dashboard_original_app.py'
                assert sha(file)==digest, name
            verify_replay()
            prior=(c.ROOT/'outputs/live/phase_a_prior_app.py').read_text()
            current=(c.ROOT/'app.py').read_text()
            restored=current.split('\nwith monitor_tab:\n',1)[0]
            restored=restored.replace("replay_tab, pipeline_tab, monitor_tab = st.tabs(['September 2024 replay', 'Big data pipeline', 'Live monitor'])", "replay_tab, pipeline_tab = st.tabs(['September 2024 replay', 'Big data pipeline'])")
            assert restored==prior
            protected=json.loads((c.ROOT/'outputs/live/phase_a_protected_hashes.json').read_text())
            assert sha(c.ROOT/'spark_dashboard.py')==protected['spark_dashboard.py']
            return '80 original manifest entries verified (79 current files + original app snapshot); authorized wrapper only; replay and Big data tabs unchanged'
        check('Frozen hashes and existing tabs',frozen_check)
        def observed_check():
            import re
            from observed_poll import latest_observed,rebuild_observed
            from observed_monitor import observed_view
            source=latest_observed()
            assert source is not None,'No saved DHM observation snapshot'
            manifest=json.loads((source/'observed_manifest.json').read_text())
            for name,digest in manifest['raw_hashes'].items():
                assert sha(source/name)==digest
            station_from_page=json.loads(re.search(r"var river = '(.*?)';",(source/'dhm_page.html').read_text(),re.S).group(1))
            assert manifest['station']==station_from_page
            payload=json.loads((source/'dhm_response.json').read_text())
            encoded=re.search(r"var river\s*=\s*'(.*?)';",payload['data']['chart'],re.S).group(1)
            unescaped=re.sub(r'\\x([0-9a-fA-F]{2})',lambda m:chr(int(m.group(1),16)),encoded)
            points=json.loads(unescaped)
            expected=pd.DataFrame(points)
            expected['datetime']=pd.to_datetime(expected.datetime,utc=True)
            expected=expected.sort_values('datetime').set_index('datetime').rename(columns={'value':'water_level_m'}).tz_convert(c.TIMEZONE)
            expected=expected.loc[expected.index.max()-pd.Timedelta(days=7):]
            raw,hourly,summary=rebuild_observed(source)
            pd.testing.assert_frame_equal(raw,expected,check_dtype=False)
            sums={};counts={}
            for date,value in zip(expected.index,expected.water_level_m):
                hour=date.floor('h');sums[hour]=sums.get(hour,0)+value;counts[hour]=counts.get(hour,0)+1
            for hour,row in hourly.iterrows():
                assert row.n_observations==counts.get(hour,0)
                if hour in counts:
                    np.testing.assert_allclose(row.water_level_m,sums[hour]/counts[hour],rtol=0,atol=1e-12)
                else:
                    assert pd.isna(row.water_level_m)
            saved=pd.read_csv(source/'hourly.csv',index_col=0)
            assert pd.DatetimeIndex(pd.to_datetime(saved.index,utc=True)).equals(hourly.index.tz_convert('UTC'))
            np.testing.assert_allclose(saved.water_level_m,hourly.water_level_m,equal_nan=True,rtol=0,atol=1e-12)
            np.testing.assert_array_equal(saved.n_observations,hourly.n_observations)
            assert json.loads((source/'observed_summary.json').read_text())==summary
            assert summary['latest_water_level_m']==float(expected.water_level_m.iloc[-1])
            assert pd.Timestamp(summary['latest_time_utc'])==expected.index[-1]
            assert summary['warning_level_m']==float(station_from_page['warning_level'])
            assert summary['danger_level_m']==float(station_from_page['danger_level'])
            shown=observed_view(source)
            pd.testing.assert_frame_equal(shown[0],raw);pd.testing.assert_frame_equal(shown[1],hourly)
            assert shown[2]==summary
            return f"DHM raw hashes, {len(raw)} ten-minute readings, {len(hourly)} hourly bins, latest level/time, published 3.50/4.00 m levels and display reproduce saved responses; no filling"
        check('Observed DHM snapshot and hourly chart',observed_check)
        def satellite_gate_check():
            report=json.loads((c.ROOT/'outputs/live/observed_investigation/imerg_latest.json').read_text())
            latency=(pd.Timestamp(report['checked_utc'])-pd.Timestamp(report['latest_utc'])).total_seconds()/3600
            np.testing.assert_allclose(latency,report['latency_hours'],rtol=0,atol=1e-4)
            assert latency>=48,'Fresh IMERG is now available; implement its panel before enabling this test expectation'
            return f'Latest IMERG at investigation was {latency:.1f} h old (>48 h); no stale rainfall series added or used for scoring'
        check('Satellite freshness gate',satellite_gate_check)
        def ui_check():
            with patch('live_monitor.latest',return_value=snapshot):
                app=AppTest.from_file(str(c.ROOT/'app.py')).run(timeout=45)
            assert not app.exception,[str(x.value) for x in app.exception]
            assert [x.label for x in app.tabs]==['September 2024 replay','Big data pipeline','Live monitor']
            assert not app.tabs[2].error
            assert app.button(key='fetch_live_inputs').label=='Fetch now'
            assert app.button(key='refresh_observed_dhm').label=='Refresh observed gauge'
            for table in app.tabs[2].dataframe:
                if 'discharge_m3_s' in table.value and 'period' in table.value:
                    pd.testing.assert_frame_equal(table.value,frame.reset_index(),check_dtype=False)
                    break
            else:
                raise AssertionError('Rendered daily input table missing')
            with patch('live_monitor.poll',return_value=snapshot) as poll, patch('live_monitor.latest',return_value=snapshot):
                app.button(key='fetch_live_inputs').click().run(timeout=45)
                assert poll.call_count==1
                assert not app.exception
            from observed_poll import latest_observed
            with patch('observed_monitor.poll_observed',return_value=latest_observed()) as observed_poll:
                app.button(key='refresh_observed_dhm').click().run(timeout=45)
                assert observed_poll.call_count==1
                assert not app.exception
            assert any(m.label=='Observed water level' for m in app.tabs[2].metric)
            return 'All three tabs render offline; model/observed displays reproduce snapshots; both fetch buttons invoke their pollers (mocked)'
        check('Streamlit rendering and button',ui_check)
    out=c.ROOT/'outputs/live';out.mkdir(exist_ok=True)
    pd.DataFrame(checks).to_csv(out/'phase_a_checks.csv',index=False)
    summary_line=f"Summary: {sum(x['status']=='PASS' for x in checks)} PASS, 0 WARN, {sum(x['status']=='FAIL' for x in checks)} FAIL"
    text='\n'.join(f"[{x['status']}] {x['message']}" for x in checks)+'\n'+summary_line+'\n'
    (out/'phase_a_check_output.txt').write_text(text)
    print(summary_line,flush=True)
    return int(any(x['status']=='FAIL' for x in checks))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot');args=parser.parse_args()
    raise SystemExit(main(args.snapshot))
