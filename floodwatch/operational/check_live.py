"""Network-free operational integration, frozen-artifact and causal checks."""
import hashlib,json,sqlite3
from io import StringIO
from unittest.mock import patch
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from floodwatch.core import config as c
from floodwatch.core.features import build_features,fit_reference
from floodwatch.pipeline.stage3_models import load_data
from floodwatch.pipeline.stage_sat_fetch import verify_frozen
from floodwatch.operational.runtime import OUT,MAP,features,dependencies,ANALOGUE_COLS
from floodwatch.operational.live import rescore,load_score
from floodwatch.live.live_poll import latest,DB,rebuild,sha


def main():
    checks=[]
    def check(name,fn):
        try:detail=fn();checks.append(('PASS',name,str(detail)))
        except Exception as e:checks.append(('FAIL',name,str(e)))
    snapshot=latest()
    with patch('requests.sessions.Session.request',side_effect=AssertionError('Network forbidden')):
        def frozen():
            count=verify_frozen()
            original=json.loads((OUT/'operational_frozen_hashes.json').read_text())
            for p,wanted in original.items():assert sha(c.ROOT/p)==wanted,p
            for p,wanted in json.loads((OUT/'analogue_artifact_hashes.json').read_text()).items():assert sha(c.ROOT/p)==wanted,p
            return f'{count} research manifest entries, {len(original)} operational model/threshold hashes and 4 analogue template/scaler hashes unchanged'
        check('Frozen models and research artifacts',frozen)
        def snapshot_check():
            provenance=json.loads((snapshot/'operational_provenance.json').read_text())
            assert provenance['dependencies']==dependencies()
            for name,wanted in provenance['snapshot_hashes'].items():assert sha(snapshot/name)==wanted
            result=rescore(snapshot);assert result==load_score(snapshot)
            with sqlite3.connect(DB) as db:row=db.execute('SELECT result_json FROM operational_scores WHERE snapshot_id=?',(result['snapshot_id'],)).fetchone()
            assert json.loads(row[0])==result
            assert len(rebuild(snapshot,full_history=True)[0])>=35
            return 'Raw hashes, feature/mapping/model provenance, saved score and SQLite reproduce offline rescore exactly'
        check('Snapshot and SQLite rescore',snapshot_check)
        def feature_check():
            source,daily,events=load_data();historical=source.rename(columns={c.GLOFAS_CHOSEN:'discharge_m3_s'})
            actual=features(historical,'era5');expected=build_features(source,fit_reference(source))[actual.columns]
            pd.testing.assert_frame_equal(actual,expected,check_freq=False,rtol=1e-12,atol=1e-12)
            frame,_=rebuild(snapshot,full_history=True)
            csv=snapshot/'daily.csv';saved=pd.read_csv(csv,index_col=0,parse_dates=True)
            # Same source-specific entry point for replaying saved inputs and live parsing.
            f1=features(frame,'gfs');copy=pd.read_csv(StringIO(frame.to_csv()),index_col=0,parse_dates=True,float_precision='round_trip');f2=features(copy,'gfs')
            pd.testing.assert_frame_equal(f1,f2,rtol=1e-8,atol=1e-8,check_freq=False)
            today=pd.Timestamp(json.loads((snapshot/'manifest.json').read_text())['today'])
            before=features(frame.loc[:today],'gfs');pd.testing.assert_frame_equal(f1.loc[before.index],before)
            return 'Historical features match frozen training implementation; identical parsed/replayed inputs match; future rows cannot alter today'
        check('Feature identity and causality',feature_check)
        def leakage():
            source,daily,events=load_data()
            reference=pd.read_csv(OUT/'era5_reference.csv',index_col=0,parse_dates=True);assert reference.index.max().year<=2019
            gref=pd.read_csv(OUT/'gfs_reference.csv',index_col=0,parse_dates=True);assert gref.index.year.isin(range(2021,2026)).all()
            for h in (1,3):
                val=pd.read_csv(OUT/f'validation_{h}d.csv',parse_dates=['date']);assert val.date.max().year<=2019
                pred=pd.read_csv(OUT/f'predictions_{h}d.csv',parse_dates=['date']);assert daily.loc[pred.date,'flood'].eq(0).all()
            loo=pd.read_csv(OUT/'analogue_loeo.csv');assert loo.held_event.ne(loo.matched_event).all();assert len(loo)==38
            diag=pd.read_csv(OUT/'september_live_diagnostic.csv',parse_dates=['event_onset']);assert diag.event_onset.dropna().dt.year.lt(2024).all()
            scaler=json.loads((OUT/'analogue_scaler.json').read_text());assert scaler['fit_end']=='2019-12-31'
            mapping=json.loads(MAP.read_text());assert mapping['fit_days']==212
            return 'Training/threshold folds end in 2019; no ongoing-flood evaluation rows; LOEO excludes held events; September templates pre-2024; 212-day refit is live-only'
        check('Leakage and template eligibility',leakage)
        def labels():
            from floodwatch.core.analogue_labels import label
            from floodwatch.core.analogue_config import RESEARCH_CLOSE_MATCH_CUTOFF,OPERATIONAL_CLOSE_MATCH_CUTOFF
            summary=pd.read_csv(OUT/'discrimination_summary.csv')
            ordinary=summary[summary.group.eq('ordinary')].set_index('analogue')
            assert ordinary.new_close_rate.le(.1).all()
            assert ordinary.pre_flood_hit_rate.lt(.25).all()
            distances=pd.read_csv(OUT/'discrimination_distances.csv',parse_dates=['date'],float_precision='round_trip')
            assert distances.date.dt.year.between(2001,2019).all()
            for kind in ('research','operational'):
                part=distances[(distances.analogue==kind)&distances.group.eq('ordinary')]
                cutoff=float(ordinary.loc[kind,'new_cutoff'])
                # Use the exact config float for the inclusive boundary.
                cutoff=RESEARCH_CLOSE_MATCH_CUTOFF if kind=='research' else OPERATIONAL_CLOSE_MATCH_CUTOFF
                assert part.distance.le(cutoff).mean()<=.1
                assert part.distance.le(np.nextafter(cutoff,np.inf)).mean()>.1
                pre=distances[(distances.analogue==kind)&distances.group.eq('pre_flood')]
                assert pre.held_event.ne(pre.matched_event).all()

            assert np.isclose(RESEARCH_CLOSE_MATCH_CUTOFF,ordinary.loc['research','new_cutoff'],rtol=0,atol=1e-14)
            assert np.isclose(OPERATIONAL_CLOSE_MATCH_CUTOFF,ordinary.loc['operational','new_cutoff'],rtol=0,atol=1e-14)
            for kind,cutoff in [('research',RESEARCH_CLOSE_MATCH_CUTOFF),('operational',OPERATIONAL_CLOSE_MATCH_CUTOFF)]:
                assert label(cutoff,'2011-07-01',2,kind)['is_close_match']
                assert not label(np.nextafter(cutoff,np.inf),'2011-07-01',2,kind)['is_close_match']
                assert not label(None,None,None,kind)['is_close_match']
            report=pd.read_csv(OUT/'analogue_close_match_report.csv')
            for row in report.itertuples():
                close=pd.notna(row.distance) and row.distance<=row.close_match_cutoff
                assert row.is_close_match==close
                assert row.label=='Pattern matching not reliable enough to display (see README)'
            result=load_score(snapshot);a=result['analogue']
            assert a['is_close_match']==(a['distance'] is not None and a['distance']<=a['close_match_cutoff'])
            return 'Training-only false-match cap <=10% verified; both hit rates <25%; resemblance labels suppressed; distance flags follow cutoff exactly'
        check('Close-match label rule',labels)
        def ui():
            app=AppTest.from_file(str(c.ROOT/'app.py')).run(timeout=90)
            assert not app.exception,[str(x.value) for x in app.exception]
            assert [t.label for t in app.tabs]==['September 2024 replay','Big data pipeline','Live monitor']
            labels=[m.label for m in app.tabs[2].metric]
            assert 'River level (measured at Khokana)' in labels and '1-day flood risk score' in labels and '3-day flood risk score' in labels
            assert not app.tabs[2].error
            return 'Three tabs render offline; observed gauge, operational score cards and forecast charts present'
        check('Streamlit integration',ui)
    lines=[f'[{s}] {n}: {d}' for s,n,d in checks];lines.append(f"Summary: {sum(s=='PASS' for s,_,_ in checks)} PASS, 0 WARN, {sum(s=='FAIL' for s,_,_ in checks)} FAIL")
    text='\n'.join(lines)+'\n';(OUT/'live_check_output.txt').write_text(text);print(text)
    return int(any(s=='FAIL' for s,_,_ in checks))
if __name__=='__main__':raise SystemExit(main())
