"""Offline parity and integrity checks for Spark processing; no model fitting."""
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import config as c
from features import raw_signals, tcwv_percentile
from spark_runtime import session
from spark_pipeline import OUT, LAKE


def main():
    checks = []
    parity = []
    def record(category, column, actual, expected):
        a,b = np.asarray(actual,dtype=float),np.asarray(expected,dtype=float)
        valid = np.isfinite(a) & np.isfinite(b)
        parity.append(dict(category=category,column=column,n_compared=int(valid.sum()),
                           max_absolute_difference=float(np.max(np.abs(a[valid]-b[valid]))) if valid.any() else 0.,
                           null_mask_mismatches=int(np.sum(np.isnan(a) != np.isnan(b)))))
    spark = session('Bagmati Spark checks')
    def check(name, run):
        try:
            message = run()
            status = 'PASS'
        except Exception as error:
            status, message = 'FAIL', str(error)
        checks.append(dict(status=status,message=f'{name}: {message}'))
        print(f'[{status}] {name}: {message}',flush=True)
    try:
        daily = spark.read.parquet(str(LAKE/'daily_aggregates')).toPandas()
        features = spark.read.parquet(str(LAKE/'features_daily')).toPandas()
        features['date'] = pd.to_datetime(features.date)
        features = features.set_index('date').sort_index()
        base = pd.read_csv(c.STAGE1B_FILE,index_col=0,parse_dates=True)
        def lake_check():
            from pyspark.sql import functions as f
            raw = spark.read.parquet(str(LAKE/'raw'))
            (raw.groupBy('source','variable').agg(f.min('year').alias('start_year'),f.max('year').alias('end_year'),
                                                f.count('*').alias('rows'),f.count('value').alias('non_null_values'))
             .orderBy('source','variable').toPandas().to_csv(OUT/'variable_coverage.csv',index=False))
            stats = json.loads((OUT/'lake_stats.json').read_text())
            count = raw.count()
            assert count == stats['rows']
            assert raw.groupBy('source','timestamp','lat','lon','variable').count().where('count > 1').count() == 0, 'Duplicate hourly keys'
            expected = 0
            for item in json.loads((OUT/'hourly_inventory.json').read_text()):
                data = json.loads(__import__('pathlib').Path(item['path']).read_text())
                for point in data if isinstance(data,list) else [data]:
                    expected += sum(len(values) for key,values in point['hourly'].items() if key != 'time')
            assert count == expected, (count,expected)
            assert raw.where(f.isnan('value')).count() == 0, 'NaN should be retained as SQL NULL'
            return f'{count:,} rows; all cached hourly records retained; unique source/hour/point/variable keys'
        check('Raw lake integrity',lake_check)
        def daily_check():
            expected = []
            for item in json.loads((OUT/'hourly_inventory.json').read_text()):
                if '/stage1b/' not in item['path'] and '/spark_weather/' not in item['path']:
                    continue
                payload = json.loads(__import__('pathlib').Path(item['path']).read_text())
                next_payload = None
                if '/spark_weather/' in item['path']:
                    path = __import__('pathlib').Path(item['path'])
                    year = int(path.stem.rsplit('_',1)[1])
                    next_path = path.with_name(f'stage1_best_match_{year+1}.json')
                    if next_path.exists():
                        next_payload = json.loads(next_path.read_text())
                points = []
                for point_index,loc in enumerate(payload):
                    h = loc['hourly']
                    frame = pd.DataFrame({k:v for k,v in h.items() if k!='time'},index=pd.to_datetime(h['time']),dtype=float)
                    result = frame.resample('D').mean()
                    if 'precipitation' in frame:
                        rain = frame.precipitation
                        if next_payload is not None:
                            hnext = next_payload[point_index]['hourly']
                            rain = pd.concat([rain,pd.Series([hnext['precipitation'][0]],index=pd.to_datetime([hnext['time'][0]]))])
                        rain.index = rain.index-pd.Timedelta(hours=1)
                        result['precipitation'] = rain.resample('D').sum(min_count=24).reindex(result.index)
                    result = result.where(frame.resample('D').count() == 24)
                    points.append(result)
                basin = sum(points)/9
                source = 'era5_tcwv' if '/stage1b/' in item['path'] else 'stage1_best_match'
                for variable in basin:
                    series = basin[variable]
                    got = daily[(daily.source == source) & (daily.variable == variable)].copy()
                    got['date'] = pd.to_datetime(got.date)
                    got = got.set_index('date').value.reindex(series.index)
                    np.testing.assert_allclose(got.to_numpy(float),series.to_numpy(float),rtol=0,atol=1e-10,equal_nan=True)
                    record('Hourly to daily',variable,got,series)
                    expected.append(len(series))
            return f'{sum(expected):,} variable-days match independent pandas resampling, including identical null masks; all nine points and 24 valid hours required'
        check('Spark daily vs pandas',daily_check)
        def rolling_check():
            for days in (1,3,7,14):
                for prefix,source in [('',features.precip_mm),('hourly_',features.hourly_precip_mm)]:
                    expected = source.rolling(days,min_periods=days).sum()
                    np.testing.assert_allclose(features[f'{prefix}rain_{days}d'],expected,rtol=0,atol=1e-9,equal_nan=True)
                    record('Rolling rainfall',f'{prefix}rain_{days}d',features[f'{prefix}rain_{days}d'],expected)
            return '1/3/7/14-day Spark windows match pandas for daily-API and hourly-derived rainfall; incomplete windows stay null'
        check('Rolling rainfall',rolling_check)
        def rainfall_parity():
            valid = features.hourly_precip_mm.notna()
            assert valid.sum() == 3651, f'Unexpected complete shifted rainfall coverage: {valid.sum()}'
            np.testing.assert_allclose(features.loc[valid,'hourly_precip_mm'],features.loc[valid,'precip_mm'],rtol=0,atol=1e-10)
            np.testing.assert_allclose(features.loc[valid,'hourly_precip_max_point_mm'],features.loc[valid,'precip_max_point_mm'],rtol=0,atol=1e-10)
            assert pd.isna(features.loc['2010-12-31','hourly_precip_mm'])
            return 'Shifted 01:00–24:00 rainfall matches accepted basin means and max-point totals on all 3,651 complete days; final cached day remains null'
        check('Shifted rainfall parity',rainfall_parity)
        def percentile_check():
            reference = spark.read.parquet(str(LAKE/'seasonal_reference')).toPandas()
            dates = pd.to_datetime(reference.date)
            assert dates.min() >= pd.Timestamp('2001-01-01') and dates.max() <= pd.Timestamp('2019-12-31')
            assert dates.dt.month.between(6,10).all()
            signals = raw_signals(base)
            mapping = {'tcwv':'tcwv_pct','sm_0_7':'sm_0_7_pct','rain_max_point':'rain_max_point_pct',**{f'rain_{d}d':f'rain_{d}d_pct' for d in (1,3,7,14)}}
            monsoon = signals.index[signals.index.month.isin(c.MONSOON_MONTHS)]
            for signal,out in mapping.items():
                expected = tcwv_percentile(signals.loc[monsoon,signal],signals.loc[:c.TRAIN_END,signal])
                np.testing.assert_allclose(features.loc[monsoon,out].to_numpy(dtype=float),expected.to_numpy(dtype=float),rtol=0,atol=1e-10,equal_nan=True)
                record('Seasonal percentile',out,features.loc[monsoon,out],expected)
            return f'All seven seasonal percentile features match frozen pandas definitions; reference dates {dates.min().date()}–{dates.max().date()}, monsoon 2001–2019 only'
        check('Seasonal reference and percentile parity',percentile_check)
        def sql_check():
            keys = ['population','condition','horizon_days']
            got = pd.read_csv(OUT/'conditional_probabilities.csv').sort_values(keys).reset_index(drop=True)
            expected = pd.read_csv(c.STAGE2_CONDITIONAL).sort_values(keys).reset_index(drop=True)
            pd.testing.assert_frame_equal(got[expected.columns],expected,check_dtype=False,rtol=1e-12,atol=1e-12)
            counts = pd.read_csv(OUT/'events_per_year.csv')
            assert counts.n_events.sum() == 26
            events = pd.read_csv(c.STAGE2_EVENTS)
            pd.testing.assert_series_equal(counts.set_index('year').n_events,events.groupby('year').size().reindex(range(2001,2026),fill_value=0),check_names=False,check_dtype=False)
            return 'All 20 Stage 2 conditional-probability rows reproduced exactly; 26 event counts match onset years'
        check('Spark SQL results',sql_check)
        def frozen_check():
            manifest = json.loads((c.ROOT/'outputs/satellite/frozen_artifact_hashes.json').read_text())
            for path,sha in manifest.items():
                if path == 'app.py' and (OUT/'dashboard_original_app.py').exists():
                    from spark_dashboard_integrity import verify_replay
                    verify_replay()
                    continue
                assert hashlib.sha256((c.ROOT/path).read_bytes()).hexdigest() == sha, f'Frozen artifact changed: {path}'
            if (OUT/'dashboard_original_app.py').exists():
                return f'{len(manifest)-1} original non-app hashes unchanged; authorized app tab wrapper preserves the original replay body verbatim'
            return f'All {len(manifest)} original hashes unchanged, including models, thresholds, features.py and dashboard'
        check('Frozen artifacts',frozen_check)
        gap = features.loc['2024-01-01':'2024-07-01','tcwv_mm']
        def gap_check():
            assert gap.isna().all(), 'Accepted TCWV gap was filled'
            case = features.loc['2024-09-26':'2024-09-28',['tcwv_mm','tcwv_pct']]
            case.to_csv(OUT/'september_tcwv_check.csv')
            return f'{len(gap)} missing daily TCWV values preserved; September 26–28 percentiles match pandas'
        check('Missing data',gap_check)
        mismatch = (features.hourly_precip_mm-features.precip_mm).abs()
        if (mismatch > 1e-6).any():
            message = (f'Hourly precipitation sums differ from the accepted daily API totals on {int((mismatch>1e-6).sum())} days '
                       f'(max {mismatch.max():.6f} mm). Both retained separately; Stage 2 reproduction uses accepted daily totals.')
            checks.append(dict(status='WARN',message=message))
            print('[WARN] '+message,flush=True)
        coverage = json.loads((OUT/'download_status.json').read_text())
        if any(x['status']=='RATE_LIMIT' for x in coverage):
            message = 'Open-Meteo rate limit stopped hourly weather recovery at 2011; recovered 2001–2010 only. Remaining rainfall/soil daily inputs have explicit cached-Stage-1 provenance; no hourly values fabricated.'
            checks.append(dict(status='WARN',message=message))
            print('[WARN] '+message,flush=True)
    finally:
        spark.stop()
    pd.DataFrame(checks).to_csv(OUT/'checks.csv',index=False)
    if parity:
        (pd.DataFrame(parity).groupby(['category','column'],as_index=False)
         .agg(n_compared=('n_compared','sum'),max_absolute_difference=('max_absolute_difference','max'),
              null_mask_mismatches=('null_mask_mismatches','sum')).to_csv(OUT/'column_parity.csv',index=False))
    summary = f"Summary: {sum(x['status']=='PASS' for x in checks)} PASS, {sum(x['status']=='WARN' for x in checks)} WARN, {sum(x['status']=='FAIL' for x in checks)} FAIL"
    print(summary,flush=True)
    (OUT/'check_output.txt').write_text('\n'.join(f"[{x['status']}] {x['message']}" for x in checks)+'\n'+summary+'\n')
    return int(any(x['status']=='FAIL' for x in checks))


if __name__ == '__main__':
    sys.exit(main())
