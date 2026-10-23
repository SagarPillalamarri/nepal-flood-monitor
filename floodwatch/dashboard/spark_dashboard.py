"""Render saved Spark reports only. No Spark imports, jobs or network access."""
import json
from pathlib import Path
import pandas as pd
import streamlit as st

REPORT = Path(__file__).resolve().parents[2]/'outputs/spark'


def render_pipeline():
    st.title('Big data pipeline')
    from floodwatch.dashboard.plain_language import caption, glossary
    st.info('What this page shows: How 25 years of weather and river data were processed with Apache Spark, and proof that the results are correct. The charts below are saved processing results, not live river conditions.')
    st.caption('Saved results from Apache Spark 4.0.1 in local[*] mode. This tab reads reports only; it does not start Spark or score models.')
    st.code('API JSON → Parquet data lake → Spark jobs → Spark SQL → models → dashboard',language=None)
    st.caption('The models are frozen. Spark reproduces the accepted processing and analysis; the September replay continues to use the saved model results. Live observations and experimental operational scores are on the Live monitor tab.')
    required = ['lake_stats.json','variable_coverage.csv','column_parity.csv','rainfall_diagnosis.json',
                'events_per_year.csv','conditional_probabilities.csv','scaling_trials.json',
                'scaling.md','daily_physical_plan.txt','checks.csv']
    missing = [name for name in required if not (REPORT/name).is_file()]
    if missing:
        st.error('Saved Spark reports missing: '+', '.join(missing))
        return

    stats = json.loads((REPORT/'lake_stats.json').read_text())
    st.subheader('Data lake')
    metrics = st.columns(4)
    metrics[0].metric('Hourly rows',f"{stats['rows']:,}")
    metrics[1].metric('Partitions / Parquet files',f"{stats['partitions']} / {stats['parquet_files']}")
    metrics[2].metric('Snappy Parquet',f"{stats['parquet_bytes']/1e6:.2f} MB")
    metrics[3].metric('Source JSON',f"{stats['raw_json_bytes']/1e6:.2f} MB")
    for card in metrics:
        with card:caption('Dataset rows, storage chunks and file sizes.','Rows count hourly measurements; smaller file size needs less storage.','saved Spark lake_stats.json')
    saving = 100*(1-stats['parquet_bytes']/stats['raw_json_bytes'])
    st.caption(f'{saving:.1f}% smaller by file bytes; small filesystem metadata excluded. Partitioned by variable and year. Requested point coordinates are retained even when points snap to the same grid cell.')
    st.warning('Hourly rainfall and all three soil layers cover 2001–2010 only: the API rate limit stopped recovery at 2011. Moisture in the air (water vapour / TCWV) covers 2001–2025 with the documented 2024 source gap. Missing values are never filled.')
    coverage = pd.read_csv(REPORT/'variable_coverage.csv')
    st.dataframe(coverage[coverage.source.ne('gfs_probe')],hide_index=True,width='stretch')
    with st.expander('Small GFS probe caches, separate from the historical basin data'):
        st.dataframe(coverage[coverage.source.eq('gfs_probe')],hide_index=True,width='stretch')
        st.caption('Two one-point, one-day probes: the displayed start/end years are endpoints, not continuous coverage. Null unsupported layers are retained. These probes do not enter historical basin means or model scoring.')
    st.caption('Soil/TCWV daily means require 24 valid hours at each of all nine points. Shifted rainfall requires the same complete coverage; 2010-12-31 lacks the following midnight and stays null. Daily-only Stage 1 supplements are explicitly labelled in the feature lake.')

    st.subheader('Spark vs pandas parity')
    checks = pd.read_csv(REPORT/'checks.csv')
    counts = checks.status.value_counts()
    st.success(f"{counts.get('PASS',0)} PASS · {counts.get('WARN',0)} WARN · {counts.get('FAIL',0)} FAIL")
    parity = pd.read_csv(REPORT/'column_parity.csv')
    st.dataframe(parity,hide_index=True,width='stretch',column_config={
        'max_absolute_difference':st.column_config.NumberColumn('Max absolute difference',format='%.3e')})
    st.caption('Measured independent comparisons of complete hourly-derived daily values, rolling totals and seasonal percentiles. Units follow each column. Null masks are checked separately. Percentile references use 2001–2019 monsoon data only.')
    rainfall = json.loads((REPORT/'rainfall_diagnosis.json').read_text())
    table = pd.DataFrame([dict(
        grouping='00:00–23:00 (before)' if r['shift_hours']==0 else '01:00–24:00 (corrected)',
        complete_days=r['days'],median_difference_mm=r['median'],p95_signed_difference_mm=r['p95'],
        p95_absolute_difference_mm=r['p95_absolute'],max_absolute_difference_mm=r['max_absolute'],
        days_over_1mm=r['days_over_1mm'],mismatches_over_1e_6_mm=r['mismatches']) for r in rainfall])
    st.dataframe(table,hide_index=True,width='stretch')
    st.info('Rainfall timestamps mark the end of the preceding accumulation hour. Spark subtracts one hour for day grouping, so the following midnight belongs to the previous day. This resolves all 1,874 mismatches on complete days; accepted daily totals are unchanged. The >1 mm count rounds numerical noise to eight decimals (365 days; 366 without rounding).')
    with st.expander('Full saved check results'):
        st.dataframe(checks,hide_index=True,width='stretch')

    st.subheader('Spark SQL')
    events = pd.read_csv(REPORT/'events_per_year.csv')
    st.bar_chart(events.set_index('year'),y='n_events',x_label='Flood onset year',y_label='Events',height=250)
    caption('Number of flood onsets per year.','Taller bars mean more events in the saved river-flow labels.','saved Spark SQL event counts from GloFAS labels')
    st.dataframe(pd.read_csv(REPORT/'conditional_probabilities.csv'),hide_index=True,width='stretch',
                 column_config={'probability':st.column_config.NumberColumn('Probability',format='%.5f')})
    st.caption('Saved SQL results match the Stage 2 pandas outputs: all 26 event counts and all 20 conditional-probability rows. Conditional probabilities use 2001–2019 monsoon days only; all-day and not-currently-flooding populations are reported separately.')

    st.subheader('Scaling: 1× / 10× / 50×')
    trials = pd.DataFrame(json.loads((REPORT/'scaling_trials.json').read_text()))
    scaling = trials.groupby('scale',as_index=False).agg(input_rows=('input_rows','first'),median_seconds=('seconds','median'),trials=('seconds','count'))
    scaling['runtime_vs_1x'] = scaling.median_seconds/scaling.loc[scaling.scale.eq(1),'median_seconds'].iloc[0]
    st.dataframe(scaling,hide_index=True,width='stretch')
    st.line_chart(scaling.set_index('input_rows')[['median_seconds']],x_label='Input rows',y_label='Median runtime (seconds)',height=260)
    caption('Processing time as the number of records grows.','Higher means slower; rows are synthetic copies for a speed test.','saved Spark scaling_trials.json benchmark')
    st.caption('Three measured trials per scale after one warm-up; local[*], 10 CPU slots, 3 GB heap and 100 shuffle partitions at every scale. Synthetic replicas are generated inside Spark, with count/checksum verification. Timings cover hourly-to-daily aggregation, not the entire pipeline or model scoring.')
    st.info('Scale illustration: 50× is 256 million hourly rows—about one variable over all of Nepal across a multi-year archive, depending on grid resolution and duration. The benchmark uses synthetic replicas of this basin dataset; it is not an actual Nepal-wide download.')
    st.warning('The first 50× attempt ran out of Java heap with 10 shuffle partitions. All scales were rerun with 100 partitions and the same 3 GB heap; the table shows those successful runs. The initial failure is preserved in the benchmark report.')
    with st.expander('Full saved scaling report'):
        st.markdown((REPORT/'scaling.md').read_text())

    plan = (REPORT/'daily_physical_plan.txt').read_text()
    with st.expander('Physical plan — highlighted shuffle (Exchange) lines'):
        st.caption('Yellow markers identify Exchange operators and hash-partitioning lines: these move records between tasks for grouping and distinct-hour counts. The remaining lines preserve the saved plan text.')
        highlighted = '\n'.join(('🟨 ' if 'Exchange' in line or 'hashpartitioning' in line else '   ')+line for line in plan.splitlines())
        st.code(highlighted,language=None)
        st.download_button('Download original physical plan',data=plan,file_name='daily_physical_plan.txt',mime='text/plain')

    glossary()
