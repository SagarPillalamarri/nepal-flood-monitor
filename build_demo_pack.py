"""Build a presentation pack from frozen artifacts without fitting or retuning."""
import json
import os
import shutil
import numpy as np
import pandas as pd
import config as c
from stage_sat_fetch import OUT as SATELLITE, verify_frozen

DEMO = c.ROOT/'outputs/demo'


def markdown(headers, rows):
    def cell(value):
        if value is None or (isinstance(value,(float,np.floating)) and np.isnan(value)):
            return '—'
        return str(value).replace('|','/')
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']
                     + ['| '+' | '.join(cell(value) for value in row)+' |' for row in rows])


def main():
    count = verify_frozen()
    satellite_ready = (SATELLITE/'summary.json').exists() and (SATELLITE/'sep2024_imerg_vs_era5.png').exists()
    if satellite_ready:
        satellite_checks = pd.read_csv(SATELLITE/'checks.csv')
        if satellite_checks.status.eq('FAIL').any():
            raise ValueError('Satellite validation has a FAIL; resolve it before packaging')
    elif not (SATELLITE/'fallback.json').exists():
        raise ValueError('Finish satellite validation or explicitly record the approved fallback before building the demo pack')
    DEMO.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(DEMO/'.matplotlib'))
    os.environ.setdefault('XDG_CACHE_HOME',str(DEMO/'.cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from xgboost import XGBClassifier
    events = pd.read_csv(c.STAGE2_EVENTS,parse_dates=['start','peak_date'])
    selected = json.loads(c.STAGE3_CALENDAR_SELECTED.read_text())['horizons']
    replay = pd.read_csv(c.STAGE5_REPLAY_FILE,parse_dates=['date','analogue_event_onset']).set_index('date')
    probability = pd.read_csv(c.STAGE2_CONDITIONAL)
    precursor = pd.read_csv(c.STAGE2_PRECURSORS).set_index('variable').loc['tcwv_mm']
    loo = pd.read_csv(c.STAGE4_DIR/'loo_summary.csv')
    shutil.copy2(c.STAGE2_DIR/'composite_tcwv_mm.png',DEMO/'tcwv_event_composite.png')
    shutil.copy2(c.STAGE2_CONDITIONAL,DEMO/'conditional_probabilities.csv')
    shutil.copy2(c.STAGE5_CASE_TABLE,DEMO/'september_20_28_diagnostics.csv')
    shutil.copy2(c.STAGE3_CALENDAR_SELECTED,DEMO/'selected_models.json')

    conditions = ['all_days','tcwv_pct_ge_90','tcwv_pct_ge_95',
                  'tcwv_pct_ge_95_and_soil_gt_75','rain_3d_pct_ge_95']
    names = ['All monsoon days','TCWV ≥90th','TCWV ≥95th','TCWV ≥95th + soil >75th','3-day rain ≥95th']
    prob = probability[probability.population.eq('all_monsoon_days')]
    fig,ax = plt.subplots(figsize=(11,5),layout='constrained')
    for offset,horizon,color in [(-.2,1,'#176b89'),(.2,3,'#d17a22')]:
        part = prob[prob.horizon_days.eq(horizon)].set_index('condition').loc[conditions]
        bars = ax.bar(np.arange(len(conditions))+offset,100*part.probability,width=.4,color=color,label=f'Onset within next {horizon} day(s)')
        ax.bar_label(bars,labels=[f'{n:.1f}%' for n in 100*part.probability],fontsize=8,padding=2)
    ns = prob[prob.horizon_days.eq(1)].set_index('condition').loc[conditions].n_days
    ax.set_xticks(np.arange(len(conditions)),[f'{name}\nn={n}' for name,n in zip(names,ns)],fontsize=8)
    ax.set(ylabel='Empirical probability (%)',title='Training monsoons 2001–2019 · conditional flood-onset frequencies')
    ax.legend(fontsize=8)
    fig.savefig(DEMO/'conditional_probabilities.png',dpi=160)
    plt.close(fig)

    all_metrics,validation_rows = [],[]
    for horizon in c.TARGET_HORIZONS:
        choice = next(row for row in selected if row['horizon']==horizon)
        selection = json.loads((c.STAGE3_CALENDAR_DIR/f'selection_{horizon}d.json').read_text())
        for variant,settings in selection['models'].items():
            validation_rows.append([horizon,variant.replace('_',' '),
                                    ', '.join(f'{value:.4f}' for value in settings['validation_fold_f1']),
                                    f'{settings["mean_validation_f1"]:.4f}',f'{settings["threshold"]:.2f}',
                                    'SELECTED' if variant==choice['selected_variant'] else 'Not selected'])
        metric = pd.read_csv(c.STAGE3_CALENDAR_DIR/f'metrics_{horizon}d.csv')
        all_metrics.append(metric)
        model = XGBClassifier()
        model.load_model(c.STAGE3_CALENDAR_DIR/f'{choice["selected_variant"]}_{horizon}d.json')
        columns = choice['columns']
        importance = pd.Series(model.feature_importances_,index=columns).sort_values()
        fig,ax = plt.subplots(figsize=(10,6),layout='constrained')
        importance.plot.barh(ax=ax,color='#176b89')
        ax.set(title=f'Selected {horizon}-day model · {choice["selected_variant"].replace("_"," ")}',xlabel='Normalized gain importance')
        fig.savefig(DEMO/f'xgboost_selected_{horizon}d_feature_importance.png',dpi=160)
        plt.close(fig)
    metrics = pd.concat(all_metrics,ignore_index=True)
    metrics.to_csv(DEMO/'model_metrics_both_variants.csv',index=False)

    case = replay.loc['2024-09-20':'2024-09-30']
    fig,axes = plt.subplots(3,1,figsize=(11,9),sharex=True,layout='constrained')
    for axis,horizon in zip(axes[:2],c.TARGET_HORIZONS):
        axis.plot(case.index,case[f'risk_{horizon}d'],marker='o',color='#176b89',label=f'Selected {horizon}-day XGBoost score')
        threshold = float(case[f'threshold_{horizon}d'].iloc[0])
        axis.axhline(threshold,color='tomato',linestyle='--',label=f'Frozen threshold {threshold:.2f}')
        axis.set(ylabel='Uncalibrated score',ylim=(0,1))
        axis.legend(fontsize=8,loc='upper left')
    axes[2].plot(case.index,case.discharge_m3_s,marker='o',color='#36558f',label='GloFAS q_north')
    threshold = json.loads(c.STAGE2_METADATA.read_text())['threshold']
    axes[2].axhline(threshold,color='tomato',linestyle='--',label=f'Flood-label threshold {threshold:.3f} m³/s')
    axes[2].set(ylabel='Discharge (m³/s)')
    axes[2].legend(fontsize=8,loc='upper left')
    for axis in axes:
        axis.axvline(pd.Timestamp('2024-09-28'),color='black',alpha=.4,linestyle=':')
    axes[0].set_title('September 2024 · frozen selected scores and discharge\nBlank risk points are excluded evaluation dates, not zero scores')
    fig.autofmt_xdate()
    fig.savefig(DEMO/'sep2024_selected_risk_vs_discharge.png',dpi=160)
    plt.close(fig)

    fig,ax = plt.subplots(figsize=(11,5),layout='constrained')
    ax.plot(case.index,case.analogue_days_before_flood,color='gray',alpha=.6)
    for onset,group in case.groupby('analogue_event_onset'):
        ax.scatter(group.index,group.analogue_days_before_flood,s=50,label=f'Past event {onset:%Y-%m-%d}')
    ax.axhline(3,color='gray',linestyle='--',alpha=.6)
    ax.axvline(pd.Timestamp('2024-09-28'),color='black',linestyle=':',label='2024 onset')
    ax.set(ylabel='Days before the matched past flood',ylim=(0,15),title='September 2024 · closest physical analogue alignment\nMatched lead is descriptive, not a calibrated current-event countdown')
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.savefig(DEMO/'sep2024_analogue_countdown.png',dpi=160)
    plt.close(fig)

    sections = ['# Bagmati flood precursor project — demo results',
                'Models are FROZEN. This pack reads saved models, predictions and selection records. It does not fit, retune, change feature sets, thresholds or selected variants.',
                '## Event catalogue',
                '26 GloFAS-proxy events: 19 training events (2001–2019), six main-test events, and the separately reported September 2024 event. Flood means q_north exceeds the training-monsoon 99th percentile, 445.191 m³/s. Onset is the first flood day; peak is the maximum discharge date.',
                markdown(['Event','Onset','Peak'],[[r.event_id,r.start.strftime('%Y-%m-%d'),r.peak_date.strftime('%Y-%m-%d')] for r in events.itertuples()]),
                '## Stage 2: TCWV precursor finding',
                f'The all-event TCWV composite first exceeds the same-calendar-day training 75th percentile at lag {int(precursor.first_lag)} ({int(precursor.days_before_event)} days before onset), with {int(precursor.paired_events_at_crossing)} paired events. This includes test/case events and is exploratory, not a validated two-day forecast lead. Composites skip missing observations and report sample counts.',
                '![TCWV event composite](tcwv_event_composite.png)',
                '![Training conditional probabilities](conditional_probabilities.png)',
                'Conditional probabilities use 2001–2019 only. Denominators are condition-day counts, not independent events; both all-days and non-ongoing-flood populations are retained in conditional_probabilities.csv.',
                '## Stage 3: validation-only calendar ablation',
                'The ablation was prompted by the September 2024 diagnosis. Thresholds were tuned on pooled out-of-fold predictions from the existing three chronological validation folds inside 2001–2019. Selection uses mean fold F1; an exact tie drops monsoon_day. Neither test metrics nor September case results selected the variant. Existing targets, split purges, native NaN handling and exclusion of ongoing flood days remain in force.',
                markdown(['Horizon (days)','Feature set','Fold F1 values','Mean validation F1','Threshold','Selection'],validation_rows)]
    test_rows,case_rows = [],[]
    for row in metrics.itertuples():
        values = [row.horizon,row.variant.replace('_',' '),'SELECTED' if row.selected_by_validation else 'Not selected',
                  f'{100*row.recall:.1f}%',f'{100*row.precision:.1f}%',f'{100*row.false_alarm_rate:.2f}%',
                  f'{row.f1:.3f}',f'{row.events_detected}/{row.events_evaluable}',
                  '—' if pd.isna(row.mean_lead_days) else f'{row.mean_lead_days:.2f}']
        (test_rows if row.split=='test' else case_rows).append(values)
    headers = ['Horizon','Variant','Status','Recall','Precision','FAR','F1','Events detected','Mean lead (days)']
    sections += ['### Main test: 2020–2025, September 2024 reported separately',markdown(headers,test_rows),
                 'Daily false alarm rate = FP/(FP+TN). Mean lead averages detected events only. There are 880 eligible one-day test dates and 878 three-day test dates; each has six evaluable events.',
                 '### Separate September 2024 results',markdown(headers,case_rows),
                 'Both selected models miss September 2024. The non-selected three-day model without monsoon_day detects it one day ahead with two false alarms in 27 eligible case dates; this did not change the selected model.',
                 '![Selected one-day importance](xgboost_selected_1d_feature_importance.png)',
                 '![Selected three-day importance](xgboost_selected_3d_feature_importance.png)',
                 '## September 20–28 replay: frozen selected scores and closest analogue']
    replay_rows=[]
    for date,row in replay.loc['2024-09-20':'2024-09-28'].iterrows():
        replay_rows.append([date.strftime('%Y-%m-%d'), 'Excluded: ongoing flood' if pd.isna(row.risk_1d) else f'{row.risk_1d:.6f}',
                            f'{row.threshold_1d:.2f}', 'Excluded: ongoing flood' if pd.isna(row.risk_3d) else f'{row.risk_3d:.6f}',
                            f'{row.threshold_3d:.2f}',row.analogue_event_onset.strftime('%Y-%m-%d'),int(row.analogue_days_before_flood)])
    sections += [markdown(['Date','1-day score','Threshold','3-day score','Threshold','Past event onset','Matched lead (days)'],replay_rows),
                 '![Selected risk versus discharge](sep2024_selected_risk_vs_discharge.png)',
                 '## Stage 4: physical analogue selection',
                 'Inputs: TCWV percentile; 1/3/7-day rainfall percentiles; topsoil moisture; one-day discharge rise. No calendar feature. Euclidean matching only; no DTW. Leave-one-event-out uses 19 training events, queried at onset-1 and onset-2. A hit means the top analogue ends within three days of its own event onset.',
                 markdown(['Window (days)','Hits / queries','Hit rate'],[[int(r.window_days),f'{int(r.sum)}/{int(r.count)}',f'{100*r.mean:.1f}%'] for r in loo.itertuples()]),
                 'The 14-day configuration is selected. September uses pre-2024 event templates only. The 38 queries are paired observations from 19 events, not 38 independent floods.',
                 '![Historical analogue alignment](sep2024_analogue_countdown.png)']
    satellite_ok = (SATELLITE/'summary.json').exists() and (SATELLITE/'sep2024_imerg_vs_era5.png').exists()
    if satellite_ok:
        sat = json.loads((SATELLITE/'summary.json').read_text())
        shutil.copy2(SATELLITE/'sep2024_imerg_vs_era5.png',DEMO/'sep2024_imerg_vs_era5.png')
        shutil.copy2(SATELLITE/'event_rainfall_percentiles.csv',DEMO/'satellite_event_rainfall_percentiles.csv')
        sections += ['## Satellite rainfall validation',
                     f'NASA/GPM_L3/IMERG_V07 precipitation from read-only Earth Engine project bdasem-7; {sat["days"]} monsoon days, 2001–2025. Each Kathmandu day integrates the 49 overlapping half-hours with 15-minute weights at both day boundaries. Spatial support is the area-weighted nine-point bounding box at the native 0.1-degree grid.',
                     f'Monsoon mean: IMERG {sat["imerg_mean_mm_day"]:.3f} mm/day; ERA5 {sat["era5_mean_mm_day"]:.3f} mm/day. Raw daily Pearson correlation {sat["raw_correlation"]:.4f}; seasonal-percentile correlation {sat["percentile_correlation"]:.4f}. Each source has its own 2001–2019 reference. {sat["provisional_days"]} dates contain non-permanent source intervals.',
                     'IMERG is validation only and is absent from model features and dashboard scoring. The 26-event before-onset rainfall percentile table is in satellite_event_rainfall_percentiles.csv.',
                     '![IMERG versus ERA5 rainfall](sep2024_imerg_vs_era5.png)']
    else:
        sections += ['## Satellite validation status','Earth Engine setup/extraction was not completed within the time box. The approved reanalysis/hydrological-data title and satellite/radar next steps apply. No substitute source or satellite validation result is claimed.']
    sections += ['## Known data limitations',
        '- ERA5 rainfall has a previously noted local wet bias. The IMERG bounding-box mean and ERA5 nine-point mean have different spatial support, so their difference is not a controlled bias estimate. Neither is local rain-gauge truth.',
        '- TCWV source gap: all nine points have 4,368 null hours from 2024-01-01 05:00 to 2024-07-01 04:00 in returned API local timestamps. Complete-day aggregation leaves 183 NaN days, including 31 monsoon dates (June plus July 1). There is no filling, interpolation or model substitution; XGBoost uses native NaN handling.',
        '- GloFAS discharge labels are modelled proxies rather than observed DHM danger-level exceedances. GloFAS reanalysis and the weather inputs share ERA5 forcing; agreement between rainfall and these labels is not fully independent validation. See the [Copernicus GloFAS FAQ](https://confluence.ecmwf.int/spaces/CEMS/pages/315563119/6+FAQ+in+GloFAS).',
        '- The main test has only six independent flood events; the September case is separate. Rare-event precision is low, especially for the three-day target. Daily positive-date counts differ from event counts.',
        '- Approximate 1.5-day basin response is a user-supplied working estimate, not calibrated here. Daily means can obscure fast flood build-up, and end-of-day predictors do not establish a guaranteed operational warning lead.',
        '- ERA5 is delayed reanalysis. This is a historical replay, with no live forecast equivalence, pollers or operational latency validation. Risk scores are class-weighted and uncalibrated, not flood probabilities.',
        '- The TCWV lag-2 finding uses all-event exploratory composites. The calendar ablation was prompted after inspecting September; training-fold selection prevents direct test tuning but does not create a new blind confirmation set.',
        '- Seasonal percentile references use only 2001–2019 monsoon observations. Three-day satellite comparisons omit the first two June reference days for both sources because satellite retrieval is monsoon-only.',
        '- IMERG is a coarse (~0.1-degree) satellite-derived product that can include gauge calibration, retrieval errors and provisional records. Small mountainous basins have few pixels; satellite rainfall is not ground truth. The requested nine-point bounding box is a study-area rectangle, not a delineated catchment polygon. See the [official IMERG V07 catalogue](https://developers.google.com/earth-engine/datasets/catalog/NASA_GPM_L3_IMERG_V07).',
        '- ONI uses a publication-lag approximation; historical values can be revised. ONI is excluded from the selected primary models and remains only in the earlier optional ablation.',
        '- Closest analogue lead is descriptive and can jump when the nearest past event changes; LOEO hit rate is 12/38, not evidence of a reliable countdown to the current flood.',
        '## Dashboard launch',
        'Run from the project folder; the existing environment is ready. Do not run model training or calibration commands for the demo.',
        "```sh\ncd '/Users/sagarp/Documents/BDA_SEM 7'\nvenv/bin/streamlit run app.py\n```",
        'The date slider replays September 1–30. Rainfall, TCWV percentile, soil moisture, discharge, selected frozen scores/thresholds and closest past event are displayed. Current-flood and target-censored dates remain unscored.',
        f'Frozen-artifact verification: {count} recorded files matched before and after pack creation.']
    (DEMO/'results_summary.md').write_text('\n\n'.join(sections)+'\n')
    (DEMO/'launch_commands.txt').write_text("cd '/Users/sagarp/Documents/BDA_SEM 7'\nvenv/bin/streamlit run app.py\n")
    verify_frozen()
    (DEMO/'manifest.json').write_text(json.dumps({'frozen_artifacts_verified':count,'satellite_validation_included':satellite_ok,
                                               'plots':sorted(p.name for p in DEMO.glob('*.png'))},indent=2))
    print((DEMO/'results_summary.md').read_text())


if __name__ == '__main__':
    main()
