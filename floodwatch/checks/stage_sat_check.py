"""Compare IMERG with ERA5 for validation only; model features remain frozen."""
import os
import json
import sys
import numpy as np
import pandas as pd
from floodwatch.core import config as c
from floodwatch.core.features import tcwv_percentile
from floodwatch.pipeline.stage_sat_fetch import DAILY_FILE, OUT, verify_frozen


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lines, checks = [], []
    def emit(text):
        lines.append(str(text))
        print(text)
    def report(status, message):
        checks.append({'status':status,'message':message})
        emit(f'[{status}] {message}')
    try:
        satellite = pd.read_csv(DAILY_FILE,parse_dates=['date']).set_index('date')
        source = pd.read_csv(c.STAGE1B_FILE,parse_dates=['date']).set_index('date')
        events = pd.read_csv(c.STAGE2_EVENTS,parse_dates=['start','peak_date','end'])
        expected = source.index[source.index.month.isin(c.MONSOON_MONTHS)]
        if not satellite.index.equals(expected) or satellite.index.has_duplicates:
            raise ValueError('IMERG daily dates differ from the 2001–2025 monsoon calendar')
        if not np.isfinite(satellite.imerg_mm).all() or satellite.imerg_mm.lt(0).any() or satellite.n_intervals.ne(49).any():
            raise ValueError('Missing/invalid rainfall or incomplete half-hourly day')
        report('PASS',f'{len(satellite)} complete monsoon days, each integrated over 24 Kathmandu hours from 49 overlapping intervals')
        start = pd.Timestamp('2024-09-27',tz=c.TIMEZONE).tz_convert('UTC')
        frames = pd.date_range(start-pd.Timedelta(minutes=15),periods=49,freq='30min')
        weights = [max(0,(min(stamp+pd.Timedelta(minutes=30),start+pd.Timedelta(days=1))-max(stamp,start)).total_seconds()/3600) for stamp in frames]
        np.testing.assert_allclose([weights[0],weights[-1],sum(weights)],[.25,.25,24])
        np.testing.assert_allclose(weights[1:-1],.5)
        report('PASS','Kathmandu boundary weights independently verified: 0.25 + 47×0.5 + 0.25 = 24 hours; no UTC-day substitution')

        paired = pd.DataFrame({'imerg_mm':satellite.imerg_mm,'era5_mm':source.precip_mm.reindex(expected)})
        reference = paired.loc[c.TRAIN_START:c.TRAIN_END]
        for name in ('imerg','era5'):
            paired[f'{name}_pct'] = tcwv_percentile(paired[f'{name}_mm'],reference[f'{name}_mm'])
        report('PASS','Each source uses its own 2001–2019 monsoon seasonal percentile reference (±15 calendar days); test years excluded')
        means = paired[['imerg_mm','era5_mm']].mean()
        raw_corr = float(paired.imerg_mm.corr(paired.era5_mm))
        pct_corr = float(paired.imerg_pct.corr(paired.era5_pct))
        daily_case = paired.loc['2024-09-20':'2024-09-30',['imerg_mm','era5_mm']]
        paired.to_csv(OUT/'daily_comparison.csv')
        daily_case.to_csv(OUT/'september_2024_daily.csv')
        emit('\nDAILY RAINFALL, 20–30 SEPTEMBER 2024 (mm/day):')
        emit(daily_case.round(3).to_string())
        emit(f'\nMONSOON MEAN, 2001–2025: IMERG={means.imerg_mm:.3f} mm/day; ERA5={means.era5_mm:.3f} mm/day; days={len(paired)}')
        emit(f'DAILY MONSOON CORRELATION: raw Pearson r={raw_corr:.4f}; seasonal-percentile Pearson r={pct_corr:.4f}; days={len(paired)}')

        # Reindex onto the full calendar before rolling; no jumping across winters.
        calendar = source.index
        rolling = paired[['imerg_mm','era5_mm']].reindex(calendar).rolling(3,min_periods=3).sum()
        common = rolling.dropna().loc[lambda x:x.index.month.isin(c.MONSOON_MONTHS)]
        ref3 = common.loc[c.TRAIN_START:c.TRAIN_END]
        records = []
        for event in events.itertuples():
            end_date = event.start-pd.Timedelta(days=1)
            values = common.loc[[end_date]]
            record = {'event_id':event.event_id,'onset':event.start,'peak':event.peak_date,
                      'rain_start':event.start-pd.Timedelta(days=3),'rain_end':end_date}
            for name in ('imerg','era5'):
                record[f'{name}_3day_mm'] = float(values.iloc[0][f'{name}_mm'])
                record[f'{name}_3day_pct'] = float(tcwv_percentile(values[f'{name}_mm'],ref3[f'{name}_mm']).iloc[0])
            records.append(record)
        event_rain = pd.DataFrame(records)
        if len(event_rain) != 26 or event_rain.filter(regex='3day').isna().any().any():
            raise ValueError('Incomplete 26-event rainfall comparison')
        event_rain.to_csv(OUT/'event_rainfall_percentiles.csv',index=False)
        emit('\nTHREE-DAY RAINFALL STRICTLY BEFORE EACH ONSET (source-specific training seasonal percentile):')
        display = event_rain[['event_id','onset','peak','imerg_3day_mm','era5_3day_mm','imerg_3day_pct','era5_3day_pct']].copy()
        display[display.select_dtypes('number').columns] = display.select_dtypes('number').round(2)
        emit(display.to_string(index=False))
        report('PASS','All 26 event totals cover onset-3 through onset-1; full-calendar rolling and matched source-specific training references used')
        provisional_days = int(satellite.provisional_intervals.gt(0).sum())
        if provisional_days:
            report('WARN',f'{provisional_days} days include non-permanent IMERG intervals; source status is retained in imerg_daily.csv')
        report('WARN','IMERG is a native 0.1-degree area-weighted bounding-box mean; ERA5 is the existing nine-point mean. IMERG is a separate satellite-derived estimate, not ground truth')
        os.environ.setdefault('MPLCONFIGDIR',str(OUT/'.matplotlib'))
        os.environ.setdefault('XDG_CACHE_HOME',str(OUT/'.cache'))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11,5),layout='constrained')
        positions = np.arange(len(daily_case))
        ax.bar(positions-.2,daily_case.imerg_mm,width=.4,label='GPM IMERG V07')
        ax.bar(positions+.2,daily_case.era5_mm,width=.4,label='ERA5 nine-point mean')
        ax.set_xticks(positions,daily_case.index.strftime('%d Sep'),rotation=30,ha='right')
        ax.set(ylabel='Rainfall (mm / Kathmandu day)',title='September 2024 rainfall validation · Bagmati bounding box')
        ax.axvline(list(daily_case.index).index(pd.Timestamp('2024-09-28')),color='black',linestyle=':',label='Flood onset: 28 Sep')
        ax.legend()
        fig.savefig(OUT/'sep2024_imerg_vs_era5.png',dpi=160)
        plt.close(fig)
        summary = {'days':len(paired),'imerg_mean_mm_day':float(means.imerg_mm),'era5_mean_mm_day':float(means.era5_mm),
                   'raw_correlation':raw_corr,'percentile_correlation':pct_corr,'provisional_days':provisional_days,
                   'percentile_reference':'own source, 2001–2019 monsoon, ±15 calendar days',
                   'event_reference_note':'matched complete three-day windows; first two June days omitted from both sources because IMERG download is monsoon-only'}
        (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
        report('PASS',f'{verify_frozen()} frozen model/scoring artifacts verified unchanged; rainfall comparison plot saved')
    except (OSError,ValueError,KeyError,AssertionError) as error:
        report('FAIL',str(error))
    failures = sum(row['status']=='FAIL' for row in checks)
    emit(f'\nSummary: {sum(row["status"]=="PASS" for row in checks)} PASS, {sum(row["status"]=="WARN" for row in checks)} WARN, {failures} FAIL')
    pd.DataFrame(checks).to_csv(OUT/'checks.csv',index=False)
    (OUT/'report.txt').write_text('\n'.join(lines)+'\n')
    return int(bool(failures))


if __name__ == '__main__':
    sys.exit(main())
