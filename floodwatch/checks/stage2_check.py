"""Validate flood labels, event independence, composites and September 2024."""
import json
import sys
import numpy as np
import pandas as pd
from floodwatch.core import config as c
from floodwatch.pipeline.stage2_analysis import load_input, label_floods, calendar_climatology

results = []


def report(status, message):
    results.append(dict(status=status, message=message))
    print(f'[{status}] {message}')


def main():
    try:
        source = load_input()
        daily = pd.read_csv(c.STAGE2_DAILY, parse_dates=['date']).set_index('date')
        events = pd.read_csv(c.STAGE2_EVENTS, parse_dates=['start','end','peak_date'])
        annual = pd.read_csv(c.STAGE2_ANNUAL)
        windows = pd.read_csv(c.STAGE2_WINDOWS, parse_dates=['date','event_start'])
        composites = pd.read_csv(c.STAGE2_COMPOSITES)
        precursor = pd.read_csv(c.STAGE2_PRECURSORS)
        climate = pd.read_csv(c.STAGE2_CLIMATOLOGY)
        metadata = json.loads(c.STAGE2_METADATA.read_text())
        pd.testing.assert_frame_equal(daily[source.columns], source)
        if not daily.flood.isin([0,1]).all():
            raise ValueError('Flood labels must be binary')
        report('PASS', f'{len(daily)} daily rows; Stage 1b inputs including TCWV NaNs preserved')

        expected, magnitude, fitted = label_floods(source)
        pd.testing.assert_series_equal(daily.flood, expected, check_dtype=False)
        if fitted['source'] == 'GloFAS':
            train = source.loc[c.TRAIN_START:c.TRAIN_END, c.GLOFAS_CHOSEN]
            train = train[train.index.month.isin(c.MONSOON_MONTHS)]
            independent_threshold = float(np.quantile(train, c.FLOOD_QUANTILE))
            np.testing.assert_allclose(metadata['threshold'], independent_threshold)
            report('PASS', f'GloFAS labels use {c.GLOFAS_CHOSEN} > {independent_threshold:.3f} m³/s; threshold fitted on training monsoons only')
        else:
            report('PASS', 'DHM labels match observed water level >= danger level')

        report('PASS' if len(events) >= c.EVENT_MIN_COUNT_WARN else 'WARN',
               f'{len(events)} flood events (warning threshold: {c.EVENT_MIN_COUNT_WARN})')
        if events.empty or events.event_id.duplicated().any():
            raise ValueError('Invalid event IDs')
        separation = (events.start.iloc[1:].reset_index(drop=True) - events.end.iloc[:-1].reset_index(drop=True)).dt.days - 1
        if len(separation) and separation.min() < c.EVENT_DRY_DAYS:
            raise ValueError('Events are not separated by seven dry days')
        flood_rows = daily[daily.flood.eq(1)]
        if flood_rows.event_id.isna().any() or daily.loc[daily.flood.eq(0), 'event_id'].notna().any():
            raise ValueError('Event membership must identify all and only flood days')
        for event in events.itertuples():
            members = flood_rows[flood_rows.event_id.eq(event.event_id)]
            if len(members) != event.flood_days or members.index.min() != event.start or members.index.max() != event.end:
                raise ValueError('Event bounds or flood-day counts disagree')
            if (members.index.to_series().diff().dt.days.dropna() > c.EVENT_DRY_DAYS).any():
                raise ValueError('An event contains an internal seven-day dry gap')
            if magnitude.loc[members.index].idxmax() != event.peak_date:
                raise ValueError('Event peak date is incorrect')
        report('PASS', 'Flood membership, event bounds and peaks consistent; events separated by at least seven dry days')
        outside_days = int((daily.flood.eq(1) & ~daily.index.month.isin(c.MONSOON_MONTHS)).sum())
        outside_events = int((~events.in_monsoon).sum())
        report('WARN' if outside_days else 'PASS', f'Outside June–October: {outside_days} flood days, {outside_events} events')
        expected_years = list(range(pd.Timestamp(c.START_DATE).year,pd.Timestamp(c.END_DATE).year+1))
        if annual.year.tolist() != expected_years or annual.events.sum() != len(events):
            raise ValueError('Annual count table does not cover all years or sum to event count')
        for row in annual.itertuples():
            if row.events != events.year.eq(row.year).sum():
                raise ValueError('Incorrect per-year event count')
        report('PASS', 'Annual event counts cover every year, including zero-event years')

        case = events[(events.start <= pd.Timestamp(c.TCWV_EVENT_END)) & (events.end >= pd.Timestamp(c.CASE_STUDY_START))]
        if len(case) != 1:
            raise ValueError('September 2024 case study must intersect one detected event')
        peak = case.peak_date.iloc[0]
        if abs((peak-pd.Timestamp(c.CASE_PEAK_DATE)).days) > c.CASE_PEAK_TOLERANCE_DAYS:
            raise ValueError(f'September case peak {peak.date()} differs from expected date')
        if case.training_eligible.any() or (events.loc[events.training_eligible, 'end'] > pd.Timestamp(c.TRAIN_END)).any():
            raise ValueError('Test/case event marked training eligible')
        report('PASS', f'September 2024 event detected: start {case.start.iloc[0].date()}, peak {peak.date()}; excluded from training')

        # Fit again using a real prefix, not altered or simulated future records.
        prefix = source.loc[:c.TRAIN_END]
        prefix_climate = calendar_climatology(prefix)
        full_climate = calendar_climatology(source)
        pd.testing.assert_frame_equal(prefix_climate, full_climate)
        pd.testing.assert_frame_equal(climate, full_climate, check_dtype=False)
        if fitted['source'] == 'GloFAS':
            early_labels, _, early_metadata = label_floods(prefix)
            np.testing.assert_allclose(early_metadata['threshold'], metadata['threshold'])
            pd.testing.assert_series_equal(early_labels, expected.loc[prefix.index])
        report('PASS', 'No test-year leakage: truncating input at 2019 leaves labels, threshold and calendar climatology unchanged')

        expected_rows = len(events) * len(source.columns) * (c.COMPOSITE_LAG_END-c.COMPOSITE_LAG_START+1)
        if len(windows) != expected_rows or windows.duplicated(['event_id','variable','lag']).any():
            raise ValueError('Event windows are incomplete or duplicated')
        for variable, group in windows.groupby('variable'):
            wanted = source[variable].reindex(pd.DatetimeIndex(group.date)).to_numpy()
            np.testing.assert_allclose(group.value.to_numpy(), wanted, equal_nan=True)
        report('PASS', 'Every variable has complete -14 to +3 windows; values match source dates, including NaNs')

        lookup = composites.set_index(['variable','lag'])
        for key, group in windows.groupby(['variable','lag']):
            paired = group.dropna(subset=['value','normal_mean','normal_q75'])
            result = lookup.loc[key]
            if result.paired_n != len(paired) or result.observed_n != group.value.notna().sum():
                raise ValueError('Composite sample counts incorrect')
            np.testing.assert_allclose(result.event_mean, paired.value.mean(), equal_nan=True)
            np.testing.assert_allclose(result.normal_mean, paired.normal_mean.mean(), equal_nan=True)
            np.testing.assert_allclose(result.anomaly_mean, (paired.value-paired.normal_mean).mean(), equal_nan=True)
        report('PASS', 'Composites skip NaNs with paired sample counts; no missing observations become zeros')
        missing = windows[(windows.variable == 'tcwv_mm') & windows.value.isna()]
        report('WARN' if len(missing) else 'PASS', f'TCWV source gap affects {len(missing)} event-window observations; retained as NaN and skipped in composite means')
        shoulder = windows[windows.normal_mean.isna() & windows.value.notna()]
        report('WARN' if len(shoulder) else 'PASS', f'{len(shoulder)} observed window values have no monsoon calendar-day baseline (season edges); excluded from paired means')
        if set(precursor.variable) != set(source.columns):
            raise ValueError('Precursor table omits variables')
        for row in precursor.itertuples():
            group = composites[(composites.variable == row.variable) & composites.lag.lt(0)]
            crossed = group[(group.paired_n >= c.PRECURSOR_MIN_EVENTS) & (group.event_mean > group.normal_q75)]
            want = crossed.lag.min() if len(crossed) else np.nan
            np.testing.assert_allclose(row.first_lag, want, equal_nan=True)
        report('PASS', 'Precursor timing uses first pre-event crossing above calendar-day 75th percentile')
        paths = [c.STAGE2_DIR / f'composite_{variable}.png' for variable in source.columns]
        paths.append(c.STAGE2_DIR / 'case_study_labels.png')
        if any(not path.exists() or path.stat().st_size == 0 for path in paths):
            raise ValueError('Missing or empty plots')
        report('PASS', f'{len(paths)} plots saved, including September 2024 labels')
        from floodwatch.core.features import fit_reference, build_features
        conditions = pd.read_csv(c.STAGE2_CONDITIONAL)
        feature_days = build_features(source,fit_reference(source)).loc[c.TRAIN_START:c.TRAIN_END]
        for row in conditions.itertuples():
            dates = feature_days.index
            eligible = (dates+pd.Timedelta(days=row.horizon_days)) <= pd.Timestamp(c.TRAIN_END)
            if row.population == 'not_currently_flooding':
                eligible &= daily.flood.reindex(dates).eq(0).to_numpy()
            if row.condition == 'tcwv_pct_ge_90':
                eligible &= feature_days.tcwv_pct.ge(c.CONDITIONAL_TCWV_LEVELS[0]).to_numpy()
            elif row.condition == 'tcwv_pct_ge_95':
                eligible &= feature_days.tcwv_pct.ge(c.CONDITIONAL_TCWV_LEVELS[1]).to_numpy()
            elif row.condition == 'tcwv_pct_ge_95_and_soil_gt_75':
                eligible &= (feature_days.tcwv_pct.ge(c.CONDITIONAL_TCWV_LEVELS[1]) & feature_days.sm_0_7_pct.gt(c.CONDITIONAL_SOIL_PERCENTILE)).to_numpy()
            elif row.condition == 'rain_3d_pct_ge_95':
                eligible &= feature_days.rain_3d_pct.ge(c.CONDITIONAL_RAIN_PERCENTILE).to_numpy()
            elif row.condition != 'all_days':
                raise ValueError('Unknown conditional probability condition')
            selected = dates[eligible]
            positive = sum(((events.start > date) & (events.start <= date+pd.Timedelta(days=row.horizon_days))).any() for date in selected)
            if row.n_days != len(selected) or row.onset_within_horizon_days != positive:
                raise ValueError('Conditional probability denominator or onset count incorrect')
            np.testing.assert_allclose(row.probability,positive/len(selected) if len(selected) else np.nan,equal_nan=True)
        report('PASS', 'Conditional probabilities: training-only percentiles, day counts and future onset counts verified for both populations')

    except (OSError, ValueError, KeyError, AssertionError) as error:
        report('FAIL', str(error))
    c.STAGE2_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(c.STAGE2_REPORT, index=False)
    statuses = [r['status'] for r in results]
    print(f"\nSummary: {statuses.count('PASS')} PASS, {statuses.count('WARN')} WARN, {statuses.count('FAIL')} FAIL")
    return int('FAIL' in statuses)


if __name__ == '__main__':
    sys.exit(main())
