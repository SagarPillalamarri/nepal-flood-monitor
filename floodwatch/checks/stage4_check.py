"""Audit leave-one-event-out selection and the physical-feature case matches."""
import json
import sys
import numpy as np
import pandas as pd
from floodwatch.core import config as c
from floodwatch.pipeline.stage4_analogues import _window

RESULTS = []


def report(status, message):
    RESULTS.append({'status': status, 'message': message})
    print(f'[{status}] {message}')


def main():
    RESULTS.clear()
    try:
        folder = c.STAGE4_DIR
        features = pd.read_csv(c.STAGE3_FEATURES, parse_dates=['date']).set_index('date')
        missing = sorted(set(c.STAGE4_MATCH_FEATURES) - set(features.columns))
        if missing or 'monsoon_day' in c.STAGE4_MATCH_FEATURES:
            raise ValueError(f'Invalid Stage 4 feature list: missing={missing}, calendar_feature_present={"monsoon_day" in c.STAGE4_MATCH_FEATURES}')
        if tuple(c.STAGE4_MATCH_FEATURES) != ('tcwv_pct','rain_1d_pct','rain_3d_pct','rain_7d_pct','sm_0_7','discharge_rise_1d'):
            raise ValueError('Stage 4 feature list differs from the approved physical variables')
        report('PASS', 'Analogue inputs are TCWV, 1/3/7-day rainfall percentiles, topsoil and 1-day discharge rise; monsoon_day is absent')

        loo = pd.read_csv(folder / 'loo_evaluation.csv', parse_dates=['held_out_onset','matched_event_onset'])
        events = pd.read_csv(c.STAGE2_EVENTS, parse_dates=['start','end','peak_date'])
        train_events = events[events.training_eligible.astype(bool) & events.in_monsoon.astype(bool)]
        expected_queries = len(train_events) * 2 * len(c.STAGE4_WINDOW_CANDIDATES)
        if len(loo) != expected_queries or set(loo.window_days) != set(c.STAGE4_WINDOW_CANDIDATES):
            raise ValueError(f'Unexpected LOEO query count/configurations: {len(loo)} vs {expected_queries}')
        if loo.held_out_event_id.eq(loo.matched_event_id).any():
            raise ValueError('Held-out event appears as its own analogue')
        if loo.held_out_onset.dt.year.gt(2019).any() or loo.matched_event_onset.dt.year.gt(2019).any():
            raise ValueError('Test-period event entered training LOEO selection')
        if not loo.query_days_before_onset.isin([1,2]).all() or not loo.matched_days_before_onset.between(1,14).all():
            raise ValueError('LOEO query or template lead is out of range')
        if not loo.hit_within_3_days.equals(loo.matched_days_before_onset.le(3).rename('hit_within_3_days')):
            raise ValueError('LOEO hit labels do not follow the <=3 day rule')
        report('PASS', f'LOEO covers {len(train_events)} training events at onset-1 and onset-2 for both windows; held-out events are excluded')

        summary = pd.read_csv(folder / 'loo_summary.csv')
        rates = loo.groupby('window_days').hit_within_3_days.mean()
        for row in summary.itertuples():
            np.testing.assert_allclose(row.mean, rates.loc[row.window_days])
        expected_selected = int(summary.sort_values(['mean','window_days'], ascending=[False,True]).iloc[0].window_days)
        selected = json.loads((folder / 'selection.json').read_text())
        meta = json.loads(c.STAGE4_METADATA.read_text())
        if selected['selected_window_days'] != expected_selected or meta['window_days'] != expected_selected:
            raise ValueError('Selected analogue window does not have the best training hit rate')
        report('PASS', f'Window selection follows training LOEO hit rate and the shorter-window tie rule: {expected_selected} days')

        scaler = pd.read_csv(c.STAGE4_SCALERS).set_index('feature').loc[list(c.STAGE4_MATCH_FEATURES)]
        train = features.loc[c.TRAIN_START:c.TRAIN_END, list(c.STAGE4_MATCH_FEATURES)]
        np.testing.assert_allclose(scaler.training_mean, train.mean().to_numpy())
        np.testing.assert_allclose(scaler.training_std, train.std(ddof=0).to_numpy())
        standardized = (features[list(c.STAGE4_MATCH_FEATURES)] - scaler.training_mean.to_numpy()) / scaler.training_std.to_numpy()
        report('PASS', 'Final daily-match standardization uses only 2001–2019 monsoon data')

        templates = pd.read_csv(c.STAGE4_TEMPLATES, parse_dates=['event_onset','event_peak','window_start','window_end'])
        matches = pd.read_csv(c.STAGE4_MATCHES, parse_dates=['date','event_onset','event_peak','matched_window_end'])
        if not matches.date.equals(pd.Series(features.index, name='date')):
            raise ValueError('Daily output does not include each monsoon feature date exactly once')
        if matches.window_days.ne(expected_selected).any():
            raise ValueError('Daily output contains another window size')
        if templates.days_before_flood.lt(1).any() or templates.days_before_flood.gt(14).any():
            raise ValueError('Template lead is outside 1–14 days')
        if (templates.window_end != templates.event_onset - pd.to_timedelta(templates.days_before_flood, unit='D')).any():
            raise ValueError('Template window end does not match event lead')
        if (templates.window_start != templates.window_end - pd.Timedelta(days=expected_selected - 1)).any():
            raise ValueError('Template length differs from selected window')
        if (matches.dropna(subset=['event_onset']).event_onset >= matches.dropna(subset=['event_onset']).date).any():
            raise ValueError('A future or same-day event was used as a daily analogue')
        case = matches[matches.date.between('2024-09-01','2024-09-30')]
        if case.closest_event_id.isna().any() or case.event_onset.dt.year.ge(2024).any():
            raise ValueError('September case is unmatched or uses a 2024+ event')
        report('PASS', 'Daily analogues use complete selected-length templates and September 2024 uses pre-2024 events only')

        for date in pd.to_datetime(['2024-09-20','2024-09-26','2024-09-28']):
            observed = matches.set_index('date').loc[date]
            query = _window(standardized, date, expected_selected)
            candidates = templates[(templates.event_onset < date)
                                   & (templates.event_onset <= pd.Timestamp(c.STAGE4_CASE_HISTORY_END))]
            distances = []
            for row in candidates.itertuples():
                vector = _window(standardized, row.window_end, expected_selected)
                if vector is None:
                    raise ValueError('A saved case template has missing feature data')
                distances.append((float(np.linalg.norm(query-vector)), int(row.event_id), int(row.days_before_flood)))
            best = min(distances)
            if (int(observed.closest_event_id), int(observed.days_before_flood)) != (best[1], best[2]):
                raise ValueError(f'Nearest case analogue does not reproduce on {date.date()}')
            np.testing.assert_allclose(observed.euclidean_distance, best[0])
        incomplete = matches[~matches.window_complete]
        if incomplete.closest_event_id.notna().any():
            raise ValueError('Incomplete query window received a fabricated match')
        report('PASS', f'September representative matches reproduce; {len(incomplete)} incomplete windows remain unmatched')
    except (OSError, ValueError, KeyError, AssertionError) as error:
        report('FAIL', str(error))
    c.STAGE4_REPORT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(RESULTS).to_csv(c.STAGE4_REPORT, index=False)
    failures = sum(row['status'] == 'FAIL' for row in RESULTS)
    print(f'\nSummary: {sum(row["status"] == "PASS" for row in RESULTS)} PASS, {sum(row["status"] == "WARN" for row in RESULTS)} WARN, {failures} FAIL')
    return int(failures > 0)


if __name__ == '__main__':
    sys.exit(main())
