"""Select and apply a physical-feature Euclidean flood analogue."""
import json
import sys
import numpy as np
import pandas as pd
from floodwatch.core import config as c


def _window(matrix, end_date, days):
    dates = pd.date_range(pd.Timestamp(end_date) - pd.Timedelta(days=days - 1), pd.Timestamp(end_date))
    if not dates.isin(matrix.index).all():
        return None
    values = matrix.loc[dates].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        return None
    return values.reshape(-1)


def _scaled(features, reference_dates):
    reference = features.loc[reference_dates, c.STAGE4_MATCH_FEATURES]
    mean, std = reference.mean(), reference.std(ddof=0)
    if mean.isna().any() or std.isna().any() or std.eq(0).any():
        raise ValueError('Stage 4 cannot standardize incomplete or constant training features')
    return (features[list(c.STAGE4_MATCH_FEATURES)] - mean) / std, mean, std


def _event_template_rows(matrix, event, window_days):
    rows = []
    for lead in c.STAGE4_TEMPLATE_LEADS:
        end = event.start - pd.Timedelta(days=lead)
        vector = _window(matrix, end, window_days)
        if vector is not None:
            rows.append({'event_id': int(event.event_id), 'event_onset': event.start,
                         'event_peak': event.peak_date, 'days_before_flood': int(lead),
                         'window_start': end - pd.Timedelta(days=window_days - 1),
                         'window_end': end, '_vector': vector})
    return rows


def leave_one_event_out(features, events):
    train_events = events[events.training_eligible.astype(bool)
                          & events.in_monsoon.astype(bool)].sort_values('start')
    train_dates = features.index[(features.index >= pd.Timestamp(c.TRAIN_START))
                                 & (features.index <= pd.Timestamp(c.TRAIN_END))]
    rows = []
    for held in train_events.itertuples():
        # Keep both query windows and the event itself out of the fold's scaling fit.
        excluded = set(pd.date_range(held.start - pd.Timedelta(days=max(c.STAGE4_WINDOW_CANDIDATES) + 1),
                                     held.start + pd.Timedelta(days=int(held.span_days) - 1)))
        reference_dates = pd.DatetimeIndex([date for date in train_dates if date not in excluded])
        for window_days in c.STAGE4_WINDOW_CANDIDATES:
            matrix, _, _ = _scaled(features, reference_dates)
            templates = []
            for event in train_events.itertuples():
                if int(event.event_id) == int(held.event_id):
                    continue
                for template in _event_template_rows(matrix, event, window_days):
                    template_dates = pd.date_range(template['window_start'], template['window_end'])
                    if set(template_dates).intersection(excluded):
                        continue
                    templates.append(template)
            for offset in (1, 2):
                query_end = held.start - pd.Timedelta(days=offset)
                query = _window(matrix, query_end, window_days)
                if query is None or not templates:
                    raise ValueError(f'Incomplete leave-one-event-out window for event {held.event_id}, lead {offset}, {window_days}d')
                distances = [float(np.linalg.norm(query - item['_vector'])) for item in templates]
                best_index = min(range(len(templates)), key=lambda i: (distances[i], templates[i]['event_id'], templates[i]['days_before_flood']))
                best = templates[best_index]
                rows.append({'window_days': window_days, 'held_out_event_id': int(held.event_id),
                             'held_out_onset': held.start, 'query_days_before_onset': offset,
                             'matched_event_id': best['event_id'], 'matched_event_onset': best['event_onset'],
                             'matched_days_before_onset': best['days_before_flood'],
                             'distance': distances[best_index], 'templates_considered': len(templates),
                             'hit_within_3_days': best['days_before_flood'] <= 3})
    return pd.DataFrame(rows)


def select_window(loo):
    summary = loo.groupby('window_days').hit_within_3_days.agg(['sum', 'count', 'mean']).reset_index()
    summary = summary.sort_values(['mean', 'window_days'], ascending=[False, True])
    return int(summary.iloc[0].window_days), summary


def build():
    c.STAGE4_DIR.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(c.STAGE3_FEATURES, parse_dates=['date']).set_index('date')
    missing = sorted(set(c.STAGE4_MATCH_FEATURES) - set(features.columns))
    if missing:
        raise ValueError(f'Stage 4 feature columns missing: {missing}')
    events = pd.read_csv(c.STAGE2_EVENTS, parse_dates=['start', 'end', 'peak_date'])
    loo = leave_one_event_out(features, events)
    loo.to_csv(c.STAGE4_DIR / 'loo_evaluation.csv', index=False)
    selected_days, summary = select_window(loo)
    summary.to_csv(c.STAGE4_DIR / 'loo_summary.csv', index=False)

    train_dates = features.index[(features.index >= pd.Timestamp(c.TRAIN_START))
                                 & (features.index <= pd.Timestamp(c.TRAIN_END))]
    standardized, mean, std = _scaled(features, train_dates)
    pd.DataFrame({'feature': c.STAGE4_MATCH_FEATURES, 'training_mean': mean.to_numpy(),
                  'training_std': std.to_numpy(), 'training_start': c.TRAIN_START,
                  'training_end': c.TRAIN_END}).to_csv(c.STAGE4_SCALERS, index=False)

    templates = []
    for event in events.itertuples():
        for item in _event_template_rows(standardized, event, selected_days):
            templates.append(item)
    if not templates:
        raise ValueError('No complete Stage 4 templates available')
    template_rows = []
    for item in templates:
        row = {key: value for key, value in item.items() if key != '_vector'}
        for day_i in range(selected_days):
            for feature_i, name in enumerate(c.STAGE4_MATCH_FEATURES):
                row[f'day_{day_i + 1:02d}__{name}'] = item['_vector'][day_i * len(c.STAGE4_MATCH_FEATURES) + feature_i]
        template_rows.append(row)
    pd.DataFrame(template_rows).to_csv(c.STAGE4_TEMPLATES, index=False)

    match_rows = []
    for date in standardized.index:
        query = _window(standardized, date, selected_days)
        eligible = [item for item in templates if item['event_onset'] < date]
        if date.strftime('%Y-%m') == '2024-09' and c.STAGE4_EXCLUDE_CASE_TEMPLATES:
            eligible = [item for item in eligible if item['event_onset'] <= pd.Timestamp(c.STAGE4_CASE_HISTORY_END)]
        if query is None or not eligible:
            match_rows.append({'date': date, 'window_days': selected_days,
                               'closest_event_id': pd.NA, 'event_onset': pd.NaT,
                               'event_peak': pd.NaT, 'days_before_flood': pd.NA,
                               'matched_window_end': pd.NaT, 'euclidean_distance': np.nan,
                               'templates_considered': len(eligible), 'window_complete': query is not None})
            continue
        distances = [float(np.linalg.norm(query - item['_vector'])) for item in eligible]
        best_index = min(range(len(eligible)), key=lambda i: (distances[i], eligible[i]['event_id'], eligible[i]['days_before_flood']))
        best = eligible[best_index]
        match_rows.append({'date': date, 'window_days': selected_days,
                           'closest_event_id': best['event_id'], 'event_onset': best['event_onset'],
                           'event_peak': best['event_peak'], 'days_before_flood': best['days_before_flood'],
                           'matched_window_end': best['window_end'],
                           'euclidean_distance': distances[best_index],
                           'templates_considered': len(eligible), 'window_complete': True})
    matches = pd.DataFrame(match_rows)
    matches.to_csv(c.STAGE4_MATCHES, index=False)

    selection = {'selected_window_days': selected_days, 'features': list(c.STAGE4_MATCH_FEATURES),
                 'leave_one_event_out_queries': len(loo),
                 'hit_rates': {str(int(row.window_days)): float(row['mean']) for _, row in summary.iterrows()},
                 'hit_counts': {str(int(row.window_days)): [int(row['sum']), int(row['count'])] for _, row in summary.iterrows()},
                 'tie_rule': 'choose 7-day window if hit rates tie',
                 'selection_period': '2001–2019 monsoon events only',
                 'case_template_cutoff': c.STAGE4_CASE_HISTORY_END}
    (c.STAGE4_DIR / 'selection.json').write_text(json.dumps(selection, indent=2))
    metadata = {'method': 'Euclidean distance on flattened standardized physical-feature windows',
                'features': list(c.STAGE4_MATCH_FEATURES), 'window_days': selected_days,
                'template_alignment': 'for each event and lead 1..14, use the selected-length window ending that many days before onset',
                'scaling': 'per-feature mean and population standard deviation from 2001–2019 monsoon rows',
                'candidate_rule': 'leave-one-event-out for training selection; event onset must precede query for daily output',
                'september_2024_rule': 'only event onsets on or before 2023-12-31 are considered',
                'missing_policy': 'skip a query or template window if any date/feature is missing; no imputation',
                'dtw_used': False, 'templates_saved': len(templates), 'matches_saved': len(matches),
                'matched_days': int(matches.closest_event_id.notna().sum()), 'selection': selection}
    c.STAGE4_METADATA.write_text(json.dumps(metadata, indent=2))
    return loo, summary, matches


def main():
    loo, summary, matches = build()
    case = matches.loc[matches.date.between('2024-09-20', '2024-09-30')]
    print('LEAVE-ONE-EVENT-OUT TRAINING HIT RATES (hit = top analogue lead is 1–3 days):')
    print(summary.to_string(index=False))
    print('\nSEPTEMBER 2024 DAILY CLOSEST PHYSICAL ANALOGUE:')
    print(case[['date', 'closest_event_id', 'event_onset', 'days_before_flood',
                'euclidean_distance', 'templates_considered']].to_string(index=False))
    print(f'\nSelected {int(summary.sort_values(["mean", "window_days"], ascending=[False, True]).iloc[0].window_days)}-day window from {len(loo)} training queries.')
    print(f'Saved {len(matches)} daily rows; {matches.closest_event_id.notna().sum()} have a match. Run stage4_check.py.')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, AssertionError) as error:
        print(f'[FAIL] {error}', file=sys.stderr)
        sys.exit(1)
