"""Build daily flood labels, independent events and descriptive composites."""
import json
import sys
import numpy as np
import pandas as pd
import config as c
from stage1b_fetch import read_daily, require_dates


def load_input():
    data = read_daily(c.STAGE1B_FILE, allow_tcwv_gap=True)
    require_dates(data, c.START_DATE, c.END_DATE, allow_tcwv_gap=True)
    return data


def label_floods(data):
    """Fit the fallback threshold on training monsoons, never on test years."""
    if c.DHM_FILE.exists():
        dhm = pd.read_csv(c.DHM_FILE, parse_dates=[c.DHM_DATE_COLUMN])
        dhm = dhm.set_index(c.DHM_DATE_COLUMN).sort_index()
        levels = dhm[[c.DHM_LEVEL_COLUMN, c.DHM_DANGER_COLUMN]]
        require_dates(levels, c.START_DATE, c.END_DATE)
        if (levels[c.DHM_DANGER_COLUMN] <= 0).any():
            raise ValueError('DHM danger levels must be positive')
        flood = levels[c.DHM_LEVEL_COLUMN] >= levels[c.DHM_DANGER_COLUMN]
        magnitude = levels[c.DHM_LEVEL_COLUMN].rename('flood_magnitude')
        metadata = dict(source='DHM', comparison='water_level >= danger_level', threshold=None)
    else:
        reference = data.loc[c.TRAIN_START:c.TRAIN_END, c.GLOFAS_CHOSEN]
        reference = reference[reference.index.month.isin(c.MONSOON_MONTHS)]
        if reference.empty or reference.isna().any():
            raise ValueError('Missing GloFAS threshold reference')
        threshold = reference.quantile(c.FLOOD_QUANTILE)
        magnitude = data[c.GLOFAS_CHOSEN].rename('flood_magnitude')
        flood = magnitude > threshold
        metadata = dict(source='GloFAS', point=c.GLOFAS_CHOSEN, threshold=float(threshold),
                        quantile=c.FLOOD_QUANTILE, comparison='discharge > threshold',
                        reference_start=c.TRAIN_START, reference_end=c.TRAIN_END,
                        reference_days=len(reference))
    return flood.astype('int8').rename('flood'), magnitude, metadata


def group_events(flood, magnitude):
    """Merge flood runs until at least seven fully non-flood days separate them."""
    dates = flood.index[flood.eq(1)]
    if dates.empty:
        raise ValueError('No floods found')
    groups = dates.to_series().diff().dt.days.gt(c.EVENT_DRY_DAYS).cumsum()
    records = []
    assignments = pd.Series(pd.NA, index=flood.index, dtype='Int64', name='event_id')
    for number, (_, group) in enumerate(groups.groupby(groups), start=1):
        # `group` contains group numbers; the index retains the actual flood dates.
        members = group.index
        peak_date = magnitude.loc[members].idxmax()
        assignments.loc[members] = number
        records.append(dict(event_id=number, start=members.min(), end=members.max(),
                            peak_date=peak_date, peak_value=float(magnitude.loc[peak_date]),
                            flood_days=len(members), span_days=(members.max()-members.min()).days+1,
                            year=members.min().year,
                            in_monsoon=bool(members.month.isin(c.MONSOON_MONTHS).all()),
                            training_eligible=bool(members.max() <= pd.Timestamp(c.TRAIN_END))))
    return pd.DataFrame(records), assignments


def calendar_climatology(data):
    """Exact month/day climatology, fit only to training-period monsoon days."""
    reference = data.loc[c.TRAIN_START:c.TRAIN_END]
    reference = reference[reference.index.month.isin(c.MONSOON_MONTHS)]
    rows = []
    for calendar_day, frame in reference.groupby(reference.index.strftime('%m-%d')):
        for variable in data.columns:
            values = frame[variable].dropna()
            rows.append(dict(calendar_day=calendar_day, variable=variable,
                             normal_mean=values.mean(), normal_q75=values.quantile(c.NORMAL_QUANTILE),
                             reference_n=len(values)))
    return pd.DataFrame(rows)


def event_windows(data, events, climatology):
    lookup = climatology.set_index(['calendar_day', 'variable'])
    rows = []
    for event in events.itertuples():
        for lag in range(c.COMPOSITE_LAG_START, c.COMPOSITE_LAG_END + 1):
            date = event.start + pd.Timedelta(days=lag)
            for variable in data.columns:
                actual = data.at[date, variable] if date in data.index else np.nan
                key = (date.strftime('%m-%d'), variable)
                normal = lookup.loc[key] if key in lookup.index else None
                mean = normal.normal_mean if normal is not None else np.nan
                q75 = normal.normal_q75 if normal is not None else np.nan
                rows.append(dict(event_id=event.event_id, event_start=event.start, date=date,
                                 lag=lag, variable=variable, value=actual, normal_mean=mean,
                                 normal_q75=q75, anomaly=actual-mean,
                                 training_event=event.training_eligible))
    return pd.DataFrame(rows)


def summarize_windows(windows):
    rows = []
    for (variable, lag), group in windows.groupby(['variable', 'lag']):
        paired = group.dropna(subset=['value', 'normal_mean', 'normal_q75'])
        rows.append(dict(variable=variable, lag=lag, events_total=len(group),
                         observed_n=int(group.value.notna().sum()), paired_n=len(paired),
                         missing_n=int(group.value.isna().sum()),
                         observed_mean=group.value.mean(),
                         event_mean=paired.value.mean(), normal_mean=paired.normal_mean.mean(),
                         normal_q75=paired.normal_q75.mean(), anomaly_mean=paired.anomaly.mean()))
    return pd.DataFrame(rows)


def precursor_timing(composites):
    rows = []
    for variable, group in composites.groupby('variable'):
        pre = group[group.lag.lt(0)].sort_values('lag')
        above = pre[(pre.paired_n >= c.PRECURSOR_MIN_EVENTS) & (pre.event_mean > pre.normal_q75)]
        first = above.iloc[0] if len(above) else None
        rows.append(dict(variable=variable,
                         first_lag=first.lag if first is not None else np.nan,
                         days_before_event=-first.lag if first is not None else np.nan,
                         left_censored=bool(first is not None and first.lag == c.COMPOSITE_LAG_START),
                         paired_events_at_crossing=int(first.paired_n) if first is not None else 0))
    return pd.DataFrame(rows)


def save_plots(composites, data, magnitude, flood):
    import os
    os.environ.setdefault('MPLCONFIGDIR', str(c.MATPLOTLIB_CACHE))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for variable, group in composites.groupby('variable'):
        group = group.sort_values('lag')
        fig, axes = plt.subplots(2, 1, figsize=c.STAGE2_PLOT_SIZE, sharex=True,
                                 gridspec_kw={'height_ratios': [3, 1]}, constrained_layout=True)
        axes[0].plot(group.lag, group.event_mean, marker='o', label='Event composite (paired samples)')
        axes[0].plot(group.lag, group.normal_mean, label='Training calendar-day normal')
        axes[0].plot(group.lag, group.normal_q75, '--', label='Training calendar-day 75th percentile')
        axes[0].axvline(0, color='black', alpha=.4)
        axes[0].set(title=variable, ylabel=c.STAGE2_UNITS.get(variable, variable))
        axes[0].legend(fontsize='small')
        axes[1].bar(group.lag, group.paired_n, color='slateblue')
        axes[1].set(xlabel='Days relative to first flood day', ylabel='Paired n')
        fig.savefig(c.STAGE2_DIR / f'composite_{variable}.png', dpi=c.STAGE2_PLOT_DPI)
        plt.close(fig)
    case = data.loc[c.CASE_STUDY_START:c.CASE_STUDY_END]
    fig, ax = plt.subplots(figsize=c.STAGE2_PLOT_SIZE, constrained_layout=True)
    ax.plot(case.index, magnitude.loc[case.index], marker='o', label='Flood source magnitude')
    for date in case.index[flood.loc[case.index].eq(1)]:
        ax.axvline(date, color='tomato', alpha=.2)
    ax.set(title='September 2024: shaded days exceed flood threshold', ylabel='Flood source magnitude')
    ax.legend()
    fig.autofmt_xdate()
    fig.savefig(c.STAGE2_DIR / 'case_study_labels.png', dpi=c.STAGE2_PLOT_DPI)
    plt.close(fig)


def conditional_probabilities(data, events, flood):
    from features import fit_reference, build_features
    from targets import onset_series, onset_target
    features = build_features(data, fit_reference(data)).loc[c.TRAIN_START:c.TRAIN_END]
    onset = onset_series(data.index, events)
    conditions = {
        'all_days': pd.Series(True, index=features.index),
        'tcwv_pct_ge_90': features.tcwv_pct >= c.CONDITIONAL_TCWV_LEVELS[0],
        'tcwv_pct_ge_95': features.tcwv_pct >= c.CONDITIONAL_TCWV_LEVELS[1],
        'tcwv_pct_ge_95_and_soil_gt_75': ((features.tcwv_pct >= c.CONDITIONAL_TCWV_LEVELS[1])
                                       & (features.sm_0_7_pct > c.CONDITIONAL_SOIL_PERCENTILE)),
        'rain_3d_pct_ge_95': features.rain_3d_pct >= c.CONDITIONAL_RAIN_PERCENTILE,
    }
    rows = []
    for horizon in c.TARGET_HORIZONS:
        target = onset_target(onset, horizon).reindex(features.index)
        known = target.notna() & ((features.index + pd.Timedelta(days=horizon)) <= pd.Timestamp(c.TRAIN_END))
        for population in ('all_monsoon_days', 'not_currently_flooding'):
            population_mask = known.copy()
            if population == 'not_currently_flooding':
                population_mask &= flood.reindex(features.index).eq(0)
            for name, condition in conditions.items():
                mask = population_mask & condition
                count = int(mask.sum())
                positives = int(target.loc[mask].sum())
                rows.append(dict(population=population, condition=name, horizon_days=horizon,
                                 n_days=count, onset_within_horizon_days=positives,
                                 probability=positives/count if count else np.nan))
    return pd.DataFrame(rows)


def main():
    data = load_input()
    c.STAGE2_DIR.mkdir(parents=True, exist_ok=True)
    flood, magnitude, metadata = label_floods(data)
    events, assignments = group_events(flood, magnitude)
    climate = calendar_climatology(data)
    windows = event_windows(data, events, climate)
    composites = summarize_windows(windows)
    precursors = precursor_timing(composites)
    daily = data.join(flood).join(assignments)
    conditional = conditional_probabilities(data, events, flood)
    conditional.to_csv(c.STAGE2_CONDITIONAL, index=False)
    annual = events.groupby('year').size().reindex(range(pd.Timestamp(c.START_DATE).year,
                                                      pd.Timestamp(c.END_DATE).year+1), fill_value=0)
    annual = annual.rename('events').rename_axis('year').reset_index()
    daily.to_csv(c.STAGE2_DAILY)
    for table, path in ((events,c.STAGE2_EVENTS), (annual,c.STAGE2_ANNUAL),
                        (climate,c.STAGE2_CLIMATOLOGY), (windows,c.STAGE2_WINDOWS),
                        (composites,c.STAGE2_COMPOSITES), (precursors,c.STAGE2_PRECURSORS)):
        table.to_csv(path, index=False)
    metadata.update(event_dry_days=c.EVENT_DRY_DAYS, alignment='first flood day',
                    climatology_start=c.TRAIN_START, climatology_end=c.TRAIN_END,
                    nan_policy='skip NaN and report paired counts; never zero-fill')
    c.STAGE2_METADATA.write_text(json.dumps(metadata, indent=2))
    save_plots(composites, data, magnitude, flood)
    print(f'Label source: {metadata}')
    print('\nAll event dates:')
    print(events[['event_id', 'start', 'peak_date']].to_string(index=False))
    print('\nConditional probabilities (training monsoons):')
    print(conditional.to_string(index=False))
    print(f'\nEvents per year ({len(events)} total):\n{annual.to_string(index=False)}')
    print(f'\nComposite precursor timing:\n{precursors.to_string(index=False)}')
    print(f'\nSaved outputs to {c.STAGE2_DIR}. Run stage2_check.py')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, ImportError) as error:
        print(f'[FAIL] {error}', file=sys.stderr)
        sys.exit(1)
