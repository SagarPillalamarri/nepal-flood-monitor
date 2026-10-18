"""Flood onset targets, observed at future dates and never used as features."""
import numpy as np
import pandas as pd
import config as c


def onset_series(index, events):
    onset = pd.Series(0, index=index, dtype='int8', name='onset')
    onset.loc[pd.DatetimeIndex(events.start)] = 1
    return onset


def onset_target(onset, horizon):
    # Shift toward today so the target covers t+1 through t+h, excluding t.
    future = pd.concat([onset.shift(-lead) for lead in range(1, horizon+1)], axis=1)
    target = future.max(axis=1).where(future.notna().all(axis=1))
    return target.rename('target')


def eligible_rows(daily, features, target, start, end, horizon, exclude_case=False):
    label_end = features.index + pd.Timedelta(days=horizon)
    mask = ((features.index >= pd.Timestamp(start)) & (features.index <= pd.Timestamp(end))
            & (label_end <= pd.Timestamp(end)) & daily.flood.reindex(features.index).eq(0)
            & target.reindex(features.index).notna())
    if exclude_case:
        # Purge dates whose outcome window intersects the separately reported month.
        case_overlap = ((features.index <= pd.Timestamp(c.CASE_REPORT_END))
                        & (label_end >= pd.Timestamp(c.CASE_REPORT_START)))
        mask &= ~case_overlap
    return features.index[mask]
