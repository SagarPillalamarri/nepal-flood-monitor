"""Shared feature definitions for historical analysis and later live scoring."""
import numpy as np
import pandas as pd
from floodwatch.core import config as c


def tcwv_percentile(values, reference):
    """Rank against a supplied, frozen source-specific monsoon climatology.

    Use a +/-15 calendar-day window. Month/day mapped onto a leap year
    keeps September aligned across leap and non-leap years. No future
    observations in values are used to fit the reference distribution.
    """
    reference = reference[reference.index.month.isin(c.MONSOON_MONTHS)].dropna()
    if reference.empty:
        raise ValueError("TCWV reference climatology is empty")
    calendar = str(c.CALENDAR_REFERENCE_YEAR) + '-'
    ref_day = pd.to_datetime(calendar + reference.index.strftime('%m-%d')).dayofyear
    days = pd.to_datetime(calendar + values.index.strftime('%m-%d')).dayofyear
    out = pd.Series(index=values.index, dtype=float, name="tcwv_percentile")
    for day in np.unique(days):
        distance = np.abs(ref_day - day)
        pool = reference.to_numpy()[np.minimum(distance, c.CALENDAR_CYCLE_DAYS - distance) <= c.PERCENTILE_WINDOW_DAYS]
        if len(pool) < c.PERCENTILE_MIN_SAMPLES:
            raise ValueError(f"Too few climatology samples for calendar day {day}: {len(pool)}")
        selected = days == day
        # Midrank handles tied measurements without assigning all ties to zero.
        x = np.round(values.to_numpy()[selected], c.PERCENTILE_ROUND_DECIMALS)
        pool = np.sort(np.round(pool, c.PERCENTILE_ROUND_DECIMALS))
        out.iloc[np.flatnonzero(selected)] = 100 * (np.searchsorted(pool, x, side='left') + np.searchsorted(pool, x, side='right')) / (2 * len(pool))
    out[values.isna()] = np.nan
    return out


def raw_signals(data):
    """Trailing windows include today and never use data after today's close."""
    if not data.index.is_monotonic_increasing or data.index.has_duplicates:
        raise ValueError('Feature input must be sorted with unique daily dates')
    if not data.index.equals(pd.date_range(data.index.min(), data.index.max(), name=data.index.name)):
        raise ValueError('Feature windows require a continuous daily input calendar')
    signals = pd.DataFrame(index=data.index)
    for days in c.RAINFALL_WINDOWS:
        signals[f'rain_{days}d'] = data.precip_mm.rolling(days, min_periods=days).sum()
    signals['rain_max_point'] = data.precip_max_point_mm
    signals['tcwv'] = data.tcwv_mm
    signals['sm_0_7'] = data.sm_0_7
    return signals


def fit_reference(data, end=None):
    """Freeze raw seasonal distributions on the specified training prefix."""
    end = c.TRAIN_END if end is None else end
    if pd.Timestamp(end) > pd.Timestamp(c.TRAIN_END):
        raise ValueError('Feature references cannot include test-period dates')
    signals = raw_signals(data.loc[:end]).loc[c.TRAIN_START:end]
    return signals.loc[signals.index.month.isin(c.MONSOON_MONTHS)]


def build_features(data, reference):
    """One shared causal implementation for training, case replay and scoring.

    Reference is supplied explicitly so live/replay code cannot silently refit it.
    Soil definitions are the existing ERA5 layer depths; operational model/source
    equivalence remains deferred and this project currently runs historical replay.
    """
    if reference.index.max() > pd.Timestamp(c.TRAIN_END):
        raise ValueError('Reference contains test-period data')
    signals = raw_signals(data)
    days = data.index[data.index.month.isin(c.MONSOON_MONTHS)]
    signals = signals.loc[days]
    out = pd.DataFrame(index=days)
    for column in signals:
        out[f'{column}_pct'] = tcwv_percentile(signals[column], reference[column])
    for column in c.SOIL_COLUMNS:
        out[column] = data.loc[days, column]
        out[f'{column}_change_{c.SOIL_CHANGE_DAYS}d'] = data[column].diff(c.SOIL_CHANGE_DAYS).loc[days]
    out['discharge'] = data.loc[days, c.GLOFAS_CHOSEN]
    for lag in c.DISCHARGE_RISE_WINDOWS:
        out[f'discharge_rise_{lag}d'] = data[c.GLOFAS_CHOSEN].diff(lag).div(lag).loc[days]
    starts = pd.to_datetime(dict(year=days.year, month=c.MONSOON_START_MONTH, day=c.MONSOON_START_DAY))
    out['monsoon_day'] = (days-pd.DatetimeIndex(starts)).days + 1
    out['oni'] = data.loc[days, 'oni']
    out.index.name = 'date'
    return out


def variant_columns(features, variant):
    columns = [name for name in features.columns if name != 'oni']
    if variant == 'xgb_core':
        columns = [name for name in columns if name != 'tcwv_pct']
    elif variant == 'xgb_oni_ablation':
        columns.append('oni')
    elif variant != 'xgb_tcwv':
        raise ValueError(f'Unknown feature variant: {variant}')
    return columns
