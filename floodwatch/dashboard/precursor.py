"""Snapshot + 2001–2019 training-table explanations; never feeds model scores."""
import pandas as pd
import streamlit as st
from floodwatch.core import config as c
from floodwatch.core.features import tcwv_percentile
from floodwatch.operational.live import load_score
from floodwatch.live.observed_poll import latest_observed,rebuild_observed
from floodwatch.live.live_poll import latest,rebuild,THRESHOLD
from floodwatch.dashboard.plain_language import caption,PERCENTILE


def panel_data(snapshot=None, panel_time=None):
    snapshot=latest() if snapshot is None else snapshot
    if snapshot is None:return {'available':False,'text':'No live snapshot is available. Use Fetch now.'}
    result=load_score(snapshot)
    if result is None or not result.get('features'):return {'available':False,'text':'No operational features are saved for this snapshot. Use Fetch now.'}
    f=result['features'];day=pd.Timestamp(result['date']);water=f['tcwv_pct'];rain7=f['rain_7d_pct'];rain14=f['rain_14d_pct']
    table=pd.read_csv(c.STAGE2_CONDITIONAL);table=table[table.population.eq('all_monsoon_days')&table.horizon_days.eq(3)].set_index('condition')
    base=table.loc['all_days'];ge90=table.loc['tcwv_pct_ge_90'];ge95=table.loc['tcwv_pct_ge_95']
    if water is None:return {'available':False,'text':'Water-vapour data is missing; no precursor summary is available.'}
    if water<90:band='<90';n=int(base.n_days-ge90.n_days);k=int(base.onset_within_horizon_days-ge90.onset_within_horizon_days)
    elif water<95:band='90–95';n=int(ge90.n_days-ge95.n_days);k=int(ge90.onset_within_horizon_days-ge95.onset_within_horizon_days)
    else:band='≥95';n=int(ge95.n_days);k=int(ge95.onset_within_horizon_days)
    probability=k/n;baseline=float(base.probability)
    train=pd.read_csv(c.STAGE1_FILE,index_col=0,parse_dates=True).loc[c.TRAIN_START:c.TRAIN_END,c.GLOFAS_CHOSEN]
    q=f['discharge'];qp=None if q is None else float(tcwv_percentile(pd.Series([q],index=pd.DatetimeIndex([day])),train).iloc[0]);gap=None if q is None else THRESHOLD-q
    observed=latest_observed();gauge=None
    if observed is not None:
        _,_,obs=rebuild_observed(observed);warning=obs['warning_level_m']
        if warning and obs['latest_water_level_m'] is not None:gauge={'percent':100*obs['latest_water_level_m']/warning,'level':obs['latest_water_level_m'],'warning':warning,'time':obs['latest_time_nepal']}
    panel_time=pd.Timestamp.now(tz='UTC') if panel_time is None else pd.Timestamp(panel_time)
    if panel_time.tzinfo is None:panel_time=panel_time.tz_localize('UTC')
    age_hours=None if gauge is None else max(0.0,(panel_time-pd.Timestamp(gauge['time'])).total_seconds()/3600)
    stale=age_hours is not None and age_hours>6
    age_text=None if age_hours is None else f'{age_hours:.0f} hours' if age_hours<48 else f'{age_hours/24:.1f} days'
    stale_warning=f'Gauge reading is {age_text} old. Press Refresh gauge.' if stale else None
    high_probability=float(ge95.probability);high_n=int(ge95.n_days);high_k=int(ge95.onset_within_horizon_days)
    def ordinal(value):
        n=int(round(value));suffix='th' if 10<=n%100<=20 else {1:'st',2:'nd',3:'rd'}.get(n%10,'th')
        return f'{n}{suffix}'
    def wet(p):return 'unavailable' if p is None else 'normal' if p<75 else 'wetter than usual' if p<95 else 'very wet for this time of year'
    score=result['scores'].get('1d',{});value=score.get('score');threshold=score.get('threshold')
    if value is not None and value>=threshold:summary='Elevated flood risk'
    elif water>=95 and value is not None and value<threshold:summary='Early warning sign: very moist air'
    elif water<90 and rain7 is not None and rain14 is not None and rain7<75 and rain14<75:summary='No warning signs active'
    else:summary='Some signals above normal; no alert'
    if stale:summary+=' (gauge reading out of date)'
    lines=[f'Precursor check — {day:%Y-%m-%d}',PERCENTILE,
           f'Moisture in the air (water vapour): {ordinal(water)} percentile; band {band}.',
           f'When moisture was in this band in 2001–2019, a flood followed within 3 days {probability:.2%} of the time ({k}/{n} days), vs {baseline:.2%} on a typical monsoon day.',
           f'For comparison: when moisture was at or above the 95th percentile, a flood followed within 3 days {high_probability:.2%} of the time ({high_k}/{high_n} days).',
           f'Rain over 7 days: {ordinal(rain7)} percentile — {wet(rain7)}.' if rain7 is not None else 'Rain over 7 days: unavailable.',
           f'Rain over 14 days: {ordinal(rain14)} percentile — {wet(rain14)}.' if rain14 is not None else 'Rain over 14 days: unavailable.',
           f'River flow (modelled): {q:.1f} m³/s; {ordinal(qp)} percentile for this time of year; about {abs(gap):.0f} m³/s '+('below' if gap>=0 else 'above')+f' the flood level (flow is {100*q/THRESHOLD:.1f}% of it).' if q is not None else 'River flow (modelled): unavailable.',
           f"River level (measured at Khokana): {gauge['percent']:.0f}% of the warning level ({gauge['level']:.3f} m / {gauge['warning']:.2f} m). Observation: {gauge['time']}; this is the latest saved reading, not necessarily today’s." if gauge else 'River level (measured at Khokana): unavailable.',
           'Summary: '+summary,
           'These are experimental indicators, not an official warning. Today’s weather includes forecast hours; the gauge reading has its own timestamp.']
    if stale_warning:lines.insert(0,stale_warning)
    return dict(panel_time=panel_time.isoformat(),gauge_age_hours=age_hours,gauge_stale=stale,gauge_warning=stale_warning,high_probability=high_probability,high_n_days=high_n,high_n_flood=high_k,flow_percent_of_flood=None if q is None else 100*q/THRESHOLD,available=True,date=str(day.date()),snapshot_id=result['snapshot_id'],water_pct=water,band=band,n_days=n,n_flood=k,probability=probability,base_probability=baseline,rain7_pct=rain7,rain14_pct=rain14,river_flow=q,river_percentile=qp,distance_to_flood=gap,gauge=gauge,summary=summary,text='\n'.join(lines))


def render_precursor():
    st.subheader('Precursor check — signs to watch')
    data=panel_data()
    if not data['available']:st.info(data['text']);return
    if data['gauge_warning']:st.warning(data['gauge_warning'])
    st.caption('Panel time: '+data['panel_time'])
    for line in data['text'].splitlines():
        if line!=data['gauge_warning']:st.write(line)
    caption('Saved weather and river warning signs.','Read each sign together; no active signs does not guarantee safety.','latest GFS/GloFAS snapshot, saved DHM gauge, Stage 2 training conditional probabilities and q_north 2001–2019 ±15-day monsoon calendar window')
    st.caption('Technical names: TCWV percentile; GFS rain_7d_pct / rain_14d_pct; GloFAS q_north; DHM station 550.05. Training band rates are derived by subtracting the saved ≥90/≥95 counts; they are not live forecast probabilities.')
