"""Streamlit historical replay for the Bagmati flood precursor project."""
from datetime import date, timedelta
import pandas as pd
import streamlit as st
import config as c
from stage5_replay import prepare_replay

st.set_page_config(page_title='Bagmati flood replay', page_icon='🌧️', layout='wide')
replay_tab, pipeline_tab = st.tabs(['September 2024 replay', 'Big data pipeline'])
with replay_tab:
    st.title('Bagmati Basin · September 2024 replay')
    st.caption('Historical daily replay using saved validation-selected models and physical-variable analogues. XGBoost scores are uncalibrated and are not probabilities.')

    replay = prepare_replay()
    minimum, maximum = replay.index.min().date(), replay.index.max().date()
    selected_day = st.slider('Step through the replay', minimum, maximum,
                             value=date(2024, 9, 20), step=timedelta(days=1), format='YYYY-MM-DD')
    day = pd.Timestamp(selected_day)
    row = replay.loc[day]

    if row['score_status'] == 'Eligible separate-case score':
        st.success(f'{day:%A, %d %B %Y} · score available')
    else:
        st.info(f'{day:%A, %d %B %Y} · {row["score_status"]}')

    first = st.columns(4)
    first[0].metric('Basin rainfall', f'{row.precip_mm:.1f} mm' if pd.notna(row.precip_mm) else '—')
    first[1].metric('TCWV seasonal percentile', f'{row.tcwv_pct:.1f}' if pd.notna(row.tcwv_pct) else '—')
    first[2].metric('Topsoil moisture', f'{row.sm_0_7:.3f} m³/m³' if pd.notna(row.sm_0_7) else '—')
    first[3].metric('GloFAS discharge', f'{row.discharge_m3_s:.1f} m³/s' if pd.notna(row.discharge_m3_s) else '—')

    second = st.columns(4)
    for index, horizon in enumerate(c.TARGET_HORIZONS):
        score = row[f'risk_{horizon}d']
        threshold = row[f'threshold_{horizon}d']
        if pd.isna(score):
            display = 'Not scored'
            delta = f'threshold {threshold:.2f}; {row.score_status}'
        else:
            display = f'{score:.4f}'
            delta = f'threshold {threshold:.2f} · ' + ('alert' if score >= threshold else 'below threshold')
        second[index].metric(f'{horizon}-day XGBoost score', display, delta)

    event_day = int(row.flood) == 1
    st.write(f'Flood label: **{"active" if event_day else "not active"}**')
    if pd.notna(row.analogue_event_onset):
        st.write(f'Closest past precursor: **event {int(row.analogue_closest_event_id)}**, onset **{row.analogue_event_onset:%d %b %Y}**, matched window ends **{int(row.analogue_days_before_flood)} days before onset** (distance {row.analogue_euclidean_distance:.2f}; {int(row.analogue_window_days)}-day window).')
    else:
        st.write('No complete historical analogue window is available for this day.')

    st.subheader('September timeline')
    chart = replay[['precip_mm','tcwv_pct','risk_1d','threshold_1d','risk_3d','threshold_3d']].copy()
    chart.index.name = 'date'
    st.line_chart(chart, height=330)
    physical = replay[['sm_0_7','discharge_m3_s','analogue_days_before_flood']].copy()
    physical.index.name = 'date'
    st.line_chart(physical, height=260)

    with st.expander('Show replay data'):
        st.dataframe(replay.reset_index(), width='stretch', hide_index=True)

with pipeline_tab:
    from spark_dashboard import render_pipeline
    render_pipeline()
