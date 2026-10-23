"""Streamlit historical replay for the Bagmati flood precursor project."""
from datetime import date, timedelta
import pandas as pd
import streamlit as st
from floodwatch.core import config as c
from floodwatch.dashboard.stage5_replay import prepare_replay

st.set_page_config(page_title='Bagmati flood replay', page_icon='🌧️', layout='wide')
replay_tab, pipeline_tab, monitor_tab = st.tabs(['September 2024 replay', 'Big data pipeline', 'Live monitor'])
with replay_tab:
    st.title('Bagmati Basin · September 2024 replay')
    st.info('What this page shows: Step through the days before the September 2024 Kathmandu flood and see which warning signs appeared, and when. The scores and weather values are saved research results, not a forecast for today.')
    from floodwatch.dashboard.plain_language import caption, glossary, PERCENTILE
    st.caption(PERCENTILE)
    st.caption('Historical daily replay using saved validation-selected models and physical-variable analogues. XGBoost scores are uncalibrated and are not probabilities.')

    from floodwatch.core.analogue_labels import annotate_replay
    replay = annotate_replay(prepare_replay())
    minimum, maximum = replay.index.min().date(), replay.index.max().date()
    selected_day = st.slider('Step through the replay', minimum, maximum,
                             value=date(2024, 9, 20), step=timedelta(days=1), format='YYYY-MM-DD')
    day = pd.Timestamp(selected_day)
    row = replay.loc[day]

    if row['score_status'] == 'Eligible separate-case score':
        st.success(f'{day:%A, %d %B %Y} · score available')
    else:
        st.info(f'{day:%A, %d %B %Y} · '+("Already flooding, so no forecast is made for this day" if row.flood == 1 else "No forecast available for this day"))

    first = st.columns(4)
    first[0].metric('Basin rainfall', f'{row.precip_mm:.1f} mm' if pd.notna(row.precip_mm) else '—')
    first[1].metric('Moisture in the air (water vapour)', f'{row.tcwv_pct:.1f}' if pd.notna(row.tcwv_pct) else '—')
    first[2].metric('Topsoil moisture', f'{row.sm_0_7:.3f} m³/m³' if pd.notna(row.sm_0_7) else '—')
    first[3].metric('River flow (modelled)', f'{row.discharge_m3_s:.1f} m³/s' if pd.notna(row.discharge_m3_s) else '—')

    descriptions=[('Daily basin rain.','Higher means more rain.','ERA5, mm/day'),('Moisture rank for this time of year.','Higher ranks mean unusually moist air.','ERA5 TCWV seasonal percentile'),('Water in the upper soil layer.','Higher means wetter soil.','ERA5 sm_0_7, m³/m³'),('Estimated river flow.','Compare with the model flood level of 445.191 m³/s.','GloFAS q_north, m³/s')]
    for col,args in zip(first,descriptions):
        with col:caption(*args)
    second = st.columns(4)
    for index, horizon in enumerate(c.TARGET_HORIZONS):
        score = row[f'risk_{horizon}d']
        threshold = row[f'threshold_{horizon}d']
        if pd.isna(score):
            display = 'Already flooding, so no forecast is made for this day' if row.flood == 1 else 'No forecast available'
            delta = f'threshold {threshold:.2f}; {row.score_status}'
        else:
            display = f'{score:.4f}'
            delta = f'threshold {threshold:.2f} · ' + ('alert' if score >= threshold else 'below threshold')
        second[index].metric(f'{horizon}-day flood risk score', display, delta)
        with second[index]:caption('Flood risk score (0–1, not a probability).','An alert fires at or above the fixed threshold.','saved research XGBoost model')

    event_day = int(row.flood) == 1
    st.write(f'Flood label: **{"active" if event_day else "not active"}**')
    st.write(row.analogue_label)
    st.caption(f"Analogue distance: {row.analogue_distance:.3f} · close-match cutoff: {row.analogue_close_match_cutoff:.3f}" if pd.notna(row.analogue_distance) else f"No complete window · close-match cutoff: {row.analogue_close_match_cutoff:.3f}")
    if row.analogue_nearest_note:
        st.caption(row.analogue_nearest_note)

    st.subheader('September timeline')
    chart = replay[['precip_mm','tcwv_pct','risk_1d','threshold_1d','risk_3d','threshold_3d']].copy()
    chart.index.name = 'date'
    st.line_chart(chart.rename(columns={'tcwv_pct':'Moisture rank (TCWV percentile)','precip_mm':'Rain (mm/day)','risk_1d':'1-day flood risk score','risk_3d':'3-day flood risk score','threshold_1d':'1-day threshold','threshold_3d':'3-day threshold'}), height=330)
    caption('Rain, moisture rank and saved flood risk scores over September.','The lines have different units; compare each score with its own threshold, not with rainfall.','ERA5 and frozen research XGBoost outputs')
    physical = replay[['sm_0_7','discharge_m3_s']].copy()
    physical.index.name = 'date'
    st.line_chart(physical.rename(columns={'sm_0_7':'Topsoil moisture (m³/m³)','discharge_m3_s':'River flow, modelled (m³/s)'}), height=260)
    caption('Soil wetness and estimated river flow.','The series have different units; rising flow means more water passing the river point.','ERA5 sm_0_7 and GloFAS q_north')

    with st.expander('Show replay data'):
        st.dataframe(replay.reset_index(), width='stretch', hide_index=True)

    glossary()

with pipeline_tab:
    from floodwatch.dashboard.spark_dashboard import render_pipeline
    render_pipeline()

with monitor_tab:
    from floodwatch.dashboard.live_monitor import render_monitor
    render_monitor()
