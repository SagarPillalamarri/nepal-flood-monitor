"""Observed gauge panel; satellite only displayed when the freshness gate passes."""
import json
import altair as alt
import pandas as pd
import streamlit as st
import config as c
from observed_poll import latest_observed,poll_observed,rebuild_observed


def observed_view(snapshot):
    return rebuild_observed(snapshot)


def render_observed():
    st.header('Observed river level — DHM Khokana gauge')
    if st.button('Refresh observed gauge',key='refresh_observed_dhm'):
        try:
            with st.spinner('Reading the public DHM gauge…'):
                poll_observed()
            st.success('Observed gauge snapshot saved.')
        except Exception as error:
            st.error(f'DHM refresh unavailable: {error}. No access workaround attempted; displaying the last successful snapshot if present.')
    snapshot=latest_observed()
    if snapshot is None:
        st.info('No observed gauge snapshot saved yet. Use Refresh observed gauge.')
    else:
        try:
            raw,hourly,summary=observed_view(snapshot)
            latest=pd.Timestamp(summary['latest_time_nepal'])
            age=(pd.Timestamp.now(tz='UTC')-pd.Timestamp(summary['latest_time_utc'])).total_seconds()/3600
            cols=st.columns(3)
            cols[0].metric('Observed water level',f"{summary['latest_water_level_m']:.3f} m")
            cols[1].metric('DHM warning level',f"{summary['warning_level_m']:.2f} m" if summary['warning_level_m'] is not None else 'Not published')
            cols[2].metric('DHM danger level',f"{summary['danger_level_m']:.2f} m" if summary['danger_level_m'] is not None else 'Not published')
            st.write(f'Latest observation: **{latest:%Y-%m-%d %H:%M} Nepal time** · age {age:.1f} hours. Fetched {summary["fetched_at"]}.')
            if age>3:
                st.warning('The latest saved gauge observation is more than three hours old. Refresh before interpreting it as current.')
            chart=hourly.reset_index()
            chart['hour']=chart.hour.dt.tz_localize(None)
            x=alt.X('hour:T',title='Nepal time',axis=alt.Axis(format='%b %d',tickCount='day'))
            y=alt.Y('water_level_m:Q',title='Gauge water level (m)',scale=alt.Scale(zero=True))
            line=alt.Chart(chart).mark_line(color='#45b89e',strokeWidth=2).encode(x=x,y=y,
                tooltip=[alt.Tooltip('hour:T',format='%Y-%m-%d %H:%M'),alt.Tooltip('water_level_m:Q',format='.3f'), 'n_observations:Q','partial_hour:N'])
            layers=[line]
            for name,color in [('warning','#e8b344'),('danger','#df675a')]:
                level=summary[f'{name}_level_m']
                if level is not None:
                    ref=alt.Chart(pd.DataFrame({'level':[level],'label':[f'DHM {name}: {level:.2f} m']}))
                    layers += [ref.mark_rule(color=color,strokeDash=[5,3]).encode(y='level:Q'),
                               ref.mark_text(align='right',baseline='bottom',dy=-4,color=color).encode(x=alt.value('width'),y='level:Q',text='label:N')]
            st.altair_chart(alt.layer(*layers).properties(height=280),width='stretch')
            st.caption('DHM station 550.05, series 927. The public seven-calendar-day window is shown as hourly means of ten-minute readings; the latest/first hours may be partial. Missing hours stay missing; no interpolation. The latest metric is the latest raw reading, not an hourly average. Gauge level in metres is not GloFAS discharge in m³/s and is not fed to the frozen models.')
            with st.expander('Observed gauge hourly data and source'):
                table=hourly.reset_index();table['hour']=table.hour.dt.strftime('%Y-%m-%d %H:%M')
                st.dataframe(table,hide_index=True,width='stretch')
                st.link_button('Public DHM Khokana page','https://dhm.gov.np/hydrology/hms-Single/50')
        except (OSError,KeyError,ValueError) as error:
            st.error(f'Cannot read saved DHM observations: {error}')
    report=c.ROOT/'outputs/live/observed_investigation/imerg_latest.json'
    if report.exists():
        info=json.loads(report.read_text())
        st.info(f"NASA IMERG is not shown as recent observed rainfall: latest image {info['latest_utc']}; freshness checked {info['checked_utc']}, latency {info['latency_hours']:.1f} hours. The required limit is under 48 hours. No separate Early/Late collection was found in the public GPM asset listing.")
