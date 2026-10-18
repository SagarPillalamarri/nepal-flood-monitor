"""Phase A Streamlit monitor; explicit button fetch, otherwise saved snapshots."""
import json
import altair as alt
import pandas as pd
import streamlit as st
from live_poll import BANNER, THRESHOLD, latest, poll, rebuild


def view_data(snapshot):
    frame,summary = rebuild(snapshot)
    today = pd.Timestamp(summary['today'])
    return frame,frame.loc[today+pd.Timedelta(days=1):],summary


def monitor_chart(frame, today, column, label, *, logarithmic=False, bars=False):
    """Presentation only: daily ticks and one shared today boundary."""
    data=frame.reset_index()
    axis=alt.Axis(format='%b %d',values=[d.isoformat() for d in frame.index],
                  labelAngle=-60,labelOverlap=False,labelFontSize=10,title='Date')
    x=alt.X('date:T',axis=axis,scale=alt.Scale(domain=[frame.index.min().isoformat(),frame.index.max().isoformat()]))
    scale=alt.Scale(type='log',zero=False) if logarithmic else alt.Scale(zero=True)
    y=alt.Y(f'{column}:Q',title=label,scale=scale)
    base=alt.Chart(data).encode(x=x,y=y,tooltip=[alt.Tooltip('date:T',format='%Y-%m-%d',title='Date'),alt.Tooltip(f'{column}:Q',title=label,format='.3f')])
    past=base.transform_filter(alt.datum.date <= today.to_pydatetime())
    # Include today's existing value as a visual anchor, not an extra input row.
    future=base.transform_filter(alt.datum.date >= today.to_pydatetime())
    if logarithmic:
        past=past.transform_filter(alt.datum[column] > 0)
        future=future.transform_filter(alt.datum[column] > 0)
    if bars:
        marks=[past.mark_bar(color='#3486d9',opacity=1),
               base.transform_filter(alt.datum.date > today.to_pydatetime()).mark_bar(color='#9dc9ef',opacity=.8)]
    else:
        marks=[past.mark_line(color='#3486d9',strokeWidth=2),
               future.mark_line(color='#9dc9ef',strokeWidth=2,strokeDash=[6,4])]
    if logarithmic:
        refs=pd.DataFrame({'reference':[THRESHOLD,3.65],
                           'name':[f'Flood threshold: {THRESHOLD:.3f} m³/s','Typical October: 3.65 m³/s']})
        marks += [alt.Chart(refs.iloc[[1]]).mark_rule(color='#a0a0a0',opacity=.45,strokeDash=[3,4]).encode(y=alt.Y('reference:Q',scale=scale)),
                  alt.Chart(refs.iloc[[0]]).mark_rule(color='#e76f51',strokeWidth=1.5).encode(y=alt.Y('reference:Q',scale=scale)),
                  alt.Chart(refs.iloc[[0]]).mark_text(align='right',baseline='bottom',dy=-4,color='#e76f51',fontSize=11)
                  .encode(x=alt.value('width'),y=alt.Y('reference:Q',scale=scale),text='name:N')]
    boundary=alt.Chart(pd.DataFrame({'date':[today]})).encode(x=x)
    marks += [boundary.mark_rule(color='#e5b94d',strokeDash=[3,3],strokeWidth=1.3),
              boundary.mark_text(text='Today',align='left',baseline='bottom',dx=4,dy=-6,color='#e5b94d').encode(y=alt.value(0))]
    return alt.layer(*marks).properties(height=290)


def render_monitor():
    st.title('Live monitor')
    from observed_monitor import render_observed
    render_observed()
    st.header('Model estimates and forecast')
    st.warning(BANNER)
    if st.button('Fetch now',key='fetch_live_inputs'):
        try:
            with st.spinner('Fetching GFS and GloFAS inputs…'):
                poll()
            st.success('New snapshot saved and logged.')
        except Exception as error:
            st.error(f'Fetch failed: {error}. Any previous successful snapshot remains displayed below.')
    snapshot = latest()
    if snapshot is None:
        st.info('No successful live snapshot yet. Use Fetch now.')
        return
    try:
        frame,forecast,summary = view_data(snapshot)
    except (OSError,ValueError,KeyError) as error:
        st.error(f'Cannot read saved snapshot: {error}')
        return
    st.caption(f"Fetch time (UTC): {summary['fetch_time']} · snapshot {summary['snapshot_id']}")
    st.caption('Past 14 days plus today and the next seven days. Past GFS values are modelled inputs, not gauge observations. The next seven days means tomorrow through today + 7.')
    fetched=pd.Timestamp(summary['fetch_time']).strftime('%Y-%m-%d %H:%M:%S UTC')
    st.write(f'Data fetched {fetched}. Up to today: latest GFS / GloFAS records (available within hours). After today: forecast.')
    if summary['exceedance_dates']:
        st.error(summary['statement'])
    elif forecast.discharge_m3_s.isna().any():
        st.warning(summary['statement'])
    else:
        st.success(summary['statement'])
    st.subheader('GloFAS discharge forecast')
    st.caption('Typical mid-October flow at this point: 3.65 m3/s (2001–2025)')
    metrics=st.columns(2)
    peak=forecast.discharge_m3_s.max()
    metrics[0].metric('Maximum next-seven-day discharge',f'{peak:.2f} m³/s' if pd.notna(peak) else 'Unavailable')
    metrics[1].metric('Flood threshold',f'{THRESHOLD:.3f} m³/s')
    discharge = forecast[['discharge_m3_s']].copy()
    discharge['flood_threshold_m3_s'] = THRESHOLD
    today=pd.Timestamp(summary['today'])
    st.altair_chart(monitor_chart(frame,today,'discharge_m3_s','Discharge (m³/s, log scale)',logarithmic=True),width='stretch')
    st.caption('Solid blue: past / today. Light dashed blue: forecast, joined at today. Yellow vertical line: Today. Log scale; non-positive/missing flows cannot be plotted. Faint grey line: typical mid-October flow (3.65 m³/s).')
    st.caption('GloFAS v4 seamless, at the training-selected q_north point. Daily discharge dates use UTC, as in the historical inputs; weather days use Asia/Kathmandu. Threshold exceedance is a GloFAS-based signal, not an XGBoost risk score.')
    st.dataframe(discharge.reset_index(),hide_index=True,width='stretch',column_config={'date':st.column_config.DateColumn('Date',format='YYYY-MM-DD')})
    st.subheader('Basin rainfall · raw GFS forecast')
    st.altair_chart(monitor_chart(frame,today,'precip_mm','Rainfall (mm/day)',bars=True),width='stretch')
    st.subheader('Total column water vapour · raw GFS forecast')
    st.altair_chart(monitor_chart(frame,today,'tcwv_mm','TCWV (mm)'),width='stretch')
    st.caption('gfs_seamless; fixed nine-point basin mean, requiring 24 valid hours at every point. Rainfall uses hours ending 01:00–24:00. These are raw GFS values, without ERA5 scale matching or percentile conversion.')
    st.dataframe(frame.reset_index(),hide_index=True,width='stretch',column_config={'date':st.column_config.DateColumn('Date',format='YYYY-MM-DD')})
    with st.expander('Snapshot provenance'):
        st.json(json.loads((snapshot/'manifest.json').read_text()))
