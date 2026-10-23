"""Presentation text only; no model or feature calculations."""
import streamlit as st

PERCENTILE='90th percentile = higher than 90% of days at this time of year'

def caption(shows,read,source):
    st.caption(f'What it shows: {shows} How to read it: {read} Source: {source}.')

def glossary():
    with st.expander('Glossary'):
        st.write('**Moisture in the air (water vapour)** — the water held in the whole column of air above the basin; technical name: TCWV.')
        st.write('**River flow (modelled)** — the estimated volume passing a river point each second, in m³/s; technical name: GloFAS discharge. It is not a gauge measurement.')
        st.write('**River level (measured at Khokana)** — the measured height of water, in metres; technical source: DHM gauge. Height and flow are different quantities.')
        st.write('**Percentile** — '+PERCENTILE+'.')
        st.write('**Flood risk score** — a number from 0 to 1, not a probability. An alert fires when the score reaches or exceeds its fixed threshold; technical model: XGBoost.')
        st.write('**Already flooding** — no new flood-onset forecast is made for an ongoing flood day.')
        st.write('**Pattern distance** — smaller means more similar. Pattern matching failed the training discrimination test and is not used to claim resemblance.')
        st.write('**Apache Spark** — software that processes many records in parallel. A partition is a chunk of those records; parity means two implementations give the same answers.')
