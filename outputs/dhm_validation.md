# DHM historical gauge validation — public archive unavailable in tested requests

No models, labels, features or dashboard files were changed. Only two historical chart-data requests were made, each using the normal public anonymous page/form session. No login, terms acceptance, block workaround or archive sweep was attempted.

Endpoint: `POST https://dhm.gov.np/site/getRiverWatchBySeriesId_Single`; Khokana station 550.05, `seriesid=927`. The page exposes an anchor date and period selector, rather than a documented arbitrary start/end API.

| Anchor date | Mode | Requested coverage | HTTP / application status | Observations |
|---|---|---|---|---:|
| 2024-10-02 | 4: 7 Days Point | 2024-09-26–2024-10-02, containing the flood | 200 / success | 0 |
| 2024-09-28 | 2: Hourly | Flood onset/peak day | 200 / success | 0 |

The seven-day response embedded an empty observation array; the hourly response also embedded an empty array and an empty Max/Min/Average table. Thus the public form accepts the requested historical dates, but does not expose observations for these tested periods. This is not evidence that no flood or gauge threshold crossing occurred. Current gauge observations remain available through the separate live monitor.

| Event | GloFAS onset | GloFAS peak date / discharge | Observed peak | First 3.50 m crossing | First 4.00 m crossing | Earlier/same/later comparison |
|---|---|---|---|---|---|---|
| September 2024 | 2024-09-28 | 2024-09-28 / 705.32 m³/s | Unavailable | Unknown | Unknown | Cannot assess |

The full 20 September–5 October 2024 range was not downloaded after both flood-window probes returned no records. The six optional 2020–2021 events were not queried, to respect the small polite-request budget once historical availability was not established. Their ±2-day warning/danger agreement remains unknown, not negative.

## Next step

Make a formal DHM historical-data request for station 550.05 (Bagmati at Khokana): 20 September–5 October 2024 and ±2 days around 2020-07-10, 2020-07-23, 2020-09-23, 2021-07-01, 2021-07-17 and 2021-08-14. Request original timestamps/timezone, gauge-level units/datum, quality flags and warning/danger levels applicable at those times. No formal request has been submitted.

Raw page HTML, exact JSON responses and request metadata are preserved under `data/dhm_history/`; `probe_summary.json` records statuses and zero record counts. No empty response is converted into synthetic readings or a non-flood label.
