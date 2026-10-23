# Analogue discrimination: negative finding

All selection quantities use 2001–2019 monsoon days, frozen scalers and existing training-event templates. The tested window is fixed at 14 days. Pre-flood queries are onset-1 through onset-7 and exclude their own event from candidate templates. A hit requires both distance <= cutoff and matched template lead 1–3, retaining the earlier hit definition. No test, September or live rows set anything.

Ordinary days exclude onset-14 through onset+7 around every training-period onset. There are 2,505 ordinary queries: 2,262 complete windows and 243 incomplete windows. Distance quantiles use complete windows; false-match rates use all ordinary queries, with incomplete windows counted as no match. All 133 pre-flood queries are complete.

| Analogue | Group | Median | P10 | P90 |
|---|---|---:|---:|---:|
| research | ordinary | 6.311239 | 4.911395 | 8.453109 |
| research | pre_flood | 6.343480 | 4.554046 | 9.790957 |
| operational | ordinary | 4.841338 | 3.651647 | 7.530398 |
| operational | pre_flood | 4.994261 | 3.685617 | 8.038792 |

| Analogue | Old false-match rate | New cutoff | New false-match rate | Pre-flood hits | Distance-only pre-flood coverage |
|---|---:|---:|---:|---:|---:|
| research | 70.34% | 4.967996365959 | 9.98% (250/2505) | 4/133 (3.01%) | 16.54% |
| operational | 70.54% | 3.68583445160981 | 9.98% (250/2505) | 5/133 (3.76%) | 10.53% |

The cutoff is the largest representable floating-point number below the first disallowed ordinary distance, accounting for ties. At the next representable float, the ordinary false-match rate exceeds 10%. Neither hit rate reaches 25%; both resemblance displays are disabled. Even distance-only pre-flood coverage is below 25%. The matching procedure does not discriminate flood run-ups from ordinary conditions adequately at the stipulated false-match rate. Distances and nearest matches remain in SQLite and reports for audit.

Both labels now read: **Pattern matching not reliable enough to display (see README)**. Model scores, thresholds, templates and scalers remain unchanged. Earlier cutoff sections in README describe superseded historical steps.
