# Bagmati monsoon flood precursors

Identifying Monsoon Flood Precursors in Nepal's Bagmati River Basin Using Satellite and Weather Data. Study area: upstream of DHM station 550.05, Bagmati at Khokana. Daily period 2001–2025, modelling June–October. Train 2001–2019, test 2020–2025. September 2024 is held out of training. No random splits, source substitutions, invented data or interpolation.

## Run

Use the project's Python 3.10+ environment from the root. Quote paths containing spaces:

```sh
cd '/Users/sagarp/Documents/BDA_SEM 7'
python -m pip install -r 'requirements.txt'
python 'stage1b_fetch.py'
python 'stage1b_check.py'
python 'stage2_analysis.py'
python 'stage2_check.py'
```

Stop after each stage for user verification. Stage 2 was explicitly authorized following a Stage 1b check with no FAILs; Stage 2 was approved, including the additions below; Stage 3 has now been built and checked. All settings, paths, dates and decision thresholds are in `config.py`.

## Stage 1

Existing `stage1_fetch.py` downloads rainfall, soil moisture and GloFAS candidates. It has not been changed. Rainfall and soil moisture retain all nine points. `stage1_check.py` now reports the configured `GLOFAS_CHOSEN` (`q_north`) rather than choosing by percentile rank, retaining its low-mean-flow warning. Verified: 8 PASS, 0 WARN, 0 FAIL; September event peak 705.3 m³/s.

## Stage 1b: TCWV and ONI

`stage1b_fetch.py` reuses existing caches and adds `tcwv_mm` and `oni` to the unchanged Stage 1 columns in `data/processed/stage1b_daily.csv` (9,131 rows). TCWV uses **all nine points throughout**; the former point exclusion has been removed. ERA5 is explicitly selected. Each point needs all 24 hourly values for a valid daily mean, and the basin mean needs all nine points. Any incomplete day stays NaN rather than averaging incomplete hours or fewer points. There are no repair requests for the confirmed gap. Missing 2025 data was fetched once; earlier years were reused.

New hourly responses are checked for expected point count, timestamps/row count, non-null values, units and physical sign before an atomic cache write. Only NaNs inside the explicitly approved ERA5 source gap are accepted; unexpected missing values, infinities and other source failures still stop processing.

NOAA ONI is cached in `data/raw/stage1b/oni.ascii.txt`; `data/processed/oni_daily.csv` contains 9,131 validated daily assignments. The index uses the three-month season ending two months before the scoring month (September -> MJJ; January -> previous-year SON). NOAA labels seasons by their centre month's year. Historical values can be revised slightly; the lag rule does not reconstruct original publication vintages. Stage 3 must ablate optional ONI and drop it if test results do not improve.

`features.py` provides the shared TCWV percentile definition, to be used for training and later scoring: 2001–2019 ERA5 monsoon reference, ±15-calendar-day seasonal window, midranks for ties and month/day alignment across leap years. Missing TCWV remains a missing percentile. No 2020–2025 ERA5 values enter this reference. Raw TCWV is stored for analysis, not a model feature.

`stage1b_check.py` checks unchanged Stage 1 data, daily coverage, equality to the all-nine-point cached aggregation including missing values, ONI timing, training-only climatology, September elevated TCWV and prefix invariance. It reports GFS deferral and the approved source gap as WARN. Results are saved to `data/processed/stage1b_checks.csv`; any FAIL exits 1.

Verified Stage 1b result: **9 PASS, 2 WARN, 0 FAIL**. September 26–28 TCWV is 44.60, 45.23 and 44.64 mm, with seasonal percentiles 98.30, 98.98 and 98.98.

### Known data limitations

- Open-Meteo's ERA5 TCWV archive has a source gap confirmed by a fresh direct request and independently confirmed by the user. All nine points have 4,368 null hours, from January 1, 2024 at 05:00 through July 1 at 04:00 in the API's returned local hourly timestamps. It is not a truncated local cache. Under the complete-day rule, **183 daily TCWV means remain NaN**, January 1–July 1 inclusive. **31 are monsoon days: all 30 June days plus partially missing July 1.** No filling, interpolation, model substitution or point exclusion is used. Stage 2 skips these NaNs and reports sample counts; Stage 3 XGBoost must use its native NaN handling, not imputation. The rainfall-only baseline remains available on these days.
- ERA5 is delayed reanalysis. An end-of-day historical replay does not by itself establish operational forecasting skill. The demo scope has no live pollers.
- Rainfall is ERA5-based and runs wet locally. Stage 3 must use percentile/anomaly features and a training-tuned percentile baseline, not fixed millimetre thresholds.
- GloFAS is a modelled flood proxy, not observed DHM danger-level exceedance. Label metadata records its source. Rainfall causation is the study scope; a discharge threshold alone cannot independently verify each event's cause.

### Deferred or excluded sources

`stage1b_gfs_calibration.py` is separate and **has not run**. It is deferred until after Monday. It downloads GFS TCWV from March 2021 onward, ranks it against its own climatology, and compares overlapping monsoon percentiles with ERA5 (WARN below correlation 0.7). This is a descriptive calibration, not an out-of-sample model metric; any historical operational replay must freeze reference data before scoring. Forecast archive: skipped. 850 hPa winds: excluded, future work only.

## Stage 2: labels and precursor analysis

`stage2_analysis.py` writes daily **same-day** flood labels and descriptive event composites. It does not create future targets or predictive features; those belong to Stage 3's shared `features.py`.

If `data/raw/dhm_khokana.csv` exists, configured columns `date`, `water_level`, `danger_level` must have complete daily coverage. Flood is water level >= danger level. An existing but incomplete DHM file fails rather than silently switching sources. Otherwise flood is `q_north` strictly above its 99th-percentile **training-period monsoon** discharge. The threshold is deliberately frozen using 2001–2019 to comply with the no-test-leakage rule. Labels can exist outside monsoon, but those days/events are flagged and excluded from future monsoon modelling.

Consecutive flood runs are merged when fewer than seven fully non-flood days separate them. Each event records first/last flood day, peak date/value, flood-day count and training eligibility. Alignment day 0 is the first flood day, not the peak. Annual counts include zero-event years.

Every input variable has an event window from day -14 to +3. Normal conditions are the mean and 75th percentile for the **same calendar day** across training monsoons only. Windows crossing season boundaries have no monsoon baseline for their outside-season days; these baselines remain NaN. Paired composites use only event observations with a valid baseline, and show observed, missing and paired counts. TCWV NaNs are skipped, never interpreted as zero. Plots compare event means, calendar-day normal means and normal 75th percentiles, with paired sample counts underneath.

The precursor table reports the earliest pre-event lag at which the paired composite exceeds the calendar-day 75th percentile, requiring at least three events. A crossing at -14 is marked left-censored: moisture or another variable was already elevated when the window began. No crossing stays blank/NaN. These are descriptive average patterns, not validated warning lead times. All-event composites include test/case events for exploration and must not be used to select model features or tune parameters on the test set; training-event flags are supplied for later training-only analyses.

Outputs:

- `data/processed/stage2_daily.csv`: Stage 1b inputs, same-day `flood`, and nullable flood-day `event_id`.
- `outputs/stage2/events.csv`: event catalogue, including training eligibility.
- `outputs/stage2/events_per_year.csv`: annual counts.
- `outputs/stage2/calendar_climatology.csv`: training-only normals, upper quartiles and sample counts.
- `outputs/stage2/event_windows.csv`: long-form windows with NaNs preserved.
- `outputs/stage2/composites.csv`: paired means/anomalies and support counts per variable and lag.
- `outputs/stage2/precursor_timing.csv`: first above-normal pre-event lag and left-censor flag.
- `outputs/stage2/metadata.json`: label source, threshold and reference dates.
- `outputs/stage2/composite_*.png` and `case_study_labels.png`: plots.
- `outputs/stage2/checks.csv`: check results.

`stage2_check.py` verifies source preservation, label rule/threshold, event count (WARN below 15), seven-day separation, membership and peaks, outside-season flags, annual counts, September 2024 detection peaking September 28 ±1 day, and its exclusion from training. It verifies that truncating input at 2019 leaves fitted labels/threshold/climatology unchanged, that window values match their source dates including NaNs, and that composite means exclude NaNs with correct counts. It also checks precursor crossings and plot files. It ends with a PASS/WARN/FAIL summary and exits 1 on any FAIL.

## Remaining demo scope

Stage 3 is complete: XGBoost + rainfall-percentile baseline, first the 1-day onset target and then the onset-within-next-3-days target after primary checks passed. The 3-day target includes t+1, t+2 and t+3. Preserve time split, no leakage, September holdout, class imbalance handling, metrics, importance plots and ONI ablation. No logistic regression. XGBoost handles missing TCWV natively. A later validation-only ablation selects whether to retain `monsoon_day`; its results are recorded below.

Stage 4 is now complete: it uses Euclidean standardized windows over physical precursor features only, no DTW. Leave-one-event-out selection chooses between 14-day and 7-day windows on the 2001–2019 events. September 2024 uses only pre-2024 event templates.

Stage 5 is complete: a Streamlit dashboard replays September 2024 day by day with rainfall, TCWV percentile, soil moisture, discharge, selected XGBoost scores and the closest historical analogue. No Kafka, live pollers or forecast archive.

Sources: [Historical Weather](https://open-meteo.com/en/docs/historical-weather-api), [Historical Forecast](https://open-meteo.com/en/docs/historical-forecast-api), [GFS](https://open-meteo.com/en/docs/gfs-api), [NOAA ONI](https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt).

## Verified Stage 2 result

**14 PASS, 0 WARN, 0 FAIL.** The training-monsoon GloFAS 99th-percentile threshold is 445.191 m³/s at `q_north`. There are 26 events: 19 in the training period and 7 in the test period, with no flagged flood days outside June–October. The September 2024 event begins and peaks September 28. Its event is not training eligible.

In the descriptive all-event composite, TCWV first exceeds the same-calendar-day 75th percentile at lag -2 (two days before first flood day); rainfall does so at lag -8. Discharge is already elevated at the -14-day window boundary, so its precursor timing is left-censored. These are exploratory averages, not demonstrated forecast lead times. ONI and the three soil-moisture composites have no pre-event upper-quartile crossing under this definition. The accepted TCWV source gap affects no detected event windows in this event catalogue. Thirteen plots were saved and the TCWV plot was visually inspected. Stage 2 was subsequently approved. Stage 3 results are below; work now stops after Stage 3.


## Stage 2 additions: conditional onset probabilities

`outputs/stage2/conditional_probabilities.csv` contains empirical onset probabilities for each requested condition and horizons 1 and 3, using only training monsoon days and training-fitted percentile references. It includes `n_days`, positive day counts and probabilities. `all_monsoon_days` is the requested all-days population; `not_currently_flooding` supplies the analogous at-risk population with flood days excluded. Each population has five conditions (base rate; TCWV percentile >=90; >=95; >=95 and topsoil seasonal percentile >75; three-day rainfall percentile >=95). These are day-level conditional frequencies, not counts of independent events. They are training-period descriptions, not test performance estimates. Zero counts produce NaN, not an invented probability.

`stage2_analysis.py` also prints all 26 event onset and peak dates. The enhanced Stage 2 checker verifies denominators and future onset counts directly against the event catalogue: **15 PASS, 0 WARN, 0 FAIL**. Stage 3 and Stage 4 results are documented below.

## Stage 3: onset models

```sh
python 'stage3_models.py' --horizon 1
python 'stage3_check.py' --horizon 1
python 'stage3_models.py' --horizon 3
python 'stage3_check.py'
python 'stage3_diagnostics.py'
python 'stage3_calendar_ablation.py'
python 'stage3_calendar_check.py'
python 'stage4_analogues.py'
python 'stage4_check.py'
python 'stage5_replay.py'
python 'stage5_check.py'
streamlit run 'app.py'
```

The stretch run requires a successful primary check for the same input file. The final check defaults to both horizons; `--horizon 1` or `--horizon 3` checks just that target. Models and all audit files are under `outputs/stage3/`; features are in `data/processed/stage3_features.csv`. scikit-learn and XGBoost are added to requirements. No logistic regression is used.

### Targets and eligible days

An onset is the first day of one of the Stage 2 events. The 1-day target is an onset tomorrow. The 3-day target is any onset during t+1 through t+3, explicitly excluding t. Every training, validation, main-test and separate-case row must have `flood[t] == 0`; ongoing floods never become persistence examples. Future target windows must be fully observed and cannot cross a split's end. All rolling calculations operate on the complete daily calendar before monsoon filtering, avoiding windows that jump across winters.

Training is 2001–2019. Main testing is 2020–2025 with **the whole September 2024 month reported separately**, plus purge of main-test rows whose target windows intersect that month. Consequently the main test contains six events, and the case report contains one. Training has 2,877 eligible days for both targets; main-test counts are 880 (1-day) and 878 (3-day), with 27 eligible case dates for each. Case plots display September 20–October 2 for context, with scores only on eligible separate-case dates; gaps are excluded dates, not zero risk.

### Shared features and references

`features.py` alone implements the predictive transformations: trailing 1/3/7/14-day rainfall percentiles, maximum-point rainfall percentile, raw soil moisture at the existing three depths and each 7-day change, topsoil percentile, discharge and 1/3-day rise rates (change divided by days), day of monsoon, TCWV percentile and optional lagged ONI. Features at t use data through the end of t only. The reference is passed explicitly to prevent implicit refitting during replay. References and model column lists are saved alongside the models.

Final seasonal percentiles use **only 2001–2019 monsoon observations**, with a ±15-calendar-day reference window. Inside each validation fold, the reference is refitted on that fold's earlier training prefix. Ranking rounds values to eight decimal places solely to identify ties consistently despite CSV floating-point round-trip noise; raw observations and missing values are unchanged. TCWV's 31 monsoon NaNs stay NaN through percentile conversion and XGBoost scoring; no imputer is used. The baseline depends only on available rainfall.

### Training and threshold selection

Fixed, shallow XGBoost hyperparameters are in config. Class imbalance uses the negative/positive training count ratio as `scale_pos_weight`. There are no random splits or test-based early stopping. Three expanding chronological validation folds are used: fit through 2009, validate 2010–2012; fit through 2012, validate 2013–2015; fit through 2015, validate 2016–2019. Target windows are purged at each fitting/validation boundary. Alert thresholds are selected on pooled out-of-fold predictions by F1, with deterministic precision and conservative-threshold tie breaks. Full 2001–2019 refitting follows threshold selection.

The baseline is `rain_3d_pct > X`, with X chosen only from training-period validation. The primary XGBoost variant is predeclared `xgb_tcwv`, excluding ONI. `xgb_core` removes TCWV for the atmospheric ablation. `xgb_oni_ablation` adds ONI. All use identical eligible dates. ONI is saved as a diagnostic optional model; it is not promoted by looking at test results. This keeps test ablation assessment separate from primary model selection. Its test F1 improves here, so the optional model is retained for review rather than dropped. Both ONI variants still miss September 2024. No successful case result is fabricated or used for threshold retuning.

XGBoost outputs are **uncalibrated, class-weighted scores**, not calibrated flood probabilities. Stored JSON models use native NaN handling. Definitions: [XGBoost Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html).

### Metrics and artifacts

`metrics.csv` and `metrics_1d.csv`/`metrics_3d.csv` report recall, precision, F1, false alarm rate **FP/(FP+TN)**, false alarm ratio **FP/(TP+FP)**, confusion counts, event recall and mean lead. Lead is the earliest alert in the preceding target window, averaged only over detected events; a missed event has NaN lead. Three-day day-level recall differs from event recall because one onset can generate three positive forecast dates. No accuracy headline metric is reported.

Artifacts include `samples_*d.csv` (eligibility audit), `predictions_*d.csv`, `validation_predictions_*d.csv`, `threshold_tuning_*d.csv`, `event_leads_*d.csv`, `metadata_*d.json`, `feature_reference.csv`, the three XGBoost JSON variants per horizon, importance CSV/PNG, separate case plots, `case_study_predictions.csv`, `oni_ablation.csv`, and `checks_*d.csv`. `checks.csv` combines both horizons. `primary_ready.json` gates the stretch run for the same input hash.

### Verified results

Each horizon: **11 PASS, 2 WARN, 0 FAIL**. The warnings are the accepted TCWV gap and the separate September XGBoost miss. Leakage checks include recomputing historical prefixes with a frozen reference, independently checking trailing rainfall and backward discharge differences, and direct future-onset target verification. The checker explicitly confirms zero ongoing-flood rows in training/evaluation, time ordering and target-window purges, chronological training-only tuning, model reload reproducibility, common dates, all metric counts and lead-time ranges. The September risk/discharge plot was visually inspected.

| Horizon | Model | Recall | Precision | False alarm rate | F1 | Test events detected | Mean lead (detected events) |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 day | Baseline | 16.7% | 5.0% | 2.17% | 0.077 | 1/6 | 1.0 day |
| 1 day | XGBoost + TCWV (primary) | 50.0% | 17.6% | 1.60% | 0.261 | 3/6 | 1.0 day |
| 1 day | XGBoost + TCWV + ONI (ablation) | 50.0% | 30.0% | 0.80% | 0.375 | 3/6 | 1.0 day |
| 3 days | Baseline | 61.1% | 5.3% | 22.79% | 0.098 | 5/6 | 2.8 days |
| 3 days | XGBoost + TCWV (primary) | 66.7% | 6.1% | 21.51% | 0.112 | 5/6 | 2.8 days |
| 3 days | XGBoost + TCWV + ONI (ablation) | 61.1% | 9.5% | 12.21% | 0.164 | 5/6 | 2.8 days |

Separate September 2024: primary XGBoost misses the onset at both horizons. The baseline detects it one day ahead for the primary target, and two days ahead for the three-day target. The low precision and six-event main test limit the evidence of generalization. These results are historical proxy-label results, not validated operational flood warnings.

### Calendar-prior ablation

The September 2024 diagnosis prompted a comparison of the current primary feature set with the same set minus `monsoon_day`. For each horizon, both variants used the existing three chronological folds and pooled out-of-fold threshold tuning, using 2001–2019 only. The chosen feature set is the one with higher mean fold F1; an exact tie drops `monsoon_day`. September 2024 was not used for selection or threshold tuning. `stage3_calendar_check.py` verifies the split, selected variant, threshold, model outputs and metrics.

The 1-day variant without `monsoon_day` was selected (mean validation F1 0.099 vs 0.095; threshold 0.70). The 3-day variant with `monsoon_day` was selected (0.119 vs 0.094; threshold 0.40). Test results for both variants are shown so the non-selected results remain visible:

| Horizon | Variant | Selection | Threshold | Test recall | Test precision | Test false alarm rate | Test F1 | Test events | Mean lead |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 day | With `monsoon_day` | Not selected | 0.90 | 50.0% | 17.6% | 1.60% | 0.261 | 3/6 | 1.00 day |
| 1 day | Without `monsoon_day` | Selected | 0.70 | 50.0% | 8.3% | 3.78% | 0.143 | 3/6 | 1.00 day |
| 3 days | With `monsoon_day` | Selected | 0.40 | 66.7% | 6.1% | 21.51% | 0.112 | 5/6 | 2.80 days |
| 3 days | Without `monsoon_day` | Not selected | 0.50 | 50.0% | 6.1% | 16.05% | 0.109 | 4/6 | 2.75 days |

On the separate September 2024 case, both 1-day variants miss. The selected 3-day variant with the calendar feature also misses; the non-selected 3-day variant without it detects the event at 1 day with 2 false alarms in the 27 eligible case dates. This case result is reported transparently and did not influence selection. All saved variants, fold metrics, threshold tuning and checks are in `outputs/stage3/calendar_ablation/`.

### Stage 3 diagnostics

`stage3_diagnostics.py` reads the saved models, thresholds and predictions. It does not train or tune. It reports September 20–27 risk scores and score ranks among eligible 2020–2025 held-out dates, including the separately reported September case rows; display percentiles for all primary model inputs; XGBoost log-odds contributions for September 25–26; the September onset and label threshold; and detection of the six main-test events. `outputs/stage3/diagnostics/protected_hashes.json` records unchanged model and threshold hashes. September diagnostics are descriptive and must not be used to retune the models.

The subsequent `monsoon_day` ablation was prompted by the September diagnosis. Selection and threshold tuning used validation folds only; the case was not used. Its separate models, test/case comparison and 6-pass check are under `outputs/stage3/calendar_ablation/`.

### Stage 4: nearest historical precursor

`stage4_analogues.py` compares only TCWV percentile, 1/3/7-day rainfall percentiles, topsoil moisture and 1-day discharge rise. It never uses `monsoon_day`. Leave-one-event-out validation on the 19 training events asks, at onset-1 and onset-2, whether the nearest template from another event ends within 1–3 days of that event's onset. Scalers are refitted with the held-out event's query and flood dates excluded. The 14-day window has the higher hit rate (12/38, 31.6%) than 7 days (9/38, 23.7%), so 14 days is selected. Daily candidates must precede the query; September 2024 uses event templates dated through 2023. Missing windows are left unmatched; there is no filling or DTW. Daily rows report closest event, onset, peak, matched lead, distance and candidate count. `stage4_check.py` verifies the LOEO selection, scaling, September restriction and representative distances.

Verified Stage 4: **6 PASS, 0 WARN, 0 FAIL**. There are 360 usable templates and 3,825 daily monsoon rows; 3,398 rows have a nearest-event match. With physical features, September 20–25 maps to the July 1, 2011 event at leads 6, 5, 4, 4, 3 and 2 days; September 26–27 maps to September 23, 2020 at leads 9 and 8 days; September 28 maps to July 10, 2020 at lead 1 day. See `outputs/stage4/daily_analogues.csv`, `loo_evaluation.csv` and `checks.csv`.

### Stage 5: September replay dashboard

Run `python 'stage5_replay.py'`, `python 'stage5_check.py'`, then `streamlit run 'app.py'`. The slider steps through September 1–30 and shows rainfall, TCWV percentile, soil moisture, discharge, the selected 1-day and 3-day model scores and thresholds, and the closest past analogue. Scores remain blank on days excluded from case evaluation under the ongoing-flood and target-window rules. The app reads saved predictions and does not retrain models. `outputs/stage5/september_2024_replay.csv` is the dashboard data; `september_20_28_diagnostics.csv` contains the requested comparison table; `outputs/stage5/checks.csv` records validation. Streamlit is listed in `requirements.txt`. Stage 5 check: **3 PASS, 0 WARN, 0 FAIL**. Streamlit AppTest rendered the date slider and six metric widgets with no app exceptions.

All five stages are now complete within the approved demo scope. The calendar-feature and analogue-window choices use training-period validation only; September 2024 remains a separate diagnostic replay.

## Frozen-model satellite rainfall validation and demo pack

The models, thresholds, selected variants, predictive features and dashboard scoring are frozen. `stage_sat_fetch.py` accesses only `NASA/GPM_L3/IMERG_V07`, band `precipitation` (mm/hour), through read-only Earth Engine project `bdasem-7`. This is a validation dataset, never a model input. The optional client dependency is listed in `requirements_satellite.txt`; the dashboard does not require Earth Engine.

The spatial mean is pixel-area-weighted over the nine-point bounding rectangle [85.28, 27.62, 85.44, 27.78] at the native 0.1-degree grid. This is a study-area box, not a delineated catchment. Each Asia/Kathmandu day spans 49 overlapping half-hour intervals: the first and last contribute 0.25 hours and the middle 47 contribute 0.5 hours each, totaling 24 hours. Every month is checked for its exact half-hour timestamp sequence and complete pixel coverage before atomic caching. Missing intervals are not filled. Source provisional/permanent status is retained. The official catalogue notes that V07 has no permanent products beyond September 2025: [IMERG V07 documentation](https://developers.google.com/earth-engine/datasets/catalog/NASA_GPM_L3_IMERG_V07).

`stage_sat_check.py` reports the September 20–30 daily comparison, monsoon means, raw and source-specific seasonal-percentile correlations, and three-day rainfall strictly before all 26 event onsets. Diagnostic percentiles use each source's own 2001–2019 monsoon reference and the existing ±15-calendar-day ranking convention. Complete three-day reference windows are matched between sources; the first two June days are omitted from both reference pools because the satellite download covers monsoon months only. No test year enters these references. Satellite checks and plots are under `outputs/satellite/`; monthly caches are under `data/raw/satellite/`; the final series is `data/processed/imerg_daily.csv`.

`build_demo_pack.py` reads the frozen artifacts and creates `outputs/demo/results_summary.md`, selected-model plots and launch commands. It only reloads saved models for feature importance; it never fits them. `outputs/satellite/frozen_artifact_hashes.json` verifies that the 80 recorded model/scoring artifacts remain unchanged. GloFAS labels share ERA5 forcing with the meteorological inputs, the main test contains only six events, and the approximate 1.5-day basin response is a working estimate rather than a calibrated result. The pack records these limitations alongside the known TCWV gap, wet rainfall bias and other caveats.

Satellite result: **5 PASS, 2 WARN, 0 FAIL** across all 3,825 monsoon dates. Mean IMERG rainfall is 11.520 mm/day versus ERA5 14.643 mm/day. Daily Pearson correlation is 0.4941 for raw rainfall and 0.4422 for source-specific seasonal percentiles. Both sources rank the three-day rainfall before the September 28, 2024 onset at the 100th percentile. Warnings flag 31 days with non-permanent source intervals and the different spatial supports of IMERG and ERA5. These are validation findings only; the frozen models and dashboard scoring are unchanged.

```sh
venv/bin/python stage_sat_fetch.py
venv/bin/python stage_sat_check.py
venv/bin/python build_demo_pack.py
# Demo only (no network fetch or model fitting required):
venv/bin/streamlit run app.py
```

## Earlier live scoring work — deferred after Phase A delivery

The earlier live scoring work was paused while Spark processing took priority. Phase A live input monitoring has now been delivered; the earlier scoring/calibration prototype remains deferred. No model, threshold, selected variant, feature definition or dashboard scoring was changed. The ECMWF probe returned all five required variables (24/24 valid hours each) for 27 September 2024 in Historical Weather (`ecmwf_ifs`) and for the current Forecast API (`ecmwf_ifs025`, also `ecmwf_ifs`). The advertised archive begins in 2017; the first local day has only 19 valid rainfall/soil hours. The attempted 2017 May–October calibration download failed validation because a requested variable had no complete daily values; it was not cached. ERA5 scale mappings and model risk scores have not been produced. Earlier probe results are preliminary and do not establish full archive coverage.

Work in progress is saved in `live_common.py`, `live_calibration.py` and the updated `stage1b_gfs_calibration.py` entry point. These are unfinished and must not be used for operational scores. The follow-up historical coverage probe was read-only. At that earlier pause, no Today tab or SQLite store had been built; the new Phase A Live monitor and input-summary SQLite store described below supersede that status. Resume by resolving archive coverage, then fit input mappings and implement snapshot/offline checks; do not refit models or modify `features.py`.

## Apache Spark processing layer

`spark_pipeline.py` uses Apache Spark 4.0.1 DataFrames in `local[*]` mode with Homebrew Java 17, selected by `spark_runtime.py` without changing the system Java installation. The Spark UI is printed on startup, normally http://localhost:4040, and is available while the process runs. Install the optional dependency with `venv/bin/python -m pip install -r requirements_spark.txt`. The existing models, features, thresholds, selected variants, September replay and dashboard remain frozen; Spark builds parallel, checked data products rather than training anything.

The raw lake `data/lake/raw/` is Snappy Parquet partitioned by `variable` and `year`. Each row retains timestamp, requested latitude/longitude, variable and nullable value, plus source, original file and snapped grid coordinates. The session timezone is Asia/Kathmandu. Requested points remain distinct even when several snap to the same ERA5 grid cell. Source distinguishes the nine-point ERA5 TCWV archive, Stage 1 weather, and the two one-point GFS availability probes; GFS probes are retained in the lake but cannot contribute to the historical basin means. Every cached hourly value, including nulls, is retained. Daily satellite summaries and ONI files have no hourly records and are not relabelled as hourly data.

Stage 1 had saved daily rainfall/soil data but no hourly API responses. `spark_fetch_hourly.py` recovered hourly precipitation and all three original soil depths at all nine requested points for 2001–2010. Open-Meteo returned HTTP 429 for 2011, so downloading stopped without retrying or requesting later years. All 25 years of existing hourly TCWV caches were reused. No data are invented, filled or interpolated. `outputs/spark/hourly_inventory.json` lists every included source file and `download_status.json` records the download limit.

The lake contains 5,128,008 rows in 80 variable/year partitions and 80 Parquet data files: 14,543,819 bytes of Parquet versus 87,244,506 bytes of original hourly JSON, excluding small filesystem metadata files. `lake_stats.json` gives row/non-null counts per source and variable. This is a local multi-million-row processing demonstration, not a distributed cluster deployment.

Daily aggregation requires 24 distinct hourly timestamps and 24 non-null values for each point, and all nine complete points for a basin day. Rainfall is summed per point before taking the basin mean; other weather variables use point-day means. Incomplete days remain null. `data/lake/daily_aggregates/` retains source coverage and complete-point counts. `outputs/spark/daily_physical_plan.txt` is the actual formatted `explain()` output. Its `Exchange hashpartitioning` operators shuffle records between tasks for point-day grouping, distinct-hour counting and subsequent basin grouping. Global chronological windows use a single ordered basin series after this large hourly reduction; they are not presented as parallel across dates.

Rainfall interval diagnosis: [Open-Meteo documents hourly precipitation as the accumulation over the preceding hour](https://open-meteo.com/en/docs/historical-weather-api). Spark now assigns hours ending 01:00 through the following 00:00 (24:00) to that Kathmandu day by subtracting one hour from precipitation timestamps for grouping only. Other variables retain their original grouping clock. The previous 00:00–23:00 sum differed on 1,874 of 3,652 days: signed median 0 mm, signed 95th percentile 1 mm, maximum absolute difference 15.311111 mm, and 365 days with absolute difference >1 mm after rounding numerical noise to eight decimals (366 with an unrounded floating-point comparison). With the corrected interval, zero mismatches remain on 3,651 complete days (tolerance 1e-6 mm; largest numerical residual 2.84e-14 mm). The last cached day, 2010-12-31, lacks the following midnight and remains null; no extra download or filling was performed. Year boundaries use the next cached year’s midnight where present. Accepted Stage 1 daily totals are unchanged. Full diagnostics are in `outputs/spark/rainfall_diagnosis.json` and the two `rain_differences_shift_*.csv` files. `data/lake/features_daily/` contains both `hourly_precip_mm` and accepted `precip_mm`, with separate Spark Window 1/3/7/14-day sums (`hourly_rain_*` and `rain_*`). Hourly-derived values remain null beyond recovered coverage. The accepted cached daily rainfall drives the historical feature-percentile and Stage 2 SQL reproduction, with explicit provenance columns. Soil and TCWV use complete hourly aggregates where available and identify daily-only cached Stage 1 coverage elsewhere. Missing hourly coverage is never manufactured from daily inputs. All of these outputs are separate from the frozen predictive artifacts.

Seasonal ranks use exact midranks, eight-decimal tie rounding and ±15 calendar days on a leap-year calendar, matching the unchanged `features.py`. Only June–October 2001–2019 observations enter `data/lake/seasonal_reference/`; no 2020–2025 observation can enter that reference. Spark SQL runs from `spark_queries.sql` and saves onset event counts by year, TCWV percentiles in the three days strictly before onset versus all monsoon days, and all 20 original Stage 2 conditional-probability rows. Both all-day and not-currently-flooding populations are reported explicitly.

`stage_spark_check.py` is offline. It checks raw row retention and unique keys, every historical hourly-derived variable-day against independent pandas resampling, complete-day null masks, all rainfall windows, all seven seasonal percentile definitions, reference-year restrictions, exact SQL agreement, the source TCWV gap and all 80 original frozen artifact hashes. Rate-limited recovery is reported as a warning. The shifted-rainfall parity check verifies both basin means and max-point daily totals against the accepted totals. The user's check-script request ended at “Spark daily”; the remaining checks were implemented as integrity safeguards without altering the scientific rules.

Verified Spark result after rainfall correction: **8 PASS, 1 WARN, 0 FAIL**, with complete output in `outputs/spark/check_output.txt`. The former rainfall discrepancy is resolved by the preceding-hour interval correction. Across all 26 events, mean TCWV percentile in the 78 strictly pre-onset days is 74.296, versus 52.449 over valid monsoon days; the event-specific table is `tcwv_by_event.csv`. The training-period three-day onset probability given TCWV percentile ≥95 is 8/81 (9.88%), versus the all-day base rate 57/2,907 (1.96%). These are reproduced descriptive Stage 2 results, not new model selection. The original 183-day TCWV gap and September 26–28 percentiles are unchanged.

```sh
# Optional recovery; already stopped at HTTP 429 for 2011 in this run:
venv/bin/python spark_fetch_hourly.py
# Local processing, no model fitting:
venv/bin/python spark_pipeline.py
# Reuse the already built raw lake:
venv/bin/python spark_pipeline.py --reuse-raw
venv/bin/python stage_spark_check.py
# Historical dashboard remains unchanged:
venv/bin/streamlit run app.py
```

### Spark scaling benchmark

`venv/bin/python spark_scaling.py` uses 100 shuffle partitions at all scales and measures the shifted hourly-to-daily aggregation at 1x, 10x and 50x input rows using `local[*]`. It performs one untimed warm-up and three measured trials at each scale, reporting median wall-clock time in `outputs/spark/scaling.md` and raw timings in `scaling_trials.json`. Replicas carry independent source grouping keys, and counts/checksums must scale accordingly. Replication occurs inside Spark; no enlarged raw files or new scientific observations are created. This benchmark includes the Parquet scan, synthetic replication, shuffles and forced aggregation; it excludes startup, ingestion, feature percentiles, model scoring and output writes. It is an aggregation benchmark on one machine, not an end-to-end or cluster benchmark.

The first 50x benchmark attempt exhausted the 3 GB Java heap with the default 10 shuffle partitions. Its logs and completed smaller-scale trials are preserved in `outputs/spark/scaling_initial_10_partitions*`. The reported comparison reruns all scales with 100 shuffle partitions and adaptive coalescing disabled, keeping the same heap and all CPU cores. This memory limit is part of the result and is not hidden.

### Big data pipeline dashboard tab

`app.py` now has a **Big data pipeline** tab alongside the unchanged September replay. `spark_dashboard.py` reads only saved JSON/CSV/text reports under `outputs/spark/`; Streamlit never imports or starts Spark. The tab shows the pipeline diagram, lake statistics and variable coverage, measured per-column Spark/pandas differences, rainfall interval correction, SQL charts/tables, scaling medians and chart, and the physical plan with yellow markers on Exchange/hash-partitioning lines. The Nepal-wide comparison is an illustrative scale analogy dependent on archive duration and resolution; the actual benchmark is synthetic basin-data replication.

`stage_spark_check.py` exports `column_parity.csv` and `variable_coverage.csv` during offline processing. `stage_spark_dashboard_check.py` blocks Spark imports, renders both tabs, exercises the original replay slider and checks immutable model/scoring files and paused live-work hashes. Dashboard result: **4 PASS, 0 WARN, 0 FAIL**; see `outputs/spark/dashboard_check_output.txt`. Starting the local HTTP server was not approved in this session; rendering was verified with Streamlit AppTest.

The authorized app change adds only the two-tab wrapper and pipeline-render call. The complete original app is preserved in `outputs/spark/dashboard_original_app.py` and still matches its original frozen hash. `spark_dashboard_integrity.py` verifies that the original imports, page configuration and entire replay body remain verbatim. The original satellite-era 80-file manifest remains intact as a historical baseline; the current Spark check verifies its 79 non-app files plus this explicit replay-integrity check, rather than claiming the app wrapper hash is unchanged. Models, features, thresholds and selected variants are unchanged, and live mode remains paused.

```sh
venv/bin/python stage_spark_dashboard_check.py
venv/bin/streamlit run app.py
```

## Live monitor — Phase A complete, model scoring deferred

`live_poll.py` fetches `gfs_seamless` precipitation and total column integrated water vapour at all nine `BASIN_POINTS`, plus `seamless_v4` GloFAS discharge at `GLOFAS_CHOSEN` (`q_north`). Each successful run saves exact raw HTTP response bytes, request parameters, UTC fetch time, SHA256 response hashes, daily inputs and a summary under `data/live/snapshots/<timestamp>/`. It appends the summary to the `runs` table in `data/live/live_scores.sqlite`. Failed requests retain their raw/error evidence but do not replace the last successful displayed snapshot. No Kafka, background poller, model training or automatic fetching is used.

The window is the previous 14 days, today, and the next seven days (tomorrow through today+7). One extra weather day is requested to supply the following-midnight accumulation hour for the last displayed rainfall day. Rainfall uses hours ending 01:00–24:00; TCWV uses 00:00–23:00 means. Every point-day requires 24 values, and every basin mean requires all nine points. Nulls are preserved. Weather days are Asia/Kathmandu; GloFAS daily dates remain UTC, consistent with training inputs. GFS past-period values are modelled/archived forecasts, not rainfall observations.

The **Live monitor** tab has a manual **Fetch now** button, fetch timestamp, seven-day discharge/threshold chart and table, raw GFS rainfall and TCWV charts with separate past/today and forecast series, a 22-day input table, and snapshot provenance. It shows exactly: “Live inputs only. Model risk scores need inputs on the ERA5 scale; see README.” This is necessary because GFS and ERA5 input distributions differ. Phase A performs no quantile mapping and no model scoring; `risk_1d` and `risk_3d` remain null. GloFAS exceedance is calculated directly against 445.191 m³/s and is not an XGBoost alert. The first snapshot fetched at 2026-10-18T14:00:11.849662+00:00 has no forecast exceedance for October 19–25; its maximum forecast flow is 2.08 m³/s.

`stage_live_check.py` runs with network requests forbidden. It independently recomputes nine-point/24-hour weather values from raw responses, checks response hashes, CSV/SQLite reproduction, all seven GloFAS forecasts and the threshold statement, rendered table equality, and the Fetch now button with a mocked poller. Result: **5 PASS, 0 WARN, 0 FAIL** (`outputs/live/phase_a_check_output.txt`). The original 80-entry freeze manifest is not overwritten: 79 current protected files retain their hashes; its original app hash is verified against the preserved original app snapshot. The authorized three-tab wrapper is checked separately against `outputs/live/phase_a_prior_app.py`. The entire September replay body and Big data tab body/helper are unchanged. A screenshot is saved as `outputs/live/phase_a_monitor.png`.

### Phase B next steps

Phase B resumed only after Phase A passed. The new nine-point ECMWF IFS archive probe is saved under `outputs/live/phase_b_probe/`. On September 27 in 2017, 2021 and 2023, rainfall and all three soil layers each have 216/216 valid point-hours, but TCWV has 0/216. The September 27, 2025 sample has 216/216 for all five variables. Therefore the advertised 2017 archive start cannot be assumed to provide a complete TCWV climatology. The earlier calibration's all-null-variable rejection was valid; these missing source values must not be filled or silently replaced.

Phase A remains the final usable state. Before enabling frozen-model scores, audit full temporal availability for ECMWF TCWV, establish a defensible paired calibration window for each variable, and verify Historical Weather IFS versus Forecast IFS product/resolution compatibility. If the earlier approved GFS fallback is selected, first validate its historical precipitation, TCWV and native soil layers and explicitly label the depth approximations. Then fit and save source-to-ERA5 quantile mappings, report overlap dates/counts and mapped-value correlations, fetch sufficient rolling-window warm-up, and use the unchanged `features.py`, model weights, thresholds, selected variants and analogue templates. Add offline mapping/feature/rescore checks before enabling any scores. Phase A continues to use raw GFS inputs until those checks pass.

```sh
venv/bin/python live_poll.py
venv/bin/python stage_live_check.py
venv/bin/streamlit run app.py
# Optional read-only archive coverage gate (not needed by the dashboard):
venv/bin/python phase_b_probe.py
```

## Public observed live data

The Live monitor now starts with **Observed river level — DHM Khokana gauge**, followed by the unchanged GFS/GloFAS section under **Model estimates and forecast**. Observed gauge level is in metres; it is not modelled discharge, is never converted using an assumed rating curve, and does not enter features or frozen model scoring. The existing model-data Fetch now button retains its behavior; Refresh observed gauge independently refreshes the public gauge snapshot.

### DHM investigation and access

The public page https://dhm.gov.np/hydrology/hms-Single/50 identifies Bagmati at Khokana, station 550.05, series ID 927. Its normal anonymous VIEW form posts to https://dhm.gov.np/site/getRiverWatchBySeriesId_Single with date, period=4 (7 Days Point), seriesid=927 and the public page's anonymous CSRF form token. The HTTP 200 response is JSON containing chart/table HTML. The chart embeds hex-escaped JSON with explicit UTC timestamps and ten-minute observed water-level values. `observed_poll.py` decodes this JSON without executing JavaScript. The HTML table's ambiguous clock labels are not used for timestamps. The page/plot publish warning 3.50 m and danger 4.00 m. No login, CAPTCHA, proxy, user-agent disguise or access bypass is used. `/robots.txt` returned 404; the public observation form required no acceptance of the separate historical-data request terms. On an HTTP block/error the poller stops and retains the previous successful snapshot, rather than working around access restrictions.

Each fetch stores exact page HTML, exact JSON response bytes, source identity, request date/period, UTC fetch time, response hashes, parsed observations, hourly data and summary in `data/live/snapshots/observed_<timestamp>/`. `data/live/observed_latest.json` points only to a validated successful snapshot. The source supplies a seven-calendar-day window with partial first/current days; hourly bins are means of the available ten-minute observations, include sample counts and partial-hour flags, and retain absent hours as NaN. There is no filling or interpolation. Latest-reading metrics show the last actual point rather than the hourly mean. All displayed times are explicitly Asia/Kathmandu, and observations older than three hours receive a stale-data warning. The first saved operational observation snapshot has 952 readings and 160 hourly bins, latest 0.941 m at 2026-10-18 14:35 UTC / 20:20 Nepal time.

### Satellite freshness investigation and next steps

Using the existing read-only Earth Engine session on `bdasem-7`, the most recent `NASA/GPM_L3/IMERG_V07` image was `20261016130000`, provisional, starting 2026-10-16 13:00 UTC. Checked at 2026-10-18 14:24:34 UTC, its age was 49.41 hours from interval start (48.91 hours after the half-hour interval ended). It therefore fails the required **under two days** gate. No stale last-14-day rainfall panel is added and no satellite data enter model scoring. The live tab states the check time and measured latency, rather than claiming this old probe is an ongoing satellite freshness check.

The read-only listing of public `NASA/GPM_L3` assets contained IMERG V04/V05/V06/V07, the legacy IMERG asset and monthly/daily older products, but no separately named Early/Late-run collection. NASA's Early/Late products exist outside that listing; their published latency is not evidence that they are accessible as a newer Earth Engine collection. Next step: recheck V07 ingestion freshness and locate a documented, publicly accessible Early/Late route if needed, without new login or terms violations. Only if actual available image latency is under 48 hours should a 14-day basin satellite panel be built, with complete Kathmandu-day integration and explicit latency/provisional status. See `outputs/live/observed_investigation/` for saved page, XHR, public asset listing and timestamp evidence; [Earth Engine V07 catalogue](https://developers.google.com/earth-engine/datasets/catalog/NASA_GPM_L3_IMERG_V07).

`stage_live_check.py` now also forbids network during independent observed-JSON decoding, verifies source hashes, source-to-display latest level/time and warning/danger thresholds, manually reproduces every hourly average/sample count, and checks satellite freshness exclusion. Both model and observation refresh buttons are tested with mocked pollers. Result: **7 PASS, 0 WARN, 0 FAIL**. The poller for GFS/GloFAS, app tab wrapper, Big data/replay tabs, model weights, features and thresholds are unchanged.

```sh
venv/bin/python observed_poll.py
venv/bin/python stage_live_check.py
venv/bin/streamlit run app.py
```

### Historical DHM gauge validation — formal request needed

A bounded optional probe of the same public Khokana endpoint returned HTTP 200 / success but **zero observations** for a seven-day window anchored at 2024-10-02 (covering the September flood) and for hourly data on 2024-09-28. Only two historical data requests were made. The page accepts past dates, but historical observations were not available for these tested periods; no conclusion about warning/danger crossing can be drawn. Further 2020–2021 event queries were skipped. Live current gauge access is unaffected. Next step: a formal DHM historical-data request for station 550.05, covering 20 September–5 October 2024 and ±2 days around the six test onsets, including timestamps/timezone, datum/units, quality flags and contemporaneous thresholds. No request was submitted and no models, labels, features or dashboard content changed. See `outputs/dhm_validation.md` and raw evidence in `data/dhm_history/`.
