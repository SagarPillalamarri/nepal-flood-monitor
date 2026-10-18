# Stage 3 results

Stage 2 additions and onset models are complete. Work stops before Stage 4.

## Event onset and peak dates

| Event | Onset | Peak |
|---:|---|---|
| 1 | 2001-08-23 | 2001-08-23 |
| 2 | 2002-07-21 | 2002-07-22 |
| 3 | 2003-06-30 | 2003-06-30 |
| 4 | 2003-07-08 | 2003-07-08 |
| 5 | 2003-08-18 | 2003-08-18 |
| 6 | 2005-07-16 | 2005-07-16 |
| 7 | 2007-07-20 | 2007-07-26 |
| 8 | 2009-07-29 | 2009-07-29 |
| 9 | 2009-08-18 | 2009-08-19 |
| 10 | 2010-07-11 | 2010-07-11 |
| 11 | 2011-07-01 | 2011-07-01 |
| 12 | 2012-06-24 | 2012-06-24 |
| 13 | 2012-07-19 | 2012-07-19 |
| 14 | 2014-06-30 | 2014-06-30 |
| 15 | 2014-08-14 | 2014-08-14 |
| 16 | 2017-08-11 | 2017-08-12 |
| 17 | 2018-07-02 | 2018-07-02 |
| 18 | 2018-08-24 | 2018-08-24 |
| 19 | 2019-07-12 | 2019-07-12 |
| 20 | 2020-07-10 | 2020-07-10 |
| 21 | 2020-07-23 | 2020-07-23 |
| 22 | 2020-09-23 | 2020-09-23 |
| 23 | 2021-07-01 | 2021-07-01 |
| 24 | 2021-07-17 | 2021-07-17 |
| 25 | 2021-08-14 | 2021-08-14 |
| 26 | 2024-09-28 | 2024-09-28 |

## Conditional probabilities (training monsoon days)

| Condition | Days | Onset next day | Onset next 3 days |
|---|---:|---:|---:|
| all_days | 2907 | 0.65% | 1.96% |
| tcwv_pct_ge_90 | 203 | 3.94% | 7.88% |
| tcwv_pct_ge_95 | 81 | 4.94% | 9.88% |
| tcwv_pct_ge_95_and_soil_gt_75 | 60 | 3.33% | 6.67% |
| rain_3d_pct_ge_95 | 130 | 1.54% | 1.54% |

## Full Stage 3 check output

```text
[PASS] Percentile reference is 2001–2019 monsoon only; no 2020–2025 values
[PASS] Feature leakage checks: prefix invariance, trailing rainfall and backward discharge shifts pass
[WARN] 31 monsoon TCWV percentiles remain NaN; XGBoost receives NaN natively
[PASS] 1-day target is onset in t+1 through t+1; verified directly against event dates
[PASS] No training, test, case or validation row has an ongoing flood on day t
[PASS] Time split and label-window purge pass: 2877 train, 880 test, 27 separate case days
[PASS] Three chronological validation folds and all threshold tuning remain inside training years
[PASS] Baseline and all XGBoost ablations use identical eligible dates; saved models reproduce scores and alerts
[PASS] Recall, precision, false alarm rate FP/(FP+TN), F1 and event lead times verified
[PASS] Test baseline comparison: XGBoost F1=0.261, baseline F1=0.077; test events=6
[PASS] ONI retained only as a reported ablation; primary model excludes ONI and is not selected on test results
[WARN] September 2024 onset: missed; earliest eligible warning lead=none day(s)
[PASS] Feature importance, separate September risk/discharge plot and model files saved

Summary (1-day): 11 PASS, 2 WARN, 0 FAIL

[PASS] Percentile reference is 2001–2019 monsoon only; no 2020–2025 values
[PASS] Feature leakage checks: prefix invariance, trailing rainfall and backward discharge shifts pass
[WARN] 31 monsoon TCWV percentiles remain NaN; XGBoost receives NaN natively
[PASS] 3-day target is onset in t+1 through t+3; verified directly against event dates
[PASS] No training, test, case or validation row has an ongoing flood on day t
[PASS] Time split and label-window purge pass: 2877 train, 878 test, 27 separate case days
[PASS] Three chronological validation folds and all threshold tuning remain inside training years
[PASS] Baseline and all XGBoost ablations use identical eligible dates; saved models reproduce scores and alerts
[PASS] Recall, precision, false alarm rate FP/(FP+TN), F1 and event lead times verified
[PASS] Test baseline comparison: XGBoost F1=0.112, baseline F1=0.098; test events=6
[PASS] ONI retained only as a reported ablation; primary model excludes ONI and is not selected on test results
[WARN] September 2024 onset: missed; earliest eligible warning lead=none day(s)
[PASS] Feature importance, separate September risk/discharge plot and model files saved

Summary (3-day): 11 PASS, 2 WARN, 0 FAIL

```

## Test metrics (September reported separately)

| Horizon | Model | Recall | Precision | False alarm rate | F1 | Mean detected-event lead |
|---:|---|---:|---:|---:|---:|---:|
| 1 | baseline | 16.7% | 5.0% | 2.17% | 0.077 | 1.0 days |
| 1 | xgb_core | 50.0% | 6.7% | 4.81% | 0.118 | 1.0 days |
| 1 | xgb_oni_ablation | 50.0% | 30.0% | 0.80% | 0.375 | 1.0 days |
| 1 | xgb_tcwv | 50.0% | 17.6% | 1.60% | 0.261 | 1.0 days |
| 3 | baseline | 61.1% | 5.3% | 22.79% | 0.098 | 2.8 days |
| 3 | xgb_core | 66.7% | 6.4% | 20.35% | 0.117 | 2.8 days |
| 3 | xgb_oni_ablation | 61.1% | 9.5% | 12.21% | 0.164 | 2.8 days |
| 3 | xgb_tcwv | 66.7% | 6.1% | 21.51% | 0.112 | 2.8 days |

False alarm rate = FP/(FP+TN). Mean lead excludes missed events. Main test has six events.

## September 2024

All XGBoost variants miss the September 28 onset at both horizons. The baseline detects it 1 day ahead for the primary target and 2 days ahead for the stretch target. No case-based retuning was performed.

![September 2024 3-day scores](/Users/sagarp/Documents/BDA_SEM 7/outputs/stage3/case_study_3d.png)

## Reproduce

```sh
python 'stage2_analysis.py'
python 'stage2_check.py'
python 'stage3_models.py' --horizon 1
python 'stage3_check.py' --horizon 1
python 'stage3_models.py' --horizon 3
python 'stage3_check.py'
```
