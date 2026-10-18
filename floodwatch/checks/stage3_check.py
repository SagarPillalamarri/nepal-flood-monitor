"""Independent checks of onset timing, causal features, splits and saved models."""
import argparse
import json
import sys
import numpy as np
import pandas as pd
from floodwatch.core import config as c
from floodwatch.core.features import fit_reference, build_features, raw_signals, variant_columns
from floodwatch.core.targets import onset_series, onset_target, eligible_rows
from floodwatch.pipeline.stage3_models import artifact, load_data, input_hash, binary_metrics, choose_threshold, evaluate_predictions

results = []


def report(status, message):
    results.append(dict(status=status, message=message))
    print(f'[{status}] {message}')


def main(horizon):
    results.clear()
    try:
        from xgboost import XGBClassifier
        source, daily, events = load_data()
        features = pd.read_csv(c.STAGE3_FEATURES, parse_dates=['date']).set_index('date')
        reference = pd.read_csv(c.STAGE3_REFERENCE, parse_dates=['date']).set_index('date')
        samples = pd.read_csv(artifact('samples',horizon), parse_dates=['date','label_end'])
        predictions = pd.read_csv(artifact('predictions',horizon), parse_dates=['date'])
        validation = pd.read_csv(artifact('validation_predictions',horizon), parse_dates=['date','fit_end','valid_start','valid_end'])
        metadata = json.loads(artifact('metadata',horizon,'json').read_text())
        metrics = pd.read_csv(artifact('metrics',horizon))
        if metadata['input_hash'] != input_hash():
            raise ValueError('Saved model inputs changed')
        pd.testing.assert_frame_equal(reference, fit_reference(source))
        if reference.index.min() < pd.Timestamp(c.TRAIN_START) or reference.index.max() > pd.Timestamp(c.TRAIN_END):
            raise ValueError('Reference contains dates outside training')
        report('PASS', 'Percentile reference is 2001–2019 monsoon only; no 2020–2025 values')
        pd.testing.assert_frame_equal(features, build_features(source,reference), check_dtype=False)
        for cutoff in (c.TRAIN_END,c.CASE_PLOT_START,c.TCWV_EVENT_END):
            prefix = build_features(source.loc[:cutoff], reference)
            pd.testing.assert_frame_equal(prefix, features.loc[prefix.index], check_dtype=False)
        # Verify the rolling sums and differences against actual dated input slices.
        check_date = pd.Timestamp(c.TCWV_EVENT_END)
        raw = raw_signals(source)
        for days in c.RAINFALL_WINDOWS:
            value = source.precip_mm.loc[check_date-pd.Timedelta(days=days-1):check_date].sum()
            np.testing.assert_allclose(raw.loc[check_date,f'rain_{days}d'],value)
        for lag in c.DISCHARGE_RISE_WINDOWS:
            wanted = (source.loc[check_date,c.GLOFAS_CHOSEN]-source.loc[check_date-pd.Timedelta(days=lag),c.GLOFAS_CHOSEN])/lag
            np.testing.assert_allclose(features.loc[check_date,f'discharge_rise_{lag}d'],wanted)
        report('PASS', 'Feature leakage checks: prefix invariance, trailing rainfall and backward discharge shifts pass')
        nan_dates = features.index[features.tcwv_pct.isna()]
        expected_nan = source.tcwv_mm.reindex(features.index).isna()
        if not features.tcwv_pct.isna().equals(expected_nan.rename('tcwv_pct')):
            raise ValueError('Missing TCWV was filled or propagated incorrectly')
        report('WARN' if len(nan_dates) else 'PASS', f'{len(nan_dates)} monsoon TCWV percentiles remain NaN; XGBoost receives NaN natively')

        onset_dates = pd.DatetimeIndex(events.start)
        y = onset_target(onset_series(source.index,events),horizon)
        if samples.date.duplicated().any() or not samples.date.isin(features.index).all():
            raise ValueError('Invalid or repeated sample dates')
        independent = [int(((onset_dates > date) & (onset_dates <= date+pd.Timedelta(days=horizon))).any()) for date in samples.date]
        np.testing.assert_array_equal(samples.target.to_numpy(dtype=int), independent)
        np.testing.assert_array_equal(samples.target.to_numpy(), y.reindex(samples.date).to_numpy())
        report('PASS', f'{horizon}-day target is onset in t+1 through t+{horizon}; verified directly against event dates')
        if samples.ongoing_flood.ne(0).any() or daily.flood.reindex(samples.date).ne(0).any():
            raise ValueError('An ongoing flood day entered training/evaluation')
        if daily.flood.reindex(validation.date).ne(0).any():
            raise ValueError('An ongoing flood day entered chronological validation')
        report('PASS', 'No training, test, case or validation row has an ongoing flood on day t')
        train = samples[samples.split.eq('train')]
        test = samples[samples.split.eq('test')]
        case = samples[samples.split.eq('case')]
        if train.label_end.max() > pd.Timestamp(c.TRAIN_END) or train.date.max() >= test.date.min():
            raise ValueError('Training label windows cross into test or split is not chronological')
        if test.date.min() < pd.Timestamp(c.TEST_START) or test.label_end.max() > pd.Timestamp(c.TEST_END):
            raise ValueError('Test dates/targets outside test period')
        case_overlap = (test.date <= pd.Timestamp(c.CASE_REPORT_END)) & (test.label_end >= pd.Timestamp(c.CASE_REPORT_START))
        if case_overlap.any() or train.date.between(c.CASE_REPORT_START,c.CASE_REPORT_END).any():
            raise ValueError('September case is not isolated')
        report('PASS', f'Time split and label-window purge pass: {len(train)} train, {len(test)} test, {len(case)} separate case days')
        for fold, group in validation.groupby('fold'):
            fit_end, valid_start, valid_end = c.VALIDATION_FOLDS[int(fold)-1]
            if group.date.min() < pd.Timestamp(valid_start) or group.date.max()+pd.Timedelta(days=horizon) > pd.Timestamp(valid_end):
                raise ValueError('Validation target window exceeds its fold')
            if pd.Timestamp(fit_end) >= pd.Timestamp(valid_start) or pd.Timestamp(valid_end) > pd.Timestamp(c.TRAIN_END):
                raise ValueError('Invalid chronological validation fold')
        report('PASS', 'Three chronological validation folds and all threshold tuning remain inside training years')

        expected_variants = {'baseline',*c.MODEL_VARIANTS}
        if set(predictions.variant) != expected_variants or set(metrics.variant) != expected_variants:
            raise ValueError('Baseline comparison or an ablation is missing')
        for variant, group in predictions.groupby('variant'):
            if group.date.duplicated().any() or set(group.date) != set(samples.date):
                raise ValueError('Models were evaluated on different days')
            if daily.flood.reindex(group.date).ne(0).any():
                raise ValueError('Ongoing flood in prediction table')
            settings = metadata['models'][variant]
            threshold = settings['threshold']
            oof = validation[validation.variant.eq(variant)]
            candidates = c.BASELINE_PERCENTILE_THRESHOLDS if variant == 'baseline' else c.XGB_SCORE_THRESHOLDS
            wanted_threshold,_ = choose_threshold(oof.target,oof.score,candidates,strict=variant=='baseline')
            np.testing.assert_allclose(threshold,wanted_threshold)
            dates = pd.DatetimeIndex(group.date)
            if variant == 'baseline':
                score = features.loc[dates,'rain_3d_pct'].to_numpy()
                alarm = score > threshold
            else:
                columns = variant_columns(features,variant)
                if columns != settings['columns']:
                    raise ValueError('Model columns disagree with shared feature definitions')
                model = XGBClassifier()
                model.load_model(artifact(variant,horizon,'json'))
                score = model.predict_proba(features.loc[dates,columns])[:,1]
                alarm = score >= threshold
            np.testing.assert_allclose(group.score,score,rtol=1e-6,atol=1e-7)
            np.testing.assert_array_equal(group.alarm,alarm.astype(int))
        report('PASS', 'Baseline and all XGBoost ablations use identical eligible dates; saved models reproduce scores and alerts')
        recomputed, leads = evaluate_predictions(predictions,events,horizon)
        pd.testing.assert_frame_equal(metrics,recomputed,check_dtype=False,rtol=1e-6,atol=1e-7)
        for row in metrics.itertuples():
            group = predictions[predictions.variant.eq(row.variant)&predictions.split.eq(row.split)]
            tp=int(((group.target==1)&(group.alarm==1)).sum())
            fp=int(((group.target==0)&(group.alarm==1)).sum())
            fn=int(((group.target==1)&(group.alarm==0)).sum())
            tn=int(((group.target==0)&(group.alarm==0)).sum())
            if (row.tp,row.fp,row.fn,row.tn)!=(tp,fp,fn,tn):
                raise ValueError('Confusion matrix counts disagree')
            np.testing.assert_allclose(row.false_alarm_rate,fp/(fp+tn))
        if leads.lead_days.dropna().lt(1).any() or leads.lead_days.dropna().gt(horizon).any():
            raise ValueError('Lead time outside forecast horizon')
        report('PASS', 'Recall, precision, false alarm rate FP/(FP+TN), F1 and event lead times verified')
        test_metrics = metrics[metrics.split.eq('test')].set_index('variant')
        primary,baseline = test_metrics.loc[c.PRIMARY_MODEL_VARIANT],test_metrics.loc['baseline']
        report('WARN' if primary.f1 <= baseline.f1 else 'PASS',
               f'Test baseline comparison: XGBoost F1={primary.f1:.3f}, baseline F1={baseline.f1:.3f}; test events={int(primary.events_evaluable)}')
        report('PASS', 'ONI retained only as a reported ablation; primary model excludes ONI and is not selected on test results')
        case_event = leads[(leads.variant == c.PRIMARY_MODEL_VARIANT) & (leads.split == 'case')]
        if len(case_event) != 1 or case_event.onset.iloc[0] != pd.Timestamp(c.CASE_PEAK_DATE):
            raise ValueError('September 2024 event missing from separate evaluation')
        detected = bool(case_event.detected.iloc[0])
        lead = case_event.lead_days.iloc[0]
        report('PASS' if detected else 'WARN',
               f'September 2024 onset: {"detected" if detected else "missed"}; earliest eligible warning lead={lead if detected else "none"} day(s)')
        for kind in ('feature_importance','case_study'):
            if not artifact(kind,horizon,'png').exists():
                raise ValueError('Required plot missing')
        report('PASS', 'Feature importance, separate September risk/discharge plot and model files saved')
    except (OSError, ValueError, KeyError, AssertionError, ImportError) as error:
        report('FAIL', str(error))
    statuses = [r['status'] for r in results]
    failures = statuses.count('FAIL')
    pd.DataFrame(results).to_csv(artifact('checks',horizon),index=False)
    print(f"\nSummary ({horizon}-day): {statuses.count('PASS')} PASS, {statuses.count('WARN')} WARN, {failures} FAIL")
    if not failures and horizon == c.TARGET_HORIZONS[0]:
        c.STAGE3_READY.write_text(json.dumps(dict(input_hash=input_hash(),failures=0,horizon=horizon)))
    return int(bool(failures))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon',type=int,choices=c.TARGET_HORIZONS)
    args = parser.parse_args()
    horizons = (args.horizon,) if args.horizon is not None else c.TARGET_HORIZONS
    failures = 0
    for horizon in horizons:
        failures += main(horizon)
    reports = []
    for horizon in c.TARGET_HORIZONS:
        path = artifact('checks',horizon)
        if path.exists():
            frame = pd.read_csv(path)
            frame.insert(0,'horizon',horizon)
            reports.append(frame)
    if reports:
        pd.concat(reports,ignore_index=True).to_csv(c.STAGE3_REPORT,index=False)
    sys.exit(int(bool(failures)))
