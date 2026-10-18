"""Audit validation-only selection and held-out reporting for calendar ablation."""
import json
import sys
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from floodwatch.core import config as c
from floodwatch.pipeline.stage3_models import (artifact, load_data, input_hash, choose_threshold,
                           binary_metrics, evaluate_predictions)
from floodwatch.pipeline.stage3_calendar_ablation import VARIANTS, model_path


def check_horizon(horizon):
    rows = []
    def report(status, message):
        rows.append({'horizon': horizon, 'status': status, 'message': message})
        print(f'[{status}] {horizon}-day: {message}')
    try:
        source, daily, events = load_data()
        folder = c.STAGE3_CALENDAR_DIR
        selection = json.loads((folder / f'selection_{horizon}d.json').read_text())
        if selection['input_hash'] != input_hash() or selection['september_used_for_selection']:
            raise ValueError('Selection inputs changed or September entered selection')
        oof = pd.read_csv(folder / f'validation_predictions_{horizon}d.csv', parse_dates=['date','fit_end','valid_start','valid_end'])
        if oof.date.max() > pd.Timestamp(c.TRAIN_END) or oof.date.min() < pd.Timestamp(c.TRAIN_START):
            raise ValueError('Validation scores leave 2001–2019')
        if oof.date.dt.year.max() > 2019 or oof.date.dt.strftime('%Y-%m').eq('2024-09').any():
            raise ValueError('Test or September case rows entered validation')
        if set(oof.fold.unique()) != {1,2,3} or set(oof.variant.unique()) != set(VARIANTS):
            raise ValueError('Missing fold or feature variant')
        with_columns = selection['models']['with_monsoon_day']['columns']
        without_columns = selection['models']['without_monsoon_day']['columns']
        if without_columns != [name for name in with_columns if name != 'monsoon_day']:
            raise ValueError('Ablation changed features other than monsoon_day')
        samples = pd.read_csv(artifact('samples', horizon), parse_dates=['date'])
        if samples.ongoing_flood.ne(0).any() or samples.date.duplicated().any():
            raise ValueError('Current-flood or duplicate rows entered the accepted sample set')
        for fold, group in oof.groupby('fold'):
            expected = c.VALIDATION_FOLDS[int(fold)-1]
            if group.date.min() < pd.Timestamp(expected[1]) or group.date.max() + pd.Timedelta(days=horizon) > pd.Timestamp(expected[2]):
                raise ValueError('Fold targets cross their validation boundary')
            if daily.flood.reindex(group.date).ne(0).any():
                raise ValueError('Ongoing flood rows entered validation')
        for fold, group in oof.groupby('fold'):
            dates_by_variant = [set(part.date) for _, part in group.groupby('variant')]
            if len(dates_by_variant) != 2 or dates_by_variant[0] != dates_by_variant[1]:
                raise ValueError('Feature variants do not share identical validation dates')
        report('PASS', 'Three chronological validation folds contain training years only; no ongoing floods or September data')

        metrics = pd.read_csv(folder / f'metrics_{horizon}d.csv')
        tuning = pd.read_csv(folder / f'threshold_tuning_{horizon}d.csv')
        fold_metrics = pd.read_csv(folder / f'validation_fold_metrics_{horizon}d.csv')
        means = {}
        for variant in VARIANTS:
            part = oof[oof.variant.eq(variant)]
            threshold, _ = choose_threshold(part.target, part.score, c.XGB_SCORE_THRESHOLDS)
            stored = selection['models'][variant]['threshold']
            np.testing.assert_allclose(threshold, stored)
            tune = tuning[tuning.variant.eq(variant)]
            wanted_tune = tune.sort_values(['f1','precision','threshold'], ascending=False).iloc[0].threshold
            if not np.isclose(wanted_tune, threshold):
                raise ValueError('Saved validation threshold tuning is inconsistent')
            means[variant] = float(fold_metrics.loc[fold_metrics.variant.eq(variant), 'f1'].mean())
        expected_selected = ('without_monsoon_day' if means['without_monsoon_day'] >= means['with_monsoon_day']
                             else 'with_monsoon_day')
        if expected_selected != selection['selected_variant']:
            raise ValueError('Selected feature set does not match mean validation F1/tie rule')
        report('PASS', f'Validation-only thresholds and selection verified; mean F1 with/without monsoon_day={means["with_monsoon_day"]:.3f}/{means["without_monsoon_day"]:.3f}; selected {expected_selected}')

        features = pd.read_csv(c.STAGE3_FEATURES, parse_dates=['date']).set_index('date')
        predictions = pd.read_csv(folder / f'predictions_{horizon}d.csv', parse_dates=['date'])
        for variant, group in predictions.groupby('variant'):
            columns = selection['models'][variant]['columns']
            if ('monsoon_day' in columns) != (variant == 'with_monsoon_day'):
                raise ValueError('Variant feature columns do not match its ablation name')
            model = XGBClassifier()
            model.load_model(model_path(variant, horizon))
            score = model.predict_proba(features.loc[pd.DatetimeIndex(group.date), columns])[:, 1]
            np.testing.assert_allclose(score, group.score.to_numpy(), rtol=1e-6, atol=1e-7)
            alarm = score >= selection['models'][variant]['threshold']
            np.testing.assert_array_equal(alarm.astype(int), group.alarm.to_numpy())
        recomputed, _ = evaluate_predictions(predictions, events, horizon)
        saved = metrics.drop(columns=['mean_validation_f1','selected_by_validation']).sort_values(['variant','split']).reset_index(drop=True)
        recomputed = recomputed.sort_values(['variant','split']).reset_index(drop=True)
        pd.testing.assert_frame_equal(saved, recomputed, check_dtype=False, rtol=1e-6, atol=1e-7)
        case = predictions[predictions.split.eq('case')]
        if daily.flood.reindex(case.date).ne(0).any():
            raise ValueError('Ongoing flood entered separate-case evaluation')
        report('PASS', 'Saved models reproduce scores, alerts and test/case metrics; both variants use identical dates')
    except (OSError, ValueError, KeyError, AssertionError, ImportError) as error:
        report('FAIL', str(error))
    return rows


def main():
    results = []
    for horizon in c.TARGET_HORIZONS:
        results.extend(check_horizon(horizon))
    c.STAGE3_CALENDAR_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(c.STAGE3_CALENDAR_DIR / 'checks.csv', index=False)
    failures = sum(row['status'] == 'FAIL' for row in results)
    passes = sum(row['status'] == 'PASS' for row in results)
    warns = sum(row['status'] == 'WARN' for row in results)
    print(f'\nSummary: {passes} PASS, {warns} WARN, {failures} FAIL')
    return int(failures > 0)


if __name__ == '__main__':
    sys.exit(main())
