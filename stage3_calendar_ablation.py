"""Select whether to retain monsoon_day using training-period validation only."""
import json
import sys
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import config as c
from features import fit_reference, build_features, variant_columns
from targets import onset_series, onset_target, eligible_rows
from stage3_models import (artifact, load_data, input_hash, make_model,
                           choose_threshold, binary_metrics, evaluate_predictions)

VARIANTS = ('with_monsoon_day', 'without_monsoon_day')


def model_path(variant, horizon):
    return c.STAGE3_CALENDAR_DIR / f'{variant}_{horizon}d.json'


def main(horizon):
    c.STAGE3_CALENDAR_DIR.mkdir(parents=True, exist_ok=True)
    source, daily, events = load_data()
    full_reference = fit_reference(source)
    features = build_features(source, full_reference)
    current_columns = variant_columns(features, c.PRIMARY_MODEL_VARIANT)
    if 'monsoon_day' not in current_columns:
        raise ValueError('Current primary model does not contain monsoon_day')
    columns_by_variant = {
        'with_monsoon_day': current_columns,
        'without_monsoon_day': [name for name in current_columns if name != 'monsoon_day'],
    }
    onset = onset_series(source.index, events)
    target = onset_target(onset, horizon)

    # Generate fold scores with the same chronological boundaries and fold-local references.
    oof_parts = []
    for fold, (fit_end, valid_start, valid_end) in enumerate(c.VALIDATION_FOLDS, 1):
        fold_features = build_features(source.loc[:valid_end], fit_reference(source, fit_end))
        train_dates = eligible_rows(daily, fold_features, target, c.TRAIN_START, fit_end, horizon)
        valid_dates = eligible_rows(daily, fold_features, target, valid_start, valid_end, horizon)
        for variant, columns in columns_by_variant.items():
            model = make_model(target.loc[train_dates])
            model.fit(fold_features.loc[train_dates, columns], target.loc[train_dates].astype(int))
            scores = model.predict_proba(fold_features.loc[valid_dates, columns])[:, 1]
            oof_parts.append(pd.DataFrame({'date': valid_dates, 'variant': variant,
                                           'score': scores, 'target': target.loc[valid_dates].to_numpy(),
                                           'fold': fold, 'fit_end': fit_end,
                                           'valid_start': valid_start, 'valid_end': valid_end}))
        print(f'{horizon}d validation fold {fold}: {len(train_dates)} fit / {len(valid_dates)} validation rows', flush=True)
    oof = pd.concat(oof_parts, ignore_index=True)
    oof.to_csv(c.STAGE3_CALENDAR_DIR / f'validation_predictions_{horizon}d.csv', index=False)

    thresholds, tuning_parts, fold_metric_rows = {}, [], []
    for variant in VARIANTS:
        part = oof[oof.variant.eq(variant)]
        threshold, tuning = choose_threshold(part.target, part.score, c.XGB_SCORE_THRESHOLDS)
        thresholds[variant] = threshold
        tuning['variant'] = variant
        tuning_parts.append(tuning)
        for fold, group in part.groupby('fold'):
            metric = binary_metrics(group.target, group.score >= threshold)
            fold_metric_rows.append({'horizon': horizon, 'variant': variant, 'fold': int(fold),
                                     'threshold': threshold, **metric})
    pd.concat(tuning_parts, ignore_index=True).to_csv(c.STAGE3_CALENDAR_DIR / f'threshold_tuning_{horizon}d.csv', index=False)
    fold_metrics = pd.DataFrame(fold_metric_rows)
    fold_metrics.to_csv(c.STAGE3_CALENDAR_DIR / f'validation_fold_metrics_{horizon}d.csv', index=False)
    mean_f1 = fold_metrics.groupby('variant').f1.mean().to_dict()
    # Exact tie rule specified by the user: prefer the version without the calendar prior.
    selected = ('without_monsoon_day' if mean_f1['without_monsoon_day'] >= mean_f1['with_monsoon_day']
                else 'with_monsoon_day')

    train_dates = eligible_rows(daily, features, target, c.TRAIN_START, c.TRAIN_END, horizon)
    split_dates = {
        'test': eligible_rows(daily, features, target, c.TEST_START, c.TEST_END, horizon, exclude_case=True),
        'case': eligible_rows(daily, features, target, c.CASE_REPORT_START, c.CASE_REPORT_END, horizon),
    }
    prediction_parts = []
    model_settings = {}
    for variant, columns in columns_by_variant.items():
        model = make_model(target.loc[train_dates])
        model.fit(features.loc[train_dates, columns], target.loc[train_dates].astype(int))
        model.save_model(model_path(variant, horizon))
        model_settings[variant] = {'threshold': thresholds[variant], 'columns': columns,
                                   'mean_validation_f1': float(mean_f1[variant]),
                                   'validation_fold_f1': fold_metrics[fold_metrics.variant.eq(variant)].f1.tolist()}
        for split, dates in split_dates.items():
            score = model.predict_proba(features.loc[dates, columns])[:, 1]
            prediction_parts.append(pd.DataFrame({'date': dates, 'horizon': horizon, 'split': split,
                                                  'variant': variant, 'target': target.loc[dates].to_numpy(),
                                                  'score': score, 'alarm': (score >= thresholds[variant]).astype(int)}))
    predictions = pd.concat(prediction_parts, ignore_index=True)
    predictions.to_csv(c.STAGE3_CALENDAR_DIR / f'predictions_{horizon}d.csv', index=False)
    metrics, leads = evaluate_predictions(predictions, events, horizon)
    metrics['mean_validation_f1'] = metrics.variant.map(mean_f1)
    metrics['selected_by_validation'] = metrics.variant.eq(selected)
    metrics.to_csv(c.STAGE3_CALENDAR_DIR / f'metrics_{horizon}d.csv', index=False)
    leads.to_csv(c.STAGE3_CALENDAR_DIR / f'event_leads_{horizon}d.csv', index=False)
    result = {'horizon': horizon, 'selected_variant': selected,
              'selection_rule': 'higher mean F1 across the three chronological training validation folds; exact tie selects without monsoon_day',
              'threshold_rule': 'pooled out-of-fold validation F1; then precision; then higher threshold',
              'validation_years_only': True, 'september_used_for_selection': False,
              'input_hash': input_hash(), 'models': model_settings}
    (c.STAGE3_CALENDAR_DIR / f'selection_{horizon}d.json').write_text(json.dumps(result, indent=2))
    summaries = []
    if c.STAGE3_CALENDAR_SELECTED.exists():
        summaries = json.loads(c.STAGE3_CALENDAR_SELECTED.read_text()).get('horizons', [])
        summaries = [row for row in summaries if row.get('horizon') != horizon]
    summaries.append({'horizon': horizon, 'selected_variant': selected,
                      'threshold': thresholds[selected], 'columns': columns_by_variant[selected],
                      'selection_file': str(c.STAGE3_CALENDAR_DIR / f'selection_{horizon}d.json')})
    c.STAGE3_CALENDAR_SELECTED.write_text(json.dumps({'horizons': sorted(summaries, key=lambda x: x['horizon'])}, indent=2))
    print(f'\n{horizon}-day mean validation F1:')
    print(fold_metrics.groupby('variant').f1.agg(['mean', 'min', 'max']).to_string())
    print(f'Selected: {selected}; validation threshold={thresholds[selected]}')
    print(f'\nHeld-out metrics (other variant is not selected):\n{metrics.to_string(index=False)}')


if __name__ == '__main__':
    try:
        for horizon in c.TARGET_HORIZONS:
            main(horizon)
    except (OSError, ValueError, KeyError, AssertionError, ImportError) as error:
        print(f'[FAIL] {error}', file=sys.stderr)
        sys.exit(1)
