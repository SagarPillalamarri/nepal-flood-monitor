"""Train onset models with chronological validation and separate case reporting."""
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import config as c
from features import fit_reference, build_features, variant_columns
from targets import onset_series, onset_target, eligible_rows
from stage2_analysis import load_input


def artifact(kind, horizon, suffix='csv'):
    return c.STAGE3_DIR / f'{kind}_{horizon}d.{suffix}'


def load_data():
    source = load_input()
    daily = pd.read_csv(c.STAGE2_DAILY, parse_dates=['date']).set_index('date')
    pd.testing.assert_frame_equal(daily[source.columns], source)
    events = pd.read_csv(c.STAGE2_EVENTS, parse_dates=['start','peak_date','end'])
    return source, daily, events


def input_hash():
    return hashlib.sha256(c.STAGE2_DAILY.read_bytes()).hexdigest()


def make_model(target):
    from xgboost import XGBClassifier
    positive = int(target.sum())
    if not 0 < positive < len(target):
        raise ValueError('Training target must contain both classes')
    return XGBClassifier(**c.XGB_PARAMS, scale_pos_weight=(len(target)-positive)/positive,
                         missing=np.nan)


def binary_metrics(target, alarm):
    y, p = np.asarray(target, dtype=int), np.asarray(alarm, dtype=int)
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    precision = tp/(tp+fp) if tp+fp else 0.0
    recall = tp/(tp+fn) if tp+fn else np.nan
    f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0
    return dict(n_days=len(y), positives=int(y.sum()), tp=tp, fp=fp, fn=fn, tn=tn,
                recall=recall, precision=precision, false_alarm_rate=fp/(fp+tn) if fp+tn else np.nan,
                false_alarm_ratio=fp/(tp+fp) if tp+fp else np.nan, f1=f1)


def choose_threshold(target, scores, candidates, strict=False):
    rows = []
    for value in candidates:
        alarms = scores > value if strict else scores >= value
        metrics = binary_metrics(target, alarms)
        rows.append(dict(threshold=value, **metrics))
    table = pd.DataFrame(rows)
    # Deterministic tie-breaks: F1, precision, then a more conservative threshold.
    best = table.sort_values(['f1','precision','threshold'], ascending=False).iloc[0]
    return float(best.threshold), table


def validation_predictions(source, daily, events, horizon):
    onset = onset_series(source.index, events)
    target = onset_target(onset, horizon)
    records = []
    for fold, (fit_end, valid_start, valid_end) in enumerate(c.VALIDATION_FOLDS, start=1):
        # Fit the percentile reference again inside each chronological fold.
        reference = fit_reference(source, fit_end)
        features = build_features(source.loc[:valid_end], reference)
        train_dates = eligible_rows(daily, features, target, c.TRAIN_START, fit_end, horizon)
        valid_dates = eligible_rows(daily, features, target, valid_start, valid_end, horizon)
        print(f'{horizon}d fold {fold}: {len(train_dates)} training/{len(valid_dates)} validation days', flush=True)
        for variant in ('baseline',) + c.MODEL_VARIANTS:
            if variant == 'baseline':
                scores = features.loc[valid_dates, 'rain_3d_pct'].to_numpy()
            else:
                columns = variant_columns(features, variant)
                model = make_model(target.loc[train_dates])
                model.fit(features.loc[train_dates, columns], target.loc[train_dates].astype(int))
                scores = model.predict_proba(features.loc[valid_dates, columns])[:, 1]
            records.append(pd.DataFrame(dict(date=valid_dates, variant=variant, score=scores,
                                              target=target.loc[valid_dates].to_numpy(), fold=fold,
                                              fit_end=fit_end, valid_start=valid_start, valid_end=valid_end)))
    return pd.concat(records, ignore_index=True)


def evaluate_predictions(predictions, events, horizon):
    rows, event_rows = [], []
    for (variant, split), group in predictions.groupby(['variant','split']):
        if split == 'train':
            continue
        metrics = binary_metrics(group.target, group.alarm)
        leads = []
        event_count = 0
        for event in events.itertuples():
            candidates = group[(group.date < event.start)
                               & (group.date >= event.start-pd.Timedelta(days=horizon))]
            if candidates.empty:
                continue
            event_count += 1
            warned = candidates[candidates.alarm.eq(1)]
            lead = (event.start-warned.date.min()).days if len(warned) else np.nan
            if len(warned):
                leads.append(lead)
            event_rows.append(dict(horizon=horizon, variant=variant, split=split,
                                   event_id=event.event_id, onset=event.start, detected=bool(len(warned)),
                                   lead_days=lead))
        rows.append(dict(horizon=horizon, variant=variant, split=split, **metrics,
                         events_evaluable=event_count, events_detected=len(leads),
                         event_recall=len(leads)/event_count if event_count else np.nan,
                         mean_lead_days=float(np.mean(leads)) if leads else np.nan))
    return pd.DataFrame(rows), pd.DataFrame(event_rows)


def save_plots(horizon, predictions, features, source):
    import os
    os.environ.setdefault('MPLCONFIGDIR', str(c.MATPLOTLIB_CACHE))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from xgboost import XGBClassifier
    model = XGBClassifier()
    model.load_model(artifact(c.PRIMARY_MODEL_VARIANT, horizon, 'json'))
    columns = variant_columns(features, c.PRIMARY_MODEL_VARIANT)
    importance = pd.Series(model.feature_importances_, index=columns).sort_values()
    importance.rename('importance').to_csv(artifact('feature_importance', horizon))
    fig, ax = plt.subplots(figsize=c.STAGE3_PLOT_SIZE, constrained_layout=True)
    importance.plot.barh(ax=ax)
    ax.set(title=f'XGBoost {horizon}-day onset feature importance', xlabel='Normalized gain importance')
    fig.savefig(artifact('feature_importance', horizon, 'png'), dpi=c.STAGE3_PLOT_DPI)
    plt.close(fig)
    case = source.loc[c.CASE_PLOT_START:c.CASE_PLOT_END]
    fig, axes = plt.subplots(2, 1, figsize=c.STAGE3_PLOT_SIZE, sharex=True, constrained_layout=True)
    for variant in ('baseline', c.PRIMARY_MODEL_VARIANT):
        part = predictions[(predictions.variant == variant) & (predictions.split == 'case')].set_index('date')
        if variant == 'baseline':
            axes[0].step(case.index, part.alarm.reindex(case.index), where='mid', label='Baseline alert')
        else:
            axes[0].plot(case.index, part.score.reindex(case.index), marker='o', label='XGBoost score (uncalibrated)')
    selected = json.loads(artifact('metadata', horizon, 'json').read_text())
    axes[0].axhline(selected['models'][c.PRIMARY_MODEL_VARIANT]['threshold'], color='tomato', linestyle='--', label='XGBoost alert threshold')
    axes[0].axvline(pd.Timestamp(c.CASE_PEAK_DATE), color='black', alpha=.3)
    axes[0].set(title=f'September 2024: onset within next {horizon} day(s)', ylabel='Score / alert', ylim=(-.05,1.05))
    axes[0].legend()
    axes[1].plot(case.index, case[c.GLOFAS_CHOSEN], marker='o', color='navy', label=c.GLOFAS_CHOSEN)
    axes[1].set(ylabel='Discharge (m³/s)')
    axes[1].legend()
    fig.autofmt_xdate()
    fig.savefig(artifact('case_study', horizon, 'png'), dpi=c.STAGE3_PLOT_DPI)
    plt.close(fig)


def summarize_all():
    tables = [pd.read_csv(artifact('metrics', h)) for h in c.TARGET_HORIZONS if artifact('metrics', h).exists()]
    metrics = pd.concat(tables, ignore_index=True)
    metrics.to_csv(c.STAGE3_METRICS, index=False)
    ablation = []
    for horizon in metrics.horizon.unique():
        for split in ('test','case'):
            group = metrics[(metrics.horizon == horizon) & (metrics.split == split)].set_index('variant')
            no_oni, with_oni = group.loc['xgb_tcwv'], group.loc['xgb_oni_ablation']
            ablation.append(dict(horizon=horizon, split=split, f1_without_oni=no_oni.f1,
                                  f1_with_oni=with_oni.f1, f1_change=with_oni.f1-no_oni.f1,
                                  improves_test=bool(split == 'test' and with_oni.f1 > no_oni.f1),
                                  primary_uses_oni=False))
    pd.DataFrame(ablation).to_csv(c.STAGE3_ONI_ABLATION, index=False)
    case_tables = [pd.read_csv(artifact('predictions', h)) for h in c.TARGET_HORIZONS if artifact('predictions', h).exists()]
    case = pd.concat(case_tables, ignore_index=True)
    case[case.split.eq('case')].to_csv(c.STAGE3_CASE, index=False)


def main(horizon):
    if horizon != c.TARGET_HORIZONS[0]:
        if not c.STAGE3_READY.exists():
            raise ValueError('Run and pass the primary 1-day check before the stretch horizon')
        ready = json.loads(c.STAGE3_READY.read_text())
        if ready.get('input_hash') != input_hash() or ready.get('failures') != 0:
            raise ValueError('Primary check is stale or did not pass')
    c.STAGE3_DIR.mkdir(parents=True, exist_ok=True)
    source, daily, events = load_data()
    reference = fit_reference(source)
    features = build_features(source, reference)
    reference.to_csv(c.STAGE3_REFERENCE)
    features.to_csv(c.STAGE3_FEATURES)
    target = onset_target(onset_series(source.index, events), horizon)
    index_sets = dict(train=eligible_rows(daily, features, target, c.TRAIN_START, c.TRAIN_END, horizon),
                      test=eligible_rows(daily, features, target, c.TEST_START, c.TEST_END, horizon, exclude_case=True),
                      case=eligible_rows(daily, features, target, c.CASE_REPORT_START, c.CASE_REPORT_END, horizon))
    sample_rows = []
    for split, dates in index_sets.items():
        sample_rows.append(pd.DataFrame(dict(date=dates, split=split, target=target.loc[dates].to_numpy(),
                                             ongoing_flood=daily.flood.loc[dates].to_numpy(),
                                             label_end=dates+pd.Timedelta(days=horizon))))
    pd.concat(sample_rows).to_csv(artifact('samples', horizon), index=False)
    oof = validation_predictions(source, daily, events, horizon)
    oof.to_csv(artifact('validation_predictions', horizon), index=False)
    predictions, settings, tuning = [], {}, []
    for variant in ('baseline',) + c.MODEL_VARIANTS:
        validation = oof[oof.variant.eq(variant)]
        candidates = c.BASELINE_PERCENTILE_THRESHOLDS if variant == 'baseline' else c.XGB_SCORE_THRESHOLDS
        threshold, table = choose_threshold(validation.target, validation.score, candidates, strict=variant == 'baseline')
        table['variant'] = variant
        tuning.append(table)
        columns = ['rain_3d_pct'] if variant == 'baseline' else variant_columns(features, variant)
        model = None
        if variant != 'baseline':
            train_dates = index_sets['train']
            model = make_model(target.loc[train_dates])
            model.fit(features.loc[train_dates, columns], target.loc[train_dates].astype(int))
            model.save_model(artifact(variant, horizon, 'json'))
        settings[variant] = dict(threshold=threshold, columns=columns,
                                 train_positive=int(target.loc[index_sets['train']].sum()),
                                 train_n=len(index_sets['train']))
        for split, dates in index_sets.items():
            scores = (features.loc[dates, 'rain_3d_pct'].to_numpy() if model is None
                      else model.predict_proba(features.loc[dates, columns])[:,1])
            alarms = scores > threshold if model is None else scores >= threshold
            predictions.append(pd.DataFrame(dict(date=dates, horizon=horizon, split=split, variant=variant,
                                                  target=target.loc[dates].to_numpy(), score=scores,
                                                  alarm=alarms.astype(int))))
    predictions = pd.concat(predictions, ignore_index=True)
    predictions.to_csv(artifact('predictions', horizon), index=False)
    pd.concat(tuning).to_csv(artifact('threshold_tuning', horizon), index=False)
    metrics, detected = evaluate_predictions(predictions, events, horizon)
    metrics.to_csv(artifact('metrics', horizon), index=False)
    detected.to_csv(artifact('event_leads', horizon), index=False)
    metadata = dict(horizon=horizon, target='event onset in t+1 through t+h',
                    ongoing_flood_days_excluded=True, input_hash=input_hash(),
                    train_start=c.TRAIN_START, train_end=c.TRAIN_END, test_start=c.TEST_START,
                    test_end=c.TEST_END, separate_case_start=c.CASE_REPORT_START,
                    separate_case_end=c.CASE_REPORT_END, feature_reference_end=c.TRAIN_END,
                    validation_folds=c.VALIDATION_FOLDS, xgb_params=c.XGB_PARAMS,
                    primary_variant=c.PRIMARY_MODEL_VARIANT, models=settings,
                    score_note='Class-weighted uncalibrated scores, not calibrated flood probabilities',
                    oni_policy='Diagnostic ablation only; primary excludes ONI to avoid test-driven selection')
    artifact('metadata', horizon, 'json').write_text(json.dumps(metadata, indent=2))
    save_plots(horizon, predictions, features, source)
    summarize_all()
    print(f'\n{horizon}-day held-out results (September reported separately):\n{metrics.to_string(index=False)}')
    print(f'Run stage3_check.py --horizon {horizon}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon', type=int, choices=c.TARGET_HORIZONS, default=c.TARGET_HORIZONS[0])
    args = parser.parse_args()
    try:
        main(args.horizon)
    except (OSError, ValueError, KeyError, ImportError, AssertionError) as error:
        print(f'[FAIL] {error}', file=sys.stderr)
        sys.exit(1)
