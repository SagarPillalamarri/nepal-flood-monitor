"""Read saved Stage 3 models and outputs; never train or tune anything."""
import hashlib
import json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, DMatrix
import config as c
from features import tcwv_percentile
from stage3_models import artifact


def protected_hashes():
    paths = [artifact('metadata', h, 'json') for h in c.TARGET_HORIZONS]
    paths += [artifact(v, h, 'json') for h in c.TARGET_HORIZONS for v in c.MODEL_VARIANTS]
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    before = protected_hashes()
    c.STAGE3_DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(c.STAGE3_FEATURES, parse_dates=['date']).set_index('date')
    dates = pd.date_range(c.DIAGNOSTIC_START,c.DIAGNOSTIC_END,name='date')
    risk = pd.DataFrame(index=dates)
    for horizon in c.TARGET_HORIZONS:
        meta = json.loads(artifact('metadata',horizon,'json').read_text())
        predictions = pd.read_csv(artifact('predictions',horizon),parse_dates=['date'])
        predictions = predictions[predictions.variant.eq(c.PRIMARY_MODEL_VARIANT)]
        # Include separately reported September so this is the complete held-out period.
        pool = predictions[predictions.split.isin(['test','case'])]
        main_pool = predictions[predictions.split.eq('test')]
        scored = predictions.set_index('date').loc[dates,'score']
        risk[f'risk_{horizon}d'] = scored
        risk[f'threshold_{horizon}d'] = meta['models'][c.PRIMARY_MODEL_VARIANT]['threshold']
        risk[f'score_pct_{horizon}d'] = [100*((pool.score < value).sum()+.5*(pool.score == value).sum())/len(pool) for value in scored]
        risk[f'reference_n_{horizon}d'] = len(pool)
        risk[f'main_test_only_pct_{horizon}d'] = [100*((main_pool.score < value).sum()+.5*(main_pool.score == value).sum())/len(main_pool) for value in scored]
    risk.to_csv(c.STAGE3_DIAGNOSTIC_DIR/'september_scores.csv')
    percentiles = features.loc[dates,[name for name in features.columns if name.endswith('_pct')]].copy()
    percentiles.to_csv(c.STAGE3_DIAGNOSTIC_DIR/'input_percentile_features.csv')
    features.loc[dates].to_csv(c.STAGE3_DIAGNOSTIC_DIR/'all_input_features_actual_units.csv')
    # For non-percentile model inputs, provide a labelled diagnostic-only ranking.
    # These are not substituted into the saved model.
    display_pct = features.loc[dates].copy()
    train = features.loc[c.TRAIN_START:c.TRAIN_END]
    for name in features.columns:
        if not name.endswith('_pct'):
            display_pct[name] = tcwv_percentile(features.loc[dates,name],train[name])
    display_pct.columns = [f'{name}_display_pct' for name in display_pct.columns]
    display_pct.to_csv(c.STAGE3_DIAGNOSTIC_DIR/'all_inputs_display_percentiles.csv')
    meta = json.loads(artifact('metadata',1,'json').read_text())
    columns = meta['models'][c.PRIMARY_MODEL_VARIANT]['columns']
    model = XGBClassifier()
    model.load_model(artifact(c.PRIMARY_MODEL_VARIANT,1,'json'))
    contribution_dates = pd.DatetimeIndex(c.DIAGNOSTIC_CONTRIBUTION_DATES)
    matrix = DMatrix(features.loc[contribution_dates,columns],missing=np.nan)
    contributions = model.get_booster().predict(matrix,pred_contribs=True)
    score = model.predict_proba(features.loc[contribution_dates,columns])[:,1]
    np.testing.assert_allclose(1/(1+np.exp(-contributions.sum(axis=1))),score,rtol=1e-5)
    records=[]
    for i,date in enumerate(contribution_dates):
        for j,name in enumerate(columns):
            value=float(contributions[i,j])
            records.append(dict(date=date,feature=name,actual_input=float(features.loc[date,name]),
                                contribution_log_odds=value,direction='up' if value > 0 else 'down' if value < 0 else 'neutral',
                                bias_log_odds=float(contributions[i,-1]),score=float(score[i])))
    all_contributions=pd.DataFrame(records)
    all_contributions.to_csv(c.STAGE3_DIAGNOSTIC_DIR/'all_contributions_1d.csv',index=False)
    top=all_contributions.assign(magnitude=lambda x:x.contribution_log_odds.abs()).sort_values('magnitude',ascending=False).groupby('date',sort=True).head(c.DIAGNOSTIC_TOP_FEATURES).sort_values(['date','magnitude'],ascending=[True,False])
    top.to_csv(c.STAGE3_DIAGNOSTIC_DIR/'top_contributions_1d.csv',index=False)
    events=pd.read_csv(c.STAGE2_EVENTS,parse_dates=['start','end','peak_date'])
    case=events[(events.start <= pd.Timestamp(c.CASE_REPORT_END)) & (events.end >= pd.Timestamp(c.CASE_REPORT_START))]
    label_metadata=json.loads(c.STAGE2_METADATA.read_text())
    leads=pd.read_csv(artifact('event_leads',1),parse_dates=['onset'])
    leads=leads[leads.variant.eq(c.PRIMARY_MODEL_VARIANT)&leads.split.eq('test')]
    leads[['event_id','onset','detected','lead_days']].to_csv(c.STAGE3_DIAGNOSTIC_DIR/'test_event_detections_1d.csv',index=False)
    if before != protected_hashes():
        raise ValueError('Saved model weights or thresholds changed')
    (c.STAGE3_DIAGNOSTIC_DIR/'protected_hashes.json').write_text(json.dumps(before,indent=2))
    print('DAILY SCORES (rank reference includes all eligible 2020–2025 main-test + separate-case dates):')
    print(risk.round(6).to_string())
    print('\nMODEL PERCENTILE INPUTS:')
    print(percentiles.round(2).to_string())
    print('\nALL PRIMARY MODEL INPUTS EXPRESSED AS DISPLAY-ONLY SEASONAL PERCENTILES:')
    print('Raw soil, discharge, rate-of-change, monsoon-day and ONI values are ranked against their 2001–2019 seasonal distributions for readability; the saved model still receives its original feature values.')
    print(display_pct.round(2).to_string())
    print('\nTOP 5 ABSOLUTE FEATURE CONTRIBUTIONS (signed log-odds, not probability points):')
    contribution_display = top[['date','feature','actual_input','contribution_log_odds','direction']].copy()
    contribution_display['actual_input'] = contribution_display.actual_input.round(6)
    contribution_display['contribution_log_odds'] = contribution_display.contribution_log_odds.round(6)
    print(contribution_display.to_string(index=False))
    print('\nCase onset:',case.start.dt.strftime('%Y-%m-%d').tolist())
    print('GloFAS flood threshold (m³/s):',label_metadata['threshold'])
    print('\nMAIN TEST EVENTS, 1-DAY PRIMARY MODEL:')
    print(leads[['event_id','onset','detected','lead_days']].to_string(index=False))
    print('\nPASS: Saved weights and thresholds unchanged; contribution sums reproduce the saved model scores.')


if __name__ == '__main__':
    main()
