"""Assemble the approved September 2024 dashboard replay from saved stage outputs."""
import json
import sys
import pandas as pd
import config as c


def prepare_replay():
    c.STAGE5_DIR.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range('2024-09-01', '2024-09-30', name='date')
    daily = pd.read_csv(c.STAGE2_DAILY, parse_dates=['date']).set_index('date')
    features = pd.read_csv(c.STAGE3_FEATURES, parse_dates=['date']).set_index('date')
    replay = daily[['precip_mm', 'tcwv_mm', 'sm_0_7', c.GLOFAS_CHOSEN, 'flood']].reindex(dates)
    replay = replay.rename(columns={c.GLOFAS_CHOSEN: 'discharge_m3_s'})
    replay['tcwv_pct'] = features.tcwv_pct.reindex(dates)
    replay['sm_0_7_pct'] = features.sm_0_7_pct.reindex(dates)

    selected = json.loads(c.STAGE3_CALENDAR_SELECTED.read_text())
    for horizon in c.TARGET_HORIZONS:
        chosen = next(row for row in selected['horizons'] if int(row['horizon']) == horizon)
        predictions = pd.read_csv(c.STAGE3_CALENDAR_DIR / f'predictions_{horizon}d.csv', parse_dates=['date'])
        predictions = predictions[(predictions.variant == chosen['selected_variant'])
                                  & (predictions.split == 'case')].set_index('date')
        replay[f'risk_{horizon}d'] = predictions.score.reindex(dates)
        replay[f'threshold_{horizon}d'] = float(chosen['threshold'])
        alerts = (replay[f'risk_{horizon}d'] >= replay[f'threshold_{horizon}d']).astype('boolean')
        alerts.loc[replay[f'risk_{horizon}d'].isna()] = pd.NA
        replay[f'alert_{horizon}d'] = alerts
        replay[f'model_{horizon}d'] = chosen['selected_variant']

    analogues = pd.read_csv(c.STAGE4_MATCHES, parse_dates=['date','event_onset','event_peak','matched_window_end']).set_index('date')
    for column in ('closest_event_id','event_onset','event_peak','days_before_flood',
                   'matched_window_end','euclidean_distance','window_days'):
        replay[f'analogue_{column}'] = analogues[column].reindex(dates)
    replay['score_status'] = 'Eligible separate-case score'
    for date in dates:
        if pd.isna(replay.loc[date, 'risk_1d']):
            replay.loc[date, 'score_status'] = ('Ongoing flood day excluded from evaluation'
                                                if replay.loc[date, 'flood'] == 1
                                                else 'Outside eligible separate-case scoring rows')
    replay.index.name = 'date'
    replay.to_csv(c.STAGE5_REPLAY_FILE)
    replay.loc['2024-09-20':'2024-09-28', [
        'risk_1d','threshold_1d','alert_1d','risk_3d','threshold_3d','alert_3d',
        'analogue_closest_event_id','analogue_event_onset','analogue_days_before_flood',
        'analogue_euclidean_distance','score_status']].to_csv(c.STAGE5_CASE_TABLE)
    return replay


def main():
    replay = prepare_replay()
    print(f'Prepared September 2024 replay: {len(replay)} days')
    print(replay.loc['2024-09-20':'2024-09-28', [
        'precip_mm','tcwv_pct','sm_0_7','discharge_m3_s','risk_1d','threshold_1d',
        'risk_3d','threshold_3d','analogue_event_onset','analogue_days_before_flood',
        'score_status']].to_string())
    print(f'Saved {c.STAGE5_REPLAY_FILE}')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, StopIteration) as error:
        print(f'[FAIL] {error}', file=sys.stderr)
        sys.exit(1)
