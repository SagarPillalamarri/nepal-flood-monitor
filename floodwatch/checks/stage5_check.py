"""Check the assembled dashboard replay against saved model and analogue artifacts."""
import sys
import numpy as np
import pandas as pd
from floodwatch.core import config as c
from floodwatch.dashboard.stage5_replay import prepare_replay


def main():
    checks = []
    def report(status, message):
        checks.append({'status': status, 'message': message})
        print(f'[{status}] {message}')
    try:
        replay = prepare_replay()
        expected_dates = pd.date_range('2024-09-01','2024-09-30',name='date')
        if not replay.index.equals(expected_dates):
            raise ValueError('Replay must contain every September date exactly once')
        required = ['precip_mm','tcwv_pct','sm_0_7','discharge_m3_s','flood',
                    'threshold_1d','threshold_3d','analogue_event_onset',
                    'analogue_days_before_flood']
        if replay[required].isna().any().any():
            raise ValueError('A required replay variable is missing')
        case_table = pd.read_csv(c.STAGE5_CASE_TABLE, parse_dates=['date','analogue_event_onset'],
                                 dtype={'alert_1d':'boolean','alert_3d':'boolean'}).set_index('date')
        expected_case = replay.loc['2024-09-20':'2024-09-28', case_table.columns]
        pd.testing.assert_frame_equal(case_table, expected_case, check_dtype=False, check_freq=False)
        report('PASS', 'Replay covers all 30 September dates with rainfall, TCWV percentile, soil moisture, discharge and flood labels')

        for horizon in c.TARGET_HORIZONS:
            selected = pd.read_csv(c.STAGE3_CALENDAR_DIR / f'predictions_{horizon}d.csv', parse_dates=['date'])
            selected_json = __import__('json').loads((c.STAGE3_CALENDAR_DIR / f'selection_{horizon}d.json').read_text())
            selected = selected[(selected.variant == selected_json['selected_variant']) & selected.split.eq('case')].set_index('date')
            dates = selected.index.intersection(replay.index)
            scored_dates = replay.index[replay[f'risk_{horizon}d'].notna()]
            if set(scored_dates) != set(dates):
                raise ValueError(f'{horizon}-day replay score coverage differs from saved case-eligible dates')
            np.testing.assert_allclose(replay.loc[dates, f'risk_{horizon}d'], selected.loc[dates, 'score'])
            if replay.loc[dates, f'alert_{horizon}d'].astype(int).ne(selected.loc[dates, 'alarm']).any():
                raise ValueError(f'{horizon}-day replay alerts differ from saved model output')
            if replay[f'threshold_{horizon}d'].nunique() != 1:
                raise ValueError(f'{horizon}-day threshold changes across replay')
        flood_dates = replay.index[replay.flood.eq(1)]
        if replay.loc[flood_dates, ['risk_1d','risk_3d']].notna().any().any():
            raise ValueError('An ongoing-flood date has a forecast score')
        report('PASS', 'Replay scores, selected thresholds and alerts match saved validation-selected model outputs; ongoing flood days are unscored')

        analogues = pd.read_csv(c.STAGE4_MATCHES, parse_dates=['date','event_onset']).set_index('date')
        np.testing.assert_array_equal(replay.analogue_closest_event_id.fillna(-1), analogues.loc[replay.index].closest_event_id.fillna(-1))
        if replay.analogue_event_onset.dt.year.ge(2024).any():
            raise ValueError('Replay uses a 2024 or later analogue template')
        report('PASS', 'Replay analogue event and lead fields match Stage 4 and use pre-2024 templates')
    except (OSError, ValueError, KeyError, AssertionError) as error:
        report('FAIL', str(error))
    c.STAGE5_REPORT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(checks).to_csv(c.STAGE5_REPORT, index=False)
    failures = sum(row['status'] == 'FAIL' for row in checks)
    print(f'\nSummary: {sum(row["status"] == "PASS" for row in checks)} PASS, {sum(row["status"] == "WARN" for row in checks)} WARN, {failures} FAIL')
    return int(failures > 0)


if __name__ == '__main__':
    sys.exit(main())
