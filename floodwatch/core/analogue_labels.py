"""Distance-based labels shared by saved records and dashboard views."""
import math
from floodwatch.core.analogue_config import RESEARCH_ANALOGUE_DISPLAY_ENABLED, OPERATIONAL_ANALOGUE_DISPLAY_ENABLED
import pandas as pd
from floodwatch.core.analogue_config import RESEARCH_CLOSE_MATCH_CUTOFF, OPERATIONAL_CLOSE_MATCH_CUTOFF


def label(distance,event_onset,lead,kind):
    cutoff={'research':RESEARCH_CLOSE_MATCH_CUTOFF,'operational':OPERATIONAL_CLOSE_MATCH_CUTOFF}[kind]
    valid=distance is not None and pd.notna(distance) and math.isfinite(float(distance)) and pd.notna(event_onset) and event_onset is not None
    close=bool(valid and float(distance)<=cutoff)
    date=str(pd.Timestamp(event_onset).date()) if valid else None
    enabled={'research':RESEARCH_ANALOGUE_DISPLAY_ENABLED,'operational':OPERATIONAL_ANALOGUE_DISPLAY_ENABLED}[kind]
    return {'display_enabled':enabled,'distance':float(distance) if valid else None,'close_match_cutoff':cutoff,'is_close_match':close,
            'label':'Pattern matching not reliable enough to display (see README)' if not enabled else f'Resembles the run-up to the {date} flood ({int(lead)} days before onset)' if close else 'No close match to any past flood run-up',
            'nearest_note':f'nearest: {date}, not a close match' if enabled and valid and not close else None}


def annotate_replay(replay):
    result=replay.copy()
    labels=pd.DataFrame([label(row.analogue_euclidean_distance,row.analogue_event_onset,row.analogue_days_before_flood,'research') for row in result.itertuples()],index=result.index)
    for column in labels:result[f'analogue_{column}']=labels[column]
    return result
