"""Bounded retrospective practice adapter, separate from forecast features."""
from dataclasses import asdict
from statistics import median
from langchain_core.tools import tool
from data.contracts import EventKey
from data.fastf1_source import load_session
from ml.features import clean_laps


@tool
def get_practice_session_data(season:int,round_number:int,session:str='FP2')->dict:
    """Read a cached practice session as hindsight; never infer a current entry list."""
    if session not in {'FP1','FP2','FP3'}:return {'status':'error','errors':['Practice session required']}
    loaded=load_session(EventKey(season,round_number),session)
    clean=clean_laps(loaded.laps)
    return dict(status=loaded.status.value,event_id=loaded.event.event_id,session=session,
                quality_flags=loaded.quality_flags,source=asdict(loaded.source),
                analysis_kind='hindsight_only',errors=loaded.errors,
                drivers=[dict(driver_id=entry['driver_id'],sample_count=len(laps),
                              median_clean_lap_s=median(row['lap_time_s'] for row in laps) if laps else None)
                         for entry in loaded.entries for laps in [[row for row in clean if row['driver_id']==entry['driver_id']]]])
