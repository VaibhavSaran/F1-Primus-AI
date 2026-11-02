"""Retired action tool; chat cannot silently start training or forecast jobs."""
from langchain_core.tools import tool


@tool
def run_race_prediction(round_number:int)->dict:
    """Link callers to explicit verified forecast controls without executing work."""
    return dict(status='handoff',route='Predictions',round_number=round_number,
                message='Choose an event, verified snapshot, cutoff and model in Predictions to submit an authorized forecast.')
