"""
Agent Prediction Tool using ML model
LangChain tool that wraps the ML model and exposes it
to the LangGraph agent as a callable tool.
"""

from __future__ import annotations
from langchain_core.tools import tool
from ml.model import run_prediction
from config import F1_CALENDAR


@tool
def run_race_prediction(round_number: int) -> dict:
    """
    Run the ML prediction model for a given F1 race round
    and return a ranked list of predicted finishing positions.

    Args:
        round_number: Race round number on the 2026 calendar (1-24).

    Returns:
        dict with predicted podium, top 10 finishing order,
        model MAE in seconds, and MLflow run ID.
    """
    race_info = F1_CALENDAR.get(round_number)
    if not race_info:
        return {"error": f"Round {round_number} not found in 2026 calendar."}

    race_name, location, country, race_date = race_info

    try:
        result = run_prediction(
            round_number=round_number,
            race_name=race_name,
        )
        return result
    except Exception as exc:
        return {"error": f"Prediction model failed: {exc}"}