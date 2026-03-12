"""
Prefect Pre Race Pipeline
Prefect flow — schedules and orchestrates the F1 Primus AI
pre-race pipeline. Runs the full LangGraph agent for a given
round and saves the report to disk.
"""

from __future__ import annotations
import os
import json
from datetime import datetime, timedelta
from pathlib import Path

from prefect import flow, task, get_run_logger

from agents.graph import run_pre_race_agent
from config import F1_CALENDAR


# Output directory 
REPORTS_DIR = Path("data") / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# Tasks 

@task(name="validate-round", retries=1)
def validate_round(round_number: int) -> dict:
    """Confirm the round exists in the 2026 calendar."""
    logger = get_run_logger()
    race_info = F1_CALENDAR.get(round_number)
    if not race_info:
        raise ValueError(f"Round {round_number} not found in 2026 calendar.")
    race_name, location, country, race_date = race_info
    logger.info(f"Validated: Round {round_number} — {race_name} @ {location}")
    return {
        "round_number": round_number,
        "race_name":    race_name,
        "location":     location,
        "country":      country,
        "race_date":    race_date,
    }


@task(name="run-agent", retries=1, retry_delay_seconds=30)
def run_agent(round_info: dict) -> dict:
    """Run the full LangGraph agent pipeline for the round."""
    logger = get_run_logger()
    round_number = round_info["round_number"]
    race_name    = round_info["race_name"]

    logger.info(f"Starting agent for {race_name} (Round {round_number})")
    result = run_pre_race_agent(round_number=round_number)

    if "error" in result:
        raise RuntimeError(f"Agent failed: {result['error']}")

    logger.info(f"Agent complete — {result['message_count']} messages exchanged")
    return result


@task(name="save-report")
def save_report(result: dict) -> Path:
    """Save the final report and tool outputs to disk."""
    logger     = get_run_logger()
    race_name  = result["race_name"].replace(" ", "_")
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename   = f"Round_{result['round']}_{race_name}_{timestamp}"

    # Save markdown report
    report_path = REPORTS_DIR / f"{filename}.md"
    report_path.write_text(result["report"], encoding="utf-8")

    # Save raw tool outputs as JSON
    json_path = REPORTS_DIR / f"{filename}_tools.json"
    json_path.write_text(
        json.dumps(result["tool_outputs"], indent=2, default=str),
        encoding="utf-8"
    )

    logger.info(f"Report saved: {report_path}")
    logger.info(f"Tool data saved: {json_path}")
    return report_path


# Main Flow 

@flow(
    name="f1-pre-race-pipeline",
    description="Autonomous pre-race intelligence pipeline for F1 2026",
)
def pre_race_pipeline(round_number: int) -> dict:
    """
    Full pre-race pipeline for a single F1 round.

    Args:
        round_number: 2026 F1 calendar round (1-24).

    Returns:
        dict with report path and metadata.
    """
    logger = get_run_logger()
    logger.info(f"F1 Primus AI pipeline starting — Round {round_number}")

    race_info   = validate_round(round_number)
    result      = run_agent(race_info)
    report_path = save_report(result)

    logger.info(f"Pipeline complete for {race_info['race_name']}")

    return {
        "round":       round_number,
        "race_name":   race_info["race_name"],
        "report_path": str(report_path),
        "messages":    result["message_count"],
        "status":      "success",
    }


# Auto-schedule helper 

def get_next_race_round() -> int | None:
    """Return the round number of the next upcoming race."""
    today = datetime.today().date()
    for round_num, (name, location, country, date_str) in F1_CALENDAR.items():
        race_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        if race_date >= today:
            return round_num
    return None


# Entry point 

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="F1 Primus AI — Pre-race pipeline")
    parser.add_argument(
        "--round", type=int, default=None,
        help="Round number to analyse (default: next upcoming race)"
    )
    args = parser.parse_args()

    round_number = args.round or get_next_race_round()

    if round_number is None:
        print("No upcoming races found in the 2026 calendar.")
    else:
        result = pre_race_pipeline(round_number=round_number)
        print(f"\nDone — report saved to: {result['report_path']}")