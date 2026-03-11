"""
Agent tool to retrieve and summarize F1 practice session data using FastF1.
Fetches F1 practice session data using the FastF1 library.
Returns top driver lap times, pace deltas, and tyre info.
"""

from __future__ import annotations
import warnings
import fastf1
import pandas as pd
from langchain_core.tools import tool
from config import CACHE_DIR, CURRENT_SEASON

# Suppress FastF1 non-critical warnings
warnings.filterwarnings("ignore", category=FutureWarning)
fastf1.Cache.enable_cache(CACHE_DIR)


@tool
def get_practice_session_data(round_number: int, session: str = "FP2") -> dict:
    """
    Fetch F1 practice session lap time data for a given race round.

    Args:
        round_number: Race round number on the 2026 calendar (1-24).
        session:      Session identifier — 'FP1', 'FP2', or 'FP3'.
                      Defaults to 'FP2' as it's most representative.

    Returns:
        dict with top 10 drivers by fastest lap, their lap times,
        tyre compounds used, and pace delta to the fastest driver.
    """
    try:
        event = fastf1.get_event(CURRENT_SEASON, round_number)
        sess  = fastf1.get_session(CURRENT_SEASON, round_number, session)

        print(f" Loading {event['EventName']} {session} data")
        sess.load(telemetry=False, weather=False, messages=False)

        laps = sess.laps.pick_quicklaps()
        if laps.empty:
            return {"error": f"No valid lap data found for Round {round_number} {session}."}

        # Best lap per driver
        best = (
            laps.groupby("Driver")["LapTime"]
            .min()
            .reset_index()
            .sort_values("LapTime")
        )

        # Tyre compound for each driver's fastest lap
        tyre_map = {}
        for driver in best["Driver"]:
            drv_laps = laps[laps["Driver"] == driver]
            fastest  = drv_laps.loc[drv_laps["LapTime"].idxmin()]
            tyre_map[driver] = fastest.get("Compound", "UNKNOWN")

        # Build output rows
        fastest_time = best.iloc[0]["LapTime"].total_seconds()
        rows = []
        for _, row in best.head(10).iterrows():
            lap_sec   = row["LapTime"].total_seconds()
            delta_sec = round(lap_sec - fastest_time, 3)
            rows.append({
                "position": len(rows) + 1,
                "driver":   row["Driver"],
                "lap_time": _fmt_time(lap_sec),
                "lap_time_s": round(lap_sec, 3),
                "delta":    f"+{delta_sec}s" if delta_sec > 0 else "FASTEST",
                "tyre":     tyre_map.get(row["Driver"], "UNKNOWN"),
            })

        return {
            "round":      round_number,
            "session":    session,
            "event_name": event["EventName"],
            "circuit":    event["Location"],
            "season":     CURRENT_SEASON,
            "top_10":     rows,
            "summary":    _summarise(rows, event["EventName"], session),
        }

    except Exception as exc:
        return {"error": f"FastF1 data fetch failed: {exc}"}


# helper functions

def _fmt_time(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins}:{secs:06.3f}"


def _summarise(rows: list, event: str, session: str) -> str:
    if not rows:
        return "No data available."
    p1 = rows[0]
    p2 = rows[1] if len(rows) > 1 else None

    summary = (
        f"{event} {session}: {p1['driver']} leads on "
        f"{p1['tyre']} tyres ({p1['lap_time']})"
    )
    if p2:
        summary += f", with {p2['driver']} {p2['delta']} behind on {p2['tyre']}s."
    return summary