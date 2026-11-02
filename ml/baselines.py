"""Deterministic baselines that preserve event and entry identities."""

from __future__ import annotations

from statistics import mean

from ml.evaluation import _eligible, _event_groups, _missing, _number, _records, _timestamp


def grid_baseline(entries):
    """Verified starting grid; an unknown grid is unavailable, never fabricated."""
    results = []
    for row in _records(entries):
        score = row.get("grid_position")
        row.update(prediction_score=score, baseline_source="grid", score_units="ordinal_position",
                   baseline_status="unavailable" if _missing(score) else "observed")
        results.append(row)
    return results


def qualifying_baseline(entries):
    """Observed qualifying order; it is not a verified final-race starting grid."""
    results = []
    for row in _records(entries):
        score = row.get('qualifying_position')
        row.update(prediction_score=score, baseline_source='qualifying', score_units='ordinal_position',
                   baseline_status='unavailable' if _missing(score) else 'observed')
        results.append(row)
    return results


def weekend_form_baseline(features, records, *, as_of_utc):
    """Explicit research heuristic: 60% qualifying, 30% practice, 10% sprint.

    Available signal weights are renormalized. No learned accuracy or win odds.
    """
    rows = _records(features)
    pace = sorted((row['practice_pace_delta_s'], row['driver_id']) for row in rows
                  if row.get('practice_pace_delta_s') is not None)
    pace_ranks = {driver: 1 + sum(value < score for value, _ in pace) for score, driver in pace}
    sprints = {}
    for record in records:
        if (record.session_id == 'S' and record.available_at_utc <= as_of_utc
                and record.source.available_by(as_of_utc)):
            value = record.values.get('sprint_position')
            if value is not None:
                sprints[record.driver_id] = _number(value, 'sprint_position', allow_missing=False)
    for row in rows:
        signals = [(0.6, row.get('qualifying_position') or row.get('grid_position')),
                   (0.3, pace_ranks.get(row['driver_id'])), (0.1, sprints.get(row['driver_id']))]
        present = [(weight, value) for weight, value in signals if value is not None]
        score = sum(weight * value for weight, value in present) / sum(weight for weight, _ in present) if present else None
        row.update(prediction_score=score, baseline_source='weekend_form',
                   score_units='ordinal_position', baseline_status='research_heuristic')
    return rows


def rolling_form_baseline(history, entries, *, as_of_utc, window=5, prior_position=10.5):
    """Past-only mean finish position: driver history, then team history, then prior.

    window counts the most recent eligible races for each driver/team. The
    configurable prior_position is an estimated ordinal prior, not observed
    rookie performance. No ranking score is converted into a win probability.
    """
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("window must be a positive integer")
    prior = _number(prior_position, "prior_position", allow_missing=False)
    if prior < 1:
        raise ValueError("prior_position must be at least 1")
    cutoff = _timestamp(as_of_utc, "as_of_utc")
    entries = _records(entries)
    target_ids = {row["event_id"] for row in entries}
    events = [event for event in _event_groups(history) if event["start"] < cutoff and
              event["publication"] is not None and event["publication"] < cutoff and
              event["completed"] and event["event_id"] not in target_ids]
    results = []
    for row in entries:
        driver_form, team_form = [], []
        for event in events:
            eligible, _ = _eligible(event["rows"])
            driver = [past["actual_position"] for past in eligible if past["driver_id"] == row["driver_id"]]
            team = [past["actual_position"] for past in eligible if not _missing(row.get("team_id")) and
                    past.get("team_id") == row["team_id"]]
            if driver:
                driver_form.append(mean(driver))
            if team:
                team_form.append(mean(team))
        available = driver_form or team_form
        score = mean(available[-window:]) if available else prior
        source = "driver_form" if driver_form else "team_form" if team_form else "prior"
        row.update(prediction_score=score, baseline_source=source, baseline_status="estimated",
                   score_units="ordinal_position", n_form_events=min(window, len(available)))
        results.append(row)
    return results
