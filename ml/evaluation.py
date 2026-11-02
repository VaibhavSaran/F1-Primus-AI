"""Offline, whole-race evaluation; no tracking or provider side effects."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isclose, isfinite, isnan, log, log2, sqrt
from numbers import Real
from random import Random
from statistics import mean


PROTOCOL_VERSION = "f1-race-macro-v1"
METRICS = (
    "rank_spearman", "position_mae", "ndcg_at_3", "podium_recall",
    "top10_overlap", "winner_accuracy", "winner_top3_hit",
    "winner_log_loss", "winner_brier", "pace_mae_seconds", "pace_rmse_seconds",
)


def _protocol(higher_score_better=False):
    return dict(version=PROTOCOL_VERSION, aggregation="unweighted race macro",
                relevance={1: 3, 2: 2, 3: 1}, ndcg_gain="linear",
                probability_clip=1e-15, probability_sum_tolerance=1e-8,
                brier="sum over eligible entries", higher_score_better=higher_score_better,
                ties="average ranks for Spearman; driver_id for ordinal/top-k ranks",
                exclusions="DNS, DSQ, unclassified DNF; retain classified DNF official positions",
                missing="None; exclude unavailable metric from its race denominator",
                position_mae="eligible predicted ordinal rank versus official position",
                top10_denominator="min(10, eligible field size)",
                bootstrap="percentile 95% CI, resample whole races")


def _missing(value):
    return value is None or (isinstance(value, Real) and isnan(value))


def _number(value, name, *, positive_integer=False, allow_missing=True):
    if _missing(value) and allow_missing:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric")
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive_integer and (value < 1 or int(value) != value):
        raise ValueError(f"{name} must be a positive integer")
    return float(value)


def _identifier(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _records(rows):
    if hasattr(rows, "to_dict") and not isinstance(rows, Mapping):
        rows = rows.to_dict("records")
    if isinstance(rows, Mapping):
        rows = [rows]
    records, seen = [], set()
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise TypeError("rows must be mappings or a pandas DataFrame")
        row = dict(raw)
        key = tuple(_identifier(row.get(name), name) for name in ("event_id", "driver_id"))
        if key in seen:
            raise ValueError(f"duplicate event/driver identity: {key}")
        seen.add(key)
        for name in ("actual_position", "predicted_rank", "grid_position"):
            if name in row:
                row[name] = _number(row[name], name, positive_integer=True)
        for name in ("prediction_score", "actual_pace_s", "predicted_pace_s"):
            if name in row:
                row[name] = _number(row[name], name)
        if "winner_probability" in row and row["winner_probability"] is not None:
            value = _number(row["winner_probability"], "winner_probability", allow_missing=False)
            if not 0 <= value <= 1:
                raise ValueError("winner_probability must be between 0 and 1")
            row["winner_probability"] = value
        if not _missing(row.get("team_id")):
            _identifier(row["team_id"], "team_id")
        status = row.get("status", "")
        if _missing(status):
            status = ""
        if not isinstance(status, str):
            raise TypeError("status must be a string")
        row["status"] = status.strip().upper()
        outcome = row.get("outcome_status", "completed")
        if _missing(outcome):
            outcome = "completed"
        if outcome not in {"pending", "completed", "official", "provisional"}:
            raise ValueError("invalid outcome_status")
        row["outcome_status"] = outcome
        records.append(row)
    return records


def _eligible(rows):
    kept, exclusions = [], {"DNS": 0, "DSQ": 0, "unclassified": 0}
    for row in rows:
        if row["status"] in {"DNS", "DSQ"}:
            exclusions[row["status"]] += 1
        elif row["status"] == "DNF" and _missing(row.get("actual_position")):
            exclusions["unclassified"] += 1
        else:
            kept.append(row)
    return kept, exclusions


def _completed(rows):
    return bool(rows) and all(row["outcome_status"] != "pending" and
                              not _missing(row.get("actual_position")) for row in rows)


def _average_ranks(values):
    ranks = [0.0] * len(values)
    indexes = sorted(range(len(values)), key=values.__getitem__)
    start = 0
    while start < len(indexes):
        end = start + 1
        while end < len(indexes) and values[indexes[end]] == values[indexes[start]]:
            end += 1
        for index in indexes[start:end]:
            ranks[index] = (start + 1 + end) / 2
        start = end
    return ranks


def _spearman(actual, predicted):
    if len(actual) < 2:
        return None
    left, right = _average_ranks(actual), _average_ranks(predicted)
    left = [value - mean(left) for value in left]
    right = [value - mean(right) for value in right]
    denominator = sqrt(sum(value * value for value in left) *
                       sum(value * value for value in right))
    return sum(a * b for a, b in zip(left, right)) / denominator if denominator else None


def evaluate_race(rows, *, higher_score_better=False):
    """Score one complete race field. Missing labels/predictions stay unavailable.

    Input rows carry event_id, driver_id, actual_position and prediction_score
    (lower is better) or predicted_rank. Optional winner probabilities must form
    a coherent distribution over the eligible field. Pace deltas are seconds.
    The caller supplies the complete common field for all compared models.
    """
    rows = _records(rows)
    event_ids = {row["event_id"] for row in rows}
    if len(event_ids) > 1:
        raise ValueError("evaluate_race accepts one event")
    eligible, exclusions = _eligible(rows)
    metrics = dict.fromkeys(METRICS)
    positions = [row["actual_position"] for row in eligible
                 if not _missing(row.get("actual_position"))]
    if len(set(positions)) != len(positions):
        raise ValueError("official actual positions must be unique within a race")
    explicitly_pending = any(row["outcome_status"] == "pending" for row in rows)
    complete = _completed(eligible) and not explicitly_pending
    probabilities = [row.get("winner_probability") for row in eligible]
    has_distribution = bool(eligible) and all(value is not None for value in probabilities)
    if has_distribution and not isclose(sum(probabilities), 1.0, rel_tol=0, abs_tol=1e-8):
        raise ValueError("eligible winner probabilities must sum to 1")
    # A score column takes precedence; do not fill individual missing scores with ranks.
    score_key = "prediction_score" if any("prediction_score" in row for row in eligible) else "predicted_rank"
    values = [row.get(score_key) for row in eligible]
    has_ranking = bool(eligible) and all(not _missing(value) for value in values)
    ordered = []
    if has_ranking:
        sign = -1 if score_key == "prediction_score" and higher_score_better else 1
        ordered = sorted(eligible, key=lambda row: (sign * row[score_key], row["driver_id"]))
    if complete and has_ranking:
        predicted = {row["driver_id"]: index for index, row in enumerate(ordered, 1)}
        metrics["position_mae"] = mean(abs(predicted[row["driver_id"]] - row["actual_position"])
                                       for row in eligible)
        metrics["rank_spearman"] = _spearman(
            [row["actual_position"] for row in eligible], [sign * value for value in values])
        actual_order = sorted(eligible, key=lambda row: row["actual_position"])
        top_n = min(10, len(eligible))
        actual_top = {row["driver_id"] for row in actual_order[:top_n]}
        metrics["top10_overlap"] = len(actual_top & {row["driver_id"] for row in ordered[:top_n]}) / top_n
        winner = next((row["driver_id"] for row in eligible if row["actual_position"] == 1), None)
        if winner is not None:
            metrics["winner_accuracy"] = float(ordered[0]["driver_id"] == winner)
            metrics["winner_top3_hit"] = float(winner in {row["driver_id"] for row in ordered[:3]})
        podium = {row["driver_id"] for row in eligible if row["actual_position"] <= 3}
        if {1, 2, 3}.issubset(positions):
            metrics["podium_recall"] = len(podium & {row["driver_id"] for row in ordered[:3]}) / 3
            dcg = sum(max(0, 4 - row["actual_position"]) / log2(index + 2)
                      for index, row in enumerate(ordered[:3]))
            metrics["ndcg_at_3"] = dcg / (3 + 2 / log2(3) + 1 / log2(4))
    pace_errors = []
    if complete:
        if has_distribution and 1 in positions:
            winner_index = positions.index(1)
            metrics["winner_log_loss"] = -log(max(probabilities[winner_index], 1e-15))
            metrics["winner_brier"] = sum((p - float(row["actual_position"] == 1)) ** 2
                                          for p, row in zip(probabilities, eligible))
    if not explicitly_pending:
        pace_errors = [row["predicted_pace_s"] - row["actual_pace_s"] for row in eligible
                       if not _missing(row.get("actual_pace_s")) and
                       not _missing(row.get("predicted_pace_s"))]
        if pace_errors:
            metrics["pace_mae_seconds"] = mean(abs(value) for value in pace_errors)
            metrics["pace_rmse_seconds"] = sqrt(mean(value * value for value in pace_errors))
    return dict(event_id=next(iter(event_ids), None), metrics=metrics, protocol=_protocol(higher_score_better),
                n_entries=len(rows), n_eligible=len(eligible), exclusions=exclusions,
                n_pace_pairs=len(pace_errors), predicted_order=[row["driver_id"] for row in ordered],
                outcome_status="completed" if complete else "pending")


def _percentile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def evaluate_predictions(rows, *, higher_score_better=False, bootstrap_samples=0, seed=0):
    """Unweighted race means with metric-specific denominators and optional race CIs."""
    if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int) or bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be a nonnegative integer")
    grouped = defaultdict(list)
    for row in _records(rows):
        grouped[row["event_id"]].append(row)
    races = [evaluate_race(grouped[key], higher_score_better=higher_score_better) for key in sorted(grouped)]
    metrics, counts, intervals = {}, {}, {}
    rng = Random(seed)
    samples = [rng.choices(races, k=len(races)) for _ in range(bootstrap_samples)] if races else []
    for metric in METRICS:
        scores = [race["metrics"][metric] for race in races if race["metrics"][metric] is not None]
        metrics[metric] = mean(scores) if scores else None
        counts[metric] = len(scores)
        intervals[metric] = None
        if len(scores) >= 2 and samples:
            means = []
            for sample in samples:
                available = [race["metrics"][metric] for race in sample if race["metrics"][metric] is not None]
                if available:
                    means.append(mean(available))
            intervals[metric] = [_percentile(means, 0.025), _percentile(means, 0.975)]
    return dict(metrics=metrics, metric_event_counts=counts, per_race=races, n_events=len(races),
                confidence_intervals=intervals, bootstrap_samples=bootstrap_samples, seed=seed,
                protocol=_protocol(higher_score_better))


def _timestamp(value, name):
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"{name} must be a valid UTC-aware timestamp") from exc
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _event_groups(rows):
    groups = defaultdict(list)
    for row in _records(rows):
        groups[row["event_id"]].append(row)
    events = []
    for event_id, records in groups.items():
        starts = {_timestamp(row.get("event_time_utc"), "event_time_utc") for row in records}
        cutoffs = {_timestamp(row.get("forecast_cutoff_utc") if not _missing(row.get("forecast_cutoff_utc"))
                              else row["event_time_utc"], "forecast_cutoff_utc") for row in records}
        if len(starts) != 1 or len(cutoffs) != 1:
            raise ValueError("event timestamps must agree across all driver rows")
        start, cutoff = starts.pop(), cutoffs.pop()
        if cutoff > start:
            raise ValueError("forecast cutoff must be at or before race start")
        publication = None
        times = [_timestamp(row["outcome_available_at_utc"], "outcome_available_at_utc")
                 for row in records if not _missing(row.get("outcome_available_at_utc"))]
        if len(times) == len(records):
            publication = max(times)
            if publication < start:
                raise ValueError("race outcome cannot be published before race start")
        eligible, _ = _eligible(records)
        events.append(dict(event_id=event_id, start=start, cutoff=cutoff, publication=publication,
                           completed=_completed(eligible) and all(row["outcome_status"] != "pending" for row in records),
                           rows=records))
    return sorted(events, key=lambda event: (event["start"], event["event_id"]))


@dataclass(frozen=True)
class WalkForwardFold:
    train_event_ids: tuple[str, ...]
    test_event_ids: tuple[str, ...]
    train_rows: tuple[dict, ...]
    test_rows: tuple[dict, ...]


def walk_forward_folds(rows, *, min_train_events=12):
    """Whole-race expanding folds, with no preprocessing fitted outside each fold.

    Required event_time_utc is the race start. Heldout forecast_cutoff_utc defaults
    to start; training requires known outcome_available_at_utc strictly before
    that cutoff. Unknown publication or incomplete labels never enter training.
    """
    if isinstance(min_train_events, bool) or not isinstance(min_train_events, int) or min_train_events < 1:
        raise ValueError("min_train_events must be a positive integer")
    events, folds = _event_groups(rows), []
    for test in events:
        if not test["completed"]:
            continue
        train = [event for event in events if event["start"] < test["cutoff"] and
                 event["completed"] and event["publication"] is not None and
                 event["publication"] < test["cutoff"]]
        if len(train) >= min_train_events:
            folds.append(WalkForwardFold(tuple(event["event_id"] for event in train), (test["event_id"],),
                                         tuple(dict(row) for event in train for row in event["rows"]),
                                         tuple(dict(row) for row in test["rows"])))
    return folds
