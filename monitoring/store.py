"""Trusted service writes; browser-facing reads belong in TrackingQueries."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from math import isfinite
from pathlib import Path
import re
import sqlite3
from uuid import NAMESPACE_URL, uuid5

from data.contracts import ForecastMode
from ml.evaluation import PROTOCOL_VERSION, evaluate_race


_PREDICTION_FIELDS = {"driver_id", "prediction_score", "predicted_rank", "winner_probability",
                      "predicted_pace_s", "grid_position", "team_id"}
_OUTCOME_FIELDS = {"driver_id", "actual_position", "status", "actual_pace_s"}


def _identifier(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:@-]{0,159}", value):
        raise ValueError(f"{name} must be a safe nonempty identifier")
    return value


def _timestamp(value, name):
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime) or value.utcoffset() is None or value.utcoffset().total_seconds() != 0:
        raise ValueError(f"{name} must be an aware UTC timestamp")
    return value.astimezone(timezone.utc).isoformat()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _number(value, name, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    if integer and int(value) != value:
        raise ValueError(f"{name} must be an integer")
    return value


def _rows(forecast, outcome):
    # Filter here too: older stored outcomes may predate ingestion validation.
    labels = {entry["driver_id"]: {key: value for key, value in entry.items() if key in _OUTCOME_FIELDS}
              for entry in outcome["entries"]} if outcome else {}
    # Complete fields must agree; evaluating a partial intersection would inflate coverage.
    complete_field = labels.keys() == {entry["driver_id"] for entry in forecast["ranking"]}
    return [{**{key: value for key, value in entry.items() if key in _PREDICTION_FIELDS},
             **labels.get(entry["driver_id"], {}), "event_id": forecast["event_id"],
             "outcome_status": outcome["status"] if outcome and complete_field else "pending"}
            for entry in forecast["ranking"]]


class TrackingStore:
    """SQLite canonical store with atomic immutable inserts and replay-safe scoring.

    Construct only with a trusted local configuration path. Write methods are
    internal service operations: callers must authorize identity/outcome ingestion.
    No path is derived from run, owner, event, or revision identifiers.
    """

    def __init__(self, path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            for table, key in (("forecasts", "run_id TEXT PRIMARY KEY"),
                               ("outcomes", "event_id TEXT, revision_id TEXT, PRIMARY KEY(event_id, revision_id)"),
                               ("scores", "run_id TEXT, revision_id TEXT, protocol_version TEXT, PRIMARY KEY(run_id, revision_id, protocol_version)"),
                               ("operational_events", "event_id TEXT PRIMARY KEY")):
                connection.execute(f"CREATE TABLE IF NOT EXISTS {table} (payload TEXT NOT NULL, {key})")
                for operation in ("UPDATE", "DELETE"):
                    connection.execute(f"CREATE TRIGGER IF NOT EXISTS {table}_{operation.lower()} "
                                       f"BEFORE {operation} ON {table} BEGIN SELECT RAISE(ABORT, 'immutable record'); END")

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _all(self, table):
        if table not in {"forecasts", "outcomes", "scores", "operational_events"}:
            raise ValueError("unknown tracking table")
        with self._connect() as connection:
            return [json.loads(row[0]) for row in connection.execute(f"SELECT payload FROM {table}")]

    def _insert(self, table, keys, payload, *, generated_fields=()):
        encoded = _json(payload)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            condition = " AND ".join(f"{name} = ?" for name in keys)
            existing = connection.execute(f"SELECT payload FROM {table} WHERE {condition}", tuple(keys.values())).fetchone()
            if existing:
                previous = json.loads(existing[0])
                candidate = dict(payload)
                for field in generated_fields:
                    candidate[field] = previous[field]
                if _json(candidate) != _json(previous):
                    raise ValueError(f"immutable {table} identity conflict")
                return previous
            names = ["payload", *keys]
            placeholders = ",".join("?" for _ in names)
            connection.execute(f"INSERT INTO {table} ({','.join(names)}) VALUES ({placeholders})", (encoded, *keys.values()))
        return json.loads(encoded)

    def save_forecast(self, payload, *, is_public=False):
        if type(is_public) is not bool:
            raise ValueError("is_public requires explicit boolean classification")
        payload = dict(payload)
        for name in ("run_id", "event_id", "owner_id", "model_id", "snapshot_id"):
            _identifier(payload.get(name), name)
        payload["mode"] = ForecastMode(payload["mode"]).value
        payload["as_of_utc"] = _timestamp(payload["as_of_utc"], "as_of_utc")
        if type(payload.get("higher_score_better", False)) is not bool:
            raise ValueError("higher_score_better must be boolean")
        ranking = payload.get("ranking")
        if not isinstance(ranking, list) or not ranking:
            raise ValueError("ranking must contain forecast entries")
        seen, ranks = set(), set()
        for entry in ranking:
            driver = _identifier(entry.get("driver_id"), "driver_id")
            if driver in seen:
                raise ValueError("duplicate driver identity")
            seen.add(driver)
            rank = _number(entry.get("predicted_rank"), "predicted_rank", integer=True)
            if rank < 1 or rank in ranks:
                raise ValueError("predicted ranks must be unique positive integers")
            ranks.add(rank)
            score = entry.get("prediction_score")
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not isfinite(score):
                raise ValueError("prediction_score must be finite")
        if ranks != set(range(1, len(ranking) + 1)):
            raise ValueError("predicted ranks must cover the field")
        evaluate_race([dict(row, event_id=payload["event_id"], outcome_status="pending") for row in ranking],
                      higher_score_better=payload.get("higher_score_better", False))
        payload.update(is_public=is_public, recorded_at_utc=_now())
        return self._insert("forecasts", {"run_id": payload["run_id"]}, payload, generated_fields=("recorded_at_utc",))

    def append_outcome(self, payload):
        payload = dict(payload)
        for name in ("event_id", "revision_id"):
            _identifier(payload.get(name), name)
        if payload.get("status") not in {"official", "provisional"}:
            raise ValueError("outcome status must be official or provisional")
        payload["published_at_utc"] = _timestamp(payload["published_at_utc"], "published_at_utc")
        if not isinstance(payload.get("entries"), list) or not payload["entries"]:
            raise ValueError("outcome entries are required")
        payload["entries"] = [{key: value for key, value in entry.items() if key in _OUTCOME_FIELDS}
                              for entry in payload["entries"]]
        seen = set()
        for entry in payload["entries"]:
            driver = _identifier(entry.get("driver_id"), "driver_id")
            if driver in seen:
                raise ValueError("duplicate driver identity")
            seen.add(driver)
            position = entry.get("actual_position")
            if position is not None and (_number(position, "actual_position", integer=True) < 1):
                raise ValueError("actual position must be positive")
            if not isinstance(entry.get("status"), str):
                raise ValueError("entry status is required")
        # Reuse canonical evaluator validation for positions and optional task labels.
        evaluate_race([dict(row, event_id=payload["event_id"], outcome_status=payload["status"]) for row in payload["entries"]])
        return self._insert("outcomes", {key: payload[key] for key in ("event_id", "revision_id")}, payload)

    def score_forecast(self, run_id, revision_id=None, *, scored_at_utc=None):
        _identifier(run_id, "run_id")
        forecast = next((row for row in self._all("forecasts") if row["run_id"] == run_id), None)
        if forecast is None:
            raise KeyError(run_id)
        outcomes = [row for row in self._all("outcomes") if row["event_id"] == forecast["event_id"] and
                    (revision_id is None or row["revision_id"] == revision_id)]
        if not outcomes:
            if revision_id is not None:
                raise KeyError(revision_id)
            return None
        outcome = max(outcomes, key=lambda row: (row["published_at_utc"], row["revision_id"]))
        if outcome["published_at_utc"] <= forecast["as_of_utc"]:
            raise ValueError("outcome publication must follow forecast cutoff")
        scored_at = _timestamp(scored_at_utc, "scored_at_utc") if scored_at_utc else _now()
        if scored_at < outcome["published_at_utc"]:
            raise ValueError("scoring time must follow outcome publication")
        evaluation = evaluate_race(_rows(forecast, outcome), higher_score_better=forecast.get("higher_score_better", False))
        identity = f"{run_id}/{outcome['revision_id']}/{PROTOCOL_VERSION}"
        payload = dict(score_id=str(uuid5(NAMESPACE_URL, identity)), run_id=run_id, event_id=forecast["event_id"],
                       revision_id=outcome["revision_id"], protocol_version=PROTOCOL_VERSION,
                       outcome_status=outcome["status"], published_at_utc=outcome["published_at_utc"],
                       scored_at_utc=scored_at, metrics=evaluation["metrics"], evaluation=evaluation)
        return self._insert("scores", {key: payload[key] for key in ("run_id", "revision_id", "protocol_version")},
                            payload, generated_fields=("scored_at_utc",))

    def record_operational_event(self, payload):
        allowed = {"event_id", "owner_id", "run_id", "model_id", "status", "occurred_at_utc", "latency_ms",
                   "input_tokens", "output_tokens", "tool_calls", "estimated_cost", "billed_cost", "cache_hit"}
        payload = {key: value for key, value in payload.items() if key in allowed}
        for name in ("event_id", "owner_id", "run_id", "model_id"):
            if name in payload or name in {"event_id", "owner_id"}:
                _identifier(payload.get(name), name)
        if payload.get("status") not in {"success", "error", "timeout", "denied", "cancelled"}:
            raise ValueError("invalid operational status")
        payload["occurred_at_utc"] = _timestamp(payload["occurred_at_utc"], "occurred_at_utc")
        for key in ("latency_ms", "input_tokens", "output_tokens", "tool_calls", "estimated_cost", "billed_cost"):
            if payload.get(key) is not None:
                _number(payload[key], key, integer=key.endswith("tokens") or key == "tool_calls")
        if "cache_hit" in payload and type(payload["cache_hit"]) is not bool:
            raise ValueError("cache_hit must be boolean")
        return self._insert("operational_events", {"event_id": payload["event_id"]}, payload)
