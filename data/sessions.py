"""Discover provider schedules without assuming weekdays or equal-round circuits."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Callable

from data.contracts import EventKey, SourceRecord, require_utc

SCHEMA_VERSION = "1"
SESSION_IDS = {
    "Practice 1": "FP1", "Practice 2": "FP2", "Practice 3": "FP3",
    "Qualifying": "Q", "Sprint Qualifying": "SQ", "Sprint Shootout": "SQ",
    "Sprint": "S", "Race": "R",
}


@dataclass(frozen=True)
class SessionRecord:
    event: EventKey
    session_id: str
    name: str
    starts_at_utc: datetime
    ends_at_utc: datetime | None = None

    def __post_init__(self):
        object.__setattr__(self, "starts_at_utc", require_utc(self.starts_at_utc))
        if self.ends_at_utc is not None:
            end = require_utc(self.ends_at_utc)
            if end < self.starts_at_utc:
                raise ValueError("session end precedes start")
            object.__setattr__(self, "ends_at_utc", end)

    def completed_by(self, cutoff: datetime) -> bool:
        return self.ends_at_utc is not None and self.ends_at_utc <= require_utc(cutoff)


@dataclass(frozen=True)
class EventInfo:
    event: EventKey
    event_name: str
    location: str
    sessions: tuple[SessionRecord, ...]
    source: SourceRecord | None = None
    quality_flags: list[str] = field(default_factory=list)


def session_cache_key(event: EventKey, session_id: str, schema_version: str = SCHEMA_VERSION) -> str:
    return f"fastf1:{schema_version}:{event.season}:{event.round_number}:{session_id}"


def _provider_utc(value: Any) -> datetime:
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    # Only provider fields explicitly named DateUtc may contain naive UTC.
    if isinstance(value, datetime) and value.utcoffset() is None:
        value = value.replace(tzinfo=timezone.utc)
    return require_utc(value)


def resolve_event(
    event: EventKey, *, loader: Callable | None = None,
    fetched_at_utc: datetime | None = None,
    circuit_map: dict[tuple[int, str], str] | None = None,
) -> EventInfo:
    if loader is None:
        import fastf1
        loader = fastf1.get_event
    row = loader(event.season, event.round_number)
    fetched_at_utc = require_utc(fetched_at_utc or datetime.now(timezone.utc))
    location = str(row.get("Location") or "")
    circuit = row.get("CircuitId") or (circuit_map or {}).get((event.season, location)) or event.circuit_id
    event = replace(event, circuit_id=circuit)
    sessions = []
    original = {}
    for index in range(1, 6):
        name = row.get(f"Session{index}")
        if name is None or str(name) in ("", "None", "nan", "NaT"):
            continue
        key = f"Session{index}DateUtc"
        start = row.get(key)
        if start is None or str(start) in ("", "None", "nan", "NaT"):
            raise ValueError(f"UTC session timestamp unavailable: {key}")
        original[key] = str(start)
        end = row.get(f"Session{index}EndDateUtc")
        if end is not None and str(end) in ("", "None", "nan", "NaT"):
            end = None
        sessions.append(SessionRecord(event, SESSION_IDS.get(str(name), str(name)), str(name), _provider_utc(start), _provider_utc(end) if end is not None else None))
    source = SourceRecord("fastf1", session_cache_key(event, "schedule"), fetched_at_utc, original_timestamps=original)
    return EventInfo(event, str(row.get("EventName") or ""), location, tuple(sessions), source, [] if circuit else ["circuit_identity_unresolved"])
