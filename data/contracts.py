"""Shared domain records; times are explicit UTC and missing evidence stays unknown."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from data.sessions import EventInfo


def require_utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValueError("timestamp must be a timezone-aware datetime")
    return value.astimezone(timezone.utc)


class ForecastMode(str, Enum):
    PRE_WEEKEND = "pre_weekend"
    POST_QUALIFYING = "post_qualifying"
    REPLAY_ANALYSIS = "replay_analysis"
    LIVE = "live"


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    DEGRADED = "degraded"
    INSUFFICIENT_DATA = "insufficient_data"
    ERROR = "error"


@dataclass(frozen=True)
class EventKey:
    season: int
    round_number: int
    # Metadata does not change the season/round event identity or joins.
    circuit_id: str | None = field(default=None, compare=False)

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in (self.season, self.round_number)):
            raise ValueError("season and round_number must be positive integers")

    @property
    def event_id(self) -> str:
        return f"{self.season}:{self.round_number}"


@dataclass(frozen=True)
class PredictionRequest:
    event: EventKey
    mode: ForecastMode
    as_of_utc: datetime

    def __post_init__(self):
        object.__setattr__(self, "mode", ForecastMode(self.mode))
        object.__setattr__(self, "as_of_utc", require_utc(self.as_of_utc))

    def validate_schedule(self, schedule: EventInfo) -> None:
        if self.event != schedule.event:
            raise ValueError("schedule event does not match request")
        if self.mode in (ForecastMode.LIVE, ForecastMode.REPLAY_ANALYSIS):
            return
        if not schedule.sessions:
            raise ValueError("session schedule is unavailable")
        if self.mode == ForecastMode.PRE_WEEKEND:
            if self.as_of_utc >= min(session.starts_at_utc for session in schedule.sessions):
                raise ValueError("pre_weekend cutoff must precede the first session")
            return
        qualifying = next((session for session in schedule.sessions if session.session_id == "Q"), None)
        race = next((session for session in schedule.sessions if session.session_id == "R"), None)
        if qualifying is None or qualifying.ends_at_utc is None:
            raise ValueError("qualifying completion time is unavailable")
        if race is None or not qualifying.ends_at_utc <= self.as_of_utc < race.starts_at_utc:
            raise ValueError("post_qualifying cutoff must follow qualifying and precede the race")


@dataclass(frozen=True)
class SourceRecord:
    provider: str
    source_id: str
    fetched_at_utc: datetime
    published_at_utc: datetime | None = None
    issued_at_utc: datetime | None = None
    available_at_utc: datetime | None = None
    url: str | None = None
    original_timestamps: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "fetched_at_utc", require_utc(self.fetched_at_utc))
        for name in ("published_at_utc", "issued_at_utc", "available_at_utc"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, require_utc(value))

    @property
    def availability_time_utc(self) -> datetime:
        evidence = [value for value in (self.published_at_utc, self.issued_at_utc, self.available_at_utc) if value is not None]
        # A fetch proves availability from that fetch onward, never before it.
        return max(evidence) if evidence else self.fetched_at_utc

    def available_by(self, cutoff: datetime) -> bool:
        return self.availability_time_utc <= require_utc(cutoff)


@dataclass(frozen=True)
class DriverSessionRecord:
    event: EventKey
    session_id: str
    driver_id: str
    available_at_utc: datetime
    values: dict[str, Any]
    source: SourceRecord

    def __post_init__(self):
        object.__setattr__(self, "available_at_utc", require_utc(self.available_at_utc))


@dataclass(frozen=True)
class DataSnapshot:
    snapshot_id: str
    event: EventKey
    as_of_utc: datetime
    payloads: dict[str, Any] = field(default_factory=dict)
    sources: list[SourceRecord] = field(default_factory=list)
    quality_flags: list[str] = field(default_factory=list)

    def __post_init__(self):
        object.__setattr__(self, "as_of_utc", require_utc(self.as_of_utc))


@dataclass(frozen=True)
class DriverForecast:
    driver_id: str
    rank: int
    score: float
    pace_delta_s: float | None = None
    probabilities: dict[str, float] | None = None
    interval: tuple[float, float] | None = None


@dataclass(frozen=True)
class PredictionResult:
    run_id: str
    request: PredictionRequest
    snapshot_id: str
    model_id: str
    ranking: list[DriverForecast]
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    status: RunStatus = RunStatus.PENDING


@dataclass(frozen=True)
class ProgressEvent:
    run_id: str
    stage: str
    status: str
    message: str


@dataclass(frozen=True)
class ReplayFrame:
    session_time_s: float
    drivers: dict[str, Any]
    weather: dict[str, Any]
    race_control: list[Any]
