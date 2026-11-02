"""Issue-time-aware weather records; observations are never forecast inputs."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from data.contracts import SourceRecord, require_utc


@dataclass(frozen=True)
class WeatherRecord:
    valid_at_utc: datetime
    kind: str
    values: dict[str, Any]
    source: SourceRecord
    model_run: str | None = None
    original_valid_at: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "valid_at_utc", require_utc(self.valid_at_utc))
        if self.kind not in ("forecast", "observation"):
            raise ValueError("weather kind must be forecast or observation")

    def usable_for_forecast(self, cutoff: datetime) -> bool:
        cutoff = require_utc(cutoff)
        return (
            self.kind == "forecast"
            and self.source.issued_at_utc is not None
            and self.source.issued_at_utc <= self.valid_at_utc
            and self.source.available_by(cutoff)
        )


@dataclass(frozen=True)
class WeatherResult:
    status: str
    records: list[WeatherRecord] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    quality_flags: list[str] = field(default_factory=list)


def _time(value: Any, *, utc_declared: bool = False) -> datetime:
    if isinstance(value, str):
        if "T" not in value and " " not in value:
            raise ValueError("weather valid time must include an hour")
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if utc_declared and isinstance(value, datetime) and value.utcoffset() is None:
        value = value.replace(tzinfo=timezone.utc)
    return require_utc(value)


def normalize_weather(
    payload: object, *, fetched_at_utc: datetime, kind: str,
    issued_at_utc: datetime | None = None, model_run: str | None = None,
    provider: str = "open-meteo", url: str | None = None,
) -> WeatherResult:
    fetched_at_utc = require_utc(fetched_at_utc)
    if kind not in ("forecast", "observation"):
        raise ValueError("weather kind must be forecast or observation")
    if not isinstance(payload, dict) or payload.get("error"):
        return WeatherResult("error", errors=["weather provider returned an error or invalid schema"])
    if payload.get("is_historical"):
        kind = "observation"
    original = {}
    if issued_at_utc is not None:
        issued_at_utc = require_utc(issued_at_utc)
    elif payload.get("issued_at_utc") is not None:
        original["issued_at_utc"] = payload["issued_at_utc"]
        try:
            issued_at_utc = _time(payload["issued_at_utc"])
        except (ValueError, TypeError):
            issued_at_utc = None
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict) or not isinstance(hourly.get("time"), list):
        return WeatherResult("error", errors=["weather hourly schema unavailable"])
    times = hourly["time"]
    columns = {key: values for key, values in hourly.items() if key != "time"}
    if any(not isinstance(values, list) or len(values) != len(times) for values in columns.values()):
        return WeatherResult("error", errors=["weather hourly columns are misaligned"])
    flags = ["forecast_issue_time_unknown"] if kind == "forecast" and issued_at_utc is None else []
    utc_declared = payload.get("timezone") in ("UTC", "GMT", "Etc/UTC") or payload.get("utc_offset_seconds") == 0
    records = []
    try:
        for index, valid in enumerate(times):
            source = SourceRecord(provider, f"{kind}:{valid}", fetched_at_utc, issued_at_utc=issued_at_utc, url=url, original_timestamps=original)
            records.append(WeatherRecord(_time(valid, utc_declared=utc_declared), kind, {key: values[index] for key, values in columns.items()}, source, model_run or payload.get("model_run"), str(valid)))
    except (ValueError, TypeError):
        return WeatherResult("error", errors=["weather valid time lacks a valid timezone"])
    return WeatherResult("ok" if records else "empty", records, quality_flags=flags)


def fetch_weather(
    url: str, params: dict[str, Any], *, fetched_at_utc: datetime,
    transport: Callable | None = None, timeout: float = 10,
    kind: str = "forecast", issued_at_utc: datetime | None = None,
    model_run: str | None = None,
) -> WeatherResult:
    require_utc(fetched_at_utc)
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if transport is None:
        import requests
        transport = requests.get
    # One bounded attempt; callers decide whether a later retry is appropriate.
    try:
        response = transport(url, params={**params, "timezone": "UTC"}, timeout=timeout)
        if isinstance(response, dict):
            payload = response
        else:
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:
        return WeatherResult("error", errors=[f"weather request failed: {type(exc).__name__}"])
    if "/archive" in url:
        kind = "observation"
    return normalize_weather(payload, fetched_at_utc=fetched_at_utc, kind=kind, issued_at_utc=issued_at_utc, model_run=model_run, url=url)
