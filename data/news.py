"""Normalize Tavily responses without confusing unavailable search with no news."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from data.contracts import SourceRecord, require_utc


def _published_time(value: Any) -> datetime | None:
    try:
        if isinstance(value, str):
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return require_utc(value) if value is not None else None
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class NewsArticle:
    title: str
    url: str | None
    content: str
    source: SourceRecord


@dataclass(frozen=True)
class NewsResult:
    status: str
    articles: list[NewsArticle] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    quality_flags: list[str] = field(default_factory=list)


def normalize_news(payload: object, *, fetched_at_utc: datetime) -> NewsResult:
    fetched_at_utc = require_utc(fetched_at_utc)
    if isinstance(payload, dict):
        if payload.get("error"):
            return NewsResult("error", errors=["news provider returned an error"])
        payload = payload.get("results")
    if not isinstance(payload, list):
        return NewsResult("error", errors=["invalid news response schema"])
    articles, errors, flags = [], [], []
    seen = set()
    for index, row in enumerate(payload):
        if not isinstance(row, dict) or row.get("error"):
            errors.append("news result unavailable or malformed")
            continue
        url = row.get("url") or None
        if (url is not None and not isinstance(url, str)) or not any(
            isinstance(row.get(key), str) and row[key].strip()
            for key in ("title", "content", "snippet")
        ):
            errors.append("news article has no text or has an invalid URL")
            continue
        if url and url in seen:
            continue
        if url:
            seen.add(url)
        original = {key: row[key] for key in ("published_at_utc", "published_at", "published_date", "available_at_utc") if row.get(key) is not None}
        publication = next((row[key] for key in ("published_at_utc", "published_at", "published_date") if row.get(key) is not None), None)
        published_at = _published_time(publication)
        if published_at is None:
            flags.append("publication_time_unknown")
        source = SourceRecord("tavily", url or f"article:{index}", fetched_at_utc, published_at_utc=published_at, available_at_utc=_published_time(row.get("available_at_utc")), url=url, original_timestamps=original)
        articles.append(NewsArticle(str(row.get("title") or ""), url, str(row.get("content") or row.get("snippet") or ""), source))
    status = "error" if errors else "ok" if articles else "empty"
    return NewsResult(status, articles, errors, sorted(set(flags)))


def fetch_news(query: str, *, search: Callable[[str], object], fetched_at_utc: datetime) -> NewsResult:
    require_utc(fetched_at_utc)
    try:
        payload = search(query)
    except Exception as exc:
        return NewsResult("error", errors=[f"news request failed: {type(exc).__name__}"])
    return normalize_news(payload, fetched_at_utc=fetched_at_utc)
