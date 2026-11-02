from datetime import datetime, timedelta, timezone

import pytest

from data.contracts import EventKey
from data.news import fetch_news, normalize_news
from data.sessions import resolve_event, session_cache_key
from data.weather import fetch_weather, normalize_weather

UTC = timezone.utc
CUT = datetime(2026, 11, 20, 12, tzinfo=UTC)


@pytest.mark.parametrize("wrapped", [False, True])
def test_tavily_list_and_dict_produce_headlines_and_unknown_publication(wrapped):
    rows = [{"title": "Grid penalty under review", "url": "https://example.org/f1", "content": "Decision pending"}]
    result = normalize_news({"results": rows} if wrapped else rows, fetched_at_utc=CUT)
    assert result.status == "ok"
    assert result.articles[0].title == "Grid penalty under review"
    assert result.articles[0].source.published_at_utc is None
    assert not result.articles[0].source.available_by(CUT - timedelta(days=1))
    assert result.articles[0].source.fetched_at_utc == CUT


def test_news_empty_error_and_malformed_states_are_distinct():
    assert normalize_news({"results": []}, fetched_at_utc=CUT).status == "empty"
    assert normalize_news({"error": "unavailable"}, fetched_at_utc=CUT).status == "error"
    assert normalize_news({"unexpected": []}, fetched_at_utc=CUT).status == "error"
    assert normalize_news([{"error": "timeout"}], fetched_at_utc=CUT).status == "error"
    result = normalize_news([{"title": "A", "url": "x", "published_date": "2026-11-19T10:00:00Z"}], fetched_at_utc=CUT)
    assert result.articles[0].source.published_at_utc == datetime(2026, 11, 19, 10, tzinfo=UTC)


def test_news_fetch_exception_is_error_without_claiming_no_penalties():
    def unavailable(query):
        raise TimeoutError("offline")
    result = fetch_news("F1", search=unavailable, fetched_at_utc=CUT)
    assert result.status == "error"
    assert result.articles == []


def test_fastf1_discovery_preserves_sprint_and_saturday_utc_without_fp2():
    event = EventKey(2026, 22)
    fixture = {
        "EventName": "Saturday GP", "Location": "Desert", "CircuitId": "desert-track",
        "Session1": "Practice 1", "Session1DateUtc": datetime(2026, 11, 19, 18),
        "Session2": "Sprint Qualifying", "Session2DateUtc": datetime(2026, 11, 20, 18),
        "Session3": "Sprint", "Session3DateUtc": datetime(2026, 11, 21, 0),
        "Session4": "Qualifying", "Session4DateUtc": datetime(2026, 11, 21, 4),
        "Session5": "Race", "Session5DateUtc": datetime(2026, 11, 22, 4),
    }
    info = resolve_event(event, loader=lambda season, round_number: fixture, fetched_at_utc=CUT)
    assert [s.session_id for s in info.sessions] == ["FP1", "SQ", "S", "Q", "R"]
    assert info.sessions[-1].starts_at_utc == datetime(2026, 11, 22, 4, tzinfo=UTC)
    assert info.sessions[-1].ends_at_utc is None
    assert info.event.circuit_id == "desert-track"
    assert session_cache_key(EventKey(2025, 22), "R") != session_cache_key(event, "R")


def test_round_number_never_supplies_cross_year_circuit_identity():
    a = resolve_event(EventKey(2025, 1), loader=lambda *_: {"EventName": "A", "Location": "Melbourne"}, fetched_at_utc=CUT, circuit_map={(2025, "Melbourne"): "albert-park"})
    b = resolve_event(EventKey(2026, 1), loader=lambda *_: {"EventName": "B", "Location": "Sakhir"}, fetched_at_utc=CUT, circuit_map={(2026, "Sakhir"): "bahrain"})
    assert a.event.circuit_id != b.event.circuit_id


def test_local_session_timestamp_is_not_assumed_utc():
    with pytest.raises(ValueError, match="UTC"):
        resolve_event(EventKey(2026, 1), loader=lambda *_: {"EventName": "GP", "Location": "X", "Session1": "Race", "Session1Date": "2026-01-01T12:00:00"}, fetched_at_utc=CUT)


def test_weather_forecast_requires_issue_time_and_preserves_valid_time():
    payload = {"timezone": "UTC", "hourly": {"time": ["2026-11-22T04:00"], "temperature_2m": [22], "precipitation_probability": [None]}}
    unknown = normalize_weather(payload, fetched_at_utc=CUT, kind="forecast")
    assert unknown.status == "ok"
    assert not unknown.records[0].usable_for_forecast(CUT)
    issued = CUT - timedelta(hours=6)
    result = normalize_weather(payload, fetched_at_utc=CUT, kind="forecast", issued_at_utc=issued, model_run="run-1")
    record = result.records[0]
    assert record.valid_at_utc == datetime(2026, 11, 22, 4, tzinfo=UTC)
    assert record.source.issued_at_utc == issued
    assert record.usable_for_forecast(CUT)
    assert record.values["precipitation_probability"] is None
    assert not record.usable_for_forecast(issued - timedelta(seconds=1))


def test_observed_weather_archive_cannot_be_a_pre_race_forecast():
    result = normalize_weather({"timezone": "UTC", "hourly": {"time": ["2026-11-22T04:00"], "temperature_2m": [22]}}, fetched_at_utc=CUT, kind="observation", issued_at_utc=CUT - timedelta(days=1))
    assert not result.records[0].usable_for_forecast(CUT)
    assert result.records[0].kind == "observation"


def test_weather_http_failure_and_empty_data_are_explicit():
    def failed(url, params, timeout):
        raise TimeoutError("offline")
    assert fetch_weather("https://example.org", {}, transport=failed, fetched_at_utc=CUT).status == "error"
    assert normalize_weather({"timezone": "UTC", "hourly": {"time": []}}, fetched_at_utc=CUT, kind="forecast").status == "empty"
    assert normalize_weather({"error": True, "reason": "bad request"}, fetched_at_utc=CUT, kind="forecast").status == "error"



def test_unzoned_weather_timestamps_do_not_gain_invented_utc():
    result = normalize_weather({"hourly": {"time": ["2026-11-22T04:00"], "temperature_2m": [22]}}, fetched_at_utc=CUT, kind="forecast", issued_at_utc=CUT)
    assert result.status == "error"


def test_missing_weather_column_is_unknown_not_zero():
    result = normalize_weather({"timezone": "UTC", "hourly": {"time": ["2026-11-22T04:00"]}}, fetched_at_utc=CUT, kind="forecast", issued_at_utc=CUT)
    assert result.records[0].values.get("precipitation_probability") is None



def test_partial_news_error_retains_evidence_without_success_status():
    result = normalize_news([{"title": "A", "url": "https://example.org/a"}, {"error": "timeout"}], fetched_at_utc=CUT)
    assert result.status == "error"
    assert [article.title for article in result.articles] == ["A"]


def test_original_news_publication_timestamp_survives_normalization():
    result = normalize_news([{"title": "A", "url": "https://example.org/a", "published_date": "2026-11-19T12:00:00+02:00"}], fetched_at_utc=CUT)
    assert result.articles[0].source.published_at_utc == datetime(2026, 11, 19, 10, tzinfo=UTC)
    assert result.articles[0].source.original_timestamps["published_date"] == "2026-11-19T12:00:00+02:00"

@pytest.mark.parametrize("row", [{}, {"title": "A", "url": ["bad"]}])
def test_malformed_news_article_is_error_instead_of_success(row):
    assert normalize_news([row], fetched_at_utc=CUT).status == "error"


def test_weather_issue_after_valid_time_is_not_a_forecast_of_that_time():
    result = normalize_weather({"timezone": "UTC", "hourly": {"time": ["2026-11-19T04:00"], "temperature_2m": [22]}}, fetched_at_utc=CUT, kind="forecast", issued_at_utc=CUT)
    assert not result.records[0].usable_for_forecast(CUT)


def test_weather_date_only_time_does_not_invent_a_valid_hour():
    result = normalize_weather({"timezone": "UTC", "hourly": {"time": ["2026-11-22"], "temperature_2m": [22]}}, fetched_at_utc=CUT, kind="forecast", issued_at_utc=CUT)
    assert result.status == "error"


def test_weather_archive_endpoint_is_observation_even_when_caller_defaults_forecast():
    result = fetch_weather("https://archive-api.open-meteo.com/v1/archive", {}, fetched_at_utc=CUT, issued_at_utc=CUT - timedelta(days=1), transport=lambda *args, **kwargs: {"timezone": "UTC", "hourly": {"time": ["2026-11-22T04:00"], "temperature_2m": [22]}})
    assert result.records[0].kind == "observation"
    assert not result.records[0].usable_for_forecast(CUT)
