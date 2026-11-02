from datetime import datetime, timedelta, timezone

import pytest

from data.contracts import (
    DataSnapshot, DriverSessionRecord, EventKey, ForecastMode,
    PredictionRequest, SourceRecord,
)
from data.sessions import EventInfo, SessionRecord

UTC = timezone.utc
CUT = datetime(2026, 3, 6, 10, tzinfo=UTC)
EVENT = EventKey(2026, 1, "albert-park")


def test_request_rejects_naive_cutoff_and_normalizes_offsets():
    with pytest.raises(ValueError, match="timezone"):
        PredictionRequest(EVENT, ForecastMode.PRE_WEEKEND, datetime(2026, 3, 6))
    request = PredictionRequest(EVENT, "live", CUT.astimezone(timezone(timedelta(hours=3))))
    assert request.as_of_utc == CUT
    assert request.as_of_utc.tzinfo == UTC
    assert request.mode is ForecastMode.LIVE


def test_event_identity_is_year_and_round_with_explicit_circuit_metadata():
    assert EventKey(2025, 1, "albert-park") != EventKey(2026, 1, "sakhir")
    assert EVENT.event_id == "2026:1"
    with pytest.raises(ValueError):
        EventKey(2026, 0)


def test_unknown_publication_cannot_prove_news_existed_before_fetch():
    source = SourceRecord("tavily", "article", CUT)
    assert not source.available_by(CUT - timedelta(seconds=1))
    assert source.available_by(CUT)
    known = SourceRecord("tavily", "article", CUT, published_at_utc=CUT - timedelta(days=1))
    assert known.available_by(CUT - timedelta(hours=1))


def test_latest_availability_evidence_controls_cutoff():
    source = SourceRecord("fastf1", "Q", CUT, published_at_utc=CUT - timedelta(days=1), available_at_utc=CUT)
    assert not source.available_by(CUT - timedelta(hours=1))
    with pytest.raises(ValueError, match="timezone"):
        SourceRecord("fastf1", "Q", datetime(2026, 3, 6))


def test_mode_validation_requires_qualifying_completion_and_before_race():
    sessions = (
        SessionRecord(EVENT, "FP1", "Practice 1", CUT, CUT + timedelta(hours=1)),
        SessionRecord(EVENT, "Q", "Qualifying", CUT + timedelta(days=1), CUT + timedelta(days=1, hours=1)),
        SessionRecord(EVENT, "R", "Race", CUT + timedelta(days=2)),
    )
    schedule = EventInfo(EVENT, "Test GP", "Test", sessions)
    PredictionRequest(EVENT, "pre_weekend", CUT - timedelta(seconds=1)).validate_schedule(schedule)
    with pytest.raises(ValueError):
        PredictionRequest(EVENT, "pre_weekend", CUT).validate_schedule(schedule)
    with pytest.raises(ValueError):
        PredictionRequest(EVENT, "post_qualifying", CUT + timedelta(days=1, minutes=30)).validate_schedule(schedule)
    PredictionRequest(EVENT, "post_qualifying", CUT + timedelta(days=1, hours=2)).validate_schedule(schedule)
    with pytest.raises(ValueError):
        PredictionRequest(EVENT, "post_qualifying", CUT + timedelta(days=2)).validate_schedule(schedule)
    unknown = EventInfo(EVENT, "GP", "Test", (SessionRecord(EVENT, "Q", "Qualifying", CUT), sessions[-1]))
    with pytest.raises(ValueError):
        PredictionRequest(EVENT, "post_qualifying", CUT + timedelta(hours=2)).validate_schedule(unknown)


def test_snapshot_and_driver_payload_preserve_missing_data():
    source = SourceRecord("fastf1", "FP1", CUT)
    row = DriverSessionRecord(EVENT, "FP1", "rookie", CUT, {"team_id": None}, source)
    snapshot = DataSnapshot("snap", EVENT, CUT, {"driver_sessions": [row]}, [source], ["missing_fp2"])
    assert snapshot.payloads["driver_sessions"][0].values["team_id"] is None
    assert snapshot.quality_flags == ["missing_fp2"]
    with pytest.raises(ValueError):
        DriverSessionRecord(EVENT, "FP1", "rookie", datetime(2026, 3, 6), {}, source)

@pytest.mark.parametrize("record", [
    lambda: SourceRecord("provider", "record", None),
    lambda: DriverSessionRecord(EVENT, "FP1", "rookie", None, {}, SourceRecord("provider", "record", CUT)),
    lambda: DataSnapshot("snapshot", EVENT, None),
])
def test_required_availability_timestamps_cannot_be_missing(record):
    with pytest.raises(ValueError, match="timezone"):
        record()
