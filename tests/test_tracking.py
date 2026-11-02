from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timedelta
from math import log
import json
import sqlite3

import pytest

from monitoring import TrackingQueries, TrackingStore


def forecast(run_id="run-1", owner_id="alice", model_id="model-v1", event_id="2026:1", **extra):
    return dict(run_id=run_id, event_id=event_id, owner_id=owner_id, model_id=model_id,
                mode="pre_weekend", as_of_utc="2026-03-01T00:00:00+00:00", snapshot_id="snap-1",
                ranking=[dict(driver_id=driver, prediction_score=i, predicted_rank=i)
                         for i, driver in enumerate(["a", "b", "c"], 1)], **extra)


def outcome(revision_id="official-1", status="official", event_id="2026:1", reverse=False,
            published_at_utc="2026-03-03T00:00:00Z"):
    return dict(event_id=event_id, revision_id=revision_id, status=status,
                published_at_utc=published_at_utc,
                entries=[dict(driver_id=driver, actual_position=i, status="FINISHED")
                         for i, driver in enumerate(["c", "b", "a"] if reverse else ["a", "b", "c"], 1)])


@pytest.fixture
def store(tmp_path):
    return TrackingStore(tmp_path / "tracking.sqlite3")


def test_persistence_idempotence_and_immutable_conflicts(store):
    original = store.save_forecast(forecast())
    assert store.save_forecast(deepcopy(forecast())) == original
    changed = forecast()
    changed["ranking"][0]["prediction_score"] = 99
    with pytest.raises(ValueError, match="conflict"):
        store.save_forecast(changed)
    reopened = TrackingStore(store.path)
    assert TrackingQueries(reopened, owner_id="alice").get_forecast("run-1") == original
    with sqlite3.connect(store.path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute("DELETE FROM forecasts")


def test_viewer_only_reads_own_or_explicitly_public_records(store):
    store.save_forecast(forecast())
    store.save_forecast(forecast("bob-private", "bob"))
    store.save_forecast(forecast("bob-public", "bob"), is_public=True)
    query = TrackingQueries(store, owner_id="alice")
    assert {row["run_id"] for row in query.forecasts()} == {"run-1", "bob-public"}
    assert query.get_forecast("bob-private") is None
    assert len(TrackingQueries(store).forecasts()) == 1
    assert len(TrackingQueries(store, role="operator").forecasts()) == 3
    with pytest.raises(PermissionError):
        query.operational_events()
    with pytest.raises(ValueError):
        TrackingQueries(store, role="admin")
    with pytest.raises(ValueError, match="conflict"):
        store.save_forecast(forecast(), is_public=True)


def test_pending_and_revision_scoring_and_published_order(store):
    store.save_forecast(forecast())
    query = TrackingQueries(store, owner_id="alice")
    assert store.score_forecast("run-1") is None
    assert query.summary()["groups"][0]["metrics"]["position_mae"] is None
    store.append_outcome(outcome("provisional", "provisional"))
    score = store.score_forecast("run-1")
    assert score["metrics"]["position_mae"] == 0
    assert store.score_forecast("run-1") == score
    store.append_outcome(outcome("corrected", reverse=True, published_at_utc="2026-03-04T00:00:00Z"))
    corrected = store.score_forecast("run-1")
    assert corrected["revision_id"] == "corrected"
    assert corrected["metrics"]["position_mae"] == pytest.approx(4 / 3)
    store.append_outcome(outcome("late-old", published_at_utc="2026-03-02T00:00:00Z"))
    assert store.score_forecast("run-1") == corrected
    assert len(query.outcomes("2026:1")) == 3
    assert len(query.score_history("run-1")) == 2
    changed = outcome("corrected", reverse=False, published_at_utc="2026-03-04T00:00:00Z")
    with pytest.raises(ValueError, match="conflict"):
        store.append_outcome(changed)
    summary = query.summary()["groups"][0]
    assert summary["n_events"] == 1
    assert summary["metric_event_counts"]["position_mae"] == 1
    assert summary["metrics"]["winner_log_loss"] is None


def test_new_revision_invalidates_current_summary_until_scored(store):
    store.save_forecast(forecast())
    store.append_outcome(outcome())
    store.score_forecast("run-1")
    store.append_outcome(outcome("correction", reverse=True, published_at_utc="2026-03-04T00:00:00Z"))
    summary = TrackingQueries(store, owner_id="alice").summary()["groups"][0]
    assert summary["metrics"]["position_mae"] is None
    assert summary["n_pending_events"] == 1


def test_race_macro_uses_latest_cutoff_and_common_events(store):
    for payload in [forecast(), forecast("again"), forecast("v2", model_id="model-v2"),
                    forecast("other", model_id="model-v2", event_id="2026:2")]:
        if payload["run_id"] == "again":
            payload["as_of_utc"] = "2026-03-01T01:00:00Z"
            payload["ranking"].reverse()
            for i, row in enumerate(payload["ranking"], 1):
                row.update(prediction_score=i, predicted_rank=i)
        store.save_forecast(payload)
    for event in ["2026:1", "2026:2"]:
        store.append_outcome(outcome(event_id=event))
    for run in ["run-1", "again", "v2", "other"]:
        store.score_forecast(run)
    query = TrackingQueries(store, owner_id="alice")
    group = query.summary(model_id="model-v1")["groups"][0]
    assert group["n_forecasts"] == 2
    assert group["n_events"] == 1
    assert group["metrics"]["position_mae"] == pytest.approx(4 / 3)
    comparison = query.compare_models(["model-v1", "model-v2"])
    assert comparison["common_event_ids"] == ["2026:1"]
    assert all(group["n_events"] == 1 for group in comparison["groups"])
    assert query.summary(window=1)["groups"][1]["n_events"] == 1


def test_partial_outcomes_and_missing_fields_do_not_invent_metrics(store):
    store.save_forecast(forecast())
    label = outcome()
    label["entries"].pop()
    store.append_outcome(label)
    score = store.score_forecast("run-1")
    assert score["metrics"]["position_mae"] is None
    assert score["evaluation"]["outcome_status"] == "pending"


@pytest.mark.parametrize("field,value", [("run_id", "../escape"), ("owner_id", ""),
    ("as_of_utc", "2026-03-01T00:00:00"), ("as_of_utc", "2026-03-01T02:00:00+02:00"),
    ("mode", "unknown")])
def test_forecast_boundary_validation(store, field, value):
    payload = forecast()
    payload[field] = value
    with pytest.raises((ValueError, TypeError)):
        store.save_forecast(payload)


def test_ranking_schema_validation_and_label_cutoff(store):
    payload = forecast()
    payload["ranking"][1]["driver_id"] = "a"
    with pytest.raises(ValueError):
        store.save_forecast(payload)
    payload = forecast()
    payload["ranking"][0]["prediction_score"] = float("nan")
    with pytest.raises(ValueError):
        store.save_forecast(payload)
    store.save_forecast(forecast())
    store.append_outcome(outcome(published_at_utc="2026-02-01T00:00:00Z"))
    with pytest.raises(ValueError, match="cutoff"):
        store.score_forecast("run-1")


def test_atomic_duplicate_inserts_from_multiple_store_instances(store):
    def insert(_):
        return TrackingStore(store.path).save_forecast(forecast())
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(insert, range(12)))
    assert all(row == rows[0] for row in rows)
    assert len(TrackingQueries(store, role="operator").forecasts()) == 1


def test_operational_allowlist_and_operator_aggregate(store):
    store.record_operational_event(dict(event_id="op-1", owner_id="alice", run_id="run-1",
        model_id="model-v1", status="success", occurred_at_utc="2026-03-03T00:00:00Z",
        latency_ms=120, input_tokens=20, output_tokens=5, estimated_cost=0.01,
        billed_cost=None, prompt="secret prompt", authorization="Bearer sensitive"))
    operator = TrackingQueries(store, role="operator")
    raw = operator.operational_events()
    assert len(raw) == 1
    assert "secret" not in json.dumps(raw)
    assert "sensitive" not in json.dumps(raw)
    assert "prompt" not in raw[0]
    assert operator.operational_summary()["p95_latency_ms"] == 120
    assert operator.operational_summary()["billed_cost"] is None
    with pytest.raises(PermissionError):
        TrackingQueries(store, owner_id="alice").operational_summary()


def test_comparison_requires_same_completed_common_cohort(store):
    store.save_forecast(forecast())
    store.save_forecast(forecast("v2", model_id="model-v2"))
    store.append_outcome(outcome())
    store.score_forecast("run-1")
    query = TrackingQueries(store, owner_id="alice")
    assert query.compare_models(["model-v1", "model-v2"])["common_event_ids"] == []
    store.score_forecast("v2")
    assert query.compare_models(["model-v1", "model-v2"])["common_event_ids"] == ["2026:1"]
    other_mode = forecast("different-mode", model_id="model-v2")
    other_mode["mode"] = "post_qualifying"
    store.save_forecast(other_mode)
    with pytest.raises(ValueError, match="mode"):
        query.compare_models(["model-v1", "model-v2"])


def test_historical_summary_uses_recording_and_scoring_times(store):
    saved = store.save_forecast(forecast())
    store.append_outcome(outcome())
    score = store.score_forecast("run-1", scored_at_utc=datetime.fromisoformat(saved["recorded_at_utc"]) + timedelta(seconds=1))
    query = TrackingQueries(store, owner_id="alice")
    assert query.summary(as_of_utc="2026-03-05T00:00:00Z")["groups"] == []
    before_score = query.summary(as_of_utc=saved["recorded_at_utc"])["groups"][0]
    assert before_score["metrics"]["position_mae"] is None
    after_score = query.summary(as_of_utc=score["scored_at_utc"])["groups"][0]
    assert after_score["metrics"]["position_mae"] == 0


def test_invalid_distribution_rejected_before_save(store):
    payload = forecast()
    for row in payload["ranking"]:
        row["winner_probability"] = 0.8
    with pytest.raises(ValueError, match="sum"):
        store.save_forecast(payload)


def test_denied_score_and_outcome_history_not_exposed(store):
    store.save_forecast(forecast())
    store.append_outcome(outcome())
    store.score_forecast("run-1")
    stranger = TrackingQueries(store, owner_id="bob")
    assert stranger.score_history("run-1") == []
    assert stranger.outcomes("2026:1") == []
    assert stranger.summary()["groups"] == []


def test_concurrent_score_replays_remain_one_event(store):
    store.save_forecast(forecast())
    store.append_outcome(outcome())
    with ThreadPoolExecutor(max_workers=4) as pool:
        scores = list(pool.map(lambda _: TrackingStore(store.path).score_forecast("run-1"), range(8)))
    assert all(score == scores[0] for score in scores)
    assert len(TrackingQueries(store, owner_id="alice").score_history("run-1")) == 1


def test_operational_denials_do_not_dilute_failure_rate(store):
    for i, status in enumerate(["error", "success", "denied", "denied"]):
        store.record_operational_event(dict(event_id=f"op-{i}", owner_id="alice", status=status,
            occurred_at_utc="2026-03-03T00:00:00Z", billed_cost=0.2 if i == 0 else None))
    summary = TrackingQueries(store, role="operator").operational_summary()
    assert summary["failure_rate"] == 0.5
    assert summary["billed_cost_event_count"] == 1


@pytest.mark.parametrize("legacy_record", [False, True])
def test_outcome_predictions_cannot_replace_saved_forecast(store, legacy_record):
    prediction = forecast()
    for row in prediction["ranking"]:
        row["winner_probability"] = 0.8 if row["driver_id"] == "a" else 0.1
        row["predicted_pace_s"] = 9 + row["predicted_rank"]
    saved = store.save_forecast(prediction)
    labels = outcome(reverse=True)
    for row in labels["entries"]:
        row.update(prediction_score=row["actual_position"], predicted_rank=row["actual_position"],
                   actual_pace_s=9 + row["actual_position"], predicted_pace_s=9 + row["actual_position"],
                   winner_probability=float(row["actual_position"] == 1))
    if legacy_record:
        # Previously accepted payloads already on disk must remain safe to score.
        with sqlite3.connect(store.path) as connection:
            connection.execute("INSERT INTO outcomes(payload, event_id, revision_id) VALUES (?, ?, ?)",
                               (json.dumps(labels), labels["event_id"], labels["revision_id"]))
    else:
        recorded = store.append_outcome(labels)
        assert all(set(row) == {"driver_id", "actual_position", "status", "actual_pace_s"}
                   for row in recorded["entries"])
    score = store.score_forecast("run-1")
    summary = TrackingQueries(store, owner_id="alice").summary()["groups"][0]
    for metrics in (score["metrics"], summary["metrics"]):
        assert metrics["position_mae"] == pytest.approx(4 / 3)
        assert metrics["winner_log_loss"] == pytest.approx(-log(0.1))
        assert metrics["winner_brier"] == pytest.approx(1.46)
        assert metrics["pace_mae_seconds"] == pytest.approx(4 / 3)
    assert TrackingQueries(store, owner_id="alice").get_forecast("run-1") == saved


@pytest.mark.parametrize("legacy_record", [False, True])
def test_event_qualified_outcome_entries_join_using_canonical_event_and_status(store, legacy_record):
    store.save_forecast(forecast())
    labels = outcome(reverse=True)
    for row in labels["entries"]:
        row.update(event_id=labels["event_id"], outcome_status="pending")
    if legacy_record:
        with sqlite3.connect(store.path) as connection:
            connection.execute("INSERT INTO outcomes(payload, event_id, revision_id) VALUES (?, ?, ?)",
                               (json.dumps(labels), labels["event_id"], labels["revision_id"]))
    else:
        store.append_outcome(labels)
    score = store.score_forecast("run-1")
    assert score["metrics"]["position_mae"] == pytest.approx(4 / 3)
    assert score["evaluation"]["outcome_status"] == "completed"


def test_denials_and_cancellations_do_not_dilute_dispatched_latency(store):
    for i, (status, latency) in enumerate([("success", 1000), *[("denied", 1)] * 20,
                                           *[("cancelled", 2)] * 20]):
        store.record_operational_event(dict(event_id=f"op-{i}", owner_id="alice", status=status,
            occurred_at_utc="2026-03-03T00:00:00Z", latency_ms=latency))
    summary = TrackingQueries(store, role="operator").operational_summary()
    assert summary["p50_latency_ms"] == 1000
    assert summary["p95_latency_ms"] == 1000
    assert summary["latency_event_count"] == 1
    assert summary["denied_count"] == 20
    assert summary["denied_p95_latency_ms"] == 1
    assert summary["denied_latency_event_count"] == 20


def test_dispatch_latency_includes_errors_and_timeouts_but_requires_measurements(store):
    for i, (status, latency) in enumerate([("success", 100), ("error", 200), ("timeout", 1000),
                                           ("success", None), ("denied", None)]):
        store.record_operational_event(dict(event_id=f"op-{i}", owner_id="alice", status=status,
            occurred_at_utc="2026-03-03T00:00:00Z", latency_ms=latency))
    summary = TrackingQueries(store, role="operator").operational_summary()
    assert summary["p50_latency_ms"] == 200
    assert summary["p95_latency_ms"] == 1000
    assert summary["latency_event_count"] == 3
    assert summary["denied_p95_latency_ms"] is None
    assert summary["denied_latency_event_count"] == 0


def test_denials_only_do_not_invent_dispatch_latency(store):
    store.record_operational_event(dict(event_id="op-1", owner_id="alice", status="denied",
        occurred_at_utc="2026-03-03T00:00:00Z", latency_ms=1))
    summary = TrackingQueries(store, role="operator").operational_summary()
    assert summary["p50_latency_ms"] is None
    assert summary["p95_latency_ms"] is None
    assert summary["latency_event_count"] == 0
    assert summary["denied_p95_latency_ms"] == 1
