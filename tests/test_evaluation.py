"""Whole-race chronology, aggregation and comparator invariants."""

import pandas as pd
import pytest

from ml import baselines, evaluation


def test_macro_aggregation_does_not_weight_large_races_more():
    rows = [dict(event_id="r1", driver_id="a", actual_position=1, prediction_score=1),
            dict(event_id="r1", driver_id="b", actual_position=2, prediction_score=0)]
    rows += [dict(event_id="r2", driver_id=str(i), actual_position=i, prediction_score=i)
             for i in range(1, 21)]
    result = evaluation.evaluate_predictions(rows, bootstrap_samples=100, seed=9)
    assert result["metrics"]["position_mae"] == 0.5
    assert result["metrics"]["rank_spearman"] == pytest.approx(0)
    assert result["metric_event_counts"]["rank_spearman"] == 2
    assert result["confidence_intervals"]["position_mae"] == [0.0, 1.0]
    assert result["n_events"] == 2
    assert result["protocol"]["version"]


def test_pending_races_not_in_metric_denominator_and_empty_cohort_is_unavailable():
    rows = [dict(event_id="r1", driver_id="a", actual_position=1, prediction_score=1),
            dict(event_id="r2", driver_id="a", actual_position=None, prediction_score=1)]
    result = evaluation.evaluate_predictions(rows)
    assert result["metrics"]["position_mae"] == 0
    assert result["metric_event_counts"]["position_mae"] == 1
    assert evaluation.evaluate_predictions([])["metrics"]["position_mae"] is None


def chronology():
    return [dict(event_id=event, driver_id=driver, event_time_utc=time,
                 outcome_available_at_utc=time.replace("12:00", "13:00"),
                 actual_position=position, team_id="team")
            for event, time in [("old", "2024-01-01T12:00:00Z"),
                                ("same", "2024-02-01T12:00:00Z"),
                                ("test", "2024-02-01T12:00:00Z"),
                                ("future", "2024-03-01T12:00:00Z")]
            for driver, position in [("a", 1), ("b", 2)]]


def test_walk_forward_is_whole_race_and_strictly_earlier_with_same_time_events():
    folds = evaluation.walk_forward_folds(pd.DataFrame(chronology()), min_train_events=1)
    heldout = {fold.test_event_ids[0]: fold for fold in folds}
    assert heldout["test"].train_event_ids == ("old",)
    assert heldout["same"].train_event_ids == ("old",)
    assert len(heldout["test"].train_rows) == 2
    assert len(heldout["test"].test_rows) == 2
    assert set(heldout["future"].train_event_ids) == {"old", "same", "test"}


def test_walk_forward_requires_consistent_aware_event_times():
    rows = chronology()
    rows[0]["event_time_utc"] = "2024-01-01T12:00:00"
    with pytest.raises(ValueError):
        evaluation.walk_forward_folds(rows, min_train_events=1)
    rows = chronology()
    rows.append(rows[0].copy())
    with pytest.raises(ValueError):
        evaluation.walk_forward_folds(rows, min_train_events=1)
    rows = chronology()
    rows[0]["event_time_utc"] = "2024-01-02T12:00:00Z"
    with pytest.raises(ValueError):
        evaluation.walk_forward_folds(rows, min_train_events=1)


def test_walk_forward_respects_result_publication_and_excludes_pending_labels():
    rows = chronology()
    for row in rows:
        if row["event_id"] == "old":
            row["outcome_available_at_utc"] = "2024-02-02T00:00:00Z"
    folds = evaluation.walk_forward_folds(rows, min_train_events=1)
    assert [fold.test_event_ids for fold in folds] == [("future",)]
    rows = chronology()
    rows[0]["actual_position"] = None
    folds = evaluation.walk_forward_folds(rows, min_train_events=1)
    assert [fold.test_event_ids for fold in folds] == [("future",)]
    assert "old" not in folds[0].train_event_ids


def test_fold_uses_forecast_cutoff_and_never_assumes_unknown_label_publication():
    rows = chronology()
    for row in rows:
        if row["event_id"] in {"test", "same"}:
            row["forecast_cutoff_utc"] = "2023-12-31T00:00:00Z"
    assert [fold.test_event_ids for fold in evaluation.walk_forward_folds(
        rows, min_train_events=1)] == [("future",)]
    rows = chronology()
    for row in rows:
        row.pop("outcome_available_at_utc")
    assert evaluation.walk_forward_folds(rows, min_train_events=1) == []


def test_nonfinite_history_position_rejected_instead_of_training_on_it():
    rows = chronology()
    rows[0]["actual_position"] = float("inf")
    with pytest.raises(ValueError):
        evaluation.walk_forward_folds(rows, min_train_events=1)


def test_grid_baseline_preserves_unknown_identity_and_marks_missing_grid():
    rows = [dict(event_id="r", driver_id="rookie", team_id="new", grid_position=3),
            dict(event_id="r", driver_id="a", grid_position=None)]
    predicted = baselines.grid_baseline(rows)
    assert predicted[0]["driver_id"] == "rookie"
    assert predicted[0]["prediction_score"] == 3
    assert predicted[1]["prediction_score"] is None
    assert predicted[1]["baseline_status"] == "unavailable"


def test_rolling_form_is_past_only_and_uses_team_then_global_prior_for_rookies():
    history = chronology()
    entries = [dict(event_id="test", driver_id="a", team_id="team"),
               dict(event_id="test", driver_id="rookie", team_id="team"),
               dict(event_id="test", driver_id="other", team_id="new")]
    predicted = baselines.rolling_form_baseline(
        history, entries, as_of_utc="2024-02-01T12:00:00Z")
    assert [row["prediction_score"] for row in predicted] == [1, 1.5, 10.5]
    assert predicted[1]["driver_id"] == "rookie"
    assert predicted[2]["baseline_source"] == "prior"
    for row in history:
        if row["event_id"] != "old":
            row["actual_position"] = 99
    assert baselines.rolling_form_baseline(
        history, entries, as_of_utc="2024-02-01T12:00:00Z") == predicted


def test_rolling_form_window_counts_races_and_waits_for_known_labels():
    history = [dict(event_id="r1", driver_id="a", actual_position=8,
                    event_time_utc="2024-01-01T00:00:00Z",
                    outcome_available_at_utc="2024-01-01T01:00:00Z"),
               dict(event_id="r2", driver_id="a", actual_position=2,
                    event_time_utc="2024-01-02T00:00:00Z",
                    outcome_available_at_utc="2024-01-02T01:00:00Z"),
               dict(event_id="r3", driver_id="a", actual_position=1,
                    event_time_utc="2024-01-03T00:00:00Z",
                    outcome_available_at_utc="2024-01-05T00:00:00Z")]
    entries = [dict(event_id="r4", driver_id="a")]
    assert baselines.rolling_form_baseline(history, entries,
        as_of_utc="2024-01-04T00:00:00Z", window=1)[0]["prediction_score"] == 2
