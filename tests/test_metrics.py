"""Hand-computed checks for the frozen race-macro scoring protocol."""

import math

import pandas as pd
import pytest

from ml import evaluation


def race(order=("a", "b", "c", "d"), event="r1"):
    positions = {"a": 1, "b": 2, "c": 3, "d": 4}
    return [dict(event_id=event, driver_id=driver, actual_position=positions[driver],
                 prediction_score=index) for index, driver in enumerate(order)]


def test_swap_has_hand_computed_ranking_and_linear_ndcg():
    result = evaluation.evaluate_race(race(("b", "a", "d", "c")))
    assert result["metrics"]["rank_spearman"] == pytest.approx(0.6)
    assert result["metrics"]["position_mae"] == 1
    # Linear relevance, not the alternative exponential 2**relevance - 1.
    assert result["metrics"]["ndcg_at_3"] == pytest.approx(0.8174935137996165)
    assert result["metrics"]["podium_recall"] == pytest.approx(2 / 3)
    assert result["metrics"]["winner_accuracy"] == 0
    assert result["metrics"]["winner_top3_hit"] == 1
    assert result["metrics"]["top10_overlap"] == 1


def test_probability_metrics_are_race_multiclass_not_independent_binary():
    rows = race()
    for row, probability in zip(rows, [0.4, 0.3, 0.2, 0.1]):
        row["winner_probability"] = probability
    metrics = evaluation.evaluate_race(rows)["metrics"]
    assert metrics["winner_log_loss"] == pytest.approx(-math.log(0.4))
    assert metrics["winner_brier"] == pytest.approx(0.5)


def test_zero_winner_probability_uses_recorded_clipping():
    rows = race()
    for row, probability in zip(rows, [0, 1, 0, 0]):
        row["winner_probability"] = probability
    result = evaluation.evaluate_race(rows)
    assert result["metrics"]["winner_log_loss"] == pytest.approx(-math.log(1e-15))
    assert result["protocol"]["probability_clip"] == 1e-15


@pytest.mark.parametrize("probabilities", [[0.5] * 4, [-0.1, 0.3, 0.3, 0.5], [float("nan"), 0, 0, 1]])
def test_invalid_coherent_distribution_is_rejected(probabilities):
    rows = race()
    for row, probability in zip(rows, probabilities):
        row["winner_probability"] = probability
    with pytest.raises(ValueError):
        evaluation.evaluate_race(rows)


def test_absent_optional_metrics_remain_unavailable():
    metrics = evaluation.evaluate_race(race())["metrics"]
    assert metrics["winner_log_loss"] is None
    assert metrics["winner_brier"] is None
    assert metrics["pace_mae_seconds"] is None
    assert metrics["pace_rmse_seconds"] is None


def test_pace_errors_are_seconds_and_use_paired_targets_only():
    rows = race()
    rows[0].update(actual_pace_s=0, predicted_pace_s=1)
    rows[1].update(actual_pace_s=2, predicted_pace_s=5)
    rows[2].update(actual_pace_s=None, predicted_pace_s=100)
    result = evaluation.evaluate_race(pd.DataFrame(rows))
    assert result["metrics"]["pace_mae_seconds"] == 2
    assert result["metrics"]["pace_rmse_seconds"] == pytest.approx(math.sqrt(5))
    assert result["n_pace_pairs"] == 2


def test_completed_clean_pace_target_is_scored_independently_of_position_target():
    rows = [dict(event_id="pace", driver_id="a", actual_position=None,
                 outcome_status="completed", actual_pace_s=0.2, predicted_pace_s=0.5)]
    result = evaluation.evaluate_race(rows)
    assert result["metrics"]["position_mae"] is None
    assert result["metrics"]["pace_mae_seconds"] == pytest.approx(0.3)


def test_pandas_nullable_missing_position_stays_unavailable():
    rows = pd.DataFrame(race())
    rows["actual_position"] = pd.array([1, 2, 3, pd.NA], dtype="Int64")
    result = evaluation.evaluate_race(rows)
    assert result["metrics"]["position_mae"] is None
    assert result["outcome_status"] == "pending"


def test_explicit_pending_outcome_suppresses_pace_labels_too():
    rows = race()
    rows[0].update(outcome_status="pending", actual_pace_s=1, predicted_pace_s=2)
    assert evaluation.evaluate_race(rows)["metrics"]["pace_mae_seconds"] is None


def test_identical_scores_have_undefined_correlation_and_stable_id_tiebreak():
    rows = race(("d", "b", "a", "c"))
    for row in rows:
        row["prediction_score"] = 7
    result = evaluation.evaluate_race(rows)
    assert result["metrics"]["rank_spearman"] is None
    assert result["metrics"]["position_mae"] == 0
    assert result["predicted_order"] == ["a", "b", "c", "d"]


def test_single_entry_spearman_is_unavailable():
    assert evaluation.evaluate_race(race()[:1])["metrics"]["rank_spearman"] is None


def test_partial_score_ties_use_average_ranks_for_correlation():
    rows = race()
    rows[1]["prediction_score"] = rows[0]["prediction_score"]
    result = evaluation.evaluate_race(rows)
    assert result["metrics"]["rank_spearman"] == pytest.approx(0.9486832980505138)


def test_partial_predictions_and_partial_probabilities_report_unavailable():
    rows = race()
    rows[0]["prediction_score"] = None
    rows[0]["winner_probability"] = 0.5
    result = evaluation.evaluate_race(rows)
    assert result["metrics"]["position_mae"] is None
    assert result["metrics"]["winner_log_loss"] is None


def test_dns_dsq_excluded_and_classified_dnf_kept_without_closing_position_gaps():
    rows = race()
    rows[1]["status"] = "DNS"
    rows[2]["status"] = "DSQ"
    rows[3]["status"] = "DNF"
    result = evaluation.evaluate_race(rows)
    assert result["n_eligible"] == 2
    assert result["exclusions"] == {"DNS": 1, "DSQ": 1, "unclassified": 0}
    assert result["metrics"]["position_mae"] == 1
    assert result["metrics"]["podium_recall"] is None


def test_unclassified_dnf_excluded_but_missing_regular_outcome_is_pending():
    rows = race()
    rows[3].update(status="DNF", actual_position=None)
    assert evaluation.evaluate_race(rows)["n_eligible"] == 3
    rows[2]["actual_position"] = None
    result = evaluation.evaluate_race(rows)
    assert result["metrics"]["position_mae"] is None
    assert result["outcome_status"] == "pending"


def test_pending_outcome_never_looks_like_zero_error():
    rows = race()
    for row in rows:
        row["outcome_status"] = "pending"
    result = evaluation.evaluate_race(rows)
    assert all(value is None for value in result["metrics"].values())


@pytest.mark.parametrize("change", [
    {"driver_id": ""}, {"event_id": 1}, {"actual_position": True},
    {"actual_position": 1.5}, {"prediction_score": "fast"},
    {"prediction_score": float("inf")}, {"winner_probability": True},
    {"outcome_status": "invented"},
])
def test_boundary_validation_rejects_invalid_types_and_values(change):
    rows = race()
    rows[0].update(change)
    with pytest.raises((TypeError, ValueError)):
        evaluation.evaluate_race(rows)


def test_duplicate_identity_and_duplicate_actual_positions_rejected():
    rows = race()
    rows.append(rows[0].copy())
    with pytest.raises(ValueError):
        evaluation.evaluate_race(rows)
    rows = race()
    rows[1]["actual_position"] = 1
    with pytest.raises(ValueError):
        evaluation.evaluate_race(rows)


def test_explicit_predicted_ranks_and_higher_scores_are_supported():
    rows = race()
    for row in rows:
        row["predicted_rank"] = row.pop("prediction_score") + 1
    assert evaluation.evaluate_race(rows)["metrics"]["position_mae"] == 0
    rows = race()
    for row in rows:
        row["prediction_score"] *= -1
    assert evaluation.evaluate_race(rows, higher_score_better=True)["metrics"]["position_mae"] == 0
