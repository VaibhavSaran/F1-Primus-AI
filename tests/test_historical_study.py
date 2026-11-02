from datetime import datetime, timezone
import pytest


def test_study_joins_independent_results_not_embedded_prediction_labels():
    from ml.historical_study import join_outcomes
    predictions = [dict(event_id='2025:19', driver_id='a', prediction_score=2, actual_position=99),
                   dict(event_id='2025:19', driver_id='b', prediction_score=1)]
    labels = [dict(event_id='2025:19', driver_id='a', actual_position=1, status='Finished'),
              dict(event_id='2025:19', driver_id='b', actual_position=2, status='Finished')]
    rows = join_outcomes(predictions, labels)
    assert rows[0]['actual_position'] == 1
    assert rows[0]['predicted_rank'] == 2
    assert rows[0]['position_error'] == 1


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'other_event', 'nan'])
def test_study_rejects_invalid_or_incomplete_predictions(change):
    from ml.historical_study import join_outcomes
    predictions = [dict(event_id='2025:19', driver_id='a', prediction_score=1)]
    labels = [dict(event_id='2025:19', driver_id='a', actual_position=1)]
    if change == 'missing': labels.append(dict(event_id='2025:19', driver_id='b', actual_position=2))
    elif change == 'duplicate': predictions *= 2
    elif change == 'other_event': predictions[0]['event_id'] = '2025:18'
    else: predictions[0]['prediction_score'] = float('nan')
    with pytest.raises(ValueError): join_outcomes(predictions, labels)


def test_paired_intervals_use_only_same_race_pairs_and_are_reproducible():
    from ml.historical_study import paired_difference
    candidate = [dict(event_id='a', metrics=dict(rank_spearman=.5)),
                 dict(event_id='b', metrics=dict(rank_spearman=.9))]
    baseline = [dict(event_id='a', metrics=dict(rank_spearman=.4)),
                dict(event_id='c', metrics=dict(rank_spearman=.1))]
    result = paired_difference(candidate, baseline, 'rank_spearman', samples=50)
    assert result['n_events'] == 1
    assert result['mean_delta'] == pytest.approx(.1)
    assert result == paired_difference(candidate, baseline, 'rank_spearman', samples=50)


def test_pace_extraction_rejects_target_race_and_cutoff_boundary():
    from ml.historical_study import eligible_pace_session
    cutoff = datetime(2025,10,19,18,tzinfo=timezone.utc)
    base = dict(event_id='2025:19', session_code='FP1', completed_at_utc='2025-10-17T19:00:00+00:00')
    assert eligible_pace_session(base, target_event_id='2025:19', cutoff_utc=cutoff)
    assert not eligible_pace_session(dict(base, session_code='R'), target_event_id='2025:19', cutoff_utc=cutoff)
    assert not eligible_pace_session(dict(base, completed_at_utc=cutoff.isoformat()), target_event_id='2025:19', cutoff_utc=cutoff)
    assert not eligible_pace_session(dict(base, event_id='2026:1'), target_event_id='2025:19', cutoff_utc=cutoff)


def test_immutable_study_json_uses_canonical_bytes_across_windows_newlines(tmp_path):
    from ml.historical_study import _write_json
    path=tmp_path/'value.json'
    _write_json(path,dict(value=1))
    assert path.read_bytes()==b'{\n  "value": 1\n}\n'
