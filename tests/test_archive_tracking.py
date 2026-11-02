import pytest
from flows.archive_tracking import archive_outcome


def test_known_result_with_a_different_field_is_not_an_awaiting_result(tmp_path):
    from monitoring import TrackingQueries, TrackingStore
    from test_tracking import forecast, outcome
    store = TrackingStore(tmp_path / 'tracking.sqlite3')
    run = forecast()
    run['ranking'] = run['ranking'][:2]
    store.save_forecast(run)
    store.append_outcome(outcome())
    query = TrackingQueries(store, owner_id='alice')
    assert query.outcome_state('run-1') == 'roster_mismatch'
    assert TrackingQueries(store, owner_id='bob').outcome_state('run-1') is None


def test_outcomes_keep_real_retrieval_lineage_and_do_not_score_locked_year():
    label = dict(event_id='2025:1', driver_id='driver-one', actual_position=1,
        status='FINISHED', original_fetched_at_utc='2026-10-31T01:00:00+00:00', source_id='pinned-version')
    outcome = archive_outcome('2025:1', [label])
    assert outcome['published_at_utc'] == label['original_fetched_at_utc']
    assert outcome['historical_publication_verified'] is False
    assert outcome['entries'][0]['actual_position'] == 1
    assert outcome == archive_outcome('2025:1', [label])
    with pytest.raises(ValueError, match='holdout'):
        archive_outcome('2026:1', [dict(label, event_id='2026:1')])
    with pytest.raises(ValueError, match='labels'):
        archive_outcome('2025:2', [label])
