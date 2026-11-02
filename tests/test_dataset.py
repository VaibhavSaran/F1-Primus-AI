from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from data.contracts import DataSnapshot, EventKey, SourceRecord
from data.snapshots import load_snapshot, save_snapshot
from ml.dataset import OutcomeRecord, build_dataset

UTC = timezone.utc
CUTOFF = datetime(2025, 7, 5, 17, tzinfo=UTC)
EVENT = EventKey(2025, 12, 'silverstone')


def snapshot():
    src = SourceRecord('fixture', 'entries', CUTOFF, available_at_utc=CUTOFF)
    return DataSnapshot('pilot', EVENT, CUTOFF,
                        {'mode': 'post_qualifying', 'entries': [{'driver_id': 'NOR', 'team_id': 'MCL'}],
                         'driver_sessions': []}, [src], [])


def outcome(position=1, status='Finished'):
    return OutcomeRecord(EVENT.event_id, 'NOR', position, CUTOFF + timedelta(days=1),
                         CUTOFF + timedelta(days=1, hours=3), status)


def test_label_changes_do_not_change_features():
    first = build_dataset([snapshot()], [outcome(1)])
    second = build_dataset([snapshot()], [outcome(9)])
    feature_keys = first.feature_columns
    assert {key: first.rows[0][key] for key in feature_keys} == {key: second.rows[0][key] for key in feature_keys}
    assert first.rows[0]['actual_position'] == 1
    assert second.rows[0]['actual_position'] == 9


def test_dataset_preserves_event_driver_cutoff_and_label_provenance():
    row = build_dataset([snapshot()], [outcome()]).rows[0]
    assert row['event_id'] == EVENT.event_id
    assert row['driver_id'] == 'NOR'
    assert row['forecast_cutoff_utc'] == CUTOFF
    assert row['outcome_available_at_utc'] > CUTOFF
    assert row['practice_pace_delta_s'] is None


def test_missing_outcome_is_pending_not_zero():
    row = build_dataset([snapshot()], []).rows[0]
    assert row['actual_position'] is None
    assert row['outcome_status'] == 'pending'


@pytest.mark.parametrize('status', ['DNS', 'DSQ'])
def test_ineligible_classification_is_recorded_without_inventing_position(status):
    row = build_dataset([snapshot()], [outcome(1, status)]).rows[0]
    assert row['eligible'] is False
    assert row['status'] == status


def test_duplicate_outcomes_are_rejected():
    with pytest.raises(ValueError, match='duplicate'):
        build_dataset([snapshot()], [outcome(), outcome()])


def test_duplicate_event_driver_snapshots_are_rejected():
    with pytest.raises(ValueError, match='duplicate'):
        build_dataset([snapshot(), snapshot()], [outcome()])


def test_snapshot_round_trip_and_content_identity(tmp_path):
    original = snapshot()
    path = save_snapshot(original, tmp_path)
    restored = load_snapshot(path)
    assert restored.event == original.event
    assert restored.as_of_utc == original.as_of_utc
    assert restored.payloads == original.payloads
    assert save_snapshot(original, tmp_path) == path


def test_snapshot_id_cannot_escape_store(tmp_path):
    with pytest.raises(ValueError):
        save_snapshot(replace(snapshot(), snapshot_id='../escape'), tmp_path)


def test_mutated_saved_snapshot_is_detected(tmp_path):
    path = save_snapshot(snapshot(), tmp_path)
    text = path.read_text(encoding='utf-8').replace('MCL', 'RBR')
    path.write_text(text, encoding='utf-8')
    with pytest.raises(ValueError, match='integrity'):
        load_snapshot(path)

def test_dataset_rows_feed_the_baseline_and_evaluator():
    from ml.baselines import grid_baseline
    from ml.evaluation import evaluate_predictions
    dataset = build_dataset([snapshot()], [outcome()])
    predictions = grid_baseline([dict(row, grid_position=1) for row in dataset.rows])
    assert evaluate_predictions(predictions)['metrics']['position_mae'] == 0


@pytest.mark.parametrize('status', ['DNS', 'DSQ', 'DNF'])
def test_resolved_excluded_outcomes_do_not_make_race_pending(status):
    from ml.evaluation import evaluate_predictions
    src = snapshot()
    payload = dict(src.payloads, entries=[{'driver_id': 'NOR', 'team_id': 'MCL'},
                                         {'driver_id': 'OTHER', 'team_id': 'OTHER'}])
    outcomes = [outcome(), OutcomeRecord(EVENT.event_id, 'OTHER', None,
                                         CUTOFF + timedelta(days=1), CUTOFF + timedelta(days=1, hours=3), status)]
    rows = build_dataset([replace(src, payloads=payload)], outcomes).rows
    assert rows[1]['outcome_status'] == 'completed'
    assert evaluate_predictions([dict(row, prediction_score=index) for index, row in enumerate(rows)])['metrics']['position_mae'] == 0


def test_history_iterators_are_reusable_across_snapshots():
    from ml.features import HistoryView
    prior = [dict(event_id='2025:1', driver_id='NOR', team_id='MCL', actual_position=3,
                   event_time_utc=CUTOFF - timedelta(days=5),
                   outcome_available_at_utc=CUTOFF - timedelta(days=4))]
    second = replace(snapshot(), snapshot_id='second', event=EventKey(2025, 13, 'spa'))
    expected = build_dataset([snapshot(), second], [], history=prior).rows
    actual = build_dataset([snapshot(), second], [], history=iter(prior)).rows
    assert [row['driver_form_position'] for row in actual] == [3, 3]
    assert actual == expected


def test_provider_records_round_trip(tmp_path):
    from data.weather import WeatherRecord
    from data.news import NewsArticle
    from data.sessions import SessionRecord
    src = snapshot().sources[0]
    records = [WeatherRecord(CUTOFF + timedelta(days=1), 'forecast', {'temperature_2m': 20}, src),
               NewsArticle('Example', 'https://example.com', 'Evidence', src),
               SessionRecord(EVENT, 'FP2', 'Practice 2', CUTOFF - timedelta(days=1))]
    original = replace(snapshot(), payloads=dict(snapshot().payloads, provider_records=records))
    assert load_snapshot(save_snapshot(original, tmp_path)).payloads == original.payloads
