from datetime import datetime, timedelta, timezone

import pytest

from data.contracts import DataSnapshot, DriverSessionRecord, EventKey, ForecastMode, SourceRecord
from ml.features import HistoryView, build_features, clean_laps

UTC = timezone.utc
CUTOFF = datetime(2025, 7, 5, 17, tzinfo=UTC)
EVENT = EventKey(2025, 12, 'silverstone')


def source(when=CUTOFF - timedelta(hours=2)):
    return SourceRecord('fixture', 'practice', when, available_at_utc=when)


def record(driver='NOR', when=CUTOFF - timedelta(hours=1), session='FP2', **values):
    return DriverSessionRecord(EVENT, session, driver, when, values, source())


def snapshot(records=(), mode='post_qualifying', entries=None):
    return DataSnapshot('snapshot-a', EVENT, CUTOFF,
                        {'mode': mode, 'entries': entries or [{'driver_id': 'NOR', 'team_id': 'MCL'},
                                                              {'driver_id': 'ROOKIE', 'team_id': 'NEW'}],
                         'driver_sessions': list(records)}, [source()], [])


def test_future_observations_do_not_change_features():
    observed = record(practice_pace_delta_s=0.2, practice_laps=8)
    before = build_features(snapshot([observed]))
    future = record(when=CUTOFF + timedelta(minutes=1), practice_pace_delta_s=-20, grid_position=1)
    after = build_features(snapshot([observed, future]))
    assert before.rows == after.rows
    assert after.rows[0]['practice_pace_delta_s'] == 0.2


def test_pre_weekend_ignores_all_current_weekend_observations():
    result = build_features(snapshot([record(grid_position=1, qualifying_position=1,
                                             practice_pace_delta_s=-1)], mode='pre_weekend'))
    assert result.rows[0]['grid_position'] is None
    assert result.rows[0]['practice_pace_delta_s'] is None


def test_missing_driver_keeps_identity_and_missing_inputs():
    result = build_features(snapshot([record(practice_pace_delta_s=0.2)]))
    rookie = next(row for row in result.rows if row['driver_id'] == 'ROOKIE')
    assert rookie['team_id'] == 'NEW'
    assert rookie['practice_pace_delta_s'] is None
    assert rookie['practice_missing'] is True


def test_late_source_is_not_smuggled_in_by_earlier_record_timestamp():
    late = SourceRecord('fixture', 'late', CUTOFF + timedelta(hours=1),
                        available_at_utc=CUTOFF + timedelta(hours=1))
    obs = DriverSessionRecord(EVENT, 'FP2', 'NOR', CUTOFF - timedelta(hours=1),
                              {'practice_pace_delta_s': -3}, late)
    assert build_features(snapshot([obs])).rows[0]['practice_pace_delta_s'] is None


def test_race_labels_are_not_feature_columns():
    result = build_features(snapshot([record(session='Q', actual_position=1, race_mean_lap_time=80,
                                             grid_position=3)]))
    assert 'actual_position' not in result.rows[0]
    assert 'race_mean_lap_time' not in result.rows[0]
    assert result.rows[0]['grid_position'] == 3


def test_current_race_observation_is_not_pre_race_feature():
    obs = DriverSessionRecord(EVENT, 'Race', 'NOR', CUTOFF - timedelta(hours=1),
                              {'practice_pace_delta_s': -9}, source())
    assert build_features(snapshot([obs])).rows[0]['practice_pace_delta_s'] is None


def test_rolling_form_only_reads_known_earlier_outcomes():
    rows = [dict(event_id='2025:1', driver_id='NOR', team_id='MCL', actual_position=4,
                 event_time_utc=CUTOFF - timedelta(days=20),
                 outcome_available_at_utc=CUTOFF - timedelta(days=19)),
            dict(event_id='2025:2', driver_id='NOR', team_id='MCL', actual_position=1,
                 event_time_utc=CUTOFF - timedelta(days=1),
                 outcome_available_at_utc=CUTOFF + timedelta(days=1))]
    features = build_features(snapshot(), HistoryView(tuple(rows)))
    assert features.rows[0]['driver_form_position'] == 4
    assert features.rows[0]['form_events'] == 1


def test_unseen_history_is_not_assigned_another_drivers_results():
    history = HistoryView((dict(event_id='2025:1', driver_id='VER', team_id='RBR', actual_position=1,
                                event_time_utc=CUTOFF - timedelta(days=2),
                                outcome_available_at_utc=CUTOFF - timedelta(days=1)),))
    result = build_features(snapshot(), history)
    assert all(row['driver_form_position'] is None for row in result.rows)


def test_observation_sequence_does_not_change_latest_eligible_feature():
    earlier = record(when=CUTOFF - timedelta(hours=3), session='Q', grid_position=8)
    later = record(when=CUTOFF - timedelta(hours=1), session='Q', grid_position=3)
    assert build_features(snapshot([earlier, later])).rows == build_features(snapshot([later, earlier])).rows


def test_duplicate_entry_ids_are_rejected():
    with pytest.raises(ValueError, match='duplicate'):
        build_features(snapshot(entries=[{'driver_id': 'NOR'}, {'driver_id': 'NOR'}]))


def test_clean_laps_excludes_invalid_and_neutralized_laps_without_fastest_only_bias():
    laps = [dict(lap_number=2, lap_time_s=91, accurate=True, track_status='1'),
            dict(lap_number=3, lap_time_s=94, accurate=True, track_status='1'),
            dict(lap_number=1, lap_time_s=95, accurate=True, track_status='1'),
            dict(lap_number=4, lap_time_s=100, accurate=True, track_status='4'),
            dict(lap_number=5, lap_time_s=90, accurate=True, track_status='1', deleted=True),
            dict(lap_number=6, lap_time_s=93, accurate=True, track_status='1', pit_in=True),
            dict(lap_number=7, lap_time_s=89, accurate=False, track_status='1')]
    result = clean_laps(laps)
    assert [row['lap_time_s'] for row in result] == [91, 94]


@pytest.mark.parametrize('value', [float('inf'), float('nan'), 'fast'])
def test_invalid_numeric_feature_is_rejected(value):
    with pytest.raises(ValueError):
        build_features(snapshot([record(practice_pace_delta_s=value)]))


def test_sprint_qualifying_is_not_race_qualifying_or_grid():
    result = build_features(snapshot([record(session='SQ', qualifying_position=1, grid_position=1)]))
    assert result.rows[0]['qualifying_position'] is None
    assert result.rows[0]['grid_position'] is None
    assert result.rows[0]['qualifying_missing'] is True


def test_weather_features_require_known_issue_and_valid_time():
    unknown = record(session='WEATHER_FORECAST', weather_rain_probability=40,
                     weather_valid_at_utc=CUTOFF + timedelta(days=1))
    assert build_features(snapshot([unknown])).rows[0]['weather_rain_probability'] is None


def test_weather_issue_after_cutoff_or_valid_time_cannot_enter_features():
    issued = CUTOFF - timedelta(hours=1)
    src = SourceRecord('weather', 'forecast', issued, issued_at_utc=issued)
    obs = DriverSessionRecord(EVENT, 'WEATHER_FORECAST', 'NOR', issued,
                              {'weather_rain_probability': 40, 'weather_valid_at_utc': CUTOFF + timedelta(days=1)}, src)
    assert build_features(snapshot([obs])).rows[0]['weather_rain_probability'] == 40
    invalid = DriverSessionRecord(EVENT, 'WEATHER_FORECAST', 'NOR', issued,
                                  {'weather_rain_probability': 40, 'weather_valid_at_utc': issued - timedelta(hours=1)}, src)
    assert build_features(snapshot([invalid])).rows[0]['weather_rain_probability'] is None
