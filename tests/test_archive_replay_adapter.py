import pandas as pd
import pytest
from pathlib import Path

from data.archive.replay_adapter import build_replay


class Reader:
    def tables(self, session_id):
        return {'meta': {'season': 2025, 'round_number': 1, 'session_code': 'FP1',
                         'version_id': 'pilot', 'created_at_utc': '2026-10-26T00:00:00+00:00'},
                'entries': pd.DataFrame([{'driver_id': 'nor', 'abbreviation': 'NOR', 'team_id': 'mclaren'}]),
                'laps': pd.DataFrame([
                    {'driver_id': 'nor', 'lap_number': 1, 'lap_end_clock_ns': 90_000_000_000,
                     'lap_time_ns': 80_000_000_000, 'pit_out_clock_ns': 12_000_000_000,
                     'provider_is_accurate': None, 'deleted': 0},
                    {'driver_id': 'nor', 'lap_number': 2, 'lap_end_clock_ns': 170_000_000_000,
                     'lap_time_ns': 80_000_000_000, 'provider_is_accurate': 1, 'deleted': None}]),
                'track_status': pd.DataFrame([{'session_clock_ns': 50_000_000_000, 'status_code_raw': '2', 'message_raw': 'Yellow'}]),
                'race_control': pd.DataFrame()}

    def dense(self, session_id, kind, driver_ids=None, columns=None):
        return pd.DataFrame([{'driver_id': 'nor', 'session_clock_ns': t * 100_000_000,
                              'x_raw': t, 'y_raw': t * 2} for t in range(10, 1800)])


def test_out_of_range_provider_markers_do_not_destroy_usable_replay():
    reader = Reader()
    tables = reader.tables('2025:1:FP1')
    tables['race_control'] = pd.DataFrame([
        {'session_clock_ns': 100_000_000_000, 'message_text': 'Observed flag'},
        {'session_clock_ns': 100_000_000_000_000, 'message_text': 'Misaligned clock'},
    ])
    reader.tables = lambda _: tables
    session = build_replay(reader, '2025:1:FP1')
    assert session.positions and session.timed_laps
    assert any(m['message'] == 'Observed flag' for m in session.markers)
    assert not any(m['message'] == 'Misaligned clock' for m in session.markers)
    assert 'excluded_markers_outside_replay_clock:1' in session.quality_flags
    assert len(tables['race_control']) == 2


def test_original_session_clock_unknown_flags_and_exact_pit_clock():
    replay = build_replay(Reader(), '2025:1:FP1')
    assert replay.positions[0]['clock_ms'] == 1000
    assert replay.laps[0]['clock_ms'] == 90_000
    assert replay.laps[0]['accurate'] is None
    assert replay.laps[0]['deleted'] is False
    assert replay.laps[1]['accurate'] is True
    assert replay.laps[1]['deleted'] is None
    pit = [m for m in replay.markers if 'pit out' in m['message']]
    assert len(pit) == 1 and pit[0]['clock_ms'] == 12_000
    assert len(replay.positions) <= 180
    assert replay.browser_payload()['position_samples_original'] == len(replay.positions)


def test_selection_is_bounded_and_validated():
    for ids in [[], ['missing'], ['nor', 'nor'], ['a', 'b', 'c', 'd', 'e']]:
        with pytest.raises(ValueError):
            build_replay(Reader(), '2025:1:FP1', driver_ids=ids)


def test_full_grid_and_causal_weather():
    reader = Reader()
    tables = reader.tables('2025:1:FP1')
    ids = [f'driver-{i}' for i in range(22)]
    tables['entries'] = pd.DataFrame([{'driver_id': d, 'abbreviation': f'D{i}', 'team_id': 'team'} for i, d in enumerate(ids)])
    tables['laps'] = pd.DataFrame([dict(tables['laps'].iloc[0], driver_id=d) for d in ids])
    tables['weather'] = pd.DataFrame([
        {'session_clock_ns': 10_000_000_000, 'air_temperature_c': 21, 'relative_humidity_pct': 75, 'rainfall_flag': 0},
        {'session_clock_ns': 60_000_000_000, 'air_temperature_c': 31, 'relative_humidity_pct': None, 'rainfall_flag': 1}])
    reader.tables = lambda _: tables
    reader.dense = lambda *a, **k: pd.DataFrame([{'driver_id': d, 'session_clock_ns': t * 1_000_000_000, 'x_raw': t, 'y_raw': t}
        for d in ids for t in range(1801 if d == ids[0] else 7201)])
    session = build_replay(reader, '2025:1:FP1', ids)
    assert len(session.entries) == 22
    payload = session.browser_payload()
    assert len(payload['positions']) == len(session.positions) <= 120_000
    for d in ids:
        clocks = [r['clock_ms'] for r in session.positions if r['driver_id'] == d]
        assert max(b-a for a,b in zip(clocks, clocks[1:])) <= 3000
    assert session.frame_at(9000).frame.weather == {}
    assert session.frame_at(59000).frame.weather['air_temperature_c'] == 21
    assert session.frame_at(60000).frame.weather['relative_humidity_pct'] is None
    assert session.frame_at(181000).frame.weather == {}
    assert session.frame_at(1_805_000).frame.drivers[ids[0]]['coordinates'] is not None
    assert session.frame_at(1_805_001).frame.drivers[ids[0]]['coordinates'] is None
    assert 'status' not in session.frame_at(1_805_001).frame.drivers[ids[0]]
    from data.replay import validate_commit
    assert validate_commit(dict(session_id=session.identity, frame_version=session.frame_version, sequence=1,
                                clock_ms=60000, driver_ids=ids, event='seek'), session)['driver_ids'] == ids


def test_nonmonotonic_lap_clock_excluded_without_modifying_source():
    reader = Reader()
    tables = reader.tables('2025:1:FP1')
    tables['laps'].loc[1, 'lap_end_clock_ns'] = 80_000_000_000
    original = tables['laps'].copy(deep=True)
    reader.tables = lambda _: tables
    replay = build_replay(reader, '2025:1:FP1')
    assert len(replay.laps) == 1
    pd.testing.assert_frame_equal(tables['laps'], original)
    assert any(flag.endswith(':1') for flag in replay.quality_flags)


def test_parser_fills_and_nonfinite_positions_are_not_observed_and_endpoint_survives():
    reader = Reader()
    dense = reader.dense('2025:1:FP1', 'positions')
    dense['quality_flags'] = '[]'
    dense['x_raw'] = dense.x_raw.astype(float)
    dense.loc[0, ['x_raw', 'y_raw', 'quality_flags']] = [0, 0, '["fastf1_parser_generated_fill"]']
    dense.loc[1, 'x_raw'] = float('inf')
    original = dense.copy(deep=True)
    reader.dense = lambda *args, **kwargs: dense
    replay = build_replay(reader, '2025:1:FP1')
    assert replay.positions[0]['clock_ms'] == 1200
    assert replay.positions[-1]['clock_ms'] == 179900
    assert not any(row['x'] == 0 for row in replay.positions)
    pd.testing.assert_frame_equal(dense, original)


def test_real_local_pilot_is_browser_bounded_when_present():
    from data.archive.reader import ArchiveReader
    root = Path(__file__).resolve().parents[1] / '.local/f1_2025_2026'
    if not (root / 'f1.sqlite3').exists():
        pytest.skip('Local pilot archive is not checked into Git')
    reader = ArchiveReader(root)
    sessions = reader.available_sessions()
    if not any(row['session_id'] == '2025:1:FP1' for row in sessions):
        pytest.skip('2025 Australian practice pilot not present')
    entries = reader.tables('2025:1:FP1')['entries']
    for count in (2, 4):
        replay = build_replay(reader, '2025:1:FP1', driver_ids=entries.driver_id.tolist()[:count])
        payload = replay.browser_payload()
        assert len(payload['positions']) == len(replay.positions) <= 120_000
        assert replay.positions and replay.timed_laps
        assert all(entry['abbreviation'] for entry in replay.entries)
        # The selected pilot streams are continuous, with original observations
        # within the three-second freshness budget after bucket sampling.
        for entry in replay.entries:
            clocks = [row['clock_ms'] for row in replay.positions if row['driver_id'] == entry['driver_id']]
            assert max(b - a for a, b in zip(clocks, clocks[1:])) < 3000
