"""Retained timing stream projection and its file trust boundary."""
from hashlib import sha256
from pathlib import Path
import pickle

import pandas as pd
import pytest

from data.archive import Archive
from data.archive.reader import ArchiveReader


@pytest.fixture
def timing_archive(tmp_path):
    archive = Archive(tmp_path / 'archive')
    calendar = archive.source('fixture', 'calendar', b'{}', '2026-10-23T00:00:00Z')
    archive.save_calendar([dict(event_id='2020:1', season=2020, round_number=1, event_name='Test')],
                          [dict(session_id='2020:1:R', event_id='2020:1', session_code='R')], calendar)
    frame = pd.DataFrame(dict(Time=pd.to_timedelta([10, 12, 14], unit='s'), Driver=['44', '44', '0'],
                              Position=[1, 2, 3], GapToLeader=['', 'LAP 1', '2.5'],
                              IntervalToPositionAhead=['', '+1.5', '1.0']))
    source = archive.source('fastf1_parser_cache', 'timing', pickle.dumps(dict(version=15, data=(pd.DataFrame(), frame, []))),
                            '2026-10-23T00:00:00Z', endpoint_kind='_extended_timing_data')
    rows = dict(drivers=[dict(driver_id='ham', display_name='Lewis Hamilton')],
                session_entries=[dict(driver_id='ham', driver_number='44')],
                laps=[dict(driver_id='ham', lap_number=1, lap_position=1, lap_end_clock_ns=90_000_000_000)])
    version = archive.publish_session('2020:1:R', rows, [], [source], {})
    return archive, source, version


def test_timing_keeps_clock_identity_and_mixed_gap_strings(timing_archive):
    archive, _, _ = timing_archive
    frame = ArchiveReader(archive.root).timing('2020:1:R')
    assert frame.driver_id.tolist() == ['ham', 'ham']
    assert frame.session_time_s.tolist() == [10., 12.]
    assert frame.gap_to_leader_raw.tolist() == [None, 'LAP 1']
    assert frame.attrs['timing_basis'] == 'timing_stream'


def test_missing_source_uses_only_lap_end_position(timing_archive):
    archive, source, _ = timing_archive
    with archive.connect() as connection:
        connection.execute('DELETE FROM session_version_sources WHERE source_id=?', (source,))
    frame = ArchiveReader(archive.root).timing('2020:1:R')
    assert frame.session_time_s.tolist() == [90.]
    assert frame.position.tolist() == [1]
    assert frame.gap_to_leader_raw.tolist() == [None]
    assert frame.attrs == dict(timing_basis='lap_end_position', source_status='absent')



def _source_path(archive, source):
    with archive.connect() as connection:
        row = connection.execute('SELECT raw_path FROM source_objects WHERE source_id=?', (source,)).fetchone()
    return archive.root / row[0]


@pytest.mark.parametrize('damage', ['hash', 'missing', 'corrupt', 'path', 'version', 'provider'])
def test_invalid_retained_sources_fall_back_without_breaking_replay(timing_archive, damage):
    archive, source, _ = timing_archive
    path = _source_path(archive, source)
    if damage == 'missing':
        path.unlink()
    elif damage in ('hash', 'corrupt'):
        path.write_bytes(b'not a pickle')
        if damage == 'corrupt':
            # Even a catalog matching the corrupt bytes cannot permit decoding.
            digest = sha256(path.read_bytes()).hexdigest()
            new_path = archive.root / 'raw' / 'fastf1_parser_cache' / digest / 'payload.bin'
            new_path.parent.mkdir(parents=True)
            new_path.write_bytes(path.read_bytes())
            with archive.connect() as connection:
                connection.execute('UPDATE source_objects SET payload_sha256=?,raw_path=? WHERE source_id=?',
                                   (digest, new_path.relative_to(archive.root).as_posix(), source))
    elif damage == 'version':
        cached = dict(version=999, data=(pd.DataFrame(), pd.DataFrame(), []))
        payload = pickle.dumps(cached)
        digest = sha256(payload).hexdigest()
        new_path = archive.root / 'raw' / 'fastf1_parser_cache' / digest / 'payload.bin'
        new_path.parent.mkdir(parents=True)
        new_path.write_bytes(payload)
        with archive.connect() as connection:
            connection.execute('UPDATE source_objects SET payload_sha256=?,raw_path=? WHERE source_id=?',
                               (digest, new_path.relative_to(archive.root).as_posix(), source))
    else:
        with archive.connect() as connection:
            if damage == 'path':
                connection.execute('UPDATE source_objects SET raw_path=? WHERE source_id=?', ('../outside.bin', source))
            else:
                connection.execute('UPDATE source_objects SET provider=? WHERE source_id=?', ('other', source))
    frame = ArchiveReader(archive.root).timing('2020:1:R')
    assert frame.attrs == dict(timing_basis='lap_end_position', source_status='invalid')
    assert frame.position.tolist() == [1]


def test_forbidden_pickle_global_is_never_executed(timing_archive, monkeypatch):
    archive, source, _ = timing_archive
    import builtins
    class Dangerous:
        def __reduce__(self):
            return eval, ('40 + 2',)
    payload = pickle.dumps(Dangerous())
    digest = sha256(payload).hexdigest()
    path = archive.root / 'raw' / 'fastf1_parser_cache' / digest / 'payload.bin'
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    with archive.connect() as connection:
        connection.execute('UPDATE source_objects SET payload_sha256=?,raw_path=? WHERE source_id=?',
                           (digest, path.relative_to(archive.root).as_posix(), source))
    def forbidden(*args):
        raise AssertionError('untrusted eval executed')
    monkeypatch.setattr(builtins, 'eval', forbidden)
    assert ArchiveReader(archive.root).timing('2020:1:R').attrs['source_status'] == 'invalid'


def test_timing_limits_apply_before_decoding(timing_archive, monkeypatch):
    archive, _, _ = timing_archive
    import data.archive.timing as timing
    monkeypatch.setattr(timing, 'MAX_TIMING_BYTES', 1)
    assert ArchiveReader(archive.root).timing('2020:1:R').attrs['source_status'] == 'invalid'


@pytest.mark.parametrize('session_id', ['2020:6:R', '2026:3:R', '2020:8:R'])
def test_real_retained_timing_when_present(session_id):
    root = Path(__file__).resolve().parents[1] / '.local/f1_2025_2026'
    if not (root / 'f1.sqlite3').exists():
        pytest.skip('Local archive absent')
    reader = ArchiveReader(root)
    if session_id not in {r['session_id'] for r in reader.available_sessions()}:
        pytest.skip('Session absent')
    frame = reader.timing(session_id)
    assert frame.attrs['timing_basis'] == 'timing_stream'
    assert len(frame) > 1000
    assert frame.session_time_s.between(0, 50000).all()
    assert set(frame.driver_id).issubset(set(reader.tables(session_id)['entries'].driver_id))
    assert frame.position.notna().any()


def test_scoped_entry_is_preserved_by_exact_session_number():
    from data.archive.timing import normalize_stream
    frame = pd.DataFrame(dict(Time=pd.to_timedelta([12, 13], unit='s'), Driver=['45', '65535'],
                              Position=[2, 1], GapToLeader=['+1.2', ''], IntervalToPositionAhead=['+1.2', '']))
    result = normalize_stream(frame, [dict(driver_id='fastf1:2020:1:R:number:45', driver_number='45'),
                                      dict(driver_id='placeholder', driver_number='65535')])
    assert result.driver_id.tolist() == ['fastf1:2020:1:R:number:45']


def test_linked_raw_directory_is_rejected_before_open(timing_archive, monkeypatch):
    from data.archive import timing
    archive, _, _ = timing_archive
    original = timing._linked
    monkeypatch.setattr(timing, '_linked', lambda path: path.name == 'raw' or original(path))
    result = ArchiveReader(archive.root).timing('2020:1:R')
    assert result.attrs['source_status'] == 'invalid'


def test_practice_never_has_race_standings(timing_archive):
    archive, _, _ = timing_archive
    with archive.connect() as connection:
        connection.execute("UPDATE sessions SET session_code='FP1' WHERE session_id='2020:1:R'")
    result = ArchiveReader(archive.root).timing('2020:1:R')
    assert result.empty
    assert result.attrs['source_status'] == 'not_applicable'


def test_unlinked_source_never_supplies_current_publication(timing_archive):
    archive, _, version = timing_archive
    with archive.connect() as connection:
        connection.execute('DELETE FROM session_version_sources WHERE version_id=?', (version,))
    # The retained source still exists in the catalog and on disk.
    with archive.connect() as connection:
        assert connection.execute("SELECT count(*) FROM source_objects WHERE endpoint_kind='_extended_timing_data'").fetchone()[0] == 1
    result = ArchiveReader(archive.root).timing('2020:1:R')
    assert result.attrs['source_status'] == 'absent'
    assert result.session_time_s.tolist() == [90.]


def test_replay_pins_timing_to_tables_when_current_publication_changes(timing_archive):
    from data.archive.replay_adapter import build_replay
    archive, source, original_version = timing_archive
    reader = ArchiveReader(archive.root)
    original_tables = reader.tables
    def tables_then_publish(session_id):
        tables = original_tables(session_id)
        rows = dict(drivers=[dict(driver_id='ham', display_name='Lewis Hamilton')],
                    session_entries=[dict(driver_id='ham', driver_number='44')],
                    laps=[dict(driver_id='ham', lap_number=i, lap_position=1,
                               lap_time_ns=80_000_000_000, lap_end_clock_ns=i*90_000_000_000)
                          for i in (1, 2)])
        # A new richer publication intentionally has no linked timing source.
        archive.publish_session(session_id, rows, [], [], {})
        return tables
    reader.tables = tables_then_publish
    replay = build_replay(reader, '2020:1:R', ['ham'])
    assert replay.source['source_id'] == original_version
    assert replay.timing_basis == 'timing_stream'
    assert len(replay.timing) == 2
    assert reader.timing('2020:1:R').attrs['source_status'] == 'absent'
    assert reader.timing('2020:1:R', version_id=original_version).attrs['source_status'] == 'available'


def test_timing_rejects_wrong_session_or_unpublished_version(timing_archive):
    archive, _, version = timing_archive
    reader = ArchiveReader(archive.root)
    for bad in ('not-a-version', 'other-session'):
        with pytest.raises(ValueError, match='published version'):
            reader.timing('2020:1:R', version_id=bad)
    with archive.connect() as connection:
        connection.execute("UPDATE session_versions SET status='staged' WHERE version_id=?", (version,))
    with pytest.raises(ValueError):
        reader.timing('2020:1:R', version_id=version)


def test_retained_state_snapshot_clears_blank_gaps_and_invalid_position(timing_archive):
    from data.archive.replay_adapter import build_replay
    archive, source, _ = timing_archive
    frame = pd.DataFrame(dict(Time=pd.to_timedelta([10, 20], unit='s'), Driver=['44', '44'],
                              Position=[1, 0], GapToLeader=['+1.2', ''],
                              IntervalToPositionAhead=['+0.8', '']))
    payload = pickle.dumps(dict(version=15, data=(pd.DataFrame(), frame, [])))
    digest = sha256(payload).hexdigest()
    path = archive.root / 'raw' / 'fastf1_parser_cache' / digest / 'payload.bin'
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    with archive.connect() as connection:
        connection.execute('UPDATE source_objects SET payload_sha256=?,raw_path=? WHERE source_id=?',
                           (digest, path.relative_to(archive.root).as_posix(), source))
    session = build_replay(ArchiveReader(archive.root), '2020:1:R', ['ham'])
    before = session.frame_at(10000).frame.drivers['ham']
    after = session.frame_at(20000).frame.drivers['ham']
    assert before['position'] == 1 and before['gap_s'] == 1.2 and before['interval_s'] == .8
    assert after['position'] is None and after['gap_s'] is None and after['interval_s'] is None


def test_only_retained_stream_rows_are_marked_as_complete_state(timing_archive, monkeypatch):
    from data.archive.replay_adapter import build_replay, ReplaySession
    archive, source, _ = timing_archive
    original = ReplaySession.from_mapping
    records = []
    def capture(data):
        records.append(data)
        return original(data)
    monkeypatch.setattr(ReplaySession, 'from_mapping', capture)
    reader = ArchiveReader(archive.root)
    build_replay(reader, '2020:1:R', ['ham'])
    with archive.connect() as connection:
        connection.execute('DELETE FROM session_version_sources WHERE source_id=?', (source,))
    build_replay(reader, '2020:1:R', ['ham'])
    assert records[0]['timing'] and all(row['state_snapshot'] is True for row in records[0]['timing'])
    assert records[1]['timing'] and all(row['state_snapshot'] is False for row in records[1]['timing'])
