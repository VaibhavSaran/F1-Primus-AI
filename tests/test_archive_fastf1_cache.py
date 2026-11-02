from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3

import fastf1
import pandas as pd
import pytest

from data.archive.collect import Collector, cutoff_utc
from data.archive.fastf1_cache import add_original_cached_streams, archived_event, configured_cache, seed_trusted_cache
from data.archive.fastf1_extract import calendar_rows
from data.archive.store import Archive


def schedule_fixture():
    values = dict(round_number=1, country='Australia', location='Melbourne',
        official_event_name='Australian GP', event_name='Australian Grand Prix',
        event_date='2025-03-16', event_format='conventional', gmt_offset='+11:00',
        f1_api_support=True)
    for index, (name, stamp) in enumerate([
        ('Practice 1', '2025-03-14T00:30:00'), ('Practice 2', '2025-03-14T16:00:00'),
        ('Practice 3', '2025-03-15T12:30:00'), ('Qualifying', '2025-03-15T16:00:00'),
        ('Race', '2025-03-16T15:00:00')], 1):
        values[f'session{index}'] = name
        values[f'session{index}_date'] = stamp
    return {name: {'0': value} for name, value in values.items()}


def setup_archive(tmp_path):
    from data.archive.collect import schedule_frame

    archive = Archive(tmp_path / 'archive')
    payload = schedule_fixture()
    source_id = archive.source('fastf1_schedule', 'schedule_2025', json.dumps(payload).encode(), '2026-10-23T00:00:00Z')
    events, sessions = calendar_rows(2025, schedule_frame(payload))
    archive.save_calendar(events, sessions, source_id)
    return archive


def test_seed_uses_sqlite_snapshot_and_never_overwrites_existing_cache(tmp_path):
    source, target = tmp_path / 'trusted', tmp_path / 'new-cache'
    relative = Path('2025/weekend/session/timing_data.ff1pkl')
    (source / relative).parent.mkdir(parents=True)
    (source / relative).write_bytes(b'trusted pickle bytes copied without evaluation')
    with sqlite3.connect(source / 'fastf1_http_cache.sqlite') as conn:
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('CREATE TABLE fixture(value INTEGER)')
        conn.execute('INSERT INTO fixture VALUES(42)')
        conn.commit()
        result = seed_trusted_cache(source, target)
        with sqlite3.connect(target / 'fastf1_http_cache.sqlite') as snapshot:
            assert snapshot.execute('SELECT value FROM fixture').fetchone()[0] == 42
    assert (target / relative).read_bytes() == (source / relative).read_bytes()
    assert result['copied'] == 2
    (source / relative).write_bytes(b'new source revision')
    seed_trusted_cache(source, target)
    assert (target / relative).read_bytes() == b'trusted pickle bytes copied without evaluation'


def test_seed_rejects_linked_files_instead_of_importing_outside_trusted_root(tmp_path):
    source, target = tmp_path / 'trusted', tmp_path / 'new-cache'
    source.mkdir()
    outside = tmp_path / 'outside.ff1pkl'
    outside.write_bytes(b'outside')
    link = source / '2025'
    try:
        link.symlink_to(tmp_path, target_is_directory=True)
    except OSError:
        pytest.skip('Windows symlink privilege unavailable')
    result = seed_trusted_cache(source, target)
    assert result['copied'] == 0
    assert result['skipped_links'] >= 1


def test_archived_event_has_local_api_date_and_utc_schedule_without_network(tmp_path, monkeypatch):
    archive = setup_archive(tmp_path)
    monkeypatch.setattr(fastf1, 'get_event_schedule', lambda *args, **kwargs: pytest.fail('network calendar used'))
    event = archived_event(archive, '2025:1')
    session = event.get_session('FP1')
    assert session.api_path.endswith('/2025-03-14_Practice_1/')
    assert session.date == pd.Timestamp('2025-03-13T13:30:00')
    assert event.year == 2025


def test_cache_context_restores_existing_global_configuration_on_exit(tmp_path):
    names = ['_CACHE_DIR', '_requests_session_cached', '_FORCE_RENEW', '_IGNORE_VERSION',
             '_tmp_disabled', '_http_tmp_disabled', '_func_tmp_disabled', '_ci_mode',
             '_default_cache_enabled', '_no_cached_warned']
    before = {name: getattr(fastf1.Cache, name) for name in names if hasattr(fastf1.Cache, name)}
    with configured_cache(tmp_path / 'cache', offline=True):
        assert fastf1.Cache._requests_session_cached.settings.only_if_cached is True
        assert all(getattr(fastf1.Cache, name) is False for name in before if name.endswith('tmp_disabled'))
    assert all(getattr(fastf1.Cache, name) is value for name, value in before.items())


def test_disabled_cache_cannot_bypass_scoped_offline_mode_and_is_restored(tmp_path, monkeypatch):
    import requests
    names = [name for name in ('_tmp_disabled', '_http_tmp_disabled', '_func_tmp_disabled')
             if hasattr(fastf1.Cache, name)]
    for name in names:
        monkeypatch.setattr(fastf1.Cache, name, True)
    def forbidden(*args, **kwargs):
        pytest.fail('offline mode attempted a network send')
    monkeypatch.setattr(requests.Session, 'send', forbidden)
    with pytest.raises(RuntimeError, match='fixture'):
        with configured_cache(tmp_path / 'cache', offline=True):
            assert fastf1.Cache.requests_get('https://example.org/uncached-fixture').status_code == 504
            raise RuntimeError('fixture')
    assert all(getattr(fastf1.Cache, name) is True for name in names)


class NoNetwork:
    @contextmanager
    def install_fastf1(self):
        yield self


def test_collector_uses_archived_event_and_cached_optional_channel_flags(tmp_path, monkeypatch):
    from data.archive import fastf1_cache

    archive = setup_archive(tmp_path)
    event = archived_event(archive, '2025:1')
    session_path = event.get_session('R').api_path.removeprefix('/static/')
    source = tmp_path / 'trusted-cache'
    for name in ['session_info', 'driver_info', '_extended_timing_data', 'timing_app_data',
                 'track_status_data', 'session_status_data', 'lap_count']:
        path = source / session_path / (name + '.ff1pkl')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'trusted fixture')
    monkeypatch.setattr(fastf1_cache, 'TRUSTED_CACHE', source)
    monkeypatch.setattr(fastf1, 'get_session', lambda *args, **kwargs: pytest.fail('network calendar used'))
    calls = []

    def load(session, **kwargs):
        calls.append(kwargs)
        session._results = pd.DataFrame([dict(DriverNumber='1', DriverId='one')])
        session._laps = pd.DataFrame([dict(DriverNumber='1', LapNumber=1, LapTime=pd.Timedelta(seconds=90))])

    monkeypatch.setattr(fastf1.core.Session, 'load', load)
    collector = Collector(archive, end=cutoff_utc('2026-10-23', 'America/Chicago'), http=NoNetwork())
    task_id = collector.queue('2025:1:R', 'fastf1_session', season=2025, event_id='2025:1',
        session_id='2025:1:R', config={'year': 2025, 'round': 1, 'code': 'R'})
    state = {'config': {'year': 2025, 'round': 1, 'code': 'R'}}
    collector.fastf1_session({'task_id': task_id, 'session_id': '2025:1:R'}, state)
    assert calls == [dict(laps=True, telemetry=False, weather=False, messages=False)]
    with archive.connect() as conn:
        assert conn.execute('SELECT count(*) FROM laps').fetchone()[0] == 1
        assert conn.execute('SELECT status FROM collection_tasks WHERE task_id=?', (task_id,)).fetchone()[0] == 'partial'
        assert conn.execute("SELECT status FROM coverage WHERE dataset_kind='positions'").fetchone()[0] == 'blocked'


def test_collector_contains_sdk_system_exit_and_keeps_loaded_evidence(tmp_path, monkeypatch):
    from data.archive import fastf1_cache

    archive = setup_archive(tmp_path)
    source = tmp_path / 'trusted-cache'
    event = archived_event(archive, '2025:1')
    cache_file = source / event.get_session('R').api_path.removeprefix('/static/') / 'driver_info.ff1pkl'
    cache_file.parent.mkdir(parents=True)
    cache_file.write_bytes(b'trusted fixture')
    monkeypatch.setattr(fastf1_cache, 'TRUSTED_CACHE', source)

    def load(session, **kwargs):
        session._results = pd.DataFrame([dict(DriverNumber='1', DriverId='one')])
        raise SystemExit('missing cached optional component')

    monkeypatch.setattr(fastf1.core.Session, 'load', load)
    collector = Collector(archive, end=cutoff_utc('2026-10-23', 'America/Chicago'), http=NoNetwork())
    task_id = collector.queue('2025:1:R', 'fastf1_session', season=2025, event_id='2025:1', session_id='2025:1:R')
    state = {'config': {'year': 2025, 'round': 1, 'code': 'R'}}
    collector.fastf1_session({'task_id': task_id, 'session_id': '2025:1:R'}, state)
    with archive.connect() as conn:
        assert conn.execute('SELECT count(*) FROM session_entries').fetchone()[0] == 1
        assert conn.execute('SELECT count(*) FROM source_objects WHERE provider=?', ('fastf1_trusted_local_cache',)).fetchone()[0] == 1
    assert 'system_exit' in state['load_error']


def test_collector_reenables_online_sdk_load_only_inside_paced_transport(tmp_path, monkeypatch):
    from data.archive import fastf1_cache

    archive = setup_archive(tmp_path)
    monkeypatch.setattr(fastf1_cache, 'TRUSTED_CACHE', tmp_path / 'no-existing-cache')
    active = []

    class Transport:
        @contextmanager
        def install_fastf1(self):
            active.append(True)
            try:
                yield self
            finally:
                active.pop()

    def load(session, **kwargs):
        assert active
        assert fastf1.Cache._requests_session_cached.settings.only_if_cached is False
        assert kwargs == dict(laps=True, telemetry=True, weather=True, messages=True)
        session._results = pd.DataFrame([dict(DriverNumber='1', DriverId='one')])
        path = Path(fastf1.Cache._CACHE_DIR) / session.api_path.removeprefix('/static/') / 'driver_info.ff1pkl'
        path.parent.mkdir(parents=True)
        path.write_bytes(b'collected fixture')

    monkeypatch.setattr(fastf1.core.Session, 'load', load)
    collector = Collector(archive, end=cutoff_utc('2026-10-23', 'America/Chicago'), http=Transport())
    collector.timing_online = True
    task_id = collector.queue('2025:1:R', 'fastf1_session', season=2025, event_id='2025:1', session_id='2025:1:R')
    collector.fastf1_session({'task_id': task_id, 'session_id': '2025:1:R'},
        {'config': {'year': 2025, 'round': 1, 'code': 'R'}})
    assert not active
    with archive.connect() as conn:
        assert conn.execute('SELECT count(*) FROM session_entries').fetchone()[0] == 1


def test_original_cached_stream_keeps_submillisecond_dates_and_marks_parser_fill(tmp_path, monkeypatch):
    from data.archive.fastf1_extract import extract_session
    from types import SimpleNamespace

    path = tmp_path / 'session'
    path.mkdir()
    (path / 'car_data.ff1pkl').write_bytes(b'trusted fixture')
    raw = pd.DataFrame({'Date': pd.to_datetime(['2025-01-01T00:00:00.000000123', '2025-01-01T00:00:00.100000456']),
        'Time': pd.to_timedelta([1, None], unit='ns'), 'Speed': [100, 0], 'FutureChannel': [7, 0]})
    source = SimpleNamespace(results=pd.DataFrame([dict(DriverNumber='1', DriverId='one')]),
        car_data={'1': raw.copy()}, pos_data={}, api_path='/static/fixture/')
    payload = extract_session(source, '2025:1:R')

    def original(path):
        assert fastf1.Cache._requests_session_cached.settings.only_if_cached is True
        return {'1': raw}

    monkeypatch.setattr(fastf1._api, 'car_data', original)
    with configured_cache(tmp_path / 'cache', offline=False):
        add_original_cached_streams(source, path, payload, '2025:1:R')
        assert fastf1.Cache._requests_session_cached.settings.only_if_cached is False
    exported = next(item['table'] for item in payload['datasets'] if item['kind'] == 'raw_car')
    assert exported['Date'].iloc[0].value == 1735689600000000123
    assert exported['FutureChannel'].tolist() == [7, 0]
    assert exported['fastf1_parser_missing_time_fill'].tolist() == [False, True]
    assert payload['metadata']['original_parser_streams']['car']['1']['valid_observed_samples'] == 1
    canonical = next(item['table'] for item in payload['datasets'] if item['kind'] == 'car')
    assert 'fastf1_parser_generated_fill' in canonical['quality_flags'].iloc[1]
    assert 'fastf1_parser_missing_time_fill' not in raw


@pytest.mark.parametrize('entrant_fill', [False, True])
def test_unmatched_raw_vehicle_fills_do_not_downgrade_entrant_coverage(tmp_path, monkeypatch, entrant_fill):
    from types import SimpleNamespace
    from data.archive.fastf1_extract import extract_session

    path = tmp_path / 'session'
    path.mkdir()
    (path / 'position_data.ff1pkl').write_bytes(b'trusted fixture')
    processed = pd.DataFrame(dict(Date=pd.date_range('2025-01-01', periods=2, freq='s'),
        Time=pd.to_timedelta([0, 1], unit='s'), SessionTime=pd.to_timedelta([0, 1], unit='s'),
        X=[1, 2], Y=[3, 4], Z=[0, 0], Status=['OnTrack', 'OnTrack']))
    source = SimpleNamespace(results=pd.DataFrame([dict(DriverNumber='1', DriverId='one')]),
        car_data={}, pos_data={'1': processed}, api_path='/static/fixture/')
    payload = extract_session(source, '2025:1:R')
    expected = 'partial' if entrant_fill else 'available'
    entrant = processed.drop(columns=['SessionTime']).copy()
    if entrant_fill:
        entrant.loc[0, 'Time'] = pd.NaT
    unmatched = processed.drop(columns=['SessionTime']).copy()
    unmatched.loc[0, 'Time'] = pd.NaT
    monkeypatch.setattr(fastf1._api, 'position_data', lambda *a: {'1': entrant, '241': unmatched})
    with configured_cache(tmp_path / 'cache', offline=True):
        add_original_cached_streams(source, path, payload, '2025:1:R')
    assert payload['metadata']['completeness']['positions']['status'] == expected
    aggregate = next(r for r in payload['metadata']['coverage'] if r['dataset_kind'] == 'positions')
    assert aggregate['status'] == expected
    raw_coverage = next(r for r in payload['metadata']['coverage']
        if r['dataset_kind'] == 'raw_positions/fastf1:2025:1:R:number:241')
    assert raw_coverage['status'] == 'partial' and raw_coverage['invalid_count'] == 1
    raw = next(d['table'] for d in payload['datasets']
        if d['kind'] == 'raw_positions' and d['driver_id'].endswith(':241'))
    assert raw.fastf1_parser_missing_time_fill.tolist() == [True, False]
    assert 'fastf1:2025:1:R:number:241' not in payload['metadata']['completeness']['positions']['drivers']


def test_collector_uses_original_sprint_shootout_name_and_checks_cutoff(tmp_path, monkeypatch):
    from data.archive import fastf1_cache
    from data.archive.collect import schedule_frame

    archive = Archive(tmp_path / 'archive')
    payload = schedule_fixture()
    payload['session4']['0'] = 'Sprint Shootout'
    raw = json.dumps(payload).encode()
    source_id = archive.source('fastf1_schedule', 'sprint_fixture', raw, '2026-10-23T00:00:00Z')
    events, sessions = calendar_rows(2025, schedule_frame(payload))
    archive.save_calendar(events, sessions, source_id)
    monkeypatch.setattr(fastf1_cache, 'TRUSTED_CACHE', tmp_path / 'no-cache')
    calls = []

    def load(session, **kwargs):
        calls.append(session.name)
        session._results = pd.DataFrame([dict(DriverNumber='1', DriverId='one')])

    monkeypatch.setattr(fastf1.core.Session, 'load', load)
    collector = Collector(archive, end=cutoff_utc('2026-10-23', 'America/Chicago'), http=NoNetwork())
    collector.timing_online = True
    task_id = collector.queue('2025:1:SQ', 'fastf1_session', season=2025, event_id='2025:1', session_id='2025:1:SQ')
    state = {'config': {'year': 2025, 'round': 1, 'code': 'SQ'}}
    collector.fastf1_session({'task_id': task_id, 'session_id': '2025:1:SQ'}, state)
    assert calls == ['Sprint Shootout']
    with archive.connect() as conn:
        conn.execute('UPDATE sessions SET scheduled_start_utc_ns=? WHERE session_id=?',
                     (int(collector.end.timestamp())*10**9 - 3600*10**9, '2025:1:SQ'))
    collector.fastf1_session({'task_id': task_id, 'session_id': '2025:1:SQ'}, state)
    assert calls == ['Sprint Shootout']
    with archive.connect() as conn:
        assert conn.execute('SELECT last_error_code FROM collection_tasks WHERE task_id=?', (task_id,)).fetchone()[0] == 'completion_before_cutoff_unverified'


def test_post_cutoff_observations_preserve_sources_and_reject_publication(tmp_path, monkeypatch):
    from data.archive import fastf1_cache

    archive = setup_archive(tmp_path)
    monkeypatch.setattr(fastf1_cache, 'TRUSTED_CACHE', tmp_path / 'no-cache')
    collector = Collector(archive, end=cutoff_utc('2026-10-23', 'America/Chicago'), http=NoNetwork())
    collector.timing_online = True

    def load(session, **kwargs):
        session._results = pd.DataFrame([dict(DriverNumber='1', DriverId='one')])
        session._laps = pd.DataFrame([dict(DriverNumber='1', LapNumber=1,
            LapStartDate=pd.Timestamp(collector.end) + pd.Timedelta(1, unit='ns'))])
        path = Path(fastf1.Cache._CACHE_DIR) / session.api_path.removeprefix('/static/') / 'driver_info.ff1pkl'
        path.parent.mkdir(parents=True)
        path.write_bytes(b'preserve this source despite cutoff rejection')

    monkeypatch.setattr(fastf1.core.Session, 'load', load)
    task_id = collector.queue('2025:1:R', 'fastf1_session', season=2025, event_id='2025:1', session_id='2025:1:R')
    state = {'config': {'year': 2025, 'round': 1, 'code': 'R'}}
    collector.fastf1_session({'task_id': task_id, 'session_id': '2025:1:R'}, state)
    with archive.connect() as conn:
        assert conn.execute('SELECT count(*) FROM session_versions').fetchone()[0] == 0
        assert conn.execute('SELECT count(*) FROM source_objects WHERE provider=?', ('fastf1_trusted_local_cache',)).fetchone()[0] == 1
        assert conn.execute('SELECT last_error_code FROM collection_tasks WHERE task_id=?', (task_id,)).fetchone()[0] == 'observations_exceed_archive_cutoff_sources_preserved'
