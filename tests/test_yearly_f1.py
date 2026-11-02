"""Offline safety checks for the notebook batch entry point."""
from contextlib import contextmanager
from datetime import datetime, timezone

import pytest

from data.archive import notebook_f1 as nf
from data.archive.http import ProviderBlocked, RetryDeferred


def test_selection_numeric_priority_and_completion():
    end = datetime(2025, 5, 1, tzinfo=timezone.utc)
    rows = [dict(session_id=f'2025:{r}:{code}', event_id=f'2025:{r}', session_code=code,
                 scheduled_start_utc_ns=int(end.timestamp()*1e9)-hours*3600*10**9)
            for r, code, hours in [(10, 'R', 5), (2, 'FP1', 3), (2, 'Q', 3), (2, 'R', 5), (1, 'R', 1)]]
    assert [r['session_id'] for r in nf.choose_sessions(rows, end, 2025)] == ['2025:2:R', '2025:2:Q', '2025:2:FP1', '2025:10:R']


def test_known_cancelled_2020_practices_are_not_selected():
    end = datetime(2021, 1, 1, tzinfo=timezone.utc)
    rows = [dict(session_id=sid, event_id=sid.rsplit(':', 1)[0], session_code=sid.rsplit(':', 1)[1],
                 scheduled_start_utc_ns=1) for sid in ('2020:2:FP1', '2020:2:FP3', '2020:11:FP1', '2020:11:FP2')]
    assert [row['session_id'] for row in nf.choose_sessions(rows, end, 2020)] == ['2020:2:FP1']


def test_end_for_is_chicago_and_clamped():
    assert nf.end_for('2020-01-01').isoformat() == '2020-01-02T06:00:00+00:00'
    assert nf.end_for('2099-01-01') <= datetime.now(timezone.utc)


def test_waiting_lock_retries_and_releases(monkeypatch, tmp_path):
    calls, messages = [], []
    @contextmanager
    def lock(path):
        calls.append('enter')
        if len(calls) == 1:
            raise RuntimeError('Another archive collector owns this root')
        try:
            yield
        finally:
            calls.append('release')
    monkeypatch.setattr(nf, 'worker_lock', lock)
    monkeypatch.setattr(nf.time, 'sleep', lambda seconds: None)
    with pytest.raises(KeyboardInterrupt):
        with nf.waiting_lock(tmp_path / 'new', emit=messages.append):
            raise KeyboardInterrupt
    assert calls == ['enter', 'enter', 'release']
    assert len(messages) == 1


def test_progress_atomic_and_schema(tmp_path):
    path = tmp_path / 'progress.json'
    nf.save_progress(path, {'schema_version': 1, 'sessions': {}})
    assert nf.read_progress(path)['sessions'] == {}
    path.write_text('{"schema_version":2,"sessions":{}}')
    with pytest.raises(ValueError):
        nf.read_progress(path)


def test_deferred_wait_honors_deadline_and_is_interruptible(monkeypatch):
    sleeps = []
    clock = [100.0]
    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds
    monkeypatch.setattr(nf.time, 'time', lambda: clock[0])
    monkeypatch.setattr(nf.time, 'sleep', sleep)
    nf.wait_until(225, emit=lambda _: None)
    assert sleeps == [30, 30, 30, 30, 5]
    monkeypatch.setattr(nf.time, 'sleep', lambda _: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        nf.wait_until(230, emit=lambda _: None)


def test_global_http_ledger(tmp_path):
    http = nf.collection_http(tmp_path)
    assert http.path == tmp_path / '.local/f1_collection_control/http.sqlite3'
    assert http.status()['buckets']['fastf1']['hour_limit'] == 200
    assert http.status()['buckets']['sdk']['hour_limit'] == 200
    with http._connect() as conn:
        conn.execute("UPDATE http_policies SET policy_json=? WHERE bucket='sdk'", ('{"min_interval":11,"hour_limit":100}',))
        conn.execute("UPDATE http_policies SET policy_json=? WHERE bucket='fastf1'", ('{"min_interval":11,"hour_limit":400}',))
        conn.execute("INSERT INTO http_attempts(provider,route,request_hash,started_at,weight,sdk) VALUES('fastf1','offline','hash',?,1,1)", (nf.time.time(),))
    updated = nf.collection_http(tmp_path).status()['buckets']
    assert updated['sdk']['hour_limit'] == 100
    assert updated['fastf1']['hour_limit'] == 200
    assert updated['fastf1']['hour_used'] == 1


@pytest.fixture
def batch(monkeypatch, tmp_path):
    """Real SQLite calendar and checkpoints, with all outbound work replaced."""
    monkeypatch.setattr(nf, 'require_runtime', lambda root: None)
    monkeypatch.setattr(nf.time, 'sleep', lambda seconds: None)
    class HTTP:
        def status(self):
            return {'routes': []}
    monkeypatch.setattr(nf, 'collection_http', lambda root: HTTP())
    def calendar(archive, http, year):
        source = archive.source('test', 'offline', b'{}', nf.utc_now())
        events = [dict(event_id=f'{year}:{i}', season=year, round_number=i, event_name='Offline') for i in range(1, 6)]
        sessions = [dict(session_id=f'{year}:{i}:R', event_id=f'{year}:{i}', session_code='R', full_name='Race', original_provider_name='Race', scheduled_start_utc_ns=1) for i in range(1, 6)]
        archive.save_calendar(events, sessions, source)
    monkeypatch.setattr(nf, 'load_calendar', calendar)
    monkeypatch.setattr(nf, 'published_record', lambda *args: None)
    return tmp_path


def test_batch_continues_after_failed_session_and_reuses_pilot(batch, monkeypatch):
    called = []
    pilot = dict(status='partial', version_id='pilot', laps=10)
    monkeypatch.setattr(nf, 'published_record', lambda archive, sid, end: pilot if sid == '2025:1:R' else None)
    def download(row, *args):
        called.append(row['session_id'])
        if row['session_id'] == '2025:2:R':
            raise ValueError('offline failure')
        return dict(status='complete', version_id='new', laps=20)
    monkeypatch.setattr(nf, 'download_session', download)
    result = nf.run_f1_year(batch, 2025, emit=lambda _: None)
    assert result['sessions']['2025:1:R'] == pilot
    assert result['sessions']['2025:2:R']['status'] == 'failed'
    assert result['sessions']['2025:3:R']['status'] == 'complete'
    assert '2025:1:R' not in called


def test_batch_stops_403_and_checkpoints(batch, monkeypatch):
    calls, messages = [], []
    def denied(row, *args):
        calls.append(row['session_id'])
        raise ProviderBlocked('fastf1', 'offline', 403)
    monkeypatch.setattr(nf, 'download_session', denied)
    result = nf.run_f1_year(batch, 2025, emit=messages.append)
    assert calls == ['2025:1:R']
    assert result['sessions']['2025:1:R']['status'] == 'blocked'
    assert any('authenticate=True' in message for message in messages)
    assert nf.read_progress(batch / '.local/f1_2025_2026/progress-f1-2025.json') == result


def test_batch_quota_wait_resume_and_bounded_persist(batch, monkeypatch):
    calls, waits = [], []
    def download(row, *args):
        calls.append(row['session_id'])
        if len(calls) == 1:
            raise RetryDeferred('fastf1', 9999999999, 'quota')
        return dict(status='partial', version_id='new', laps=20)
    monkeypatch.setattr(nf, 'download_session', download)
    monkeypatch.setattr(nf, 'wait_until', lambda deadline, emit: waits.append(deadline))
    result = nf.run_f1_year(batch, 2025, session_limit=1, emit=lambda _: None)
    assert waits == [9999999999]
    assert calls == ['2025:1:R', '2025:1:R']
    assert result['sessions']['2025:1:R']['status'] == 'partial'
    monkeypatch.setattr(nf, 'MAX_DEFERRED_WAITS', 0)
    monkeypatch.setattr(nf, 'download_session', lambda *args: (_ for _ in ()).throw(RetryDeferred('fastf1', 9999999999)))
    result = nf.run_f1_year(batch, 2025, session_limit=1, emit=lambda _: None)
    assert result['sessions']['2025:1:R']['status'] == 'deferred'


def test_three_consecutive_failures_stop_batch(batch, monkeypatch):
    calls = []
    def fail(row, *args):
        calls.append(row['session_id'])
        raise ValueError('offline')
    monkeypatch.setattr(nf, 'download_session', fail)
    result = nf.run_f1_year(batch, 2025, emit=lambda _: None)
    assert len(calls) == 9
    assert len(result['sessions']) == 3


def test_explicit_remaining_session_selection_limits_downloads(batch, monkeypatch):
    calls = []
    def download(row, *args):
        calls.append(row['session_id'])
        return {'status': 'partial', 'version_id': 'offline', 'laps': 1}
    monkeypatch.setattr(nf, 'published_record', lambda *args: None)
    monkeypatch.setattr(nf, 'download_session', download)
    nf.run_f1_year(batch, 2025, session_ids={'2025:2:R'}, emit=lambda _: None)
    assert calls == ['2025:2:R']


def test_explicit_identity_reprocess_bypasses_complete_publication(batch, monkeypatch):
    calls = []
    monkeypatch.setattr(nf, 'published_record', lambda *args: {'status': 'complete', 'version_id': 'old', 'laps': 1})
    def download(row, *args):
        calls.append(row['session_id'])
        return {'status': 'complete', 'version_id': 'corrected', 'laps': 1}
    monkeypatch.setattr(nf, 'download_session', download)
    nf.run_f1_year(batch, 2025, session_ids={'2025:2:R'}, reprocess_session_ids={'2025:2:R'}, emit=lambda _: None)
    assert calls == ['2025:2:R']


def test_download_preserves_sources_before_cutoff_rejection(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from contextlib import nullcontext
    end = datetime(2025, 1, 1, tzinfo=timezone.utc)
    cache_file = tmp_path / 'fastf1-cache/2025/race/car_data.ff1pkl'
    cache_file.parent.mkdir(parents=True)
    cache_file.write_bytes(b'offline-original')
    seen = []
    class Session:
        api_path = '/static/2025/race'
        def load(self, **options):
            pass
    monkeypatch.setattr(nf, 'archived_event', lambda *args: SimpleNamespace(get_session=lambda name: Session()))
    monkeypatch.setattr(nf, 'configured_cache', lambda *args, **kw: nullcontext())
    monkeypatch.setattr(nf, 'add_original_cached_streams', lambda *args: None)
    result = {'metadata': {'observed_end_utc_ns': int(end.timestamp()*1e9)}, 'tables': {'laps': [1]}, 'datasets': []}
    monkeypatch.setattr(nf, 'extract_session', lambda *args: result)
    archive = SimpleNamespace(root=tmp_path, source=lambda *args, **kw: seen.append(args) or 'source',
                              publish_session=lambda *args: pytest.fail('ineligible publication'))
    http = SimpleNamespace(install_fastf1=lambda: nullcontext(), last_sdk_deferred=None)
    row = dict(event_id='2025:1', session_id='2025:1:R', original_provider_name='Race')
    with pytest.raises(ValueError, match='cross cutoff'):
        nf.download_session(row, archive, http, end)
    assert seen[0][2] == b'offline-original'
    http.last_sdk_deferred = RetryDeferred('fastf1', 9999999999)
    with pytest.raises(RetryDeferred):
        nf.download_session(row, archive, http, end)


def test_auth_restores_environment_and_arguments(monkeypatch):
    import os
    import runpy
    import sys
    monkeypatch.setenv('F1_USERNAME', 'local@example.invalid')
    monkeypatch.setenv('F1_PASSWORD', 'local-secret')
    before = sys.argv[:]
    def login(*args, **kw):
        assert sys.argv[1:3] == ['auth', 'f1tv']
        raise RuntimeError('offline')
    monkeypatch.setattr(runpy, 'run_module', login)
    with pytest.raises(RuntimeError):
        nf.authenticate_in_notebook()
    assert sys.argv == before
    assert os.environ['F1_PASSWORD'] == 'local-secret'


def test_published_record_validates_original_sources_without_json_progress(tmp_path):
    archive = nf.Archive(tmp_path)
    source = archive.source('test', 'offline', b'original', nf.utc_now())
    archive.save_calendar([dict(event_id='2025:1', season=2025, round_number=1)],
                          [dict(session_id='2025:1:R', event_id='2025:1', session_code='R')], source)
    coverage = {key: {'status': 'available'} for key in nf.REQUIRED}
    vid = archive.publish_session('2025:1:R', {'drivers': [{'driver_id': 'driver'}],
        'session_entries': [{'driver_id': 'driver'}], 'laps': [{'driver_id': 'driver', 'lap_number': 1}]},
        [], [source], {'completeness': coverage, 'observed_end_utc_ns': 100})
    end = datetime(2025, 1, 1, tzinfo=timezone.utc)
    record = nf.published_record(archive, '2025:1:R', end)
    assert record['version_id'] == vid
    assert record['status'] == 'complete'
    with archive.connect() as conn:
        raw = conn.execute('SELECT raw_path FROM source_objects WHERE source_id=?', (source,)).fetchone()[0]
    (tmp_path / raw).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='Corrupt or missing published source'):
        nf.published_record(archive, '2025:1:R', end)


@pytest.mark.parametrize('laps,extract_fails', [([{'lap_number': 1}], False), ([], False), ([], True)])
def test_denied_primary_preserves_usable_mirror_or_cached_laps(monkeypatch, tmp_path, laps, extract_fails):
    from contextlib import nullcontext
    from types import SimpleNamespace
    published = []
    class Session:
        api_path = '/static/2025/race'
        def load(self, **kwargs):
            pass
    monkeypatch.setattr(nf, 'archived_event', lambda *args: SimpleNamespace(get_session=lambda name: Session()))
    monkeypatch.setattr(nf, 'configured_cache', lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(nf, 'add_original_cached_streams', lambda *args: None)
    coverage = {key: {'status': 'available' if key == 'laps' else 'unavailable'} for key in nf.REQUIRED}
    result = {'metadata': {'completeness': coverage}, 'tables': {'laps': laps}, 'datasets': []}
    def extract(*args):
        if extract_fails:
            raise ValueError('No loaded data')
        return result
    monkeypatch.setattr(nf, 'extract_session', extract)
    def publish(*args):
        published.append(args[-1])
        return 'mirror-version'
    archive = SimpleNamespace(root=tmp_path, publish_session=publish)
    http = SimpleNamespace(install_fastf1=lambda: nullcontext(), last_sdk_deferred=None,
        status=lambda: {'routes': [{'provider': 'fastf1', 'route': 'fastf1:primary', 'blocked_status': 403}]})
    row = dict(event_id='2025:1', session_id='2025:1:R', original_provider_name='Race')
    if laps:
        record = nf.download_session(row, archive, http, datetime(2025, 1, 1, tzinfo=timezone.utc))
        assert record['status'] == 'partial'
        assert record['laps'] == 1
        assert published[0]['denied_routes'][0]['status'] == 403
        assert 'cached or mirror evidence retained' in published[0]['access_note']
    else:
        with pytest.raises(ProviderBlocked):
            nf.download_session(row, archive, http, datetime(2025, 1, 1, tzinfo=timezone.utc))
        assert not published
