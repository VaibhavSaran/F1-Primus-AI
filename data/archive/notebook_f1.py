"""Visible, resumable yearly notebook collection using the existing local archive.

Importing this module performs no authentication or network access.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import time
from uuid import uuid4

from .collect import cutoff_utc, schedule_frame, worker_lock
from .fastf1_cache import archived_event, configured_cache, add_original_cached_streams
from .fastf1_extract import calendar_rows, extract_session
from .http import PacedHTTP, Policy, ProviderBlocked, RetryDeferred
from .store import Archive, utc_now

REQUIRED = ('laps', 'car', 'positions', 'weather', 'race_control')
MAX_DEFERRED_WAITS = 12


@contextmanager
def waiting_lock(path, emit=print):
    """Wait for another notebook kernel; the OS releases ownership on interrupt."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    announced = False
    while True:
        manager = worker_lock(path)
        try:
            manager.__enter__()
        except RuntimeError as exc:
            if str(exc) != 'Another archive collector owns this root':
                raise
            if not announced:
                emit('Waiting for the other archive notebook to finish; interrupt to cancel.')
                announced = True
            time.sleep(2)
            continue
        break
    try:
        yield
    finally:
        manager.__exit__(None, None, None)


def collection_http(root):
    # FastF1's auth adapter can retry one wire request inside one Session.send.
    # Count logical sends conservatively at <=200/hour, reserving retry headroom.
    http = PacedHTTP(Path(root) / '.local/f1_collection_control',
                     policies={'fastf1': Policy(hour_limit=200), 'sdk': Policy(hour_limit=200)})
    with http._connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        for bucket in ('fastf1', 'sdk'):
            policy = json.loads(conn.execute('SELECT policy_json FROM http_policies WHERE bucket=?', (bucket,)).fetchone()[0])
            policy['hour_limit'] = min(policy['hour_limit'], 200)
            conn.execute('UPDATE http_policies SET policy_json=? WHERE bucket=?', (json.dumps(policy), bucket))
    return http


def end_for(through):
    return min(cutoff_utc(through, 'America/Chicago'), datetime.now(timezone.utc))


def load_calendar(archive, http, year):
    url = f'https://raw.githubusercontent.com/theOehrly/f1schedule/master/schedule_{year}.json'
    response = http.get(url, provider='fastf1')
    source = archive.source('fastf1_schedule', url, response.content, utc_now(), url=url,
                            endpoint_kind='calendar')
    events, sessions = calendar_rows(year, schedule_frame(response.json()))
    archive.save_calendar(events, sessions, source)
    return events, sessions


def choose_sessions(rows, end, year, rounds=None):
    from .notebook_audit import NOT_HELD
    priorities = {'R': 0, 'S': 1, 'Q': 2, 'SQ': 3, 'FP1': 4, 'FP2': 5, 'FP3': 6}
    selected = []
    for original in rows:
        row = dict(original)
        if row.get('session_id') in NOT_HELD:
            continue
        season, number = map(int, row['event_id'].split(':'))
        code, start = row['session_code'], row.get('scheduled_start_utc_ns')
        margin = (4 if code == 'R' else 2) * 3600 * 10**9
        if (season == year and code in priorities and start is not None
                and start + margin <= int(end.timestamp()*1e9)
                and (rounds is None or number in rounds)):
            selected.append(row)
    return sorted(selected, key=lambda r: (int(r['event_id'].split(':')[1]), priorities[r['session_code']]))


def read_progress(path):
    if not path.exists():
        return {'schema_version': 1, 'sessions': {}}
    value = json.loads(path.read_text(encoding='utf-8'))
    if value.get('schema_version') != 1 or not isinstance(value.get('sessions'), dict):
        raise ValueError('Unknown checkpoint schema; preserve and inspect it before continuing.')
    return value


def save_progress(path, progress):
    temporary = path.with_name(f'.{path.name}.{uuid4().hex}.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(progress, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def wait_until(retry_at, emit=print):
    emit('Provider pause until ' + datetime.fromtimestamp(retry_at, timezone.utc).isoformat()
         + '; this cell will resume automatically. Interrupt to stop safely.')
    while retry_at > time.time():
        time.sleep(min(30, retry_at - time.time()))


def require_runtime(root):
    import fastf1
    import sys
    from packaging.version import Version
    if Path(sys.executable).resolve() != (Path(root) / 'primusEnv/Scripts/python.exe').resolve():
        raise RuntimeError('Select this project\'s primusEnv notebook kernel.')
    if Version(fastf1.__version__) != Version(version('fastf1')):
        raise RuntimeError('FastF1 changed on disk. Restart the kernel.')
    if not Version('3.9') <= Version(fastf1.__version__) < Version('4'):
        raise RuntimeError('This notebook requires FastF1 >=3.9,<4 in primusEnv.')


def authenticate_in_notebook():
    """Explicit local opt-in only; restore credentials and argv even on failure."""
    import getpass
    import runpy
    import sys
    username = os.environ.get('F1_USERNAME') or input('F1 account email: ').strip()
    password = os.environ.get('F1_PASSWORD') or getpass.getpass('F1 account password (hidden): ')
    if not username or not password:
        raise ValueError('Sign-in cancelled: email and password are required locally.')
    previous = {key: os.environ.get(key) for key in ('F1_USERNAME', 'F1_PASSWORD')}
    arguments = sys.argv[:]
    try:
        os.environ.update(F1_USERNAME=username, F1_PASSWORD=password)
        sys.argv = ['fastf1', 'auth', 'f1tv', '--authenticate', '--method', 'password']
        try:
            runpy.run_module('fastf1', run_name='__main__')
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise RuntimeError('Local sign-in did not complete; inspect the SDK message above.') from None
    finally:
        sys.argv = arguments
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        username = password = None


def access_denied(http):
    for route in http.status()['routes']:
        if route['provider'] == 'fastf1' and route['blocked_status'] in (401, 403):
            raise ProviderBlocked('fastf1', route['route'], route['blocked_status'])


def published_record(archive, session_id, end):
    """The immutable archive is authoritative, including the original pilot."""
    with archive.connect() as conn:
        versions = conn.execute("SELECT * FROM session_versions WHERE session_id=? AND status='published' ORDER BY created_at_utc DESC",
                                (session_id,)).fetchall()
        for row in versions:
            if row['observed_end_utc_ns'] is not None and row['observed_end_utc_ns'] >= int(end.timestamp()*1e9):
                continue
            archive._verify_files(conn, row['version_id'])
            coverage = json.loads(row['completeness_json'] or '{}')
            metadata = json.loads(row['metadata_json'] or '{}')
            count = conn.execute('SELECT count(*) FROM laps WHERE version_id=?', (row['version_id'],)).fetchone()[0]
            if not count:
                continue
            complete = not metadata.get('load_error') and all(coverage.get(k, {}).get('status') == 'available' for k in REQUIRED)
            return dict(status='complete' if complete else 'partial', version_id=row['version_id'],
                        coverage=coverage, laps=count, saved_at_utc=row['created_at_utc'])
    return None


def download_session(row, archive, http, end):
    cache = archive.root / 'fastf1-cache'
    event = archived_event(archive, row['event_id'])
    session = event.get_session(row['original_provider_name'] or row['full_name'])
    with configured_cache(cache, offline=False), http.install_fastf1():
        load_error = None
        try:
            session.load(laps=True, telemetry=True, weather=True, messages=True)
        except SystemExit:
            load_error = 'SDK exited while loading'
        except Exception:
            access_denied(http)
            raise
        if http.last_sdk_deferred is not None:
            raise http.last_sdk_deferred
        try:
            result = extract_session(session, row['session_id'])
        except Exception:
            access_denied(http)
            raise
        path = (cache / session.api_path.removeprefix('/static/')).resolve()
        if not path.is_relative_to(cache.resolve()):
            raise ValueError('Unsafe SDK cache path')
        add_original_cached_streams(session, path, result, row['session_id'])
        if http.last_sdk_deferred is not None:
            raise http.last_sdk_deferred
        sources = []
        for file in sorted(path.glob('*.ff1pkl')):
            if file.is_symlink() or not file.resolve().is_relative_to(cache.resolve()):
                raise ValueError('Unsafe parser cache file')
            sources.append(archive.source('fastf1_parser_cache', str(file.relative_to(cache)),
                file.read_bytes(), utc_now(), endpoint_kind=file.stem,
                availability_basis='hindsight_download_original_publication_unverified'))
        observed_end = result['metadata'].get('observed_end_utc_ns')
        if observed_end is not None and observed_end >= int(end.timestamp()*1e9):
            raise ValueError('Session observations cross cutoff; evidence retained without publication')
        if not result['tables'].get('laps'):
            access_denied(http)
            raise ValueError('No lap data returned')
        result['metadata'].update(load_error=load_error, collection_scope={'cutoff_utc': end.isoformat()})
        coverage = result['metadata']['completeness']
        denied_routes = [dict(provider=route['provider'], route=route['route'], status=route['blocked_status'])
                         for route in http.status()['routes']
                         if route['provider'] == 'fastf1' and route['blocked_status'] in (401, 403)]
        if denied_routes:
            result['metadata']['denied_routes'] = denied_routes
            result['metadata']['access_note'] = 'Some origins denied access; usable cached or mirror evidence retained. Coverage describes actual returned data.'
        count = len(result['tables']['laps'])
        vid = archive.publish_session(row['session_id'], result['tables'], result['datasets'], sources, result['metadata'])
        complete = not load_error and all(coverage.get(k, {}).get('status') == 'available' for k in REQUIRED)
        return dict(status='complete' if complete else 'partial', version_id=vid, coverage=coverage,
                    laps=count, saved_at_utc=utc_now())


def run_f1_year(root, year, through='2026-10-26', round_numbers=None, session_limit=None,
                retry_partial=False, authenticate=False, emit=print, *, session_ids=None,
                reprocess_session_ids=None):
    """Run one year, serializing kernels and checkpointing every session outcome."""
    if isinstance(year, bool) or not isinstance(year, int) or not 2020 <= year <= 2026:
        raise ValueError('year must be an integer from 2020 through 2026')
    if session_limit is not None and (isinstance(session_limit, bool) or not isinstance(session_limit, int) or session_limit < 1):
        raise ValueError('session_limit must be a positive integer or None')
    if round_numbers is not None:
        round_numbers = set(round_numbers)
        if any(isinstance(r, bool) or not isinstance(r, int) or r < 1 for r in round_numbers):
            raise ValueError('round_numbers must contain positive integers')
    if session_ids is not None:
        session_ids = set(session_ids)
        if any(not isinstance(sid, str) or not sid.startswith(f'{year}:') for sid in session_ids):
            raise ValueError('session_ids must contain session identifiers for the selected year')
    reprocess_session_ids = set(reprocess_session_ids or ())
    if any(not isinstance(sid, str) or not sid.startswith(f'{year}:') for sid in reprocess_session_ids):
        raise ValueError('reprocess_session_ids must belong to the selected year')
    root = Path(root).resolve()
    require_runtime(root)
    end = end_for(through)
    output = root / '.local/f1_2025_2026'
    with waiting_lock(output, emit):
        if authenticate:
            authenticate_in_notebook()
        archive, http = Archive(output), collection_http(root)
        if authenticate:
            http.reset_denied('fastf1')
        progress_path = output / f'progress-f1-{year}.json'
        progress = read_progress(progress_path)
        progress.update(year=year, cutoff_utc=end.isoformat())
        waits = 0
        try:
            while True:
                try:
                    load_calendar(archive, http, year)
                    break
                except RetryDeferred as exc:
                    progress['calendar'] = dict(status='deferred', retry_after_utc=datetime.fromtimestamp(exc.retry_at, timezone.utc).isoformat())
                    save_progress(progress_path, progress)
                    if waits >= MAX_DEFERRED_WAITS:
                        return progress
                    waits += 1
                    wait_until(exc.retry_at, emit)
            progress['calendar'] = {'status': 'complete'}
            with archive.connect() as conn:
                rows = conn.execute('SELECT s.*, e.event_name FROM sessions s JOIN events e USING(event_id) WHERE e.season=?', (year,)).fetchall()
            rows = choose_sessions(rows, end, year, round_numbers)
            if session_ids is not None:
                rows = [row for row in rows if row['session_id'] in session_ids]
            emit(f'{year}: {len(rows)} completed sessions eligible; archive {output}')
            attempted, failures = 0, 0
            for row in rows:
                sid = row['session_id']
                existing = published_record(archive, sid, end)
                if existing and sid not in reprocess_session_ids and (existing['status'] == 'complete' or not retry_partial):
                    progress['sessions'][sid] = existing
                    save_progress(progress_path, progress)
                    emit(f'{sid}: reusing validated {existing["status"]} publication')
                    continue
                if session_limit is not None and attempted >= session_limit:
                    break
                previous = progress['sessions'].get(sid, {})
                if previous.get('status') == 'deferred':
                    retry_at = datetime.fromisoformat(previous['retry_after_utc']).timestamp()
                    if retry_at > time.time():
                        if waits >= MAX_DEFERRED_WAITS:
                            return progress
                        waits += 1
                        wait_until(retry_at, emit)
                attempted += 1
                emit(f'{sid}: loading {row["original_provider_name"] or row["full_name"]}')
                attempt = 0
                while attempt < 3:
                    try:
                        record = download_session(row, archive, http, end)
                        progress['sessions'][sid] = record
                        failures = 0
                        emit(f'{sid}: {record["status"]}, {record["laps"]} laps')
                        if record['status'] == 'partial':
                            emit('Coverage is partial; inspect saved coverage. If an origin denied access, local sign-in or entitlement may be needed for missing channels.')
                        break
                    except ProviderBlocked:
                        progress['sessions'][sid] = {'status': 'blocked', 'reason': 'Local sign-in or account entitlement required'}
                        raise
                    except RetryDeferred as exc:
                        progress['sessions'][sid] = dict(status='deferred', retry_after_utc=datetime.fromtimestamp(exc.retry_at, timezone.utc).isoformat(), reason=exc.reason)
                        save_progress(progress_path, progress)
                        if waits >= MAX_DEFERRED_WAITS:
                            emit('Bounded automatic pause limit reached; rerun this cell to resume.')
                            return progress
                        waits += 1
                        wait_until(exc.retry_at, emit)
                    except Exception as exc:
                        attempt += 1
                        if attempt == 3:
                            failures += 1
                            progress['sessions'][sid] = dict(status='failed', error_type=type(exc).__name__, reason='Session failed after three attempts; inspect SDK warnings', saved_at_utc=utc_now())
                            emit(f'{sid}: failed ({type(exc).__name__}); checkpoint saved')
                        else:
                            emit(f'{sid}: retry {attempt}/3 after {30*attempt}s ({type(exc).__name__})')
                            time.sleep(30*attempt)
                    finally:
                        save_progress(progress_path, progress)
                if failures >= 3:
                    emit('Stopped after three consecutive session failures; rerun to resume.')
                    break
        except ProviderBlocked:
            emit('Access denied (401/403). Stopped. Sign in locally with authenticate=True and verify account access, then rerun.')
        except KeyboardInterrupt:
            emit('Interrupted safely; checkpoints and cache preserved. Rerun to resume.')
            raise
        finally:
            save_progress(progress_path, progress)
        emit(f'Year cell finished; session coverage and resume state: {progress_path}')
        return progress
