"""Resumable ERA5 event weekends for the independent yearly weather notebooks.

Original observations are retrospective reanalysis, never archived forecasts.
Importing this module performs no network access or authentication.
"""
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq

from .http import ProviderBlocked, RetryDeferred
from .notebook_f1 import collection_http, end_for, load_calendar, waiting_lock, wait_until
from .store import Archive, canonical, file_hash, utc_now
from .weather_extract import fetch_historical_weather, normalize_weather

VERSION = 'yearly_weather_v1'
JOLPICA = 'https://api.jolpi.ca/ergast/f1'


def _now():
    return datetime.now(timezone.utc)


def _source(archive, source_id):
    with archive.connect() as conn:
        row = conn.execute('SELECT * FROM source_objects WHERE source_id=?', (source_id,)).fetchone()
    if row is None:
        raise ValueError('Preserved source metadata missing: ' + source_id)
    path = archive._safe_path(row['raw_path'])
    if not path.is_file() or file_hash(path) != row['payload_sha256']:
        raise ValueError('Preserved source missing or hash mismatch: ' + source_id)
    return path.read_bytes()


def _circuits(archive, http, year):
    """Retain public circuit coordinates without touching authenticated timing APIs."""
    url = f'{JOLPICA}/{year}.json'
    with archive.connect() as conn:
        cached = conn.execute("SELECT source_id FROM source_objects WHERE provider='jolpica' AND request_fingerprint=? ORDER BY first_fetched_at_utc DESC LIMIT 1", (url,)).fetchone()
    if cached:
        source = cached[0]
        payload = json.loads(_source(archive, source))
    else:
        response = http.get(url, params={'limit': 100, 'offset': 0}, provider='jolpica')
        source = archive.source('jolpica', url, response.content, utc_now(), url=response.url,
                                endpoint_kind='calendar', availability_basis='fetched_now_not_historical_publication')
        payload = response.json()
    mr = payload['MRData']
    races = mr['RaceTable']['Races']
    if int(mr['offset']) != 0 or int(mr['total']) > len(races):
        raise ValueError('Incomplete Jolpica schedule; refuse to silently omit circuits')
    circuits = {}
    for race in races:
        try:
            item, number = race['Circuit'], int(race['round'])
            latitude, longitude = float(item['Location']['lat']), float(item['Location']['long'])
            if not math.isfinite(latitude) or not math.isfinite(longitude) or not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
                raise ValueError('invalid circuit coordinates')
            circuit_id = 'jolpica:circuit:' + item['circuitId']
            with archive.connect() as conn:
                archive._upsert_reference(conn, 'circuits', dict(circuit_id=circuit_id,
                    name=item.get('circuitName'), reference_latitude=latitude, reference_longitude=longitude))
                conn.execute('UPDATE events SET circuit_id=? WHERE event_id=?', (circuit_id, f'{year}:{number}'))
            circuits[number] = dict(circuit_id=circuit_id, latitude=latitude, longitude=longitude,
                coordinate_basis='jolpica_circuit_reference', coordinate_source_id=source)
        except (KeyError, TypeError, ValueError):
            # The corresponding event is explicitly blocked below; other weekends continue.
            continue
    return circuits


def _save_task(archive, event_id, state, status, retry=None, error=None):
    task_id = VERSION + ':' + event_id
    with archive.connect() as conn:
        archive._upsert_reference(conn, 'collection_tasks', dict(task_id=task_id,
            requested_scope=event_id, dataset_kind='weather', config_version=VERSION,
            season=int(event_id.split(':')[0]), event_id=event_id, status=status,
            attempt_count=state.get('failures', 0), next_retry_utc=retry,
            checkpoint_json=canonical(state), last_error_code=error,
            expected_count=state.get('coverage', {}).get('expected_count'),
            observed_count=state.get('observations')))


def _publish(archive, event_id, state, result, source):
    if result['coverage']['status'] == 'unavailable':
        raise ValueError('ERA5 response has no valid observations for this window')
    series_id = sha256(canonical([event_id, source, state['start'], state['end']]).encode()).hexdigest()
    rows = [dict(row, series_id=series_id, source_id=source) for row in result['observations']]
    relative = f'weather/{event_id.replace(":", "/")}/{series_id}.parquet'
    final = archive._safe_path(relative)
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = final.with_name('.' + uuid4().hex + '.tmp')
    try:
        pq.write_table(pa.Table.from_pylist(rows), temporary)
        with temporary.open('r+b') as stream:
            os.fsync(stream.fileno())
        temporary.replace(final)
    finally:
        temporary.unlink(missing_ok=True)
    with archive.connect() as conn:
        archive._upsert_reference(conn, 'weather_series', dict(result['series'], series_id=series_id, source_id=source))
        conn.execute('DELETE FROM weather_observations WHERE series_id=?', (series_id,))
        archive.insert_rows(conn, 'weather_observations', rows)
    state.update(source_id=source, series_id=series_id, observations=len(rows), coverage=result['coverage'],
                 parquet_path=relative, parquet_sha256=file_hash(final))


def _verified(archive, state):
    if not state.get('source_id'):
        return False
    _source(archive, state['source_id'])
    _source(archive, state['circuit']['coordinate_source_id'])
    path = archive._safe_path(state.get('parquet_path', 'missing.parquet'))
    if not path.is_file() or file_hash(path) != state.get('parquet_sha256'):
        return False
    expected = state.get('observations', 0)
    if expected <= 0 or pq.read_metadata(path).num_rows != expected:
        return False
    with archive.connect() as conn:
        series = conn.execute('SELECT source_id FROM weather_series WHERE series_id=?', (state.get('series_id'),)).fetchone()
        count = conn.execute('SELECT count(*) FROM weather_observations WHERE series_id=? AND source_id=?',
                             (state.get('series_id'), state['source_id'])).fetchone()[0]
    return bool(series and series[0] == state['source_id'] and count == expected)


def run_weather_year(root, year, through='2026-10-26', round_numbers=None, event_limit=None, emit=print, *, retry_failed=False):
    """Collect one year's completed weekends, then return without waiting for ERA5 lag.

    Seven weather kernels share an OS lock; F1 notebooks use a different output
    lock and all workers share persisted HTTP limits. Re-running verifies raw
    hashes, SQL observation counts and Parquet evidence before skipping an event.
    Session IDs and precise UTC window live in collection_tasks.checkpoint_json.
    Explicit retry_failed grants a fresh three-attempt budget and source fetch
    for failed events, retaining previous source IDs and failure history.
    """
    if isinstance(year, bool) or not isinstance(year, int) or not 2020 <= year <= 2026:
        raise ValueError('year must be 2020 through 2026')
    if event_limit is not None and (isinstance(event_limit, bool) or not isinstance(event_limit, int) or event_limit < 1):
        raise ValueError('event_limit must be a positive integer or None')
    rounds = None if round_numbers is None else list(round_numbers)
    if rounds is not None and any(type(number) is not int or number < 1 for number in rounds):
        raise ValueError('round_numbers must contain positive integers')
    rounds = None if rounds is None else set(rounds)
    end = end_for(through)
    root = Path(root).resolve()
    output = root / '.local/f1_weather'
    counts = {}
    with waiting_lock(output, emit=emit):
        archive, http = Archive(output), collection_http(root)
        deferred_waits, calendar_failures = 0, 0
        while True:
            try:
                events, sessions = load_calendar(archive, http, year)
                circuits = _circuits(archive, http, year)
                break
            except RetryDeferred as exc:
                deferred_waits += 1
                if deferred_waits > 12:
                    emit(str(exc))
                    return dict(root=str(output), year=year, events=0, counts={'pending': 1})
                wait_until(exc.retry_at, emit=emit)
            except ProviderBlocked as exc:
                emit('Weather calendar collection blocked: ' + str(exc))
                return dict(root=str(output), year=year, events=0, counts={'blocked': 1})
            except Exception as exc:
                calendar_failures += 1
                emit(f'Weather calendar attempt {calendar_failures}/3 failed: {exc}')
                if calendar_failures >= 3:
                    return dict(root=str(output), year=year, events=0, counts={'failed': 1})
        eligible = {}
        for session in sessions:
            number = int(session['event_id'].split(':')[1])
            stamp = session.get('scheduled_start_utc_ns')
            duration = timedelta(hours=4 if session['session_code'] == 'R' else 2)
            if stamp is None or (rounds is not None and number not in rounds):
                continue
            start = datetime.fromtimestamp(stamp / 10**9, timezone.utc)
            if start + duration <= end:
                eligible.setdefault(session['event_id'], []).append((session, start, start + duration))
        selected = sorted(eligible, key=lambda key: int(key.split(':')[1]))
        if event_limit is not None:
            selected = selected[:event_limit]
        for event_id in selected:
            if (output / 'STOP').exists():
                emit('Weather STOP file found; rerun after removing it to resume.')
                break
            chosen = eligible[event_id]
            start = min(item[1] for item in chosen) - timedelta(hours=1)
            finish = min(end - timedelta(microseconds=1), max(item[2] for item in chosen) + timedelta(hours=1))
            with archive.connect() as conn:
                previous = conn.execute('SELECT * FROM collection_tasks WHERE task_id=?', (VERSION + ':' + event_id,)).fetchone()
            state = json.loads(previous['checkpoint_json']) if previous else {}
            if retry_failed and previous and previous['status'] == 'failed':
                prior_sources = state.setdefault('prior_failed_source_ids', [])
                failed_source = state.pop('source_id', None)
                if failed_source and failed_source not in prior_sources:
                    prior_sources.append(failed_source)
                state.setdefault('failure_history', []).append(dict(
                    at_utc=utc_now(), attempts=state.get('failures', 0),
                    error=previous['last_error_code'], source_id=failed_source))
                state['failures'] = 0
            same_window = state.get('start') == start.isoformat() and state.get('end') == finish.isoformat()
            if not same_window:
                state = {}
            state.update(start=start.isoformat(), end=finish.isoformat(),
                         session_ids=sorted(item[0]['session_id'] for item in chosen))
            circuit = circuits.get(int(event_id.split(':')[1]))
            if circuit is None:
                _save_task(archive, event_id, state, 'blocked', error='missing_or_invalid_jolpica_coordinates')
                counts['blocked'] = counts.get('blocked', 0) + 1
                emit(event_id + ': blocked; circuit coordinates unavailable')
                continue
            state['circuit'] = circuit
            # Six full days after the UTC request date provides an ERA5 publication margin.
            retry = datetime.combine(finish.date() + timedelta(days=6), datetime.min.time(), timezone.utc)
            if _now() < retry:
                _save_task(archive, event_id, state, 'pending', retry=retry.isoformat(), error='era5_publication_lag')
                counts['pending'] = counts.get('pending', 0) + 1
                emit(event_id + ': ERA5 publication pending; rerun on/after ' + retry.isoformat())
                continue
            status, blocked = 'failed', False
            if state.get('failures', 0) >= 3:
                emit(event_id + ': stopped after three failures; inspect task error before resetting failures')
            while state.get('failures', 0) < 3:
                try:
                    if not _verified(archive, state):
                        if state.get('source_id'):
                            payload = json.loads(_source(archive, state['source_id']))
                            result = normalize_weather(payload, circuit, start=start, end=finish)
                        else:
                            def preserve(response):
                                state['source_id'] = archive.source('open_meteo', response.url, response.content,
                                    utc_now(), url=response.url, endpoint_kind='era5_hourly',
                                    availability_basis='retrospective_reanalysis')
                                _save_task(archive, event_id, state, 'running')
                            result = fetch_historical_weather(http, circuit, start, finish, on_response=preserve)
                        _publish(archive, event_id, state, result, state['source_id'])
                    status = 'complete' if state['coverage']['status'] == 'available' else 'partial'
                    _save_task(archive, event_id, state, status)
                    break
                except RetryDeferred as exc:
                    deferred_waits += 1
                    status = 'pending'
                    _save_task(archive, event_id, state, status,
                        retry=datetime.fromtimestamp(exc.retry_at, timezone.utc).isoformat(), error=exc.reason)
                    if deferred_waits > 12:
                        break
                    emit(event_id + ': provider pause; shared HTTP ledger preserves Retry-After')
                    wait_until(exc.retry_at, emit=emit)
                except ProviderBlocked as exc:
                    status, blocked = 'blocked', True
                    _save_task(archive, event_id, state, status, error=str(exc))
                    emit('Weather provider blocked: ' + str(exc))
                    break
                except Exception as exc:
                    state['failures'] = state.get('failures', 0) + 1
                    _save_task(archive, event_id, state, 'failed' if state['failures'] >= 3 else 'pending',
                               error=f'{type(exc).__name__}: {str(exc)[:400]}')
                    emit(f'{event_id}: attempt {state["failures"]}/3 failed: {exc}')
            counts[status] = counts.get(status, 0) + 1
            emit(event_id + ': ' + status)
            if blocked:
                emit('Stopped weather year after provider access block; other unprocessed events remain resumable.')
                break
    return dict(root=str(output), year=year, events=sum(counts.values()), counts=counts)



