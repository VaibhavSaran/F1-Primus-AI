"""Resumable local F1 backfill. Run with ``python -m data.archive.collect``.

One worker owns the archive; HTTP accounting and page checkpoints survive restart.
Original provider payloads are retained even when a field has no SQL projection.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import date, datetime, time as dt_time, timedelta, timezone
from hashlib import sha256
from importlib.metadata import version
import json
import logging
import os
from pathlib import Path
import sys
import time
from uuid import uuid4
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from .fastf1_extract import calendar_rows, extract_session
from .http import PacedHTTP, ProviderBlocked, RetryDeferred
from .store import Archive, canonical, utc_now
from .weather_extract import fetch_historical_weather

LOG = logging.getLogger('archive')
VERSION = '1'
JOLPICA = 'https://api.jolpi.ca/ergast/f1'
KINDS = ('results', 'qualifying', 'sprint', 'driverStandings', 'constructorStandings', 'pitstops', 'laps')


def cutoff_utc(through, timezone_name):
    return datetime.combine(date.fromisoformat(through) + timedelta(days=1), dt_time(), ZoneInfo(timezone_name)).astimezone(timezone.utc)


def schedule_frame(payload):
    frame = pd.DataFrame({key: list(value.values()) if isinstance(value, dict) else value for key, value in payload.items()})
    frame = frame.rename(columns={key: ''.join(part.capitalize() for part in key.split('_')) for key in frame})
    for i in range(1, 6):
        name = f'Session{i}Date'
        if name not in frame:
            continue
        dates = []
        for _, row in frame.iterrows():
            value = row[name]
            if pd.isna(value):
                dates.append(pd.NaT)
                continue
            stamp = pd.Timestamp(value)
            if stamp.tzinfo is None:
                offset = str(row.get('GmtOffset') or '+00:00')
                sign = -1 if offset.startswith('-') else 1
                hours, minutes = [int(part) for part in offset.lstrip('+-').split(':')[:2]]
                stamp = stamp.tz_localize(timezone(sign * timedelta(hours=hours, minutes=minutes)))
            dates.append(stamp.tz_convert('UTC'))
        frame[f'{name}Utc'] = dates
    if 'EventDate' in frame:
        frame['EventDate'] = pd.to_datetime(frame['EventDate'])
    return frame


@contextmanager
def worker_lock(root):
    """OS lock releases on process exit, including crash; stale PID cannot block resume."""
    stream = (root / 'worker.lock').open('a+b')
    try:
        # Windows byte locks can reject the initial read before locking is attempted.
        stream.seek(0)
        if stream.read(1) == b'':
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        stream.close()
        raise RuntimeError('Another archive collector owns this root') from None
    try:
        yield
    finally:
        stream.close()


class Collector:
    def __init__(self, archive, *, end, http=None, event=None, kinds=''):
        self.archive, self.end = archive, end
        self.http = http or PacedHTTP(archive.root)
        self.scope = end.isoformat()
        self.timing_online = False
        self.event = event
        self.kinds = kinds
        self.run_id = None

    def preflight(self, year):
        """A denied archive is recorded once, rather than retried for every channel."""
        evidence = []
        for origin in ('https://livetiming.formula1.com', 'https://livetiming-mirror.fastf1.dev'):
            url = f'{origin}/static/{year}/Index.json'
            try:
                response = self.http.get(url, provider='fastf1')
                source_id = self.source('fastf1_access_probe', response, 'index_preflight')
                evidence.append({'url': url, 'status': response.status_code, 'source_id': source_id})
                self.timing_online = True
            except ProviderBlocked as exc:
                evidence.append({'url': url, 'status': exc.status_code, 'reason': 'access_denied'})
            except requests.HTTPError as exc:
                evidence.append({'url': url, 'status': exc.response.status_code, 'reason': 'public_index_unavailable'})
            except (requests.RequestException, RetryDeferred) as exc:
                evidence.append({'url': url, 'reason': type(exc).__name__})
        report = {'checked_at_utc': utc_now(), 'evidence': evidence, 'timing_online': self.timing_online,
                  'caveat': 'An absent index does not prove every individual mirror file is absent. Detailed sessions remain blocked pending a working public timing source.'}
        self.archive.source('collection_manifest', 'timing_access', canonical(report).encode(), utc_now(), endpoint_kind='source_access')
        (self.archive.root / 'source-access.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        LOG.warning('Timing preflight online=%s evidence=%s', self.timing_online, evidence)

    def queue(self, scope, kind, *, season=None, event_id=None, session_id=None, config=None):
        task_id = sha256(f'{scope}:{kind}:{self.scope}:{VERSION}'.encode()).hexdigest()
        checkpoint = {'config': config or {}}
        with self.archive.connect() as conn:
            conn.execute('''INSERT OR IGNORE INTO collection_tasks
                (task_id,requested_scope,dataset_kind,config_version,season,event_id,session_id,status,checkpoint_json)
                VALUES(?,?,?,?,?,?,?,'pending',?)''',
                (task_id, scope, kind, self.scope + ':' + VERSION, season, event_id, session_id, canonical(checkpoint)))
        return task_id

    def save_task(self, task_id, status, checkpoint, error=None, retry_at=None):
        with self.archive.connect() as conn:
            conn.execute('UPDATE collection_tasks SET status=?,checkpoint_json=?,last_error_code=?,next_retry_utc=? WHERE task_id=?',
                         (status, canonical(checkpoint), error, retry_at, task_id))

    def next_task(self):
        with self.archive.connect() as conn:
            row = conn.execute('''SELECT * FROM collection_tasks WHERE status='pending'
                AND config_version=? AND (? IS NULL OR event_id=?)
                AND (?='' OR instr(','||?||',',','||dataset_kind||',')>0)
                AND (next_retry_utc IS NULL OR next_retry_utc<=?)
                ORDER BY CASE dataset_kind WHEN 'fastf1_calendar' THEN 0 WHEN 'jolpica_calendar' THEN 1
                WHEN 'jolpica_drivers' THEN 2 WHEN 'jolpica_constructors' THEN 2
                ELSE 3 END,season,event_id,
                CASE dataset_kind WHEN 'weather' THEN 0 WHEN 'fastf1_session' THEN 1 WHEN 'results' THEN 2
                WHEN 'qualifying' THEN 3 WHEN 'sprint' THEN 3 WHEN 'publish' THEN 4 WHEN 'pitstops' THEN 6 WHEN 'laps' THEN 7 ELSE 5 END,task_id LIMIT 1''',
                (self.scope + ':' + VERSION, self.event, self.event, self.kinds, self.kinds, utc_now())).fetchone()
            if row:
                conn.execute("UPDATE collection_tasks SET status='running',attempt_count=attempt_count+1 WHERE task_id=?", (row['task_id'],))
                return dict(row)
        return None

    def source(self, provider, response, kind):
        return self.archive.source(provider, response.url, response.content, utc_now(), url=response.url, endpoint_kind=kind,
                                   availability_basis='fetched_now_not_historical_publication')

    def plan_sessions(self, year):
        with self.archive.connect() as conn:
            sessions = conn.execute('SELECT s.* FROM sessions s JOIN events e USING(event_id) WHERE e.season=? AND s.scheduled_start_utc_ns<?',
                                    (year, int(self.end.timestamp()) * 10**9)).fetchall()
        for session in sessions:
            self.queue(session['session_id'], 'fastf1_session', season=year, event_id=session['event_id'], session_id=session['session_id'],
                       config={'year': year, 'round': int(session['event_id'].split(':')[1]), 'code': session['session_code']})

    def calendar(self, task, state):
        year = task['season']
        url = f'https://raw.githubusercontent.com/theOehrly/f1schedule/master/schedule_{year}.json'
        response = self.http.get(url, provider='fastf1')
        source_id = self.source('fastf1_schedule', response, 'calendar')
        events, sessions = calendar_rows(year, schedule_frame(response.json()))
        self.archive.save_calendar(events, sessions, source_id)
        self.plan_sessions(year)
        return True

    def jolpica_page(self, task, state):
        if state.get('source_ids') and state.get('offset', 0) >= state.get('total', float('inf')):
            return True  # Retry publication/queueing from evidence, never fetch beyond the final page.
        year, kind = task['season'], task['dataset_kind']
        config = state['config']
        endpoint = config['endpoint']
        url = f'{JOLPICA}/{year}' + (f'/{endpoint}' if endpoint else '') + '.json'
        response = self.http.get(url, params={'limit': 100, 'offset': state.get('offset', 0)}, provider='jolpica')
        source_id = self.source('jolpica', response, kind)
        payload = response.json()
        mr = payload['MRData']
        # Provider total is the count of endpoint records, including nested lap timings.
        limit, offset, total = int(mr['limit']), int(mr['offset']), int(mr['total'])
        if limit <= 0 or offset != state.get('offset', 0) or total < 0:
            raise ValueError('Invalid pagination metadata')
        if kind == 'jolpica_calendar':
            self.enrich_calendar(year, mr['RaceTable']['Races'], source_id)
        elif kind in ('jolpica_drivers', 'jolpica_constructors'):
            self.references(mr, kind)
        state.setdefault('source_ids', []).append(source_id)
        state['offset'], state['total'] = offset + limit, total
        return offset + limit >= total

    def references(self, mr, kind):
        with self.archive.connect() as conn:
            if kind == 'jolpica_drivers':
                for driver in mr['DriverTable']['Drivers']:
                    self.archive._upsert_reference(conn, 'drivers', {'driver_id': 'jolpica:driver:' + driver['driverId'],
                        'source_driver_id': driver['driverId'], 'first_name': driver.get('givenName'), 'last_name': driver.get('familyName'),
                        'display_name': ' '.join(filter(None, [driver.get('givenName'), driver.get('familyName')]))})
            else:
                for team in mr['ConstructorTable']['Constructors']:
                    self.archive._upsert_reference(conn, 'teams', {'team_id': 'jolpica:team:' + team['constructorId'],
                        'canonical_name': team.get('name'), 'source_team_id': team['constructorId']})

    def enrich_calendar(self, year, races, source_id):
        for race in races:
            round_number = int(race['round'])
            event_id = f'{year}:{round_number}'
            circuit = race['Circuit']
            location = circuit['Location']
            circuit_id = 'jolpica:circuit:' + circuit['circuitId']
            with self.archive.connect() as conn:
                event = conn.execute('SELECT * FROM events WHERE event_id=?', (event_id,)).fetchone()
            if not event:
                # A calendar outage must not prevent archiving known race metadata.
                stamp = pd.Timestamp(race['date'] + 'T' + race.get('time', '00:00:00Z'))
                self.archive.save_calendar([{'event_id': event_id, 'season': year, 'round_number': round_number,
                    'event_name': race['raceName'], 'local_race_date': race['date'], 'status': 'scheduled'}],
                    [{'session_id': event_id + ':R', 'event_id': event_id, 'session_code': 'R', 'full_name': 'Race',
                      'scheduled_start_utc_ns': int(stamp.value), 'status': 'scheduled'}], source_id)
            with self.archive.connect() as conn:
                self.archive._upsert_reference(conn, 'circuits', {'circuit_id': circuit_id, 'name': circuit.get('circuitName'),
                    'country': location.get('country'), 'locality': location.get('locality'),
                    'reference_latitude': float(location['lat']), 'reference_longitude': float(location['long'])})
                conn.execute('UPDATE events SET circuit_id=? WHERE event_id=?', (circuit_id, event_id))
                sessions = conn.execute('SELECT * FROM sessions WHERE event_id=? AND scheduled_start_utc_ns<?',
                                       (event_id, int(self.end.timestamp()) * 10**9)).fetchall()
            if not sessions:
                continue
            start = datetime.fromtimestamp(min(s['scheduled_start_utc_ns'] for s in sessions) / 10**9, timezone.utc)
            finish = min(self.end - timedelta(microseconds=1), datetime.fromtimestamp(max(s['scheduled_start_utc_ns'] for s in sessions) / 10**9, timezone.utc) + timedelta(hours=4))
            self.queue(event_id, 'weather', season=year, event_id=event_id, config={
                'circuit': {'circuit_id': circuit_id, 'latitude': float(location['lat']), 'longitude': float(location['long']),
                            'coordinate_basis': 'jolpica_circuit_reference', 'coordinate_source_id': source_id},
                'start': start.isoformat(), 'end': finish.isoformat()})
            eligible = {s['session_code'] for s in sessions if
                        s['scheduled_start_utc_ns'] + (4 if s['session_code'] == 'R' else 2) * 3600 * 10**9 <= int(self.end.timestamp()) * 10**9}
            for kind in KINDS:
                if kind == 'sprint' and 'S' not in eligible:
                    continue
                if kind == 'qualifying' and 'Q' not in eligible:
                    continue
                if kind not in ('qualifying', 'sprint') and 'R' not in eligible:
                    continue
                self.queue(event_id, kind, season=year, event_id=event_id,
                           config={'endpoint': f'{round_number}/{kind}', 'round': round_number})

    def weather(self, task, state):
        config = state['config']
        start, end = (datetime.fromisoformat(config[key]) for key in ('start', 'end'))
        # ERA5 is published after several days. Preserve a pending task, never invent recent observations.
        if end.date() > datetime.now(timezone.utc).date() - timedelta(days=6):
            raise RetryDeferred('open_meteo', (end + timedelta(days=7)).timestamp(), 'era5_publication_lag')
        def preserve(response):
            source_id = self.archive.source('open_meteo', response.url, response.content, utc_now(), url=response.url,
                endpoint_kind='era5_hourly', availability_basis='retrospective_reanalysis')
            state['source_ids'] = [source_id]
        result = fetch_historical_weather(self.http, config['circuit'], start, end, on_response=preserve)
        source_id = state['source_ids'][0]
        series_id = sha256(f'{task["event_id"]}:{source_id}'.encode()).hexdigest()
        with self.archive.connect() as conn:
            if not conn.execute('SELECT 1 FROM weather_series WHERE series_id=?', (series_id,)).fetchone():
                self.archive.insert_rows(conn, 'weather_series', [dict(result['series'], series_id=series_id, source_id=source_id)])
                self.archive.insert_rows(conn, 'weather_observations', [dict(row, series_id=series_id, source_id=source_id) for row in result['observations']])
        state['source_ids'], state['observations'] = [source_id], len(result['observations'])
        state['coverage'] = result['coverage']
        if result['coverage']['status'] == 'unavailable':
            raise ValueError('Weather response contains no observations for requested window')
        return True

    def fastf1_session(self, task, state):
        import fastf1
        from .fastf1_cache import (TRUSTED_CACHE, add_original_cached_streams, archived_event,
                                  configured_cache, seed_trusted_cache)

        config = state['config']
        cutoff_ns = int(self.end.timestamp()) * 10**9
        with self.archive.connect() as conn:
            scheduled = conn.execute('SELECT scheduled_start_utc_ns,original_provider_name,full_name FROM sessions WHERE session_id=?',
                                     (task['session_id'],)).fetchone()
        duration_ns = (4 if config['code'] == 'R' else 2) * 3600 * 10**9
        if scheduled is None or scheduled[0] is None or scheduled[0] + duration_ns > cutoff_ns:
            self.save_task(task['task_id'], 'blocked', state, 'completion_before_cutoff_unverified')
            return None
        session_name = scheduled['original_provider_name'] or scheduled['full_name'] or config['code']
        cache = self.archive.root / 'fastf1-cache'
        state['cache_seed'] = seed_trusted_cache(TRUSTED_CACHE, cache)
        online_allowed = getattr(self, 'timing_online', False)
        with configured_cache(cache, offline=not online_allowed), self.http.install_fastf1():
            try:
                event = archived_event(self.archive, task['session_id'].rsplit(':', 1)[0])
                session = event.get_session(session_name)
            except ValueError as exc:
                if not online_allowed:
                    self.save_task(task['task_id'], 'blocked', state, f'offline_archived_schedule_unavailable: {exc}')
                    return None
                session = fastf1.get_session(config['year'], config['round'], session_name)
            path = (cache / session.api_path.removeprefix('/static/')).resolve()
            if not path.is_relative_to(cache.resolve()):
                raise ValueError('FastF1 session cache path escapes archive cache')
            files = {file.stem: file for file in path.glob('*.ff1pkl')}
            if not online_allowed and not files:
                self.save_task(task['task_id'], 'blocked', state, 'fastf1_archive_auth_required_no_local_cache')
                return None
            # Missing optional parser files can make FastF1 exit instead of raising.
            # Offline loads request only locally available component groups.
            optional = {'telemetry': {'car_data', 'position_data'}, 'weather': {'weather_data'},
                        'messages': {'race_control_messages'}}
            load_options = dict(laps=online_allowed or {'_extended_timing_data', 'timing_app_data',
                'session_status_data', 'track_status_data'}.issubset(files))
            load_options.update({kind: online_allowed or required.issubset(files)
                                 for kind, required in optional.items()})
            try:
                session.load(**load_options)
            except SystemExit as exc:
                state['load_error'] = f'fastf1_sdk_system_exit: {exc}'
            result = extract_session(session, task['session_id'])
            add_original_cached_streams(session, path, result, task['session_id'])
            result['metadata']['ingestion_run_id'] = getattr(self, 'run_id', None)
            sources = []
            for file in sorted(path.glob('*.ff1pkl')):
                if file.is_symlink() or not file.resolve().is_relative_to(cache.resolve()):
                    raise ValueError('FastF1 parser cache file escapes trusted archive cache')
                sources.append(self.archive.source('fastf1_trusted_local_cache', str(file.relative_to(cache)), file.read_bytes(), utc_now(),
                    endpoint_kind=file.stem, availability_basis='collected_now_or_seeded_cache_original_fetch_time_unknown'))
            state['source_ids'] = sources
            observed_end = result['metadata'].get('observed_end_utc_ns')
            if observed_end is not None and observed_end > cutoff_ns:
                self.save_task(task['task_id'], 'blocked', state, 'observations_exceed_archive_cutoff_sources_preserved')
                return None
            if not online_allowed:
                for kind in ['car', 'positions', 'weather', 'race_control']:
                    item = result['metadata']['completeness'][kind]
                    if item['status'] == 'unavailable':
                        item.update(status='blocked', reason='Timing origin access unavailable and no matching cached channel')
                        for row in result['metadata']['coverage']:
                            if row['dataset_kind'] == kind or row['dataset_kind'].startswith(kind + '/'):
                                row.update(status='blocked', reason=item['reason'])
            has_rows = any(len(dataset['table']) for dataset in result['datasets'])
            if not has_rows and not result['tables'].get('session_entries'):
                self.save_task(task['task_id'], 'blocked', state, state.get('load_error', 'cache_missing_timing'))
                return None
            state['version_id'] = self.archive.publish_session(task['session_id'], result['tables'], result['datasets'], sources, result['metadata'])
            partial = not online_allowed or state.get('load_error') or any(
                result['metadata']['completeness'][kind]['status'] != 'available'
                for kind in ['laps', 'car', 'positions', 'weather', 'race_control'])
            self.save_task(task['task_id'], 'partial' if partial else 'complete', state,
                state.get('load_error') or ('cached_session_only_new_downloads_blocked' if not online_allowed else
                                          'provider_channel_coverage_partial' if partial else None))
            return None

    def publish_jolpica(self, event_id):
        from .fastf1_extract import normalize_jolpica
        with self.archive.connect() as conn:
            tasks = conn.execute('SELECT * FROM collection_tasks WHERE event_id=? AND config_version=?',
                                 (event_id, self.scope + ':' + VERSION)).fetchall()
            sources = {row['source_id']: dict(row) for row in conn.execute('SELECT * FROM source_objects WHERE provider=\'jolpica\'')}
        for code, kinds in [('R', ('results', 'laps', 'pitstops')), ('Q', ('qualifying',)), ('S', ('sprint',))]:
            payloads, source_ids = {}, {}
            for task in tasks:
                kind = task['dataset_kind']
                if kind not in kinds or task['status'] not in ('complete', 'unavailable'):
                    continue
                key = {'qualifying': 'qualifying_results', 'sprint': 'sprint_results', 'pitstops': 'pit_stops'}.get(kind, kind)
                ids = json.loads(task['checkpoint_json']).get('source_ids', [])
                payloads[key] = [json.loads((self.archive.root / sources[s]['raw_path']).read_bytes()) for s in ids]
                source_ids[key] = ids
            if not payloads:
                continue
            result = normalize_jolpica(event_id + ':' + code, payloads, source_ids)
            if result['tables'].get('session_entries'):
                result['metadata']['ingestion_run_id'] = self.run_id
                result['metadata']['collection_tasks'] = [dict(dataset_kind=t['dataset_kind'], status=t['status']) for t in tasks if t['dataset_kind'] in KINDS]
                for table in ('session_entries', 'session_results', 'laps', 'pit_stop_records'):
                    rows = result['tables'].get(table)
                    if rows:
                        result['datasets'].append({'kind': table, 'table': pd.DataFrame(rows)})
                self.archive.publish_session(event_id + ':' + code, result['tables'], result['datasets'],
                    [s for ids in source_ids.values() for s in ids], result['metadata'])

    def queue_publication(self, task):
        with self.archive.connect() as conn:
            row = conn.execute('SELECT checkpoint_json FROM collection_tasks WHERE task_id=?', (task['task_id'],)).fetchone()
        ids = json.loads(row['checkpoint_json']).get('source_ids', [])
        evidence_hash = sha256(canonical(ids).encode()).hexdigest()
        self.queue(task['event_id'] + ':' + task['dataset_kind'] + ':' + evidence_hash, 'publish', season=task['season'], event_id=task['event_id'])

    def pending_count(self):
        with self.archive.connect() as conn:
            return conn.execute("""SELECT count(*) FROM collection_tasks WHERE status='pending' AND config_version=?
                AND (? IS NULL OR event_id=?) AND (?='' OR instr(','||?||',',','||dataset_kind||',')>0)""",
                (self.scope + ':' + VERSION, self.event, self.event, self.kinds, self.kinds)).fetchone()[0]

    def retry(self, status, reset_provider=None):
        with worker_lock(self.archive.root), self.archive.connect() as conn:
            if reset_provider:
                self.http.reset_denied(reset_provider)
            rows = conn.execute('SELECT * FROM collection_tasks WHERE status=? AND config_version=?', (status, self.scope + ':' + VERSION)).fetchall()
            for row in rows:
                state = json.loads(row['checkpoint_json'])
                state.pop('_failures', None)
                if status == 'unavailable':
                    state = {'config': state.get('config', {})}
                conn.execute("UPDATE collection_tasks SET status='pending',attempt_count=0,next_retry_utc=NULL,last_error_code=NULL,checkpoint_json=? WHERE task_id=?", (canonical(state), row['task_id']))

    def execute(self, task):
        state = json.loads(task['checkpoint_json'])
        kind = task['dataset_kind']
        try:
            if kind == 'fastf1_calendar':
                done = self.calendar(task, state)
            elif kind == 'weather':
                done = self.weather(task, state)
            elif kind == 'fastf1_session':
                done = self.fastf1_session(task, state)
            elif kind == 'publish':
                self.publish_jolpica(task['event_id'])
                done = True
            else:
                done = self.jolpica_page(task, state)
            if done is not None:
                status = 'complete' if done else 'pending'
                if done and kind == 'weather' and state.get('coverage', {}).get('status') == 'partial':
                    status = 'partial'
                if done and kind in KINDS and state.get('total') == 0:
                    status = 'unavailable'
                self.save_task(task['task_id'], status, state)
                if task['event_id'] and kind in KINDS and done:
                    self.queue_publication(task)
            LOG.info('%s %s %s offset=%s', task['season'], kind, task['event_id'] or '', state.get('offset', ''))
        except RetryDeferred as exc:
            retry = datetime.fromtimestamp(exc.retry_at, timezone.utc).isoformat()
            self.save_task(task['task_id'], 'pending', state, exc.reason, retry)
            LOG.warning('%s deferred until %s: %s', kind, retry, exc.reason)
        except ProviderBlocked as exc:
            self.save_task(task['task_id'], 'blocked', state, str(exc))
            LOG.warning('%s blocked: %s', kind, exc)
        except Exception as exc:
            # Bounded automatic retries; a bad session cannot stop the remaining seasons.
            retry = datetime.now(timezone.utc) + timedelta(minutes=5 * (task['attempt_count'] + 1))
            state['_failures'] = state.get('_failures', 0) + 1
            status = 'failed' if state['_failures'] >= 4 else 'pending'
            self.save_task(task['task_id'], status, state, f'{type(exc).__name__}: {str(exc)[:500]}', retry.isoformat())
            LOG.exception('%s failed for %s', kind, task['event_id'] or task['season'])

    def status(self):
        with self.archive.connect() as conn:
            counts = {table: conn.execute(f'SELECT count(*) FROM {table}').fetchone()[0] for table in
                      ('events', 'sessions', 'drivers', 'teams', 'laps', 'session_results', 'pit_stop_records', 'weather_observations', 'dataset_files', 'source_objects')}
            tasks = [dict(row) for row in conn.execute('SELECT dataset_kind,status,count(*) count FROM collection_tasks GROUP BY dataset_kind,status ORDER BY dataset_kind,status')]
            errors = [dict(row) for row in conn.execute('SELECT dataset_kind,last_error_code,count(*) count FROM collection_tasks WHERE last_error_code IS NOT NULL GROUP BY dataset_kind,last_error_code LIMIT 30')]
        return {'root': str(self.archive.root), 'through_exclusive_utc': self.scope, 'counts': counts, 'tasks': tasks, 'errors': errors, 'http': self.http.status()}

    def run(self, first_year, max_tasks=None, idle_exit=False):
        with worker_lock(self.archive.root):
            self.run_id = uuid4().hex
            manifest_path = self.archive.root / 'runs' / (self.run_id + '.json')
            manifest_path.parent.mkdir(exist_ok=True)
            run = {'ingestion_run_id': self.run_id, 'script_version': VERSION, 'parser_version': version('fastf1'),
                   'schema_version': '1', 'config_hash': sha256(canonical([first_year, self.scope]).encode()).hexdigest(),
                   'start_utc': utc_now(), 'status': 'running', 'manifest_path': str(manifest_path.relative_to(self.archive.root))}
            with self.archive.connect() as conn:
                conn.execute("UPDATE ingestion_runs SET status='interrupted' WHERE status='running'")
                self.archive.insert_rows(conn, 'ingestion_runs', [run])
            (self.archive.root / 'worker.json').write_text(json.dumps({'pid': os.getpid(), **run}, indent=2), encoding='utf-8')
            manifest_path.write_text(json.dumps(run, indent=2), encoding='utf-8')
            self.preflight(first_year)
            with self.archive.connect() as conn:
                conn.execute("UPDATE collection_tasks SET status='pending' WHERE status='running'")
                completed = conn.execute("SELECT * FROM collection_tasks WHERE dataset_kind IN ('results','qualifying','sprint','driverStandings','constructorStandings','pitstops','laps') AND status IN ('complete','unavailable') AND config_version=?", (self.scope + ':' + VERSION,)).fetchall()
            for task in completed:
                self.queue_publication(task)
            for year in range(first_year, (self.end - timedelta(days=1)).year + 1):
                self.queue(str(year), 'fastf1_calendar', season=year)
                for kind, endpoint in [('calendar', ''), ('drivers', 'drivers'), ('constructors', 'constructors')]:
                    self.queue(str(year), 'jolpica_' + kind, season=year, config={'endpoint': endpoint})
            processed = 0
            while not (self.archive.root / 'STOP').exists():
                task = self.next_task()
                if task is None:
                    if idle_exit:
                        break
                    pending = self.pending_count()
                    if not pending:
                        break
                    time.sleep(30)
                    continue
                self.execute(task)
                processed += 1
                (self.archive.root / 'status.json').write_text(json.dumps(self.status(), indent=2), encoding='utf-8')
                if max_tasks is not None and processed >= max_tasks:
                    break
            with self.archive.connect() as conn:
                remaining = conn.execute("SELECT count(*) FROM collection_tasks WHERE status!='complete' AND config_version=?", (self.scope + ':' + VERSION,)).fetchone()[0]
                run.update(end_utc=utc_now(), status='partial' if remaining else 'complete')
                conn.execute('UPDATE ingestion_runs SET end_utc=?,status=? WHERE ingestion_run_id=?', (run['end_utc'], run['status'], self.run_id))
            manifest_path.write_text(json.dumps(dict(run, processed_tasks=processed), indent=2), encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'status', 'retry'])
    parser.add_argument('--root', default='.local/f1_archive')
    parser.add_argument('--from-year', type=int, default=2018)
    parser.add_argument('--through', default='2026-10-23')
    parser.add_argument('--timezone', default='America/Chicago')
    parser.add_argument('--max-tasks', type=int)
    parser.add_argument('--idle-exit', action='store_true')
    parser.add_argument('--event', help='Optional pilot event ID, for example 2018:1; calendars must already be present')
    parser.add_argument('--kinds', default='', help='Optional comma-separated task kinds for a scoped pilot')
    parser.add_argument('--status', choices=['failed', 'blocked', 'unavailable'], default='failed')
    parser.add_argument('--reset-provider', choices=['fastf1', 'jolpica', 'open_meteo'], help='Explicitly clear an access-denied circuit after legitimate access is restored; quota history is retained')
    args = parser.parse_args(argv)
    if args.reset_provider and (args.command != 'retry' or args.status != 'blocked'):
        parser.error('--reset-provider requires retry --status blocked')
    end = cutoff_utc(args.through, args.timezone)
    if not 2018 <= args.from_year <= date.fromisoformat(args.through).year:
        parser.error('from-year must be between 2018 and through year')
    archive = Archive(args.root)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', handlers=[logging.StreamHandler(), logging.FileHandler(archive.root / 'collector.log', encoding='utf-8')])
    collector = Collector(archive, end=end, event=args.event, kinds=args.kinds)
    if args.command == 'run':
        collector.run(args.from_year, args.max_tasks, args.idle_exit)
    elif args.command == 'retry':
        collector.retry(args.status, args.reset_provider)
    print(json.dumps(collector.status(), indent=2))


if __name__ == '__main__':
    main()
