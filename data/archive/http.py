"""Persistent, conservative HTTP accounting shared by archive workers.

Policies are our safety margins, not provider minimum request intervals. FastF1's
own send method and limiters remain in the request path, including on retries.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import date
from email.utils import parsedate_to_datetime
import hashlib
import json
from pathlib import Path
import random
import sqlite3
import threading
import time
from urllib.parse import parse_qs, urlsplit

import requests


@dataclass(frozen=True)
class Policy:
    min_interval: float = 11
    hour_limit: float = 400
    minute_limit: float | None = None
    day_limit: float | None = None


DEFAULT_POLICIES = {
    'fastf1': Policy(),
    'jolpica': Policy(min_interval=21, hour_limit=180),
    'sdk': Policy(),
    'open_meteo': Policy(min_interval=2.5, hour_limit=4000, minute_limit=450, day_limit=8000),
}


class ProviderBlocked(requests.RequestException):
    def __init__(self, provider, route, status_code):
        self.provider, self.route, self.status_code = provider, route, status_code
        super().__init__(f'{provider} route {route} blocked: HTTP {status_code}')


class RetryDeferred(requests.RequestException):
    def __init__(self, provider, retry_at, reason='provider_backoff'):
        self.provider, self.retry_at, self.reason = provider, retry_at, reason
        super().__init__(f'{provider}: {reason}; next retry at {retry_at}')


class QuotaExceeded(RetryDeferred):
    pass


def weather_weight(params):
    """Conservative weighted calls: variables/10, days/14, models, locations.

    Source: https://open-meteo.com/en/pricing (fractional call accounting).
    Round up input ranges, never down; count all requested weather variable sets.
    """
    def values(key):
        value = params.get(key, '')
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value if str(item)]
        return [item for item in str(value).split(',') if item]

    variables = sum(len(values(key)) for key in ('hourly', 'daily', 'current', 'minutely_15'))
    days = 14
    if params.get('start_date') and params.get('end_date'):
        days = (date.fromisoformat(str(params['end_date'])) - date.fromisoformat(str(params['start_date']))).days + 1
        if days <= 0:
            raise ValueError('weather end date precedes start date')
    else:
        days = max(14, int(params.get('forecast_days', 0)) + int(params.get('past_days', 0)))
    return max(1, variables / 10) * max(1, days / 14) * max(1, len(values('models'))) * max(1, len(values('latitude')))


def _provider(url):
    host = (urlsplit(url).hostname or '').lower()
    if host == 'ergast.com' or host.endswith('.ergast.com') or host == 'api.jolpi.ca':
        return 'jolpica'
    if host == 'open-meteo.com' or host.endswith('.open-meteo.com'):
        return 'open_meteo'
    return 'fastf1'


_install_lock = threading.Lock()


class PacedHTTP:
    def __init__(self, root, policies=None, session=None, *, clock=time.time,
                 sleep=time.sleep, jitter=random.random, max_attempts=4, max_wait=60):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / 'http.sqlite3'
        self.session = session if session is not None else requests.Session()
        self.clock, self.sleep, self.jitter = clock, sleep, jitter
        if max_attempts < 1 or not 0 < max_wait <= 60:
            raise ValueError('max_attempts must be positive; max_wait must be in (0,60]')
        self.max_attempts, self.max_wait = max_attempts, max_wait
        # FastF1's soft loaders may swallow RequestException subclasses.
        self.last_sdk_deferred = None
        defaults = dict(DEFAULT_POLICIES)
        defaults.update(policies or {})
        with self._connect() as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS http_policies(bucket TEXT PRIMARY KEY, policy_json TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS http_routes(route TEXT PRIMARY KEY, provider TEXT NOT NULL,
                    blocked_status INTEGER, next_retry REAL NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS http_attempts(attempt_id INTEGER PRIMARY KEY,
                    provider TEXT NOT NULL, route TEXT NOT NULL, request_hash TEXT NOT NULL,
                    started_at REAL NOT NULL, completed_at REAL, status INTEGER,
                    weight REAL NOT NULL, sdk INTEGER NOT NULL, error_type TEXT,
                    bytes_received INTEGER, retry_at REAL);
                CREATE INDEX IF NOT EXISTS ix_http_attempts_time ON http_attempts(started_at);
            ''')
            for name, policy in defaults.items():
                if isinstance(policy, dict):
                    policy = Policy(**policy)
                if policy.min_interval < 0 or policy.hour_limit <= 0 or any(value is not None and value <= 0 for value in (policy.minute_limit, policy.day_limit)):
                    raise ValueError('invalid HTTP policy')
                conn.execute('INSERT OR IGNORE INTO http_policies VALUES (?,?)', (name, json.dumps(asdict(policy))))

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _reserve(self, provider, route, request_hash, weight, sdk):
        """Reserve actual sends atomically. Interrupted sends still consume quota."""
        while True:
            wait = 0
            with self._connect() as conn:
                conn.execute('BEGIN IMMEDIATE')
                now = self.clock()
                state = conn.execute('SELECT * FROM http_routes WHERE route=?', (route,)).fetchone()
                if state:
                    if state['blocked_status']:
                        raise ProviderBlocked(provider, route, state['blocked_status'])
                    wait = max(0, state['next_retry'] - now)
                    if wait > self.max_wait:
                        raise RetryDeferred(provider, state['next_retry'])
                buckets = [(provider, weight)] + ([('sdk', 1)] if sdk else [])
                for bucket, charge in buckets:
                    row = conn.execute('SELECT policy_json FROM http_policies WHERE bucket=?', (bucket,)).fetchone()
                    if row is None:
                        raise ValueError(f'unknown provider policy {bucket}')
                    policy = Policy(**json.loads(row[0]))
                    clause, args = ('sdk=1', []) if bucket == 'sdk' else ('provider=?', [bucket])
                    metric = '1' if bucket == 'sdk' else 'weight'
                    for seconds, limit in ((60, policy.minute_limit), (3600, policy.hour_limit), (86400, policy.day_limit)):
                        if limit is None:
                            continue
                        rows = conn.execute(f'SELECT started_at,{metric} AS charge FROM http_attempts WHERE {clause} AND started_at>? ORDER BY started_at', [*args, now - seconds]).fetchall()
                        used = sum(row['charge'] for row in rows)
                        if used + charge > limit:
                            retry_at = (rows[0]['started_at'] + seconds + 0.001) if rows else now + seconds
                            raise QuotaExceeded(provider, retry_at, f'{bucket}_{seconds}s_budget')
                    previous = conn.execute(f'SELECT MAX(started_at) FROM http_attempts WHERE {clause}', args).fetchone()[0]
                    if previous is not None:
                        wait = max(wait, previous + policy.min_interval - now)
                if wait <= 0:
                    cursor = conn.execute('INSERT INTO http_attempts(provider,route,request_hash,started_at,weight,sdk) VALUES (?,?,?,?,?,?)', (provider, route, request_hash, now, weight, int(sdk)))
                    return cursor.lastrowid
            if wait > self.max_wait:
                raise RetryDeferred(provider, self.clock() + wait, 'pacing_delay')
            self.sleep(wait)

    def _finish(self, attempt_id, status=None, error=None, retry_at=None, size=None):
        with self._connect() as conn:
            conn.execute('UPDATE http_attempts SET completed_at=?,status=?,error_type=?,bytes_received=?,retry_at=? WHERE attempt_id=?', (self.clock(), status, error, size, retry_at, attempt_id))

    def _route_state(self, provider, route, *, blocked=None, retry_at=0):
        with self._connect() as conn:
            conn.execute('''INSERT INTO http_routes(route,provider,blocked_status,next_retry) VALUES (?,?,?,?)
                ON CONFLICT(route) DO UPDATE SET blocked_status=COALESCE(excluded.blocked_status,http_routes.blocked_status), next_retry=MAX(excluded.next_retry,http_routes.next_retry)''', (route, provider, blocked, retry_at))

    def _perform(self, request, send, provider, *, sdk=False, allow_blocked_response=False):
        host = (urlsplit(request.url).hostname or '').lower()
        route = f'{provider}:{host}'
        # Hash only: persisted state cannot reveal API keys, URL credentials or headers.
        fingerprint = hashlib.sha256(f'{request.method} {request.url}'.encode()).hexdigest()
        params = {key: ','.join(value) for key, value in parse_qs(urlsplit(request.url).query).items()}
        weight = weather_weight(params) if provider == 'open_meteo' else 1
        for attempt in range(self.max_attempts):
            try:
                attempt_id = self._reserve(provider, route, fingerprint, weight, sdk)
            except ProviderBlocked as exc:
                if not allow_blocked_response:
                    raise
                # Return the quarantined status so FastF1's existing public mirror
                # fallback remains reachable without probing a denied origin again.
                response = requests.Response()
                response.status_code, response.request, response.url = exc.status_code, request, request.url
                response._content = b''
                response.archive_quarantined = True
                return response
            try:
                response = send()
            except (requests.Timeout, requests.ConnectionError) as exc:
                delay = min(60, 30 * 2**attempt + self.jitter())
                retry_at = self.clock() + delay
                self._finish(attempt_id, error=type(exc).__name__, retry_at=retry_at)
                self._route_state(provider, route, retry_at=retry_at)
                if attempt + 1 == self.max_attempts:
                    raise
                if delay > self.max_wait:
                    raise RetryDeferred(provider, retry_at) from exc
                continue
            status = response.status_code
            if status in (401, 403):
                self._finish(attempt_id, status=status)
                self._route_state(provider, route, blocked=status)
                if allow_blocked_response:
                    return response
                raise ProviderBlocked(provider, route, status)
            if status == 429 or 500 <= status <= 599:
                delay = min(60, 30 * 2**attempt + self.jitter())
                retry_after = response.headers.get('Retry-After')
                if retry_after:
                    try:
                        delay = max(delay, float(retry_after))
                    except ValueError:
                        try:
                            delay = max(delay, parsedate_to_datetime(retry_after).timestamp() - self.clock())
                        except (ValueError, TypeError, OverflowError):
                            pass
                retry_at = self.clock() + delay
                self._finish(attempt_id, status=status, retry_at=retry_at)
                self._route_state(provider, route, retry_at=retry_at)
                if attempt + 1 == self.max_attempts:
                    response.raise_for_status()
                response.close()
                if delay > self.max_wait:
                    raise RetryDeferred(provider, retry_at)
                continue
            size = None if getattr(response, '_content', False) is False else len(response.content)
            self._finish(attempt_id, status=status, size=size)
            return response
        raise RuntimeError('unreachable request retry state')

    def get(self, url, params=None, provider='open_meteo', timeout=30):
        if timeout <= 0:
            raise ValueError('timeout must be positive')
        request = self.session.prepare_request(requests.Request('GET', url, params=params))
        # Redirects are followed here so every network hop reserves its own call.
        for _ in range(self.session.max_redirects + 1):
            options = self.session.merge_environment_settings(request.url, {}, False, None, None)
            response = self._perform(request, lambda: self.session.send(request, timeout=timeout, allow_redirects=False, **options), provider, sdk=provider in ('fastf1', 'jolpica'))
            if not response.is_redirect:
                response.raise_for_status()
                return response
            previous_host = urlsplit(request.url).hostname
            request = response.next
            response.close()
            if request is None:
                raise requests.TooManyRedirects('redirect target unavailable')
            if urlsplit(request.url).hostname != previous_host:
                provider = _provider(request.url)
        raise requests.TooManyRedirects('archive redirect limit exceeded')

    @contextmanager
    def install_fastf1(self):
        from fastf1.req import _SessionWithRateLimiting
        if not _install_lock.acquire(blocking=False):
            raise RuntimeError('FastF1 paced transport already installed in this process')
        original = _SessionWithRateLimiting.send
        client = self
        self.last_sdk_deferred = None

        def paced_send(session, request, **kwargs):
            kwargs.setdefault('timeout', 30)
            try:
                return client._perform(request, lambda: original(session, request, **kwargs), _provider(request.url), sdk=True, allow_blocked_response=True)
            except RetryDeferred as exc:
                client.last_sdk_deferred = exc
                raise

        _SessionWithRateLimiting.send = paced_send
        try:
            yield self
        finally:
            _SessionWithRateLimiting.send = original
            _install_lock.release()

    def reset_denied(self, provider):
        """Explicit operator action after legitimate source access has changed.

        Quota history and Retry-After times remain intact.
        """
        if provider not in DEFAULT_POLICIES or provider == 'sdk':
            raise ValueError('Unknown source provider')
        with self._connect() as conn:
            conn.execute('UPDATE http_routes SET blocked_status=NULL WHERE provider=?', (provider,))

    def status(self):
        now = self.clock()
        with self._connect() as conn:
            buckets = {}
            for row in conn.execute('SELECT * FROM http_policies'):
                bucket = row['bucket']
                clause, args = ('sdk=1', []) if bucket == 'sdk' else ('provider=?', [bucket])
                metric = '1' if bucket == 'sdk' else 'weight'
                used = conn.execute(f'SELECT COALESCE(SUM({metric}),0) FROM http_attempts WHERE {clause} AND started_at>?', [*args, now - 3600]).fetchone()[0]
                buckets[bucket] = {**json.loads(row['policy_json']), 'hour_used': used}
            routes = [dict(row) for row in conn.execute('SELECT * FROM http_routes')]
        return {'buckets': buckets, 'routes': routes}
