from datetime import datetime, timezone
from email.utils import format_datetime
import sqlite3
import io
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests
from requests.adapters import BaseAdapter
from urllib3.response import HTTPResponse

from data.archive.http import PacedHTTP, Policy, ProviderBlocked, QuotaExceeded, RetryDeferred, weather_weight


class Clock:
    def __init__(self):
        self.now = 1700000000.0
        self.sleeps = []

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class Adapter(BaseAdapter):
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def send(self, request, **kwargs):
        self.calls.append((request.url, kwargs))
        item = next(self.responses)
        if isinstance(item, Exception):
            raise item
        status, headers = item if isinstance(item, tuple) else (item, {})
        response = requests.Response()
        response.status_code = status
        response.headers.update(headers)
        response.url = request.url
        response.request = request
        response._content = b'{"ok": true}'
        response._content_consumed = True
        response.raw = HTTPResponse(body=io.BytesIO(response.content), preload_content=False, request_url=request.url)
        return response

    def close(self):
        pass


def http(tmp_path, responses, clock=None, **kwargs):
    clock = clock or Clock()
    session = requests.Session()
    adapter = Adapter(responses)
    session.mount('https://', adapter)
    client = PacedHTTP(tmp_path, session=session, clock=clock.time, sleep=clock.sleep, jitter=lambda: 0, **kwargs)
    return client, adapter, clock


def test_spacing_and_shared_budgets_persist_across_instances(tmp_path):
    client, adapter, clock = http(tmp_path, [200, 200])
    client.get('https://example.org/a', provider='fastf1')
    client.get('https://example.org/b', provider='fastf1')
    assert clock.sleeps == [11]
    other, _, _ = http(tmp_path, [200], clock)
    other.get('https://api.jolpi.ca/ergast/f1/2024/', provider='jolpica')
    assert clock.sleeps == [11, 11]
    assert other.status()['buckets']['sdk']['hour_used'] == 3
    assert len(adapter.calls) == 2


def test_401_blocks_persistently_without_retry_and_other_host_works(tmp_path):
    client, adapter, clock = http(tmp_path, [401, 200])
    with pytest.raises(ProviderBlocked):
        client.get('https://archive-api.open-meteo.com/v1/archive')
    other, unused, _ = http(tmp_path, [200], clock)
    with pytest.raises(ProviderBlocked):
        other.get('https://archive-api.open-meteo.com/v1/archive')
    client.get('https://api.jolpi.ca/ergast/f1/', provider='jolpica')
    assert len(adapter.calls) == 2 and not unused.calls


def test_explicit_reset_preserves_quota_and_other_provider_denials(tmp_path):
    client, adapter, _ = http(tmp_path, [401, 401, 200])
    with pytest.raises(ProviderBlocked):
        client.get('https://archive-api.open-meteo.com/v1/archive')
    with pytest.raises(ProviderBlocked):
        client.get('https://api.jolpi.ca/ergast/f1/', provider='jolpica')
    client.reset_denied('open_meteo')
    assert client.status()['buckets']['open_meteo']['hour_used'] == 1
    assert client.get('https://archive-api.open-meteo.com/v1/archive').status_code == 200
    with pytest.raises(ProviderBlocked):
        client.get('https://api.jolpi.ca/ergast/f1/', provider='jolpica')
    assert len(adapter.calls) == 3


def test_swallowed_sdk_quota_is_observable_after_load(tmp_path, monkeypatch):
    from fastf1.req import _SessionWithRateLimiting
    client, _, clock = http(tmp_path, [])
    deferred = QuotaExceeded('fastf1', clock.now + 3600, 'hour_budget')
    def deny(*args, **kwargs):
        raise deferred
    monkeypatch.setattr(client, '_perform', deny)
    request = requests.Request('GET', 'https://livetiming.formula1.com/static/test').prepare()
    with client.install_fastf1():
        try:
            _SessionWithRateLimiting().send(request)
        except Exception:
            pass  # Same behavior as an SDK soft loader.
        assert client.last_sdk_deferred is deferred
    with client.install_fastf1():
        assert client.last_sdk_deferred is None


def test_retry_after_http_date_and_transient_failures_count_every_attempt(tmp_path):
    clock = Clock()
    retry = format_datetime(datetime.fromtimestamp(clock.now + 40, timezone.utc), usegmt=True)
    client, adapter, _ = http(tmp_path, [(429, {'Retry-After': retry}), requests.Timeout(), 503, 200], clock)
    assert client.get('https://archive-api.open-meteo.com/v1/archive').status_code == 200
    assert clock.sleeps == [40, 60, 60]
    assert len(adapter.calls) == 4
    assert client.status()['buckets']['open_meteo']['hour_used'] == 4


def test_long_retry_after_defers_and_restart_obeys_it(tmp_path):
    client, adapter, clock = http(tmp_path, [(429, {'Retry-After': '3600'})])
    with pytest.raises(RetryDeferred) as error:
        client.get('https://archive-api.open-meteo.com/v1/archive')
    assert error.value.retry_at == clock.now + 3600
    other, unused, _ = http(tmp_path, [200], clock)
    with pytest.raises(RetryDeferred):
        other.get('https://archive-api.open-meteo.com/v1/archive')
    assert len(adapter.calls) == 1 and not unused.calls and not clock.sleeps


def test_weighted_weather_quota_is_atomic_and_persistent(tmp_path):
    policy = Policy(min_interval=0, hour_limit=2, day_limit=2, minute_limit=2)
    client, adapter, clock = http(tmp_path, [200], policies={'open_meteo': policy})
    params = {'hourly': ','.join(f'v{x}' for x in range(15)), 'start_date': '2024-01-01', 'end_date': '2024-01-14'}
    assert weather_weight(params) == 1.5
    client.get('https://archive-api.open-meteo.com/v1/archive', params=params)
    other, unused, _ = http(tmp_path, [200], clock)
    with pytest.raises(QuotaExceeded):
        other.get('https://archive-api.open-meteo.com/v1/archive')
    assert not unused.calls and len(adapter.calls) == 1


def test_redirects_are_counted_and_404_is_not_retried(tmp_path):
    client, adapter, _ = http(tmp_path, [(302, {'Location': '/b'}), 404])
    with pytest.raises(requests.HTTPError):
        client.get('https://example.org/a', provider='jolpica')
    assert len(adapter.calls) == 2
    assert client.status()['buckets']['jolpica']['hour_used'] == 2


def test_fastf1_real_send_hook_cache_hits_and_host_quarantine(tmp_path, monkeypatch):
    from fastf1.req import _SessionWithRateLimiting, _CachedSessionWithRateLimiting
    # Keep the SDK limiters present; their own unit is outside this offline test.
    class Limiter:
        calls = 0
        def limit(self):
            self.calls += 1
    limiter = Limiter()
    import re
    monkeypatch.setattr(_SessionWithRateLimiting, '_RATE_LIMITS', {re.compile('.*'): [limiter]})
    cached = _CachedSessionWithRateLimiting(backend='memory', expire_after=3600)
    plain = _SessionWithRateLimiting()
    adapter = Adapter([200, 401, 200, 200])
    cached.mount('https://', adapter)
    plain.mount('https://', adapter)
    # SDK 3.9 mounts an authentication adapter on this longer URL prefix.
    cached.mount('https://livetiming.formula1.com/', adapter)
    plain.mount('https://livetiming.formula1.com/', adapter)
    client, _, _ = http(tmp_path, [])
    original = _SessionWithRateLimiting.send
    with client.install_fastf1():
        cached.get('https://cache.example.org/data')
        assert cached.get('https://cache.example.org/data').from_cache
        assert plain.get('https://livetiming.formula1.com/static/a').status_code == 401
        assert plain.get('https://livetiming.formula1.com/static/b').status_code == 401
        assert plain.get('https://livetiming-mirror.fastf1.dev/static/a').status_code == 200
        # Sessions created after installation inherit the same accounting.
        late = _SessionWithRateLimiting()
        late.mount('https://', adapter)
        late.get('https://other.example.org/data')
    assert _SessionWithRateLimiting.send is original
    assert len(adapter.calls) == limiter.calls == 4
    assert client.status()['buckets']['sdk']['hour_used'] == 4
    with sqlite3.connect(client.path) as conn:
        assert conn.execute('SELECT COUNT(*) FROM http_attempts').fetchone()[0] == 4


def test_retries_are_bounded_and_timeout_is_forwarded(tmp_path):
    client, adapter, _ = http(tmp_path, [503, 503, 503, 503, 200])
    with pytest.raises(requests.HTTPError):
        client.get('https://example.org/data', provider='fastf1', timeout=7)
    assert len(adapter.calls) == 4
    assert all(kwargs['timeout'] == 7 for _, kwargs in adapter.calls)


def test_independent_workers_cannot_overspend_the_same_sqlite_budget(tmp_path):
    policy = Policy(min_interval=0, hour_limit=2)
    clients = [http(tmp_path, [200], policies={'open_meteo': policy}) for _ in range(8)]
    def send(item):
        try:
            item[0].get('https://archive-api.open-meteo.com/v1/archive')
            return 'sent'
        except QuotaExceeded:
            return 'quota'
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(send, clients))
    assert results.count('sent') == 2 and results.count('quota') == 6
    assert sum(len(adapter.calls) for _, adapter, _ in clients) == 2


def test_sdk_fetch_page_reaches_existing_mirror_without_reprobing_denied_origin(tmp_path, monkeypatch):
    from fastf1 import _api
    from fastf1.req import Cache, _SessionWithRateLimiting
    plain = _SessionWithRateLimiting()
    adapter = Adapter([401, 200, 200])
    plain.mount('https://', adapter)
    plain.mount('https://livetiming.formula1.com/', adapter)
    monkeypatch.setattr(Cache, '_requests_session', plain)
    monkeypatch.setattr(Cache, '_requests_session_cached', None)
    if hasattr(Cache, '_http_tmp_disabled'):
        monkeypatch.setattr(Cache, '_http_tmp_disabled', True)
        monkeypatch.setattr(Cache, '_ensure_caching', classmethod(lambda cls: None))
    else:
        monkeypatch.setattr(Cache, '_tmp_disabled', True)
        monkeypatch.setattr(Cache, '_enable_default_cache', classmethod(lambda cls: None))
    client, _, _ = http(tmp_path, [])
    with client.install_fastf1():
        assert _api.fetch_page('/static/test/', 'index') == {'ok': True}
        assert _api.fetch_page('/static/test2/', 'index') == {'ok': True}
    assert len(adapter.calls) == 3
    assert 'livetiming.formula1.com' in adapter.calls[0][0]
    assert all('livetiming-mirror.fastf1.dev' in url for url, _ in adapter.calls[1:])
