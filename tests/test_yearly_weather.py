from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from data.archive import notebook_weather as weather
from data.archive.store import Archive
from data.archive.weather_extract import FIELDS


class Response:
    def __init__(self, payload, url):
        self.payload, self.url = payload, url
        self.content = json.dumps(payload).encode()
    def json(self):
        return self.payload


class HTTP:
    def __init__(self):
        self.calls = []
    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if 'jolpi' in url:
            races = [{'round': str(i), 'Circuit': {'circuitId': 'track' + str(i),
                'circuitName': 'Track', 'Location': {'lat': '40', 'long': '-80'}}} for i in (1, 2)]
            return Response({'MRData': {'total': '2', 'limit': '100', 'offset': '0', 'RaceTable': {'Races': races}}}, url)
        params = kwargs['params']
        start = datetime.fromisoformat(params['start_date']).replace(tzinfo=timezone.utc)
        end = datetime.fromisoformat(params['end_date']).replace(tzinfo=timezone.utc) + timedelta(hours=23)
        times = [start + timedelta(hours=i) for i in range(int((end-start).total_seconds() // 3600) + 1)]
        payload = {'latitude': 40., 'longitude': -80., 'timezone': 'UTC', 'utc_offset_seconds': 0,
                   'hourly_units': {key: unit for key, (_, unit) in FIELDS.items()},
                   'hourly': {'time': [stamp.isoformat() for stamp in times],
                              **{key: [1.] * len(times) for key in FIELDS}}}
        return Response(payload, url)


@pytest.fixture
def setup(monkeypatch, tmp_path):
    http = HTTP()
    monkeypatch.setattr(weather, 'collection_http', lambda root: http)
    monkeypatch.setattr(weather, 'end_for', lambda through: datetime(2026, 10, 27, 5, tzinfo=timezone.utc))
    monkeypatch.setattr(weather, '_now', lambda: datetime(2026, 10, 27, 5, tzinfo=timezone.utc))
    def calendar(archive, http, year):
        source = archive.source('fastf1_schedule', str(year), b'calendar', '2026-10-27T05:00:00+00:00')
        events, sessions = [], []
        for number, day in [(1, '2026-10-01'), (2, '2026-10-25')]:
            event = f'{year}:{number}'
            events.append(dict(event_id=event, season=year, round_number=number, event_name='Race'))
            for code, hour in [('FP1', 10), ('R', 15)]:
                stamp = datetime.fromisoformat(day).replace(hour=hour, tzinfo=timezone.utc)
                sessions.append(dict(session_id=event+':'+code, event_id=event, session_code=code,
                    scheduled_start_utc_ns=int(stamp.timestamp()) * 10**9))
        archive.save_calendar(events, sessions, source)
        return events, sessions
    monkeypatch.setattr(weather, 'load_calendar', calendar)
    return tmp_path, http


def tasks(root):
    with Archive(root / '.local/f1_weather').connect() as conn:
        return [dict(row) for row in conn.execute("SELECT * FROM collection_tasks WHERE dataset_kind='weather' ORDER BY event_id")]


def test_two_events_recent_era5_is_deferred_and_completed_event_resumes(setup):
    root, http = setup
    result = weather.run_weather_year(root, 2026)
    rows = tasks(root)
    assert result['counts'] == {'complete': 1, 'pending': 1}
    assert rows[1]['next_retry_utc'].startswith('2026-10-31')
    assert len([url for url, _ in http.calls if 'archive-api' in url]) == 1
    checkpoint = json.loads(rows[0]['checkpoint_json'])
    assert checkpoint['session_ids'] == ['2026:1:FP1', '2026:1:R']
    assert checkpoint['start'] == '2026-10-01T09:00:00+00:00'
    assert checkpoint['end'] == '2026-10-01T20:00:00+00:00'
    assert (root / '.local/f1_weather' / checkpoint['parquet_path']).exists()
    weather.run_weather_year(root, 2026)
    assert len([url for url, _ in http.calls if 'archive-api' in url]) == 1


def test_missing_projection_rebuilds_from_preserved_source_without_http(setup):
    root, http = setup
    weather.run_weather_year(root, 2025, round_numbers=[1])
    archive = Archive(root / '.local/f1_weather')
    row = tasks(root)[0]
    checkpoint = json.loads(row['checkpoint_json'])
    (archive.root / checkpoint['parquet_path']).unlink()
    with archive.connect() as conn:
        conn.execute('DELETE FROM weather_observations')
    weather.run_weather_year(root, 2025, round_numbers=[1])
    assert len([url for url, _ in http.calls if 'archive-api' in url]) == 1
    with archive.connect() as conn:
        assert conn.execute('SELECT count(*) FROM weather_observations').fetchone()[0] == 12
    assert (archive.root / checkpoint['parquet_path']).exists()


def test_uncompleted_session_is_not_requested(setup, monkeypatch):
    root, http = setup
    monkeypatch.setattr(weather, 'end_for', lambda through: datetime(2026, 10, 1, 16, tzinfo=timezone.utc))
    weather.run_weather_year(root, 2026, round_numbers=[1])
    checkpoint = json.loads(tasks(root)[0]['checkpoint_json'])
    assert checkpoint['session_ids'] == ['2026:1:FP1']
    assert checkpoint['end'] == '2026-10-01T13:00:00+00:00'


def test_corrupt_source_stops_instead_of_trusting_progress(setup):
    root, http = setup
    weather.run_weather_year(root, 2025, round_numbers=[1])
    archive = Archive(root / '.local/f1_weather')
    with archive.connect() as conn:
        path = conn.execute("SELECT raw_path FROM source_objects WHERE provider='open_meteo'").fetchone()[0]
    (archive.root / path).write_bytes(b'corrupt')
    result = weather.run_weather_year(root, 2025, round_numbers=[1])
    assert result['counts'] == {'failed': 1}
    assert len([url for url, _ in http.calls if 'archive-api' in url]) == 1


def test_partial_payload_is_honest_partial_not_complete(setup):
    root, http = setup
    original = http.get
    def missing(url, **kwargs):
        response = original(url, **kwargs)
        if 'archive-api' in url:
            response.payload['hourly']['temperature_2m'][10] = None
            response.content = json.dumps(response.payload).encode()
        return response
    http.get = missing
    assert weather.run_weather_year(root, 2025, round_numbers=[1])['counts'] == {'partial': 1}
    assert json.loads(tasks(root)[0]['checkpoint_json'])['coverage']['missing_count'] == 0


def test_bad_coordinates_block_event_without_request(setup):
    root, http = setup
    original = http.get
    def bad(url, **kwargs):
        response = original(url, **kwargs)
        if 'jolpi' in url:
            response.payload['MRData']['RaceTable']['Races'][0]['Circuit']['Location']['lat'] = 'nan'
            response.content = json.dumps(response.payload).encode()
        return response
    http.get = bad
    assert weather.run_weather_year(root, 2025, round_numbers=[1])['counts'] == {'blocked': 1}
    assert len(http.calls) == 1


def test_provider_block_stops_year_and_preserves_pending_events(setup):
    root, http = setup
    original = http.get
    def blocked(url, **kwargs):
        if 'archive-api' in url:
            raise weather.ProviderBlocked('open_meteo', '/v1/archive', 403)
        return original(url, **kwargs)
    http.get = blocked
    result = weather.run_weather_year(root, 2025, round_numbers=[1])
    assert result['counts'] == {'blocked': 1}
    assert tasks(root)[0]['last_error_code'].endswith('HTTP 403')


def test_retry_after_is_honored_then_finishes(setup, monkeypatch):
    root, http = setup
    original = http.get
    retries, waits = [], []
    def deferred(url, **kwargs):
        if 'archive-api' in url and not retries:
            retries.append(1)
            raise weather.RetryDeferred('open_meteo', 1234567890, 'http_429')
        return original(url, **kwargs)
    http.get = deferred
    monkeypatch.setattr(weather, 'wait_until', lambda retry_at, emit: waits.append(retry_at))
    assert weather.run_weather_year(root, 2025, round_numbers=[1])['counts'] == {'complete': 1}
    assert waits == [1234567890]


def test_failures_bounded_across_reruns_and_original_bad_response_retained(setup):
    root, http = setup
    original = http.get
    def empty(url, **kwargs):
        response = original(url, **kwargs)
        if 'archive-api' in url:
            response.payload['hourly'] = {'time': []}
            response.content = json.dumps(response.payload).encode()
        return response
    http.get = empty
    weather.run_weather_year(root, 2025, round_numbers=[1])
    assert tasks(root)[0]['status'] == 'failed'
    assert tasks(root)[0]['attempt_count'] == 3
    weather.run_weather_year(root, 2025, round_numbers=[1])
    assert len([url for url, _ in http.calls if 'archive-api' in url]) == 1
    assert tasks(root)[0]['attempt_count'] == 3


def test_calendar_errors_have_bounded_retry_and_clear_summary(setup, monkeypatch):
    root, http = setup
    attempts = []
    def broken(*args):
        attempts.append(1)
        raise ValueError('malformed calendar')
    monkeypatch.setattr(weather, 'load_calendar', broken)
    assert weather.run_weather_year(root, 2025)['counts'] == {'failed': 1}
    assert len(attempts) == 3

def test_explicit_retry_failed_fetches_fresh_payload_and_keeps_original_source(setup):
    root, http = setup
    original = http.get
    def empty(url, **kwargs):
        response = original(url, **kwargs)
        if 'archive-api' in url:
            response.payload['hourly'] = {'time': []}
            response.content = json.dumps(response.payload).encode()
        return response
    http.get = empty
    assert weather.run_weather_year(root, 2025, round_numbers=[1])['counts'] == {'failed': 1}
    old = json.loads(tasks(root)[0]['checkpoint_json'])['source_id']
    http.get = original
    assert weather.run_weather_year(root, 2025, round_numbers=[1], retry_failed=True)['counts'] == {'complete': 1}
    state = json.loads(tasks(root)[0]['checkpoint_json'])
    assert old in state['prior_failed_source_ids']
    assert state['source_id'] != old
    archive = Archive(root / '.local/f1_weather')
    with archive.connect() as conn:
        sources = list(conn.execute("SELECT raw_path FROM source_objects WHERE provider='open_meteo'"))
    assert len(sources) == 2
    assert all((archive.root / row[0]).is_file() for row in sources)


@pytest.mark.parametrize('rounds', [[True], [1.5], ['1'], [0], [-1]])
def test_invalid_round_numbers_fail_before_collection(setup, rounds):
    root, http = setup
    with pytest.raises(ValueError, match='round_numbers'):
        weather.run_weather_year(root, 2025, round_numbers=rounds)
    assert http.calls == []
