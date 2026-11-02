import json
import sqlite3
from hashlib import sha256

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data.archive.weather_audit import audit_weather


@pytest.fixture
def saved(tmp_path):
    root = tmp_path / 'weather'
    root.mkdir()
    pq.write_table(pa.table({'sample_index': [0, 1]}), root / 'weather.parquet')
    (root / 'raw.json').write_bytes(b'weather')
    (root / 'coordinates.json').write_bytes(b'coordinates')
    state = dict(series_id='series', source_id='weather-source', observations=2,
                 circuit={'coordinate_source_id': 'coordinates-source'},
                 parquet_path='weather.parquet',
                 parquet_sha256=sha256((root / 'weather.parquet').read_bytes()).hexdigest(),
                 coverage={'status': 'available'})
    with sqlite3.connect(root / 'f1.sqlite3') as conn:
        conn.executescript('''
            CREATE TABLE collection_tasks(task_id TEXT, event_id TEXT, dataset_kind TEXT,
                config_version TEXT, status TEXT, next_retry_utc TEXT, checkpoint_json TEXT);
            CREATE TABLE weather_series(series_id TEXT, source_id TEXT);
            CREATE TABLE weather_observations(series_id TEXT, source_id TEXT);
            CREATE TABLE source_objects(source_id TEXT, raw_path TEXT, payload_sha256 TEXT);
        ''')
        conn.execute('INSERT INTO collection_tasks VALUES (?,?,?,?,?,?,?)',
                     ('yearly_weather_v1:2026:1', '2026:1', 'weather', 'yearly_weather_v1', 'complete', None, json.dumps(state)))
        conn.execute('INSERT INTO weather_series VALUES (?,?)', ('series', 'weather-source'))
        conn.executemany('INSERT INTO weather_observations VALUES (?,?)', [('series', 'weather-source')] * 2)
        for source, path in [('weather-source', 'raw.json'), ('coordinates-source', 'coordinates.json')]:
            conn.execute('INSERT INTO source_objects VALUES (?,?,?)',
                         (source, path, sha256((root / path).read_bytes()).hexdigest()))
    return root


def test_complete_saved_evidence_and_read_only(saved):
    before = (saved / 'f1.sqlite3').read_bytes()
    result = audit_weather(saved, ['2026:1'], verify_files=True)
    assert result['counts'] == {'complete': 1}
    assert result['observations'] == 2
    assert result['integrity']['file_hashes_checked'] == 3
    assert result['integrity']['raw_sources_verified'] == 2
    assert (saved / 'f1.sqlite3').read_bytes() == before


def test_missing_database_does_not_create_directory(tmp_path):
    target = tmp_path / 'absent'
    result = audit_weather(target, ['2026:1'])
    assert result['events'][0]['status'] == 'unavailable'
    assert not target.exists()


def test_missing_task_and_expected_ids_are_the_only_scope(saved):
    result = audit_weather(saved, ['2026:2'])
    assert result['counts'] == {'missing': 1}
    assert result['expected_events'] == 1
    assert result['events'][0]['event_id'] == '2026:2'


@pytest.mark.parametrize('damage', ['parquet', 'rows', 'source', 'unsafe', 'version', 'ambiguous', 'checkpoint'])
def test_bad_saved_evidence_is_repair_local(saved, damage):
    with sqlite3.connect(saved / 'f1.sqlite3') as conn:
        if damage == 'rows':
            conn.execute("UPDATE weather_observations SET source_id='other'")
        elif damage == 'version':
            conn.execute("UPDATE collection_tasks SET config_version='other'")
        elif damage == 'unsafe':
            conn.execute("UPDATE source_objects SET raw_path='../outside.json'")
        elif damage == 'ambiguous':
            conn.execute('INSERT INTO collection_tasks SELECT * FROM collection_tasks')
        elif damage == 'checkpoint':
            conn.execute("UPDATE collection_tasks SET checkpoint_json='[]'")
    if damage == 'parquet':
        (saved / 'weather.parquet').write_bytes(b'corrupt')
    elif damage == 'source':
        (saved / 'raw.json').write_bytes(b'changed')
    result = audit_weather(saved, ['2026:1'], verify_files=True)
    assert result['counts'] == {'repair_local': 1}
    assert result['complete_events'] == 0


def test_default_checks_parquet_rows_and_missing_raw_files(saved):
    pq.write_table(pa.table({'sample_index': [0]}), saved / 'weather.parquet')
    assert audit_weather(saved, ['2026:1'])['counts'] == {'repair_local': 1}
    pq.write_table(pa.table({'sample_index': [0, 1]}), saved / 'weather.parquet')
    (saved / 'coordinates.json').unlink()
    assert audit_weather(saved, ['2026:1'])['counts'] == {'repair_local': 1}


def test_default_explicitly_reports_hashes_not_checked(saved):
    result = audit_weather(saved, ['2026:1'])
    assert result['counts'] == {'complete': 1}
    assert result['integrity']['file_hashes_checked'] == 0
    assert 'not checked' in result['events'][0]['detail']


def test_pending_retry_is_reported_without_exposing_checkpoint(saved):
    with sqlite3.connect(saved / 'f1.sqlite3') as conn:
        conn.execute("UPDATE collection_tasks SET status='pending', next_retry_utc='2026-11-01T00:00:00Z', checkpoint_json=?",
                     (json.dumps({'secret': 'never-print-this'}),))
    result = audit_weather(saved, ['2026:1'])
    assert result['counts'] == {'pending': 1}
    assert result['events'][0]['next_retry_utc'] == '2026-11-01T00:00:00Z'
    assert 'never-print-this' not in str(result)
