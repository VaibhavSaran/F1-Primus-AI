from hashlib import sha256

import pandas as pd
import pyarrow as pa
import pytest

from data.archive import Archive
from data.archive.reader import ArchiveReader


@pytest.fixture
def saved(tmp_path):
    archive = Archive(tmp_path / 'archive')
    source = archive.source('fixture', 'calendar', b'{}', '2026-10-23T00:00:00Z')
    archive.save_calendar(
        [{'event_id': '2025:1', 'season': 2025, 'round_number': 1,
          'event_name': 'Test GP', 'circuit_id': 'test', 'circuit_name': 'Test'}],
        [{'session_id': '2025:1:FP1', 'event_id': '2025:1', 'session_code': 'FP1'}], source)
    rows = {
        'drivers': [{'driver_id': 'a', 'display_name': 'A Driver'}, {'driver_id': 'b', 'display_name': 'B Driver'}],
        'session_entries': [{'driver_id': driver, 'driver_number': str(index + 1),
            'abbreviation': driver.upper(), 'team_name_at_event': 'Test Team'} for index, driver in enumerate(('a', 'b'))],
        'laps': [{'driver_id': 'a', 'lap_number': 1, 'lap_time_ns': 90_000_000_001,
            'sector1_ns': 30_000_000_000, 'lap_start_clock_ns': 10_000_000_001,
            'lap_start_utc_ns': 1_700_000_000_000_000_001, 'stint_number': 1,
            'lap_end_clock_ns': 100_000_000_002, 'pit_out_clock_ns': 10_000_000_001}],
        'weather_series': [{'series_id': 'weather', 'provider': 'fixture', 'kind': 'track_observation', 'source_id': source}],
        'weather_observations': [{'series_id': 'weather', 'sample_index': 0,
            'session_clock_ns': 10_000_000_001, 'relative_humidity_pct': 70,
            'valid_time_utc_ns': 1_700_000_000_000_000_001, 'source_id': source}]}
    datasets = [{'kind': kind, 'driver_id': driver, 'table': pa.table({
        'driver_id': [driver, driver], 'session_clock_ns': [10_000_000_001] * 2,
        'timestamp_utc_ns': [1_700_000_000_000_000_001, None], 'sample_index': [0, 1],
        'speed_kmh': [100., 101.], 'unknown_channel': ['x', 'y']})}
        for driver in ('a', 'b') for kind in ('car', 'positions')]
    version = archive.publish_session('2025:1:FP1', rows, datasets, [source], {'t0_utc_ns': 1_699_999_990_000_000_000})
    return archive, source, rows, version


def test_missing_reader_never_initializes(tmp_path):
    root = tmp_path / 'absent'
    reader = ArchiveReader(root)
    assert reader.available_sessions() == []
    with pytest.raises(ValueError, match='missing'):
        reader.tables('2025:1:FP1')
    assert not root.exists()


def test_rich_published_version_selected_consistently(saved):
    archive, source, rows, version = saved
    sparse = {**rows, 'laps': [{'driver_id': 'a', 'lap_number': 1, 'lap_time_ns': 91_000_000_000}]}
    archive.publish_session('2025:1:FP1', sparse, [], [source], {})
    reader = ArchiveReader(archive.root)
    assert reader.available_sessions()[0]['version_id'] == version
    tables = reader.tables('2025:1:FP1')
    assert tables['meta']['version_id'] == version
    assert tables['laps'].iloc[0]['lap_start_utc_ns'] == 1_700_000_000_000_000_001
    assert tables['laps'].iloc[0]['lap_time_s'] == pytest.approx(90.000000001)
    assert tables['entries'].iloc[0]['display_name'] == 'A Driver'
    assert tables['weather'].iloc[0]['relative_humidity_pct'] == 70
    assert tables['weather'].iloc[0]['session_time_s'] == pytest.approx(10.000000001)
    with archive.connect() as connection:
        assert connection.execute('SELECT version_id FROM v_current_laps').fetchone()[0] == version
        connection.execute("UPDATE session_versions SET status='staged' WHERE version_id=?", (version,))
    assert reader.available_sessions()[0]['version_id'] != version


def test_no_laps_falls_back_to_latest_published_and_unknown_rejected(saved):
    archive, source, rows, version = saved
    with archive.connect() as connection:
        connection.execute('DELETE FROM laps')
    newest = archive.publish_session('2025:1:FP1', {**rows, 'laps': []}, [], [source], {})
    reader = ArchiveReader(archive.root)
    assert reader.tables('2025:1:FP1')['meta']['version_id'] == newest
    with pytest.raises(ValueError, match='No published'):
        reader.tables("x' OR 1=1 --")


def test_dense_pushdown_retains_duplicate_clocks_exact_ns_and_selected_driver(saved, monkeypatch):
    archive, _, _, _ = saved
    reader = ArchiveReader(archive.root)
    import data.archive.reader as module
    original = module.pq.ParquetFile
    opened = []
    def tracked(path):
        opened.append(path)
        return original(path)
    monkeypatch.setattr(module.pq, 'ParquetFile', tracked)
    result = reader.dense('2025:1:FP1', 'car', ['a'], ['speed_kmh'])
    assert len(opened) == 1
    assert len(result) == 2
    assert result['driver_id'].tolist() == ['a', 'a']
    assert result.iloc[0]['timestamp_utc_ns'] == 1_700_000_000_000_000_001
    assert pd.isna(result.iloc[1]['timestamp_utc_ns'])
    assert result['session_time_s'].tolist() == pytest.approx([10.000000001] * 2)
    assert 'unknown_channel' not in result
    assert 'unknown_channel' in reader.dense('2025:1:FP1', 'positions', ['a'])
    with pytest.raises(ValueError, match='kind'):
        reader.dense('2025:1:FP1', 'raw_car')


def test_dense_rejects_corruption_and_manifest_path_escape(saved, tmp_path):
    archive, _, _, version = saved
    reader = ArchiveReader(archive.root)
    with archive.connect() as connection:
        record = dict(connection.execute("SELECT * FROM dataset_files WHERE kind='car' AND driver_id='a'").fetchone())
    path = archive.root / record['relative_path']
    path.write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='hash'):
        reader.dense('2025:1:FP1', 'car', ['a'])
    outside = tmp_path / 'outside.parquet'
    outside.write_bytes(b'corrupted')
    with archive.connect() as connection:
        connection.execute('UPDATE dataset_files SET relative_path=?,file_sha256=? WHERE file_id=?',
            ('../outside.parquet', sha256(b'corrupted').hexdigest(), record['file_id']))
    with pytest.raises(ValueError, match='path'):
        reader.dense('2025:1:FP1', 'car', ['a'])


def test_manifest_row_limits_apply_before_opening_parquet(saved, monkeypatch):
    archive, _, _, _ = saved
    with archive.connect() as connection:
        connection.execute("UPDATE dataset_files SET row_count=2000001 WHERE kind='car' AND driver_id='a'")
    with pytest.raises(ValueError, match='limit'):
        ArchiveReader(archive.root).dense('2025:1:FP1', 'car', ['a'])


def test_empty_dense_selection_and_missing_columns(saved):
    archive, _, _, _ = saved
    reader = ArchiveReader(archive.root)
    for drivers in ([], ['missing']):
        empty = reader.dense('2025:1:FP1', 'positions', drivers)
        assert empty.empty
        assert 'session_time_s' in empty
    with pytest.raises(ValueError, match='column'):
        reader.dense('2025:1:FP1', 'car', ['a'], ['invented_channel'])


def test_readonly_connection_and_row_count_verification(saved):
    archive, _, _, _ = saved
    reader = ArchiveReader(archive.root)
    import sqlite3
    with reader._connect() as connection:
        with pytest.raises(sqlite3.OperationalError, match='readonly'):
            connection.execute('DELETE FROM laps')
    with archive.connect() as connection:
        connection.execute("UPDATE dataset_files SET row_count=3 WHERE kind='car' AND driver_id='a'")
    with pytest.raises(ValueError, match='row count'):
        reader.dense('2025:1:FP1', 'car', ['a'])
