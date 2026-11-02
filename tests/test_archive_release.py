"""Publication preparation never needs raw cache or provider access."""
import hashlib
import json
import sqlite3
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data.archive.release import (checked_path, write_bundle, safe_source,
    validate_release, build_year_release, build_release_index, write_context_availability)


def test_paths_reject_traversal_and_links(tmp_path):
    (tmp_path / 'ok').mkdir()
    assert checked_path(tmp_path, 'ok/a.parquet') == tmp_path / 'ok/a.parquet'
    for bad in ('../secret', '/secret', 'C:/secret', 'ok/../../secret', 'ok\\..\\secret',
                'ok/.. /secret', 'ok/alternate:stream'):
        with pytest.raises(ValueError):
            checked_path(tmp_path, bad)


def test_bundle_streams_exact_bytes_and_rejects_corruption(tmp_path):
    source = tmp_path / 'archive'
    source.mkdir()
    pq.write_table(pa.table({'driver_id': ['a'], 'session_clock_ns': [1]}), source / 'a.parquet')
    digest = hashlib.sha256((source / 'a.parquet').read_bytes()).hexdigest()
    item = {'relative_path': 'a.parquet', 'file_sha256': digest, 'row_count': 1,
            'kind': 'car', 'driver_id': 'a'}
    target = tmp_path / 'round.zip'
    records = write_bundle(source, target, [('events/round=01/sessions/R/car/a.parquet', item)])
    assert records[0]['sha256'] == digest
    with zipfile.ZipFile(target) as bundle:
        assert bundle.read(records[0]['member']) == (source / 'a.parquet').read_bytes()
        assert bundle.testzip() is None
    item['file_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='hash'):
        write_bundle(source, tmp_path / 'bad.zip', [('a.parquet', item)])
    item['file_sha256'] = digest
    item['row_count'] = 2
    with pytest.raises(ValueError, match='row'):
        write_bundle(source, tmp_path / 'badrows.zip', [('a.parquet', item)])
    item['row_count'] = 1
    item['driver_id'] = 'wrong'
    with pytest.raises(ValueError, match='identity'):
        write_bundle(source, tmp_path / 'badidentity.zip', [('a.parquet', item)])


def test_source_allowlist_drops_credentials_paths_and_urls():
    assert safe_source({'source_id': 's', 'provider': 'p', 'payload_sha256': 'h',
                        'raw_path': 'private', 'canonical_url': 'https://host/?token=secret',
                        'headers': {'authorization': 'secret'}}) == {
                            'source_id': 's', 'provider': 'p', 'payload_sha256': 'h'}


def test_validation_detects_changed_payload(tmp_path):
    (tmp_path / 'README.md').write_text('safe')
    record = {'path': 'README.md', 'sha256': hashlib.sha256(b'safe').hexdigest(), 'bytes': 4}
    (tmp_path / 'manifest.json').write_text(json.dumps({'files': [record], 'members': []}))
    assert validate_release(tmp_path)['files'] == 1
    (tmp_path / 'README.md').write_text('changed')
    with pytest.raises(ValueError, match='hash|size'):
        validate_release(tmp_path)


@pytest.mark.parametrize('private', ['C:\\private\\cache', 'archive:///local/file',
    'somebody@example.com', 'Bearer abcdefghijklmnopqrstuv', 'api_key=abcdefghijklmnop'])
def test_validation_scans_actual_payload_metadata(tmp_path, private):
    path = tmp_path / 'README.md'
    path.write_text(private, encoding='utf-8')
    (tmp_path / 'manifest.json').write_text(json.dumps({'files': [{'path': path.name,
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}]}))
    with pytest.raises(ValueError, match='Private'):
        validate_release(tmp_path)


def test_year_release_is_portable_pinned_and_immutable(tmp_path):
    from data.archive import Archive
    archive = Archive(tmp_path / 'f1')
    weather = Archive(tmp_path / 'weather')
    source = archive.source('fixture', 'calendar', b'{}', '2026-11-01T00:00:00Z')
    archive.save_calendar([{'event_id': '2025:1', 'season': 2025, 'round_number': 1,
        'event_name': 'Fixture', 'local_race_date': '2025-03-01'}],
        [{'session_id': '2025:1:R', 'event_id': '2025:1', 'session_code': 'R'},
         {'session_id': '2025:1:FP1', 'event_id': '2025:1', 'session_code': 'FP1', 'status': 'cancelled'},
         {'session_id': '2025:1:FP2', 'event_id': '2025:1', 'session_code': 'FP2', 'status': 'scheduled'}], source)
    rows = {'drivers': [{'driver_id': 'a', 'display_name': 'A'}, {'driver_id': 'b', 'display_name': 'B'}],
        'session_entries': [{'driver_id': 'a', 'driver_number': '1'}, {'driver_id': 'b', 'driver_number': '2'}],
        'session_results': [{'driver_id': 'a', 'status_raw': 'Retired', 'normalized_outcome_status': 'retired'},
            {'driver_id': 'b', 'status_raw': 'Did not start', 'normalized_outcome_status': 'dns'}],
        'laps': [{'driver_id': 'a', 'lap_number': 1, 'lap_time_ns': 90_000_000_001}]}
    version = archive.publish_session('2025:1:R', rows, [], [source], {})
    cohort = {'sessions': [{'session_id': '2025:1:R', 'version_id': version, 'season': 2025,
        'round_number': 1, 'session_code': 'R', 'local_race_date': '2025-03-01',
        'files': [], 'source_links': [], 'quality_flags': ['partial'],
        'required_channel_status': {'car': 'unavailable'}, 'strict_review_subset': False}], 'weather_tasks': []}
    pinned = tmp_path / 'cohort' / 'manifest.json'
    pinned.parent.mkdir()
    pinned.write_text(json.dumps(cohort))
    context = tmp_path / 'context'
    context.mkdir()
    with sqlite3.connect(context / 'context.sqlite3') as db:
        db.execute('CREATE TABLE context_records(record_id TEXT,record_sha256 TEXT,event_id TEXT,known_at TEXT,canonical_record TEXT)')
        db.execute('INSERT INTO context_records VALUES (?,?,?,?,?)', ('record', 'original', '2025:1',
            '2026-11-01T00:00:00Z', json.dumps({'event_id': '2025:1', 'version_id': version,
                'summary': 'Recorded outcome', 'archive_path': 'C:/private/archive'})))
    before = hashlib.sha256(archive.db_path.read_bytes()).hexdigest()
    root = build_year_release(archive.root, weather.root, tmp_path / 'releases', 2025,
                              cohort_path=pinned, context_root=context)
    assert validate_release(root)['passed']
    with sqlite3.connect(root / 'catalogue.sqlite') as db:
        assert db.execute('SELECT lap_time_ns FROM laps').fetchone()[0] == 90_000_000_001
        assert db.execute("SELECT status_raw FROM session_results WHERE driver_id='a'").fetchone()[0] == 'Retired'
        assert db.execute("SELECT normalized_outcome_status FROM session_results WHERE driver_id='b'").fetchone()[0] == 'dns'
        clean_context = json.loads(db.execute('SELECT canonical_record FROM curated_context').fetchone()[0])
        assert clean_context['known_at'] == '2026-11-01T00:00:00Z'
        assert 'archive_path' not in clean_context
    quality = json.loads((root / 'quality.json').read_text())
    assert quality['sessions'][0]['quality_flags'] == ['partial']
    assert quality['calendar_sessions_without_publication'][0]['status'] == 'cancelled'
    assert quality['calendar_sessions_without_publication'][1]['status'] == 'scheduled'
    assert hashlib.sha256(archive.db_path.read_bytes()).hexdigest() == before
    sidecars = write_context_availability(tmp_path / 'releases', context)
    index = build_release_index(tmp_path / 'releases')
    assert len(index['datasets']) == 1
    assert index['datasets'][0]['kaggle_url'] is None
    indexed = index['datasets'][0]['context_availability']
    sidecar = tmp_path / 'releases' / indexed['path']
    assert indexed['sha256'] == hashlib.sha256(sidecar.read_bytes()).hexdigest() == sidecars[0]['sha256']
    availability = json.loads(sidecar.read_text())
    with sqlite3.connect(context / 'context.sqlite3') as db:
        assert availability['records'][0]['known_at'] == db.execute('SELECT known_at FROM context_records WHERE record_id=?', ('record',)).fetchone()[0]
    assert availability['records'][0]['exported_record_sha256'] == hashlib.sha256(json.dumps(clean_context, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    assert availability['known_at_basis'] == 'local_context_store_knowledge_time_not_event_date'
    assert any(s['session_id'] == '2021:15:FP3' for s in index['known_cancelled_sessions'])
    with pytest.raises(ValueError, match='immutable'):
        build_year_release(archive.root, weather.root, tmp_path / 'releases', 2025, cohort_path=pinned)
    other = build_year_release(archive.root, weather.root, tmp_path / 'other', 2025,
                               cohort_path=pinned, context_root=context)
    assert (root / 'manifest.json').read_bytes() == (other / 'manifest.json').read_bytes()
    with pytest.raises(ValueError, match='separate'):
        build_year_release(archive.root, weather.root, archive.root / 'exports', 2025, cohort_path=pinned)
    cohort['sessions'][0].update(season=2026, local_race_date='2026-11-08')
    pinned.write_text(json.dumps(cohort))
    with pytest.raises(ValueError, match='cutoff'):
        build_year_release(archive.root, weather.root, tmp_path / 'future', 2026, cohort_path=pinned)
