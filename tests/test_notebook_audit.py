import json
from datetime import datetime, timezone

from test_archive_reader import saved


def test_audit_separates_missing_partial_and_known_cancelled(saved, tmp_path):
    from data.archive.notebook_audit import audit_year
    archive, source, rows, version = saved
    # The fixture publication is partial; a newer sparse publication must not hide it.
    archive.save_calendar([], [{'session_id': '2025:1:Q', 'event_id': '2025:1', 'session_code': 'Q',
                               'full_name': 'Qualifying', 'scheduled_start_utc_ns': 1}], source)
    progress = {'schema_version': 1, 'sessions': {'2025:1:Q': {'status': 'deferred', 'reason': 'quota'}}}
    (archive.root / 'progress-f1-2025.json').write_text(json.dumps(progress))
    report = audit_year(archive.root, 2025)
    by_id = {row['session_id']: row for row in report['sessions']}
    assert by_id['2025:1:FP1']['action'] == 'review_partial'
    assert by_id['2025:1:Q']['action'] == 'download'
    assert by_id['2025:1:Q']['checkpoint_status'] == 'deferred'
    assert by_id['2025:1:FP1']['laps'] == 1


def test_cancelled_sessions_not_queued_and_corruption_detected(saved):
    from data.archive.notebook_audit import audit_year
    archive, source, _, _ = saved
    archive.save_calendar([{'event_id': '2020:2', 'season': 2020, 'round_number': 2, 'event_name': 'Styrian GP'}],
        [{'session_id': '2020:2:FP3', 'event_id': '2020:2', 'session_code': 'FP3', 'full_name': 'Practice 3'}], source)
    report = audit_year(archive.root, 2020)
    assert report['sessions'][0]['action'] == 'not_held'
    assert report['sessions'][0]['reference'].startswith('https://www.formula1.com/')
    with archive.connect() as connection:
        path = archive.root / connection.execute('SELECT relative_path FROM dataset_files LIMIT 1').fetchone()[0]
    path.write_bytes(b'corrupt')
    report = audit_year(archive.root, 2025, verify_files=True)
    assert report['sessions'][0]['action'] == 'repair_local'
    assert report['integrity']['file_errors']


def test_audit_missing_archive_never_creates_one(tmp_path):
    from data.archive.notebook_audit import audit_year
    import pytest
    root = tmp_path / 'missing'
    with pytest.raises(FileNotFoundError):
        audit_year(root, 2020)
    assert not root.exists()


def test_legacy_identity_sentinels_are_reprocess_targets(saved):
    from data.archive.notebook_audit import audit_year
    archive, _, _, _ = saved
    with archive.connect() as connection:
        connection.execute("UPDATE session_entries SET source_driver_id='nan' WHERE driver_id='a'")
    report = audit_year(archive.root, 2025)
    assert report['sessions'][0]['action'] == 'reprocess_identity'
    assert report['remaining_session_ids'] == ['2025:1:FP1']
    assert report['reprocess_session_ids'] == ['2025:1:FP1']


def test_race_date_cutoff_excludes_later_weekends_and_files(saved):
    from data.archive.notebook_audit import audit_year
    archive, source, _, _ = saved
    archive.save_calendar([
        dict(event_id='2026:18', season=2026, round_number=18, event_name='United States GP', local_race_date='2026-10-25'),
        dict(event_id='2026:19', season=2026, round_number=19, event_name='Later GP', local_race_date='2026-11-01')],
        [dict(session_id=f'2026:{number}:R', event_id=f'2026:{number}', session_code='R') for number in (18, 19)], source)
    report = audit_year(archive.root, 2026, through_date='2026-10-25', verify_files=True)
    assert [row['session_id'] for row in report['sessions']] == ['2026:18:R']
    assert report['integrity']['files_verified'] == 0


def test_2021_cancelled_russian_practice_is_not_missing(saved):
    from data.archive.notebook_audit import audit_year
    archive, source, _, _ = saved
    archive.save_calendar([dict(event_id='2021:15', season=2021, round_number=15, event_name='Russian GP')],
        [dict(session_id='2021:15:FP3', event_id='2021:15', session_code='FP3')], source)
    report = audit_year(archive.root, 2021)
    assert report['counts'] == {'not_held': 1}
    assert report['remaining_session_ids'] == []


def test_unreadable_or_malformed_checkpoint_does_not_hide_publications(saved):
    from data.archive.notebook_audit import audit_year
    archive, _, _, version = saved
    for content in ('{"sessions":', '[]', '{"sessions": []}', '{"sessions": {"bad": 5}}'):
        (archive.root/'progress-f1-2025.json').write_text(content)
        report = audit_year(archive.root, 2025)
        assert report['sessions'][0]['version_id'] == version
        assert report['checkpoint_warning']
