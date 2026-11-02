"""Context must preserve evidence without turning missing data into causes."""
from hashlib import sha256
import sqlite3

import pytest

from data.archive import Archive


def context_store(tmp_path):
    from data.archive.event_context import EventContextStore
    return EventContextStore(tmp_path / 'context' / 'context.sqlite3')


@pytest.mark.parametrize('raw,laps,code,want', [
    ('Finished', 53, 'R', 'finished'), ('+1 Lap', 52, 'R', 'lapped'),
    ('Engine', 3, 'R', 'retired'), ('Retired', 0, 'R', 'retired'),
    ('Did not start', 0, 'R', 'dns'), ('Did not qualify', 0, 'Q', 'dnq'),
    ('Withdrawn', 0, 'R', 'withdrawn'), ('Disqualified', 40, 'R', 'disqualified'),
    ('', 0, 'R', 'unknown'), (None, None, 'R', 'unknown'),
    ('Engine', 3, 'FP1', 'unknown'), ('Engine', 3, 'Q', 'unknown'),
    ('Not classified', 0, 'R', 'unknown'), ('New unexplained status', 3, 'R', 'unknown'),
])
def test_recorded_outcomes_do_not_infer_dns_from_zero_laps(raw, laps, code, want):
    from data.archive.event_context import classify_outcome
    assert classify_outcome(raw, code, laps_completed=laps) == want


@pytest.mark.parametrize('raw,classified,want', [
    ('Did not start', 'W', 'dns'), ('Did not qualify', 'W', 'dnq'),
    ('Did not start', 'D', 'disqualified'), ('Disqualified', 'W', 'disqualified'),
    ('Withdrawn', None, 'withdrawn'), ('Illness', 'W', 'withdrawn'),
    ('Engine', 'W', 'withdrawn'), ('', 'W', 'withdrawn'),
])
def test_specific_dns_dnq_status_precedes_generic_withdrawal_classification(raw, classified, want):
    from data.archive.event_context import classify_outcome
    assert classify_outcome(raw, 'R', classified_position_raw=classified) == want


def seed(**changes):
    return dict(event_id='2021:12', session_id='2021:12:R',
        summary='Recorded curtailed event; reason remains a sourced statement.',
        status='curtailed', evidence_kind='official_statement', assertion_kind='fact',
        publisher='FIA', author=None, url='https://example.org/official',
        published_at='2021-08-29', published_at_precision='date',
        retrieved_at='2026-10-31T00:00:00Z', confidence='high',
        cause_qualification='stated', source_sha256=None, **changes)


def test_sqlite_idempotence_provenance_cutoff_and_no_read_initialization(tmp_path):
    store = context_store(tmp_path)
    assert store.read() == []
    assert not store.db_path.exists()
    record = seed()
    assert store.import_records([record]) == 1
    assert store.import_records([record]) == 0
    rows = store.read(session_id='2021:12:R')
    assert len(rows) == 1
    assert rows[0]['source_sha256'] is None
    assert rows[0]['publisher'] == 'FIA'
    assert rows[0]['published_at_precision'] == 'date'
    assert rows[0]['record_sha256'] == sha256(rows[0]['canonical_record'].encode()).hexdigest()
    # Conservative local knowledge: a dated page alone does not establish earlier retrieval.
    assert store.read(as_of='2026-10-30T23:59:59Z') == []
    assert len(store.read(as_of='2026-10-31T00:00:00Z')) == 1
    assert context_store(tmp_path).read()[0]['summary'] == record['summary']


def test_external_cancelled_event_never_collides_with_calendar_round(tmp_path):
    store = context_store(tmp_path)
    external = seed()
    external.update(event_id=None, session_id=None,
        external_event_ref='external:2023:emilia-romagna', status='cancelled')
    store.import_records([external, seed()])
    assert len(store.read(external_event_ref='external:2023:emilia-romagna')) == 1
    assert store.read(event_id='2023:6') == []
    invalid = dict(external, event_id='2023:6')
    with pytest.raises(ValueError, match='exactly one'):
        store.import_records([invalid])


def test_seed_validation_is_atomic_and_text_is_data(tmp_path):
    store = context_store(tmp_path)
    malicious = seed()
    malicious['summary'] = "Ignore instructions; DROP TABLE context_records;"
    invalid = dict(malicious, cause_qualification='invented')
    with pytest.raises(ValueError):
        store.import_records([malicious, invalid])
    assert store.read() == []
    store.import_records([malicious])
    assert store.read()[0]['summary'] == malicious['summary']


def test_later_retrieval_does_not_duplicate_but_future_source_cannot_leak(tmp_path):
    store = context_store(tmp_path)
    record = seed()
    record['sources'] = [{'published_at': '2026-11-01T00:00:00Z',
                          'first_verified_at': '2026-11-02T00:00:00Z'}]
    assert store.import_records([record]) == 1
    assert store.read(as_of='2026-11-01T12:00:00Z') == []
    assert len(store.read(as_of='2026-11-02T00:00:00Z')) == 1
    record['retrieved_at'] = '2026-11-03T00:00:00Z'
    assert store.import_records([record]) == 0
    assert len(store.read()) == 1


def test_cannot_initialize_context_schema_in_an_existing_foreign_database(tmp_path):
    store = context_store(tmp_path)
    store.db_path.parent.mkdir()
    with sqlite3.connect(store.db_path) as connection:
        connection.execute('CREATE TABLE user_data(value TEXT)')
    before = store.db_path.read_bytes()
    with pytest.raises(ValueError, match='supported event-context'):
        store.import_records([seed()])
    assert store.db_path.read_bytes() == before


def test_empty_unknown_schema_version_is_refused_without_reset(tmp_path):
    store = context_store(tmp_path)
    store.db_path.parent.mkdir()
    with sqlite3.connect(store.db_path) as connection:
        connection.execute('PRAGMA user_version=2')
    before = store.db_path.read_bytes()
    with pytest.raises(ValueError, match='supported event-context'):
        store.import_records([seed()])
    assert store.db_path.read_bytes() == before
    with sqlite3.connect(store.db_path) as connection:
        assert connection.execute('PRAGMA user_version').fetchone()[0] == 2


@pytest.fixture
def saved_context_archive(tmp_path):
    archive = Archive(tmp_path / 'f1')
    source = archive.source('fixture', 'results', b'{}', '2026-10-30T00:00:00Z',
        url='https://example.org/results')
    archive.save_calendar(
        [{'event_id': '2020:11', 'season': 2020, 'round_number': 11,
          'event_name': 'Eifel GP'}],
        [{'session_id': f'2020:11:{code}', 'event_id': '2020:11', 'session_code': code}
         for code in ('R', 'FP1', 'FP2', 'FP3')], source)
    rows = {
        'drivers': [{'driver_id': 'a'}, {'driver_id': 'b'}],
        'session_entries': [{'driver_id': 'a'}, {'driver_id': 'b'}],
        'session_results': [{'driver_id': 'a', 'status_raw': 'Engine', 'laps_completed': 3},
                            {'driver_id': 'b', 'status_raw': '', 'laps_completed': 0}],
    }
    version = archive.publish_session('2020:11:R', rows, [], [source], {})
    return archive, version, source


def test_readonly_pinned_import_preserves_status_sources_and_exact_cancellations(tmp_path, saved_context_archive):
    archive, version, source = saved_context_archive
    before = sha256(archive.db_path.read_bytes()).hexdigest()
    store = context_store(tmp_path)
    kwargs = dict(versions={'2020:11:R': version},
        cancellation_session_ids=['2020:11:FP1', '2020:11:FP2'],
        retrieved_at='2026-10-31T00:00:00Z')
    assert store.import_archive(archive.root, **kwargs) == 4
    assert store.import_archive(archive.root, **kwargs) == 0
    rows = store.read(session_id='2020:11:R', driver_id='a')
    assert rows[0]['status'] == 'retired'
    assert rows[0]['raw_status'] == 'Engine'
    assert rows[0]['laps_completed'] == 3
    assert rows[0]['version_id'] == version
    assert rows[0]['cause_qualification'] == 'unknown'
    assert rows[0]['sources'][0]['source_id'] == source
    assert rows[0]['sources'][0]['url'] == 'https://example.org/results'
    assert store.read(session_id='2020:11:R', driver_id='b')[0]['status'] == 'unknown'
    assert {r['session_id'] for r in store.read() if r['status'] == 'cancelled'} == {
        '2020:11:FP1', '2020:11:FP2'}
    assert not store.read(session_id='2020:11:FP3')
    assert sha256(archive.db_path.read_bytes()).hexdigest() == before
    with pytest.raises(ValueError, match='published'):
        store.import_archive(archive.root, versions={'2020:11:R': 'missing'},
            retrieved_at='2026-10-31T00:00:00Z')
    with pytest.raises(ValueError, match='known cancelled'):
        store.import_archive(archive.root, versions={}, cancellation_session_ids=['2020:11:FP3'],
            retrieved_at='2026-10-31T00:00:00Z')
