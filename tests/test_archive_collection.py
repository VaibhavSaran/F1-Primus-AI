from datetime import datetime, timezone
import json

from data.archive.collect import Collector, cutoff_utc, schedule_frame
from data.archive.store import Archive
import pytest


def test_collector_lock_reports_busy_across_processes(tmp_path):
    import subprocess
    import sys
    from data.archive.collect import worker_lock
    code = '''
import sys
from pathlib import Path
from data.archive.collect import worker_lock
try:
    with worker_lock(Path(sys.argv[1])):
        raise AssertionError('Second collector acquired the lock')
except RuntimeError as error:
    assert str(error) == 'Another archive collector owns this root'
    print('busy')
'''
    with worker_lock(tmp_path):
        child = subprocess.run([sys.executable, '-c', code, str(tmp_path)], capture_output=True, text=True, timeout=20)
    assert child.returncode == 0, child.stderr
    assert child.stdout.strip() == 'busy'


def test_cutoff_includes_local_october_23():
    assert cutoff_utc('2026-10-23', 'America/Chicago') == datetime(2026, 10, 24, 5, tzinfo=timezone.utc)


def test_schedule_preserves_local_date_and_utc():
    raw = {'round_number': {'0': 1}, 'event_date': {'0': '2026-10-23'},
           'gmt_offset': {'0': '-05:00'}, 'session1_date': {'0': '2026-10-23T23:30:00'}}
    frame = schedule_frame(raw)
    assert str(frame.iloc[0]['EventDate'].date()) == '2026-10-23'
    assert frame.iloc[0]['Session1DateUtc'].isoformat() == '2026-10-24T04:30:00+00:00'


def test_tasks_resume_without_losing_page_checkpoint(tmp_path):
    archive = Archive(tmp_path)
    collector = Collector(archive, end=cutoff_utc('2026-10-23', 'America/Chicago'))
    collector.queue('calendar:2018', 'jolpica_calendar', season=2018, config={'year': 2018})
    task = collector.next_task()
    collector.save_task(task['task_id'], 'pending', {'offset': 100, 'source_ids': ['abc']})
    collector.queue('calendar:2018', 'jolpica_calendar', season=2018, config={'year': 2018})
    resumed = collector.next_task()
    assert json.loads(resumed['checkpoint_json'])['offset'] == 100
    collector.save_task(resumed['task_id'], 'complete', {'offset': 100})
    assert collector.next_task() is None


def test_future_sessions_never_get_data_tasks(tmp_path):
    archive = Archive(tmp_path)
    collector = Collector(archive, end=cutoff_utc('2026-10-23', 'America/Chicago'))
    source = archive.source('fixture', 'schedule', b'{}', '2026-10-24T00:00:00Z')
    archive.save_calendar([{'event_id': '2026:1', 'season': 2026, 'round_number': 1}], [
        {'session_id': '2026:1:FP1', 'event_id': '2026:1', 'session_code': 'FP1', 'scheduled_start_utc_ns': 1792767600000000000},
        {'session_id': '2026:1:R', 'event_id': '2026:1', 'session_code': 'R', 'scheduled_start_utc_ns': 1793026800000000000}], source)
    collector.plan_sessions(2026)
    with archive.connect() as conn:
        rows = conn.execute('SELECT session_id FROM collection_tasks').fetchall()
    assert [r[0] for r in rows] == ['2026:1:FP1']


class Response:
    url = 'https://fixture/2018.json'
    content = b'{"fixture": true}'
    def json(self):
        return {'MRData': {'limit': '100', 'offset': '0', 'total': '1', 'RaceTable': {'Races': []}}}


class HTTP:
    def get(self, *args, **kwargs):
        return Response()


def test_page_side_effect_failure_never_advances_checkpoint(tmp_path, monkeypatch):
    archive = Archive(tmp_path)
    collector = Collector(archive, end=cutoff_utc('2026-10-23', 'America/Chicago'), http=HTTP())
    collector.queue('2018', 'jolpica_calendar', season=2018, config={'endpoint': ''})
    task = collector.next_task()
    monkeypatch.setattr(collector, 'enrich_calendar', lambda *args: (_ for _ in ()).throw(ValueError('failure applying page')))
    collector.execute(task)
    with archive.connect() as conn:
        saved = conn.execute('SELECT * FROM collection_tasks').fetchone()
        assert conn.execute('SELECT count(*) FROM source_objects').fetchone()[0] == 1
    state = json.loads(saved['checkpoint_json'])
    assert state.get('offset', 0) == 0
    assert state.get('source_ids', []) == []


def test_publishing_is_recoverable_without_network(tmp_path, monkeypatch):
    archive = Archive(tmp_path)
    collector = Collector(archive, end=cutoff_utc('2026-10-23', 'America/Chicago'), http=HTTP())
    source = archive.source('fixture', 'calendar', b'{}', '2026-10-24T00:00:00Z')
    archive.save_calendar([{'event_id': '2018:1', 'season': 2018, 'round_number': 1}], [], source)
    collector.queue('2018:1:results', 'publish', season=2018, event_id='2018:1')
    task = collector.next_task()
    task['attempt_count'] = 100  # Quota-only attempts never consume the genuine failure budget.
    monkeypatch.setattr(collector, 'publish_jolpica', lambda *args: (_ for _ in ()).throw(ValueError('publication failed')))
    collector.execute(task)
    with archive.connect() as conn:
        saved = conn.execute('SELECT * FROM collection_tasks').fetchone()
    assert saved['status'] == 'pending'
    assert json.loads(saved['checkpoint_json'])['_failures'] == 1


def test_scoped_pending_count_ignores_other_kinds(tmp_path):
    collector = Collector(Archive(tmp_path), end=cutoff_utc('2026-10-23', 'America/Chicago'), kinds='results')
    collector.queue('2018', 'jolpica_calendar', season=2018)
    assert collector.pending_count() == 0
    collector.queue('2018', 'results', season=2018)
    assert collector.pending_count() == 1


def test_manual_retry_resets_failure_budget_and_unavailable_pagination(tmp_path):
    collector = Collector(Archive(tmp_path), end=cutoff_utc('2026-10-23', 'America/Chicago'))
    task_id = collector.queue('2018', 'results', season=2018, config={'endpoint': '1/results'})
    collector.save_task(task_id, 'unavailable', {'config': {'endpoint': '1/results'}, 'offset': 100, 'total': 0, 'source_ids': ['old'], '_failures': 4})
    collector.retry('unavailable')
    task = collector.next_task()
    assert json.loads(task['checkpoint_json']) == {'config': {'endpoint': '1/results'}}
    collector.save_task(task_id, 'failed', {'config': {}, 'offset': 100, '_failures': 4})
    collector.retry('failed')
    state = json.loads(collector.next_task()['checkpoint_json'])
    assert '_failures' not in state
    assert state['offset'] == 100


def test_retrieved_new_evidence_queues_a_fresh_publication(tmp_path):
    archive = Archive(tmp_path)
    collector = Collector(archive, end=cutoff_utc('2026-10-23', 'America/Chicago'))
    source = archive.source('fixture', 'calendar', b'{}', '2026-10-24T00:00:00Z')
    archive.save_calendar([{'event_id': '2018:1', 'season': 2018, 'round_number': 1}], [], source)
    task_id = collector.queue('2018:1', 'results', season=2018, event_id='2018:1')
    task = {'task_id': task_id, 'event_id': '2018:1', 'dataset_kind': 'results', 'season': 2018}
    collector.save_task(task_id, 'unavailable', {'source_ids': ['empty-response']})
    collector.queue_publication(task)
    publication = collector.next_task()
    collector.save_task(publication['task_id'], 'complete', {})
    collector.save_task(task_id, 'complete', {'source_ids': ['real-results']})
    collector.queue_publication(task)
    assert collector.next_task()['task_id'] != publication['task_id']
