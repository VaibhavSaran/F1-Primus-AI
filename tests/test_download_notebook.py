"""Offline checks for notebook structure, safe defaults and collection helpers."""
import ast
import json
import getpass
import os
import runpy
import sys
from datetime import datetime, timezone
from pathlib import Path
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks/f1_2025_2026_download.ipynb'


def cell_source(cell):
    source = cell['source']
    return ''.join(source) if isinstance(source, list) else source


@pytest.fixture
def notebook_namespace(monkeypatch, tmp_path):
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(requests.sessions.Session, 'send', lambda *args, **kwargs: pytest.fail('Default notebook attempted HTTP'))
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    namespace = {'__name__': 'notebook_offline_test', '_test_output': tmp_path / 'archive'}
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = cell_source(cell)
            if 'configuration' in cell['metadata'].get('tags', []):
                # A user's active run configuration must not make pytest fetch/sign in.
                source += '\nRUN_DOWNLOAD = False\nAUTHENTICATE_LOCALLY = False\nUPGRADE_SDK = False\n'
                # Validation belongs to fixtures, never the user's growing multi-GB archive.
                source += '\nOUTPUT = _test_output\nCACHE = OUTPUT / "fastf1-cache"\nPROGRESS_FILE = OUTPUT / "progress.json"\n'
            exec(compile(source, f"{NOTEBOOK.name}:{cell['id']}", 'exec'), namespace)
    return namespace


def test_notebook_structure_and_python_cells_compile():
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    assert notebook['nbformat'] == 4 and notebook['nbformat_minor'] == 5
    assert 'primusenv' in notebook['metadata']['kernelspec']['display_name'].lower()
    identifiers = [cell['id'] for cell in notebook['cells']]
    assert len(set(identifiers)) == len(identifiers)
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(cell_source(cell))


def test_default_run_is_offline_and_does_not_create_a_dataset(notebook_namespace):
    scope = notebook_namespace
    assert scope['YEARS'] == (2025, 2026)
    assert not scope['RUN_DOWNLOAD'] and not scope['AUTHENTICATE_LOCALLY']
    assert scope['SESSION_LIMIT'] == 1
    assert scope['archive'] is None and scope['http'] is None


def test_session_plan_excludes_other_years_future_and_boundary_sessions(notebook_namespace):
    scope = notebook_namespace
    end = datetime(2026, 10, 24, 5, tzinfo=timezone.utc)
    start_ns = int(datetime(2025, 3, 1, tzinfo=timezone.utc).timestamp()) * 10**9
    rows = [
        dict(event_id='2025:10', session_id='2025:10:R', session_code='R', scheduled_start_utc_ns=start_ns),
        dict(event_id='2025:2', session_id='2025:2:R', session_code='R', scheduled_start_utc_ns=start_ns),
        dict(event_id='2024:1', session_id='2024:1:R', session_code='R', scheduled_start_utc_ns=start_ns),
        dict(event_id='2026:20', session_id='2026:20:R', session_code='R', scheduled_start_utc_ns=int(end.timestamp())*10**9-3600*10**9),
        dict(event_id='2026:21', session_id='2026:21:R', session_code='R', scheduled_start_utc_ns=int(end.timestamp())*10**9+1),
    ]
    selected = scope['choose_sessions'](rows, end, {2025, 2026}, {'R'})
    assert [r['session_id'] for r in selected] == ['2025:2:R', '2025:10:R']
    assert len(scope['choose_sessions'](rows, end, {2025, 2026}, {'R'}, {10})) == 1


def test_progress_roundtrip_and_invalid_schema(notebook_namespace, tmp_path):
    scope = notebook_namespace
    scope['OUTPUT'] = tmp_path
    scope['PROGRESS_FILE'] = tmp_path / 'progress.json'
    assert scope['read_progress']() == {'schema_version': 1, 'sessions': {}}
    progress = {'schema_version': 1, 'sessions': {'2025:1:R': {'status': 'partial', 'version_id': 'example'}}}
    scope['save_progress'](progress)
    assert scope['read_progress']() == progress
    assert not list(tmp_path.glob('*.tmp'))
    scope['PROGRESS_FILE'].write_text('{"schema_version":99,"sessions":{}}')
    with pytest.raises(ValueError, match='checkpoint schema'):
        scope['read_progress']()


def test_session_collection_uses_original_name_and_publishes_coverage(notebook_namespace, monkeypatch, tmp_path):
    scope = notebook_namespace
    scope['CACHE'] = tmp_path / 'cache'
    path = scope['CACHE'] / '2025/weekend/session'
    path.mkdir(parents=True)
    (path / 'timing.ff1pkl').write_bytes(b'trusted-test-source')
    seen = {'sources': [], 'published': []}
    raw = SimpleNamespace(api_path='/static/2025/weekend/session/', load=lambda **kw: None)
    def get_session(name):
        assert name == 'Sprint Shootout'
        return raw
    scope['archived_event'] = lambda *args: SimpleNamespace(get_session=get_session)
    completeness = {key: {'status': 'available'} for key in ('laps','car','positions','weather','race_control')}
    result = {'tables': {'laps': [{'lap_number': 1}]}, 'datasets': [],
              'metadata': {'completeness': completeness, 'observed_end_utc_ns': 1}}
    scope['extract_session'] = lambda *args: result
    scope['add_original_cached_streams'] = lambda *args: None
    monkeypatch.setattr(scope['fastf1'].Cache, 'enable_cache', lambda *args: None)
    monkeypatch.setattr(scope['fastf1'].Cache, 'offline_mode', lambda *args: None)
    def source(*args, **kwargs):
        seen['sources'].append(args)
        return 'source-id'
    def publish(*args):
        seen['published'].append(args)
        return 'version-id'
    archive = SimpleNamespace(source=source, publish_session=publish)
    http = SimpleNamespace(install_fastf1=lambda: nullcontext(), status=lambda: {'routes': []}, last_sdk_deferred=None)
    row = {'event_id': '2025:1', 'session_id':'2025:1:SQ', 'original_provider_name':'Sprint Shootout', 'full_name':'Sprint qualifying'}
    record = scope['download_session'](row, archive, http)
    assert record['status'] == 'complete' and record['version_id'] == 'version-id'
    assert seen['sources'][0][2] == b'trusted-test-source'
    assert len(seen['published']) == 1
    result['metadata']['observed_end_utc_ns'] = int(scope['cutoff'].timestamp()) * 10**9 + 1
    with pytest.raises(ValueError, match='cutoff'):
        scope['download_session'](row, archive, http)
    assert len(seen['sources']) == 2 and len(seen['published']) == 1
    http.last_sdk_deferred = scope['RetryDeferred']('fastf1', 1792828800, 'quota')
    result['metadata']['observed_end_utc_ns'] = 1
    with pytest.raises(scope['RetryDeferred']):
        scope['download_session'](row, archive, http)
    assert len(seen['published']) == 1


def test_authentication_denial_stops_batch_after_one_session(notebook_namespace, tmp_path):
    scope = notebook_namespace
    scope.update(RUN_DOWNLOAD=True, archive=object(), http=object(), OUTPUT=tmp_path,
                 PROGRESS_FILE=tmp_path / 'progress.json', SESSION_LIMIT=None)
    rows = [{'session_id':f'2025:{round_number}:R', 'event_name':'Australian GP', 'full_name':'Race'} for round_number in (1, 2)]
    scope['planned_sessions'] = rows
    calls = []
    def deny(row, *args):
        calls.append(row['session_id'])
        raise scope['ProviderBlocked']('fastf1','timing_archive',401)
    scope['download_session'] = deny
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    batch = next(c for c in notebook['cells'] if 'batch' in c['metadata'].get('tags', []))
    exec(cell_source(batch), scope)
    assert calls == ['2025:1:R']
    assert scope['read_progress']()['sessions']['2025:1:R']['status'] == 'blocked'


@pytest.mark.parametrize('fail', [False, True])
def test_notebook_login_uses_masked_input_and_restores_environment(notebook_namespace, monkeypatch, fail):
    scope = notebook_namespace
    monkeypatch.delenv('F1_USERNAME', raising=False)
    monkeypatch.delenv('F1_PASSWORD', raising=False)
    monkeypatch.setattr('builtins.input', lambda prompt: 'fixture@example.invalid')
    monkeypatch.setattr(getpass, 'getpass', lambda prompt: 'fixture-not-a-real-password')
    original_args = sys.argv[:]
    seen = []
    def fake_run(*args, **kwargs):
        assert os.environ['F1_USERNAME'] == 'fixture@example.invalid'
        assert os.environ['F1_PASSWORD'] == 'fixture-not-a-real-password'
        seen.append(True)
        if fail:
            raise SystemExit(1)
    monkeypatch.setattr(runpy, 'run_module', fake_run)
    if fail:
        with pytest.raises(SystemExit):
            scope['authenticate_in_notebook']()
    else:
        scope['authenticate_in_notebook']()
    assert seen == [True]
    assert 'F1_USERNAME' not in os.environ and 'F1_PASSWORD' not in os.environ
    assert sys.argv == original_args
