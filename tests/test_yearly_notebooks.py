"""Run-All contracts under isolated configuration; saved user choices may differ."""
import ast
import importlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'notebooks/collection_2020_2026'
NAMES = [f'{kind}_{year}.ipynb' for kind in ('f1', 'weather') for year in range(2020, 2027)]


def canonical_configuration(scope, *, dry_run=False):
    # Configuration cells and outputs are user-owned. Test dispatch using an
    # explicit safe scope instead of assuming their last saved options.
    scope.update(THROUGH_DATE='2026-11-02', ROUND_NUMBERS=None, RUN_DOWNLOAD=True,
                 DRY_RUN=dry_run, SESSION_LIMIT=None, EVENT_LIMIT=None,
                 AUTHENTICATE_LOCALLY=False, RETRY_PARTIAL=False,
                 RETRY_FAILED=True, VERIFY_EXISTING_FILES=False)


def test_every_year_has_separate_f1_and_weather_notebooks():
    assert DIRECTORY.is_dir()
    for name in NAMES + ['00_setup.ipynb', '99_status_validate.ipynb']:
        assert (DIRECTORY / name).is_file(), name


@pytest.mark.parametrize('name', NAMES)
def test_year_configuration_and_offline_run_all(name, monkeypatch, tmp_path):
    path = DIRECTORY / name
    notebook = json.loads(path.read_text(encoding='utf8'))
    assert notebook['nbformat'] == 4
    assert notebook['metadata']['kernelspec']['name'] in {'primusenv', 'python3'}
    # VS Code may serialize its selected Python interpreter as python3; runtime checks the exact executable.
    assert 'primusenv' in notebook['metadata']['kernelspec']['display_name'].lower()
    assert len({cell['id'] for cell in notebook['cells']}) == len(notebook['cells'])
    calls = []
    for kind in ('f1', 'weather'):
        module = importlib.import_module('data.archive.notebook_' + kind)
        monkeypatch.setattr(module, f'run_{kind}_year', lambda *args, **kwargs: calls.append(args))
    import socket
    def forbidden(*args, **kwargs):
        pytest.fail('Offline preview attempted network')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.chdir(ROOT)
    scope = {'__name__': 'notebook_offline'}
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        assert cell['execution_count'] is None or isinstance(cell['execution_count'], int)
        assert isinstance(cell['outputs'], list)  # Users may save notebook execution outputs.
        source = ''.join(cell['source'])
        ast.parse(source)
        exec(compile(source, str(path), 'exec'), scope)
        if 'configuration' in cell['metadata'].get('tags', []):
            canonical_configuration(scope, dry_run=True)
            scope['ROOT'] = tmp_path  # The preview never inspects the live archive.
    assert scope['YEAR'] == int(name[-10:-6])
    assert scope['THROUGH_DATE'] == '2026-11-02'
    assert scope['RUN_DOWNLOAD'] is True
    assert scope['ROUND_NUMBERS'] is None
    assert not calls


def test_setup_and_status_cells_compile_with_user_saved_outputs_allowed():
    for name in ('00_setup.ipynb', '99_status_validate.ipynb'):
        notebook = json.loads((DIRECTORY / name).read_text(encoding='utf8'))
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                ast.parse(''.join(cell['source']))
                assert isinstance(cell['outputs'], list)


@pytest.mark.parametrize('name', NAMES)
def test_run_all_canonical_scope_dispatches_correct_year_to_shared_runner(name, monkeypatch):
    kind, year = name.removesuffix('.ipynb').split('_')
    module = importlib.import_module('data.archive.notebook_' + kind)
    calls = []
    def runner(*args, **kwargs):
        calls.append((args, kwargs))
        return {'sessions': {}} if kind == 'f1' else {'counts': {}}
    monkeypatch.setattr(module, f'run_{kind}_year', runner)
    if kind == 'f1':
        from data.archive import notebook_audit
        monkeypatch.setattr(notebook_audit, 'audit_year', lambda *args, **kwargs: {
            'integrity': {'sqlite': 'ok', 'foreign_key_errors': 0}, 'counts': {'download': 1},
            'sessions': [{'session_id': '2020:1:R', 'round_number': 1, 'event': 'Offline GP',
                          'session': 'Race', 'action': 'download', 'checkpoint_status': 'not_attempted',
                          'laps': 0, 'detail': 'No publication'}]})
    monkeypatch.chdir(ROOT)
    scope = {'__name__': 'notebook_dispatch_test'}
    notebook = json.loads((DIRECTORY / name).read_text(encoding='utf8'))
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            exec(compile(''.join(cell['source']), name, 'exec'), scope)
            if 'configuration' in cell.get('metadata', {}).get('tags', []):
                canonical_configuration(scope)
    assert len(calls) == 1
    args, options = calls[0]
    assert args == (ROOT, int(year))
    assert options['through'] == '2026-11-02'
    assert options['round_numbers'] is None
    if kind == 'f1':
        assert options['session_limit'] is None and not options['authenticate']
    else:
        assert options['event_limit'] is None and options['retry_failed'] is True


def test_2020_existing_archive_without_year_calendar_starts_planning(monkeypatch):
    from data.archive import notebook_audit, notebook_f1
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(notebook_audit, 'audit_year', lambda *args, **kwargs: {
        'integrity': {'sqlite': 'ok', 'foreign_key_errors': 0}, 'counts': {}, 'sessions': []})
    calls = []
    monkeypatch.setattr(notebook_f1, 'run_f1_year', lambda *args, **kwargs: calls.append(kwargs) or {'sessions': {}})
    scope = {'__name__': 'notebook_empty_year_test'}
    book = json.loads((DIRECTORY / 'f1_2020.ipynb').read_text(encoding='utf8'))
    for cell in book['cells']:
        if cell['cell_type'] == 'code':
            exec(compile(''.join(cell['source']), 'f1_2020.ipynb', 'exec'), scope)
            if 'configuration' in cell.get('metadata', {}).get('tags', []):
                canonical_configuration(scope)
    assert len(calls) == 1 and calls[0]['session_ids'] is None
