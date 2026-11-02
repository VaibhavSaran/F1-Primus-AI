"""The normal 2020 audit resumes only eligible missing/repair sessions.

The former optional recovery cell has been replaced by the Run-All audit and
collection cells. User notebook configuration and saved outputs are untouched.
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('retry_partial', [False, True])
@pytest.mark.parametrize('dry_run,run_download', [(False, True), (True, True), (False, False)])
def test_targeted_resume_excludes_completed_cancelled_corrupt_and_out_of_scope(
        retry_partial, dry_run, run_download, monkeypatch, tmp_path):
    from data.archive import notebook_audit, notebook_f1
    book = json.loads((ROOT / 'notebooks/collection_2020_2026/f1_2020.ipynb').read_text(encoding='utf8'))
    tagged = {tag: c for c in book['cells'] for tag in c.get('metadata', {}).get('tags', [])}
    assert 'audit-2020' in tagged and 'collection' in tagged
    rows = [dict(session_id=f'2020:1:{code}', round_number=1, event='Fixture GP', session=code,
                 checkpoint_status='saved', laps=1, detail='', action=action)
            for code, action in [('R', 'download'), ('Q', 'review_partial'), ('FP1', 'reprocess_identity'),
                                 ('FP2', 'complete'), ('FP3', 'not_held'), ('SQ', 'repair_local')]]
    rows.append(dict(rows[0], session_id='2020:2:R', round_number=2))
    audits = []
    def audit(*args, **kwargs):
        audits.append(kwargs)
        return {'sessions': rows, 'counts': {}, 'integrity': {'sqlite': 'ok', 'foreign_key_errors': 0}}
    monkeypatch.setattr(notebook_audit, 'audit_year', audit)
    calls = []
    monkeypatch.setattr(notebook_f1, 'run_f1_year', lambda *args, **kwargs: calls.append(kwargs) or {'sessions': {}})
    output = tmp_path / '.local/f1_2025_2026'; output.mkdir(parents=True); (output / 'f1.sqlite3').touch()
    scope = dict(ROOT=tmp_path, YEAR=2020, THROUGH_DATE='2026-11-02', ROUND_NUMBERS={1},
                 DRY_RUN=dry_run, RUN_DOWNLOAD=run_download, RETRY_PARTIAL=retry_partial,
                 SESSION_LIMIT=None, VERIFY_EXISTING_FILES=False, AUTHENTICATE_LOCALLY=False,
                 display=lambda *args: None)
    import pandas as pd
    scope['pd'] = pd
    for tag in ('audit-2020', 'collection'):
        exec(compile(''.join(tagged[tag]['source']), '2020-targeted-resume', 'exec'), scope)
    expected = {'2020:1:R', '2020:1:FP1'} | ({'2020:1:Q'} if retry_partial else set())
    assert scope['REMAINING_SESSION_IDS'] == expected
    assert scope['REPROCESS_SESSION_IDS'] == {'2020:1:FP1'}
    assert audits == [{'verify_files': False}]
    assert len(calls) == int(run_download and not dry_run)
    if calls:
        assert calls[0]['session_ids'] == expected
        assert calls[0]['retry_partial'] is retry_partial
        assert calls[0]['authenticate'] is False
        assert calls[0]['reprocess_session_ids'] == {'2020:1:FP1'}
        assert calls[0]['through'] == '2026-11-02'


def test_targeted_resume_refuses_corrupt_archive(monkeypatch, tmp_path):
    from data.archive import notebook_audit, notebook_f1
    book = json.loads((ROOT / 'notebooks/collection_2020_2026/f1_2020.ipynb').read_text(encoding='utf8'))
    audit_cell = next(c for c in book['cells'] if 'audit-2020' in c.get('metadata', {}).get('tags', []))
    monkeypatch.setattr(notebook_audit, 'audit_year', lambda *args, **kwargs: {
        'sessions': [], 'counts': {}, 'integrity': {'sqlite': 'corrupt', 'foreign_key_errors': 0}})
    def forbidden(*args, **kwargs):
        pytest.fail('Integrity failure must not start collection')
    monkeypatch.setattr(notebook_f1, 'run_f1_year', forbidden)
    output = tmp_path / '.local/f1_2025_2026'; output.mkdir(parents=True); (output / 'f1.sqlite3').touch()
    with pytest.raises(RuntimeError, match='integrity failed'):
        exec(compile(''.join(audit_cell['source']), '2020-corrupt-audit', 'exec'),
             dict(ROOT=tmp_path, YEAR=2020, VERIFY_EXISTING_FILES=False))
