"""2026 collection delegates the inclusive Nov 2/current-time bounds to the runner."""
from datetime import datetime, timezone
import json
from pathlib import Path


def test_2026_collection_forwards_authorized_november2_scope_without_auth(monkeypatch, tmp_path):
    from data.archive import notebook_f1
    source_path = Path(__file__).resolve().parents[1] / 'notebooks/collection_2020_2026/f1_2026.ipynb'
    book = json.loads(source_path.read_text(encoding='utf8'))
    cell = next(c for c in book['cells'] if 'collection' in c.get('metadata', {}).get('tags', []))
    calls = []
    monkeypatch.setattr(notebook_f1, 'run_f1_year', lambda *args, **kwargs: calls.append((args, kwargs)) or {'sessions': {}})
    exec(compile(''.join(cell['source']), '2026-collection', 'exec'),
         dict(ROOT=tmp_path, YEAR=2026, THROUGH_DATE='2026-11-02', ROUND_NUMBERS=None,
              DRY_RUN=False, RUN_DOWNLOAD=True, SESSION_LIMIT=None, RETRY_PARTIAL=False,
              AUTHENTICATE_LOCALLY=False))
    assert calls == [((tmp_path, 2026), dict(through='2026-11-02', round_numbers=None,
                        session_limit=None, retry_partial=False, authenticate=False))]


def test_2026_runner_cutoff_includes_november1_but_excludes_later_and_unfinished_sessions():
    from data.archive.notebook_f1 import choose_sessions
    def row(number, start):
        return dict(event_id=f'2026:{number}', session_id=f'2026:{number}:R', session_code='R',
                    scheduled_start_utc_ns=int(datetime.fromisoformat(start).timestamp()*1e9))
    rows = [row(18,'2026-10-25T19:00:00+00:00'), row(19,'2026-11-01T20:00:00+00:00'),
            row(20,'2026-11-03T20:00:00+00:00')]
    after = datetime(2026,11,3,6,tzinfo=timezone.utc)  # Nov 2 local end, inclusive.
    assert [r['session_id'] for r in choose_sessions(rows,after,2026)] == ['2026:18:R','2026:19:R']
    ongoing = datetime(2026,11,1,21,tzinfo=timezone.utc)
    assert [r['session_id'] for r in choose_sessions(rows,ongoing,2026)] == ['2026:18:R']


def test_authorized_future_cutoff_is_still_bounded_by_current_time(monkeypatch):
    from data.archive import notebook_f1
    now = datetime(2026,11,1,5,tzinfo=timezone.utc)
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now
    monkeypatch.setattr(notebook_f1,'datetime',Clock)
    assert notebook_f1.end_for('2026-11-02') == now
