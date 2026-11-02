import json

from data.archive.status import summarize, markdown, save_status, collect_status


def audit(sessions):
    return dict(year=2026, through_date='2026-10-25', sessions=sessions,
                integrity=dict(sqlite='ok', foreign_key_errors=0, file_errors=[]))


def row(action, version='v'):
    return dict(action=action, version_id=version, date='2026-10-25')


def test_summary_counts_saved_partial_and_cancelled_separately():
    result = summarize(audit([row('complete'), row('review_partial'), row('download', None), row('not_held', None)]), {})
    assert (result['saved'], result['expected'], result['missing'], result['partial'], result['cancelled']) == (2, 3, 1, 1, 1)
    assert not result['coverage_complete']


def test_complete_coverage_retains_quality_warning_and_missing_calendar_is_unknown():
    result = summarize(audit([row('complete'), row('review_partial')]), {})
    assert result['coverage_complete'] and result['partial'] == 1
    unknown = summarize(audit([]), {})
    assert unknown['expected'] is None and unknown['missing'] is None
    assert unknown['state'] == 'Calendar unavailable'
    assert not unknown['coverage_complete']


def test_corruption_or_identity_repair_cannot_claim_complete():
    for action in ('repair_local', 'reprocess_identity'):
        assert not summarize(audit([row(action)]), {})['coverage_complete']
    data = audit([row('complete')])
    data['integrity']['foreign_key_errors'] = 1
    assert not summarize(data, {})['coverage_complete']


def test_cutoff_passed_only_to_2026_and_weather_uses_same_event_scope(monkeypatch, tmp_path):
    import data.archive.status as status
    calls, weather_calls = [], []
    def f1(directory, year, **kwargs):
        calls.append((year, kwargs['through_date']))
        data = audit([dict(row('complete'), session_id=f'{year}:18:R')])
        data['year'] = year
        return data
    def weather(directory, ids, **kwargs):
        weather_calls.append(ids)
        return dict(expected_events=1, complete_events=1)
    monkeypatch.setattr(status, 'audit_year', f1)
    monkeypatch.setattr(status, 'audit_weather', weather)
    result = collect_status(tmp_path, emit=lambda _: None)
    assert calls == [(2020, None), (2021, None), (2022, None), (2023, None),
                     (2024, None), (2025, None), (2026, '2026-11-02')]
    assert weather_calls[-1] == ['2026:18']
    assert len(result['years']) == 7


def test_saved_ledger_has_current_files_and_immutable_history(tmp_path):
    f1 = audit([dict(row('download', None), event='United States GP', session='Race',
                    session_id='2026:18:R', checkpoint_status='deferred', detail='Request budget')])
    weather = dict(expected_events=1, complete_events=0,
                   events=[dict(event_id='2026:18', status='pending', detail='Publication lag')])
    report = dict(checked_at_utc='2026-10-27T22:00:00+00:00', through_2026='2026-10-25', file_hashes_checked=False,
                  years=[dict(summary=summarize(f1, weather), f1=f1, weather=weather)])
    save_status(report, tmp_path)
    save_status(report, tmp_path)
    assert len(list((tmp_path/'history').glob('*.json'))) == 2
    assert json.loads((tmp_path/'current.json').read_text()) == report
    text = (tmp_path/'current.md').read_text(encoding='utf-8')
    assert '2026-10-25' in text and 'United States GP: Race' in text
    assert 'Publication lag' in text and 'Request budget' in text
    assert 'not checked' in text
