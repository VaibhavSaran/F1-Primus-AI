from scripts.update_data_tracker import public_markdown


def test_public_tracker_projects_counts_without_private_diagnostics():
    report = {
        'checked_at_utc': '2026-11-02T12:00:00+00:00',
        'through_2026': '2026-11-02',
        'file_hashes_checked': False,
        'years': [{
            'summary': dict(year=2026, saved=94, expected=95, complete_sessions=84,
                            partial=10, reprocess=0, repair=0, missing=1, cancelled=0,
                            weather_complete=18, weather_expected=19),
            'f1': {'private_error': 'SECRET_TOKEN and C:/private/archive'},
            'weather': {'private_id': 'owner-private-run'},
        }],
    }
    result = public_markdown(report)
    assert '| 2026 | 94/95 | 84 | 10 | 0 | 1 | 0 | 18/19 |' in result
    assert '**not checked**' in result
    assert 'SECRET_TOKEN' not in result
    assert 'C:/private' not in result
    assert 'owner-private-run' not in result
    report['file_hashes_checked'] = True
    assert 'checked during this refresh.' in public_markdown(report)
