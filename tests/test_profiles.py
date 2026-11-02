from copy import deepcopy
from datetime import datetime, timezone

import pytest

from data.profiles import build_profiles


T = datetime(2025, 7, 5, tzinfo=timezone.utc)


def row(event='2025:1', driver='NOR', team='mclaren', position=2, **extra):
    return dict(event_id=event, driver_id=driver, team_id=team, actual_position=position,
                status='FINISHED', event_time_utc='2025-03-01T00:00:00Z',
                outcome_available_at_utc='2025-03-02T00:00:00Z', circuit_id='albert-park',
                source_ids=['official-results-1'], **extra)


def test_profiles_cutoff_lineage_counts_and_no_invented_traits():
    profiles = build_profiles([row(), row('2025:2', position=4)], T)
    driver = profiles['drivers'][0]
    assert driver['recent_form']['mean_position'] == 3
    assert driver['recent_form']['sample_count'] == 2
    assert driver['reliability']['classified_dnf_count'] == 0
    assert driver['source_ids'] == ['official-results-1']
    assert driver['circuit_samples'][0]['sample_count'] == 2
    assert driver['affiliation']['team_id'] == 'mclaren'
    assert 'personality' not in driver and 'feature_score' not in driver
    assert profiles['teams'][0]['recent_form']['sample_count'] == 2


def test_future_and_unknown_publication_cannot_change_snapshot():
    original = build_profiles([row()], T)
    future = row('2025:9'); future['event_time_utc'] = '2025-08-01T00:00:00Z'
    late = row('2025:2'); late['outcome_available_at_utc'] = '2025-08-02T00:00:00Z'
    unknown = row('2025:3'); unknown.pop('outcome_available_at_utc')
    assert build_profiles([future, late, unknown, row()], T) == original


def test_team_switch_keeps_dated_affiliations_and_snapshot_identity():
    old = row(team='old-team')
    new = row('2025:2', team='new-team'); new['event_time_utc'] = '2025-05-01T00:00:00Z'
    new['outcome_available_at_utc'] = '2025-05-02T00:00:00Z'
    profiles = build_profiles([old, new], T)
    driver = profiles['drivers'][0]
    assert driver['affiliation']['team_id'] == 'new-team'
    assert [a['team_id'] for a in driver['affiliation_history']] == ['old-team', 'new-team']
    assert build_profiles([new, old], T) == profiles
    changed = deepcopy(new); changed['actual_position'] = 9
    assert build_profiles([old, changed], T)['snapshot_id'] != profiles['snapshot_id']


def test_dnf_is_observed_status_not_mechanical_or_driver_blame():
    dnf = row(position=17); dnf['status'] = 'DNF'
    profiles = build_profiles([dnf], T)
    assert profiles['drivers'][0]['reliability']['classified_dnf_count'] == 1
    assert 'cause' not in profiles['drivers'][0]['reliability']


def test_duplicates_fail_instead_of_double_counting_and_empty_is_unknown():
    with pytest.raises(ValueError, match='duplicate'):
        build_profiles([row(), row()], T)
    assert build_profiles([], T)['drivers'] == []
    with pytest.raises(ValueError):
        build_profiles([row()], datetime(2025, 7, 5))


def test_explicitly_pending_outcome_not_profile_evidence():
    pending = row(outcome_status='pending')
    assert build_profiles([pending], T)['drivers'] == []
