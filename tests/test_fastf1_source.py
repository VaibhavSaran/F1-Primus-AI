from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from data.contracts import EventKey, RunStatus
from data.fastf1_source import coverage_report, load_session
from ml.features import clean_laps

UTC = timezone.utc
EVENT = EventKey(2025, 1, 'albert_park')
FETCH = datetime(2026, 10, 23, tzinfo=UTC)
ZERO = datetime(2025, 3, 16, 3, tzinfo=UTC)


def fixture_session():
    result = pd.DataFrame([
        {'DriverNumber': '4', 'DriverId': 'norris', 'Abbreviation': 'NOR', 'TeamName': 'McLaren',
         'Position': 1.0, 'ClassifiedPosition': '1', 'Status': 'Finished', 'GridPosition': 1.0},
        {'DriverNumber': '99', 'DriverId': 'rookie', 'Abbreviation': 'NEW', 'TeamName': 'New Team',
         'Position': 2.0, 'ClassifiedPosition': '2', 'Status': '+1 Lap', 'GridPosition': 20.0},
    ])
    laps = []
    for number, driver in [('4', 'NOR'), ('99', 'NEW')]:
        for lap in range(1, 5):
            laps.append(dict(DriverNumber=number, Driver=driver, LapNumber=lap,
                             LapTime=pd.Timedelta(seconds=90 + lap), Time=pd.Timedelta(seconds=lap * 92),
                             LapStartTime=pd.Timedelta(seconds=(lap - 1) * 92),
                             IsAccurate=True, Deleted=False, PitInTime=pd.NaT, PitOutTime=pd.NaT,
                             TrackStatus='1', Compound='MEDIUM', TyreLife=lap, Stint=1))
    return SimpleNamespace(results=result, laps=pd.DataFrame(laps), t0_date=pd.Timestamp(ZERO),
                           session_status=pd.DataFrame({'Time': [pd.Timedelta(0), pd.Timedelta(seconds=400)],
                                                        'Status': ['Started', 'Finished']}),
                           name='Race', event={'EventName': 'Australian Grand Prix'}, api_path='/static/test/')


def load(raw=None, **kwargs):
    return load_session(EVENT, 'R', loader=lambda *args: raw or fixture_session(), fetched_at_utc=FETCH, **kwargs)


def test_normalizes_explicit_entries_lap_quality_and_clock_without_fake_publication():
    loaded = load()
    assert loaded.status == RunStatus.DEGRADED
    assert [entry['driver_id'] for entry in loaded.entries] == ['norris', 'rookie']
    assert loaded.entries[1]['status'] == '+1 Lap'
    assert loaded.entries[1]['position'] == 2
    assert len(clean_laps(loaded.laps)) == 6
    assert loaded.laps[1]['driver_id'] == 'norris'
    assert loaded.laps[1]['lap_end_utc'] == ZERO + timedelta(seconds=184)
    assert loaded.laps[1]['session_time_s'] == 184
    assert loaded.session_start_utc == ZERO
    assert loaded.session_end_utc == ZERO + timedelta(seconds=400)
    assert loaded.source.fetched_at_utc == FETCH
    assert loaded.source.published_at_utc is None
    assert loaded.source.available_at_utc is None
    assert not loaded.source.available_by(ZERO + timedelta(days=1))
    assert 'historical_availability_unverified' in loaded.quality_flags


def test_unknown_deletion_is_excluded_and_bad_laps_logged_in_coverage():
    raw = fixture_session()
    raw.laps['Deleted'] = raw.laps['Deleted'].astype(object)
    raw.laps.loc[1, 'Deleted'] = None
    raw.laps.loc[2, 'LapTime'] = pd.NaT
    raw.laps.loc[3, 'TrackStatus'] = '14'
    loaded = load(raw)
    assert loaded.laps[1]['deleted'] is None
    assert loaded.laps[1]['accurate'] is False
    assert loaded.laps[1]['provider_is_accurate'] is True
    report = coverage_report([loaded])
    assert report['sessions_loaded'] == 1
    assert report['total_laps'] == 8
    assert report['eligible_clean_laps'] == 3
    assert report['laps_with_deletion_evidence'] == 7
    assert report['strict_historical_eligible_sessions'] == 0
    assert report['sessions'][0]['historical_analysis_status'] == 'hindsight_only'
    assert report['sessions'][0]['forecast_eligibility'] == 'excluded_historical_availability_unverified'
    assert 'archived_weather_issue_times' in report['unavailable_features']


def test_missing_identity_never_aliases_to_another_driver_and_duplicate_laps_fail_closed():
    raw = fixture_session()
    raw.results.loc[1, 'DriverId'] = None
    raw.laps = pd.concat([raw.laps, raw.laps.iloc[[2]]], ignore_index=True)
    loaded = load(raw)
    assert loaded.entries[1]['driver_id'] == 'fastf1:number:99'
    assert len(loaded.laps) == 8
    assert 'duplicate_laps_removed' in loaded.quality_flags


def test_provider_failure_and_no_laps_are_unavailable_without_exception_details():
    def broken(*args):
        raise RuntimeError('private provider message')
    failed = load_session(EVENT, 'R', loader=broken, fetched_at_utc=FETCH)
    assert failed.status == RunStatus.ERROR
    assert failed.errors == ('provider_load_failed',)
    assert 'private' not in repr(failed)
    raw = fixture_session()
    raw.laps = pd.DataFrame()
    assert load(raw).status == RunStatus.INSUFFICIENT_DATA


@pytest.mark.parametrize('session,timeout', [('bad', 45), ('R', 0), ('R', float('nan'))])
def test_invalid_requests_fail_before_provider_call(session, timeout):
    with pytest.raises(ValueError):
        load_session(EVENT, session, timeout_s=timeout)


def test_default_loader_is_bounded_cache_only_and_reports_cache_miss(tmp_path):
    loaded = load_session(EVENT, 'R', cache_path=tmp_path / 'missing', fetched_at_utc=FETCH)
    assert loaded.status == RunStatus.INSUFFICIENT_DATA
    assert loaded.errors == ('cache_unavailable',)


def test_process_timeout_has_honest_error(monkeypatch, tmp_path):
    import subprocess
    import data.fastf1_source as adapter
    def timeout(*args, **kwargs):
        assert kwargs['timeout'] == 0.1
        raise subprocess.TimeoutExpired('worker', 0.1)
    monkeypatch.setattr(adapter.subprocess, 'run', timeout)
    loaded = load_session(EVENT, 'R', cache_path=tmp_path, timeout_s=0.1, fetched_at_utc=FETCH,
                          include_telemetry=True)
    assert loaded.status == RunStatus.ERROR
    assert loaded.errors == ('provider_timeout',)


def test_absent_telemetry_origin_does_not_invent_absolute_lap_times():
    class DataNotLoadedError(Exception):
        pass
    class NoTelemetry:
        def __getattr__(self, name):
            if name == 't0_date':
                raise DataNotLoadedError('telemetry not loaded')
            return getattr(fixture_session(), name)
    loaded = load(NoTelemetry())
    assert loaded.laps
    assert loaded.laps[0]['lap_end_utc'] is None
    assert loaded.session_start_utc is None
    assert 'absolute_lap_times_unavailable' in loaded.quality_flags
