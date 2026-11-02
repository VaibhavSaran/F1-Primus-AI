from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from data.bootstrap import normalized_positions, retrospective_snapshot, write_calendar
from data.catalog import load_calendar, event_label, session_label
from data.contracts import EventKey, SourceRecord, RunStatus
from data.fastf1_source import LoadedSession
from data.sessions import EventInfo, SessionRecord

UTC = timezone.utc
RACE = datetime(2025, 3, 16, 4, tzinfo=UTC)
EVENT = EventKey(2025, 1)


def test_calendar_keeps_real_names_dates_and_session_labels(tmp_path):
    row = dict(RoundNumber=1, EventName='Australian Grand Prix', Location='Melbourne',
               Session1='Practice 1', Session1DateUtc=RACE-timedelta(days=2),
               Session5='Race', Session5DateUtc=RACE)
    path = tmp_path / 'calendar.json'
    events = write_calendar([2025], loader=lambda *a, **k: pd.DataFrame([row]), path=path)
    assert load_calendar(path) == events
    assert event_label(events[0]) == 'Australian Grand Prix · 2025-03-16'
    assert session_label('R') == 'Grand Prix Race'
    assert events[0]['sessions'][0]['session_id'] == 'FP1'
    assert load_calendar(tmp_path / 'missing') == []


def test_positions_are_real_sampled_ontrack_samples_only():
    raw = SimpleNamespace(pos_data={'4': pd.DataFrame([
        dict(SessionTime=pd.Timedelta(seconds=t), X=x, Y=20, Status=status)
        for t, x, status in [(0, 1, 'OnTrack'), (1, 2, 'OnTrack'), (2, 3, 'OffTrack'), (4, 4, 'OnTrack')]])})
    rows = normalized_positions(raw, [{'driver_id': 'norris', 'driver_number': '4'}])
    assert [(r['session_time_s'], r['x']) for r in rows] == [(0, 1), (4, 4)]


def test_calendar_uses_race_local_date_for_night_races(tmp_path):
    row=dict(RoundNumber=22, EventName='Las Vegas Grand Prix', Location='Las Vegas',
             Session5='Race', Session5DateUtc=datetime(2025,11,23,4,tzinfo=UTC),
             Session5Date=datetime(2025,11,22,20,tzinfo=timezone(timedelta(hours=-8))))
    events=write_calendar([2025],loader=lambda *a,**k:pd.DataFrame([row]),path=tmp_path/'calendar.json')
    assert events[0]['race_date']=='2025-11-22'


def test_reconstruction_excludes_final_race_labels_and_keeps_fetch_provenance():
    fetched = RACE + timedelta(days=500)
    source = SourceRecord('jolpica', 'q', fetched)
    entry = dict(driver_id='norris', driver_number='4', abbreviation='NOR', team_id='mclaren',
                 position=2, status='Q result')
    q = LoadedSession(EVENT, 'Q', (entry,), (), source, RunStatus.DEGRADED)
    race = LoadedSession(EVENT, 'R', (dict(entry, position=1),),
                         ({'actual_position': 1},), source, RunStatus.DEGRADED)
    schedule = EventInfo(EVENT, 'Australian Grand Prix', 'Melbourne',
                         (SessionRecord(EVENT, 'Q', 'Qualifying', RACE-timedelta(days=1)),
                          SessionRecord(EVENT, 'R', 'Race', RACE)))
    snapshot = retrospective_snapshot(schedule, [q, race])
    assert snapshot.payloads['historical_reconstruction'] is True
    assert snapshot.sources == [source]
    assert source.fetched_at_utc == fetched
    assert not source.available_by(snapshot.as_of_utc)
    assert snapshot.payloads['driver_sessions'][0].values['qualifying_position'] == 2
    assert 'position' not in snapshot.payloads['entries'][0]
    assert all(record.session_id != 'R' for record in snapshot.payloads['driver_sessions'])
    assert snapshot.payloads['completion_assumptions']['Q']
    assert snapshot.payloads['schedule'].sessions[0].ends_at_utc == snapshot.as_of_utc


def test_missing_qualifying_cannot_make_snapshot():
    with pytest.raises(ValueError, match='qualifying'):
        retrospective_snapshot(EventInfo(EVENT, '', '', ()), [])


def test_historical_race_number_takes_precedence_over_current_driver_number(monkeypatch):
    import requests
    from data.bootstrap import jolpica_classification
    class Response:
        def raise_for_status(self): pass
        def json(self):
            return {'MRData': {'RaceTable': {'Races': [{'QualifyingResults': [
                {'number':'4','position':'1','Driver':{'driverId':'norris','permanentNumber':'1','code':'NOR'},
                 'Constructor':{'constructorId':'mclaren'}}]}]}}}
    monkeypatch.setattr(requests,'get',lambda *args,**kwargs:Response())
    session=jolpica_classification(EVENT,'Q')
    assert session.entries[0]['driver_number']=='4'
