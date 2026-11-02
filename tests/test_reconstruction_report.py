import json
from dataclasses import replace
from datetime import timedelta

import pytest

from test_historical_forecast import reconstruction
from data.reconstruction_report import populate
from monitoring.queries import TrackingQueries
from monitoring.store import TrackingStore


def test_saved_reconstruction_metrics_are_separate_reproducible_and_readable(tmp_path):
    request, snapshot, _ = reconstruction()
    outcome = {'event_id':request.event.event_id, 'revision_id':'official-v1', 'status':'official',
               'published_at_utc':snapshot.sources[0].fetched_at_utc.isoformat(),
               'entries':[{'driver_id':'A','actual_position':1,'status':'Finished'},
                          {'driver_id':'B','actual_position':2,'status':'Finished'}]}
    path = populate([snapshot], {request.event.event_id:outcome}, tmp_path)
    assert populate([snapshot], {request.event.event_id:outcome}, tmp_path) == path
    result = json.loads(path.read_text(encoding='utf-8'))
    assert result['status']=='evaluated_reconstruction'
    assert result['models']['grid']['metrics']['position_mae']==1
    query=TrackingQueries(TrackingStore(tmp_path/'tracking.sqlite3'),owner_id='local')
    assert query.summary()['groups']==[]
    groups=query.summary(evidence_kind='historical_reconstruction')['groups']
    assert len(groups)==2 and all(row['n_scored_events']==1 for row in groups)
    assert len(query.forecasts())==2
    assert snapshot.payloads.get('history') is None


def test_official_result_field_cannot_silently_drop_driver(tmp_path):
    request,snapshot,_=reconstruction()
    outcome={'entries':[{'driver_id':'A','actual_position':1,'status':'Finished'}]}
    with pytest.raises(ValueError,match='fields differ'):
        populate([snapshot],{request.event.event_id:outcome},tmp_path)


def test_repeated_event_snapshot_revisions_select_latest_without_double_scoring(tmp_path):
    request,snapshot,_=reconstruction()
    newer=replace(snapshot,snapshot_id='newer-revision',sources=[replace(snapshot.sources[0],
                  fetched_at_utc=snapshot.sources[0].fetched_at_utc+timedelta(days=1))])
    outcome=dict(event_id=request.event.event_id,revision_id='official-v1',status='official',
                 published_at_utc=newer.sources[0].fetched_at_utc.isoformat(),
                 entries=[dict(driver_id='A',actual_position=1,status='Finished'),
                          dict(driver_id='B',actual_position=2,status='Finished')])
    first=populate([snapshot,newer,snapshot],{request.event.event_id:outcome},tmp_path)
    assert populate([newer,snapshot],{request.event.event_id:outcome},tmp_path)==first
    report=json.loads(first.read_text(encoding='utf-8'))
    assert report['cohort_event_ids']==[request.event.event_id]
    assert {row['snapshot_id'] for row in report['lineage']}=={'newer-revision'}
