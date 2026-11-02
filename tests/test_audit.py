from datetime import datetime,timezone
import json
from data.contracts import EventKey,SourceRecord,RunStatus
from data.fastf1_source import LoadedSession
from data.audit import persist_session


def test_audit_serializes_hindsight_only_and_is_content_addressed(tmp_path):
    source=SourceRecord('fastf1','test',datetime.now(timezone.utc))
    session=LoadedSession(EventKey(2025,1),'R',({'driver_id':'A'},),({'driver_id':'A','session_time_s':90,'lap_number':1},),source,RunStatus.DEGRADED)
    path=persist_session(session,directory=tmp_path)
    assert persist_session(session,directory=tmp_path)==path
    body=json.loads(path.read_text())
    assert body['schema_version']=='session-v1'
    assert body['forecast_eligibility']=='excluded_historical_availability_unverified'
    assert body['historical_analysis_status']=='hindsight_only'
    assert body['positions']==[]
    assert body['laps'][0]['session_time_s']==90
