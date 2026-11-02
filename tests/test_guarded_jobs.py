from test_prediction_service import inputs
import pytest
from security.access import Principal,AccessDenied
from security.limits import SQLiteLimiter,LimitRejected
from monitoring.store import TrackingStore
from flows.jobs import guarded_forecast


def test_guarded_job_persists_once_and_replays_without_dispatch(tmp_path,monkeypatch):
    request,snapshot,schedule=inputs()
    principal=Principal('local','ai_user')
    limiter=SQLiteLimiter(tmp_path/'limits.sqlite3');store=TrackingStore(tmp_path/'tracking.sqlite3')
    first=guarded_forecast(principal,limiter,store,request=request,snapshot=snapshot,schedule=schedule,model_id='grid')
    import flows.jobs
    def forbidden(*args,**kwargs):raise AssertionError('duplicate dispatch')
    monkeypatch.setattr(flows.jobs,'generate_forecast',forbidden)
    second=guarded_forecast(principal,limiter,store,request=request,snapshot=snapshot,schedule=schedule,model_id='grid')
    assert second==first
    from monitoring.queries import TrackingQueries
    health=TrackingQueries(store,owner_id='local',role='operator').operational_summary()
    assert health['latency_event_count']==1


def test_unauthorized_does_not_call_inference(tmp_path,monkeypatch):
    import flows.jobs
    monkeypatch.setattr(flows.jobs,'generate_forecast',lambda *a,**k:pytest.fail('unauthorized inference'))
    request,snapshot,schedule=inputs()
    with pytest.raises(AccessDenied):
        guarded_forecast(Principal('v','viewer'),SQLiteLimiter(tmp_path/'limits.sqlite3'),TrackingStore(tmp_path/'tracking.sqlite3'),request=request,snapshot=snapshot,schedule=schedule,model_id='grid')


def test_synchronous_busy_request_does_not_strand_a_queue_lease(tmp_path):
    from security.limits import LimitPolicy
    limiter=SQLiteLimiter(tmp_path/'limits.sqlite3',policy=LimitPolicy(global_active=1))
    blocker=Principal('other','ai_user')
    limiter.reserve(blocker,'blocker','payload')
    request,snapshot,schedule=inputs()
    with pytest.raises(LimitRejected,match='busy'):
        guarded_forecast(Principal('local','ai_user'),limiter,TrackingStore(tmp_path/'tracking.sqlite3'),request=request,snapshot=snapshot,schedule=schedule,model_id='grid',request_id='synchronous-job')
    later=limiter.reserve(Principal('local','ai_user'),'new-job','new-payload')
    assert later.status=='queued'


def test_saved_history_job_identity_matches_inference_and_replays(tmp_path):
    from test_prediction_service import history_inputs
    from flows.prediction_service import forecast_run_id
    request,snapshot,schedule=history_inputs()
    principal=Principal('local','ai_user')
    limiter=SQLiteLimiter(tmp_path/'limits.sqlite3');store=TrackingStore(tmp_path/'tracking.sqlite3')
    result=guarded_forecast(principal,limiter,store,request=request,snapshot=snapshot,schedule=schedule)
    assert [(r['driver_id'],r['prediction_score']) for r in result['ranking']]==[('B',1.0),('A',2.0)]
    assert result['run_id']==forecast_run_id(request,snapshot,history=snapshot.payloads['history'])
    assert guarded_forecast(principal,limiter,store,request=request,snapshot=snapshot,schedule=schedule)==result
