from dataclasses import replace
from datetime import datetime,timedelta,timezone
import pytest
from data.contracts import DataSnapshot, DriverSessionRecord, EventKey, PredictionRequest, SourceRecord
from data.sessions import EventInfo, SessionRecord
from flows.prediction_service import generate_forecast

T=datetime(2025,7,5,18,tzinfo=timezone.utc)
E=EventKey(2025,12,'silverstone')
S=SourceRecord('fixture','entries',T-timedelta(hours=3))


def inputs():
    request=PredictionRequest(E,'post_qualifying',T)
    snapshot=DataSnapshot('verified-snapshot',E,T,{'mode':'post_qualifying','entries':[{'driver_id':'A'},{'driver_id':'B'}], 'entries_source':S,
        'driver_sessions':[DriverSessionRecord(E,'GRID',d,T-timedelta(hours=1),{'grid_position':g},S) for d,g in [('A',2),('B',1)]]},[S])
    schedule=EventInfo(E,'British Grand Prix','Silverstone',(SessionRecord(E,'Q','Qualifying',T-timedelta(hours=3),T-timedelta(hours=2)), SessionRecord(E,'R','Race',T+timedelta(days=1))), S)
    return request,snapshot,schedule


def history_inputs():
    request,snapshot,schedule=inputs()
    history=[]
    for event_id,at,positions in [('2025:11',T-timedelta(days=7),[('A',2),('B',1)]),
                                   ('2025:13',T+timedelta(days=7),[('A',1),('B',2)])]:
        history.extend(dict(event_id=event_id,driver_id=driver,actual_position=position,
                            event_time_utc=at,outcome_available_at_utc=at+timedelta(hours=2),
                            outcome_status='completed',status='FINISHED',source_id='past-results')
                       for driver,position in positions)
    return request,replace(snapshot,payloads=dict(snapshot.payloads,history=history,schedule=schedule)),schedule


def test_grid_forecast_is_stable_and_ordinal_with_actual_progress():
    request,snapshot,schedule=inputs(); events=[]
    a=generate_forecast(request,snapshot,model_id='grid',schedule=schedule,progress=events.append)
    b=generate_forecast(request,snapshot,model_id='grid',schedule=schedule)
    assert a['run_id']==b['run_id']
    assert [r['driver_id'] for r in a['ranking']]==['B','A']
    assert a['ranking'][0]['score_units']=='ordinal_position'
    assert 'winner_probability' not in a['ranking'][0]
    assert a['provenance']['snapshot_id']=='verified-snapshot'
    assert [e.stage for e in events]==['validate','features','predict','persist']
    assert a['metrics']=={}


def test_entry_provenance_and_schedule_are_required():
    request,snapshot,schedule=inputs()
    payload=dict(snapshot.payloads);payload.pop('entries_source')
    with pytest.raises(ValueError,match='entry'):
        generate_forecast(request,replace(snapshot,payloads=payload),schedule=schedule)
    with pytest.raises(ValueError,match='schedule'):
        generate_forecast(request,snapshot)


def test_late_entry_source_and_mismatched_cutoff_rejected():
    request,snapshot,schedule=inputs()
    payload=dict(snapshot.payloads,entries_source=SourceRecord('fixture','late',T+timedelta(hours=1)))
    with pytest.raises(ValueError):
        generate_forecast(request,replace(snapshot,payloads=payload),schedule=schedule)
    with pytest.raises(ValueError):
        generate_forecast(request,replace(snapshot,as_of_utc=T+timedelta(seconds=1)),schedule=schedule)


def test_unknown_grid_is_insufficient_and_untrained_model_is_not_faked():
    request,snapshot,schedule=inputs();payload=dict(snapshot.payloads,driver_sessions=[])
    result=generate_forecast(request,replace(snapshot,payloads=payload),model_id='grid',schedule=schedule)
    assert result['status']=='insufficient_data'
    assert result['ranking']==[]
    with pytest.raises(ValueError,match='artifact'):
        generate_forecast(request,snapshot,model_id='ridge',schedule=schedule)
    with pytest.raises(ValueError,match='historical research'):
        generate_forecast(request,snapshot,model_id='tabfm',schedule=schedule)


def test_form_prior_is_disclosed_as_estimate_not_validated_forecast():
    request,snapshot,schedule=inputs()
    result=generate_forecast(request,snapshot,schedule=schedule)
    assert result['status']=='degraded'
    assert 'prior' in ' '.join(result['warnings']).lower()
    assert result['provenance']['validation_status']=='unevaluated'


def test_baseline_rejects_unrelated_artifact():
    from types import SimpleNamespace
    request,snapshot,schedule=inputs()
    with pytest.raises(ValueError,match='baseline'):
        generate_forecast(request,snapshot,model_id='grid',artifact=SimpleNamespace(model_id='unrelated'),schedule=schedule)


def test_observer_failure_does_not_change_canonical_forecast():
    request,snapshot,schedule=inputs()
    def failing(event): raise RuntimeError('observer failure')
    a=generate_forecast(request,snapshot,model_id='grid',schedule=schedule,progress=failing)
    b=generate_forecast(request,snapshot,model_id='grid',schedule=schedule)
    assert a==b


def test_saved_history_is_default_and_explicit_empty_history_overrides_it():
    from flows.prediction_service import forecast_run_id
    request,snapshot,schedule=history_inputs()
    result=generate_forecast(request,snapshot,schedule=schedule)
    assert [(r['driver_id'],r['prediction_score'],r['baseline_source']) for r in result['ranking']]==[('B',1.0,'driver_form'),('A',2.0,'driver_form')]
    assert forecast_run_id(request,snapshot)==result['run_id']
    assert forecast_run_id(request,snapshot,history=snapshot.payloads['history'])==result['run_id']
    prior=generate_forecast(request,snapshot,schedule=schedule,history=())
    assert prior['run_id']!=result['run_id']
    assert all(r['baseline_source']=='prior' and r['prediction_score']==1.5 for r in prior['ranking'])


def test_saved_history_reaches_learned_model_features_without_future_labels(monkeypatch):
    from types import SimpleNamespace
    import ml.training
    request,snapshot,schedule=history_inputs()
    artifact=SimpleNamespace(model_id='fitted',candidate='ridge',target='finishing_position',model_version='1',
        metadata=dict(training_availability_verified=True,training_max_available_at_utc=(T-timedelta(days=1)).isoformat(),
                      training_max_event_time_utc=(T-timedelta(days=2)).isoformat()))
    def predict(artifact,features):
        assert [(r['driver_id'],r['driver_form_position'],r['form_events']) for r in features.rows]==[('A',2,1),('B',1,1)]
        assert all('actual_position' not in r for r in features.rows)
        return [dict(r,prediction_score=r['driver_form_position'],score_units='ordinal_position') for r in features.rows]
    monkeypatch.setattr(ml.training,'predict_model',predict)
    result=generate_forecast(request,snapshot,schedule=schedule,model_id='ridge',artifact=artifact)
    assert [r['driver_id'] for r in result['ranking']]==['B','A']
