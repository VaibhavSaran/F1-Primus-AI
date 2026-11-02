import pytest
from data.contracts import ProgressEvent
from test_prediction_service import inputs
from agents.graph import build_forecast_graph, run_pre_race_agent


def test_graph_numeric_flow_requires_no_model_client_or_key(monkeypatch):
    monkeypatch.delenv('ANTHROPIC_API_KEY',raising=False)
    request,snapshot,schedule=inputs()
    result=run_pre_race_agent(request=request,snapshot=snapshot,schedule=schedule,model_id='grid')
    assert result['run_id'].startswith('forecast-')
    assert result['report'].startswith('# Research forecast')
    assert result['ranking'][0]['driver_id']=='B'
    assert result['message_count']==0
    assert 'win odds' in result['report']


def test_legacy_round_only_does_not_train_or_guess_context():
    with pytest.raises(ValueError,match='explicit'):
        run_pre_race_agent(round_number=1)


def test_direct_service_and_graph_share_one_immutable_result(tmp_path):
    from flows.prediction_service import generate_forecast
    from monitoring.store import TrackingStore
    request,snapshot,schedule=inputs()
    store=TrackingStore(tmp_path/'tracking.sqlite3')
    direct=generate_forecast(request,snapshot,schedule=schedule,model_id='grid',store=store)
    graphed=run_pre_race_agent(request=request,snapshot=snapshot,schedule=schedule,model_id='grid',store=store)
    assert direct==graphed
