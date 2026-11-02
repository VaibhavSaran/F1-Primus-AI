"""Hindsight reconstructions remain a separate performance cohort."""
import pytest
from streamlit.testing.v1 import AppTest
from monitoring import TrackingQueries, TrackingStore
from streamlit_app import services
from test_tracking import forecast, outcome


def seed(store, run_id, model_id, *, historical=False, mode='pre_weekend'):
    row=forecast(run_id=run_id,model_id=model_id,
                 provenance={'validation_status':'historical_reconstruction' if historical else 'unevaluated'})
    row['mode']=mode
    store.save_forecast(row)
    store.score_forecast(run_id)


def test_summary_excludes_hindsight_and_requires_explicit_reconstruction_cohort(tmp_path):
    store=TrackingStore(tmp_path/'tracking.sqlite3');store.append_outcome(outcome())
    seed(store,'before-race','model-v1')
    seed(store,'reconstructed','model-v1',historical=True)
    queries=TrackingQueries(store,owner_id='alice')
    assert queries.summary()['groups'][0]['selected_run_ids']==['before-race']
    reconstructed=queries.summary(evidence_kind='historical_reconstruction')
    assert reconstructed['evidence_kind']=='historical_reconstruction'
    assert reconstructed['groups'][0]['selected_run_ids']==['reconstructed']
    with pytest.raises(ValueError,match='evidence_kind'):
        queries.summary(evidence_kind='all')


def test_model_comparison_defaults_to_prospective_cohort_and_can_explicitly_compare_reconstructions(tmp_path):
    store=TrackingStore(tmp_path/'tracking.sqlite3');store.append_outcome(outcome())
    for model in ['model-v1','model-v2']:
        seed(store,'forecast-'+model,model)
        seed(store,'reconstruction-'+model,model,historical=True,mode='post_qualifying')
    queries=TrackingQueries(store,owner_id='alice')
    prospective=queries.compare_models(['model-v1','model-v2'])
    assert all(group['selected_run_ids']==['forecast-'+group['model_id']] for group in prospective['groups'])
    retrospective=queries.compare_models(['model-v1','model-v2'],evidence_kind='historical_reconstruction')
    assert all(group['selected_run_ids']==['reconstruction-'+group['model_id']] for group in retrospective['groups'])


def validation():
    from streamlit_app.screens import model_validation
    model_validation()


def test_validation_auto_selects_only_historical_runs_and_displays_caveat(monkeypatch,tmp_path):
    store=TrackingStore(tmp_path/'tracking.sqlite3');store.append_outcome(outcome())
    seed(store,'reconstruction-v1','ridge',historical=True)
    queries=TrackingQueries(store,owner_id='alice')
    monkeypatch.setattr(services,'tracking_queries',lambda:queries)
    monkeypatch.setattr(services,'evaluations',lambda:[])
    app=AppTest.from_function(validation).run()
    assert not app.exception
    assert app.selectbox(key='tracking_evidence_kind').value=='historical_reconstruction'
    assert any('hindsight' in item.value.lower() for item in app.warning)
    assert any(item.label=='Scored model / race pairs' and item.value=='1' for item in app.metric)
    app.selectbox(key='tracking_evidence_kind').select('forecast').run()
    assert not app.exception
    assert not any('Run' in item.value.columns for item in app.dataframe)
    assert any('No saved pre-race forecasts' in item.value for item in app.info)
