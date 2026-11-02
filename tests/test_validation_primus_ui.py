"""Readable validation and grounded Primus AI presentation without providers."""
import pytest
from streamlit.testing.v1 import AppTest
from streamlit_app import services


@pytest.fixture(autouse=True)
def local_ui(monkeypatch, tmp_path):
    monkeypatch.setattr(services, 'LOCAL', tmp_path)


def validation():
    from streamlit_app.screens import model_validation
    model_validation()


def assistant():
    from streamlit_app.screens import ask_f1
    ask_f1()


def test_evaluation_shows_readable_metrics_and_model_filter(monkeypatch):
    report = dict(status='evaluated_development', mode='post_qualifying', final_holdout='locked',
        cohort_event_ids=['2025:1', '2025:2'], folds=[{}, {}], protocol={'version':'test-v1'},
        models={'ridge':dict(n_events=2, metrics={'rank_spearman':0.8, 'position_mae':1.2},
                            metric_event_counts={'rank_spearman':2, 'position_mae':2}),
                'random_forest':dict(n_events=2, metrics={'rank_spearman':0.7},
                                     metric_event_counts={'rank_spearman':2})})
    monkeypatch.setattr(services, 'evaluations', lambda:[('evaluation-test.json',report)])
    monkeypatch.setattr(services, 'tracking_queries', lambda:None)
    app=AppTest.from_function(validation).run()
    assert not app.exception
    assert not app.json
    tables=[item.value for item in app.dataframe if 'Model' in item.value.columns]
    assert tables and set(tables[0]['Model'])=={'Ridge','Random forest'}
    app.selectbox(key='evaluation_model').select('ridge').run()
    assert not app.exception
    tables=[item.value for item in app.dataframe if 'Model' in item.value.columns]
    assert set(tables[0]['Model'])=={'Ridge'}
    assert '1.200' in str(tables[0].values)


def test_coverage_report_displays_counts_not_code_or_fake_accuracy(monkeypatch):
    report=dict(status='insufficient_data',models={},cohort_event_ids=[],folds=[],protocol={},
                coverage={'sessions_loaded':4,'total_laps':3023,'eligible_clean_laps':0},
                readiness={'reasons':['insufficient_verified_development_folds']})
    monkeypatch.setattr(services,'evaluations',lambda:[('evaluation-test.json',report)])
    monkeypatch.setattr(services,'tracking_queries',lambda:None)
    app=AppTest.from_function(validation).run()
    assert not app.exception and not app.json and not app.code
    assert any(item.label=='Observed laps' and item.value=='3023' for item in app.metric)
    assert any('accuracy' in item.value.lower() for item in app.warning)


def test_primus_ai_clean_name_context_and_citations_survive_rerender(monkeypatch):
    monkeypatch.setattr(services,'snapshots',lambda:([],0))
    monkeypatch.setattr(services,'tracking_queries',lambda:None)
    monkeypatch.setattr(services,'local_identity',lambda:object())
    class Grounded:
        def ask(self,*args,**kwargs):
            return dict(answer='Saved F1 evidence. [1]',citations=[dict(id='1',kind='lap_summary',reference='snapshot-verified',as_of_utc='2025-01-01T00:00:00Z')])
    monkeypatch.setattr(services,'chat_service',lambda snapshot:Grounded())
    app=AppTest.from_function(assistant).run()
    assert not app.exception and app.title[0].value=='Primus AI'
    assert any(item.label=='Additional context' and not item.proto.expanded for item in app.expander)
    app.chat_input[0].set_value('Explain the saved F1 evidence').run()
    assert not app.exception and not app.json
    assert any('Sources: Internal data' in item.value for item in app.caption)
    assert app.session_state['chat_messages'][-1]['citations'][0]['reference'] == 'snapshot-verified'
    app.run()
    assert any('Sources: Internal data' in item.value for item in app.caption)


def test_tracking_model_and_outcome_filters_do_not_show_earlier_runs_as_scored(monkeypatch):
    records=[dict(run_id='latest-ridge',model_id='ridge',mode='post_qualifying',event_id='2025:1',as_of_utc='2025-03-01'),
             dict(run_id='older-ridge',model_id='ridge',mode='post_qualifying',event_id='2025:1',as_of_utc='2025-02-28'),
             dict(run_id='pending-forest',model_id='random_forest',mode='post_qualifying',event_id='2025:2',as_of_utc='2025-03-08')]
    class Queries:
        def forecasts(self): return records
        def request_health(self): return dict(requests=0, failures=0, denied=0, p95_latency_ms=None)
        def outcome_state(self, run_id): return 'pending' if run_id == 'pending-forest' else 'completed'
        def summary(self, model_id=None, evidence_kind="forecast"):
            groups=[dict(model_id='ridge',mode='post_qualifying',n_scored_events=1,n_pending_events=0,
                         selected_run_ids=['latest-ridge'],metrics={'position_mae':1.5},metric_event_counts={'position_mae':1},
                         per_race=[dict(event_id='2025:1',outcome_status='completed',metrics={'position_mae':1.5})]),
                    dict(model_id='random_forest',mode='post_qualifying',n_scored_events=0,n_pending_events=1,
                         selected_run_ids=['pending-forest'],metrics={'position_mae':None},metric_event_counts={'position_mae':0},
                         per_race=[dict(event_id='2025:2',outcome_status='pending',metrics={'position_mae':None})])]
            return dict(groups=[g for g in groups if model_id is None or g['model_id']==model_id])
    monkeypatch.setattr(services,'evaluations',lambda:[])
    monkeypatch.setattr(services,'tracking_queries',lambda:Queries())
    app=AppTest.from_function(validation).run()
    assert not app.exception and not app.json
    run_table=next(item.value for item in app.dataframe if 'Run' in item.value.columns)
    assert run_table.loc[run_table['Run']=='older-ridge','Outcome'].iloc[0]=='Earlier cutoff · outside current summary'
    app.selectbox(key='tracked_model').select('random_forest').run()
    app.selectbox(key='tracked_outcome').select('Awaiting outcome').run()
    assert not app.exception
    runs=next(item.value for item in app.dataframe if 'Run' in item.value.columns)
    assert runs['Run'].tolist()==['pending-forest']
    history=next(item.value for item in app.dataframe if 'Race' in item.value.columns and 'Run' not in item.value.columns)
    assert history['Race'].tolist()==['2025:2']
