from hashlib import md5,sha256
import json
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest
from streamlit_app import services


@pytest.fixture(autouse=True)
def offline(monkeypatch,tmp_path):
    import streamlit as st
    st.config.set_option('server.address','127.0.0.1')
    monkeypatch.setattr(services,'LOCAL',tmp_path)


def test_primus_is_home_and_context_is_optional():
    app=AppTest.from_file('streamlit_app/app.py').run(timeout=20)
    assert not app.exception
    assert app.title[0].value=='Primus AI'
    assert any(x.label=='Additional context' for x in app.expander)
    assert not any(x.label=='Season' for x in app.selectbox)
    assert app.checkbox(key='chat_bind_context').value is False
    app.checkbox(key='chat_bind_context').check().run(timeout=20)
    assert [x.label for x in app.selectbox[:3]]==['Season','Event / round','Session']


def test_legacy_ask_path_remains_available():
    app=AppTest.from_file('streamlit_app/app.py');app._page_hash=md5(b'ask').hexdigest()
    app.run(timeout=20)
    assert not app.exception and app.title[0].value=='Primus AI'


def test_chat_forecast_attachment_has_readable_labels(monkeypatch):
    from types import SimpleNamespace
    from streamlit_app import screens
    monkeypatch.setattr(screens,'context',lambda:dict(season=2026,round_number=19,session='Q'))
    monkeypatch.setattr(screens,'archive_profile_panel',lambda *_:None)
    monkeypatch.setattr(screens,'choose_snapshot',lambda *_:None)
    run=dict(run_id='forecast-internal-id',event_id='2026:19',model_id='qualifying',
             as_of_utc='2026-11-01T15:33:00+00:00')
    monkeypatch.setattr(screens,'tracking',lambda:SimpleNamespace(forecasts=lambda:[run]))
    monkeypatch.setattr(services,'event_labels',lambda:{'2026:19':'Mexico City Grand Prix · 2026'})
    monkeypatch.setattr(services,'driver_labels',lambda *_:{})
    monkeypatch.setattr(services,'chat_service',lambda *_:SimpleNamespace(provider=None))
    def example():
        import streamlit as st
        from streamlit_app.screens import ask_f1
        st.session_state['chat_bind_context']=True
        ask_f1()
    app=AppTest.from_function(example).run()
    assert not app.exception
    field=app.selectbox(key='chat_attached_run')
    assert field.options[0]=='No forecast attached'
    assert 'Mexico City Grand Prix' in field.options[1]
    assert 'Qualifying baseline' in field.options[1]
    assert '01 Nov 2026 15:33 UTC' in field.options[1]
    assert 'forecast-internal-id' not in field.options[1]
    field.select('forecast-internal-id').run()
    assert not app.exception and app.session_state['chat_attached_run']=='forecast-internal-id'


def test_about_has_credits_without_requiring_a_race():
    app=AppTest.from_file('streamlit_app/app.py');app._page_hash=md5(b'about').hexdigest()
    app.run(timeout=20)
    assert not app.exception and app.title[0].value=='About Primus AI'
    assert not any(x.label=='Season' for x in app.selectbox)
    assert 'FastF1' in ' '.join(x.value for x in app.markdown)
    text=' '.join(x.value for x in app.markdown)+ ' '.join(x.value for x in app.caption)
    assert 'Tom Shaw' in text and 'Mariana Antaya' in text
    assert 'https://github.com/IAmTomShaw)' in text
    assert 'https://github.com/mar-antaya)' in text
    assert 'installed Python' not in text and 'Python 3.12.10' not in text
    assert any(x.label=='Page guide' for x in app.tabs)


def test_about_does_not_discard_shared_race_selection():
    app=AppTest.from_file('streamlit_app/app.py').run(timeout=20)
    app.checkbox(key='chat_bind_context').check().run(timeout=20)
    selected={key:app.session_state[key] for key in ('season','round_number','session')}
    app._page_hash=md5(b'about').hexdigest();app.run(timeout=20)
    assert not app.exception
    assert {key:app.session_state[key] for key in selected}==selected


def test_archived_legacy_report_remains_available_without_public_output_files(monkeypatch,tmp_path):
    monkeypatch.setattr(services,'ROOT',tmp_path/'clean-checkout')
    folder=services.LOCAL/'legacy_reports';folder.mkdir()
    report=folder/'old-preview.md';report.write_text('Historical narrative',encoding='utf-8')
    assert services.reports()==[report]


def test_cached_forecast_requires_exact_snapshot_and_cutoff(monkeypatch):
    from types import SimpleNamespace
    from ml import pending_forecast
    from datetime import datetime,timezone
    snapshot=SimpleNamespace(snapshot_id='pending-test',event=SimpleNamespace(event_id='2026:19'),
        as_of_utc=datetime(2026,11,1,15,tzinfo=timezone.utc))
    saved=dict(snapshot_id=snapshot.snapshot_id,as_of_utc=snapshot.as_of_utc.isoformat(),ranking=[{}])
    monkeypatch.setattr(pending_forecast,'cached_pending_forecast',lambda *a,**kw:saved)
    monkeypatch.setattr(services,'local_identity',lambda:None)
    assert services.cached_forecast(snapshot,'tabfm')==saved
    saved['snapshot_id']='another'
    assert services.cached_forecast(snapshot,'tabfm') is None


def test_completed_chat_is_saved_and_reports_can_download_after_new_conversation(monkeypatch):
    from types import SimpleNamespace
    import streamlit as st
    downloads={}
    original=st.download_button
    def download(label,data,*a,**kw):
        downloads[label]=data
        return original(label,data,*a,**kw)
    monkeypatch.setattr(st,'download_button',download)
    monkeypatch.setattr(services,'chat_service',lambda *a:SimpleNamespace(provider=None,
        ask=lambda *a,**kw:dict(answer='A sourced F1 answer. [L1]',status='answered',
            context=dict(event_id='2026:7',session_id='2026:7:Q'),citations=[dict(id='L1',kind='archive',reference='fixture')])) )
    app=AppTest.from_file('streamlit_app/app.py').run(timeout=20)
    app.chat_input[0].set_value('How did Kimi compare?').run(timeout=20)
    assert not app.exception
    chat_id=app.session_state['chat_conversation_id']
    app.button(key='chat_new_conversation').click().run(timeout=20)
    assert len(services.chat_history().list(services.local_identity()))==1
    report=AppTest.from_file('streamlit_app/app.py');report._page_hash=md5(b'reports').hexdigest();report.run(timeout=20)
    assert not report.exception and report.selectbox(key='report_chat').value==chat_id
    assert 'How did Kimi compare?' in downloads['Download conversation Markdown']
    assert '2026:7:Q' in downloads['Download conversation Markdown']


def test_active_evaluation_is_preferred_and_zero_race_audits_are_not_accuracy_options(tmp_path):
    folder=tmp_path/'model_studies'/('a'*64);folder.mkdir(parents=True)
    report=dict(status='evaluated_reconstruction',models=dict(tabfm={},xgboost={}),
        cohort_event_ids=['2025:1'],protocol_version='nested-latest-100-v1')
    raw=json.dumps(report).encode();(folder/'report.json').write_bytes(raw)
    (tmp_path/'archive_ml').mkdir()
    (tmp_path/'archive_ml'/'active_study.json').write_text(json.dumps(dict(path=str(folder.resolve()),report_sha256=sha256(raw).hexdigest())))
    (tmp_path/'evaluations').mkdir()
    (tmp_path/'evaluations'/'evaluation-audit.json').write_text(json.dumps(dict(status='insufficient_data',cohort_event_ids=[])))
    result=services.evaluations()
    assert result[0][1]['models'].keys()==report['models'].keys()
    assert all(r.get('cohort_event_ids') for _,r in result)


def validation_example():
    from streamlit_app.screens import model_validation
    model_validation()


def test_standard_size_changes_actual_evaluation_denominator(monkeypatch):
    def evaluation(n):
        result=dict(n_events=n,metrics=dict(rank_spearman=.7),confidence_intervals={},metric_event_counts=dict(rank_spearman=n))
        return dict(cohort_event_ids=[f'2020:{i}' for i in range(n)],models=dict(tabfm=result,xgboost=result),paired_qualifying_differences={})
    report=dict(evaluation(100),status='evaluated_reconstruction',protocol=dict(version='nested-latest-100-v1'),
        benchmark_subsets={str(n):evaluation(n) for n in (5,10,50,100)},final_holdout='2026 locked')
    monkeypatch.setattr(services,'evaluations',lambda:[('active',report)])
    monkeypatch.setattr(services,'model_studies',lambda:[])
    app=AppTest.from_function(validation_example).run(timeout=20)
    assert not app.exception
    assert 'TabFM' in str(app.selectbox(key='evaluation_model').options)
    app.selectbox(key='evaluation_size').select(5).run(timeout=20)
    assert any(m.label=='Evaluated races' and str(m.value)=='5' for m in app.metric)
    assert not any(m.label=='Walk-forward folds' for m in app.metric)
    assert not app.get('code') and not app.get('json')
