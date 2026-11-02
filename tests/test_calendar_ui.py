import json

from test_streamlit import offline_ui, page, saved_input
from streamlit_app import services


def test_named_dated_calendar_and_full_session_labels(tmp_path):
    catalog = tmp_path/'catalog'
    catalog.mkdir()
    (catalog/'calendar.json').write_text(json.dumps({'schema_version':'calendar-v1','events':[
        {'season':2025,'round_number':8,'event_name':'Monaco Grand Prix','race_date':'2025-05-25',
         'sessions':[{'session_id':'FP1'},{'session_id':'Q'},{'session_id':'R'}]},
        {'season':2025,'round_number':24,'event_name':'Abu Dhabi Grand Prix','race_date':'2025-12-07',
         'sessions':[{'session_id':'FP1'},{'session_id':'Q'},{'session_id':'R'}]}]}),encoding='utf-8')
    app=page()
    app.selectbox(key='season').select(2025).run()
    assert not app.exception
    assert app.selectbox(key='round_number').options == ['Monaco Grand Prix · 2025-05-25','Abu Dhabi Grand Prix · 2025-12-07']
    assert app.selectbox(key='round_number').value is None
    app.selectbox(key='round_number').select(8).run()
    assert app.selectbox(key='session').options == ['Grand Prix Race','Qualifying','Free Practice 1']
    assert app.session_state['workspace_context']['round_number']==8


def test_prediction_uses_saved_prerace_cutoff_automatically(tmp_path, monkeypatch):
    import streamlit.elements.lib.policies as policies
    monkeypatch.setattr(policies, '_shown_default_value_warning', False)
    snapshot=saved_input(tmp_path)
    app=page('predictions')
    assert not app.exception
    assert app.session_state['forecast_request_context']['cutoff']==snapshot.as_of_utc
    assert not next(button for button in app.button if button.label=='Generate forecast').disabled
    assert any('Grand Prix' in caption.value for caption in app.caption)
    assert not any('default value' in warning.value for warning in app.warning)


def test_foundation_model_selection_is_explicit_and_never_falls_back():
    app=page('predictions')
    for model in ('tabfm',):
        app.selectbox(key='prediction_model').select(model).run()
        assert not app.exception
        assert next(button for button in app.button if button.label=='Generate forecast').disabled
        assert any('TabFM has no saved result' in item.value for item in app.info)
