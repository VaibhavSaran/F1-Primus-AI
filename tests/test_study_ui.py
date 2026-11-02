import json
from streamlit.testing.v1 import AppTest
from streamlit_app import services


def example():
    from streamlit_app.screens import historical_studies
    historical_studies()


def test_studies_show_actual_comparison_without_json_or_code(monkeypatch):
    model = dict(n_events=109, metrics=dict(rank_spearman=.67, position_mae=3.2, winner_accuracy=.5))
    case = dict(event_id='2025:19', event_name='United States Grand Prix', cutoff_utc='2025-10-19T18:59:59+00:00',
        input_sessions=['FP1','SQ','S','Q'], models=dict(ridge=[dict(driver_name='Max Verstappen',
        predicted_rank=1, actual_position=1, position_error=0, status='Finished')]))
    report = dict(cohort_event_ids=['2025:19'], models=dict(ridge=model), case_study=case,
        paired_qualifying_differences={}, foundation_metadata={})
    monkeypatch.setattr(services, 'model_studies', lambda:[('study',report)], raising=False)
    monkeypatch.setattr(services, 'study_race', lambda *a:case, raising=False)
    app=AppTest.from_function(example).run()
    assert not app.exception and not app.get('json') and not app.get('code')
    assert any('Max Verstappen' in str(x.value) for x in app.dataframe)
    assert any('United States Grand Prix' in str(x.value) for x in app.subheader)


def test_model_studies_ignore_other_cohorts_and_untrusted_directory_names(monkeypatch,tmp_path):
    monkeypatch.setattr(services,'LOCAL',tmp_path)
    (tmp_path/'archive_ml').mkdir()
    (tmp_path/'archive_ml'/'manifest.json').write_text(json.dumps(dict(cohort_sha256='current')))
    for name,cohort in [('a'*64,'current'),('b'*64,'old'),('arbitrary','current')]:
        folder=tmp_path/'model_studies'/name;folder.mkdir(parents=True)
        (folder/'report.json').write_text(json.dumps(dict(version='historical-comparison-v1',cohort_sha256=cohort)))
    assert [x[0] for x in services.model_studies()]==['a'*64]


def test_race_view_checks_prediction_file_digest_and_selects_other_historical_races(monkeypatch,tmp_path):
    from hashlib import sha256
    import pytest
    monkeypatch.setattr(services,'LOCAL',tmp_path)
    folder=tmp_path/'model_studies'/('a'*64);folder.mkdir(parents=True)
    predictions=dict(ridge=[dict(event_id='2024:1',driver_name='Driver One',predicted_rank=1,actual_position=2)])
    raw=json.dumps(predictions).encode();(folder/'predictions.json').write_bytes(raw)
    report=dict(races={'2024:1':dict(event_id='2024:1',event_name='Bahrain Grand Prix')},
        models={'ridge':{}},predictions_sha256=sha256(raw).hexdigest())
    (folder/'report.json').write_text(json.dumps(report))
    result=services.study_race('a'*64,'2024:1')
    assert result['event_name']=='Bahrain Grand Prix'
    assert result['models']['ridge'][0]['driver_name']=='Driver One'
    (folder/'predictions.json').write_text('{}')
    with pytest.raises(ValueError,match='digest'):services.study_race('a'*64,'2024:1')
