"""Readiness labels must never imply a research model produced a forecast."""
from ml.foundation_models import foundation_registry


def test_models_are_visible_with_distinct_prediction_roles():
    models = foundation_registry()
    assert set(models) == {'tabfm', 'timesfm'}
    assert models['tabfm']['targets'] == ['finishing_position', 'clean_pace']
    assert models['timesfm']['targets'] == ['pace_sequence']
    assert models['timesfm']['direct_race_ranking'] is False
    assert models['tabfm']['direct_race_ranking'] is True


def test_unconfigured_models_cannot_masquerade_as_working_candidates(tmp_path):
    for model_id, model in foundation_registry(tmp_path).items():
        assert model['id'] == model_id
        assert model['label'] and model['role']
        assert model['available'] is False
        assert model['production_eligible'] is False
        assert model['status'] == 'not_configured'
        assert model['reason']
        assert model['source_url'].startswith('https://github.com/google-research/')
        assert model['weights_license'] == ('non-commercial, non-production' if model_id == 'tabfm'
                                           else 'Apache-2.0')


def test_tabfm_runtime_and_timesfm_auxiliary_limits_are_explained():
    models = foundation_registry()
    assert 'Python 3.11' in models['tabfm']['next_step']
    import sys
    if sys.version_info >= (3,11):
        assert 'satisfies' in models['tabfm']['next_step']
    else:
        assert '3.10' in models['tabfm']['next_step']
    assert 'pace' in models['timesfm']['description'].lower()
    assert '2.5' in models['timesfm']['next_step']
    assert 'Apache-2.0' in models['timesfm']['next_step']


def test_ui_modification_does_not_mutate_registry_between_calls():
    models = foundation_registry()
    models['tabfm']['targets'].append('winner_probability')
    assert 'winner_probability' not in foundation_registry()['tabfm']['targets']
