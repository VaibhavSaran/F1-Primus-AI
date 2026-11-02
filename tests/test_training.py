"""Offline training/inference contracts; synthetic scores are not accuracy evidence."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from importlib import import_module

import numpy as np
import pandas as pd
import pytest

from ml.dataset import DatasetBundle
from ml.features import FEATURE_COLUMNS, FeatureFrame


def training_rows():
    rows = []
    for event in range(3):
        start = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=14 * event)
        for driver, position in [('a', 1), ('b', 2), ('c', 3)]:
            row = dict.fromkeys(FEATURE_COLUMNS)
            row.update(event_id=f'2024:{event + 1}', driver_id=driver, team_id='old-team',
                       circuit_id='old-track', driver_form_position=float(position),
                       grid_position=position, form_events=3, practice_missing=True,
                       qualifying_missing=False, actual_position=position,
                       actual_pace_s=position * 0.5, eligible=True, outcome_status='completed',
                       status='FINISHED', event_time_utc=start,
                       forecast_cutoff_utc=start - timedelta(hours=1),
                       outcome_available_at_utc=start + timedelta(hours=3),
                       snapshot_id=f'snapshot-{event}', source_ids=(f'source-{event}',))
            rows.append(row)
    return rows


@pytest.mark.parametrize('candidate', ['ridge', 'random_forest', 'gradient_boosting'])
def test_candidates_train_separate_targets_and_predict_without_fitting(candidate, monkeypatch):
    training = import_module('ml.training')
    for target, units in [('finishing_position', 'ordinal_position'), ('clean_pace', 'seconds')]:
        artifact = training.train_model(DatasetBundle(tuple(training_rows())),
                                        training.TrainingConfig(candidate=candidate, target=target))
        def forbidden_fit(*args, **kwargs):
            raise AssertionError('inference must not train')
        monkeypatch.setattr(artifact.pipeline, 'fit', forbidden_fit)
        predictions = training.predict_model(artifact, FeatureFrame(tuple(training_rows()[:3])))
        assert [row['driver_id'] for row in predictions] == ['a', 'b', 'c']
        assert [row['predicted_rank'] for row in predictions] == [1, 2, 3]
        assert all(np.isfinite(row['prediction_score']) and row['score_units'] == units
                   for row in predictions)
        assert all(row['model_id'] == artifact.model_id for row in predictions)
        assert all(('predicted_pace_s' in row) == (target == 'clean_pace') for row in predictions)
        assert artifact.metadata['validation_status'] == 'unevaluated'
        assert artifact.metadata['training_source_ids'] == ['source-0', 'source-1', 'source-2']
        assert artifact.metadata['training_snapshot_ids'] == ['snapshot-0', 'snapshot-1', 'snapshot-2']


def test_rookie_new_team_and_no_practice_keep_identity_and_missing_columns():
    training = import_module('ml.training')
    artifact = training.train_model(training_rows(), training.TrainingConfig())
    rookie = dict(training_rows()[0], driver_id='rookie', team_id='new-team',
                  circuit_id='new-track', compound='UNKNOWN', driver_form_position=None)
    prediction = training.predict_model(artifact, FeatureFrame((rookie,)))[0]
    assert prediction['driver_id'] == 'rookie'
    assert np.isfinite(prediction['prediction_score'])
    assert artifact.pipeline.named_steps['preprocessing'].transform(
        pd.DataFrame([{key: np.nan if rookie[key] is None else rookie[key]
                       for key in FEATURE_COLUMNS}])).shape[1] > 20


def test_tied_scores_rank_by_driver_identity_within_each_event():
    training = import_module('ml.training')
    artifact = training.train_model(training_rows(), training.TrainingConfig())
    row = training_rows()[0]
    tied = [dict(row, event_id='second', driver_id='z'),
            dict(row, event_id='first', driver_id='z'),
            dict(row, event_id='first', driver_id='a')]
    predictions = training.predict_model(artifact, tied)
    assert [(row['event_id'], row['driver_id'], row['predicted_rank']) for row in predictions] == [
        ('first', 'a', 1), ('first', 'z', 2), ('second', 'z', 1)]


def test_fold_training_preprocessing_never_sees_heldout_values():
    training = import_module('ml.training')
    from ml.evaluation import walk_forward_folds
    rows = training_rows()
    for row in rows[3:]:
        row['driver_form_position'] = 10000.0
        row['team_id'] = 'future-team'
    fold = walk_forward_folds(rows, min_train_events=1)[0]
    artifact = training.train_model(fold.train_rows, training.TrainingConfig(
        split_manifest={'train_event_ids': list(fold.train_event_ids),
                        'test_event_ids': list(fold.test_event_ids)}))
    table = artifact.pipeline.named_steps['preprocessing']
    assert table.named_transformers_['numeric'].named_steps['imputer'].statistics_[0] == 2.0
    assert 'future-team' not in table.named_transformers_['categorical'].named_steps['encoder'].categories_[1]
    assert artifact.metadata['training_event_ids'] == ['2024:1']
    assert artifact.metadata['split_manifest']['test_event_ids'] == ['2024:2']
    assert len(training.predict_model(artifact, fold.test_rows)) == 3


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf'), '2.0'])
def test_nonfinite_or_nonnumeric_input_fails_closed(value):
    training = import_module('ml.training')
    rows = training_rows()
    artifact = training.train_model(rows, training.TrainingConfig())
    rows[0]['grid_position'] = value
    with pytest.raises(ValueError, match='grid_position'):
        training.train_model(rows, training.TrainingConfig())
    with pytest.raises(ValueError, match='grid_position'):
        training.predict_model(artifact, rows[:1])


def test_schema_mismatch_or_missing_feature_is_rejected():
    training = import_module('ml.training')
    rows = training_rows()
    artifact = training.train_model(rows, training.TrainingConfig())
    with pytest.raises(ValueError, match='schema'):
        training.train_model(DatasetBundle(tuple(rows), schema_version='future'), training.TrainingConfig())
    with pytest.raises(ValueError, match='schema'):
        training.predict_model(artifact, FeatureFrame(tuple(rows), schema_version='future'))
    with pytest.raises(ValueError, match='schema'):
        training.predict_model(replace(artifact, schema_version='future'), rows)
    del rows[0]['grid_position']
    with pytest.raises(ValueError, match='grid_position'):
        training.predict_model(artifact, rows)


def test_ineligible_rows_are_excluded_and_missing_targets_never_fabricated():
    training = import_module('ml.training')
    rows = training_rows()
    rows[0].update(status='DSQ', actual_position=99)
    rows[1].update(status='DNS', actual_position=99)
    rows[2].update(outcome_status='pending', actual_position=None)
    artifact = training.train_model(rows, training.TrainingConfig())
    assert artifact.metadata['n_training_rows'] == 6
    assert artifact.metadata['n_excluded_rows'] == 3
    with pytest.raises(ValueError, match='eligible'):
        training.train_model(rows[:3], training.TrainingConfig())
    for row in rows:
        row['actual_pace_s'] = None
    with pytest.raises(ValueError, match='eligible'):
        training.train_model(rows, training.TrainingConfig(target='clean_pace'))


def test_registry_exposes_tabfm_as_unavailable_without_fallback():
    training = import_module('ml.training')
    registry = training.candidate_registry()
    assert set(registry) == {'ridge', 'random_forest', 'gradient_boosting', 'xgboost', 'tabfm'}
    assert registry['tabfm']['available'] is False
    assert registry['tabfm']['production_eligible'] is False
    assert registry['tabfm']['reason']
    with pytest.raises(ValueError, match='TabFM|tabfm'):
        training.train_model(training_rows(), training.TrainingConfig(candidate='tabfm'))


def test_nonfinite_target_duplicate_entries_and_unknown_candidate_are_rejected():
    training = import_module('ml.training')
    rows = training_rows()
    rows[0]['actual_position'] = float('inf')
    with pytest.raises(ValueError, match='actual_position'):
        training.train_model(rows, training.TrainingConfig())
    with pytest.raises(ValueError, match='duplicate'):
        training.train_model(training_rows()[:1] * 2, training.TrainingConfig())
    with pytest.raises(ValueError, match='candidate'):
        training.train_model(training_rows(), training.TrainingConfig(candidate='imaginary'))


def test_clean_pace_definition_is_explicit_and_bad_units_fail_closed():
    training = import_module('ml.training')
    artifact = training.train_model(training_rows(), training.TrainingConfig(target='clean_pace'))
    assert artifact.metadata['target_definition'] == 'event_relative_clean_pace_delta'
    predictions = training.predict_model(artifact, training_rows()[:1])
    assert predictions[0]['target_definition'] == 'event_relative_clean_pace_delta'
    bad = replace(artifact, metadata=dict(artifact.metadata, score_units='ordinal_position'))
    with pytest.raises(ValueError, match='target|units'):
        training.predict_model(bad, training_rows()[:1])


def test_frozen_seed_repeats_scores_and_empty_inference_is_empty():
    training = import_module('ml.training')
    config = training.TrainingConfig(candidate='random_forest', seed=17)
    first = training.train_model(training_rows(), config)
    second = training.train_model(training_rows(), config)
    assert [row['prediction_score'] for row in training.predict_model(first, training_rows())] == [
        row['prediction_score'] for row in training.predict_model(second, training_rows())]
    assert training.predict_model(first, FeatureFrame(())) == []


def test_split_manifest_cannot_claim_a_training_race_as_heldout():
    training = import_module('ml.training')
    with pytest.raises(ValueError, match='split|heldout'):
        training.train_model(training_rows(), training.TrainingConfig(
            split_manifest={'train_event_ids': ['2024:1'], 'test_event_ids': ['2024:2']}))


def test_training_artifact_records_latest_verified_label_and_event_times():
    training = import_module('ml.training')
    artifact = training.train_model(training_rows(), training.TrainingConfig())
    assert artifact.metadata['training_availability_verified'] is True
    assert artifact.metadata['training_max_available_at_utc'] == '2024-01-29T03:00:00+00:00'
    assert artifact.metadata['training_max_event_time_utc'] == '2024-01-29T00:00:00+00:00'
    assert artifact.metadata['dataset_sha256']


@pytest.mark.parametrize('missing_column', ['outcome_available_at_utc', 'event_time_utc'])
def test_unknown_training_times_remain_explicitly_unverified(missing_column):
    training = import_module('ml.training')
    rows = training_rows()
    rows[0].pop(missing_column)
    artifact = training.train_model(rows, training.TrainingConfig())
    assert artifact.metadata['training_availability_verified'] is False
    assert artifact.metadata['n_training_rows'] == 9
    for row in rows:
        row.pop('outcome_available_at_utc', None)
        row.pop('event_time_utc', None)
    unknown = training.train_model(rows, training.TrainingConfig())
    assert unknown.metadata['training_availability_verified'] is False
    assert unknown.metadata['training_max_available_at_utc'] is None
    assert unknown.metadata['training_max_event_time_utc'] is None


@pytest.mark.parametrize('column', ['outcome_available_at_utc', 'event_time_utc'])
@pytest.mark.parametrize('value', ['not-a-time', '2024-01-01T00:00:00', datetime(2024, 1, 1)])
def test_malformed_present_training_times_fail_before_fit(column, value):
    training = import_module('ml.training')
    rows = training_rows()
    rows[0][column] = value
    with pytest.raises(ValueError, match=column):
        training.train_model(rows, training.TrainingConfig())


def test_training_outcome_cannot_be_available_before_its_event():
    training = import_module('ml.training')
    rows = training_rows()
    rows[0]['outcome_available_at_utc'] = datetime(2023, 12, 31, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match='outcome.*before.*event'):
        training.train_model(rows, training.TrainingConfig())
