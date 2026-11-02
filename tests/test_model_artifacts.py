"""Only locally produced artifacts can be selected, by versioned ID."""
from dataclasses import replace
from importlib import import_module

import pytest

from test_training import training_rows


def test_local_round_trip_preserves_predictions_and_fitted_metadata(tmp_path):
    training = import_module('ml.training')
    artifacts = import_module('ml.artifacts')
    artifact = training.train_model(training_rows(), training.TrainingConfig(model_version='pilot-1'))
    store = artifacts.ArtifactStore(tmp_path)
    before = training.predict_model(artifact, training_rows()[:3])
    assert store.save(artifact) == artifact.model_id
    loaded = store.load(artifact.model_id)
    assert training.predict_model(loaded, training_rows()[:3]) == before
    assert loaded.model_version == 'pilot-1'
    assert loaded.metadata == artifact.metadata
    with pytest.raises(FileExistsError):
        store.save(artifact)


@pytest.mark.parametrize('model_id', ['../outside', 'C:\\unsafe.joblib', '/tmp/upload', 'upload.joblib', ''])
def test_artifact_selector_rejects_paths_and_upload_names(tmp_path, model_id):
    artifacts = import_module('ml.artifacts')
    with pytest.raises(ValueError, match='model_id'):
        artifacts.ArtifactStore(tmp_path).load(model_id)


def test_missing_artifact_fails_explicitly_without_training(tmp_path):
    artifacts = import_module('ml.artifacts')
    with pytest.raises(FileNotFoundError):
        artifacts.ArtifactStore(tmp_path).load('ridge-finishing_position-' + '0' * 32)


def test_incompatible_artifact_is_rejected_on_save_and_load(tmp_path):
    training = import_module('ml.training')
    artifacts = import_module('ml.artifacts')
    artifact = training.train_model(training_rows(), training.TrainingConfig())
    store = artifacts.ArtifactStore(tmp_path)
    with pytest.raises(ValueError, match='schema'):
        store.save(replace(artifact, schema_version='future'))
    store.save(artifact)
    import joblib
    joblib.dump(replace(artifact, feature_schema_version='future'), tmp_path / f'{artifact.model_id}.joblib')
    with pytest.raises(ValueError, match='schema'):
        store.load(artifact.model_id)


def test_selected_id_and_pinned_runtime_must_match_stored_artifact(tmp_path):
    training = import_module('ml.training')
    artifacts = import_module('ml.artifacts')
    artifact = training.train_model(training_rows(), training.TrainingConfig())
    store = artifacts.ArtifactStore(tmp_path)
    with pytest.raises(ValueError, match='version'):
        store.save(replace(artifact, metadata=dict(artifact.metadata, sklearn_version='old')))
    store.save(artifact)
    import joblib
    joblib.dump(replace(artifact, model_id='ridge-finishing_position-' + '1' * 32),
                tmp_path / f'{artifact.model_id}.joblib')
    with pytest.raises(ValueError, match='model_id'):
        store.load(artifact.model_id)


def test_failed_save_never_exposes_partial_artifact(tmp_path, monkeypatch):
    training = import_module('ml.training')
    artifacts = import_module('ml.artifacts')
    artifact = training.train_model(training_rows(), training.TrainingConfig())
    def failed_dump(value, path, **kwargs):
        with open(path, 'wb') as handle:
            handle.write(b'partial')
        raise OSError('disk failure')
    monkeypatch.setattr(artifacts.joblib, 'dump', failed_dump)
    with pytest.raises(OSError, match='disk failure'):
        artifacts.ArtifactStore(tmp_path).save(artifact)
    assert list(tmp_path.iterdir()) == []
