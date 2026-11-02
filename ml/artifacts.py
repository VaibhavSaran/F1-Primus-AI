"""Versioned fitted models in an operator-controlled, trusted local directory.

Joblib executes Python while loading. This store must never receive uploaded or
otherwise untrusted files; model selection accepts IDs, never user paths.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re
import tempfile
from typing import Any

import joblib
import sklearn

from ml.features import FEATURE_COLUMNS

MODEL_SCHEMA_VERSION = 'model-v1'
FEATURE_SCHEMA_VERSION = 'features-v1'
DATASET_SCHEMA_VERSION = 'dataset-v1'
TARGETS = {'finishing_position': ('actual_position', 'ordinal_position', 'official_classified_finishing_position'),
           'clean_pace': ('actual_pace_s', 'seconds', 'event_relative_clean_pace_delta')}
MODEL_ID = re.compile(r'(ridge|random_forest|gradient_boosting|xgboost)-(finishing_position|clean_pace)-[0-9a-f]{32}')


@dataclass(frozen=True)
class ModelArtifact:
    model_id: str
    candidate: str
    target: str
    model_version: str
    pipeline: Any
    metadata: dict[str, Any]
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS
    schema_version: str = MODEL_SCHEMA_VERSION
    feature_schema_version: str = FEATURE_SCHEMA_VERSION
    dataset_schema_version: str = DATASET_SCHEMA_VERSION


def validate_artifact(artifact: ModelArtifact) -> None:
    if not isinstance(artifact, ModelArtifact):
        raise ValueError('expected a ModelArtifact')
    if (artifact.schema_version != MODEL_SCHEMA_VERSION or
            artifact.feature_schema_version != FEATURE_SCHEMA_VERSION or
            artifact.dataset_schema_version != DATASET_SCHEMA_VERSION or
            artifact.feature_columns != FEATURE_COLUMNS):
        raise ValueError('incompatible model or feature schema')
    if not isinstance(artifact.model_id, str) or not MODEL_ID.fullmatch(artifact.model_id):
        raise ValueError('invalid model_id')
    if not artifact.model_id.startswith(f'{artifact.candidate}-{artifact.target}-'):
        raise ValueError('model_id does not match candidate and target')
    if artifact.metadata.get('sklearn_version') != sklearn.__version__:
        raise ValueError('incompatible sklearn version; retrain in the pinned runtime')
    if artifact.candidate == 'xgboost':
        import xgboost
        if artifact.metadata.get('xgboost_version') != xgboost.__version__:
            raise ValueError('incompatible XGBoost version; retrain in the pinned runtime')
    target_column, units, definition = TARGETS[artifact.target]
    if (artifact.metadata.get('target_column') != target_column or
            artifact.metadata.get('score_units') != units or
            artifact.metadata.get('target_definition') != definition or
            artifact.metadata.get('lower_score_better') is not True):
        raise ValueError('incompatible target definition or score units')
    if not hasattr(artifact.pipeline, 'predict'):
        raise ValueError('artifact has no fitted prediction pipeline')


class ArtifactStore:
    """The root is trusted operator configuration, not a request parameter."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def _path(self, model_id: str) -> Path:
        if not isinstance(model_id, str) or not MODEL_ID.fullmatch(model_id):
            raise ValueError('model_id must be a generated local artifact ID')
        path = self.root / f'{model_id}.joblib'
        if path.is_symlink() or path.resolve().parent != self.root:
            raise ValueError('model_id must refer to a file inside the trusted store')
        return path

    def save(self, artifact: ModelArtifact) -> str:
        validate_artifact(artifact)
        path = self._path(artifact.model_id)
        self.root.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)
        descriptor, temporary = tempfile.mkstemp(prefix='.model-', suffix='.tmp', dir=self.root)
        os.close(descriptor)
        try:
            joblib.dump(artifact, temporary, compress=3)
            with open(temporary, 'r+b') as handle:
                os.fsync(handle.fileno())
            # Windows rename refuses overwrite; POSIX hard links provide the
            # same atomic publication without replacing an existing version.
            if os.name == 'nt':
                os.rename(temporary, path)
            else:
                os.link(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return artifact.model_id

    def load(self, model_id: str) -> ModelArtifact:
        path = self._path(model_id)
        artifact = joblib.load(path)
        validate_artifact(artifact)
        if artifact.model_id != model_id:
            raise ValueError('stored model_id does not match selected artifact')
        return artifact
