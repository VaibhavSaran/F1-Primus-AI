"""Offline candidate fitting and inference; training never implies promotion."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import isfinite
from numbers import Real
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data.contracts import require_utc
from ml.artifacts import (DATASET_SCHEMA_VERSION, FEATURE_SCHEMA_VERSION, TARGETS, ModelArtifact,
                          validate_artifact)
from ml.dataset import DatasetBundle
from ml.features import FEATURE_COLUMNS, FeatureFrame

CATEGORICAL = ('circuit_id', 'team_id', 'compound')


@dataclass(frozen=True)
class TrainingConfig:
    candidate: str = 'ridge'
    target: str = 'finishing_position'
    model_version: str = '1'
    seed: int = 42
    split_manifest: dict[str, Any] = field(default_factory=dict)


def candidate_registry() -> dict[str, dict]:
    """Availability is separate from evaluation and production eligibility."""
    registry = {name: dict(available=True, production_eligible=False, status='unevaluated',
                           targets=list(TARGETS), reason='Requires frozen held-out evaluation')
                for name in ('ridge', 'random_forest', 'gradient_boosting', 'xgboost')}
    from importlib.util import find_spec
    if find_spec('xgboost') is None:
        registry['xgboost'].update(available=False, reason='XGBoost is not installed in the pinned runtime')
    registry['tabfm'] = dict(available=False, production_eligible=False, status='not_configured',
                            targets=list(TARGETS),
                            reason='TabFM permitted access/license and runtime/service are not validated')
    return registry


def _records(data: Any, *, training: bool) -> list[dict]:
    if isinstance(data, DatasetBundle):
        if data.schema_version != DATASET_SCHEMA_VERSION or data.feature_columns != FEATURE_COLUMNS:
            raise ValueError('incompatible dataset schema')
        data = data.rows
    elif isinstance(data, FeatureFrame):
        if training:
            raise ValueError('training requires labelled dataset rows')
        if data.schema_version != FEATURE_SCHEMA_VERSION or data.feature_columns != FEATURE_COLUMNS:
            raise ValueError('incompatible feature schema')
        data = data.rows
    elif isinstance(data, pd.DataFrame):
        data = data.to_dict('records')
    rows, seen = [], set()
    for row in data:
        if not isinstance(row, dict):
            raise ValueError('model rows must be dictionaries')
        key = row.get('event_id'), row.get('driver_id')
        if any(not isinstance(value, str) or not value.strip() for value in key):
            raise ValueError('event_id and driver_id are required')
        if key in seen:
            raise ValueError('duplicate event/driver model row')
        seen.add(key)
        rows.append(dict(row))
    return rows


def _feature_table(rows: list[dict], columns: tuple[str, ...]) -> pd.DataFrame:
    records = []
    for row in rows:
        values = {}
        for column in columns:
            if column not in row:
                raise ValueError(f'missing feature column: {column}')
            value = row[column]
            if column in CATEGORICAL:
                if value is not None and not isinstance(value, str):
                    raise ValueError(f'{column} must be text or None')
            elif value is not None:
                if isinstance(value, (bool, np.bool_)) and column.endswith('_missing'):
                    value = int(value)
                elif isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not isfinite(value):
                    raise ValueError(f'{column} must be a finite number or None')
            values[column] = np.nan if value is None else value
        records.append(values)
    return pd.DataFrame(records, columns=columns)


def _estimator(config: TrainingConfig):
    if config.candidate == 'xgboost':
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                            objective='reg:squarederror', random_state=config.seed,
                            n_jobs=1, tree_method='hist')
    if config.candidate == 'ridge':
        return Ridge(alpha=1.0)
    if config.candidate == 'random_forest':
        return RandomForestRegressor(n_estimators=100, max_depth=6, random_state=config.seed, n_jobs=1)
    return GradientBoostingRegressor(n_estimators=100, max_depth=2, random_state=config.seed)


def _training_times(rows: list[dict]) -> dict:
    times = {'event_time_utc': [], 'outcome_available_at_utc': []}
    verified = True
    for row in rows:
        observed = {}
        for column in times:
            value = row.get(column)
            if value is None:
                verified = False
                continue
            try:
                if isinstance(value, str):
                    value = datetime.fromisoformat(value.replace('Z', '+00:00'))
                observed[column] = require_utc(value)
            except (ValueError, TypeError) as exc:
                raise ValueError(f'{column} must be a valid timezone-aware timestamp or None') from exc
            times[column].append(observed[column])
        if (len(observed) == 2 and
                observed['outcome_available_at_utc'] < observed['event_time_utc']):
            raise ValueError('training outcome cannot be available before its event')
    return dict(training_availability_verified=verified,
                training_max_available_at_utc=max(times['outcome_available_at_utc']).isoformat()
                if times['outcome_available_at_utc'] else None,
                training_max_event_time_utc=max(times['event_time_utc']).isoformat()
                if times['event_time_utc'] else None)


def train_model(dataset: DatasetBundle | Any, config: TrainingConfig) -> ModelArtifact:
    """Fit ONLY the supplied training rows, including all preprocessing.

    Call separately with each WalkForwardFold.train_rows; never pass the complete
    dataset before splitting. clean_pace labels are actual_pace_s clean race-pace
    deltas relative to the event's reference, supplied by the dataset producer.
    This function neither evaluates nor promotes an artifact.
    """
    registry = candidate_registry()
    if config.candidate not in registry:
        raise ValueError(f'unknown candidate: {config.candidate}')
    if not registry[config.candidate]['available']:
        raise ValueError(registry[config.candidate]['reason'])
    if config.target not in TARGETS:
        raise ValueError(f'unknown target: {config.target}')
    if not isinstance(config.model_version, str) or not config.model_version.strip():
        raise ValueError('model_version is required')
    if type(config.seed) is not int or not 0 <= config.seed < 2 ** 32:
        raise ValueError('seed must be an unsigned 32-bit integer')
    rows = _records(dataset, training=True)
    _feature_table(rows, FEATURE_COLUMNS)
    target_column, units, definition = TARGETS[config.target]
    eligible, labels = [], []
    for row in rows:
        if (str(row.get('status', '')).upper() in {'DNS', 'DSQ', 'PENDING'} or
                row.get('outcome_status') == 'pending' or row.get('eligible') is False):
            continue
        value = row.get(target_column)
        if value is None:
            continue
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not isfinite(value):
            raise ValueError(f'{target_column} must be finite')
        if config.target == 'finishing_position' and (value < 1 or int(value) != value):
            raise ValueError('actual_position must be a positive classified integer')
        eligible.append(row)
        labels.append(float(value))
    if not eligible:
        raise ValueError(f'no eligible training rows with {target_column}')
    training_times = _training_times(eligible)
    manifest = json.loads(json.dumps(config.split_manifest))
    event_ids = sorted({row['event_id'] for row in eligible})
    if ('train_event_ids' in manifest and set(manifest['train_event_ids']) != set(event_ids) or
            set(manifest.get('test_event_ids', ())) & set(event_ids)):
        raise ValueError('split manifest must match training events and exclude heldout events')
    categorical = [column for column in FEATURE_COLUMNS if column in CATEGORICAL]
    numeric = [column for column in FEATURE_COLUMNS if column not in CATEGORICAL]
    preprocessing = ColumnTransformer([
        ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median', keep_empty_features=True,
                                                       add_indicator=True)),
                              ('scale', StandardScaler())]), numeric),
        ('categorical', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='__MISSING__',
                                                           keep_empty_features=True)),
                                  ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]),
         categorical),
    ])
    pipeline = Pipeline([('preprocessing', preprocessing), ('estimator', _estimator(config))])
    pipeline.fit(_feature_table(eligible, FEATURE_COLUMNS), labels)
    fingerprint_rows = [{key: row.get(key) for key in (*FEATURE_COLUMNS, 'event_id', 'driver_id', target_column,
                                                      'event_time_utc', 'outcome_available_at_utc')}
                        for row in eligible]
    metadata = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                    sklearn_version=sklearn.__version__, numpy_version=np.__version__, pandas_version=pd.__version__,
                    validation_status='unevaluated', production_eligible=False,
                    target_column=target_column, score_units=units, target_definition=definition,
                    lower_score_better=True,
                    n_training_rows=len(eligible), n_excluded_rows=len(rows) - len(eligible), seed=config.seed,
                    training_event_ids=event_ids,
                    training_source_ids=sorted({source for row in eligible for source in row.get('source_ids', ())}),
                    training_snapshot_ids=sorted({row['snapshot_id'] for row in eligible if row.get('snapshot_id')}),
                    split_manifest=manifest,
                    dataset_sha256=sha256(json.dumps(fingerprint_rows, sort_keys=True, default=str).encode()).hexdigest(),
                    estimator_parameters=pipeline.named_steps['estimator'].get_params())
    metadata.update(training_times)
    if config.candidate == 'xgboost':
        import xgboost
        metadata['xgboost_version'] = xgboost.__version__
        # XGBoost's default NaN sentinel needs an explicit JSON-safe representation.
        metadata['estimator_parameters']['missing'] = 'NaN'
    artifact = ModelArtifact(model_id=f'{config.candidate}-{config.target}-{uuid4().hex}',
                             candidate=config.candidate, target=config.target, model_version=config.model_version,
                             pipeline=pipeline, metadata=metadata)
    validate_artifact(artifact)
    return artifact


def predict_model(artifact: ModelArtifact, features: FeatureFrame | Any) -> list[dict]:
    """Predict without fitting; ordinal ranking scores are never race times or odds."""
    validate_artifact(artifact)
    rows = _records(features, training=False)
    if not rows:
        return []
    scores = artifact.pipeline.predict(_feature_table(rows, artifact.feature_columns))
    if len(scores) != len(rows) or not np.isfinite(scores).all():
        raise ValueError('model returned invalid prediction scores')
    result = []
    for row, score in zip(rows, scores):
        prediction = dict(row, prediction_score=float(score), score_units=TARGETS[artifact.target][1],
                          target_definition=artifact.metadata['target_definition'],
                          model_id=artifact.model_id, model_version=artifact.model_version)
        if artifact.target == 'clean_pace':
            prediction['predicted_pace_s'] = float(score)
        result.append(prediction)
    result.sort(key=lambda row: (row['event_id'], row['prediction_score'], row['driver_id']))
    ranks = {}
    for row in result:
        ranks[row['event_id']] = ranks.get(row['event_id'], 0) + 1
        row['predicted_rank'] = ranks[row['event_id']]
    return result
