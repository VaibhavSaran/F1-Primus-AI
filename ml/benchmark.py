"""Fold-local benchmark runner. Final holdouts stay locked in development mode."""
from __future__ import annotations
from dataclasses import asdict
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from ml.artifacts import DATASET_SCHEMA_VERSION,FEATURE_SCHEMA_VERSION,MODEL_SCHEMA_VERSION
from ml.baselines import grid_baseline,rolling_form_baseline
from ml.evaluation import PROTOCOL_VERSION,evaluate_predictions,walk_forward_folds
from ml.features import FEATURE_COLUMNS
from ml.protocols import EvaluationProtocol
from ml.training import TrainingConfig,train_model,predict_model,candidate_registry


def _timestamp(value):
    if isinstance(value,datetime):return value.isoformat()
    raise TypeError('Unsupported evaluation metadata')


def _digest(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(',',':'),default=_timestamp,allow_nan=False).encode()).hexdigest()


def _dataset_manifest(rows):
    # Hash every field, including provenance, without retaining raw input rows.
    row_hashes=[_digest(row) for row in rows]
    return dict(sha256=_digest(row_hashes),row_count=len(rows),row_sha256=row_hashes,
                source_ids=sorted({source for row in rows for source in row.get('source_ids',())}),
                snapshot_ids=sorted({row['snapshot_id'] for row in rows if row.get('snapshot_id')}))


def _implementation_manifest():
    root=Path(__file__).resolve().parents[1]
    files=('benchmark.py','training.py','artifacts.py','features.py','dataset.py','evaluation.py','baselines.py','protocols.py')
    sources={f'ml/{name}':sha256((root/'ml'/name).read_bytes()).hexdigest() for name in files}
    revision=None
    try:
        value=subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],capture_output=True,text=True,check=True,timeout=2).stdout.strip()
        if len(value)==40 and all(c in '0123456789abcdef' for c in value):revision=value
    except (OSError,subprocess.SubprocessError):pass
    return dict(git_revision=revision,source_sha256=_digest(sources),source_files=sources)


def _model_manifest(artifact,config):
    fields=('sklearn_version','numpy_version','pandas_version','validation_status','production_eligible',
            'target_column','score_units','target_definition','lower_score_better','n_training_rows','n_excluded_rows',
            'seed','training_event_ids','training_source_ids','training_snapshot_ids','split_manifest','dataset_sha256',
            'estimator_parameters','training_availability_verified','training_max_available_at_utc','training_max_event_time_utc')
    return dict(artifact_status='metadata_only_not_persisted',candidate=artifact.candidate,target=artifact.target,
                model_version=artifact.model_version,model_schema_version=artifact.schema_version,
                feature_schema_version=artifact.feature_schema_version,dataset_schema_version=artifact.dataset_schema_version,
                training_config=asdict(config),metadata={key:artifact.metadata[key] for key in fields if key in artifact.metadata})


def benchmark(rows,protocol:EvaluationProtocol,*,candidates=('ridge','random_forest','gradient_boosting'),
              mode='post_qualifying',seed=42):
    """Evaluate development events only, preprocessing fitted inside every fold.

    Input rows must come from audited cutoff-safe snapshots. This function does
    not prove upstream provenance or promote a model. Final events are excluded
    from both the reported cohort and development training populations.
    """
    if mode not in {'pre_weekend','post_qualifying'}:
        raise ValueError('unsupported forecast mode')
    candidates=tuple(candidates)
    if len(set(candidates))!=len(candidates): raise ValueError('duplicate candidate')
    registry=candidate_registry()
    if any(candidate not in registry for candidate in candidates): raise ValueError('unknown candidate')
    rows=list(rows.rows if hasattr(rows,'rows') else rows)
    manifest=dict(version='evaluation-manifest-v1',dataset=_dataset_manifest(rows),
                  dataset_schema_version=DATASET_SCHEMA_VERSION,feature_schema_version=FEATURE_SCHEMA_VERSION,
                  model_schema_version=MODEL_SCHEMA_VERSION,feature_columns=list(FEATURE_COLUMNS),
                  protocol_version=protocol.version,protocol_sha256=_digest(asdict(protocol)),metric_protocol_version=PROTOCOL_VERSION,
                  seed=seed,candidate_configurations={candidate:asdict(TrainingConfig(candidate=candidate,seed=seed)) for candidate in candidates},
                  baseline=dict(candidate='grid' if mode=='post_qualifying' else 'rolling_form',model_version='baseline-v1',
                                window=5 if mode=='pre_weekend' else None,prior='field_midpoint' if mode=='pre_weekend' else None),
                  implementation=_implementation_manifest())
    final=set(protocol.final_events)
    folds=walk_forward_folds([row for row in rows if row['event_id'] not in final],min_train_events=protocol.min_train_events)
    wanted=set(protocol.development_events)
    baseline_name='grid' if mode=='post_qualifying' else 'rolling_form'
    prepared=[];excluded={}
    for fold in folds:
        event_id=fold.test_event_ids[0]
        if event_id not in wanted: continue
        features=[{k:v for k,v in row.items() if k not in {'actual_position','actual_pace_s','outcome_status','status','eligible','outcome_available_at_utc'}} for row in fold.test_rows]
        baseline=grid_baseline(features) if baseline_name=='grid' else rolling_form_baseline(fold.train_rows,features,as_of_utc=features[0]['forecast_cutoff_utc'],prior_position=(len(features)+1)/2)
        if any(row['prediction_score'] is None for row in baseline):
            excluded[event_id]='baseline_full_field_unavailable';continue
        prepared.append((fold,features,baseline))
    result=dict(status='insufficient_data',promotion_allowed=False,protocol=asdict(protocol),mode=mode,evaluation_manifest=manifest,
                cohort_event_ids=[fold.test_event_ids[0] for fold,_,_ in prepared],excluded_events=excluded,
                models={},folds=[],unavailable_candidates={candidate:registry[candidate]['reason'] for candidate in candidates if not registry[candidate]['available']},
                final_holdout='locked',warning='Development evidence only; upstream audit and untouched final evaluation required before promotion.')
    if len(prepared)<protocol.min_development_events:
        return result
    predictions={baseline_name:[]}
    for candidate in candidates:
        if registry[candidate]['available']: predictions[candidate]=[]
    for fold,features,baseline in prepared:
        outputs={baseline_name:baseline}
        models={baseline_name:dict(manifest['baseline'],artifact_status='deterministic_baseline_no_fitted_artifact')}
        for candidate in predictions:
            if candidate==baseline_name:continue
            config=TrainingConfig(candidate=candidate,seed=seed,
                    split_manifest=dict(train_event_ids=list(fold.train_event_ids),test_event_ids=list(fold.test_event_ids)))
            artifact=train_model(fold.train_rows,config)
            outputs[candidate]=predict_model(artifact,features)
            models[candidate]=_model_manifest(artifact,config)
        labels={row['driver_id']:row for row in fold.test_rows}
        for name,output in outputs.items():
            # Attach labels only after inference; every model sees the same entries.
            predictions[name].extend(dict(row,actual_position=labels[row['driver_id']].get('actual_position'),
                                           status=labels[row['driver_id']].get('status',''),
                                           outcome_status=labels[row['driver_id']].get('outcome_status','completed')) for row in output)
        result['folds'].append(dict(train_event_ids=list(fold.train_event_ids),test_event_ids=list(fold.test_event_ids),
                                    train_dataset=_dataset_manifest(fold.train_rows),test_dataset=_dataset_manifest(fold.test_rows),models=models))
    result['models']={name:evaluate_predictions(output,bootstrap_samples=protocol.bootstrap_samples,seed=seed) for name,output in predictions.items()}
    result['status']='evaluated_development'
    return result
