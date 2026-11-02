"""Operator-only local projection of archive research runs into tracking/MLflow.

No provider requests or training occur. 2026 remains a locked evaluation holdout.
"""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

from data.contracts import PredictionRequest
from data.snapshots import load_snapshot
from flows.prediction_service import generate_forecast
from ml.artifacts import ArtifactStore, MODEL_ID
from monitoring.store import TrackingStore

ROOT = Path(__file__).resolve().parents[1]


def archive_outcome(event_id, labels):
    if int(event_id.split(':')[0]) >= 2026:
        raise ValueError('The 2026 holdout is locked; no automatic outcome scoring')
    selected = sorted((row for row in labels if row['event_id'] == event_id), key=lambda row: row['driver_id'])
    if not selected:
        raise ValueError('No archive outcome labels exist for this event')
    revision = sha256(json.dumps(selected, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return dict(event_id=event_id, revision_id='archive-' + revision, status='official',
        published_at_utc=max(row['original_fetched_at_utc'] for row in selected),
        historical_publication_verified=False, availability_basis='saved_archive_retrieval',
        source_versions=sorted({row['source_id'] for row in selected}),
        entries=[dict(driver_id=row['driver_id'], actual_position=row['actual_position'], status=row['status']) for row in selected])


def _artifact(root, snapshot, candidate):
    choices = []
    for path in (root / 'models').glob(candidate + '-finishing_position-*.json'):
        if path.is_symlink() or not MODEL_ID.fullmatch(path.stem):
            continue
        record = json.loads(path.read_text(encoding='utf-8'))
        meta = record['metadata']
        stamp = datetime.fromisoformat(meta['training_max_event_time_utc'])
        if (record['model_id'] == path.stem and meta.get('research_reconstruction_only') is True
                and meta.get('cohort_sha256') == snapshot.payloads.get('cohort_sha256')
                and snapshot.event.event_id not in meta.get('training_event_ids', []) and stamp < snapshot.as_of_utc):
            choices.append((stamp, path.stem, record))
    if not choices:
        return None
    _, identifier, record = max(choices)
    binary = root / 'models' / (identifier + '.joblib')
    if sha256(binary.read_bytes()).hexdigest() != record['artifact_sha256']:
        raise ValueError('Model artifact digest mismatch')
    return ArtifactStore(root / 'models').load(identifier)


def seed_archive_tracking(project=ROOT, *, n_events=5, export_mlflow=True):
    """Record measured retrospective research history from common development folds."""
    project = Path(project).resolve()
    local = project / '.local'
    workflow = local / 'archive_ml'
    manifest = json.loads((workflow / 'manifest.json').read_text(encoding='utf-8'))
    evaluation = json.loads((workflow / 'benchmark.json').read_text(encoding='utf-8'))
    labels = json.loads((workflow / 'labels.json').read_text(encoding='utf-8'))
    snapshot_paths = {row['event_id']: row['path'] for row in manifest['snapshots']}
    events = evaluation['cohort_event_ids'][-n_events:]
    store = TrackingStore(local / 'tracking.sqlite3')
    existing = {row['run_id'] for row in store._all('forecasts')}
    runs, scored = [], []
    for event_id in events:
        snapshot = load_snapshot(snapshot_paths[event_id])
        request = PredictionRequest(snapshot.event, snapshot.payloads['mode'], snapshot.as_of_utc)
        store.append_outcome(archive_outcome(event_id, labels))
        for candidate in ('qualifying', 'rolling_form', 'weekend_form', 'ridge', 'random_forest', 'gradient_boosting'):
            artifact = _artifact(local, snapshot, candidate) if candidate in {'ridge', 'random_forest', 'gradient_boosting'} else None
            if candidate in {'ridge', 'random_forest', 'gradient_boosting'} and artifact is None:
                continue
            started = perf_counter()
            result = generate_forecast(request, snapshot, model_id=candidate, artifact=artifact,
                schedule=snapshot.payloads['schedule'], owner_id='local', store=store,
                research_reconstruction=bool(artifact))
            if not result['ranking']:
                continue
            runs.append(result['run_id'])
            score = store.score_forecast(result['run_id'])
            if score['evaluation']['outcome_status'] == 'completed':
                scored.append(result['run_id'])
            if result['run_id'] not in existing:
                store.record_operational_event(dict(event_id='batch-' + result['run_id'], owner_id='local',
                    run_id=result['run_id'], model_id=result['model_id'], status='success',
                    occurred_at_utc=datetime.now(timezone.utc).isoformat(), latency_ms=(perf_counter() - started) * 1000,
                    input_tokens=0, output_tokens=0, tool_calls=0, estimated_cost=0, billed_cost=0, cache_hit=False))
    projection = dict(status='disabled')
    if export_mlflow:
        from monitoring.mlflow_adapter import export_evaluation
        projection = export_evaluation(evaluation, tracking_uri='sqlite:///' + str((local / 'mlflow.sqlite3').resolve()).replace('\\', '/'))
    report = dict(events=events, saved_runs=len(set(runs)), scored_runs=len(set(scored)),
        holdout_locked=True, evidence_kind='historical_reconstruction', mlflow=projection)
    (workflow / 'tracking_projection.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    print(json.dumps(seed_archive_tracking(), indent=2))
