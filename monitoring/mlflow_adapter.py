"""Optional MLflow projection. Canonical results survive tracking outages."""
from __future__ import annotations
from datetime import datetime
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path


def export_evaluation(result, *, tracking_uri=None,client=None,experiment='f1-primus-ai-evaluation'):
    if client is None and not tracking_uri:
        return dict(status='disabled',reason='No explicit tracking target configured')
    # Never export raw prompts, rows, identities or provider payloads.
    payload={key:result[key] for key in ('status','models','protocol','mode','cohort_event_ids','excluded_events','promotion_allowed','final_holdout','evaluation_manifest','folds',
        'version','cohort_sha256','dataset_sha256','implementation_sha256','runtime_versions','seed',
        'historical_publication_verified','holdout_locked','evidence_kind','research_only') if key in result}
    def timestamp(value):
        if isinstance(value,datetime):return value.isoformat()
        raise TypeError('Unsupported tracking metadata')
    encoded=json.dumps(payload,sort_keys=True,allow_nan=False,default=timestamp)
    payload=json.loads(encoded)
    digest=sha256(encoded.encode()).hexdigest()
    try:
        if client is None:
            from mlflow import MlflowClient
            client=MlflowClient(tracking_uri=tracking_uri)
        found=client.get_experiment_by_name(experiment)
        if found is None:
            artifacts=Path('.local/mlflow-artifacts').resolve()
            artifacts.mkdir(parents=True,exist_ok=True)
            experiment_id=client.create_experiment(experiment,artifact_location=artifacts.as_uri())
        else: experiment_id=found.experiment_id
        previous=client.search_runs([experiment_id],filter_string=f"tags.canonical_digest = '{digest}'",max_results=10)
        for run in previous:
            if run.info.status=='FINISHED':return dict(status='complete',run_id=run.info.run_id,replayed=True)
        purpose='historical_reconstruction' if result.get('status')=='evaluated_reconstruction' else 'development_evaluation'
        run=client.create_run(experiment_id,tags={'canonical_digest':digest,'purpose':purpose,'promotion_allowed':'false'})
        run_id=run.info.run_id
        for model,record in payload.get('models',{}).items():
            for metric,value in record.get('metrics',{}).items():
                if isinstance(value,(int,float)) and not isinstance(value,bool) and isfinite(value):
                    client.log_metric(run_id,f'{model}.{metric}',float(value))
        client.log_dict(run_id,payload,'evaluation.json')
        client.set_terminated(run_id,'FINISHED')
        return dict(status='complete',run_id=run_id,replayed=False)
    except Exception:
        return dict(status='unavailable',reason='MLflow export failed; canonical numeric results are retained')
