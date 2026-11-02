"""Operator preparation of unscored forecasts; never unlocks the 2026 benchmark."""
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from ml.active_study import ACTIVE_MODELS
from ml.historical_study import _read, _write_json, _time
from ml.foundation_study import context_before

LABEL_COLUMNS = {'actual_position','actual_pace_s','status','outcome_status','eligible','outcome_available_at_utc'}


def inference_rows(rows,event_id,cutoff):
    train = context_before(rows,event_id,cutoff)
    target = [{k:v for k,v in r.items() if k not in LABEL_COLUMNS} for r in rows if r['event_id']==event_id]
    if not target:
        raise ValueError('target event has no frozen pre-race features')
    return train,target


def pending_snapshot(event_id, *, archive_root='.local/archive_ml', now=None):
    from data.snapshots import load_snapshot, save_snapshot
    root=Path(archive_root); manifest=_read(root/'manifest.json')
    pins=[p for p in manifest['snapshots'] if p['event_id']==event_id]
    if len(pins)!=1: raise ValueError('one frozen post-qualifying snapshot required')
    source=load_snapshot(pins[0]['path'])
    now=_time(now or datetime.now(timezone.utc))
    race=next(s for s in source.payloads['schedule'].sessions if s.session_id=='R')
    if now>=race.starts_at_utc:
        raise ValueError('new pending forecasts must be frozen before race start')
    cutoff=min(now,source.as_of_utc)
    if any(s.fetched_at_utc>cutoff for s in source.sources):
        raise ValueError('original archive retrieval must precede prospective cutoff')
    if any(r.session_id not in {'FP1','FP2','FP3','Q','SQ','S'} or r.available_at_utc>cutoff
           for r in source.payloads['driver_sessions']):
        raise ValueError('only completed earlier weekend sessions may enter pending forecast')
    history=[h for h in source.payloads.get('history',()) if h['event_id']!=event_id
             and h.get('outcome_available_at_utc') and _time(h['outcome_available_at_utc'])<cutoff]
    payload=dict(source.payloads,prospective_saved_forecast=True,history=history,
                 source_snapshot_id=source.snapshot_id,forecast_frozen_at_utc=now.isoformat())
    identifier='pending-'+sha256((source.snapshot_id+cutoff.isoformat()).encode()).hexdigest()[:32]
    snapshot=replace(source,snapshot_id=identifier,as_of_utc=cutoff,payloads=payload)
    path=save_snapshot(snapshot,'.local/snapshots')
    return snapshot,path


def build_pending_forecasts(event_id='2026:19', *, owner_id='local', archive_root='.local/archive_ml', include_tabfm=True):
    """Trusted operator CLI only: fitting and optional real TabFM CPU inference occur here."""
    from data.contracts import PredictionRequest
    from data.snapshots import load_snapshot
    from flows.prediction_service import generate_forecast
    from flows.reconstruction import project_completed_sessions
    from ml.archive_workflow import artifact_for_event
    from ml.features import build_features, FEATURE_COLUMNS
    from monitoring.store import TrackingStore
    root=Path(archive_root)
    summary_path=root/('pending-'+event_id.replace(':','-')+'.json')
    previous=_read(summary_path) if summary_path.exists() else None
    if previous and previous.get('owner_id')==owner_id:
        snapshot=load_snapshot(previous['snapshot_path']); snapshot_path=Path(previous['snapshot_path'])
    else:
        snapshot,snapshot_path=pending_snapshot(event_id,archive_root=root)
    request=PredictionRequest(snapshot.event,snapshot.payloads['mode'],snapshot.as_of_utc)
    request.validate_schedule(snapshot.payloads['schedule'])
    store=TrackingStore('.local/tracking.sqlite3')
    summary=dict(event_id=event_id,owner_id=owner_id,snapshot_id=snapshot.snapshot_id,
        snapshot_path=str(snapshot_path.resolve()),as_of_utc=snapshot.as_of_utc.isoformat(),
        prospective_saved_forecast=True,holdout_locked=True,models={})
    saved={r['run_id']:r for r in store._all('forecasts') if r['owner_id']==owner_id}
    for candidate in ACTIVE_MODELS:
        old=(previous or {}).get('models',{}).get(candidate,{})
        if old.get('run_id') in saved and saved[old['run_id']]['snapshot_id']==snapshot.snapshot_id:
            summary['models'][candidate]=old; continue
        try:
            artifact=None; receipt=None
            if candidate=='tabfm':
                if not include_tabfm: raise ValueError('TabFM operator inference has not been requested')
                receipt=_tabfm_receipt(snapshot,root)
            elif candidate in {'ridge','random_forest','gradient_boosting','xgboost'}:
                artifact=artifact_for_event(event_id,candidate,output_root=root)
            result=generate_forecast(request,snapshot,model_id=candidate,artifact=artifact,
                schedule=snapshot.payloads['schedule'],owner_id=owner_id,store=None,
                research_reconstruction=artifact is not None,foundation_receipt=receipt)
            result['provenance'].update(prospective_saved_forecast=True,benchmark_holdout_locked=True,
                forecast_frozen_at_utc=snapshot.payloads['forecast_frozen_at_utc'])
            result['briefing']='Pre-race forecast saved before the Grand Prix. Actual results are pending; no 2026 benchmark accuracy is reported.'
            store.save_forecast(result,is_public=False)
            summary['models'][candidate]=dict(status='ready',run_id=result['run_id'],model_id=result['model_id'],ranking=result['ranking'])
        except (ValueError,RuntimeError,OSError) as error:
            summary['models'][candidate]=dict(status='unavailable',error=str(error))
        root.mkdir(parents=True,exist_ok=True)
        summary_path.write_text(json.dumps(summary,sort_keys=True,indent=2)+'\n',encoding='utf-8')
        print(candidate,summary['models'][candidate]['status'],flush=True)
    return summary


def _tabfm_receipt(snapshot,root):
    from ml.foundation_runtime import TabFMAdapter
    from ml.foundation_study import ensure_memory_headroom
    from flows.foundation_prediction import save_receipt,load_receipt
    from flows.reconstruction import project_completed_sessions
    from ml.features import build_features, FEATURE_COLUMNS
    rows=_read(root/'dataset.json')
    projected=project_completed_sessions(snapshot,snapshot.payloads['schedule'])
    target=list(build_features(projected,projected.payloads['history']).rows)
    context=context_before(rows,snapshot.event.event_id,snapshot.as_of_utc)
    adapter=TabFMAdapter(root='.local/foundation_models')
    try:
        ensure_memory_headroom('tabfm')
        checkpoint=adapter.checkpoint_info()
        output=adapter.predict(context,target,feature_columns=FEATURE_COLUMNS,
            cutoff_utc=snapshot.as_of_utc,target_event_id=snapshot.event.event_id)
        payload=dict(schema='foundation-receipt-v1',candidate='tabfm',event_id=snapshot.event.event_id,
            snapshot_id=snapshot.snapshot_id,as_of_utc=snapshot.as_of_utc.isoformat(),
            cohort_sha256=snapshot.payloads['cohort_sha256'],research_only=True,checkpoint=checkpoint,
            source_ids=sorted(s.source_id for s in snapshot.sources),
            input_sessions=sorted({r.session_id for r in snapshot.payloads['driver_sessions']}),
            predictions=output,details=dict(prospective_saved_forecast=True,training_years='pre-2026',context_limit=100),
            protocol_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
        path=save_receipt(payload)
        return load_receipt(path,snapshot,'tabfm')
    finally:
        adapter.close()


def cached_pending_forecast(event_id,model_id,*,owner_id='local',archive_root='.local/archive_ml'):
    """Owner-scoped numeric forecast for UI/Primus; no fitting, weights or network."""
    from monitoring.store import TrackingStore
    if model_id not in ACTIVE_MODELS or not isinstance(event_id,str) or not __import__('re').fullmatch(r'2026:[1-9][0-9]*',event_id):
        raise ValueError('unsupported pending event or active model')
    path=Path(archive_root)/('pending-'+event_id.replace(':','-')+'.json')
    summary=_read(path) if path.exists() else {}
    item=summary.get('models',{}).get(model_id,{})
    if summary.get('owner_id')!=owner_id or item.get('status')!='ready':
        raise ValueError('no prepared owner-visible forecast for requested model/event')
    result=next((r for r in TrackingStore('.local/tracking.sqlite3')._all('forecasts')
                 if r['run_id']==item['run_id'] and r['owner_id']==owner_id),None)
    if not result or result['snapshot_id']!=summary['snapshot_id']:
        raise ValueError('prepared forecast tracking receipt unavailable')
    return result


def prepared_pending_snapshot(event_id,*,owner_id='local',archive_root='.local/archive_ml'):
    """Read the already frozen operator snapshot without creating a new cutoff."""
    from data.snapshots import load_snapshot
    import re
    if not isinstance(event_id,str) or not re.fullmatch(r'2026:[1-9][0-9]*',event_id):
        raise ValueError('unsupported pending event')
    path=Path(archive_root)/('pending-'+event_id.replace(':','-')+'.json')
    summary=_read(path) if path.exists() else {}
    if summary.get('owner_id')!=owner_id:
        raise ValueError('no prepared owner-visible pending snapshot')
    snapshot=load_snapshot(summary['snapshot_path'])
    if snapshot.event.event_id!=event_id or snapshot.snapshot_id!=summary['snapshot_id']:
        raise ValueError('pending snapshot identity mismatch')
    return snapshot


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--event',default='2026:19');p.add_argument('--without-tabfm',action='store_true')
    a=p.parse_args();print(json.dumps(build_pending_forecasts(a.event,include_tabfm=not a.without_tabfm),indent=2))
