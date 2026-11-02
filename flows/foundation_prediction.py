"""Trusted operator-computed challenger receipts; no backend loading on UI requests."""
from __future__ import annotations
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path


def _digest(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def save_receipt(payload, *, root='.local/foundation_models/predictions'):
    value=dict(payload);value.pop('receipt_id',None)
    digest=_digest(value);value['receipt_id']=digest
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    path=root/f'{digest}.json'
    encoded=json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n'
    if path.exists():
        if path.read_text(encoding='utf-8')!=encoded:raise ValueError('immutable foundation receipt conflict')
    else:
        with path.open('x',encoding='utf-8') as f:f.write(encoded)
    return path


def validate_receipt(value,snapshot,candidate):
    from ml.foundation_runtime import REVISIONS
    if not isinstance(value,dict):raise ValueError('foundation receipt required')
    content=dict(value);digest=content.pop('receipt_id',None)
    if digest!=_digest(content):raise ValueError('foundation receipt digest mismatch')
    if (value.get('schema')!='foundation-receipt-v1' or value.get('candidate')!=candidate
            or candidate not in REVISIONS or value.get('research_only') is not True
            or value.get('checkpoint',{}).get('revision')!=REVISIONS[candidate]
            or value.get('event_id')!=snapshot.event.event_id or value.get('snapshot_id')!=snapshot.snapshot_id
            or value.get('as_of_utc')!=snapshot.as_of_utc.isoformat()
            or value.get('cohort_sha256')!=snapshot.payloads.get('cohort_sha256')):
        raise ValueError('foundation receipt must match exact research snapshot and checkpoint')
    source_ids=value.get('source_ids',[])
    if not source_ids or not set(source_ids)<={s.source_id for s in snapshot.sources}:
        raise ValueError('foundation sources must belong to pinned pre-race snapshot')
    rows=value.get('predictions',[])
    ids=[r['driver_id'] for r in rows]
    if (not rows or len(ids)!=len(set(ids))
            or set(ids)!={e['driver_id'] for e in snapshot.payloads.get('entries',[])}):
        raise ValueError('foundation prediction needs complete unique driver coverage')
    for row in rows:
        score=row.get('prediction_score')
        if row.get('event_id')!=snapshot.event.event_id or isinstance(score,bool) or not isinstance(score,(int,float)) or not isfinite(score):
            raise ValueError('foundation output must contain finite scores for selected event')
    return value


def load_receipt(path,snapshot,candidate):
    path=Path(path)
    if path.is_symlink() or path.stat().st_size>500_000:raise ValueError('invalid foundation receipt file')
    return validate_receipt(json.loads(path.read_text(encoding='utf-8')),snapshot,candidate)


def foundation_forecast(request,snapshot,receipt,*,model_id,owner_id='local',store=None,history=None):
    from flows.prediction_service import forecast_run_id
    from flows.reporting import forecast_report
    if snapshot.payloads.get('historical_reconstruction') is not True:
        raise ValueError('foundation challengers are restricted to explicit historical research')
    receipt=validate_receipt(receipt,snapshot,model_id)
    ordered=sorted(receipt['predictions'],key=lambda r:(r['prediction_score'],r['driver_id']))
    ranking=[dict(driver_id=r['driver_id'],predicted_rank=i,prediction_score=r['prediction_score'])
             for i,r in enumerate(ordered,1)]
    pace=model_id=='timesfm'
    input_sessions=receipt.get('input_sessions',[])
    if pace and receipt.get('details',{}).get('pace_contexts'):
        input_sessions=sorted({r['session_code'] for r in receipt['details']['pace_contexts']})
    # Event subsets/retry attempts change run provenance, not the fitted model.
    model_version=_digest(dict(candidate=model_id,checkpoint=receipt['checkpoint'],
        cohort_sha256=receipt.get('cohort_sha256'),inference_parameters=dict(
            context_rows=100 if model_id=='tabfm' else None,horizon=5 if pace else None,
            minimum_clean_laps=3 if pace else None)))
    result=dict(run_id=forecast_run_id(request,snapshot,model_id=model_id,owner_id=owner_id,foundation_receipt=receipt,history=history),
        event_id=request.event.event_id,season=request.event.season,round_number=request.event.round_number,
        owner_id=owner_id,model_id=model_id+'-'+model_version[:16],mode=request.mode.value,
        as_of_utc=request.as_of_utc.isoformat(),snapshot_id=snapshot.snapshot_id,ranking=ranking,
        metrics={},status='research',artifacts={},message_count=0,
        provenance=dict(validation_status='historical_reconstruction',historical_publication_verified=False,
            target_session='R',research_reconstruction=True,source_ids=receipt['source_ids'],
            foundation_receipt_id=receipt['receipt_id'],checkpoint=receipt['checkpoint'],
            foundation_run_schema='foundation-forecast-v2',
            score_units='seconds' if pace else 'ordinal_position',
            input_sessions=input_sessions),
        warnings=['Operator-computed frozen research inference; original publication timing is unverified.',
                  'Pretrained data overlap is unknown; this is not a live or uncontaminated backtest.',
                  'Pace ordering does not simulate pit strategy, reliability or calibrated win odds.' if pace else 'Tabular scores are uncalibrated position estimates.'],
        briefing='TimesFM forecasts next-lap pace from completed pre-race runs; ordering is an experimental pace heuristic.' if pace else 'TabFM estimates finishing position from earlier labelled races and pre-Grand-Prix tabular features.')
    result['report']=forecast_report(result)
    if store is not None:store.save_forecast(result,is_public=False)
    return result
