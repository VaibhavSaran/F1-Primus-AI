"""Resumable operator-only foundation inference on frozen development races."""
from __future__ import annotations
from datetime import datetime
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from time import perf_counter
import shutil

from ml.historical_study import _read,_hash,_time,_write_json,pre_race_pace_rows


def context_before(rows,target_event_id,cutoff_utc):
    cutoff=_time(cutoff_utc)
    return [r for r in rows if r['event_id']!=target_event_id and int(r['event_id'].split(':')[0])<2026
            and r.get('status','').upper() not in {'DNS','DSQ','PENDING'}
            and isinstance(r.get('actual_position'),(int,float)) and not isinstance(r['actual_position'],bool)
            and isfinite(r['actual_position']) and r['actual_position']>0
            and r.get('outcome_available_at_utc') and _time(r['outcome_available_at_utc'])<cutoff
            and r.get('event_time_utc') and _time(r['event_time_utc'])<cutoff]


def validate_lineage(manifest,benchmark,snapshot=None,pin=None):
    if manifest['cohort_sha256']!=benchmark['cohort_sha256']:
        raise ValueError('benchmark and snapshot manifest cohorts differ')
    if snapshot is not None and (snapshot.snapshot_id!=pin['snapshot_id']
            or snapshot.event.event_id!=pin['event_id']
            or snapshot.payloads.get('cohort_sha256')!=manifest['cohort_sha256']):
        raise ValueError('snapshot identity or cohort differs from manifest pin')


def attempt_directory(root,candidate,run_id,attempt):
    if type(attempt) is not int or not 0<=attempt<=99:
        raise ValueError('attempt must be an integer between zero and 99')
    return Path(root)/(candidate+'-'+run_id+(f'-attempt-{attempt}' if attempt else ''))


def ensure_memory_headroom(candidate):
    import psutil
    trim=None
    if candidate=='tabfm' and psutil.virtual_memory().available<750_000_000:
        from ml.foundation_runtime import trim_cpu_working_set
        trim=trim_cpu_working_set()
    if psutil.virtual_memory().available<350_000_000:
        raise RuntimeError('insufficient host memory headroom; use a new --attempt after closing unused applications')
    return trim


def run_foundation_study(candidate,*,archive_root='.local/archive_ml',output_root='.local/foundation_evaluations',
                         model_root='.local/foundation_models',event_ids=None,attempt=0):
    from data.snapshots import load_snapshot
    from flows.foundation_prediction import save_receipt,load_receipt
    from ml.features import FEATURE_COLUMNS
    from ml.foundation_runtime import TimesFMAdapter,TabFMAdapter,PaceSeries
    if candidate not in {'tabfm','timesfm'}:raise ValueError('unknown foundation candidate')
    root=Path(archive_root);manifest=_read(root/'manifest.json');benchmark=_read(root/'benchmark.json')
    validate_lineage(manifest,benchmark)
    rows=_read(root/'dataset.json')
    if _hash(root/'dataset.json')!=benchmark['dataset_sha256']:raise ValueError('dataset hash mismatch')
    events=benchmark['cohort_event_ids']
    if event_ids is not None:
        if not set(event_ids)<=set(events):raise ValueError('requested races must belong to frozen development cohort')
        events=[e for e in events if e in event_ids]
    if any(int(e.split(':')[0])>=2026 for e in events):raise ValueError('2026 evaluation holdout locked')
    # Execute the requested case first, then the remaining races chronologically.
    if '2025:19' in events:events=['2025:19',*[e for e in events if e!='2025:19']]
    adapter=(TimesFMAdapter if candidate=='timesfm' else TabFMAdapter)(root=model_root)
    checkpoint=adapter.checkpoint_info()
    protocol=dict(version='foundation-development-v1',candidate=candidate,cohort_sha256=manifest['cohort_sha256'],
        checkpoint=checkpoint,events=events,seed=42,context_limit=100,pace_min_laps=3,horizon=5,
        score_interpretation='median next-five-lap pace heuristic' if candidate=='timesfm' else 'tabular finishing-position estimate',
        implementation_sha256={name:_hash(name) for name in ('ml/foundation_study.py','ml/foundation_runtime.py','ml/historical_study.py')},
        dataset_sha256=benchmark['dataset_sha256'],historical_publication_verified=False,holdout_locked=True)
    run_id=sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    dest=attempt_directory(output_root,candidate,run_id,attempt);dest.mkdir(parents=True,exist_ok=True)
    _write_json(dest/'protocol.json',protocol)
    for name,digest in protocol['implementation_sha256'].items():
        source=dest/'source'/name;source.parent.mkdir(parents=True,exist_ok=True)
        if source.exists():
            if _hash(source)!=digest:raise ValueError('frozen implementation copy changed')
        else:shutil.copyfile(name,source)
    pins={p['event_id']:p for p in manifest['snapshots']}
    predictions,dispositions={},[]
    started=perf_counter()
    try:
        for index,event in enumerate(events,1):
            target=[r for r in rows if r['event_id']==event]
            snapshot=load_snapshot(pins[event]['path'])
            validate_lineage(manifest,benchmark,snapshot,pins[event])
            done=dest/(event.replace(':','-')+'.json')
            if done.exists():
                record=_read(done)
                if record['status']=='completed':
                    value=load_receipt(record['receipt_path'],snapshot,candidate)
                    predictions[event]=value['predictions']
                dispositions.append(record);continue
            tick=perf_counter()
            print(f'{candidate} {index}/{len(events)} {event}',flush=True)
            try:
                if not checkpoint.get('available'):
                    raise RuntimeError('checkpoint unavailable: '+checkpoint.get('reason','unknown'))
                trim=ensure_memory_headroom(candidate)
                sources=sorted(s.source_id for s in snapshot.sources)
                details={}
                if candidate=='tabfm':
                    context=context_before(rows,event,snapshot.as_of_utc)
                    output=adapter.predict(context,target,feature_columns=FEATURE_COLUMNS,
                        cutoff_utc=snapshot.as_of_utc,target_event_id=event)
                    context=sorted(context,key=lambda r:(_time(r['outcome_available_at_utc']),r['event_id'],r['driver_id']))[-100:]
                    details=dict(context_rows=len(context),context_event_ids=sorted({r['event_id'] for r in context}))
                else:
                    pace=pre_race_pace_rows(snapshot,manifest['cohort_path'])
                    field={r['driver_id'] for r in target}
                    pace=[p for p in pace if p['driver_id'] in field]
                    selected=[]
                    for driver in sorted(field):
                        eligible=[p for p in pace if p['driver_id']==driver]
                        if not eligible:raise ValueError('incomplete clean pace field: '+driver)
                        selected.append(max(eligible,key=lambda p:(_time(p['completed_at_utc']),len(p['values']),p['run_number'])))
                    series=[PaceSeries(p['driver_id'],p['event_id'],p['session_code'],p['completed_at_utc'],tuple(p['values'])) for p in selected]
                    output=adapter.predict(series,cutoff_utc=snapshot.as_of_utc,target_event_id=event,
                        driver_ids=[r['driver_id'] for r in target],horizon=5)
                    sources=sorted({p['source_id'] for p in selected})
                    details=dict(pace_contexts=[dict(driver_id=p['driver_id'],session_code=p['session_code'],
                        n_laps=len(p['values']),source_id=p['source_id'],compound=p['compound'],
                        stint_number=p['stint_number'],run_number=p['run_number']) for p in selected],
                        short_context_warning='Minimum three clean consecutive laps; short sequences are padded upstream. Cross-session pace is not a full strategy forecast.')
                payload=dict(schema='foundation-receipt-v1',candidate=candidate,event_id=event,
                    snapshot_id=snapshot.snapshot_id,as_of_utc=snapshot.as_of_utc.isoformat(),
                    cohort_sha256=manifest['cohort_sha256'],research_only=True,checkpoint=checkpoint,
                    source_ids=sources,input_sessions=sorted({p['session_code'] for p in selected}) if candidate=='timesfm' else
                    sorted({r.session_id for r in snapshot.payloads['driver_sessions'] if r.session_id.startswith('FP') or r.session_id=='Q'}),
                    predictions=output,details=details,protocol_sha256=run_id)
                path=save_receipt(payload,root=Path(model_root)/'predictions')
                load_receipt(path,snapshot,candidate)
                record=dict(event_id=event,status='completed',elapsed_seconds=perf_counter()-tick,
                    receipt_path=str(path.resolve()),receipt_sha256=_hash(path),memory_trim=trim)
                predictions[event]=output
            except (ValueError,RuntimeError,OSError) as error:
                # Keep exact missing coverage or dependency failures; never substitute a baseline.
                record=dict(event_id=event,status='unavailable',reason=str(error),elapsed_seconds=perf_counter()-tick)
            _write_json(done,record);dispositions.append(record)
            print(record['status'],record.get('reason',''),f"{record['elapsed_seconds']:.2f}s",flush=True)
            progress=dict(candidate=candidate,attempted=len(dispositions),requested=len(events),
                completed=len(predictions),elapsed_seconds=perf_counter()-started)
            (dest/'progress.json').write_text(json.dumps(progress,indent=2)+'\n',encoding='utf-8')
    finally:
        adapter.close()
    _write_json(dest/'predictions.json',{candidate:predictions} if predictions else {})
    _write_json(dest/'metadata.json',{candidate:dict(protocol=protocol,dispositions=dispositions,
        completed_events=len(predictions),requested_events=len(events),
        note='Exact-field coverage only; no target Grand Prix laps, labels or future measured weather enter inference.')})
    print(str(dest.resolve()),flush=True)
    return dest


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('candidate',choices=['tabfm','timesfm']);p.add_argument('--event',action='append')
    p.add_argument('--attempt',type=int,default=0,help='New immutable retry namespace; retains earlier unavailable evidence')
    a=p.parse_args();run_foundation_study(a.candidate,event_ids=a.event,attempt=a.attempt)
