"""Offline version-pinned archive research. No collectors, network or original-time claims.

2026 is a locked evaluation cohort. Historical labels are reconstructed at session
completion for research only; their original publication time is unverified.
"""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
from statistics import median, pstdev

from data.archive.reader import ArchiveReader
from data.contracts import DataSnapshot, DriverSessionRecord, EventKey, SourceRecord
from data.sessions import EventInfo, SessionRecord
from data.snapshots import save_snapshot
from flows.reconstruction import project_completed_sessions
from ml.features import build_features

DEFAULT_COHORT = '.local/review_cohorts/87cf0ea1a028ed7d738409699e472741e251a9a4b3c952176fb73fab02535c83/manifest.json'
CANDIDATES = ('ridge', 'random_forest', 'gradient_boosting', 'xgboost')

def _dt(ns):
    return datetime.fromtimestamp(ns / 1e9, timezone.utc) if ns is not None else None

def _write(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    encoded=json.dumps(value,sort_keys=True,indent=2,allow_nan=False,default=lambda x:x.isoformat())
    if path.exists() and path.read_text(encoding='utf-8')==encoded:
        return str(path.resolve())
    temporary=path.with_suffix(path.suffix+'.tmp'); temporary.write_text(encoded,encoding='utf-8'); temporary.replace(path)
    return str(path.resolve())

def prepare_archive_workflow(cohort_path=DEFAULT_COHORT, *, output_root='.local/archive_ml', snapshot_root='.local/snapshots'):
    """Read pinned SQL only; save immutable snapshots, separate labels and research rows."""
    cohort_path=Path(cohort_path); raw=cohort_path.read_bytes(); cohort=json.loads(raw)
    cohort_hash=sha256(raw).hexdigest(); root=Path(output_root); reader=ArchiveReader(cohort['archive_roots']['f1'])
    grouped=defaultdict(list)
    for pin in cohort['sessions']:
        grouped[f"{pin['season']}:{pin['round_number']}"].append(pin)
    labels=[]; histories=[]; snapshots=[]; rows=[]; skipped=[]
    with reader._connect() as c:
        c.row_factory=__import__('sqlite3').Row
        # Validate every pin, including labels, before producing anything.
        metadata={}
        for pin in cohort['sessions']:
            found=c.execute("SELECT v.*,s.session_code,s.scheduled_start_utc_ns,s.event_id FROM session_versions v JOIN sessions s USING(session_id) WHERE v.version_id=? AND v.session_id=? AND v.status='published'",(pin['version_id'],pin['session_id'])).fetchone()
            if (found is None or found['dataset_hash']!=pin['dataset_hash']
                    or found['event_id']!=f"{pin['season']}:{pin['round_number']}"
                    or found['session_code']!=pin['session_code']):
                raise ValueError('archive cohort pin no longer matches published data')
            metadata[pin['session_id']]=dict(found)
        for event_id,pins in sorted(grouped.items(), key=lambda p:tuple(map(int,p[0].split(':')))):
            e=dict(c.execute('SELECT * FROM events WHERE event_id=?',(event_id,)).fetchone())
            event=EventKey(e['season'],e['round_number'],e.get('circuit_id'))
            metas=[metadata[p['session_id']] for p in pins]
            race=next((m for m in metas if m['session_code']=='R'),None)
            if race and race['observed_end_utc_ns']:
                end=_dt(race['observed_end_utc_ns']); fetched=datetime.fromisoformat(race['created_at_utc'])
                for r in c.execute('SELECT r.driver_id,r.finishing_position,r.status_raw,e.team_id FROM session_results r JOIN session_entries e USING(version_id,driver_id) WHERE r.version_id=?',(race['version_id'],)):
                    pos=r['finishing_position']; pos=int(pos) if pos is not None and pos>0 and int(pos)==pos else None
                    status=(r['status_raw'] or 'UNKNOWN').upper()
                    if 'DISQUAL' in status: status='DSQ'
                    elif status in {'DID NOT START','WITHDRAWN'}: status='DNS'
                    item=dict(event_id=event_id,driver_id=r['driver_id'],team_id=r['team_id'],circuit_id=event.circuit_id,actual_position=pos,status=status,event_time_utc=(_dt(race['scheduled_start_utc_ns']) or _dt(race['observed_start_utc_ns'])).isoformat(),outcome_available_at_utc=end.isoformat(),original_fetched_at_utc=fetched.isoformat(),historical_publication_verified=False,availability_basis='retrospective_session_completion_assumption',source_id=race['version_id'])
                    labels.append(item); histories.append(item)
        for event_id,pins in sorted(grouped.items(), key=lambda p:tuple(map(int,p[0].split(':')))):
            e=dict(c.execute('SELECT * FROM events WHERE event_id=?',(event_id,)).fetchone()); event=EventKey(e['season'],e['round_number'],e.get('circuit_id'))
            metas=[metadata[p['session_id']] for p in pins]
            q=next((m for m in metas if m['session_code']=='Q'),None)
            race_start=c.execute("SELECT scheduled_start_utc_ns FROM sessions WHERE event_id=? AND session_code='R'",(event_id,)).fetchone()
            if not q or not q['observed_end_utc_ns'] or not race_start or race_start[0] is None:
                skipped.append(dict(event_id=event_id,reason='qualifying completion or race schedule unavailable')); continue
            cutoff=_dt(race_start[0])-timedelta(seconds=1)
            eligible=[m for m in metas if m['session_code'] in {'FP1','FP2','FP3','Q','SQ','S'} and m['observed_end_utc_ns'] is not None and _dt(m['observed_end_utc_ns'])<=cutoff]
            if q not in eligible:
                skipped.append(dict(event_id=event_id,reason='qualifying not completed before race')); continue
            sources=[]; records=[]; sessions=[]; entries={}
            # Qualifying's evidenced roster is known before R. Practice reserves,
            # archive metadata-only identities and R starters cannot define it.
            roster=list(c.execute('''SELECT e.*,d.display_name FROM session_entries e
                JOIN drivers d USING(driver_id) LEFT JOIN session_results r USING(version_id,driver_id)
                WHERE e.version_id=? AND (r.finishing_position>0 OR r.q1_ns>0 OR r.q2_ns>0 OR r.q3_ns>0
                OR EXISTS(SELECT 1 FROM laps l WHERE l.version_id=e.version_id AND l.driver_id=e.driver_id
                    AND (l.lap_time_ns>0 OR (l.lap_start_clock_ns IS NOT NULL AND l.lap_end_clock_ns IS NOT NULL))))''',(q['version_id'],)))
            for ent in roster:
                entries[ent['driver_id']]=dict(driver_id=ent['driver_id'],team_id=ent['team_id'],display_name=ent['display_name'])
            for m in sorted(eligible,key=lambda m:m['observed_end_utc_ns']):
                end=_dt(m['observed_end_utc_ns']); start=_dt(m['scheduled_start_utc_ns']) or _dt(m['observed_start_utc_ns'])
                if start>end: start=_dt(m['observed_start_utc_ns'])
                sessions.append(SessionRecord(event,m['session_code'],m['session_code'],start,end))
                source=SourceRecord('local_archive',m['version_id'],datetime.fromisoformat(m['created_at_utc']),original_timestamps=dict(session_id=m['session_id'],dataset_hash=m['dataset_hash'],availability_basis='original_publication_unknown'))
                sources.append(source)
                participating=list(c.execute('SELECT e.*,d.display_name FROM session_entries e JOIN drivers d USING(driver_id) WHERE version_id=? AND EXISTS(SELECT 1 FROM laps l WHERE l.version_id=e.version_id AND l.driver_id=e.driver_id)',(m['version_id'],)))
                clean=defaultdict(list)
                for lap in c.execute("SELECT driver_id,lap_time_ns,tyre_age_laps,compound_normalized FROM laps WHERE version_id=? AND lap_number>1 AND lap_time_ns>0 AND provider_is_accurate=1 AND coalesce(deleted,0)=0 AND pit_in_clock_ns IS NULL AND pit_out_clock_ns IS NULL AND track_status_raw='1'",(m['version_id'],)):
                    clean[lap['driver_id']].append(dict(lap))
                reference=min((median(l['lap_time_ns']/1e9 for l in laps) for laps in clean.values()),default=None)
                result_map={r['driver_id']:dict(r) for r in c.execute('SELECT driver_id,finishing_position,q1_ns,q2_ns,q3_ns FROM session_results WHERE version_id=?',(m['version_id'],))}
                best={d:min(v for v in (r['q1_ns'],r['q2_ns'],r['q3_ns']) if v is not None) for d,r in result_map.items() if any(r[k] is not None for k in ('q1_ns','q2_ns','q3_ns'))}
                for ent in participating:
                    driver=ent['driver_id']; values={}; laps=clean.get(driver,[])
                    if m['session_code'].startswith('FP') and laps:
                        times=[l['lap_time_ns']/1e9 for l in laps]
                        values=dict(practice_pace_delta_s=median(times)-reference,practice_variability_s=pstdev(times),practice_laps=len(times),tyre_age_laps=laps[-1]['tyre_age_laps'],compound=laps[-1]['compound_normalized'])
                    elif m['session_code']=='Q':
                        r=result_map.get(driver,{})
                        position=r.get('finishing_position')
                        if position is not None and position>0: values['qualifying_position']=position
                        if driver in best: values['qualifying_gap_s']=(best[driver]-min(best.values()))/1e9
                    elif m['session_code']=='S':
                        values['sprint_position']=result_map.get(driver,{}).get('finishing_position')
                    records.append(DriverSessionRecord(event,m['session_code'],driver,end,values,source))
            if not entries:
                skipped.append(dict(event_id=event_id,reason='no earlier-session lap participants')); continue
            sessions.append(SessionRecord(event,'R','Race',_dt(race_start[0])))
            schedule=EventInfo(event,e['event_name'] or event_id,e.get('location') or '',tuple(sessions))
            past=[h for h in histories if h['event_id']!=event_id and h['outcome_available_at_utc']<cutoff.isoformat()]
            sid='archive-'+sha256(f'{cohort_hash}:{event_id}:archive-features-v2'.encode()).hexdigest()[:32]
            snapshot=DataSnapshot(sid,event,cutoff,dict(mode='post_qualifying',historical_reconstruction=True,cohort_sha256=cohort_hash,entries=list(entries.values()),entries_source=next(s for s in sources if s.source_id==q['version_id']),driver_sessions=records,history=past,schedule=schedule),sources,['Retrospective archive; original publication timing unverified','Partial session coverage is retained; missing feature values stay unknown'])
            path=save_snapshot(snapshot,snapshot_root); snapshots.append(dict(event_id=event_id,path=str(path.resolve()),snapshot_id=sid))
            projected=project_completed_sessions(snapshot,schedule)
            target={l['driver_id']:l for l in labels if l['event_id']==event_id}
            for feature in build_features(projected,past).rows:
                label=target.get(feature['driver_id'],{})
                rows.append(dict(feature,snapshot_id=sid,actual_position=label.get('actual_position'),status=label.get('status','PENDING'),event_time_utc=label.get('event_time_utc'),outcome_available_at_utc=label.get('outcome_available_at_utc'),outcome_status='completed' if label else 'pending',eligible=bool(label.get('actual_position') and label.get('status') not in {'DNS','DSQ'}),historical_publication_verified=False))
    manifest=dict(version='archive-workflow-v1',cohort_sha256=cohort_hash,cohort_path=str(cohort_path.resolve()),snapshots=snapshots,skipped=skipped,session_count=len(cohort['sessions']),row_count=len(rows),historical_publication_verified=False,availability_basis='retrospective_session_completion_assumption',holdout_locked=True,holdout_year=2026)
    manifest['dataset_path']=_write(root/'dataset.json',rows); manifest['labels_path']=_write(root/'labels.json',labels)
    _write(root/'manifest.json',manifest)
    return manifest


def chronological_folds(rows, *, min_train_events=12):
    """Whole-event expanding development folds; 2026 is never read for evaluation."""
    groups=defaultdict(list)
    for r in rows:
        if int(r['event_id'].split(':')[0])<2026:
            groups[r['event_id']].append(r)
    ordered=sorted(groups,key=lambda e:groups[e][0]['forecast_cutoff_utc']); folds=[]
    for event in ordered:
        test=groups[event]; cutoff=test[0]['forecast_cutoff_utc']
        train=[r for prev in ordered for r in groups[prev] if prev!=event and r.get('outcome_available_at_utc') and r.get('event_time_utc') and r['outcome_available_at_utc']<cutoff and r['event_time_utc']<cutoff]
        if len({r['event_id'] for r in train})>=min_train_events: folds.append((train,test))
    return folds


def matching_field(predictions, outcomes):
    """A common race metric needs the full recorded field in both directions."""
    return bool(predictions) and {r['driver_id'] for r in predictions} == {r['driver_id'] for r in outcomes}


def run_archive_benchmark(*, output_root='.local/archive_ml', min_train_events=12):
    import sklearn,numpy,pandas
    from ml.training import train_model,predict_model,TrainingConfig
    from ml.evaluation import evaluate_predictions
    from ml.protocols import EvaluationProtocol,freeze_protocol
    root=Path(output_root); manifest=json.loads((root/'manifest.json').read_text()); rows=json.loads((root/'dataset.json').read_text())
    labels_by_event=defaultdict(list)
    for row in json.loads((root/'labels.json').read_text()): labels_by_event[row['event_id']].append(row)
    freeze_time_path=root/'protocol_frozen_at-v4.json'
    if not freeze_time_path.exists():
        _write(freeze_time_path,dict(cohort_sha256=manifest['cohort_sha256'],frozen_at_utc=datetime.now(timezone.utc).isoformat()))
    freeze_time=json.loads(freeze_time_path.read_text())
    if freeze_time['cohort_sha256']!=manifest['cohort_sha256']:
        raise ValueError('new cohort requires a separate benchmark output directory')
    protocol_path=freeze_protocol(EvaluationProtocol(tuple(sorted({r['event_id'] for r in rows if int(r['event_id'].split(':')[0])<2026})),tuple(sorted({r['event_id'] for r in rows if int(r['event_id'].split(':')[0])==2026})),datetime.fromisoformat(freeze_time['frozen_at_utc']),version='archive-full-field-ranking-v4',baseline='qualifying',min_train_events=min_train_events,bootstrap_samples=200),root/'protocols')
    predictions=defaultdict(list); fold_manifest=[]; exclusions=[]
    for train,test in chronological_folds(rows,min_train_events=min_train_events):
        if not matching_field(test, labels_by_event[test[0]['event_id']]):
            predicted={r['driver_id'] for r in test}; observed={r['driver_id'] for r in labels_by_event[test[0]['event_id']]}
            exclusions.append(dict(event_id=test[0]['event_id'],reason='qualifying/race roster mismatch; no partial intersection scoring',missing_predictions=sorted(observed-predicted),missing_labels=sorted(predicted-observed))); continue
        if any(r.get('qualifying_position') is None for r in test):
            exclusions.append(dict(event_id=test[0]['event_id'],reason='missing qualifying position; common benchmark cohort excludes whole event for all models')); continue
        if not all(r.get('actual_position') for r in test if r['status'] not in {'DNS','DSQ'}):
            exclusions.append(dict(event_id=test[0]['event_id'],reason='qualifying roster differs from known race label field; no target-R roster substitution',missing_labels=[r['driver_id'] for r in test if not r.get('actual_position') and r['status'] not in {'DNS','DSQ'}])); continue
        train_ids=sorted({r['event_id'] for r in train}); event=test[0]['event_id']
        split=dict(train_event_ids=train_ids,test_event_ids=[event]); fold_manifest.append(split)
        for candidate in CANDIDATES:
            artifact=train_model(train,TrainingConfig(candidate=candidate,split_manifest=split))
            predictions[candidate].extend(predict_model(artifact,test))
        for baseline,column in [('qualifying','qualifying_position'),('rolling_form','driver_form_position')]:
            predictions[baseline].extend(dict(r,prediction_score=r.get(column) if r.get(column) is not None else (len(test)+1)/2) for r in test)
    report=dict(version='archive-development-v4',status='evaluated_reconstruction',evidence_kind='historical_reconstruction',cohort_sha256=manifest['cohort_sha256'],historical_publication_verified=False,research_only=True,holdout_locked=True,final_holdout='2026 locked; not evaluated',final_events=sorted({r['event_id'] for r in rows if int(r['event_id'].split(':')[0])==2026}),cohort_event_ids=[f['test_event_ids'][0] for f in fold_manifest],folds=fold_manifest,excluded_events=exclusions,protocol_path=str(protocol_path.resolve()),models={k:evaluate_predictions(v,bootstrap_samples=200,seed=42) for k,v in predictions.items()},readiness=dict(reasons=['Reconstructed publication times; research-only chronological development evidence. No promotion or live accuracy claim.']))
    report.update(seed=42,dataset_sha256=sha256((root/'dataset.json').read_bytes()).hexdigest(),
                  runtime_versions=dict(sklearn=sklearn.__version__,numpy=numpy.__version__,pandas=pandas.__version__),
                  implementation_sha256={str(p):sha256(p.read_bytes()).hexdigest() for p in map(Path,('ml/archive_workflow.py','ml/training.py','ml/features.py','ml/evaluation.py','ml/baselines.py','flows/prediction_service.py','flows/reconstruction.py'))})
    _write(root/'benchmark_predictions.json',dict(predictions)); _write(root/'benchmark.json',report)
    _write(Path('.local/evaluations')/f"evaluation-archive-{manifest['cohort_sha256'][:24]}-v4.json",report)
    return report


def artifact_for_event(event_id,candidate, *, output_root='.local/archive_ml', model_root='.local/models'):
    """Train only pre-event pre-2026 labelled rows; reuse versioned local artifacts."""
    from ml.training import train_model,TrainingConfig
    from ml.artifacts import ArtifactStore
    if candidate not in CANDIDATES: raise ValueError('unsupported archive candidate')
    root=Path(output_root); manifest=json.loads((root/'manifest.json').read_text()); rows=json.loads((root/'dataset.json').read_text())
    target=[r for r in rows if r['event_id']==event_id]
    if not target: raise ValueError('event has no frozen archive snapshot')
    cutoff=target[0]['forecast_cutoff_utc']
    train=[r for r in rows if int(r['event_id'].split(':')[0])<2026 and r['event_id']!=event_id and r.get('outcome_available_at_utc') and r['outcome_available_at_utc']<cutoff and r['event_time_utc']<cutoff]
    if len({r['event_id'] for r in train})<12: raise ValueError('at least 12 prior development events required')
    key=sha256(json.dumps(dict(event=event_id,candidate=candidate,cohort=manifest['cohort_sha256'],version='archive-v2'),sort_keys=True).encode()).hexdigest()
    index_path=root/'model_index.json'; index=json.loads(index_path.read_text()) if index_path.exists() else {}; store=ArtifactStore(model_root)
    if key in index: return store.load(index[key])
    artifact=train_model(train,TrainingConfig(candidate=candidate,model_version='archive-v2',split_manifest=dict(train_event_ids=sorted({r['event_id'] for r in train}),test_event_ids=[event_id])))
    artifact.metadata.update(research_reconstruction_only=True,historical_publication_verified=False,availability_basis='retrospective_session_completion_assumption',cohort_sha256=manifest['cohort_sha256'])
    store.save(artifact)
    _write(Path(model_root)/f'{artifact.model_id}.json',dict(model_id=artifact.model_id,candidate=artifact.candidate,target=artifact.target,model_version=artifact.model_version,metadata=artifact.metadata,artifact_sha256=sha256((Path(model_root)/f'{artifact.model_id}.joblib').read_bytes()).hexdigest()))
    index[key]=artifact.model_id; _write(index_path,index)
    return artifact


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--cohort',default=DEFAULT_COHORT); p.add_argument('--output-root',default='.local/archive_ml'); p.add_argument('--benchmark',action='store_true'); p.add_argument('--event'); a=p.parse_args()
    manifest=prepare_archive_workflow(a.cohort,output_root=a.output_root)
    print(json.dumps({k:manifest[k] for k in ('row_count','session_count','holdout_locked')}))
    if a.benchmark:
        report=run_archive_benchmark(output_root=a.output_root); print(json.dumps(dict(development_folds=len(report['folds']),models=list(report['models']))))
    if a.event:
        for candidate in CANDIDATES: print(artifact_for_event(a.event,candidate,output_root=a.output_root).model_id)

if __name__=='__main__': main()
