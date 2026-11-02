"""Active-model comparison derived from immutable receipts and fold-local XGBoost."""
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

from ml.evaluation import evaluate_predictions
from ml.historical_study import _read, _hash, _write_json, join_outcomes, paired_difference

ACTIVE_MODELS = ('qualifying','rolling_form','weekend_form','ridge','random_forest','gradient_boosting','xgboost','tabfm')
SOURCE_STUDY = '.local/model_studies/691db355d012e9a7e386d556347a736b80fb724cbf155d88de08e99dbc2f628a'


def nested_benchmarks(predictions, events):
    """Latest 100 eligible chronological races; every model keeps every driver."""
    if len(events) < 100 or len(events) != len(set(events)) or any(int(e.split(':')[0]) >= 2026 for e in events):
        raise ValueError('at least 100 unique pre-2026 chronological events required')
    fields = {}
    for model, rows in predictions.items():
        groups = defaultdict(list)
        for row in rows:
            groups[row['event_id']].append(row['driver_id'])
        current = {e: set(groups[e]) for e in events}
        if any(len(groups[e]) != len(current[e]) or not current[e] for e in events):
            raise ValueError('models require same complete unique fields')
        if fields and fields != current:
            raise ValueError('models require same complete fields')
        fields = current
    output = {}
    for n in (5,10,50,100):
        selected = events[-n:]
        models = {name:evaluate_predictions([r for r in rows if r['event_id'] in selected],bootstrap_samples=1000,seed=42)
                  for name,rows in predictions.items()}
        output[str(n)] = dict(cohort_event_ids=selected, models=models,
            paired_qualifying_differences={name:{metric:paired_difference(result['per_race'],models['qualifying']['per_race'],metric)
                for metric in ('rank_spearman','position_mae','winner_accuracy','ndcg_at_3')}
                for name,result in models.items() if name != 'qualifying'})
    return output


def xgboost_folds(root, benchmark, *, output_root='.local/xgboost_evaluations'):
    from ml.training import TrainingConfig, train_model, predict_model
    from ml.archive_workflow import chronological_folds
    import xgboost
    root = Path(root)
    protocol = dict(version='xgboost-full-field-v1', dataset_sha256=_hash(root/'dataset.json'),
        cohort_sha256=benchmark['cohort_sha256'], seed=42, xgboost_version=xgboost.__version__,
        folds=benchmark['folds'], implementation_sha256={p:_hash(p) for p in ('ml/training.py','ml/active_study.py','ml/features.py')})
    key = sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    dest = Path(output_root)/key
    _write_json(dest/'protocol.json',protocol)
    data = _read(root/'dataset.json')
    wanted = {f['test_event_ids'][0]:f for f in benchmark['folds']}
    predictions,telemetry = [],[]
    for train,test in chronological_folds(data,min_train_events=12):
        event = test[0]['event_id']
        if event not in wanted:
            continue
        split = dict(train_event_ids=sorted({r['event_id'] for r in train}),test_event_ids=[event])
        if split != wanted[event]:
            raise ValueError('XGBoost fold differs from frozen conventional protocol')
        path = dest/(event.replace(':','-')+'.json')
        if path.exists():
            record = _read(path)
        else:
            started = perf_counter()
            artifact = train_model(train,TrainingConfig(candidate='xgboost',split_manifest=split))
            fitted = perf_counter()
            # Targets are separated before inference even though the feature selector also rejects them.
            features = [{k:v for k,v in r.items() if k not in ('actual_position','actual_pace_s','status','outcome_status','eligible','outcome_available_at_utc')} for r in test]
            scores = predict_model(artifact,features)
            finished = perf_counter()
            record = dict(event_id=event,predictions=scores,train_seconds=fitted-started,
                inference_seconds=finished-fitted,elapsed_seconds=finished-started,
                training_metadata=artifact.metadata)
            _write_json(path,record)
            print('xgboost',event,f"{record['elapsed_seconds']:.3f}s",flush=True)
        predictions.extend(record['predictions'])
        telemetry.append({k:v for k,v in record.items() if k not in ('predictions','training_metadata')})
    if len(telemetry) != len(wanted):
        raise ValueError('XGBoost must complete all frozen folds')
    _write_json(dest/'predictions.json',predictions)
    return predictions,dict(protocol=protocol,fold_runtime=telemetry,output_path=str(dest.resolve()))


def run_active_study(*, source_study=SOURCE_STUDY, archive_root='.local/archive_ml', output_root='.local/model_studies'):
    source = Path(source_study); root = Path(archive_root)
    report = deepcopy(_read(source/'report.json'))
    if _hash(source/'predictions.json') != report['predictions_sha256']:
        raise ValueError('immutable source predictions mismatch')
    benchmark = _read(root/'benchmark.json')
    if report['dataset_sha256'] != _hash(root/'dataset.json') or report['cohort_sha256'] != benchmark['cohort_sha256']:
        raise ValueError('source study and current dataset lineage differ')
    predictions = {k:v for k,v in _read(source/'predictions.json').items() if k in ACTIVE_MODELS}
    xgb, runtime = xgboost_folds(root,benchmark)
    labels = defaultdict(list)
    for row in _read(root/'labels.json'): labels[row['event_id']].append(row)
    by_event = defaultdict(list)
    for row in xgb: by_event[row['event_id']].append(row)
    predictions['xgboost'] = [r for e in benchmark['cohort_event_ids'] for r in join_outcomes(by_event[e],labels[e])]
    names = {r['driver_id']:r.get('driver_name',r['driver_id']) for r in predictions['qualifying']}
    for row in predictions['xgboost']: row['driver_name'] = names.get(row['driver_id'],row['driver_id'])
    events = sorted(benchmark['cohort_event_ids'],key=lambda e:report['races'][e]['cutoff_utc'])
    report.update(version='active-historical-comparison-v1',protocol_version='nested-latest-100-v1',
        protocol=dict(version='nested-latest-100-v1',baseline='qualifying',min_train_events=12,seed=42,
            bootstrap_samples=1000,development_events=events,final_events=[],holdout_locked=True,
            selection='latest 100 chronological eligible pre-2026 races; nested last 5/10/50/100',
            preprocessing='fitted separately inside each expanding chronological training fold'),
        cohort_event_ids=events,active_models=list(ACTIVE_MODELS),
        benchmark_subsets=nested_benchmarks(predictions,events),
        source_study=dict(path=str(source.resolve()),report_sha256=_hash(source/'report.json'),predictions_sha256=_hash(source/'predictions.json')),
        runtime_telemetry=dict(xgboost=runtime,other_models=dict(status='cached_existing_evidence',note='No missing timings inferred; see original per-event foundation dispositions.')),
        models={k:evaluate_predictions(v,bootstrap_samples=1000,seed=42) for k,v in predictions.items()},
        foundation_metadata={k:v for k,v in report.get('foundation_metadata',{}).items() if k=='tabfm'})
    report['paired_qualifying_differences']={k:{m:paired_difference(v['per_race'],report['models']['qualifying']['per_race'],m)
        for m in ('rank_spearman','position_mae','winner_accuracy','ndcg_at_3')} for k,v in report['models'].items() if k!='qualifying'}
    report['case_study']['models']={k:[r for r in rows if r['event_id']==report['case_study']['event_id']] for k,rows in predictions.items()}
    report['implementation_sha256'].update({p:_hash(p) for p in ('ml/active_study.py','ml/training.py','ml/artifacts.py')})
    encoded=json.dumps(predictions,sort_keys=True,indent=2,allow_nan=False)+'\n'
    report['predictions_sha256']=sha256(encoded.encode()).hexdigest()
    key=sha256(json.dumps(report,sort_keys=True).encode()).hexdigest()
    dest=Path(output_root)/key
    _write_json(dest/'report.json',report);_write_json(dest/'predictions.json',predictions)
    (dest/'report.md').write_text('# Active model comparison\n\nEight active models; 109 matched development races and nested latest 5/10/50/100 cohorts.\n\nTimesFM removed from active choices; original evidence retained. 2026 remains locked.\n',encoding='utf-8')
    (root/'active_study.json').write_text(json.dumps(dict(path=str(dest.resolve()),report_sha256=_hash(dest/'report.json')),indent=2)+'\n',encoding='utf-8')
    print(str(dest.resolve()),flush=True)
    return dest


if __name__=='__main__':
    run_active_study()
