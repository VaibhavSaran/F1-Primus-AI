"""Immutable, offline race studies using independently joined held-out outcomes."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import csv
import json
from math import isfinite
from pathlib import Path
from random import Random
from statistics import mean

from ml.evaluation import evaluate_predictions

VERSION = 'historical-comparison-v1'


def _time(value):
    value = datetime.fromisoformat(value) if isinstance(value, str) else value
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValueError('UTC-aware time required')
    return value.astimezone(timezone.utc)


def join_outcomes(predictions, labels):
    """Never trust labels carried alongside a cached/model prediction."""
    predictions, labels = list(predictions), list(labels)
    pids = [(r['event_id'], r['driver_id']) for r in predictions]
    lids = [(r['event_id'], r['driver_id']) for r in labels]
    if (not pids or len(set(pids)) != len(pids) or len(set(lids)) != len(lids)
            or set(pids) != set(lids) or len({e for e, _ in pids}) != 1):
        raise ValueError('exact unique matching event/driver fields required')
    if any(isinstance(r.get('prediction_score'), bool) or
           not isinstance(r.get('prediction_score'), (int, float)) or
           not isfinite(r['prediction_score']) for r in predictions):
        raise ValueError('complete finite model scores required')
    ranks = {r['driver_id']: i for i, r in enumerate(sorted(predictions,
             key=lambda r: (r['prediction_score'], r['driver_id'])), 1)}
    observed = {r['driver_id']: r for r in labels}
    result = []
    for row in predictions:
        outcome = observed[row['driver_id']]
        rank, actual = ranks[row['driver_id']], outcome.get('actual_position')
        result.append(dict(event_id=row['event_id'], driver_id=row['driver_id'],
                           prediction_score=row['prediction_score'], predicted_rank=rank,
                           actual_position=actual, status=outcome.get('status', 'UNKNOWN'),
                           position_error=rank-actual if actual is not None else None,
                           outcome_status='completed'))
    return result


def paired_difference(candidate, baseline, metric, *, samples=1000, seed=42):
    """Resample paired whole races, never individual drivers."""
    left = {r['event_id']: r['metrics'].get(metric) for r in candidate}
    right = {r['event_id']: r['metrics'].get(metric) for r in baseline}
    ids = sorted(e for e in left.keys() & right.keys() if left[e] is not None and right[e] is not None)
    deltas = [left[e]-right[e] for e in ids]
    if not deltas:
        return dict(metric=metric, n_events=0, mean_delta=None, interval_95=None, event_ids=[])
    rng = Random(seed)
    boot = sorted(mean(rng.choices(deltas, k=len(deltas))) for _ in range(samples))
    return dict(metric=metric, n_events=len(ids), mean_delta=mean(deltas),
                interval_95=[boot[int(.025*(samples-1))], boot[int(.975*(samples-1))]], event_ids=ids)


def eligible_pace_session(session, *, target_event_id, cutoff_utc):
    return (session['event_id'] == target_event_id
            and session['session_code'] in {'FP1', 'FP2', 'FP3', 'S'}
            and session.get('completed_at_utc') is not None
            and _time(session['completed_at_utc']) < _time(cutoff_utc))


def pre_race_pace_rows(snapshot, cohort_path):
    """Clean observed laps of the same weekend; never target R or reanalysis weather.

    Each session remains a separate sequence: practice runs are not silently
    concatenated across pit stops, compounds, sessions or weekends.
    """
    from data.archive.reader import ArchiveReader
    cohort = json.loads(Path(cohort_path).read_text())
    event = snapshot.event.event_id
    result = []
    with ArchiveReader(cohort['archive_roots']['f1'])._connect() as c:
        for pin in cohort['sessions']:
            if f"{pin['season']}:{pin['round_number']}" != event:
                continue
            meta = c.execute('SELECT dataset_hash,observed_end_utc_ns FROM session_versions WHERE version_id=? AND status=\'published\'', (pin['version_id'],)).fetchone()
            if not meta or meta[0] != pin['dataset_hash']:
                raise ValueError('pace source pin mismatch')
            completed = datetime.fromtimestamp(meta[1]/1e9, timezone.utc) if meta[1] else None
            session = dict(event_id=event, session_code=pin['session_code'], completed_at_utc=completed)
            if not eligible_pace_session(session, target_event_id=event, cutoff_utc=snapshot.as_of_utc):
                continue
            # A pit/outlap, compound change or missing/nonclean lap breaks a run.
            runs, previous, counters = defaultdict(list), {}, defaultdict(int)
            sql = '''SELECT driver_id,lap_number,lap_time_ns,stint_number,compound_normalized,
                     provider_is_accurate,deleted,pit_in_clock_ns,pit_out_clock_ns,track_status_raw
                     FROM laps WHERE version_id=? ORDER BY driver_id,lap_number'''
            for driver, lap, ns, stint, compound, accurate, deleted, pitin, pitout, status in c.execute(sql, (pin['version_id'],)):
                last = previous.get(driver)
                valid = lap is not None and lap > 1 and ns is not None and ns > 0 and accurate == 1 and not deleted and pitin is None and pitout is None and status == '1'
                if not valid or not last or last != (lap-1, stint, compound):
                    counters[driver] += 1
                previous[driver] = (lap, stint, compound) if valid else None
                if valid:
                    runs[(driver, counters[driver], stint, compound)].append(ns/1e9)
            for (driver, run, stint, compound), values in runs.items():
                if len(values) >= 3:
                    result.append(dict(session, driver_id=driver, values=values,
                                       source_id=pin['version_id'], stint_number=stint,
                                       compound=compound, run_number=run))
    return result


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, sort_keys=True, indent=2, allow_nan=False, default=lambda x:x.isoformat())+'\n'
    if path.exists():
        if path.read_text(encoding='utf-8') != text:
            raise ValueError('immutable study output conflicts')
    else:
        path.write_bytes(text.encode('utf-8'))


def run_study(*, archive_root='.local/archive_ml', output_root='.local/model_studies', additional=None,
              additional_metadata=None, case_event='2025:19'):
    """Extend the saved development comparison without refitting or rerunning collection.

    Additional candidate maps contain event->raw score rows from real adapters.
    Coverage is reported separately; comparisons always use exact paired races.
    """
    from data.archive.reader import ArchiveReader
    from data.snapshots import load_snapshot
    from flows.reconstruction import project_completed_sessions
    from ml.baselines import weekend_form_baseline
    from ml.features import build_features
    root = Path(archive_root)
    benchmark, manifest = _read(root/'benchmark.json'), _read(root/'manifest.json')
    dataset, labels, cache = _read(root/'dataset.json'), _read(root/'labels.json'), _read(root/'benchmark_predictions.json')
    if benchmark['dataset_sha256'] != _hash(root/'dataset.json') or benchmark['cohort_sha256'] != manifest['cohort_sha256']:
        raise ValueError('saved comparison lineage mismatch')
    events = benchmark['cohort_event_ids']
    if any(int(e.split(':')[0]) >= 2026 for e in events):
        raise ValueError('2026 accuracy holdout remains locked')
    by_label, by_features = defaultdict(list), defaultdict(list)
    for row in labels: by_label[row['event_id']].append(row)
    for row in dataset: by_features[row['event_id']].append(row)
    predictions = {}
    for candidate, rows in cache.items():
        groups = defaultdict(list)
        for row in rows:
            if row['event_id'] in events: groups[row['event_id']].append(row)
        if set(groups) != set(events): raise ValueError('cached model coverage differs from frozen cohort')
        predictions[candidate] = dict(groups)
    snapshot_pins = {p['event_id']: p for p in manifest['snapshots']}
    case_snapshot = None
    race_context = {}
    predictions['weekend_form'] = {}
    for event in events:
        snapshot = load_snapshot(snapshot_pins[event]['path'])
        race_context[event]=dict(event_id=event,cutoff_utc=snapshot.as_of_utc.isoformat(),
            input_sessions=sorted({r.session_id for r in snapshot.payloads['driver_sessions']}))
        projected = project_completed_sessions(snapshot, snapshot.payloads['schedule'])
        features = build_features(projected, projected.payloads['history'])
        predictions['weekend_form'][event] = weekend_form_baseline(features.rows,
            projected.payloads['driver_sessions'], as_of_utc=projected.as_of_utc)
        if event == case_event: case_snapshot = snapshot
    for candidate, candidate_events in (additional or {}).items():
        if candidate in predictions or not set(candidate_events) <= set(events):
            raise ValueError('challenger identity/cohort must be new and development-only')
        predictions[candidate] = candidate_events
    models, joined = {}, {}
    for candidate, groups in predictions.items():
        rows = [r for e in events if e in groups for r in join_outcomes(groups[e], by_label[e])]
        joined[candidate] = rows
        evaluation = evaluate_predictions(rows, bootstrap_samples=1000, seed=42)
        models[candidate] = evaluation
        # Reproduce cached model scores against the independently held label file.
        if candidate in benchmark['models']:
            for metric, value in benchmark['models'][candidate]['metrics'].items():
                actual = evaluation['metrics'][metric]
                if value is not None and (actual is None or abs(actual-value) > 1e-10):
                    raise ValueError('cached prediction metrics fail independent reproduction')
    if case_snapshot is None: raise ValueError('case race is outside the frozen comparison field')
    cohort = _read(manifest['cohort_path'])
    with ArchiveReader(cohort['archive_roots']['f1'])._connect() as c:
        names = dict(c.execute('SELECT driver_id,display_name FROM drivers'))
        event_names = dict(c.execute('SELECT event_id,event_name FROM events'))
    for event,details in race_context.items():details['event_name']=event_names.get(event,event)
    for rows in joined.values():
        for row in rows:row['driver_name']=names.get(row['driver_id']) or row['driver_id']
    pairs = {candidate: {metric: paired_difference(result['per_race'], models['qualifying']['per_race'], metric)
                         for metric in ('rank_spearman','position_mae','winner_accuracy','ndcg_at_3')}
             for candidate, result in models.items() if candidate != 'qualifying'}
    case = dict(event_id=case_event, event_name=event_names.get(case_event, case_event),
                cutoff_utc=case_snapshot.as_of_utc.isoformat(),
                input_sessions=sorted({r.session_id for r in case_snapshot.payloads['driver_sessions']}),
                source_ids=sorted({s.source_id for s in case_snapshot.sources}),
                training_event_ids=next(f['train_event_ids'] for f in benchmark['folds'] if f['test_event_ids']==[case_event]),
                models={candidate: [dict(r, driver_name=names.get(r['driver_id']) or r['driver_id'])
                        for r in rows if r['event_id']==case_event] for candidate, rows in joined.items()})
    input_manifest = {str(p):_hash(p) for p in [root/'benchmark.json',root/'benchmark_predictions.json',root/'dataset.json',root/'labels.json',root/'manifest.json',Path(__file__)]}
    report = dict(version=VERSION, evidence_kind='historical_reconstruction', research_only=True,
                  historical_publication_verified=False, holdout_locked=True, cohort_event_ids=events,
                  status='evaluated_reconstruction',seed=42,dataset_sha256=benchmark['dataset_sha256'],
                  baseline_benchmark_sha256=_hash(root/'benchmark.json'),
                  final_holdout='2026 locked; not evaluated',
                  implementation_sha256={name:_hash(name) for name in (
                      'ml/historical_study.py','ml/foundation_study.py','ml/foundation_runtime.py',
                      'flows/foundation_prediction.py','ml/features.py','ml/evaluation.py')},
                  cohort_sha256=manifest['cohort_sha256'], input_sha256=input_manifest, models=models,
                  paired_qualifying_differences=pairs, case_study=case,
                  foundation_metadata=additional_metadata or {}, excluded_events=benchmark['excluded_events'],
                  races=race_context,
                  predictions_sha256=sha256((json.dumps(joined,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()).hexdigest(),
                  feature_inputs=dict(learned='practice + qualifying + prior driver/team/circuit form; no sprint feature in v1',
                      weekend_form='60% qualifying, 30% practice pace rank, 10% sprint finish; available weights renormalized'),
                  limitations=['Original publication timing is reconstructed, not verified live evidence.',
                      'No calibrated win probabilities or model promotion.',
                      'Future race weather, actual pit losses, DNFs and target race telemetry are excluded from inputs.',
                      'Foundation pretraining data may overlap historical races; no uncontaminated-pretraining claim.'])
    study_id = sha256(json.dumps(report,sort_keys=True,default=str).encode()).hexdigest()
    dest = Path(output_root)/study_id; dest.mkdir(parents=True,exist_ok=True)
    _write_json(dest/'report.json', report)
    lines = ['# Historical race model comparison', '', f"Common development cohort: {len(events)} races. 2026 remains locked.",
             '', 'Research reconstruction; original input publication times are unverified.', '',
             '| Model | Races | Order correlation | Position MAE | Winner accuracy |', '|---|---:|---:|---:|---:|']
    for candidate, result in models.items():
        m=result['metrics']; lines.append(f"| {candidate} | {result['n_events']} | {m['rank_spearman']:.4f} | {m['position_mae']:.4f} | {m['winner_accuracy']:.2%} |")
    lines += ['', f"## {case['event_name']} {case_event}", '', f"Cutoff: {case['cutoff_utc']}. Inputs: {', '.join(case['input_sessions'])}.",
              '', '| Model | Predicted top three | Actual winner |', '|---|---|---|']
    for candidate, rows in case['models'].items():
        top=sorted(rows,key=lambda r:r['predicted_rank'])[:3]
        winner=next((r['driver_name'] for r in rows if r['actual_position']==1),'Unavailable')
        lines.append(f"| {candidate} | {', '.join(r['driver_name'] for r in top)} | {winner} |")
    lines += ['', 'Full driver-level results and signed predicted-minus-actual errors are in case-comparison.csv.',
              'Matched paired confidence intervals, exclusions and source hashes are in report.json.', '', *report['limitations']]
    (dest/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with (dest/'case-comparison.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['model','event_id','driver_id','driver_name','predicted_rank','actual_position','position_error','status','prediction_score'])
        writer.writeheader()
        for candidate,rows in case['models'].items():
            for row in sorted(rows,key=lambda r:r['predicted_rank']):
                writer.writerow(dict(model=candidate,**{k:row[k] for k in writer.fieldnames if k!='model'}))
    _write_json(dest/'predictions.json',joined)
    if _hash(dest/'predictions.json')!=report['predictions_sha256']:
        raise ValueError('saved prediction bytes do not match report checksum')
    print(str(dest.resolve()),flush=True)
    return dest


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('--additional'); parser.add_argument('--metadata')
    args=parser.parse_args()
    run_study(additional=_read(args.additional) if args.additional else None,
              additional_metadata=_read(args.metadata) if args.metadata else None)
