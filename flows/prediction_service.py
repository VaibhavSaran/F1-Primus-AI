"""Deterministic forecast service shared by UI/jobs; no provider calls on import."""
from __future__ import annotations
from dataclasses import asdict,is_dataclass
from datetime import datetime
from enum import Enum
from hashlib import sha256
import json
import logging
from typing import Any
from data.contracts import DataSnapshot, PredictionRequest, ProgressEvent, SourceRecord
from ml.baselines import grid_baseline, qualifying_baseline, rolling_form_baseline, weekend_form_baseline
from ml.features import HistoryView, build_features


def _json(value):
    if isinstance(value,datetime): return value.isoformat()
    if isinstance(value,Enum): return value.value
    if is_dataclass(value): return _json(asdict(value))
    if isinstance(value,dict): return {str(k):_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [_json(v) for v in value]
    return value


def resolve_history(snapshot,history=None):
    """Omission uses saved evidence; an explicit empty iterable overrides it."""
    if history is None:
        history=snapshot.payloads.get('history',())
    return HistoryView(tuple(history.rows if isinstance(history,HistoryView) else history))


def forecast_run_id(request,snapshot,*,history=None,model_id='rolling_form',artifact=None,owner_id='local',foundation_receipt=None):
    history=resolve_history(snapshot,history)
    identity=dict(request=request,snapshot=snapshot,history=history,model_id=artifact.model_id if artifact else model_id,owner_id=owner_id)
    if foundation_receipt is not None:
        identity['foundation_receipt']=foundation_receipt
        identity['foundation_run_schema']='foundation-forecast-v2'
    return 'forecast-'+sha256(json.dumps(_json(identity),sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()[:32]


def generate_forecast(request: PredictionRequest, snapshot: DataSnapshot, *, history=None,
                      model_id='rolling_form', artifact=None, schedule=None,
                      owner_id='local', store=None, progress=None, research_reconstruction=False,foundation_receipt=None) -> dict[str,Any]:
    """Use an explicit verified snapshot and fitted artifact, never train on render.

    This internal service accepts trusted records. Public callers must pass the
    shared authorization/compute limiter before entry; it is not an HTTP route.
    Baselines/unevaluated artifacts remain private research output, not champions.
    """
    if not isinstance(owner_id,str) or not owner_id.strip():
        raise ValueError('owner identity is required')
    if request.event!=snapshot.event or request.as_of_utc!=snapshot.as_of_utc:
        raise ValueError('request event/cutoff must match immutable snapshot')
    if snapshot.payloads.get('mode')!=request.mode.value:
        raise ValueError('snapshot forecast mode must match request')
    if schedule is None:
        raise ValueError('verified session schedule is required')
    request.validate_schedule(schedule)
    if model_id in {'tabfm','timesfm'}:
        from flows.foundation_prediction import foundation_forecast
        if artifact is not None:
            raise ValueError('foundation challenger must use its own immutable receipt')
        return foundation_forecast(request,snapshot,foundation_receipt,model_id=model_id,owner_id=owner_id,store=store,history=history)
    if snapshot.payloads.get('historical_reconstruction') is True:
        from flows.reconstruction import project_completed_sessions
        from flows.reporting import forecast_report
        learned = model_id in {'ridge', 'random_forest', 'gradient_boosting', 'xgboost'}
        if learned and (not research_reconstruction or artifact is None or
                        artifact.metadata.get('research_reconstruction_only') is not True):
            raise ValueError('learned historical reconstruction requires an explicit research artifact and flag')
        if not learned and (model_id not in {'grid', 'qualifying', 'rolling_form', 'weekend_form'} or artifact is not None):
            raise ValueError('historical reconstruction supports explicit research models only')
        projected = project_completed_sessions(snapshot, schedule)
        result = generate_forecast(request, projected, history=resolve_history(snapshot,history), model_id=model_id,
                                   artifact=artifact, research_reconstruction=research_reconstruction,
                                   schedule=schedule, owner_id=owner_id, store=None)
        result['run_id'] = forecast_run_id(request, snapshot, history=history, model_id=model_id, artifact=artifact, owner_id=owner_id)
        result['provenance'].update(validation_status='historical_reconstruction',
            original_fetch_times=sorted({source.fetched_at_utc.isoformat() for source in snapshot.sources}),
            historical_publication_verified=False, target_session='R', research_reconstruction=True)
        result['warnings'].insert(0, 'Historical reconstruction: completed-session event times assumed; original publication timing is unverified. Not a live forecast or validated backtest.')
        result['briefing'] = 'Grand Prix finishing-order reconstruction using only earlier weekend sessions. Final race results are excluded. Research scores are not win probabilities.'
        result['report'] = forecast_report(result)
        if store is not None and result['ranking']:
            store.save_forecast(result, is_public=False)
        return result
    if artifact is not None and getattr(artifact,'metadata',{}).get('research_reconstruction_only') is True:
        if not research_reconstruction or snapshot.payloads.get('reconstruction_projection') is not True:
            raise ValueError('research reconstruction artifact cannot enter live forecasting')
    entry_source=snapshot.payloads.get('entries_source')
    if not isinstance(entry_source,SourceRecord) or not entry_source.available_by(request.as_of_utc):
        raise ValueError('pre-cutoff entry list provenance is required')
    if model_id not in {'grid','qualifying','rolling_form','weekend_form','ridge','random_forest','gradient_boosting','xgboost'}:
        raise ValueError('unknown forecast model')
    if model_id in {'grid','qualifying','rolling_form','weekend_form'} and artifact is not None:
        raise ValueError('baseline forecasts must not receive a learned artifact')
    if model_id not in {'grid','qualifying','rolling_form','weekend_form'}:
        if artifact is None or artifact.candidate!=model_id:
            raise ValueError('matching fitted model artifact is required')
        if artifact.target!='finishing_position':
            raise ValueError('rank forecast requires finishing-position artifact')
    history=resolve_history(snapshot,history)
    run_id=forecast_run_id(request,snapshot,history=history,model_id=model_id,artifact=artifact,owner_id=owner_id)
    warnings=list(snapshot.quality_flags)
    def emit(stage,message):
        if progress is not None:
            try: progress(ProgressEvent(run_id,stage,'complete',message))
            except Exception:
                logging.getLogger(__name__).warning('Forecast progress observer failed',extra={'run_id':run_id,'stage':stage})
    emit('validate','Request, schedule and entry provenance validated')
    features=build_features(snapshot,history,mode=request.mode)
    emit('features',f'Built {len(features.rows)} cutoff-safe entry rows')
    if model_id=='grid':
        rows=grid_baseline(features.rows)
    elif model_id=='qualifying':
        rows=qualifying_baseline(features.rows)
    elif model_id=='weekend_form':
        rows=weekend_form_baseline(features.rows,snapshot.payloads.get('driver_sessions',()),as_of_utc=request.as_of_utc)
        warnings.append('Weekend form is an untrained heuristic: qualifying 60%, practice pace rank 30%, sprint finish 10%; available weights renormalized')
    elif model_id=='rolling_form':
        rows=rolling_form_baseline(history.rows,features.rows,as_of_utc=request.as_of_utc,
                                   prior_position=(len(features.rows)+1)/2)
        if any(row['baseline_source']=='prior' for row in rows):
            warnings.append('Missing driver/team history uses an explicit field-midpoint prior')
    else:
        from ml.training import predict_model
        # An artifact with historical/current target labels cannot forecast itself.
        training_ids=set(artifact.metadata.get('training_event_ids',()))
        if request.event.event_id in training_ids:
            raise ValueError('forecast event is present in the training artifact')
        if artifact.metadata.get('training_availability_verified') is not True:
            raise ValueError('training label availability is unverified')
        for field in ('training_max_available_at_utc','training_max_event_time_utc'):
            value=artifact.metadata.get(field)
            if not value:
                raise ValueError('training timestamp evidence is missing')
            known=datetime.fromisoformat(value)
            if known.utcoffset() is None or known>=request.as_of_utc:
                raise ValueError('training data must be known strictly before forecast cutoff')
        rows=predict_model(artifact,features)
    available=[row for row in rows if row.get('prediction_score') is not None]
    if len(available)!=len(rows):
        # A partial order would silently change the scored population.
        ranking=[]
        status='insufficient_data'
        warnings.append('Full-field ranking unavailable: one or more entries lack a model score')
    else:
        ranking=[]
        for rank,row in enumerate(sorted(available,key=lambda row:(row['prediction_score'],row['driver_id'])),1):
            ranking.append(dict(event_id=request.event.event_id,driver_id=row['driver_id'],team_id=row.get('team_id'),
                                prediction_score=float(row['prediction_score']),predicted_rank=rank,
                                score_units=row['score_units'],baseline_source=row.get('baseline_source')))
        status='degraded'
    warnings.append('Not validated on a frozen historical cohort; no winner probabilities or accuracy claim')
    emit('predict','Numeric prediction completed' if ranking else 'Insufficient data for a full-field forecast')
    provenance=dict(snapshot_id=snapshot.snapshot_id,feature_schema=features.schema_version,
                    model_version=artifact.model_version if artifact else ('weekend-form-v1' if model_id=='weekend_form' else 'baseline-v1'),
                    validation_status='unevaluated',source_ids=sorted({entry_source.source_id,*[s.source_id for s in snapshot.sources],
                                                                      *[s for row in features.rows for s in row['source_ids']]}),
                    missing_inputs={row['driver_id']:[k for k in features.feature_columns if row.get(k) is None] for row in features.rows})
    provenance['target_session']='R'
    provenance['input_sessions']=sorted({record.session_id for record in snapshot.payloads.get('driver_sessions',())
                                       if record.available_at_utc<=request.as_of_utc and record.source.available_by(request.as_of_utc)})
    result=_json(dict(run_id=run_id,event_id=request.event.event_id,season=request.event.season,round_number=request.event.round_number,
                      owner_id=owner_id,model_id=artifact.model_id if artifact else model_id,mode=request.mode.value,
                      as_of_utc=request.as_of_utc,snapshot_id=snapshot.snapshot_id,ranking=ranking,metrics={},
                      provenance=provenance,warnings=warnings,status=status,artifacts={},
                      briefing='Research forecast using information available at the saved cutoff. Scores are ordinal positions, not win odds.'))
    from flows.reporting import forecast_report
    result['report']=forecast_report(result)
    result['message_count']=0
    if store is not None and ranking:
        store.save_forecast(result,is_public=False)
    emit('persist','Saved immutable forecast' if store is not None and ranking else 'Returned forecast; persistence not requested or data insufficient')
    return result
