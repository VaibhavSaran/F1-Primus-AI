"""Server-side authorization/admission boundary for explicit forecast jobs."""
from __future__ import annotations
from time import perf_counter
from uuid import uuid4
from security.access import authorize
from security.limits import LimitRejected
from flows.prediction_service import forecast_run_id,generate_forecast,resolve_history
from monitoring.queries import TrackingQueries


def guarded_forecast(principal,limiter,store,*,request,snapshot,request_id=None,history=None,**options):
    authorize(principal,'forecast',resource_owner=principal.identity)
    history=resolve_history(snapshot,history)
    if len(snapshot.payloads.get('entries',[]))>32:
        raise ValueError('forecast entry limit exceeded')
    run_id=forecast_run_id(request,snapshot,history=history,model_id=options.get('model_id','rolling_form'),
                           artifact=options.get('artifact'),owner_id=principal.identity,
                           foundation_receipt=options.get('foundation_receipt'))
    job_id=request_id or run_id
    lease=limiter.reserve(principal,job_id,run_id,action='forecast',paid=False,public=False)
    if lease.replayed:
        if lease.status=='completed':
            saved=TrackingQueries(store,owner_id=principal.identity,role=principal.role).get_forecast(run_id)
            if saved:
                return {key:value for key,value in saved.items() if key not in {'recorded_at_utc','is_public'}}
        raise LimitRejected('request_already_submitted',lease.deadline)
    if lease.status!='active':
        # This synchronous local entrypoint has no queue worker; reject honestly.
        limiter.settle(principal,job_id,actual_cost=0,failed=True)
        raise LimitRejected('busy',lease.deadline)
    started=perf_counter()
    try:
        limiter.checkpoint(principal,job_id)
        result=generate_forecast(request,snapshot,history=history,owner_id=principal.identity,store=None,**options)
        limiter.checkpoint(principal,job_id)
        if result['ranking']:store.save_forecast(result,is_public=False)
        limiter.settle(principal,job_id,actual_cost=0)
        state='success'
        return result
    except Exception:
        state='error'
        limiter.settle(principal,job_id,actual_cost=0,failed=True)
        raise
    finally:
        try:
            from datetime import datetime,timezone
            store.record_operational_event(dict(event_id='operation-'+uuid4().hex,owner_id=principal.identity,run_id=run_id,
                occurred_at_utc=datetime.now(timezone.utc),status=state,latency_ms=(perf_counter()-started)*1000,
                model_id=options.get('model_id','rolling_form'),estimated_cost=0,billed_cost=0))
        except Exception:
            # Observability cannot invalidate already persisted numeric output.
            pass

