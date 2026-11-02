"""Explicit-input forecast pipeline and optional Prefect scheduling wrapper."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import os
import tempfile
from agents.graph import run_pre_race_agent
from data.contracts import PredictionRequest
from data.snapshots import load_snapshot

REPORTS_DIR=Path('.local/reports')


def save_report(result: dict, *, directory=REPORTS_DIR) -> Path:
    """Publish one canonical JSON bundle; all display/export formats derive from it."""
    import re
    run_id=result.get('run_id','')
    if not re.fullmatch(r'forecast-[0-9a-f]{32}',run_id):
        raise ValueError('generated forecast run_id required')
    # Output location is excluded from the immutable result's content.
    payload={key:value for key,value in result.items() if key!='report_path'}
    encoded=json.dumps(payload,sort_keys=True,indent=2,allow_nan=False)
    root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    target=root/f'{run_id}.json'
    if target.exists():
        if json.loads(target.read_text(encoding='utf-8'))!=payload:
            raise ValueError('immutable report conflict')
        return target
    handle,temporary=tempfile.mkstemp(prefix='.report-',suffix='.tmp',dir=root)
    try:
        with os.fdopen(handle,'w',encoding='utf-8') as stream:
            stream.write(encoded);stream.flush();os.fsync(stream.fileno())
        # Do not overwrite a prior immutable report on concurrent requests.
        if os.name=='nt': os.rename(temporary,target)
        else: os.link(temporary,target)
    except FileExistsError:
        if json.loads(target.read_text(encoding='utf-8'))!=payload:
            raise ValueError('immutable report conflict')
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target


def pre_race_pipeline(round_number=None, *, request=None,snapshot=None,reports_dir=REPORTS_DIR,**options):
    result=run_pre_race_agent(round_number=round_number,request=request,snapshot=snapshot,**options)
    return dict(result,report_path=str(save_report(result,directory=reports_dir)))


def prefect_pre_race_pipeline(**kwargs):
    """Create the installed Prefect flow only when scheduling is explicitly requested."""
    from prefect import flow
    return flow(name='f1-pre-race-pipeline',retries=0)(pre_race_pipeline)(**kwargs)


def main(argv=None):
    parser=argparse.ArgumentParser(description='Run a forecast from an operator-controlled saved snapshot')
    parser.add_argument('--snapshot',required=True,help='Trusted saved snapshot JSON, never a user upload')
    parser.add_argument('--model',choices=['grid','rolling_form','ridge','random_forest','gradient_boosting'],default='rolling_form')
    parser.add_argument('--artifact-id',help='Generated fitted artifact ID in .local/models')
    parser.add_argument('--prefect',action='store_true',help='Explicitly run through Prefect')
    args=parser.parse_args(argv)
    snapshot=load_snapshot(args.snapshot)
    request=PredictionRequest(snapshot.event,snapshot.payloads['mode'],snapshot.as_of_utc)
    artifact=None
    if args.artifact_id:
        from ml.artifacts import ArtifactStore
        artifact=ArtifactStore('.local/models').load(args.artifact_id)
    from monitoring.store import TrackingStore
    options=dict(request=request,snapshot=snapshot,schedule=snapshot.payloads.get('schedule'),
                 model_id=args.model,artifact=artifact,owner_id='local',store=TrackingStore('.local/tracking.sqlite3'))
    runner=prefect_pre_race_pipeline if args.prefect else pre_race_pipeline
    result=runner(**options)
    print(json.dumps({'run_id':result['run_id'],'status':result['status'],'report_path':result['report_path']},indent=2))


if __name__=='__main__': main()
