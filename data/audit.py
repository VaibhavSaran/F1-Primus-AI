"""Operator CLI for bounded cache coverage and historical replay exports."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime
from enum import Enum
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from data.contracts import EventKey
from data.fastf1_source import LoadedSession,load_session,coverage_report


def _plain(value):
    if isinstance(value,datetime):return value.isoformat()
    if isinstance(value,Enum):return value.value
    if isinstance(value,dict):return {key:_plain(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [_plain(item) for item in value]
    return value


def persist_session(session:LoadedSession,*,directory='.local/sessions',positions=(),race_control=())->Path:
    body=_plain(dict(schema_version='session-v1',event_id=session.event.event_id,season=session.event.season,
        round_number=session.event.round_number,session_id=session.session_id,fetched_at_utc=session.source.fetched_at_utc,
        entries=session.entries,laps=session.laps,positions=positions,race_control=race_control,
        session_start_utc=session.session_start_utc,session_end_utc=session.session_end_utc,
        source=asdict(session.source),quality_flags=session.quality_flags,
        status=session.status,errors=session.errors,historical_analysis_status='hindsight_only',
        forecast_eligibility='excluded_historical_availability_unverified'))
    encoded=json.dumps(body,sort_keys=True,separators=(',',':'),allow_nan=False)
    digest=sha256(encoded.encode()).hexdigest()
    root=Path(directory).resolve();root.mkdir(parents=True,exist_ok=True)
    path=root/f'session-{digest}.json'
    if path.is_symlink():raise ValueError('session path must not be a symlink')
    if path.exists():
        if path.read_text(encoding='utf-8')!=encoded:raise ValueError('session integrity conflict')
        return path
    handle,temporary=tempfile.mkstemp(prefix='.session-',suffix='.tmp',dir=root)
    try:
        with os.fdopen(handle,'w',encoding='utf-8') as stream:
            stream.write(encoded);stream.flush();os.fsync(stream.fileno())
        if os.name=='nt':os.rename(temporary,path)
        else:os.link(temporary,path)
    except FileExistsError:
        if path.read_text(encoding='utf-8')!=encoded:raise ValueError('session integrity conflict')
    finally:Path(temporary).unlink(missing_ok=True)
    return path


def main(argv=None):
    parser=argparse.ArgumentParser(description='Audit a FastF1 session for descriptive historical analysis')
    parser.add_argument('--season',type=int,required=True)
    parser.add_argument('--round',type=int,required=True,dest='round_number')
    parser.add_argument('--session',choices=['FP1','FP2','FP3','Q','SQ','S','R'],default='R')
    parser.add_argument('--fetch',action='store_true',help='Explicitly permit bounded free FastF1 HTTP; default is offline')
    args=parser.parse_args(argv)
    session=load_session(EventKey(args.season,args.round_number),args.session,allow_fetch=args.fetch)
    path=persist_session(session)
    report=coverage_report([session]);report['saved_session']=str(path)
    print(json.dumps(report,indent=2,default=str))


if __name__=='__main__':main()
