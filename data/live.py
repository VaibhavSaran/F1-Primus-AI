"""Recorded-feed ingestion core. No live provider connection is enabled here.

One server collector lease per session; source and receive cutoffs are separate.
The operator must supply an authorized feed adapter before live release.
"""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
import sqlite3
from data.contracts import require_utc


def _time(value):
    if isinstance(value,str):value=datetime.fromisoformat(value.replace('Z','+00:00'))
    return require_utc(value)


def _id(value):
    if not isinstance(value,str) or not value.strip() or len(value)>256 or any(ord(c)<32 for c in value):
        raise ValueError('invalid feed identity')
    return value


def _source_content(event):
    return json.dumps({key:value for key,value in event.items() if key!='received_at_utc'},
                      sort_keys=True,allow_nan=False)


class LiveStore:
    def __init__(self,path,*,lease_seconds=30,max_events=100000):
        if type(lease_seconds) is not int or not 1<=lease_seconds<=300:raise ValueError('invalid collector lease')
        if type(max_events) is not int or not 1<=max_events<=1000000:raise ValueError('invalid feed capacity')
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.lease_seconds,self.max_events=lease_seconds,max_events
        with self._db() as db:
            db.execute('CREATE TABLE IF NOT EXISTS collectors(session_id TEXT PRIMARY KEY,owner TEXT,expires REAL)')
            db.execute('CREATE TABLE IF NOT EXISTS events(session_id TEXT,event_id TEXT,payload TEXT,source_at REAL,received_at REAL,sequence INTEGER,PRIMARY KEY(session_id,event_id))')
            db.execute('CREATE INDEX IF NOT EXISTS source_clock ON events(session_id,source_at,received_at)')

    @contextmanager
    def _db(self):
        db=sqlite3.connect(self.path,timeout=2)
        try:
            with db:yield db
        finally:db.close()

    def claim(self,session_id,owner,*,now=None):
        now=require_utc(now or datetime.now(timezone.utc)).timestamp()
        _id(session_id);_id(owner)
        with self._db() as db:
            db.execute('BEGIN IMMEDIATE')
            prior=db.execute('SELECT owner,expires FROM collectors WHERE session_id=?',(session_id,)).fetchone()
            if prior and prior[0]!=owner and prior[1]>now:return False
            db.execute('INSERT OR REPLACE INTO collectors VALUES(?,?,?)',(session_id,owner,now+self.lease_seconds))
            return True

    def disconnect(self,session_id,owner):
        with self._db() as db:
            db.execute('DELETE FROM collectors WHERE session_id=? AND owner=?',(session_id,owner))

    def append(self,owner,event,*,now=None):
        event=dict(event)
        for field in ('event_id','session_id','driver_id'):
            if field in event:_id(event[field])
        for field in ('event_id','session_id'):
            if field not in event:raise ValueError('feed event identity missing')
        if event.get('kind') not in {'position','lap','weather','race_control','heartbeat'}:raise ValueError('unsupported feed kind')
        sequence=event.get('sequence')
        if type(sequence) is not int or not 0<=sequence<=10**9:raise ValueError('invalid feed sequence')
        source,received=_time(event['source_at_utc']),_time(event['received_at_utc'])
        if source>received+timedelta(seconds=5):raise ValueError('source clock is ahead of receive clock')
        event['source_at_utc'],event['received_at_utc']=source.isoformat(),received.isoformat()
        encoded=json.dumps(event,sort_keys=True,allow_nan=False)
        if len(encoded.encode())>8192:raise ValueError('feed event too large')
        now=require_utc(now or datetime.now(timezone.utc)).timestamp()
        with self._db() as db:
            db.execute('BEGIN IMMEDIATE')
            lease=db.execute('SELECT owner,expires FROM collectors WHERE session_id=?',(event['session_id'],)).fetchone()
            if not lease or lease[0]!=owner or lease[1]<=now:raise PermissionError('collector lease unavailable')
            old=db.execute('SELECT payload FROM events WHERE session_id=? AND event_id=?',(event['session_id'],event['event_id'])).fetchone()
            if old:
                if _source_content(json.loads(old[0]))!=_source_content(event):
                    raise ValueError('conflicting duplicate feed event')
                # Keep the first receipt as the historical availability boundary.
                return False
            count=db.execute('SELECT COUNT(*) FROM events WHERE session_id=?',(event['session_id'],)).fetchone()[0]
            if count>=self.max_events:raise ValueError('feed capacity reached; archive before ingesting more')
            db.execute('INSERT INTO events VALUES(?,?,?,?,?,?)',(event['session_id'],event['event_id'],encoded,source.timestamp(),received.timestamp(),sequence))
            return True

    def count(self,session_id):
        with self._db() as db:return db.execute('SELECT COUNT(*) FROM events WHERE session_id=?',(session_id,)).fetchone()[0]

    def prefix(self,session_id,*,source_cutoff,received_cutoff):
        with self._db() as db:
            return [json.loads(row[0]) for row in db.execute('SELECT payload FROM events WHERE session_id=? AND source_at<=? AND received_at<=? ORDER BY source_at,sequence,event_id',
                    (session_id,_time(source_cutoff).timestamp(),_time(received_cutoff).timestamp()))]

    def state(self,session_id,*,now=None,connected=False,stale_seconds=15):
        if isinstance(stale_seconds,bool) or not 0<stale_seconds<=300:raise ValueError('invalid freshness window')
        at=require_utc(now or datetime.now(timezone.utc))
        with self._db() as db:
            rows=db.execute('SELECT source_at,received_at,sequence FROM events WHERE session_id=? AND received_at<=? ORDER BY sequence',(session_id,at.timestamp())).fetchall()
            lease=db.execute('SELECT expires FROM collectors WHERE session_id=?',(session_id,)).fetchone()
        if not rows:return dict(status='unavailable',sequence_gaps=[],n_events=0)
        last_source,last_received=max(row[0] for row in rows),max(row[1] for row in rows)
        gaps=[];sequences=sorted(set(row[2] for row in rows))
        for first,last in zip(sequences,sequences[1:]):
            gaps.extend(list(range(first+1,min(last,first+1+1000-len(gaps)))))
            if len(gaps)>=1000:break
        stale=at.timestamp()-last_received>stale_seconds or at.timestamp()-last_source>stale_seconds
        status='disconnected' if connected is not True else 'stale' if stale or not lease or lease[0]<=at.timestamp() else 'degraded' if gaps else 'live'
        return dict(status=status,n_events=len(rows),sequence_gaps=gaps,source_age_s=at.timestamp()-last_source,
                    receive_age_s=at.timestamp()-last_received,provider_access='not_configured_by_this_module')
