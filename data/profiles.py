"""Timestamped descriptive profiles, never extra numeric forecasting features."""
from __future__ import annotations

from datetime import datetime
from hashlib import sha256
import json
from math import isfinite
import re
from statistics import mean

from data.contracts import SourceRecord, require_utc

VERSION = 'profiles-v1'


def _time(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace('Z', '+00:00'))
    return require_utc(value)


def _id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:@-]{0,159}', value):
        raise ValueError('invalid profile identifier')
    return value


def _form(rows):
    positions = [r['actual_position'] for r in rows
                 if r['actual_position'] is not None and r['status'] not in {'DNS', 'DSQ'}]
    return dict(mean_position=mean(positions) if positions else None, sample_count=len(positions),
                event_count=len({r['event_id'] for r in rows}),
                window_start_utc=min((r['event_time_utc'] for r in rows), default=None),
                window_end_utc=max((r['event_time_utc'] for r in rows), default=None),
                definition='Mean recorded classified position; DNS/DSQ excluded, classified DNF retained')


def _profile(rows, *, entity_id, kind, cutoff):
    events = sorted({(r['event_time_utc'], r['event_id']) for r in rows}, reverse=True)[:5]
    recent = [r for r in rows if (r['event_time_utc'], r['event_id']) in events]
    circuits = sorted({r['circuit_id'] for r in rows if r['circuit_id']})
    return dict(entity_id=entity_id, kind=kind, as_of_utc=cutoff,
                recent_form=_form(recent),
                reliability=dict(sample_count=len(rows),
                                 classified_dnf_count=sum(r['status'] == 'DNF' for r in rows),
                                 dns_count=sum(r['status'] == 'DNS' for r in rows),
                                 dsq_count=sum(r['status'] == 'DSQ' for r in rows),
                                 window_start_utc=min(r['event_time_utc'] for r in rows),
                                 window_end_utc=max(r['event_time_utc'] for r in rows),
                                 definition='Recorded status counts only; DNF cause/attribution unknown'),
                circuit_samples=[dict(circuit_id=c, **_form([r for r in rows if r['circuit_id'] == c]))
                                 for c in circuits],
                source_ids=sorted({s for r in rows for s in r['source_ids']}),
                lineage=[dict(event_id=r['event_id'], driver_id=r['driver_id'], team_id=r['team_id'],
                              event_time_utc=r['event_time_utc'], actual_position=r['actual_position'],
                              status=r['status'], circuit_id=r['circuit_id'],
                              outcome_available_at_utc=r['outcome_available_at_utc'], source_ids=r['source_ids'])
                         for r in rows],
                quality=('descriptive; small samples and confounding possible'
                         if all(r['source_ids'] for r in rows) else 'missing source lineage; do not cite as verified'),
                freshness_utc=max((r['outcome_available_at_utc'] for r in rows), default=None),
                feature_use='explanatory only; not an additional model feature')


def build_profiles(history, as_of_utc):
    """Build stable snapshots from trusted history rows with publication evidence.

    Affiliation means latest observed event affiliation, not a claim about a
    current contract. Unknown publication times are excluded, including outcomes
    with no timestamp. A snapshot contains no personality or inferred DNF cause.
    """
    cutoff = _time(as_of_utc)
    rows, seen = [], set()
    for item in history.rows if hasattr(history, 'rows') else history:
        if item.get('outcome_status') == 'pending':
            continue
        event_time = item.get('event_time_utc')
        publication = item.get('outcome_available_at_utc')
        if event_time is None or publication is None:
            continue
        event_time, publication = _time(event_time), _time(publication)
        if event_time >= cutoff or publication > cutoff:
            continue
        if publication < event_time:
            raise ValueError('outcome publication precedes event')
        driver, event = _id(item.get('driver_id')), _id(item.get('event_id'))
        key = (event, driver)
        if key in seen:
            raise ValueError('duplicate profile event/driver')
        seen.add(key)
        status = str(item.get('status', 'pending')).strip().upper()
        if status == 'PENDING':
            continue
        position = item.get('actual_position')
        if position is not None and (isinstance(position, bool) or not isinstance(position, (int, float))
                                     or not isfinite(position) or position < 1 or int(position) != position):
            raise ValueError('invalid historical position')
        if position is None and status not in {'DNF', 'DNS', 'DSQ'}:
            continue
        sources = item.get('source_ids', ())
        if not isinstance(sources, (list, tuple)):
            raise ValueError('source_ids must be a sequence')
        sources = [_id(s) for s in sources]
        source = item.get('source')
        if isinstance(source, SourceRecord):
            if not source.available_by(cutoff):
                continue
            sources.append(_id(source.source_id))
        if not sources and item.get('snapshot_id'):
            sources.append(_id(item['snapshot_id']))
        rows.append(dict(event_id=event, driver_id=driver,
                         team_id=_id(item['team_id']) if item.get('team_id') else None,
                         circuit_id=_id(item['circuit_id']) if item.get('circuit_id') else None,
                         event_time_utc=event_time.isoformat(), outcome_available_at_utc=publication.isoformat(),
                         actual_position=position, status=status, source_ids=sorted(set(sources))))
    rows.sort(key=lambda r: (r['event_time_utc'], r['event_id'], r['driver_id']))
    drivers, teams = [], []
    for driver in sorted({r['driver_id'] for r in rows}):
        records = [r for r in rows if r['driver_id'] == driver]
        profile = _profile(records, entity_id=driver, kind='driver', cutoff=cutoff.isoformat())
        affiliations = [dict(team_id=r['team_id'], event_id=r['event_id'],
                             observed_at_utc=r['event_time_utc'], available_at_utc=r['outcome_available_at_utc'])
                        for r in records if r['team_id']]
        profile.update(driver_id=driver, affiliation=affiliations[-1] if affiliations else None,
                       affiliation_history=affiliations)
        drivers.append(profile)
    for team in sorted({r['team_id'] for r in rows if r['team_id']}):
        records = [r for r in rows if r['team_id'] == team]
        profile = _profile(records, entity_id=team, kind='team', cutoff=cutoff.isoformat())
        latest_event = records[-1]['event_id']
        profile.update(team_id=team, lineup=dict(event_id=latest_event,
                       driver_ids=sorted(r['driver_id'] for r in records if r['event_id'] == latest_event),
                       observed_at_utc=records[-1]['event_time_utc']))
        teams.append(profile)
    payload = dict(version=VERSION, as_of_utc=cutoff.isoformat(), drivers=drivers, teams=teams)
    payload['snapshot_id'] = 'profiles-' + sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()[:20]
    return payload
