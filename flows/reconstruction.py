"""Explicit event-time reconstruction; never proof of historical publication."""
from dataclasses import replace

from data.contracts import DriverSessionRecord, SourceRecord


def project_completed_sessions(snapshot, schedule):
    if snapshot.payloads.get('mode') != 'post_qualifying':
        raise ValueError('historical reconstruction requires completed qualifying')
    sessions = {session.session_id: session for session in schedule.sessions}
    qualifying = sessions.get('Q')
    if qualifying is None or not qualifying.completed_by(snapshot.as_of_utc):
        raise ValueError('qualifying completion must precede reconstruction cutoff')
    entry_source = snapshot.payloads.get('entries_source')
    if not isinstance(entry_source, SourceRecord):
        raise ValueError('entry source is required')

    def event_time_source(source, observed):
        # This copy is confined to this explicitly labelled reconstruction.
        # Original fetched provenance remains on the immutable saved snapshot.
        return replace(source, published_at_utc=None, issued_at_utc=None,
                       available_at_utc=observed)

    records = []
    for record in snapshot.payloads.get('driver_sessions', ()):
        if not isinstance(record, DriverSessionRecord) or record.event != snapshot.event:
            raise ValueError('invalid reconstruction session record')
        if record.session_id not in {'FP1', 'FP2', 'FP3', 'Q', 'GRID', 'SQ', 'S'}:
            raise ValueError('only earlier weekend sessions can enter reconstruction')
        if record.available_at_utc > snapshot.as_of_utc:
            raise ValueError('session observation follows reconstruction cutoff')
        session = sessions.get('Q' if record.session_id == 'GRID' else record.session_id)
        if session is None or not session.completed_by(snapshot.as_of_utc):
            raise ValueError('session completion is required before reconstruction cutoff')
        if record.available_at_utc < session.ends_at_utc:
            raise ValueError('session availability cannot precede observed completion')
        records.append(replace(record, source=event_time_source(record.source, record.available_at_utc)))
    payload = dict(snapshot.payloads, historical_reconstruction=False, reconstruction_projection=True,
                   entries_source=event_time_source(entry_source, qualifying.ends_at_utc),
                   driver_sessions=records)
    return replace(snapshot, payloads=payload)
