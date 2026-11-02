"""Saved FastF1 calendars; UI reads are offline and never infer race names."""
from __future__ import annotations
import json
from pathlib import Path

CALENDAR_PATH = Path(__file__).resolve().parents[1] / '.local/catalog/calendar.json'


def load_calendar(path=CALENDAR_PATH):
    try:
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
        if payload.get('schema_version') != 'calendar-v1':
            raise ValueError('unsupported calendar schema')
        events = payload['events']
        if not isinstance(events, list):
            raise ValueError('invalid events')
        return events
    except (OSError, ValueError, KeyError, TypeError):
        return []


def event_label(event):
    return f"{event['event_name']} · {event['race_date']}"


def session_label(session_id):
    return {'FP1': 'Free Practice 1', 'FP2': 'Free Practice 2', 'FP3': 'Free Practice 3',
            'Q': 'Qualifying', 'SQ': 'Sprint Qualifying', 'S': 'Sprint',
            'R': 'Grand Prix Race'}.get(session_id, session_id)
