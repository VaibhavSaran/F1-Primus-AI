"""Past-only event/driver features; labels never enter this module's output."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from statistics import mean
from typing import Any, Iterable

from data.contracts import DataSnapshot, DriverSessionRecord, ForecastMode, require_utc

PRACTICE = ('practice_pace_delta_s', 'practice_variability_s', 'practice_laps',
            'tyre_age_laps', 'stint_length', 'compound')
QUALIFYING = ('qualifying_position', 'qualifying_gap_s', 'grid_position')
CONTEXT = ('weather_rain_probability', 'weather_air_temperature_c', 'weather_wind_speed',
           'expected_pit_loss_s', 'team_stop_time_s', 'penalty_grid_places')
FEATURE_COLUMNS = ('circuit_id', 'team_id', 'driver_form_position', 'team_form_position',
                   'circuit_form_position', 'form_events', *PRACTICE, *QUALIFYING, *CONTEXT,
                   'practice_missing', 'qualifying_missing')


@dataclass(frozen=True)
class HistoryView:
    rows: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class FeatureFrame:
    rows: tuple[dict[str, Any], ...]
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS
    schema_version: str = 'features-v1'


def _time(value: Any) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace('Z', '+00:00'))
    return require_utc(value)


def _number(value: Any, name: str) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f'{name} must be a finite number or None')
    return value


def clean_laps(laps: Iterable[dict]) -> list[dict]:
    """Keep representative clean laps, not only a driver's fastest laps."""
    result = []
    for lap in laps:
        time = lap.get('lap_time_s')
        if (isinstance(time, bool) or not isinstance(time, (int, float)) or not isfinite(time)
                or time <= 0 or lap.get('lap_number', 0) <= 1):
            continue
        if (lap.get('accurate') is not True or lap.get('deleted') or lap.get('pit_in')
                or lap.get('pit_out') or str(lap.get('track_status', '')) != '1'):
            continue
        result.append(dict(lap))
    return result


def _history(history: HistoryView | Iterable[dict], snapshot: DataSnapshot) -> list[dict]:
    rows = history.rows if isinstance(history, HistoryView) else history
    eligible, seen = [], set()
    for item in rows:
        if item.get('event_id') == snapshot.event.event_id or item.get('status') in {'DNS', 'DSQ'}:
            continue
        published, event_time = item.get('outcome_available_at_utc'), item.get('event_time_utc')
        if published is None or event_time is None or item.get('actual_position') is None:
            continue
        if _time(published) > snapshot.as_of_utc or _time(event_time) >= snapshot.as_of_utc:
            continue
        position = _number(item['actual_position'], 'actual_position')
        if position <= 0:
            raise ValueError('actual_position must be positive')
        key = (item['event_id'], item['driver_id'])
        if key in seen:
            raise ValueError('duplicate historical event/driver')
        seen.add(key)
        eligible.append(dict(item, actual_position=position))
    return sorted(eligible, key=lambda item: (_time(item['event_time_utc']), item['event_id']), reverse=True)


def build_features(snapshot: DataSnapshot, history: HistoryView | Iterable[dict] = (),
                   *, mode: ForecastMode | str | None = None) -> FeatureFrame:
    mode = ForecastMode(mode or snapshot.payloads.get('mode', 'post_qualifying'))
    if mode not in (ForecastMode.PRE_WEEKEND, ForecastMode.POST_QUALIFYING):
        raise ValueError('pre-race features require pre_weekend or post_qualifying mode')
    entries = snapshot.payloads.get('entries', [])
    if not entries:
        raise ValueError('an explicit event entry list is required')
    ids = [entry.get('driver_id') for entry in entries]
    if any(not isinstance(value, str) or not value.strip() for value in ids):
        raise ValueError('entry driver_id is required')
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate event entry driver_id')
    observations = []
    for record in snapshot.payloads.get('driver_sessions', []):
        if not isinstance(record, DriverSessionRecord) or record.event != snapshot.event:
            raise ValueError('driver session record must match snapshot event')
        if record.available_at_utc <= snapshot.as_of_utc and record.source.available_by(snapshot.as_of_utc):
            observations.append(record)
    observations.sort(key=lambda record: (record.available_at_utc, record.source.source_id, record.session_id))
    past = _history(history, snapshot)
    result = []
    for entry in entries:
        driver, team = entry['driver_id'], entry.get('team_id')
        previous = [item for item in past if item['driver_id'] == driver][:5]
        team_rows = [item for item in past if team is not None and item.get('team_id') == team]
        recent_events = list(dict.fromkeys(item['event_id'] for item in team_rows))[:5]
        team_rows = [item for item in team_rows if item['event_id'] in recent_events]
        circuit_rows = [item for item in past if item['driver_id'] == driver
                        and snapshot.event.circuit_id is not None
                        and item.get('circuit_id') == snapshot.event.circuit_id][:5]
        row = {column: None for column in FEATURE_COLUMNS}
        row.update(event_id=snapshot.event.event_id, season=snapshot.event.season,
                   round_number=snapshot.event.round_number, circuit_id=snapshot.event.circuit_id,
                   driver_id=driver, team_id=team, forecast_cutoff_utc=snapshot.as_of_utc,
                   driver_form_position=mean(item['actual_position'] for item in previous) if previous else None,
                   team_form_position=mean(item['actual_position'] for item in team_rows) if team_rows else None,
                   circuit_form_position=mean(item['actual_position'] for item in circuit_rows) if circuit_rows else None,
                   form_events=len(previous))
        sources = {item.get('source_id', f"outcome:{item['event_id']}") for item in previous + team_rows + circuit_rows}
        for record in observations:
            if record.driver_id != driver:
                continue
            session = record.session_id.upper()
            allowed = ()
            if mode == ForecastMode.POST_QUALIFYING:
                if session.startswith('FP') or session.startswith('PRACTICE'):
                    allowed = PRACTICE
                elif session in {'Q', 'QUALIFYING', 'GRID'}:
                    allowed = QUALIFYING
            if session == 'WEATHER_FORECAST':
                issued, valid = record.source.issued_at_utc, record.values.get('weather_valid_at_utc')
                if issued is not None and valid is not None and issued <= _time(valid):
                    allowed = CONTEXT[:3]
            elif session == 'PIT_ESTIMATE':
                allowed = ('expected_pit_loss_s', 'team_stop_time_s')
            elif session == 'VERIFIED_PENALTY':
                allowed = ('penalty_grid_places',)
            for column in allowed:
                if column not in record.values:
                    continue
                value = record.values[column]
                if column == 'compound':
                    if value is not None and not isinstance(value, str):
                        raise ValueError('compound must be text or None')
                else:
                    value = _number(value, column)
                row[column] = value
                sources.add(record.source.source_id)
        row['practice_missing'] = row['practice_pace_delta_s'] is None
        row['qualifying_missing'] = row['qualifying_position'] is None
        row['source_ids'] = tuple(sorted(sources))
        result.append(row)
    return FeatureFrame(tuple(result))
