"""Bounded historical replay. Relative clock filtering is not publication proof.

Final classifications never enter a clocked tower. Retained timing observations
provide race ranks and gaps; completion order remains an explicit fallback.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import json
import re
from math import isfinite
from pathlib import Path
from typing import Any

from data.contracts import EventKey, ReplayFrame, require_utc

FRAME_VERSION = 'replay-v1'
MAX_FILE_BYTES = 20_000_000
POSITION_MAX_AGE_MS = 5_000
WEATHER_MAX_AGE_MS = 120_000
TIMING_MAX_AGE_MS = 120_000
MAX_POSITION_SAMPLES = 120_000
MAX_PAYLOAD_BYTES = 20_000_000
WEATHER_FIELDS = ('air_temperature_c', 'track_temperature_c', 'relative_humidity_pct',
                  'pressure_hpa', 'wind_speed_mps', 'wind_direction_deg', 'rainfall_flag')


def _number(value, name, *, minimum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f'{name} must be finite numeric data')
    if minimum is not None and value < minimum:
        raise ValueError(f'{name} is below its minimum')
    return float(value)


def _text(value, name, *, optional=False):
    if optional and value is None:
        return None
    if not isinstance(value, str) or not value.strip() or len(value) > 2048:
        raise ValueError(f'{name} must be bounded nonempty text')
    return value


def _rows(data, name, limit):
    value = data.get(name, [])
    if not isinstance(value, (list, tuple)) or len(value) > limit or any(not isinstance(r, dict) for r in value):
        raise ValueError(f'{name} must be a bounded list of records')
    return value


def _clock(row):
    seconds = _number(row.get('session_time_s'), 'session_time_s', minimum=0)
    if seconds > 86400:
        raise ValueError('session clock exceeds 24 hour bound')
    return round(seconds * 1000)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _gap(raw):
    """FastF1 uppercase LAP N is a leader lap counter, never a deficit."""
    if not isinstance(raw, str) or len(raw) > 2048:
        raise ValueError('timing gap must be bounded text')
    raw = raw.strip()
    if re.fullmatch(r'LAP \d+', raw):
        return None, None, 'leader_counter'
    if re.fullmatch(r'\+?\d+(?:\.\d+)?', raw):
        seconds = float(raw.lstrip('+'))
        if isfinite(seconds):
            return seconds, None, f'+{seconds:.3f}s'
    match = re.fullmatch(r'\+?\s*(\d+)\s*L(?:AP(?:S)?)?', raw, re.IGNORECASE)
    if match and int(match[1]) > 0:
        laps = int(match[1])
        return None, laps, f'+{laps} lap' + ('s' if laps != 1 else '')
    return None, None, None


def _timing_at(rows, clock_ms, basis):
    latest = {}
    for row in rows:
        if row['clock_ms'] > clock_ms:
            break
        driver = latest.setdefault(row['driver_id'], {})
        for field in ('position', 'gap', 'interval'):
            if field in row:
                driver[field] = (row['clock_ms'], row[field])
    result = {}
    for driver, fields in latest.items():
        values = dict(position=None, gap_s=None, gap_laps=None, gap_label=None,
                      interval_s=None, interval_laps=None, interval_label=None)
        for field in ('position', 'gap', 'interval'):
            stamp, value = fields.get(field, (-TIMING_MAX_AGE_MS - 1, None))
            fresh = clock_ms - stamp <= TIMING_MAX_AGE_MS
            values[field + '_fresh'] = fresh
            if fresh:
                if field == 'position':
                    values['position'] = value
                else:
                    values[field + '_s'], values[field + '_laps'], values[field + '_label'] = value
        for field in ('gap', 'interval'):
            if values[field + '_label'] == 'leader_counter':
                values[field + '_label'] = 'Leader' if field == 'gap' and values['position'] == 1 else None
        values['timing_fresh'] = any(values[field + '_fresh'] for field in ('position', 'gap', 'interval'))
        values['timing_basis'] = basis
        result[driver] = values
    return result


@dataclass(frozen=True)
class ReplaySnapshot:
    frame: ReplayFrame
    laps: tuple[dict[str, Any], ...]
    stints: tuple[dict[str, Any], ...]
    markers: tuple[dict[str, Any], ...]
    provenance: dict[str, Any]


@dataclass(frozen=True)
class ReplaySession:
    identity: str
    event: EventKey
    session_id: str
    frame_version: str
    entries: tuple[dict[str, Any], ...]
    laps: tuple[dict[str, Any], ...]
    positions: tuple[dict[str, Any], ...]
    markers: tuple[dict[str, Any], ...]
    weather: tuple[dict[str, Any], ...]
    timing: tuple[dict[str, Any], ...]
    timing_basis: str
    source: dict[str, Any]
    quality_flags: tuple[str, ...]
    duration_ms: int

    @classmethod
    def from_mapping(cls, data):
        if not isinstance(data, dict) or data.get('schema_version') != 'session-v1':
            raise ValueError('unsupported normalized session schema')
        event = EventKey(data.get('season'), data.get('round_number'))
        if data.get('event_id') != event.event_id:
            raise ValueError('event identity mismatch')
        session_id = _text(data.get('session_id'), 'session_id')
        if session_id not in {'FP1', 'FP2', 'FP3', 'Q', 'SQ', 'S', 'R'}:
            raise ValueError('unsupported session identity')
        raw_source = data.get('source')
        if not isinstance(raw_source, dict):
            raise ValueError('source provenance required')
        source = {key: _text(raw_source.get(key), key) for key in ('provider', 'source_id', 'fetched_at_utc')}
        try:
            require_utc(datetime.fromisoformat(source['fetched_at_utc'].replace('Z', '+00:00')))
        except (ValueError, TypeError):
            raise ValueError('source fetch timestamp must include timezone') from None
        flags = data.get('quality_flags', [])
        if not isinstance(flags, (list, tuple)) or len(flags) > 100:
            raise ValueError('quality_flags must be bounded')
        flags = tuple(_text(flag, 'quality_flag') for flag in flags)
        entries, drivers = [], set()
        for row in _rows(data, 'entries', 64):
            driver = _text(row.get('driver_id'), 'driver_id')
            if driver in drivers:
                raise ValueError('duplicate driver identity')
            drivers.add(driver)
            # Outcome fields in session results belong to the end of a race.
            entries.append({key: _text(row.get(key), key, optional=True)
                            for key in ('driver_id', 'abbreviation', 'team_id', 'display_name')})
        if not drivers:
            raise ValueError('replay requires session entries')
        laps, seen, markers = [], set(), []
        for row in _rows(data, 'laps', 20_000):
            driver, lap_number = _text(row.get('driver_id'), 'driver_id'), row.get('lap_number')
            if driver not in drivers or type(lap_number) is not int or lap_number <= 0:
                raise ValueError('invalid lap identity')
            if (driver, lap_number) in seen:
                raise ValueError('duplicate lap identity')
            seen.add((driver, lap_number))
            lap = dict(driver_id=driver, lap_number=lap_number, clock_ms=_clock(row))
            for key in ('lap_time_s', 'tyre_age_laps', 'stint'):
                value = row.get(key)
                lap[key] = _number(value, key, minimum=0) if value is not None else None
            for key in ('compound', 'track_status'):
                lap[key] = _text(row.get(key), key, optional=True)
            for key in ('pit_in', 'pit_out', 'accurate', 'deleted'):
                value = row.get(key)
                if value is not None and type(value) is not bool:
                    raise ValueError(f'{key} must be boolean or unknown')
                lap[key] = value
            laps.append(lap)
            timing_known = lap['lap_time_s'] is not None and lap['lap_time_s'] > 0
            for kind in ('pit_in', 'pit_out'):
                if lap[kind]:
                    markers.append(dict(clock_ms=lap['clock_ms'], kind=kind, driver_id=driver,
                                        message=f'{kind.replace("_", " ")} provider record for lap {lap_number}'
                                        + ('' if timing_known else ' (unverified lap timing)')))
            if lap['track_status'] and lap['track_status'] != '1':
                markers.append(dict(clock_ms=lap['clock_ms'], kind='track_status', driver_id=driver,
                                    message=f'Lap {lap_number} provider track status code {lap["track_status"]}'
                                    + ('' if timing_known else ' (unverified lap timing)')))
        laps.sort(key=lambda row: (row['clock_ms'], row['driver_id'], row['lap_number']))
        previous = {}
        for lap in laps:
            if lap['lap_number'] <= previous.get(lap['driver_id'], 0):
                raise ValueError('lap numbers must increase with the session clock')
            previous[lap['driver_id']] = lap['lap_number']
        positions, seen = [], set()
        for row in _rows(data, 'positions', 250_000):
            driver, clock = _text(row.get('driver_id'), 'driver_id'), _clock(row)
            if driver not in drivers or (driver, clock) in seen:
                raise ValueError('invalid or duplicate position sample identity')
            seen.add((driver, clock))
            positions.append(dict(driver_id=driver, clock_ms=clock,
                                  x=_number(row.get('x'), 'x'), y=_number(row.get('y'), 'y')))
        positions.sort(key=lambda row: (row['clock_ms'], row['driver_id']))
        for row in _rows(data, 'race_control', 10_000):
            markers.append(dict(clock_ms=_clock(row), kind='race_control',
                                message=_text(row.get('message'), 'race_control message')))
        markers.sort(key=lambda row: (row['clock_ms'], row['kind'], row.get('driver_id', '')))
        weather = []
        for row in _rows(data, 'weather', 10_000):
            sample = dict(clock_ms=_clock(row))
            for key in WEATHER_FIELDS:
                value = row.get(key)
                sample[key] = _number(value, key) if value is not None else None
            if sample['rainfall_flag'] not in (None, 0, 1):
                raise ValueError('rainfall_flag must be zero, one or unknown')
            weather.append(sample)
        weather.sort(key=lambda row: row['clock_ms'])
        timing = []
        basis = data.get('timing_basis', 'derived_timed_lap_record_order')
        if basis not in {'timing_stream', 'lap_end_position', 'derived_timed_lap_record_order'}:
            raise ValueError('unsupported timing basis')
        if session_id not in {'R', 'S'}:
            basis = 'derived_timed_lap_record_order'
        for row in _rows(data, 'timing', 250_000):
            driver = _text(row.get('driver_id'), 'timing driver_id')
            if driver not in drivers:
                raise ValueError('unknown timing driver')
            sample = dict(driver_id=driver, clock_ms=_clock(row))
            state_snapshot = row.get('state_snapshot', False)
            if type(state_snapshot) is not bool:
                raise ValueError('timing state_snapshot must be boolean')
            position = row.get('position')
            if position is not None:
                if type(position) is not int or not 0 <= position <= 64:
                    raise ValueError('invalid timing position')
                # Input zero is an explicit unavailable update, never a rank.
                sample['position'] = position or None
            elif state_snapshot:
                sample['position'] = None
            for raw_key, field in (('gap_to_leader_raw', 'gap'), ('interval_to_ahead_raw', 'interval')):
                if row.get(raw_key) is not None:
                    sample[field] = _gap(row[raw_key])
                elif state_snapshot:
                    sample[field] = (None, None, None)
            if basis != 'derived_timed_lap_record_order':
                timing.append(sample)
        timing.sort(key=lambda row: (row['clock_ms'], row['driver_id']))
        duration = max((row['clock_ms'] for rows in (laps, positions, markers, weather, timing) for row in rows), default=0)
        if duration > 24 * 60 * 60 * 1000:
            raise ValueError('replay duration exceeds 24 hour bound')
        normalized = dict(entries=entries, laps=laps, positions=positions, markers=markers, weather=weather,
                          timing=timing, timing_basis=basis, source=source, flags=flags)
        version = FRAME_VERSION + ':' + sha256(_json(normalized).encode()).hexdigest()
        return cls(f'{event.event_id}:{session_id}', event, session_id, version, tuple(entries), tuple(laps),
                   tuple(positions), tuple(markers), tuple(weather), tuple(timing), basis, source, flags, duration)

    @property
    def track_unavailable_reason(self):
        return None if self.positions else 'No recorded coordinates in this source; track positions and overtakes are unavailable.'

    @property
    def timed_laps(self):
        return tuple(lap for lap in self.laps if lap['lap_time_s'] is not None and lap['lap_time_s'] > 0)

    def _validate_clock(self, clock_ms):
        if type(clock_ms) is not int or not 0 <= clock_ms <= self.duration_ms:
            raise ValueError('clock_ms must be an integer within this session')

    def frame_at(self, clock_ms: int, *, hindsight=False) -> ReplaySnapshot:
        self._validate_clock(clock_ms)
        if type(hindsight) is not bool:
            raise ValueError('hindsight must be boolean')
        raw_prefix = [lap for lap in self.laps if lap['clock_ms'] <= clock_ms]
        prefix = [lap for lap in self.timed_laps if lap['clock_ms'] <= clock_ms]
        latest = {lap['driver_id']: lap for lap in prefix}
        counts = {}
        for lap in prefix:
            counts[lap['driver_id']] = counts.get(lap['driver_id'], 0) + 1
        coordinates = {sample['driver_id']: sample for sample in self.positions if sample['clock_ms'] <= clock_ms}
        timing = _timing_at(self.timing, clock_ms, self.timing_basis)
        drivers = {}
        for entry in self.entries:
            driver = entry['driver_id']
            lap, sample = latest.get(driver), coordinates.get(driver)
            valid_sample = sample and clock_ms - sample['clock_ms'] <= POSITION_MAX_AGE_MS
            drivers[driver] = dict(entry, completed_laps=counts.get(driver, 0),
                                   last_lap_number=lap['lap_number'] if lap else None,
                                   last_lap_time_s=lap['lap_time_s'] if lap else None,
                                   last_lap_clock_ms=lap['clock_ms'] if lap else None,
                                   compound=lap['compound'] if lap else None,
                                   tyre_age_laps=lap['tyre_age_laps'] if lap else None,
                                   position=None, gap_s=None, gap_laps=None, gap_label=None,
                                   interval_s=None, interval_laps=None, interval_label=None,
                                   position_fresh=False, gap_fresh=False, interval_fresh=False,
                                   timing_fresh=False, timing_basis=self.timing_basis,
                                   coordinates=dict(x=sample['x'], y=sample['y'], sample_clock_ms=sample['clock_ms'])
                                   if valid_sample else None)
            drivers[driver].update(timing.get(driver, {}))
        ordered = sorted(drivers, key=lambda d: (-drivers[d]['completed_laps'],
                         drivers[d]['last_lap_clock_ms'] if drivers[d]['last_lap_clock_ms'] is not None else float('inf'), d))
        completion = {driver: index + 1 if drivers[driver]['completed_laps'] else None for index, driver in enumerate(ordered)}
        if self.timing_basis != 'derived_timed_lap_record_order':
            ordered = sorted(drivers, key=lambda d: (drivers[d]['position'] or float('inf'), d))
        drivers = {driver: dict(drivers[driver], completion_order=completion[driver]) for driver in ordered}
        observed_markers = [marker for marker in self.markers if marker['clock_ms'] <= clock_ms]
        analytical_laps = self.timed_laps if hindsight else tuple(prefix)
        excluded_timing = len(self.laps) - len(analytical_laps) if hindsight else len(raw_prefix) - len(prefix)
        stints = {}
        for lap in analytical_laps:
            if lap['stint'] is None and lap['compound'] is None:
                continue
            key = (lap['driver_id'], lap['stint'], lap['compound'])
            stint = stints.setdefault(key, dict(driver_id=key[0], stint=key[1], compound=key[2],
                                                first_lap=lap['lap_number'], last_lap=lap['lap_number'], observed_laps=0))
            stint['last_lap'] = lap['lap_number']
            stint['observed_laps'] += 1
        provenance = dict(session_id=self.identity, frame_version=self.frame_version, committed_clock_ms=clock_ms,
                          source=deepcopy(self.source), quality_flags=list(self.quality_flags),
                          analysis_scope='full_session_hindsight' if hindsight else 'observed_clock_prefix',
                          forecast_eligible=False, historical_availability='unverified',
                          tower_basis=self.timing_basis, untimed_lap_records_excluded=excluded_timing)
        weather = next((row for row in reversed(self.weather) if row['clock_ms'] <= clock_ms), None)
        weather = deepcopy(weather) if weather and clock_ms - weather['clock_ms'] <= WEATHER_MAX_AGE_MS else {}
        return ReplaySnapshot(ReplayFrame(clock_ms / 1000, drivers, weather,
                              deepcopy([m for m in observed_markers if m['kind'] == 'race_control'])),
                              deepcopy(tuple(analytical_laps)), deepcopy(tuple(stints.values())),
                              deepcopy(tuple(observed_markers)), provenance)

    def browser_payload(self, *, max_bytes=MAX_PAYLOAD_BYTES, max_position_samples=MAX_POSITION_SAMPLES):
        if type(max_bytes) is not int or max_bytes <= 0 or type(max_position_samples) is not int or max_position_samples < 2:
            raise ValueError('payload bounds must be positive integers')
        grouped = {entry['driver_id']: [] for entry in self.entries}
        for sample in self.positions:
            grouped[sample['driver_id']].append(sample)
        occupied = [samples for samples in grouped.values() if samples]
        if sum(min(2, len(samples)) for samples in occupied) > max_position_samples:
            raise ValueError('payload sample bound cannot retain driver endpoints')
        positions = list(self.positions)
        if len(positions) > max_position_samples:
            counts = [min(2, len(samples)) for samples in occupied]
            while sum(counts) < max_position_samples:
                index = max(range(len(occupied)), key=lambda i: len(occupied[i]) / counts[i] if counts[i] < len(occupied[i]) else -1)
                counts[index] += 1
            positions = [samples[round(i * (len(samples) - 1) / (count - 1))] if count > 1 else samples[0]
                         for samples, count in zip(occupied, counts) for i in range(count)]
            positions.sort(key=lambda row: (row['clock_ms'], row['driver_id']))
        payload = dict(schema_version=FRAME_VERSION, session_id=self.identity, frame_version=self.frame_version,
                       duration_ms=self.duration_ms, entries=self.entries, laps=self.laps, positions=positions,
                       markers=self.markers, weather=self.weather, weather_max_age_ms=WEATHER_MAX_AGE_MS,
                       timing=self.timing, timing_basis=self.timing_basis, timing_max_age_ms=TIMING_MAX_AGE_MS,
                       position_max_age_ms=POSITION_MAX_AGE_MS,
                       position_samples_original=len(self.positions), track_unavailable_reason=self.track_unavailable_reason)
        if len(_json(payload).encode('utf-8')) > max_bytes:
            raise ValueError('replay payload exceeds byte limit; native static seek remains available')
        return deepcopy(payload)


def load_replay(path: str | Path, *, max_bytes=MAX_FILE_BYTES) -> ReplaySession:
    path = Path(path)
    if path.stat().st_size > max_bytes:
        raise ValueError('normalized session exceeds file limit')
    raw = path.read_bytes()
    if len(raw) > max_bytes:
        raise ValueError('normalized session exceeds file limit')
    try:
        return ReplaySession.from_mapping(json.loads(raw))
    except (UnicodeError, json.JSONDecodeError, TypeError, KeyError) as error:
        raise ValueError('invalid normalized session') from error


def validate_commit(message, session: ReplaySession, *, last_sequence=0):
    required = {'session_id', 'frame_version', 'sequence', 'clock_ms', 'driver_ids', 'event'}
    if not isinstance(message, dict) or set(message) != required:
        raise ValueError('invalid component event schema')
    if message['session_id'] != session.identity or message['frame_version'] != session.frame_version:
        raise ValueError('stale component session or version')
    if type(message['sequence']) is not int or not last_sequence < message['sequence'] < 2**53:
        raise ValueError('stale or invalid component sequence')
    session._validate_clock(message['clock_ms'])
    drivers = message['driver_ids']
    if (not isinstance(drivers, list) or not 1 <= len(drivers) <= len(session.entries) or any(not isinstance(d, str) for d in drivers)
            or len(set(drivers)) != len(drivers)):
        raise ValueError('invalid component driver selection')
    if any(not isinstance(d, str) or d not in {e['driver_id'] for e in session.entries} for d in drivers):
        raise ValueError('unknown component driver')
    if not isinstance(message['event'], str) or message['event'] not in {'seek', 'pause', 'selection'}:
        raise ValueError('unsupported component commit event')
    return deepcopy(message)
