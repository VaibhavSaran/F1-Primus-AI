"""Read-only archive projection into the bounded historical replay contract.

Clock zero is the archive's original FastF1 session t0, including pre-session
time, not the scheduled green light or the first selected driver's sample.
Nanosecond source clocks remain unchanged in SQLite/Parquet. Coordinates are
observed samples, never interpolated; the browser may hide stale observations.
"""
from __future__ import annotations

from math import ceil, isfinite
import json

import pandas as pd

from data.replay import ReplaySession, MAX_POSITION_SAMPLES, WEATHER_FIELDS


def _known(value):
    return value is not None and not pd.isna(value)


def _number(value):
    return float(value) if _known(value) and isfinite(float(value)) else None


def _seconds(value):
    value = _number(value)
    return value / 1_000_000_000 if value is not None else None


def _text(value):
    return str(value) if _known(value) and str(value).strip() else None


def _boolean(value):
    return bool(value) if _known(value) and value in (0, 1) else None


def observed_samples(frame):
    """Exclude parser-created missing-Time fills without changing archive evidence."""
    if 'quality_flags' not in frame:
        return frame
    def observed(value):
        try:
            flags = json.loads(value) if isinstance(value, str) else value
            return isinstance(flags, (list, tuple)) and 'fastf1_parser_generated_fill' not in flags
        except (ValueError, TypeError):
            return False
    return frame.loc[frame.quality_flags.map(observed)]


def build_replay(reader, session_id: str, driver_ids=None) -> ReplaySession:
    """Build a replay for any selected recorded drivers (first two by default).

    A one-second sample grid is used when it fits the 120,000-sample budget;
    longer selections use a coarser grid. Every emitted point is an original
    observation. Empty buckets stay empty, so missing evidence is never filled.
    """
    tables = reader.tables(session_id)
    meta = tables['meta']
    entries = tables['entries'].to_dict('records')
    names = {str(row['driver_id']): _text(row.get('display_name')) or _text(row.get('abbreviation')) or 'Unknown driver'
             for row in entries}
    available = {str(row['driver_id']) for row in entries}
    selected = list(driver_ids) if driver_ids is not None else [str(r['driver_id']) for r in entries[:2]]
    if (not 1 <= len(selected) <= min(64, len(available)) or len(set(selected)) != len(selected)
            or any(driver not in available for driver in selected)):
        raise ValueError('replay requires distinct known drivers within the recorded grid')
    laps, messages, excluded = [], [], 0
    last_clock = {}
    raw_laps = tables['laps'].sort_values(['driver_id', 'lap_number'])
    for row in raw_laps.to_dict('records'):
        driver = str(row['driver_id'])
        if driver not in selected:
            continue
        # Exact pit observations precede lap completion and must not be moved
        # to the lap-end clock by ReplaySession's synthetic marker fallback.
        for kind in ('pit_in', 'pit_out'):
            clock = _seconds(row.get(kind + '_clock_ns'))
            if clock is not None and clock >= 0:
                messages.append({'session_time_s': clock,
                                 'message': f"{names[driver]}: {kind.replace('_', ' ')} observed (lap {row['lap_number']})"})
        clock = _seconds(row.get('lap_end_clock_ns'))
        if clock is None or clock <= 0 or clock <= last_clock.get(driver, -1):
            excluded += 1
            continue
        last_clock[driver] = clock
        laps.append(dict(driver_id=driver, lap_number=int(row['lap_number']), session_time_s=clock,
                         lap_time_s=_seconds(row.get('lap_time_ns')),
                         compound=_text(row.get('compound_normalized')) or _text(row.get('compound_raw')),
                         tyre_age_laps=_number(row.get('tyre_age_laps')), stint=_number(row.get('stint_number')),
                         accurate=_boolean(row.get('provider_is_accurate')), deleted=_boolean(row.get('deleted')),
                         pit_in=None, pit_out=None, track_status=None))
    excluded_markers = 0
    for table, text_column in [('race_control', 'message_text'), ('track_status', 'message_raw')]:
        for row in tables.get(table, pd.DataFrame()).to_dict('records'):
            clock = _seconds(row.get('session_clock_ns'))
            message = _text(row.get(text_column))
            if table == 'track_status':
                message = f"Track: {message or 'status changed'}"
            if clock is not None and clock >= 0 and message:
                if clock > 86400:
                    excluded_markers += 1
                    continue
                messages.append({'session_time_s': clock, 'message': message[:2048]})
    dense = reader.dense(session_id, 'positions', driver_ids=selected)
    positions = []
    if not dense.empty and {'driver_id', 'session_clock_ns', 'x_raw', 'y_raw'}.issubset(dense.columns):
        dense = observed_samples(dense[dense.driver_id.isin(selected)]).copy()
        dense['session_time_s'] = pd.to_numeric(dense.session_clock_ns, errors='coerce') / 1_000_000_000
        for column in ('x_raw', 'y_raw'):
            dense[column] = pd.to_numeric(dense[column], errors='coerce')
        dense = dense.replace([float('inf'), -float('inf')], float('nan'))
        dense = dense.dropna(subset=['session_time_s', 'x_raw', 'y_raw'])
        dense = dense[dense.session_time_s >= 0].sort_values(['driver_id', 'session_time_s'])
        if not dense.empty:
            span = dense.groupby('driver_id').session_time_s.agg(['min', 'max'])
            interval = max(1, ceil(float((span['max'] - span['min']).sum()) / (MAX_POSITION_SAMPLES - 2 * len(selected))))
            dense['_bucket'] = (dense.session_time_s / interval).astype('int64')
            endpoints = dense.groupby('driver_id', sort=False).tail(1)
            dense = pd.concat([dense.drop_duplicates(['driver_id', '_bucket']), endpoints])
            dense = dense.drop_duplicates(['driver_id', 'session_time_s']).sort_values(['driver_id', 'session_time_s'])
            for row in dense.to_dict('records'):
                x, y = _number(row['x_raw']), _number(row['y_raw'])
                if x is not None and y is not None:
                    positions.append(dict(driver_id=str(row['driver_id']), session_time_s=float(row['session_time_s']), x=x, y=y))
    season, round_number, code = session_id.split(':')
    flags = ['historical_archive_not_publication_proof', 'positions_observed_bucket_sampling_no_interpolation',
             'clock_origin_original_fastf1_t0_includes_pre_session']
    if excluded:
        flags.append(f'excluded_laps_without_monotonic_positive_end_clock:{excluded}')
    if excluded_markers:
        flags.append(f'excluded_markers_outside_replay_clock:{excluded_markers}')
    weather = []
    for row in tables.get('weather', pd.DataFrame()).to_dict('records'):
        clock = _seconds(row.get('session_clock_ns'))
        if clock is not None and clock >= 0:
            weather.append(dict(session_time_s=clock, **{key: _number(row.get(key)) for key in WEATHER_FIELDS}))
    timing = []
    timing_basis = 'derived_timed_lap_record_order'
    if hasattr(reader, 'timing'):
        timing_frame = reader.timing(session_id, version_id=meta['version_id'])
        timing_basis = timing_frame.attrs.get('timing_basis', 'unavailable')
        if timing_basis == 'unavailable':
            timing_basis = 'derived_timed_lap_record_order'
        flags.append('timing_source_status:' + timing_frame.attrs.get('source_status', 'absent'))
        if timing_basis == 'timing_stream':
            flags.append('parsed_provider_state_forward_filled')
        for row in timing_frame.to_dict('records'):
            if str(row['driver_id']) not in selected:
                continue
            position = _number(row.get('position'))
            timing.append(dict(driver_id=str(row['driver_id']), session_time_s=float(row['session_time_s']),
                               state_snapshot=timing_basis == 'timing_stream',
                               position=int(position) if position is not None else None,
                               gap_to_leader_raw=_text(row.get('gap_to_leader_raw')),
                               interval_to_ahead_raw=_text(row.get('interval_to_ahead_raw'))))
    return ReplaySession.from_mapping(dict(
        schema_version='session-v1', season=int(season), round_number=int(round_number), event_id=f'{season}:{round_number}',
        session_id=code, entries=[{key: _text(row.get(key)) for key in ('driver_id', 'abbreviation', 'team_id', 'display_name')}
                                 for row in entries if str(row['driver_id']) in selected],
        laps=laps, positions=positions, race_control=messages, weather=weather, quality_flags=flags,
        timing=timing, timing_basis=timing_basis,
        source={'provider': 'local_fastf1_archive', 'source_id': str(meta['version_id']),
                'fetched_at_utc': meta['created_at_utc']}))
