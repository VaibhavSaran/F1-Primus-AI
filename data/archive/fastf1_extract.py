"""Lossless projections of already loaded FastF1 data; no fetching/interpolation.

Raw exports retain all provider columns. Normalized clocks are integer nanoseconds;
FastF1's naive Date/t0_date and schedule DateUtc values are documented UTC values.
Unknown driver identities remain session scoped rather than matching by car number.
"""

from __future__ import annotations

import hashlib
import json
import re
from decimal import Decimal, InvalidOperation

import numpy as np
import pandas as pd

NORMALIZER_VERSION = 'fastf1-local-2'
SESSION_CODES = {'Practice 1': 'FP1', 'Practice 2': 'FP2', 'Practice 3': 'FP3',
                 'Qualifying': 'Q', 'Sprint Shootout': 'SQ',
                 'Sprint Qualifying': 'SQ', 'Sprint': 'S', 'Race': 'R'}


def _missing(value):
    return value is None or bool(pd.isna(value))


def _text(value):
    return None if _missing(value) or str(value).strip() == '' else str(value)


def _identity(value):
    # FastF1 string columns can turn missing provider IDs into literal sentinels.
    value = _text(value)
    return None if value is None or value.strip().casefold() in {'nan', '<na>', 'none', 'null', 'nat'} else value


def _number(value):
    if _missing(value):
        return None
    return value.item() if isinstance(value, np.generic) else value


def _integer(value):
    value = _number(value)
    if value is None:
        return None
    try:
        integer = int(value)
        return integer if float(value) == integer else None
    except (TypeError, ValueError, OverflowError):
        return None


def _boolean(value):
    if _missing(value):
        return None
    if isinstance(value, str):
        return {'true': 1, 'false': 0, '1': 1, '0': 0}.get(value.lower())
    return int(value) if value in (True, False) else None


def _ns(value, *, utc=False):
    if _missing(value):
        return None
    if utc:
        stamp = pd.Timestamp(value)
        stamp = stamp.tz_localize('UTC') if stamp.tzinfo is None else stamp.tz_convert('UTC')
        return int(stamp.value)
    return int(pd.Timedelta(value).value)


def _utc(clock, t0):
    return None if clock is None or t0 is None else clock + t0


def _rows(frame):
    # iterrows can coerce an integer timestamp to float in a mixed numeric row.
    columns = {name: frame[name] for name in frame.columns}
    for index in range(len(frame)):
        yield {name: values.iat[index] for name, values in columns.items()}


def _slug(value):
    return re.sub(r'[^a-z0-9]+', '-', value.casefold()).strip('-')


def calendar_rows(year: int, schedule: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """Return championship events/sessions, without declaring observations complete.

    Testing entries have round zero and are outside this championship archive.
    Circuit identity is a provider location/country reference, not a layout claim.
    """
    events, sessions = [], []
    for row in _rows(schedule):
        round_number = _integer(row.get('RoundNumber'))
        if round_number is None or round_number <= 0:
            continue
        event_id = f'{year}:{round_number}'
        country, location = _text(row.get('Country')), _text(row.get('Location'))
        circuit_id = (f'fastf1-location:{_slug(country or "unknown")}:{_slug(location)}'
                      if location else None)
        date = row.get('EventDate')
        events.append(dict(event_id=event_id, season=int(year), round_number=round_number,
            event_name=_text(row.get('EventName')), official_name=_text(row.get('OfficialEventName')),
            circuit_id=circuit_id, circuit_name=location, country=country, location=location,
            event_format_raw=_text(row.get('EventFormat')),
            local_race_date=None if _missing(date) else str(pd.Timestamp(date).date()),
            status='scheduled'))
        for index in range(1, 6):
            name = _text(row.get(f'Session{index}'))
            if not name:
                continue
            code = SESSION_CODES.get(name, f'OTHER-{_slug(name)}')
            if name == 'Sprint Qualifying' and int(year) <= 2022:
                code = 'S'
            sessions.append(dict(session_id=f'{event_id}:{code}', event_id=event_id,
                session_code=code, full_name=name, original_provider_name=name,
                scheduled_start_utc_ns=_ns(row.get(f'Session{index}DateUtc'), utc=True),
                source_timezone_name='UTC', status='scheduled'))
    return events, sessions


def _get(raw, name):
    try:
        return getattr(raw, name), None
    except Exception as exc:
        # Unloaded FastF1 properties raise DataNotLoadedError instead of returning None.
        return None, f'{type(exc).__name__}: {exc}'


def _dense(frame, driver_id, kind):
    table = pd.DataFrame(frame).copy(deep=True)
    table['driver_id'] = driver_id
    table['sample_index'] = range(len(table))
    for original, normalized, is_utc in [('Date', 'timestamp_utc_ns', True),
            ('SessionTime', 'session_clock_ns', False), ('Time', 'slice_time_ns', False)]:
        values = table[original] if original in table else [None] * len(table)
        table[normalized] = pd.array([_ns(value, utc=is_utc) for value in values], dtype='Int64')
    channels = ({'Speed': 'speed_kmh', 'RPM': 'rpm', 'nGear': 'gear',
                 'Throttle': 'throttle_pct_raw', 'DRS': 'drs_code', 'Source': 'sample_source'}
                if kind == 'car' else {'X': 'x_raw', 'Y': 'y_raw', 'Z': 'z_raw',
                    'Status': 'position_status_raw', 'Source': 'sample_source'})
    for original, normalized in channels.items():
        values = table[original] if original in table else [None] * len(table)
        table[normalized] = (pd.array([_integer(value) for value in values], dtype='Int64')
                             if normalized in ('gear', 'drs_code') else values)
    if kind == 'car':
        values = table['Brake'] if 'Brake' in table else [None] * len(table)
        table['brake_applied'] = pd.array([_boolean(value) for value in values], dtype='boolean')
    flags = []
    for row in _rows(table):
        quality = []
        if _missing(row.get('timestamp_utc_ns')):
            quality.append('missing_original_timestamp')
        throttle = row.get('Throttle')
        if kind == 'car' and not _missing(throttle) and not 0 <= throttle <= 100:
            quality.append('throttle_out_of_range')
        flags.append(json.dumps(quality))
    table['quality_flags'] = flags
    return table


def extract_session(raw_session, session_id: str) -> dict:
    """Extract normalized SQLite rows and original per-driver Parquet streams.

    The collector must request optional circuit information and attach it as
    ``circuit_info`` (or ``circuitinfo``); this function never calls get_circuit_info.
    Rows lack version_id, which the storage transaction injects atomically.
    """
    tables = {name: [] for name in ['drivers', 'teams', 'session_entries', 'session_results',
        'laps', 'track_status_events', 'session_status_events', 'race_control_messages',
        'weather_series', 'weather_observations', 'circuit_markers']}
    datasets, completeness, quality_flags = [], {}, []
    t0_value, t0_reason = _get(raw_session, 't0_date')
    t0 = _ns(t0_value, utc=True)
    if t0 is None:
        quality_flags.append('unknown_utc_clock_origin')
    frames = {}
    properties = {'results': 'results', 'laps': 'laps', 'weather': 'weather_data',
                  'track_status': 'track_status', 'session_status': 'session_status',
                  'race_control': 'race_control_messages'}
    for kind, attribute in properties.items():
        value, reason = _get(raw_session, attribute)
        frames[kind] = pd.DataFrame(value).copy(deep=True) if isinstance(value, pd.DataFrame) else pd.DataFrame()
        frame = frames[kind]
        if isinstance(value, pd.DataFrame):
            datasets.append(dict(kind=f'raw_{kind}', driver_id=None, table=frame))
        completeness[kind] = dict(status='available' if len(frame) else 'unavailable',
            rows=len(frame), reason=reason if reason else (None if len(frame) else 'No loaded rows'))
    streams = {}
    for kind, attribute in [('car', 'car_data'), ('positions', 'pos_data')]:
        value, reason = _get(raw_session, attribute)
        streams[kind] = {str(number): pd.DataFrame(frame).copy(deep=True)
                         for number, frame in (value or {}).items() if isinstance(frame, pd.DataFrame)}
        completeness[kind] = dict(reason=reason)

    results = list(_rows(frames['results']))
    laps = list(_rows(frames['laps']))
    by_number = {}
    for row in results:
        number = _text(row.get('DriverNumber'))
        if number:
            by_number[number] = row
    for row in laps:
        number = _text(row.get('DriverNumber'))
        if number and number not in by_number:
            by_number[number] = {'DriverNumber': number, 'Abbreviation': row.get('Driver'),
                                 'TeamName': row.get('Team')}
    for stream in streams.values():
        for number in stream:
            by_number.setdefault(number, {'DriverNumber': number})
    number_to_id, teams = {}, {}
    for number, row in by_number.items():
        source_driver = _identity(row.get('DriverId'))
        if source_driver is None and 'unresolved_driver_identity' not in quality_flags:
            quality_flags.append('unresolved_driver_identity')
        driver_id = f'fastf1:driver:{source_driver}' if source_driver else f'fastf1:{session_id}:number:{number}'
        number_to_id[number] = driver_id
        source_team, team_name = _identity(row.get('TeamId')), _text(row.get('TeamName'))
        if source_team is None and 'unresolved_team_identity' not in quality_flags:
            quality_flags.append('unresolved_team_identity')
        team_id = (f'fastf1:team:{source_team}' if source_team else
                   f'fastf1:{session_id}:team:{_slug(team_name)}' if team_name else None)
        if team_id:
            teams[team_id] = dict(team_id=team_id, canonical_name=team_name, source_team_id=source_team)
        tables['drivers'].append(dict(driver_id=driver_id, first_name=_text(row.get('FirstName')),
            last_name=_text(row.get('LastName')), display_name=_text(row.get('FullName')),
            country_code=_text(row.get('CountryCode')), source_driver_id=source_driver))
        tables['session_entries'].append(dict(driver_id=driver_id, team_id=team_id,
            driver_number=number, abbreviation=_text(row.get('Abbreviation')),
            broadcast_name=_text(row.get('BroadcastName')), team_name_at_event=team_name,
            team_color_at_event=_text(row.get('TeamColor')), source_driver_id=source_driver,
            source_team_id=source_team))
    tables['teams'] = list(teams.values())
    for row in results:
        driver_id = number_to_id.get(_text(row.get('DriverNumber')))
        if not driver_id:
            quality_flags.append('result_without_driver_number_retained_raw')
            continue
        status = _text(row.get('Status'))
        classified = _text(row.get('ClassifiedPosition'))
        outcome = ('finished' if status == 'Finished' else 'lapped' if status and status.startswith('+')
                   else 'disqualified' if classified == 'D' or status == 'Disqualified'
                   else 'withdrawn' if classified == 'W' else None)
        tables['session_results'].append(dict(driver_id=driver_id,
            finishing_position=_number(row.get('Position')), classified_position_raw=classified,
            grid_position=_number(row.get('GridPosition')), q1_ns=_ns(row.get('Q1')),
            q2_ns=_ns(row.get('Q2')), q3_ns=_ns(row.get('Q3')), reported_time_ns=_ns(row.get('Time')),
            time_semantics='fastf1_reported_total_race_time' if _ns(row.get('Time')) is not None else None,
            points=_number(row.get('Points')), laps_completed=_number(row.get('Laps')),
            status_raw=status, normalized_outcome_status=outcome, original_publication_utc=None))

    lap_clocks = {'LapStartTime': 'lap_start_clock_ns', 'Time': 'lap_end_clock_ns',
        'LapTime': 'lap_time_ns', 'Sector1Time': 'sector1_ns', 'Sector2Time': 'sector2_ns',
        'Sector3Time': 'sector3_ns', 'Sector1SessionTime': 'sector1_end_clock_ns',
        'Sector2SessionTime': 'sector2_end_clock_ns', 'Sector3SessionTime': 'sector3_end_clock_ns',
        'PitInTime': 'pit_in_clock_ns', 'PitOutTime': 'pit_out_clock_ns'}
    lap_numbers = {'Stint': 'stint_number', 'SpeedI1': 'speed_i1_kmh', 'SpeedI2': 'speed_i2_kmh',
        'SpeedFL': 'speed_finish_line_kmh', 'SpeedST': 'speed_trap_kmh',
        'TyreLife': 'tyre_age_laps', 'Position': 'lap_position'}
    lap_bools = {'FreshTyre': 'fresh_tyre', 'IsPersonalBest': 'is_personal_best',
        'Deleted': 'deleted', 'IsAccurate': 'provider_is_accurate', 'FastF1Generated': 'fastf1_generated'}
    normalized_laps = {}
    for row in laps:
        driver_id = number_to_id.get(_text(row.get('DriverNumber')))
        lap_number = _integer(row.get('LapNumber'))
        if not driver_id or lap_number is None or lap_number <= 0:
            quality_flags.append('unkeyed_lap_retained_raw')
            continue
        lap = dict(driver_id=driver_id, lap_number=lap_number)
        lap.update({target: _ns(row.get(original)) for original, target in lap_clocks.items()})
        lap.update({target: (_integer(row.get(original)) if original == 'Stint' else _number(row.get(original)))
                    for original, target in lap_numbers.items()})
        lap.update({target: _boolean(row.get(original)) for original, target in lap_bools.items()})
        compound = _text(row.get('Compound'))
        flags = []
        if lap['fastf1_generated'] == 1:
            flags.append('provider_generated_lap')
        if lap['provider_is_accurate'] != 1:
            flags.append('provider_accuracy_not_confirmed')
        if lap['lap_time_ns'] is None:
            flags.append('missing_lap_time')
        if (driver_id, lap_number) in normalized_laps:
            flags.append('duplicate_lap_key_last_source_row')
            quality_flags.append('duplicate_lap_key_retained_raw')
        lap.update(lap_start_utc_ns=_ns(row.get('LapStartDate'), utc=True), compound_raw=compound,
            compound_normalized=compound.upper() if compound else None,
            track_status_raw=_text(row.get('TrackStatus')), position_basis='fastf1_lap_order',
            deleted_reason=_text(row.get('DeletedReason')), lap_quality_flags_json=json.dumps(flags))
        normalized_laps[driver_id, lap_number] = lap
    tables['laps'] = list(normalized_laps.values())

    for kind, target in [('track_status', 'track_status_events'), ('session_status', 'session_status_events')]:
        for index, row in enumerate(_rows(frames[kind])):
            clock = _ns(row.get('Time'))
            item = dict(event_index=index, session_clock_ns=clock, timestamp_utc_ns=_utc(clock, t0))
            if kind == 'track_status':
                item.update(status_code_raw=_text(row.get('Status')), message_raw=_text(row.get('Message')))
            else:
                item['status_raw'] = _text(row.get('Status'))
            tables[target].append(item)
    for index, row in enumerate(_rows(frames['race_control'])):
        timestamp = _ns(row.get('Time'), utc=True)
        number = _text(row.get('RacingNumber'))
        tables['race_control_messages'].append(dict(message_index=index, timestamp_utc_ns=timestamp,
            session_clock_ns=None if timestamp is None or t0 is None else timestamp - t0,
            category_raw=_text(row.get('Category')), message_text=_text(row.get('Message')),
            status_raw=_text(row.get('Status')), flag_raw=_text(row.get('Flag')),
            scope_raw=_text(row.get('Scope')), sector=_integer(row.get('Sector')),
            driver_number_raw=number, driver_id=number_to_id.get(number), lap_number=_integer(row.get('Lap'))))

    weather = frames['weather']
    if not weather.empty:
        digest = hashlib.sha256(pd.util.hash_pandas_object(weather, index=True).values.tobytes()
            + str(t0).encode() + str(list(weather.columns)).encode()).hexdigest()
        series_id = f'fastf1:weather:{session_id}:{digest}'
        tables['weather_series'].append(dict(series_id=series_id, provider='fastf1',
            kind='track_observation', coordinate_basis='provider_track_sensor_location_unknown',
            original_timezone='UTC', dataset_version=NORMALIZER_VERSION,
            units_metadata_json=json.dumps({'temperature': 'degC', 'pressure': 'hPa',
                'wind_speed': 'm/s', 'rainfall': 'boolean', 'humidity': '%'}),
            quality_flags_json=json.dumps(['sensor_coordinates_unknown', 'pressure_reference_unspecified'])))
        for index, row in enumerate(_rows(weather)):
            clock = _ns(row.get('Time'))
            item = dict(series_id=series_id, sample_index=index,
                session_clock_ns=clock, valid_time_utc_ns=_utc(clock, t0),
                rainfall_flag=_boolean(row.get('Rainfall')), pressure_kind='unspecified_air_pressure',
                quality_flags=json.dumps([] if t0 is not None else ['unknown_utc_clock_origin']))
            item.update({target: _number(row.get(original)) for original, target in {
                'AirTemp': 'air_temperature_c', 'TrackTemp': 'track_temperature_c',
                'Humidity': 'relative_humidity_pct', 'Pressure': 'pressure_hpa',
                'WindSpeed': 'wind_speed_mps', 'WindDirection': 'wind_direction_deg'}.items()})
            tables['weather_observations'].append(item)

    for kind, stream in streams.items():
        per_driver = {}
        required = ({'Date', 'SessionTime', 'Speed', 'RPM', 'nGear', 'Throttle', 'Brake', 'DRS'}
                    if kind == 'car' else {'Date', 'SessionTime', 'X', 'Y', 'Z', 'Status'})
        for number, driver_id in number_to_id.items():
            frame = stream.get(number)
            missing = sorted(required - set(frame.columns)) if frame is not None else sorted(required)
            valid = int(frame['Date'].notna().sum()) if frame is not None and 'Date' in frame else 0
            row_count = len(frame) if frame is not None else 0
            counts = {column: int(frame[column].notna().sum())
                      if frame is not None and column in frame else 0 for column in sorted(required)}
            per_driver[driver_id] = dict(rows=row_count, valid_timestamps=valid, missing_columns=missing,
                channel_valid_counts=counts, status='unavailable' if not row_count else
                'partial' if missing or any(count < row_count for count in counts.values()) else 'available')
            if frame is not None:
                datasets.append(dict(kind=kind, driver_id=driver_id, table=_dense(frame, driver_id, kind)))
        with_rows = sum(item['rows'] > 0 for item in per_driver.values())
        complete = bool(per_driver) and all(item['status'] == 'available' for item in per_driver.values())
        completeness[kind].update(status='available' if complete else 'partial' if with_rows else 'unavailable',
            rows=sum(item['rows'] for item in per_driver.values()), drivers_with_rows=with_rows,
            expected_drivers=len(number_to_id), drivers=per_driver)

    circuit_info, _ = _get(raw_session, 'circuit_info')
    if circuit_info is None:
        circuit_info, _ = _get(raw_session, 'circuitinfo')
    rotation = None
    if circuit_info is not None:
        rotation = _number(getattr(circuit_info, 'rotation', None))
        for attribute, kind in [('corners', 'corner'), ('marshal_lights', 'marshal_light'),
                                 ('marshal_sectors', 'marshal_sector')]:
            frame = getattr(circuit_info, attribute, None)
            if not isinstance(frame, pd.DataFrame):
                continue
            datasets.append(dict(kind=f'raw_circuit_{attribute}', driver_id=None,
                                 table=pd.DataFrame(frame).copy(deep=True)))
            for index, row in enumerate(_rows(frame)):
                tables['circuit_markers'].append(dict(marker_kind=kind, marker_index=index,
                    kind=kind, number=_integer(row.get('Number')), letter=_text(row.get('Letter')),
                    x_raw=_number(row.get('X')), y_raw=_number(row.get('Y')),
                    angle_deg=_number(row.get('Angle')), distance_m=_number(row.get('Distance')),
                    geometry_basis='fastf1_multiviewer_manual_marker_provider_frame'))
    completeness['circuit_info'] = dict(status='available' if circuit_info is not None else 'unavailable',
        rows=len(tables['circuit_markers']), reason=None if circuit_info is not None else 'Optional circuit information not supplied by collector')
    completeness['timing'] = dict(status='unsupported', rows=0,
        reason='No version-pinned low-level timing stream adapter; normalized laps are not running gap observations')
    timestamps = []
    for rows in tables.values():
        for row in rows:
            for field in ['timestamp_utc_ns', 'valid_time_utc_ns', 'lap_start_utc_ns']:
                if row.get(field) is not None:
                    timestamps.append(row[field])
            if row.get('lap_end_clock_ns') is not None and t0 is not None:
                timestamps.append(t0 + row['lap_end_clock_ns'])
    for dataset in datasets:
        frame = dataset['table']
        if 'timestamp_utc_ns' in frame:
            timestamps.extend(int(value) for value in frame['timestamp_utc_ns'].dropna())
    coverage = [dict(dataset_kind=kind, status=item['status'], rows=item.get('rows', 0),
                     reason=item.get('reason')) for kind, item in completeness.items()]
    for kind in ['car', 'positions']:
        coverage.extend(dict(dataset_kind=f'{kind}/{driver}', status=item['status'], rows=item['rows'],
            valid_count=item['valid_timestamps'], missing_columns_json=json.dumps(item['missing_columns']))
            for driver, item in completeness[kind]['drivers'].items())
    return dict(tables=tables, datasets=datasets, metadata=dict(normalizer_version=NORMALIZER_VERSION,
        t0_utc_ns=t0, t0_unavailable_reason=t0_reason, observed_start_utc_ns=min(timestamps) if timestamps else None,
        observed_end_utc_ns=max(timestamps) if timestamps else None, completeness=completeness,
        quality_flags=sorted(set(quality_flags)), coverage=coverage, circuit_info_rotation_deg=rotation))


def _jolpica_duration(value):
    """Parse the provider's mm:ss.sss or seconds string without a float round trip."""
    text = _text(value)
    if text is None:
        return None
    try:
        seconds = Decimal(0)
        for component in text.lstrip('+').split(':'):
            seconds = seconds * 60 + Decimal(component)
        nanos = seconds * 1_000_000_000
        return int(nanos) if nanos.is_finite() and nanos == int(nanos) else None
    except (InvalidOperation, ValueError, OverflowError):
        return None


def normalize_jolpica(session_id: str, payloads: dict, source_ids: dict | None = None) -> dict:
    """Normalize archived Jolpica JSON pages, without inventing FastF1 channels.

    ``payloads`` maps endpoint kinds to lists of original MRData page dictionaries.
    ``source_ids`` maps each endpoint to its source ID or page-ordered source IDs;
    pit-stop records require the corresponding source evidence FK. The collector
    archives original JSON bytes; this adapter does not serialize replacements.
    """
    source_ids = source_ids or {}
    tables = {name: [] for name in ['drivers', 'teams', 'session_entries', 'session_results',
        'laps', 'pit_stop_records', 'track_status_events', 'session_status_events',
        'race_control_messages', 'weather_series', 'weather_observations', 'circuit_markers']}
    season, round_number, code = session_id.split(':')
    result_key = {'R': 'Results', 'S': 'SprintResults', 'Q': 'QualifyingResults',
                  'SQ': 'QualifyingResults'}.get(code)
    drivers, teams, entries, results, laps, pits = {}, {}, {}, {}, {}, {}
    flags = ['jolpica_fallback_no_absolute_session_clock', 'original_publication_time_unknown']
    raw_rows = {'results': 0, 'laps': 0, 'pit_stops': 0}

    def register(source_driver, constructor=None, event_number=None):
        source_driver_id = _text(source_driver.get('driverId'))
        if not source_driver_id:
            flags.append('row_without_source_driver_id_retained_in_original_json')
            return None
        driver_id = f'jolpica:driver:{source_driver_id}'
        prior = drivers.get(driver_id, {})
        first_name = _text(source_driver.get('givenName')) or prior.get('first_name')
        last_name = _text(source_driver.get('familyName')) or prior.get('last_name')
        drivers[driver_id] = dict(driver_id=driver_id, source_driver_id=source_driver_id,
            first_name=first_name, last_name=last_name,
            display_name=' '.join(part for part in [first_name, last_name] if part) or None,
            country_code=None)
        entry = entries.setdefault(driver_id, dict(driver_id=driver_id, team_id=None,
            driver_number=None, abbreviation=None, broadcast_name=None, team_name_at_event=None,
            team_color_at_event=None, source_driver_id=source_driver_id, source_team_id=None))
        if event_number is not None:
            entry['driver_number'] = _text(event_number)
        if _text(source_driver.get('code')):
            entry['abbreviation'] = _text(source_driver.get('code'))
        constructor = constructor or {}
        source_team_id = _text(constructor.get('constructorId'))
        if source_team_id:
            team_id = f'jolpica:team:{source_team_id}'
            name = _text(constructor.get('name'))
            teams[team_id] = dict(team_id=team_id, source_team_id=source_team_id, canonical_name=name)
            entry.update(team_id=team_id, source_team_id=source_team_id, team_name_at_event=name)
        return driver_id

    for endpoint, pages in payloads.items():
        if isinstance(pages, dict):
            pages = [pages]
        evidence = source_ids.get(endpoint)
        for page_index, page in enumerate(pages):
            source_id = (evidence[page_index] if isinstance(evidence, (list, tuple)) and page_index < len(evidence)
                         else evidence if isinstance(evidence, str) else None)
            races = page.get('MRData', {}).get('RaceTable', {}).get('Races', [])
            for race in races:
                if ((_text(race.get('season')) is not None and int(race['season']) != int(season))
                        or (_text(race.get('round')) is not None and int(race['round']) != int(round_number))):
                    raise ValueError('Jolpica payload event does not match target session')
                if code != 'R' and (race.get('Laps') or race.get('PitStops')):
                    raise ValueError('Jolpica lap and pit-stop endpoints are race observations')
                for row in race.get(result_key, []) if result_key else []:
                    raw_rows['results'] += 1
                    driver_id = register(row.get('Driver', {}), row.get('Constructor'), row.get('number'))
                    if not driver_id:
                        continue
                    time_value = row.get('Time') or {}
                    if isinstance(time_value, str):
                        time_value = {'time': time_value}
                    milliseconds = _integer(time_value.get('millis'))
                    displayed_time = _text(time_value.get('time'))
                    reported_time = (milliseconds * 1_000_000 if milliseconds is not None
                                     else _jolpica_duration(displayed_time))
                    semantics = ('jolpica_total_race_time_millis' if milliseconds is not None
                                 else 'jolpica_reported_gap_to_winner' if displayed_time and displayed_time.startswith('+')
                                 else 'jolpica_reported_total_race_time' if reported_time is not None else None)
                    status = _text(row.get('status'))
                    classified = _text(row.get('positionText'))
                    outcome = ('finished' if status == 'Finished' else
                               'lapped' if status and status.startswith('+') else
                               'disqualified' if status == 'Disqualified' or classified == 'D' else None)
                    results[driver_id] = dict(driver_id=driver_id, finishing_position=_integer(row.get('position')),
                        classified_position_raw=classified, grid_position=_integer(row.get('grid')),
                        q1_ns=_jolpica_duration(row.get('Q1')), q2_ns=_jolpica_duration(row.get('Q2')),
                        q3_ns=_jolpica_duration(row.get('Q3')), reported_time_ns=reported_time,
                        time_semantics=semantics, points=float(row['points']) if row.get('points') is not None else None,
                        laps_completed=_integer(row.get('laps')), status_raw=status,
                        normalized_outcome_status=outcome, original_publication_utc=None)
                for lap in race.get('Laps', []):
                    lap_number = _integer(lap.get('number'))
                    for row in lap.get('Timings', []):
                        raw_rows['laps'] += 1
                        driver_id = register({'driverId': row.get('driverId')})
                        if not driver_id or lap_number is None or lap_number <= 0:
                            flags.append('unkeyed_lap_retained_in_original_json')
                            continue
                        item = dict(driver_id=driver_id, lap_number=lap_number,
                            lap_time_ns=_jolpica_duration(row.get('time')), lap_position=_integer(row.get('position')),
                            position_basis='jolpica_lap_timing_position', lap_quality_flags_json=json.dumps(
                                ['lap_time_and_lap_order_only', 'no_sector_tyre_or_clock_evidence']))
                        for name in ['stint_number', 'lap_start_clock_ns', 'lap_end_clock_ns', 'lap_start_utc_ns',
                                'sector1_ns', 'sector2_ns', 'sector3_ns', 'sector1_end_clock_ns',
                                'sector2_end_clock_ns', 'sector3_end_clock_ns', 'speed_i1_kmh', 'speed_i2_kmh',
                                'speed_finish_line_kmh', 'speed_trap_kmh', 'pit_in_clock_ns', 'pit_out_clock_ns',
                                'compound_raw', 'compound_normalized', 'tyre_age_laps', 'fresh_tyre',
                                'is_personal_best', 'track_status_raw', 'deleted', 'deleted_reason',
                                'provider_is_accurate', 'fastf1_generated']:
                            item[name] = None
                        laps[driver_id, lap_number] = item
                for row in race.get('PitStops', []):
                    raw_rows['pit_stops'] += 1
                    driver_id = register({'driverId': row.get('driverId')})
                    stop = _integer(row.get('stop'))
                    if not driver_id or stop is None or source_id is None:
                        flags.append('pit_stop_requires_source_id_and_stop_number')
                        continue
                    pits[driver_id, stop, source_id] = dict(driver_id=driver_id, source_id=source_id,
                        stop_index=stop, duration_ns=_jolpica_duration(row.get('duration')),
                        duration_kind='provider_reported_pit_stop_duration', source_evidence_json=json.dumps(row, sort_keys=True))
    tables.update(drivers=list(drivers.values()), teams=list(teams.values()), session_entries=list(entries.values()),
                  session_results=list(results.values()), laps=list(laps.values()), pit_stop_records=list(pits.values()))
    completeness = {}
    for kind, rows in [('results', tables['session_results']), ('laps', tables['laps']),
                       ('pit_stops', tables['pit_stop_records'])]:
        available = len(rows)
        completeness[kind] = dict(status='partial' if available and available < raw_rows[kind] else
            'available' if available else 'unavailable', rows=available, source_rows=raw_rows[kind],
            reason='Jolpica normalized fields only; missing channels remain NULL')
    for kind in ['car', 'positions', 'weather', 'track_status', 'session_status', 'race_control', 'circuit_info', 'timing']:
        completeness[kind] = dict(status='unsupported', rows=0,
            reason=f'Jolpica does not provide {kind}; original FastF1 collection status is tracked separately')
    coverage = [dict(dataset_kind=kind, status=item['status'], rows=item['rows'], reason=item.get('reason'))
                for kind, item in completeness.items()]
    return dict(tables=tables, datasets=[], metadata=dict(normalizer_version=f'{NORMALIZER_VERSION}-jolpica',
        t0_utc_ns=None, observed_start_utc_ns=None, observed_end_utc_ns=None,
        completeness=completeness, coverage=coverage, quality_flags=sorted(set(flags))))
