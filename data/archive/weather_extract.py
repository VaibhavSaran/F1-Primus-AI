"""Circuit-level ERA5 reanalysis. Never a historical pre-race forecast.

Variable units and preceding-hour precipitation intervals follow:
https://open-meteo.com/en/docs/historical-weather-api
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
import json
import math


ENDPOINT = 'https://archive-api.open-meteo.com/v1/archive'
DATASET_VERSION = 'open_meteo_era5_hourly_v1'
FIELDS = {
    'temperature_2m': ('air_temperature_c', '°C'),
    'relative_humidity_2m': ('relative_humidity_pct', '%'),
    'dew_point_2m': ('dew_point_c', '°C'),
    'surface_pressure': ('pressure_hpa', 'hPa'),
    'wind_speed_10m': ('wind_speed_mps', 'm/s'),
    'wind_direction_10m': ('wind_direction_deg', '°'),
    'wind_gusts_10m': ('wind_gust_mps', 'm/s'),
    'rain': ('rain_mm', 'mm'),
    'precipitation': ('precipitation_mm', 'mm'),
    'cloud_cover': ('cloud_cover_pct', '%'),
    'weather_code': ('weather_code', 'wmo code'),
}
FORECAST_COVERAGE = {
    'status': 'unsupported',
    'reason': 'Historical forecast publication and model-run availability are unverified; reanalysis is retrospective only.',
}


def _utc(value):
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValueError('weather window requires timezone-aware datetimes')
    return value.astimezone(timezone.utc)


def _coordinates(circuit):
    if not circuit.get('circuit_id') or not circuit.get('coordinate_source_id') or not circuit.get('coordinate_basis'):
        raise ValueError('circuit coordinates require circuit ID and versioned source provenance')
    try:
        latitude, longitude = float(circuit['latitude']), float(circuit['longitude'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('circuit latitude/longitude unavailable') from exc
    if not math.isfinite(latitude) or not math.isfinite(longitude) or not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError('invalid circuit latitude/longitude')
    return latitude, longitude


def _ns(value):
    return int(value.timestamp()) * 10**9 + value.microsecond * 1000


def normalize_weather(payload, circuit, *, start, end):
    latitude, longitude = _coordinates(circuit)
    start, end = _utc(start), _utc(end)
    if end < start:
        raise ValueError('weather window end precedes start')
    if not isinstance(payload, dict) or payload.get('error'):
        raise ValueError('invalid historical weather response')
    if payload.get('utc_offset_seconds') != 0 or payload.get('timezone') not in ('UTC', 'GMT', 'Etc/UTC', 'Etc/GMT'):
        raise ValueError('weather response must explicitly use UTC')
    hourly, units = payload.get('hourly'), payload.get('hourly_units')
    if not isinstance(hourly, dict) or not isinstance(hourly.get('time'), list) or not isinstance(units, dict):
        raise ValueError('historical weather hourly schema unavailable')
    times = hourly['time']
    columns = {key: values for key, values in hourly.items() if key != 'time'}
    if any(not isinstance(values, list) or len(values) != len(times) for values in columns.values()):
        raise ValueError('historical weather columns are misaligned')
    if not set(columns).intersection(FIELDS):
        raise ValueError('historical weather measurements unavailable')
    missing_columns = sorted(set(FIELDS) - set(columns))
    for key in columns:
        if key not in FIELDS:
            continue
        if units.get(key) != FIELDS[key][1]:
            raise ValueError(f'unexpected historical weather unit for {key}: {units.get(key)}')
    try:
        returned_latitude = float(payload['latitude'])
        returned_longitude = float(payload['longitude'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('historical weather grid coordinates unavailable') from exc
    if not math.isfinite(returned_latitude) or not math.isfinite(returned_longitude) or not -90 <= returned_latitude <= 90 or not -180 <= returned_longitude <= 180:
        raise ValueError('invalid historical weather grid coordinates')
    flags = ['regional_reanalysis_not_track_sensor', 'track_temperature_unavailable']
    if missing_columns:
        flags.append('weather_columns_missing')
    observations = []
    # Preserve the hour at/before session start for bounded as-of joins.
    lower = start.replace(minute=0, second=0, microsecond=0)
    expected_times = {_ns(lower + timedelta(hours=hour)) for hour in range(int((end - lower).total_seconds() // 3600) + 1)}
    for index, raw_time in enumerate(times):
        if not isinstance(raw_time, str) or 'T' not in raw_time:
            raise ValueError('historical weather time must include hour')
        valid = datetime.fromisoformat(raw_time.replace('Z', '+00:00'))
        if valid.utcoffset() is None:
            valid = valid.replace(tzinfo=timezone.utc)
        valid = _utc(valid)
        if not lower <= valid <= end:
            continue
        row = {
            'sample_index': index, 'valid_time_utc_ns': _ns(valid),
            'session_clock_ns': None, 'track_temperature_c': None,
            'rainfall_flag': None, 'pressure_kind': 'surface',
            'precipitation_interval_start_utc_ns': _ns(valid - timedelta(hours=1)),
            'precipitation_interval_end_utc_ns': _ns(valid),
            'quality_flags': json.dumps(flags),
        }
        for key, (target, _) in FIELDS.items():
            value = hourly[key][index] if key in hourly else None
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
                raise ValueError(f'invalid historical weather numeric value: {key}')
            row[target] = value
        if any(row[target] is None for target, _ in FIELDS.values()):
            row['quality_flags'] = json.dumps([*flags, 'weather_values_missing'])
        observations.append(row)
    valid_count = sum(any(row[target] is not None for target, _ in FIELDS.values()) for row in observations)
    values_missing = any(any(row[target] is None for target, _ in FIELDS.values()) for row in observations)
    time_counts = Counter(row['valid_time_utc_ns'] for row in observations)
    missing_count = len(expected_times - set(time_counts))
    duplicate_count = sum(count - 1 for count in time_counts.values())
    unexpected_times = set(time_counts) - expected_times
    for row in observations:
        extra_flags = []
        if time_counts[row['valid_time_utc_ns']] > 1:
            extra_flags.append('duplicate_weather_valid_time')
        if row['valid_time_utc_ns'] in unexpected_times:
            extra_flags.append('unexpected_weather_valid_time')
        if extra_flags:
            row['quality_flags'] = json.dumps([*json.loads(row['quality_flags']), *extra_flags])
    has_expected_values = any(row['valid_time_utc_ns'] in expected_times and any(row[target] is not None for target, _ in FIELDS.values()) for row in observations)
    partial = bool(missing_columns or values_missing or missing_count or duplicate_count or unexpected_times)
    return {
        'series': {
            'circuit_id': circuit['circuit_id'], 'session_version_id': None,
            'provider': 'open_meteo', 'kind': 'reanalysis', 'model_name': 'era5',
            'requested_latitude': latitude, 'requested_longitude': longitude,
            'returned_latitude': returned_latitude, 'returned_longitude': returned_longitude,
            'coordinate_basis': circuit['coordinate_basis'],
            'elevation_m': payload.get('elevation'), 'spatial_resolution_km': None,
            'original_timezone': payload['timezone'],
            'units_metadata_json': json.dumps(units, sort_keys=True),
            'aggregation_metadata_json': json.dumps({'resolution': 'hourly', 'precipitation': 'preceding_hour_sum', 'coordinate_source_id': circuit['coordinate_source_id'], 'grid_resolution_degrees': 0.25}, sort_keys=True),
            'dataset_version': DATASET_VERSION, 'quality_flags_json': json.dumps(flags),
        },
        'observations': observations,
        'coverage': {
            'status': ('partial' if partial else 'available') if has_expected_values else 'unavailable',
            'rows': len(observations), 'missing_columns': missing_columns,
            'valid_count': valid_count, 'invalid_count': len(observations) - valid_count,
            'expected_count': len(expected_times), 'missing_count': missing_count,
            'duplicate_count': duplicate_count, 'unexpected_time_count': len(unexpected_times),
        },
        'forecast_coverage': dict(FORECAST_COVERAGE),
    }


def fetch_historical_weather(http, circuit, start, end, *, on_response=None):
    latitude, longitude = _coordinates(circuit)
    start, end = _utc(start), _utc(end)
    if end < start:
        raise ValueError('weather window end precedes start')
    params = {
        'latitude': latitude, 'longitude': longitude,
        'start_date': start.date().isoformat(), 'end_date': end.date().isoformat(),
        'hourly': ','.join(FIELDS), 'models': 'era5', 'timezone': 'UTC',
        'temperature_unit': 'celsius', 'wind_speed_unit': 'ms',
        'precipitation_unit': 'mm',
    }
    response = http.get(ENDPOINT, params=params, provider='open_meteo', timeout=30)
    if on_response is not None:
        on_response(response)
    result = normalize_weather(response.json(), circuit, start=start, end=end)
    result.update(raw_payload=response.content, request_url=response.url, request_params=params)
    return result
