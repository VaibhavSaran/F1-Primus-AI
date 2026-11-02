from copy import deepcopy
from datetime import datetime, timezone

import pytest

from data.archive.weather_extract import FIELDS, fetch_historical_weather, normalize_weather


UTC = timezone.utc
CIRCUIT = {'circuit_id': 'silverstone', 'latitude': 52.0786, 'longitude': -1.01694, 'coordinate_basis': 'jolpica_circuit_reference', 'coordinate_source_id': 'circuit-source'}
START = datetime(2024, 7, 7, 12, tzinfo=UTC)
END = datetime(2024, 7, 7, 15, tzinfo=UTC)


def payload():
    return {
        'latitude': 52.1, 'longitude': -1.0, 'elevation': 150,
        'timezone': 'GMT', 'utc_offset_seconds': 0,
        'hourly_units': {'temperature_2m': '°C', 'relative_humidity_2m': '%', 'surface_pressure': 'hPa', 'wind_speed_10m': 'm/s', 'rain': 'mm', 'precipitation': 'mm'},
        'hourly': {'time': ['2024-07-07T12:00', '2024-07-07T13:00'], 'temperature_2m': [15, None], 'relative_humidity_2m': [90, 95], 'surface_pressure': [996, 995], 'wind_speed_10m': [2.5, 3], 'rain': [0.4, 0], 'precipitation': [0.5, 0.1]},
    }


def test_reanalysis_units_pressure_and_missing_track_temperature():
    result = normalize_weather(payload(), CIRCUIT, start=START, end=END)
    series = result['series']
    first, second = result['observations']
    assert series['kind'] == 'reanalysis' and series['model_name'] == 'era5'
    assert series['requested_latitude'] != series['returned_latitude']
    assert first['track_temperature_c'] is None and first['rainfall_flag'] is None
    assert first['rain_mm'] == 0.4 and first['precipitation_mm'] == 0.5
    assert first['wind_speed_mps'] == 2.5 and first['pressure_kind'] == 'surface'
    assert second['air_temperature_c'] is None
    assert first['precipitation_interval_end_utc_ns'] - first['precipitation_interval_start_utc_ns'] == 3600 * 10**9
    assert result['forecast_coverage']['status'] == 'unsupported'
    assert 'forecast_runs' not in result


@pytest.mark.parametrize('change', ['unit', 'length', 'timezone', 'coordinate', 'error'])
def test_invalid_payloads_and_unproven_coordinates_are_rejected(change):
    value, circuit = deepcopy(payload()), dict(CIRCUIT)
    if change == 'unit':
        value['hourly_units']['wind_speed_10m'] = 'km/h'
    elif change == 'length':
        value['hourly']['rain'].pop()
    elif change == 'timezone':
        value['timezone'], value['utc_offset_seconds'] = 'Europe/London', 3600
    elif change == 'coordinate':
        circuit.pop('coordinate_source_id')
    else:
        value['error'] = True
    with pytest.raises(ValueError):
        normalize_weather(value, circuit, start=START, end=END)


def test_fetch_uses_utc_explicit_units_and_paced_transport():
    class HTTP:
        def get(self, url, **kwargs):
            self.url, self.kwargs = url, kwargs
            class Response:
                def json(self):
                    return payload()
                content = b'original'
                url = 'https://archive-api.open-meteo.com/v1/archive?test=1'
            return Response()
    http = HTTP()
    result = fetch_historical_weather(http, CIRCUIT, START, END)
    assert http.kwargs['provider'] == 'open_meteo'
    assert http.kwargs['params']['models'] == 'era5'
    assert http.kwargs['params']['wind_speed_unit'] == 'ms'
    assert http.kwargs['params']['timezone'] == 'UTC'
    assert result['raw_payload'] == b'original'
    assert result['series']['coordinate_basis'] == 'jolpica_circuit_reference'


def test_missing_measurements_are_partial_and_invalid_grid_is_rejected():
    value = payload()
    result = normalize_weather(value, CIRCUIT, start=START, end=END)
    assert result['coverage']['status'] == 'partial'
    assert 'wind_gusts_10m' in result['coverage']['missing_columns']
    value['latitude'] = 'not-a-coordinate'
    with pytest.raises(ValueError, match='grid coordinate'):
        normalize_weather(value, CIRCUIT, start=START, end=END)


def test_all_null_recent_reanalysis_is_unavailable_not_complete():
    value = payload()
    for key, (_, unit) in FIELDS.items():
        value['hourly'][key] = [None, None]
        value['hourly_units'][key] = unit
    result = normalize_weather(value, CIRCUIT, start=START, end=END)
    assert result['coverage']['status'] == 'unavailable'
    assert result['coverage']['valid_count'] == 0
    assert result['coverage']['invalid_count'] == 2


def test_hourly_window_coverage_detects_missing_and_duplicate_times():
    value = payload()
    for key, (_, unit) in FIELDS.items():
        value['hourly'][key] = [1, 1]
        value['hourly_units'][key] = unit
    result = normalize_weather(value, CIRCUIT, start=START, end=END)
    assert result['coverage']['status'] == 'partial'
    assert result['coverage']['expected_count'] == 4
    assert result['coverage']['missing_count'] == 2
    assert result['coverage']['duplicate_count'] == 0
    value['hourly']['time'] = ['2024-07-07T12:00', '2024-07-07T12:00']
    result = normalize_weather(value, CIRCUIT, start=START, end=END)
    assert len(result['observations']) == 2
    assert result['coverage']['missing_count'] == 3
    assert result['coverage']['duplicate_count'] == 1
    assert 'duplicate_weather_valid_time' in result['observations'][0]['quality_flags']


def test_full_hourly_window_is_available_and_empty_window_unavailable():
    value = payload()
    for key, (_, unit) in FIELDS.items():
        value['hourly'][key] = [1, 1]
        value['hourly_units'][key] = unit
    result = normalize_weather(value, CIRCUIT, start=START, end=datetime(2024, 7, 7, 13, tzinfo=UTC))
    assert result['coverage']['status'] == 'available'
    assert result['coverage']['expected_count'] == 2
    value['hourly']['time'] = ['2024-07-06T12:00', '2024-07-06T13:00']
    result = normalize_weather(value, CIRCUIT, start=START, end=END)
    assert result['coverage']['status'] == 'unavailable'
    assert result['coverage']['missing_count'] == result['coverage']['expected_count'] == 4


@pytest.mark.parametrize('failure', ['json', 'validation'])
def test_raw_response_callback_runs_before_json_or_normalization_failure(failure):
    stored = []
    class Response:
        content = b'original-provider-bytes'
        def json(self):
            if failure == 'json':
                raise ValueError('invalid JSON')
            value = payload()
            value['hourly_units']['rain'] = 'inch'
            return value
    class HTTP:
        def get(self, *args, **kwargs):
            return Response()
    with pytest.raises(ValueError):
        fetch_historical_weather(HTTP(), CIRCUIT, START, END, on_response=lambda response: stored.append(response.content))
    assert stored == [b'original-provider-bytes']
