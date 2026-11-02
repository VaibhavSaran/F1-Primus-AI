from dataclasses import replace
from datetime import timedelta

import pytest

from test_prediction_service import inputs
from data.contracts import DriverSessionRecord, SourceRecord
from flows.prediction_service import generate_forecast


def reconstruction():
    request, snapshot, schedule = inputs()
    fetched = request.as_of_utc + timedelta(days=30)
    source = SourceRecord('fastf1', 'downloaded-qualifying', fetched)
    records = [replace(row, source=source) for row in snapshot.payloads['driver_sessions']]
    payload = dict(snapshot.payloads, historical_reconstruction=True,
                   schedule=schedule, entries_source=source, driver_sessions=records)
    return request, replace(snapshot, payloads=payload, sources=[source]), schedule


def test_reconstruction_runs_without_claiming_historical_publication():
    request, snapshot, schedule = reconstruction()
    result = generate_forecast(request, snapshot, model_id='grid', schedule=schedule)
    assert [row['driver_id'] for row in result['ranking']] == ['B', 'A']
    assert result['provenance']['validation_status'] == 'historical_reconstruction'
    assert result['provenance']['original_fetch_times'][0] > request.as_of_utc.isoformat()
    assert result['run_id'].startswith('forecast-')
    assert any('reconstruction' in warning.lower() for warning in result['warnings'])
    assert snapshot.payloads['entries_source'].fetched_at_utc > request.as_of_utc


def test_reconstruction_rejects_current_race_records_and_late_sessions():
    request, snapshot, schedule = reconstruction()
    bad = DriverSessionRecord(snapshot.event, 'R', 'A', request.as_of_utc - timedelta(hours=1),
                              {'qualifying_position': 1}, snapshot.sources[0])
    with pytest.raises(ValueError, match='session'):
        generate_forecast(request, replace(snapshot, payloads=dict(snapshot.payloads,
                          driver_sessions=[bad])), model_id='grid', schedule=schedule)
    late = replace(snapshot.payloads['driver_sessions'][0], available_at_utc=request.as_of_utc+timedelta(hours=2))
    with pytest.raises(ValueError, match='cutoff'):
        generate_forecast(request, replace(snapshot, payloads=dict(snapshot.payloads,
                          driver_sessions=[late])), model_id='grid', schedule=schedule)


def test_weekend_form_uses_qualifying_practice_and_sprint_not_race():
    request, snapshot, schedule = inputs()
    source = snapshot.sources[0]
    records = list(snapshot.payloads['driver_sessions'])
    for driver, pace, sprint in [('A', 0., 1), ('B', 2., 2)]:
        records.extend([DriverSessionRecord(snapshot.event, 'FP2', driver, source.fetched_at_utc,
                                           {'practice_pace_delta_s': pace}, source),
                        DriverSessionRecord(snapshot.event, 'S', driver, source.fetched_at_utc,
                                           {'sprint_position': sprint}, source)])
    snapshot = replace(snapshot, payloads=dict(snapshot.payloads, driver_sessions=records))
    result = generate_forecast(request, snapshot, model_id='weekend_form', schedule=schedule)
    assert len(result['ranking']) == 2
    assert result['provenance']['model_version'] == 'weekend-form-v1'
    assert result['provenance']['input_sessions'] == ['FP2', 'GRID', 'S']
    assert result['metrics'] == {}
