from types import SimpleNamespace

import pandas as pd
import pyarrow.parquet as pq
import pytest

from data.archive.fastf1_extract import calendar_rows, extract_session, normalize_jolpica


def session_fixture():
    return SimpleNamespace(
        t0_date=pd.Timestamp('2025-01-01T00:00:00.000000123Z'),
        results=pd.DataFrame([
            dict(DriverNumber='1', DriverId='one', FirstName='A', LastName='One',
                 FullName='A One', TeamId='team', TeamName='Team', Position=1,
                 GridPosition=2, Time=pd.Timedelta(90_000_000_001, unit='ns'),
                 Q1=pd.Timedelta(20_000_000_003, unit='ns'), Status='Finished'),
            dict(DriverNumber='2', DriverId='two', FullName='B Two', TeamId='team',
                 TeamName='Team', Position=2, Time=pd.NaT, Status='+1 Lap'),
        ]),
        laps=pd.DataFrame([
            dict(DriverNumber='1', Driver='ONE', LapNumber=1, Stint=1,
                 Time=pd.Timedelta(90_000_000_001, unit='ns'),
                 LapStartTime=pd.Timedelta(1, unit='ns'),
                 LapStartDate=pd.Timestamp('2025-01-01T00:00:00.000000124Z'),
                 LapTime=pd.Timedelta(90_000_000_000, unit='ns'),
                 Sector1Time=pd.Timedelta(30_000_000_001, unit='ns'),
                 Sector1SessionTime=pd.Timedelta(30_000_000_002, unit='ns'),
                 Sector2Time=pd.Timedelta(30_000_000_002, unit='ns'),
                 Sector2SessionTime=pd.Timedelta(60_000_000_004, unit='ns'),
                 Sector3Time=pd.Timedelta(29_999_999_997, unit='ns'),
                 Sector3SessionTime=pd.Timedelta(90_000_000_001, unit='ns'),
                 SpeedI1=280.1, SpeedI2=290.2, SpeedFL=300.3, SpeedST=310.4,
                 IsPersonalBest=None, FreshTyre=None, Deleted=None, IsAccurate=True,
                 FastF1Generated=True, Compound='SOFT', TyreLife=2, Position=1,
                 TrackStatus='12', DeletedReason=None, ExtraField='retain me'),
        ]),
        car_data={'1': pd.DataFrame({
            'Date': [pd.Timestamp('2025-01-01T00:00:01.000000124Z')]*2,
            'SessionTime': pd.to_timedelta([1_000_000_001, 1_000_000_001], unit='ns'),
            'Time': pd.to_timedelta([0, 0], unit='ns'), 'Speed': [300, 301],
            'RPM': [10000, 10001], 'nGear': [8, 8], 'Throttle': [104, 99],
            'Brake': [None, False], 'DRS': [12, 14], 'Source': ['car', 'car'],
            'FutureChannel': [5, 6],
        })},
        pos_data={'1': pd.DataFrame({
            'Date': [pd.Timestamp('2025-01-01T00:00:01.000000125Z')],
            'SessionTime': pd.to_timedelta([1_000_000_002], unit='ns'),
            'X': [123], 'Y': [456], 'Z': [7], 'Status': ['OnTrack'], 'Source': ['pos'],
        })},
        weather_data=pd.DataFrame([dict(Time=pd.Timedelta(1_000_000_003, unit='ns'),
            AirTemp=20, TrackTemp=30, Humidity=50, Pressure=1010, Rainfall=None,
            WindDirection=180, WindSpeed=2.4)]),
        track_status=pd.DataFrame([dict(Time=pd.Timedelta(3, unit='ns'), Status='2', Message='Yellow')]),
        session_status=pd.DataFrame([dict(Time=pd.Timedelta(2, unit='ns'), Status='Started')]),
        race_control_messages=pd.DataFrame([dict(Time=pd.Timestamp('2025-01-01T00:00:00.000000128Z'),
            Category='Flag', Message='YELLOW', Flag='YELLOW', Scope='Sector', Sector=2,
            RacingNumber='2', Lap=1)]),
        circuit_info=SimpleNamespace(rotation=90, corners=pd.DataFrame([dict(
            Number=1, Letter='A', X=123, Y=456, Angle=20, Distance=100)]),
            marshal_lights=pd.DataFrame(), marshal_sectors=pd.DataFrame()),
    )


def test_calendar_uses_provider_location_not_round_for_circuit_identity():
    schedule = pd.DataFrame([dict(RoundNumber=3, EventName='Australian Grand Prix',
        OfficialEventName='Official', Country='Australia', Location='Melbourne',
        EventDate=pd.Timestamp('2025-03-16'), EventFormat='conventional',
        Session1='Practice 1', Session1DateUtc=pd.Timestamp('2025-03-14T01:30Z'),
        Session2='Race', Session2DateUtc=pd.Timestamp('2025-03-16T04:00Z'))])
    events, sessions = calendar_rows(2025, schedule)
    other = schedule.copy()
    other['RoundNumber'] = 1
    assert events[0]['circuit_id'] == calendar_rows(2026, other)[0][0]['circuit_id']
    assert events[0]['event_id'] == '2025:3'
    assert [row['session_id'] for row in sessions] == ['2025:3:FP1', '2025:3:R']
    assert sessions[0]['scheduled_start_utc_ns'] == 1741915800000000000
    assert events[0].get('timezone_name') is None


def test_extract_retains_original_frames_and_nanoseconds_without_fetching():
    source = session_fixture()
    payload = extract_session(source, '2025:1:R')
    tables = payload['tables']
    lap = tables['laps'][0]
    assert lap['lap_end_clock_ns'] == 90_000_000_001
    assert lap['lap_start_utc_ns'] == 1735689600000000124
    assert lap['sector1_ns'] == 30_000_000_001
    assert lap['sector2_end_clock_ns'] == 60_000_000_004
    assert lap['speed_trap_kmh'] == 310.4
    assert lap['fastf1_generated'] == 1
    assert lap['fresh_tyre'] is None
    assert lap['is_personal_best'] is None
    assert lap['deleted'] is None
    assert tables['session_results'][0]['reported_time_ns'] == 90_000_000_001
    assert tables['session_results'][0]['q1_ns'] == 20_000_000_003
    assert tables['session_results'][0]['grid_position'] == 2
    assert tables['race_control_messages'][0]['session_clock_ns'] == 5
    assert tables['weather_observations'][0]['rainfall_flag'] is None
    assert tables['weather_observations'][0]['wind_speed_mps'] == 2.4
    assert tables['circuit_markers'][0]['letter'] == 'A'
    assert all('version_id' not in row for rows in tables.values() for row in rows)
    datasets = {row['kind']: row['table'] for row in payload['datasets']}
    pd.testing.assert_frame_equal(datasets['raw_laps'], source.laps)
    pd.testing.assert_frame_equal(datasets['raw_results'], source.results)
    assert datasets['car']['timestamp_utc_ns'].tolist() == [1735689601000000124]*2
    assert datasets['car']['sample_index'].tolist() == [0, 1]
    assert datasets['car']['FutureChannel'].tolist() == [5, 6]
    assert datasets['car']['brake_applied'].tolist() == [pd.NA, False]
    assert 'throttle_out_of_range' in datasets['car']['quality_flags'].iloc[0]
    assert source.car_data['1'].columns.tolist() == list(datasets['car'].columns[:11])
    assert payload['metadata']['completeness']['positions']['status'] == 'partial'
    assert payload['metadata']['completeness']['positions']['drivers_with_rows'] == 1
    assert payload['metadata']['completeness']['positions']['expected_drivers'] == 2
    assert payload['metadata']['t0_utc_ns'] == 1735689600000000123


def test_missing_t0_never_uses_schedule_and_missing_streams_stay_missing():
    source = SimpleNamespace(results=pd.DataFrame([dict(DriverNumber='99')]),
        laps=pd.DataFrame(), date=pd.Timestamp('2025-01-01T00:00Z'))
    payload = extract_session(source, '2025:1:Q')
    assert payload['metadata']['t0_utc_ns'] is None
    assert payload['metadata']['completeness']['car']['status'] == 'unavailable'
    assert payload['metadata']['completeness']['timing']['status'] == 'unsupported'
    assert payload['tables']['drivers'][0]['source_driver_id'] is None
    assert '2025:1:Q' in payload['tables']['drivers'][0]['driver_id']


def test_calendar_maps_historical_sprint_qualifying_to_race_session():
    schedule = pd.DataFrame([dict(RoundNumber=10, EventName='British Grand Prix',
        Country='UK', Location='Silverstone', EventDate='2021-07-18',
        Session1='Sprint Qualifying', Session1DateUtc=pd.Timestamp('2021-07-17T16:30Z'))])
    assert calendar_rows(2021, schedule)[1][0]['session_code'] == 'S'


def test_telemetry_only_driver_becomes_entry_without_guessing_identity():
    source = SimpleNamespace(car_data={'44': pd.DataFrame({'Date': [pd.NaT], 'Speed': [10]})})
    payload = extract_session(source, '2025:1:FP1')
    assert payload['tables']['session_entries'][0]['driver_number'] == '44'
    assert payload['tables']['session_entries'][0]['source_driver_id'] is None
    car = next(item['table'] for item in payload['datasets'] if item['kind'] == 'car')
    assert str(car['timestamp_utc_ns'].dtype) == 'Int64'
    assert car['timestamp_utc_ns'].isna().all()


def test_extracted_payload_publishes_with_exact_clocks_and_original_fields(tmp_path):
    from data.archive.store import Archive

    archive = Archive(tmp_path)
    source_id = archive.source('fixture', 'loaded-session', b'fixture evidence', '2026-10-23T00:00:00Z')
    events, sessions = calendar_rows(2025, pd.DataFrame([dict(RoundNumber=1,
        EventName='Fixture GP', Country='Country', Location='Fixture circuit',
        Session1='Race', Session1DateUtc=pd.Timestamp('2025-01-01T00:00Z'))]))
    archive.save_calendar(events, sessions, source_id)
    payload = extract_session(session_fixture(), '2025:1:R')
    version_id = archive.publish_session('2025:1:R', source_ids=[source_id], **payload)
    assert archive.publish_session('2025:1:R', source_ids=[source_id], **payload) == version_id
    with archive.connect() as conn:
        assert conn.execute('SELECT lap_start_utc_ns FROM laps').fetchone()[0] == 1735689600000000124
        assert conn.execute('SELECT rainfall_flag FROM weather_observations').fetchone()[0] is None
        assert conn.execute("SELECT status FROM coverage WHERE dataset_kind='positions'").fetchone()[0] == 'partial'
        path = conn.execute("SELECT relative_path FROM dataset_files WHERE kind='car'").fetchone()[0]
        circuit = conn.execute('SELECT * FROM circuits').fetchone()
        assert circuit['reference_latitude'] is None
        assert circuit['timezone_name'] is None
    car = pq.read_table(tmp_path / path)
    assert car['timestamp_utc_ns'].to_pylist() == [1735689601000000124]*2
    assert car['FutureChannel'].to_pylist() == [5, 6]


def test_duplicate_or_unkeyed_laps_remain_in_raw_evidence():
    source = session_fixture()
    duplicate = source.laps.copy()
    duplicate['LapTime'] = pd.to_timedelta([91_000_000_000], unit='ns')
    unkeyed = source.laps.copy()
    unkeyed['LapNumber'] = None
    source.laps = pd.concat([source.laps, duplicate, unkeyed], ignore_index=True)
    payload = extract_session(source, '2025:1:R')
    assert len(payload['tables']['laps']) == 1
    assert payload['tables']['laps'][0]['lap_time_ns'] == 91_000_000_000
    assert next(dataset['table'] for dataset in payload['datasets'] if dataset['kind'] == 'raw_laps').shape[0] == 3
    assert 'duplicate_lap_key_retained_raw' in payload['metadata']['quality_flags']
    assert 'unkeyed_lap_retained_raw' in payload['metadata']['quality_flags']


def test_dense_discrete_channels_are_nullable_integers_and_empty_channel_is_partial():
    source = session_fixture()
    source.car_data['1']['nGear'] = [8, None]
    source.car_data['1']['DRS'] = [12, None]
    source.pos_data['1']['X'] = None
    payload = extract_session(source, '2025:1:R')
    car = next(item['table'] for item in payload['datasets'] if item['kind'] == 'car')
    assert str(car['gear'].dtype) == 'Int64'
    assert str(car['drs_code'].dtype) == 'Int64'
    positions = payload['metadata']['completeness']['positions']['drivers']['fastf1:driver:one']
    assert positions['channel_valid_counts']['X'] == 0
    assert positions['status'] == 'partial'
    assert car['gear'].iloc[0] == 8
    assert pd.isna(car['gear'].iloc[1])


def test_jolpica_result_lap_and_pit_pages_keep_unknown_observations_unknown():
    def page(**race):
        return {'MRData': {'RaceTable': {'Races': [race]}}}

    payloads = {'race_results': [page(Results=[dict(number='44', position='2', positionText='2',
        grid='1', laps='57', points='18', status='Finished', Time={'millis': '5400123', 'time': '+1.123'},
        Driver={'driverId': 'hamilton', 'givenName': 'Lewis', 'familyName': 'Hamilton',
                'code': 'HAM', 'permanentNumber': '44'},
        Constructor={'constructorId': 'mercedes', 'name': 'Mercedes'})])],
        'laps': [page(Laps=[dict(number='1', Timings=[dict(driverId='hamilton', position='2', time='1:32.123456789')])])],
        'pit_stops': [page(PitStops=[dict(driverId='hamilton', stop='1', lap='20', time='17:34:32', duration='22.345')])],
    }
    payload = normalize_jolpica('2025:1:R', payloads, source_ids={'pit_stops': ['pit-source']})
    result = payload['tables']['session_results'][0]
    lap = payload['tables']['laps'][0]
    pit = payload['tables']['pit_stop_records'][0]
    assert result['reported_time_ns'] == 5_400_123_000_000
    assert result['grid_position'] == 1
    assert lap['lap_time_ns'] == 92_123_456_789
    assert lap['lap_start_clock_ns'] is None
    assert lap['lap_end_clock_ns'] is None
    assert lap['compound_raw'] is None
    assert lap['fresh_tyre'] is None
    assert pit['duration_ns'] == 22_345_000_000
    assert pit['duration_kind'] == 'provider_reported_pit_stop_duration'
    assert pit['source_id'] == 'pit-source'
    assert payload['metadata']['t0_utc_ns'] is None
    assert payload['metadata']['completeness']['car']['status'] == 'unsupported'
    assert not payload['datasets']


def test_jolpica_qualifying_times_are_not_official_race_grid_and_lap_only_driver_has_no_number():
    page = {'MRData': {'RaceTable': {'Races': [{'QualifyingResults': [dict(position='1',
        Q1='1:20.001', Q2='1:19.222', Q3='1:18.333',
        Driver={'driverId': 'driver', 'permanentNumber': '7'},
        Constructor={'constructorId': 'team', 'name': 'Team'})]}]}}}
    payload = normalize_jolpica('2025:1:Q', {'qualifying_results': [page]})
    result = payload['tables']['session_results'][0]
    assert result['grid_position'] is None
    assert result['q3_ns'] == 78_333_000_000
    assert payload['tables']['session_entries'][0]['driver_number'] is None


def test_jolpica_pit_source_and_unknown_timing_survive_storage(tmp_path):
    from data.archive.store import Archive

    archive = Archive(tmp_path)
    source_id = archive.source('jolpica', 'pit-fixture', b'{"fixture":"pit"}', '2026-10-23T00:00:00Z')
    events, sessions = calendar_rows(2025, pd.DataFrame([dict(RoundNumber=1, Country='Country',
        Location='Fixture circuit', Session1='Race', Session1DateUtc=pd.Timestamp('2025-01-01T00:00Z'))]))
    archive.save_calendar(events, sessions, source_id)
    page = {'MRData': {'RaceTable': {'Races': [{'Laps': [dict(number='1', Timings=[dict(
        driverId='unknown_number', position='4', time='1:32.123')])], 'PitStops': [dict(
        driverId='unknown_number', stop='1', lap='20', time='17:34:32', duration='22.345')]}]}}}
    payload = normalize_jolpica('2025:1:R', {'pit_stops': [page]}, {'pit_stops': source_id})
    archive.publish_session('2025:1:R', source_ids=[source_id], **payload)
    with archive.connect() as conn:
        row = conn.execute('SELECT * FROM pit_stop_records').fetchone()
        assert row['source_id'] == source_id
        assert row['duration_ns'] == 22_345_000_000
        assert conn.execute('SELECT driver_number FROM session_entries').fetchone()[0] is None
        assert conn.execute('SELECT lap_start_utc_ns FROM laps').fetchone()[0] is None


def test_jolpica_rejects_wrong_event_and_race_laps_assigned_to_practice():
    wrong_event = {'MRData': {'RaceTable': {'Races': [{'season': '2024', 'round': '2'}]}}}
    with pytest.raises(ValueError, match='event'):
        normalize_jolpica('2025:1:R', {'results': [wrong_event]})
    race_laps = {'MRData': {'RaceTable': {'Races': [{'season': '2025', 'round': '1', 'Laps': [
        {'number': '1', 'Timings': [{'driverId': 'driver', 'time': '1:32.000'}]}]}]}}}
    with pytest.raises(ValueError, match='race'):
        normalize_jolpica('2025:1:FP1', {'laps': [race_laps]})

@pytest.mark.parametrize('sentinel', ['nan', 'NaN', '<NA>', 'None', 'null', 'NaT', ''])
def test_missing_identity_strings_keep_substitute_car_partitions_distinct(sentinel, tmp_path):
    from data.archive.store import Archive

    source = session_fixture()
    source.results['DriverId'] = sentinel
    source.results['TeamId'] = sentinel
    other_lap = source.laps.copy(deep=True)
    other_lap['DriverNumber'] = '2'
    source.laps = pd.concat([source.laps, other_lap], ignore_index=True)
    source.car_data['2'] = source.car_data['1'].copy(deep=True)
    payload = extract_session(source, '2025:1:R')
    entries = payload['tables']['session_entries']
    assert [row['driver_id'] for row in entries] == [
        'fastf1:2025:1:R:number:1', 'fastf1:2025:1:R:number:2']
    assert all(row['source_driver_id'] is None and row['source_team_id'] is None for row in entries)
    assert len(payload['tables']['laps']) == 2
    assert {row['team_id'] for row in entries} == {'fastf1:2025:1:R:team:team'}
    assert 'unresolved_driver_identity' in payload['metadata']['quality_flags']
    assert 'unresolved_team_identity' in payload['metadata']['quality_flags']
    assert payload['metadata']['normalizer_version'] == 'fastf1-local-2'
    pd.testing.assert_frame_equal(next(item['table'] for item in payload['datasets']
                                     if item['kind'] == 'raw_results'), source.results)
    archive = Archive(tmp_path)
    sid = archive.source('fixture', 'identity-evidence', b'fixture', '2026-10-26T00:00Z')
    events, sessions = calendar_rows(2025, pd.DataFrame([dict(RoundNumber=1,
        EventName='Fixture', Session1='Race')]))
    archive.save_calendar(events, sessions, sid)
    version = archive.publish_session('2025:1:R', source_ids=[sid], **payload)
    with archive.connect() as conn:
        assert conn.execute('SELECT count(*) FROM session_entries WHERE version_id=?', (version,)).fetchone()[0] == 2
        assert conn.execute("SELECT count(*) FROM dataset_files WHERE version_id=? AND kind='car'", (version,)).fetchone()[0] == 2


def test_three_british_unknown_numbers_publish_without_name_matching(tmp_path):
    from data.archive.store import Archive

    source = SimpleNamespace(results=pd.DataFrame([dict(DriverNumber=number,
        DriverId='nan', TeamId='nan', FullName='Sergio Perez', TeamName='Racing Point')
        for number in ['11', '0', '65535']]), race_control_messages=pd.DataFrame([
            dict(Message='None', Category='nan')]))
    payload = extract_session(source, '2020:4:FP1')
    assert len({row['driver_id'] for row in payload['tables']['session_entries']}) == 3
    assert payload['tables']['race_control_messages'][0]['message_text'] == 'None'
    assert payload['tables']['race_control_messages'][0]['category_raw'] == 'nan'
    archive = Archive(tmp_path)
    sid = archive.source('fixture', 'identity-evidence', b'fixture', '2026-10-26T00:00Z')
    events, sessions = calendar_rows(2020, pd.DataFrame([dict(RoundNumber=4,
        EventName='British Grand Prix', Session1='Practice 1')]))
    archive.save_calendar(events, sessions, sid)
    version = archive.publish_session('2020:4:FP1', source_ids=[sid], **payload)
    with archive.connect() as conn:
        assert conn.execute('SELECT count(*) FROM session_entries WHERE version_id=?', (version,)).fetchone()[0] == 3
        assert conn.execute('PRAGMA foreign_key_check').fetchall() == []
