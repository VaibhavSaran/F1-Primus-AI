"""Offline recovery checks use real immutable archive fixtures."""
import json

import pandas as pd
import pyarrow.parquet as pq
import pytest

from data.archive import Archive


SESSION = '2023:17:S'
T0 = 1696680000000000000


def fixture(tmp_path, *, bad_driver=False, bad_clock=False, fills=False):
    archive = Archive(tmp_path / 'archive')
    source = archive.source('fixture', 'calendar', b'calendar', '2023-10-07T00:00:00Z')
    archive.save_calendar([dict(event_id='2023:17', season=2023, round_number=17)],
        [dict(session_id=SESSION, event_id='2023:17', session_code='S')], source)
    times = pd.to_timedelta([1, 2], unit='s')
    raw = pd.DataFrame(dict(Time=times, Date=pd.to_datetime([T0 + 10**9, T0 + 2*10**9]),
        RPM=[10000, 11000], Speed=[100, 110], nGear=[3, 4], Throttle=[50, 60],
        Brake=[False, False], DRS=[0, 0], Source=['car', 'car']))
    if bad_clock:
        raw['SessionTime'] = times
        raw.loc[1, 'Date'] += pd.Timedelta(1, 'ns')
    if fills:
        raw.loc[1, 'Time'] = pd.NaT
        raw['fastf1_parser_missing_time_fill'] = [False, True]
    tables = dict(drivers=[dict(driver_id='driver'), dict(driver_id='unknown')],
        session_entries=[dict(driver_id='driver', driver_number='1')],
        laps=[dict(driver_id='driver', lap_number=1, lap_time_ns=90000000000)],
        weather_series=[dict(series_id='sensor', provider='fixture', kind='track_observation')],
        weather_observations=[dict(series_id='sensor', sample_index=0, session_clock_ns=1)])
    metadata = dict(t0_utc_ns=T0, normalizer_version='old', custom={'keep': True},
        timing={'retained_source_id': source}, completeness={'car': {'status': 'unavailable'},
        'laps': {'status': 'partial'}}, quality_flags=['old_flag'],
        coverage=[dict(dataset_kind='car', status='unavailable', rows=0)])
    version = archive.publish_session(SESSION, tables,
        [dict(kind='raw_car', driver_id='unknown' if bad_driver else 'driver', table=raw,
              source_id=source, transform_version='parser'),
         dict(kind='timing', table=pd.DataFrame({'session_time_s': [1.], 'driver_id': ['driver']}),
              source_id=source, transform_version='timing-v1')],
        [dict(source_id=source, dataset_kind='timing')], metadata)
    return archive, source, version


def test_preview_is_readonly_and_execute_is_explicit_and_idempotent(tmp_path):
    from data.archive.recovery import preview_raw_stream_recovery, execute_raw_stream_recovery
    archive, source, old = fixture(tmp_path)
    before = archive.db_path.read_bytes()
    preview = preview_raw_stream_recovery(archive, SESSION, old)
    assert preview['drivers']['driver']['observed_rows'] == 2
    assert preview['base_version_id'] == old
    assert archive.db_path.read_bytes() == before
    with pytest.raises(ValueError, match='execute=True'):
        execute_raw_stream_recovery(archive, SESSION, old, expected_plan_hash=preview['plan_hash'])
    new = execute_raw_stream_recovery(archive, SESSION, old, execute=True,
                                      expected_plan_hash=preview['plan_hash'])
    assert new != old
    assert execute_raw_stream_recovery(archive, SESSION, old, execute=True,
                                      expected_plan_hash=preview['plan_hash']) == new
    with archive.connect() as conn:
        assert conn.execute('SELECT count(*) FROM session_versions').fetchone()[0] == 2
        row = conn.execute('SELECT * FROM session_versions WHERE version_id=?', (new,)).fetchone()
        metadata = json.loads(row['metadata_json'])
        assert metadata['custom'] == {'keep': True}
        assert metadata['timing'] == {'retained_source_id': source}
        assert metadata['completeness']['laps']['status'] == 'partial'
        assert metadata['completeness']['car']['status'] == 'available'
        assert conn.execute('SELECT lap_time_ns FROM laps WHERE version_id=?', (new,)).fetchone()[0] == 90000000000
        assert conn.execute('SELECT count(*) FROM weather_observations').fetchone()[0] == 2
        assert tuple(conn.execute('SELECT source_id,dataset_kind FROM session_version_sources WHERE version_id=?', (new,)).fetchone()) == (source, 'timing')
        files = [dict(r) for r in conn.execute('SELECT * FROM dataset_files WHERE version_id=?', (new,))]
    assert {r['kind'] for r in files} == {'raw_car', 'car', 'timing'}
    car = pq.read_table(archive.root / next(r['relative_path'] for r in files if r['kind'] == 'car')).to_pandas()
    assert car['version_id'].unique().tolist() == [new]
    assert car['session_clock_ns'].tolist() == [10**9, 2*10**9]
    assert car['timestamp_utc_ns'].tolist() == [T0 + 10**9, T0 + 2*10**9]
    assert car['speed_kmh'].tolist() == [100, 110]
    timing = next(r for r in files if r['kind'] == 'timing')
    assert timing['source_id'] == source and timing['transform_version'] == 'timing-v1'
    assert pq.read_table(archive.root / timing['relative_path'])['version_id'].to_pylist() == [new]


@pytest.mark.parametrize('argument,reason', [('bad_driver', 'driver'), ('bad_clock', 'clock')])
def test_preview_refuses_identity_or_clock_mismatch(tmp_path, argument, reason):
    from data.archive.recovery import preview_raw_stream_recovery
    archive, _, old = fixture(tmp_path, **{argument: True})
    with pytest.raises(ValueError, match=reason):
        preview_raw_stream_recovery(archive, SESSION, old)


def test_generated_fills_are_retained_raw_but_never_observed(tmp_path):
    from data.archive.recovery import preview_raw_stream_recovery, execute_raw_stream_recovery
    archive, _, old = fixture(tmp_path, fills=True)
    preview = preview_raw_stream_recovery(archive, SESSION, old)
    assert preview['drivers']['driver'] == {'raw_rows': 2, 'observed_rows': 1, 'excluded_parser_fills': 1}
    new = execute_raw_stream_recovery(archive, SESSION, old, execute=True, expected_plan_hash=preview['plan_hash'])
    with archive.connect() as conn:
        assert conn.execute("SELECT row_count FROM dataset_files WHERE version_id=? AND kind='raw_car'", (new,)).fetchone()[0] == 2
        assert conn.execute("SELECT row_count FROM dataset_files WHERE version_id=? AND kind='car'", (new,)).fetchone()[0] == 1
        coverage = conn.execute("SELECT rows,valid_count,invalid_count FROM coverage WHERE version_id=? AND dataset_kind='car/driver'", (new,)).fetchone()
        assert tuple(coverage) == (2, 1, 1)


def test_refuses_corrupt_linked_sources_and_wrong_version_pin(tmp_path):
    from data.archive.recovery import preview_raw_stream_recovery, execute_raw_stream_recovery
    archive, source, old = fixture(tmp_path)
    preview = preview_raw_stream_recovery(archive, SESSION, old)
    with pytest.raises(ValueError, match='plan'):
        execute_raw_stream_recovery(archive, SESSION, old, execute=True, expected_plan_hash='wrong')
    with pytest.raises(ValueError, match='session'):
        preview_raw_stream_recovery(archive, '2023:17:R', old)
    with archive.connect() as conn:
        path = archive.root / conn.execute('SELECT raw_path FROM source_objects WHERE source_id=?', (source,)).fetchone()[0]
    path.write_bytes(b'broken')
    with pytest.raises(ValueError, match='[Cc]orrupt'):
        preview_raw_stream_recovery(archive, SESSION, old)


def test_unknown_t0_recovers_utc_channels_without_fabricating_session_clock(tmp_path):
    from data.archive.recovery import preview_raw_stream_recovery, execute_raw_stream_recovery
    archive, _, old = fixture(tmp_path)
    with archive.connect() as conn:
        row = conn.execute('SELECT metadata_json FROM session_versions WHERE version_id=?', (old,)).fetchone()
        metadata = json.loads(row[0]); metadata['t0_utc_ns'] = None
        conn.execute('UPDATE session_versions SET t0_utc_ns=NULL, metadata_json=? WHERE version_id=?', (json.dumps(metadata), old))
    preview = preview_raw_stream_recovery(archive, SESSION, old)
    assert preview['clock_basis'] == 'UTC_only_unknown_session_origin'
    new = execute_raw_stream_recovery(archive, SESSION, old, execute=True, expected_plan_hash=preview['plan_hash'])
    with archive.connect() as conn:
        path = conn.execute("SELECT relative_path FROM dataset_files WHERE version_id=? AND kind='car'", (new,)).fetchone()[0]
        status = json.loads(conn.execute('SELECT completeness_json FROM session_versions WHERE version_id=?', (new,)).fetchone()[0])
    frame = pq.read_table(archive.root / path).to_pandas()
    assert frame.session_clock_ns.isna().all()
    assert frame.slice_time_ns.tolist() == [10**9, 2*10**9]
    assert frame.timestamp_utc_ns.tolist() == [T0 + 10**9, T0 + 2*10**9]
    assert status['car']['status'] == 'partial'


def test_recovery_refuses_replacing_nonempty_processed_stream(tmp_path):
    from data.archive.recovery import preview_raw_stream_recovery
    archive, _, old = fixture(tmp_path)
    with archive.connect() as conn:
        conn.execute("UPDATE dataset_files SET kind='car' WHERE version_id=? AND kind='timing'", (old,))
    with pytest.raises(ValueError, match='nonempty'):
        preview_raw_stream_recovery(archive, SESSION, old)


def test_corrupt_retained_parquet_is_refused_before_preview(tmp_path):
    from data.archive.recovery import preview_raw_stream_recovery
    archive, _, old = fixture(tmp_path)
    with archive.connect() as conn:
        path = archive.root / conn.execute("SELECT relative_path FROM dataset_files WHERE version_id=? AND kind='raw_car'", (old,)).fetchone()[0]
    path.write_bytes(b'bad parquet')
    with pytest.raises(ValueError, match='Corrupt'):
        preview_raw_stream_recovery(archive, SESSION, old)


def test_known_t0_uses_nanoseconds_for_microsecond_arrow_clocks(tmp_path):
    from data.archive.recovery import execute_raw_stream_recovery, preview_raw_stream_recovery
    archive, source, _ = fixture(tmp_path)
    times = pd.Series(pd.to_timedelta([1, 2], unit='s')).dt.as_unit('us')
    dates = pd.Series(pd.to_datetime([T0 + 10**9, T0 + 2*10**9])).dt.as_unit('us')
    frame = pd.DataFrame(dict(Time=times, SessionTime=times, Date=dates,
        RPM=[10000, 11000], Speed=[100, 110], nGear=[3, 4], Throttle=[50, 60],
        Brake=[False, False], DRS=[0, 0]))
    old = archive.publish_session(SESSION,
        dict(session_entries=[dict(driver_id='driver', driver_number='1')]),
        [dict(kind='raw_car', driver_id='driver', table=frame)], [source],
        dict(t0_utc_ns=T0, completeness={'car': {'status': 'unavailable'}}))
    preview = preview_raw_stream_recovery(archive, SESSION, old)
    new = execute_raw_stream_recovery(archive, SESSION, old, execute=True,
                                      expected_plan_hash=preview['plan_hash'])
    with archive.connect() as conn:
        path = conn.execute("SELECT relative_path FROM dataset_files WHERE version_id=? AND kind='car'", (new,)).fetchone()[0]
    result = pq.read_table(archive.root / path).to_pandas()
    assert result.session_clock_ns.tolist() == [10**9, 2*10**9]
    assert result.timestamp_utc_ns.tolist() == [T0 + 10**9, T0 + 2*10**9]
