"""Explicit, version-pinned recovery from retained Parquet; no provider/cache loads."""
from hashlib import sha256
import json

import pandas as pd
import pyarrow.parquet as pq

from .fastf1_extract import _dense
from .store import canonical, file_hash

RECOVERY_VERSION = 'retained-raw-stream-1'
_SESSION_TABLES = ('session_entries', 'session_results', 'laps', 'track_status_events',
    'session_status_events', 'race_control_messages', 'circuit_markers', 'pit_stop_records', 'coverage')
_REQUIRED = {'car': {'Date', 'SessionTime', 'RPM', 'Speed', 'nGear', 'Throttle', 'Brake', 'DRS'},
             'positions': {'Date', 'SessionTime', 'X', 'Y', 'Z', 'Status'}}


def _prepare(archive, session_id, version_id, kind):
    if kind not in _REQUIRED:
        raise ValueError('Recovery kind must be car or positions')
    with archive.connect() as conn:
        version = conn.execute('SELECT * FROM session_versions WHERE version_id=? AND session_id=? AND status="published"',
                               (version_id, session_id)).fetchone()
        if version is None:
            raise ValueError('Pinned published session version does not exist')
        version = dict(version)
        archive._verify_files(conn, version_id)
        files = [dict(row) for row in conn.execute('SELECT * FROM dataset_files WHERE version_id=? ORDER BY relative_path', (version_id,))]
        tables = {name: [{key: value for key, value in dict(row).items() if key != 'version_id'}
                  for row in conn.execute(f'SELECT * FROM {name} WHERE version_id=? ORDER BY rowid', (version_id,))]
                  for name in _SESSION_TABLES}
        series = [dict(row) for row in conn.execute('SELECT * FROM weather_series WHERE session_version_id=? ORDER BY series_id', (version_id,))]
        if any(row['kind'] == 'forecast' for row in series):
            raise ValueError('Session forecast series cannot be copied by raw stream recovery')
        tables['weather_series'] = [{key: value for key, value in row.items() if key != 'session_version_id'} for row in series]
        tables['weather_observations'] = [dict(row) for row in conn.execute('''SELECT w.* FROM weather_observations w
            JOIN weather_series s USING(series_id) WHERE s.session_version_id=? ORDER BY w.series_id,w.sample_index''', (version_id,))]
        sources = [dict(row) for row in conn.execute('SELECT source_id,dataset_kind FROM session_version_sources WHERE version_id=? ORDER BY source_id,dataset_kind', (version_id,))]
        # File provenance may refer to a source outside the version's source links.
        for file in files:
            if file['source_id']:
                source = conn.execute('SELECT * FROM source_objects WHERE source_id=?', (file['source_id'],)).fetchone()
                if source is None:
                    raise ValueError('Missing dataset provenance source')
                path = archive._safe_path(source['raw_path'])
                if not path.is_file() or file_hash(path) != source['payload_sha256']:
                    raise ValueError('Corrupt dataset provenance source')
    metadata = json.loads(version['metadata_json'] or '{}')
    metadata['completeness'] = json.loads(version['completeness_json'] or '{}')
    metadata['quality_flags'] = json.loads(version['quality_flags_json'] or '[]')
    metadata.pop('completeness_json', None)
    metadata.pop('quality_flags_json', None)
    metadata['t0_utc_ns'] = version['t0_utc_ns']
    entries = {row['driver_id']: row for row in tables['session_entries']}
    raw_files = [file for file in files if file['kind'] == f'raw_{kind}']
    if not raw_files:
        raise ValueError('No retained raw stream partitions')
    raw_ids = [file['driver_id'] for file in raw_files]
    if len(set(raw_ids)) != len(raw_ids) or any(driver not in entries for driver in raw_ids):
        raise ValueError('Raw partition driver must exactly match pinned session entries')
    if any(file['kind'] == kind and file['row_count'] for file in files):
        raise ValueError('Recovery refuses to replace an existing nonempty processed stream')
    datasets = []
    for file in files:
        if file['kind'] == kind:
            continue
        table = pq.read_table(archive._safe_path(file['relative_path']))
        if not file['kind'].startswith('raw_') and 'version_id' in table.column_names:
            if any(value != version_id for value in table['version_id'].to_pylist()):
                raise ValueError('Processed dataset version pin mismatch')
            table = table.drop(['version_id'])
        datasets.append(dict(kind=file['kind'], driver_id=file['driver_id'], table=table,
                             source_id=file['source_id'], transform_version=file['transform_version']))
    details, per_driver = {}, {}
    clock_basis = 'Date_minus_pinned_t0' if version['t0_utc_ns'] is not None else 'UTC_only_unknown_session_origin'
    for file in raw_files:
        driver_id = file['driver_id']
        frame = pq.read_table(archive._safe_path(file['relative_path'])).to_pandas()
        if not {'Time', 'Date'}.issubset(frame) or not pd.api.types.is_timedelta64_dtype(frame.Time) or not pd.api.types.is_datetime64_any_dtype(frame.Date):
            raise ValueError('Raw stream clock requires original timedelta Time and datetime Date')
        if 'driver_id' in frame and any(value != driver_id for value in frame.driver_id):
            raise ValueError('Raw row driver conflicts with partition driver')
        fill = frame.Time.isna()
        if 'fastf1_parser_missing_time_fill' in frame:
            flags = frame['fastf1_parser_missing_time_fill']
            if flags.isna().any() or not pd.api.types.is_bool_dtype(flags):
                raise ValueError('Invalid parser fill flags')
            if not flags.equals(fill):
                raise ValueError('Parser fill flag disagrees with missing Time')
        observed = frame.loc[~fill].copy()
        if observed.empty or observed.Date.isna().any():
            raise ValueError('Observed raw samples require valid Date')
        dates = pd.to_datetime(observed.Date, utc=True)
        if not dates.is_monotonic_increasing or not observed.Time.is_monotonic_increasing:
            raise ValueError('Raw stream clock is not monotonic')
        if version['t0_utc_ns'] is not None:
            clocks = dates.dt.as_unit('ns').astype('int64') - version['t0_utc_ns']
            if 'SessionTime' in observed:
                if not pd.api.types.is_timedelta64_dtype(observed.SessionTime) or observed.SessionTime.isna().any() or not (observed.SessionTime.dt.as_unit('ns').astype('int64').to_numpy() == clocks.to_numpy()).all():
                    raise ValueError('Raw stream SessionTime clock conflicts with pinned UTC origin')
            observed['SessionTime'] = pd.to_timedelta(clocks, unit='ns')
        elif 'SessionTime' in observed and observed.SessionTime.notna().any():
            raise ValueError('Cannot verify existing SessionTime clock without pinned UTC origin')
        else:
            observed['SessionTime'] = pd.Series(pd.NaT, index=observed.index, dtype='timedelta64[ns]')
        table = _dense(observed, driver_id, kind)
        extra = ['recovered_from_retained_parser_parquet']
        if version['t0_utc_ns'] is None:
            extra.append('unknown_session_clock_origin')
        table['quality_flags'] = [json.dumps(json.loads(value) + extra) for value in table.quality_flags]
        counts = {column: int(observed[column].notna().sum()) if column in observed else 0 for column in sorted(_REQUIRED[kind])}
        missing = sorted(_REQUIRED[kind] - set(frame.columns))
        if version['t0_utc_ns'] is not None and 'SessionTime' in missing:
            missing.remove('SessionTime')
        status = 'partial' if fill.any() or missing or any(value != len(observed) for value in counts.values()) else 'available'
        per_driver[driver_id] = dict(status=status, rows=len(frame), processed_rows=len(observed), valid_timestamps=len(observed),
            missing_columns=missing, channel_valid_counts=counts, parser_generated_fills=int(fill.sum()))
        details[driver_id] = dict(raw_rows=len(frame), observed_rows=len(observed), excluded_parser_fills=int(fill.sum()))
        datasets.append(dict(kind=kind, driver_id=driver_id, table=table, source_id=file['source_id'], transform_version=RECOVERY_VERSION))
    for driver in entries.keys() - per_driver.keys():
        per_driver[driver] = dict(status='unavailable', rows=0, valid_timestamps=0, missing_columns=sorted(_REQUIRED[kind]),
                                  channel_valid_counts={column: 0 for column in sorted(_REQUIRED[kind])})
    total = sum(row['rows'] for row in per_driver.values())
    status = 'available' if all(row['status'] == 'available' for row in per_driver.values()) else 'partial'
    metadata['completeness'][kind] = dict(status=status, rows=total, drivers=per_driver,
        reason='Recovered retained parser samples; raw originals preserved; SessionTime derived only from pinned t0')
    coverage = [row for row in tables['coverage'] if row['dataset_kind'] != kind and not row['dataset_kind'].startswith(f'{kind}/')]
    coverage.append(dict(dataset_kind=kind, status=status, rows=total, reason=metadata['completeness'][kind]['reason']))
    for driver, item in per_driver.items():
        coverage.append(dict(dataset_kind=f'{kind}/{driver}', status=item['status'], rows=item['rows'],
            valid_count=item['valid_timestamps'], invalid_count=item.get('parser_generated_fills', 0),
            missing_columns_json=json.dumps(item['missing_columns'])))
    tables['coverage'] = coverage
    metadata['coverage'] = [{key: value for key, value in row.items() if key != 'checked_at_utc'} for row in coverage]
    metadata['quality_flags'] = sorted(set(metadata['quality_flags'] + ['recovered_from_retained_parser_parquet']))
    metadata['raw_stream_recovery'] = dict(version=RECOVERY_VERSION, base_version_id=version_id, kind=kind,
        clock_basis=clock_basis, drivers=details)
    metadata['normalizer_version'] = f"{version['normalizer_version']}+{RECOVERY_VERSION}"
    preview = dict(session_id=session_id, base_version_id=version_id, kind=kind, drivers=details,
                   clock_basis=clock_basis, status=status)
    preview['plan_hash'] = sha256(canonical(dict(preview=preview,
        dataset_hash=version['dataset_hash'], metadata=metadata, tables=tables,
        files=[{key: file[key] for key in ('file_id', 'file_sha256', 'source_id', 'transform_version')} for file in files])).encode()).hexdigest()
    return preview, tables, datasets, sources, metadata


def preview_raw_stream_recovery(archive, session_id, version_id, *, kind='car'):
    """Validate all pinned evidence and return a read-only, hash-pinned preview."""
    return _prepare(archive, session_id, version_id, kind)[0]


def execute_raw_stream_recovery(archive, session_id, version_id, *, expected_plan_hash, execute=False, kind='car'):
    """Revalidate preview and publish a new immutable version only with execute=True."""
    if execute is not True:
        raise ValueError('Recovery publication requires execute=True')
    preview, tables, datasets, sources, metadata = _prepare(archive, session_id, version_id, kind)
    if preview['plan_hash'] != expected_plan_hash:
        raise ValueError('Recovery plan hash differs from reviewed preview')
    return archive.publish_session(session_id, tables, datasets, sources, metadata)
