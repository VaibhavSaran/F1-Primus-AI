"""Read-only, session-scoped access to published local archive evidence."""

from contextlib import contextmanager
from hashlib import sha256
import os
from pathlib import Path
import sqlite3

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .schema import _LAP_EVIDENCE_SCORE


MAX_ROWS = 2_000_000
MAX_FILES = 200

_SESSIONS = f"""
WITH evidence AS (
 SELECT v.version_id, sum({_LAP_EVIDENCE_SCORE}) score, count(*) lap_count
 FROM session_versions v JOIN laps l USING(version_id)
 WHERE v.status='published' GROUP BY v.version_id
), ranked AS (
 SELECT v.*,row_number() OVER(PARTITION BY v.session_id
 ORDER BY coalesce(e.score,0) DESC,coalesce(e.lap_count,0) DESC,v.rowid DESC) preference
 FROM session_versions v LEFT JOIN evidence e USING(version_id) WHERE v.status='published'
)
SELECT v.session_id,v.version_id,e.season,e.round_number,e.event_name,e.local_race_date,
 s.session_code,s.full_name,s.scheduled_start_utc_ns,v.t0_utc_ns,v.created_at_utc,
 v.observed_start_utc_ns,v.observed_end_utc_ns
FROM ranked v JOIN sessions s USING(session_id) JOIN events e USING(event_id)
WHERE v.preference=1
"""


def _frame(cursor):
    """Preserve nullable integer nanoseconds without a float round-trip."""
    rows = cursor.fetchmany(MAX_ROWS + 1)
    if len(rows) > MAX_ROWS:
        raise ValueError("Archive table exceeds the pilot row limit")
    columns = [item[0] for item in cursor.description]
    result = {}
    for index, name in enumerate(columns):
        values = [row[index] for row in rows]
        present = [value for value in values if value is not None]
        integer = (present and all(isinstance(value, int) for value in present)) or name.endswith('_ns')
        result[name] = pd.array(values, dtype='Int64') if integer else values
    return pd.DataFrame(result)


def _seconds(frame, columns):
    for original, derived in columns.items():
        if original in frame:
            frame[derived] = frame[original] / 1_000_000_000
    return frame


class ArchiveReader:
    """Never initializes, migrates or downloads anything from the supplied root."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.db_path = self.root / 'f1.sqlite3'
        # Hosted releases may place the local catalogue and mounted telemetry separately.
        mounted_parquet = os.environ.get('F1_SERVING_PARQUET_ROOT')
        self.parquet_root = Path(mounted_parquet or self.root / 'parquet').resolve()
        if not mounted_parquet and not self.parquet_root.is_relative_to(self.root):
            raise ValueError('Archive telemetry is outside the archive root')

    @contextmanager
    def _connect(self):
        if not self.db_path.is_file() or not self.db_path.resolve().is_relative_to(self.root):
            raise ValueError("Local archive database is missing or outside the archive root")
        connection = sqlite3.connect(self.db_path.as_uri() + '?mode=ro', uri=True)
        try:
            connection.execute('PRAGMA query_only=ON')
            yield connection
        finally:
            connection.close()

    def available_sessions(self):
        if not self.db_path.is_file():
            return []
        with self._connect() as connection:
            return _frame(connection.execute(_SESSIONS + ' ORDER BY e.season,e.round_number,s.scheduled_start_utc_ns')).to_dict('records')

    def _meta(self, connection, session_id):
        frame = _frame(connection.execute(_SESSIONS + ' AND v.session_id=?', (session_id,)))
        if frame.empty:
            raise ValueError(f"No published local archive session: {session_id}")
        return frame.iloc[0].to_dict()

    def tables(self, session_id):
        with self._connect() as connection:
            meta = self._meta(connection, session_id)
            version = meta['version_id']
            frames = {'meta': meta}
            frames['entries'] = _frame(connection.execute(
                'SELECT e.*,d.display_name FROM session_entries e JOIN drivers d USING(driver_id) '
                'WHERE e.version_id=? ORDER BY e.driver_id', (version,)))
            for key, table in {'laps': 'laps', 'results': 'session_results',
                    'track_status': 'track_status_events', 'race_control': 'race_control_messages',
                    'coverage': 'coverage', 'stints': 'v_stints', 'pit_visits': 'v_pit_visits'}.items():
                frames[key] = _frame(connection.execute(f'SELECT * FROM {table} WHERE version_id=?', (version,)))
            frames['weather'] = _frame(connection.execute(
                'SELECT w.*,s.provider,s.kind,s.units_metadata_json FROM weather_observations w '
                'JOIN weather_series s USING(series_id) WHERE s.session_version_id=? '
                'ORDER BY w.session_clock_ns,w.sample_index', (version,)))
        _seconds(frames['laps'], {'lap_time_ns': 'lap_time_s', 'sector1_ns': 'sector1_s',
            'sector2_ns': 'sector2_s', 'sector3_ns': 'sector3_s',
            'lap_start_clock_ns': 'lap_start_session_s', 'lap_end_clock_ns': 'lap_end_session_s',
            'pit_in_clock_ns': 'pit_in_session_s', 'pit_out_clock_ns': 'pit_out_session_s'})
        for key in ('weather', 'track_status', 'race_control'):
            _seconds(frames[key], {'session_clock_ns': 'session_time_s'})
        _seconds(frames['pit_visits'], {'pit_in_clock_ns': 'pit_in_session_s',
            'pit_out_clock_ns': 'pit_out_session_s', 'duration_ns': 'duration_s'})
        return frames

    def dense(self, session_id, kind, driver_ids=None, columns=None):
        if kind not in ('car', 'positions'):
            raise ValueError("Dense kind must be car or positions")
        with self._connect() as connection:
            version = self._meta(connection, session_id)['version_id']
            query = 'SELECT * FROM dataset_files WHERE version_id=? AND kind=?'
            parameters = [version, kind]
            if driver_ids is not None:
                driver_ids = list(dict.fromkeys(driver_ids))
                if not driver_ids:
                    return pd.DataFrame(columns=list(dict.fromkeys(list(columns or []) +
                        ['driver_id', 'session_clock_ns', 'timestamp_utc_ns', 'session_time_s'])))
                if len(driver_ids) > 100:
                    raise ValueError('Too many selected drivers')
                query += ' AND driver_id IN (' + ','.join('?' for _ in driver_ids) + ')'
                parameters.extend(driver_ids)
            files = _frame(connection.execute(query + ' ORDER BY driver_id,relative_path', parameters)).to_dict('records')
        if len(files) > MAX_FILES or sum(item['row_count'] for item in files) > MAX_ROWS:
            raise ValueError('Dense selection exceeds the pilot file or row limit')
        selected = None if columns is None else list(dict.fromkeys(
            [name for name in columns if name != 'session_time_s'] + ['driver_id', 'session_clock_ns', 'timestamp_utc_ns']))
        frames = []
        for item in files:
            relative = Path(item['relative_path'])
            version_root = (self.parquet_root / version).resolve()
            if relative.parts[:2] != ('parquet', version) or relative.anchor:
                raise ValueError('Unsafe archive parquet path')
            path = (self.parquet_root / Path(*relative.parts[1:])).resolve()
            if (Path(version).name != version or version_root.parent != self.parquet_root
                    or not path.is_relative_to(version_root)
                    or not path.is_file() or path.suffix != '.parquet'):
                raise ValueError('Untrusted archive Parquet path')
            digest = sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() != item['file_sha256']:
                raise ValueError('Archive Parquet hash mismatch')
            parquet = pq.ParquetFile(path)
            if parquet.metadata.num_rows != item['row_count'] or parquet.metadata.num_rows > MAX_ROWS:
                raise ValueError('Archive Parquet row count mismatch or pilot limit exceeded')
            if selected is not None and set(selected) - set(parquet.schema_arrow.names):
                raise ValueError('Requested dense column is unavailable')
            frame = parquet.read(columns=selected).to_pandas(types_mapper=lambda dtype:
                pd.Int64Dtype() if pa.types.is_integer(dtype) else None)
            if 'driver_id' in frame and (frame['driver_id'] != item['driver_id']).any():
                raise ValueError('Archive Parquet driver identity mismatch')
            frames.append(frame)
        result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=selected or ['driver_id', 'session_clock_ns', 'timestamp_utc_ns'])
        return _seconds(result, {'session_clock_ns': 'session_time_s'})

    def timing(self, session_id, *, version_id=None):
        """Project the selected publication's retained race timing, with lap fallback.

        Parser snapshots carry previous fields; their clock is a snapshot clock,
        not proof that every carried field was refreshed by the provider.
        """
        from .timing import COLUMNS, retained_stream, normalize_stream, lap_positions

        with self._connect() as connection:
            meta = self._meta(connection, session_id)
            if version_id is not None and not connection.execute(
                    "SELECT 1 FROM session_versions WHERE version_id=? AND session_id=? AND status='published'",
                    (version_id, session_id)).fetchone():
                raise ValueError('Timing version must be a published version of the selected session')
            if meta['session_code'] not in ('R', 'S'):
                frame = pd.DataFrame(columns=COLUMNS)
                frame.attrs.update(timing_basis='unavailable', source_status='not_applicable')
                return frame
            version = version_id if version_id is not None else meta['version_id']
            entries = _frame(connection.execute(
                'SELECT * FROM session_entries WHERE version_id=?', (version,))).to_dict('records')
            sources = _frame(connection.execute(
                'SELECT DISTINCT o.* FROM source_objects o JOIN session_version_sources s USING(source_id) '
                "WHERE s.version_id=? AND o.endpoint_kind='_extended_timing_data'", (version,))).to_dict('records')
            source_status = 'absent'
            if len(sources) == 1:
                try:
                    frame = normalize_stream(retained_stream(self.root, sources[0]), entries)
                    if not frame.empty:
                        frame.attrs.update(timing_basis='timing_stream', source_status='available',
                                           source_id=sources[0]['source_id'])
                        return frame
                    source_status = 'empty'
                except (ValueError, OSError, EOFError, TypeError, KeyError, AttributeError, ImportError,
                        OverflowError):
                    # Invalid or missing retained files must not execute general pickle
                    # globals or prevent an otherwise usable historical replay.
                    source_status = 'invalid'
            elif sources:
                source_status = 'ambiguous'
            laps = _frame(connection.execute('SELECT * FROM laps WHERE version_id=?', (version,)))
        frame = lap_positions(laps.to_dict('records'))
        frame.attrs.update(timing_basis='lap_end_position' if not frame.empty else 'unavailable',
                           source_status=source_status)
        return frame
