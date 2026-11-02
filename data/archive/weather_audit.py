"""Read-only checks of yearly weather checkpoints and their retained evidence."""
from collections import Counter
import json
from pathlib import Path
import sqlite3
import stat

import pyarrow.parquet as pq

from .store import file_hash


VERSION = 'yearly_weather_v1'


def _safe_file(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError('Missing or unsafe archive file')
    path = root / relative
    if '..' in Path(relative).parts or not path.resolve().is_relative_to(root):
        raise ValueError('Missing or unsafe archive file')
    # Do not follow links, even links whose current targets are inside the root.
    for item in [path, *path.parents]:
        if item == root or not item.is_relative_to(root):
            continue
        attributes = getattr(item.lstat(), 'st_file_attributes', 0)
        if item.is_symlink() or attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            raise ValueError('Missing or unsafe archive file')
    if not path.is_file():
        raise ValueError('Missing or unsafe archive file')
    return path


def audit_weather(directory, expected_event_ids, *, verify_files=False):
    """Audit only the caller's eligible calendar events; never infer future rounds.

    Every saved event requires SQL links, safe existing files and Parquet row
    counts. ``verify_files`` additionally checks retained SHA-256 digests. This
    function opens an existing database in read-only mode and never initializes
    an Archive, calls providers, or modifies collection checkpoints.
    """
    root = Path(directory).resolve()
    expected = list(dict.fromkeys(expected_event_ids))
    integrity = dict(sqlite='unavailable', foreign_key_errors=0, file_errors=[],
                     files_verified=0, raw_sources_verified=0, file_hashes_checked=0)
    events = [dict(event_id=event_id, status='unavailable', observations=0,
                   next_retry_utc=None, detail='Weather archive unavailable') for event_id in expected]
    result = dict(events=events, counts={}, expected_events=len(expected),
                  complete_events=0, observations=0, integrity=integrity)
    database = root / 'f1.sqlite3'
    parquet_checked, raw_checked, hash_checked = set(), set(), {}
    connection = None
    try:
        if not database.is_file():
            result['error'] = 'Weather archive database missing'
        else:
            _safe_file(root, 'f1.sqlite3')
            connection = sqlite3.connect(database.as_uri() + '?mode=ro', uri=True, timeout=30)
            connection.row_factory = sqlite3.Row
            connection.execute('PRAGMA query_only=ON')
            checks = [row[0] for row in connection.execute('PRAGMA quick_check')]
            integrity['sqlite'] = 'ok' if checks == ['ok'] else 'failed'
            integrity['foreign_key_errors'] = len(connection.execute('PRAGMA foreign_key_check').fetchall())
            for event in events:
                tasks = connection.execute('''SELECT config_version,status,next_retry_utc,checkpoint_json
                    FROM collection_tasks WHERE dataset_kind='weather' AND event_id=?''',
                    (event['event_id'],)).fetchall()
                if not tasks:
                    event.update(status='missing', detail='No yearly weather collection task saved')
                    continue
                if len(tasks) != 1 or tasks[0]['config_version'] != VERSION:
                    event.update(status='repair_local', detail='Ambiguous or unsupported weather task configuration')
                    continue
                task = tasks[0]
                status = task['status']
                event.update(status=status if status in {'complete', 'partial', 'pending', 'running', 'failed', 'blocked'} else 'repair_local',
                             next_retry_utc=task['next_retry_utc'], detail='Saved weather collection task: ' +
                             (status if status in {'complete', 'partial', 'pending', 'running', 'failed', 'blocked'} else 'unknown'))
                if status not in {'complete', 'partial'}:
                    continue
                try:
                    if integrity['sqlite'] != 'ok' or integrity['foreign_key_errors']:
                        raise ValueError('Weather database integrity check failed')
                    state = json.loads(task['checkpoint_json'])
                    if not isinstance(state, dict):
                        raise ValueError('Invalid weather checkpoint')
                    count = state.get('observations')
                    if type(count) is not int or count <= 0:
                        raise ValueError('Missing weather observation count')
                    series_id, source_id = state['series_id'], state['source_id']
                    series = connection.execute('SELECT source_id FROM weather_series WHERE series_id=?', (series_id,)).fetchall()
                    total, matched = connection.execute('''SELECT count(*),coalesce(sum(source_id=?),0)
                        FROM weather_observations WHERE series_id=?''', (source_id, series_id)).fetchone()
                    if len(series) != 1 or series[0]['source_id'] != source_id or total != count or matched != count:
                        raise ValueError('Weather SQL series or observation source/count mismatch')
                    path = _safe_file(root, state['parquet_path'])
                    if pq.read_metadata(path).num_rows != count:
                        raise ValueError('Weather Parquet row count mismatch')
                    if verify_files:
                        if path not in hash_checked:
                            hash_checked[path] = file_hash(path)
                        if hash_checked[path] != state['parquet_sha256']:
                            raise ValueError('Weather Parquet SHA-256 mismatch')
                    parquet_checked.add(path)
                    for sid in {source_id, state['circuit']['coordinate_source_id']}:
                        sources = connection.execute('SELECT raw_path,payload_sha256 FROM source_objects WHERE source_id=?', (sid,)).fetchall()
                        if len(sources) != 1:
                            raise ValueError('Missing weather or circuit coordinate source metadata')
                        source = sources[0]
                        raw = _safe_file(root, source['raw_path'])
                        if verify_files:
                            if raw not in hash_checked:
                                hash_checked[raw] = file_hash(raw)
                            if hash_checked[raw] != source['payload_sha256']:
                                raise ValueError('Retained weather or coordinate source SHA-256 mismatch')
                        raw_checked.add(raw)
                    event.update(observations=count, detail='Saved reanalysis verified' +
                                 (' including SHA-256 digests' if verify_files else ' (SHA-256 digests not checked)'))
                    coverage = state.get('coverage')
                    if status == 'complete' and (not isinstance(coverage, dict) or coverage.get('status') != 'available'):
                        event.update(status='partial', detail='Saved weather coverage is not complete')
                except (KeyError, TypeError, ValueError, OSError, sqlite3.Error):
                    # Never return checkpoint contents, filesystem paths or provider messages.
                    event.update(status='repair_local', observations=0,
                                 detail='Saved weather evidence is missing, inconsistent or unreadable')
                    integrity['file_errors'].append(dict(event_id=event['event_id'], detail=event['detail']))
    except (OSError, ValueError, sqlite3.Error):
        result['error'] = 'Weather archive could not be read or has an unsupported schema'
        for event in events:
            event.update(status='unavailable', observations=0, detail=result['error'])
    finally:
        if connection is not None:
            connection.close()
    integrity.update(files_verified=len(parquet_checked), raw_sources_verified=len(raw_checked),
                     file_hashes_checked=len(hash_checked))
    result.update(counts=dict(Counter(event['status'] for event in events)),
                  complete_events=sum(event['status'] == 'complete' for event in events),
                  observations=sum(event['observations'] for event in events))
    return result
