"""Read-only calendar-versus-publication checks for notebook resume planning."""
from collections import Counter
from datetime import date
import json
from pathlib import Path
import sqlite3

import pyarrow.parquet as pq

from .notebook_f1 import REQUIRED
from .reader import ArchiveReader
from .store import file_hash


NOT_HELD = {
    '2020:2:FP3': ('Cancelled due to weather', 'https://www.formula1.com/en/racing/2020/styria'),
    '2020:11:FP1': ('Cancelled due to weather', 'https://www.formula1.com/en/racing/2020/germany'),
    '2020:11:FP2': ('Cancelled due to weather', 'https://www.formula1.com/en/racing/2020/germany'),
    '2021:15:FP3': ('Cancelled due to wet weather', 'https://press.pirelli.com/2021-russian-grand-prix---saturday/'),
}


def audit_year(directory, year=2020, *, verify_files=False, through_date=None):
    """Return a fresh inventory without creating an archive or contacting a provider.

    Published coverage takes precedence over progress JSON. Partial parser fills
    are review items, not proof that another download can restore observations.
    """
    if through_date is not None:
        through_date = date.fromisoformat(through_date).isoformat()
    directory = Path(directory).resolve()
    database = directory / 'f1.sqlite3'
    if not database.is_file():
        raise FileNotFoundError('No archive calendar exists yet: ' + str(database))
    progress_path = directory / f'progress-f1-{year}.json'
    checkpoint_warning = None
    try:
        progress = json.loads(progress_path.read_text(encoding='utf8')) if progress_path.exists() else {}
        if not isinstance(progress, dict) or not isinstance(progress.get('sessions', {}), dict):
            raise ValueError('Invalid checkpoint structure')
        checkpoints = {key: value for key, value in progress.get('sessions', {}).items() if isinstance(value, dict)}
        if len(checkpoints) != len(progress.get('sessions', {})):
            checkpoint_warning = 'Invalid checkpoint rows ignored; published coverage takes precedence.'
    except (OSError, ValueError, UnicodeError):
        checkpoints = {}
        checkpoint_warning = 'Collection checkpoint unreadable; published coverage is still audited.'
    published = {row['session_id']: row for row in ArchiveReader(directory).available_sessions() if row['season'] == year}
    errors, checked = [], set()
    connection = sqlite3.connect(database.as_uri() + '?mode=ro', uri=True, timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute('PRAGMA query_only=ON')
        integrity = {'sqlite': connection.execute('PRAGMA quick_check').fetchone()[0],
                     'foreign_key_errors': len(connection.execute('PRAGMA foreign_key_check').fetchall())}
        calendar = connection.execute('''SELECT s.*,e.event_name,e.round_number,e.local_race_date FROM sessions s
            JOIN events e USING(event_id) WHERE e.season=? ORDER BY e.round_number,s.scheduled_start_utc_ns''', (year,)).fetchall()
        sessions = []
        for original in calendar:
            item = dict(original)
            if through_date and item['local_race_date'] and item['local_race_date'] > through_date:
                continue
            sid = item['session_id']
            checkpoint = checkpoints.get(sid, {})
            row = dict(session_id=sid, round_number=item['round_number'], event=item['event_name'],
                       date=item['local_race_date'], session=item['full_name'] or item['session_code'],
                       checkpoint_status=checkpoint.get('status', 'not_attempted'),
                       laps=0, drivers=0, version_id=None, detail='', reference='')
            if sid in NOT_HELD:
                row.update(action='not_held', detail=NOT_HELD[sid][0], reference=NOT_HELD[sid][1])
            elif sid not in published:
                row.update(action='download', detail=checkpoint.get('reason', 'No published session saved'))
                if checkpoint.get('error_type'):
                    row['detail'] += ' · ' + checkpoint['error_type']
                if checkpoint.get('retry_after_utc'):
                    row['detail'] += ' · retry after ' + checkpoint['retry_after_utc']
            else:
                version = published[sid]['version_id']
                row['version_id'] = version
                record = connection.execute('SELECT metadata_json,completeness_json FROM session_versions WHERE version_id=?', (version,)).fetchone()
                metadata = json.loads(record['metadata_json'] or '{}')
                coverage = json.loads(record['completeness_json'] or '{}')
                row['laps'] = connection.execute('SELECT count(*) FROM laps WHERE version_id=?', (version,)).fetchone()[0]
                row['drivers'] = connection.execute('SELECT count(*) FROM session_entries WHERE version_id=?', (version,)).fetchone()[0]
                flags = [f'{kind}: {coverage.get(kind, {}).get("status", "unknown")} '
                         f'({coverage.get(kind, {}).get("reason") or "see coverage"})'
                         for kind in REQUIRED if coverage.get(kind, {}).get('status') != 'available']
                row.update(action='complete' if not flags and not metadata.get('load_error') and row['laps'] else 'review_partial',
                           detail='; '.join(flags) or metadata.get('load_error') or '')
                identities = connection.execute('SELECT source_driver_id,source_team_id FROM session_entries WHERE version_id=?', (version,)).fetchall()
                if any(str(value).strip().casefold() in {'nan', 'nat', '<na>', 'null', 'none'}
                       for entry in identities for value in entry if value is not None):
                    row.update(action='reprocess_identity', detail='Legacy missing identity stored as text; reprocess from retained cache with corrected normalizer. ' + row['detail'])
                if verify_files:
                    files = [dict(r) for r in connection.execute('SELECT relative_path,file_sha256,row_count FROM dataset_files WHERE version_id=?', (version,))]
                    sources = [dict(r) for r in connection.execute('''SELECT o.raw_path,o.payload_sha256 FROM source_objects o
                        JOIN session_version_sources v USING(source_id) WHERE v.version_id=?''', (version,))]
                    for source in files + sources:
                        relative = source.get('relative_path', source.get('raw_path'))
                        path = (directory / relative).resolve()
                        try:
                            if not path.is_relative_to(directory) or not path.is_file():
                                raise ValueError('Missing or unsafe archive file')
                            if relative not in checked:
                                if file_hash(path) != source.get('file_sha256', source.get('payload_sha256')):
                                    raise ValueError('SHA-256 mismatch')
                                if 'row_count' in source and pq.ParquetFile(path).metadata.num_rows != source['row_count']:
                                    raise ValueError('Parquet row count mismatch')
                                checked.add(relative)
                        except (OSError, ValueError) as error:
                            errors.append({'session_id': sid, 'file': relative, 'error': str(error)})
                            row.update(action='repair_local', detail='Local file verification failed; inspect audit errors before retrying')
            sessions.append(row)
    finally:
        connection.close()
    integrity.update(file_errors=errors, files_verified=len(checked), file_hashes_checked=verify_files)
    return {'year': year, 'through_date': through_date, 'sessions': sessions, 'counts': dict(Counter(row['action'] for row in sessions)),
            'checkpoint_warning': checkpoint_warning,
            'integrity': integrity,
            'remaining_session_ids': [row['session_id'] for row in sessions if row['action'] in ('download', 'reprocess_identity')],
            'reprocess_session_ids': [row['session_id'] for row in sessions if row['action'] == 'reprocess_identity']}
