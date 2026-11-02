"""Local sourced context, separate from immutable F1 publications.

Source text is untrusted data. Nothing in this module executes it, fetches a URL,
infers an incident cause, or changes an archive database.
"""
from contextlib import closing
from datetime import datetime, time, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3


DEFAULT_DB = Path(__file__).resolve().parents[2] / '.local/event_context/context.sqlite3'
SCHEMA_VERSION = 1
_SCHEMA = """
CREATE TABLE context_records (
 record_id TEXT PRIMARY KEY, record_sha256 TEXT NOT NULL,
 event_id TEXT, external_event_ref TEXT, session_id TEXT, version_id TEXT,
 driver_id TEXT, status TEXT NOT NULL, known_at TEXT NOT NULL,
 canonical_record TEXT NOT NULL);
CREATE INDEX context_session ON context_records(session_id,driver_id);
CREATE INDEX context_event ON context_records(event_id,external_event_ref,known_at);
PRAGMA user_version=1;
"""
_RETIREMENT_STATUSES = {
    'retired', 'accident', 'collision', 'collision damage', 'spun off', 'engine',
    'gearbox', 'transmission', 'clutch', 'hydraulics', 'electrical', 'suspension',
    'brakes', 'differential', 'overheating', 'mechanical', 'driveshaft', 'fuel system',
    'fuel pressure', 'fuel pump', 'out of fuel', 'oil leak', 'oil pressure',
    'water leak', 'water pressure', 'exhaust', 'wheel', 'wheel bearing', 'puncture',
    'tyre', 'rear wing', 'front wing', 'power unit', 'power loss', 'damage',
    'steering', 'turbo', 'radiator', 'vibrations', 'battery', 'undertray', 'ers',
}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def _utc(value, *, date_allowed=False):
    if not isinstance(value, str):
        raise ValueError('A UTC timestamp is required')
    if date_allowed and re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
        parsed = datetime.combine(datetime.strptime(value, '%Y-%m-%d').date(), time.max, timezone.utc)
    else:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if parsed.tzinfo is None:
            raise ValueError('Timestamp must include a timezone')
    return parsed.astimezone(timezone.utc).isoformat(timespec='microseconds')


def classify_outcome(raw_status, session_code, *, laps_completed=None, classified_position_raw=None):
    """Conservative recorded classification; zero laps never establishes DNS.

    Retirement-like provider status is used only for R/S sessions. The raw value
    remains evidence, not a separately verified explanation for the retirement.
    """
    status = str(raw_status or '').strip().casefold()
    classified = str(classified_position_raw or '').strip().upper()
    if status in {'disqualified', 'dsq'} or classified == 'D':
        return 'disqualified'
    if status in {'did not start', 'dns', 'not started'}:
        return 'dns'
    if status in {'did not qualify', 'did not prequalify', 'dnq', 'dnpq'}:
        return 'dnq'
    if status in {'withdrawn', 'withdrew'} or classified == 'W':
        return 'withdrawn'
    if status == 'finished':
        return 'finished'
    if re.fullmatch(r'\+\s*\d+\s+laps?', status):
        return 'lapped'
    if session_code in {'R', 'S'} and status in _RETIREMENT_STATUSES:
        return 'retired'
    return 'unknown'


class EventContextStore:
    """Explicit imports write only this store; read() never creates it."""

    def __init__(self, db_path=DEFAULT_DB):
        self.db_path = Path(db_path).resolve()

    def _check_schema(self, connection):
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if tables != {'context_records'} or connection.execute('PRAGMA user_version').fetchone()[0] != SCHEMA_VERSION:
            raise ValueError('Not a supported event-context database')

    def import_records(self, records):
        """Validate the entire curated batch before an atomic, idempotent insert.

        Extra JSON fields are preserved. Missing source hashes stay null; the
        separately computed record hash never claims to hash a web page.
        """
        prepared = []
        for supplied in records:
            record = dict(supplied)
            event, external = record.get('event_id'), record.get('external_event_ref')
            if bool(event) == bool(external):
                raise ValueError('Provide exactly one event_id or external_event_ref')
            if external and (not external.startswith('external:') or record.get('session_id')):
                raise ValueError('External events require external: references and no archive session_id')
            if record.get('session_id') and not record['session_id'].startswith(event + ':'):
                raise ValueError('Session must belong to the supplied event')
            if record.get('version_id') and not record.get('session_id'):
                raise ValueError('A version_id requires a session_id')
            for required in ('summary', 'status', 'publisher', 'url', 'confidence'):
                if not isinstance(record.get(required), str) or not record[required].strip():
                    raise ValueError('Missing context field: ' + required)
            if not record['url'].startswith(('https://', 'http://', 'archive:')):
                raise ValueError('Context citation must be HTTP(S) or archive:')
            for field, allowed in (
                ('assertion_kind', {'fact', 'report'}),
                ('cause_qualification', {'stated', 'reported', 'unknown'}),
                ('evidence_kind', {'official_statement', 'reported_context', 'cancellation', 'archive_result'}),
            ):
                if record.get(field) not in allowed:
                    raise ValueError('Invalid context field: ' + field)
            source_hash = record.get('source_sha256')
            if source_hash is not None and not re.fullmatch('[0-9a-f]{64}', source_hash):
                raise ValueError('source_sha256 must be a lowercase SHA256 or null')
            record.setdefault('source_sha256', None)
            record.setdefault('author', None)
            record.setdefault('published_at', None)
            record.setdefault('sources', [])
            retrieved = _utc(record.get('retrieved_at'))
            verified = _utc(record.get('first_verified_at') or record['retrieved_at'])
            published = _utc(record['published_at'], date_allowed=True) if record['published_at'] else verified
            record['first_verified_at'] = record.get('first_verified_at') or record['retrieved_at']
            knowledge_times = [retrieved, verified, published]
            if record.get('archive_published_at'):
                knowledge_times.append(_utc(record['archive_published_at']))
            for source in record['sources']:
                for field in ('published_at', 'first_verified_at'):
                    if source.get(field):
                        knowledge_times.append(_utc(source[field], date_allowed=field == 'published_at'))
            known_at = max(knowledge_times)
            canonical = _canonical(record)
            identity = {key: value for key, value in record.items()
                        if key not in {'retrieved_at', 'first_verified_at'}}
            prepared.append((sha256(_canonical(identity).encode()).hexdigest(),
                sha256(canonical.encode()).hexdigest(), event, external, record.get('session_id'),
                record.get('version_id'), record.get('driver_id'), record['status'], known_at, canonical))
        if not prepared:
            return 0
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.db_path)) as connection:
            if (connection.execute('PRAGMA user_version').fetchone()[0] == 0
                    and not connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchone()):
                connection.executescript(_SCHEMA)
            self._check_schema(connection)
            with connection:
                before = connection.total_changes
                connection.executemany('INSERT OR IGNORE INTO context_records VALUES(?,?,?,?,?,?,?,?,?,?)', prepared)
                return connection.total_changes - before

    def read(self, *, event_id=None, external_event_ref=None, session_id=None,
             version_id=None, driver_id=None, as_of=None):
        """Return independent facts/reports with citations; no synthesis or cause inference.

        as_of uses conservative local knowledge time, never just the event date.
        Multiple publications remain separate; callers should pin version_id.
        """
        if not self.db_path.is_file():
            return []
        query, parameters = [], []
        for key, value in {'event_id': event_id, 'external_event_ref': external_event_ref,
                'session_id': session_id, 'version_id': version_id, 'driver_id': driver_id}.items():
            if value is not None:
                query.append(key + '=?')
                parameters.append(value)
        if as_of is not None:
            query.append('known_at<=?')
            parameters.append(_utc(as_of))
        sql = 'SELECT record_id,record_sha256,known_at,canonical_record FROM context_records'
        if query:
            sql += ' WHERE ' + ' AND '.join(query)
        with closing(sqlite3.connect(self.db_path.as_uri() + '?mode=ro', uri=True)) as connection:
            connection.execute('PRAGMA query_only=ON')
            self._check_schema(connection)
            return [dict(json.loads(raw), record_id=identifier, record_sha256=digest,
                         known_at=known_at, canonical_record=raw)
                    for identifier, digest, known_at, raw in connection.execute(
                        sql + ' ORDER BY known_at,record_id', parameters)]

    def import_archive(self, archive_root, *, versions, retrieved_at, cancellation_session_ids=()):
        """Copy explicit published version evidence using a read-only source connection.

        No current-version reselection, network, archive initialization, or source
        mutation occurs. Cancellation IDs must be both curated and in this calendar.
        """
        from .notebook_audit import NOT_HELD
        source_db = (Path(archive_root).resolve() / 'f1.sqlite3').resolve()
        if not source_db.is_file() or source_db == self.db_path:
            raise ValueError('A separate existing source archive is required')
        _utc(retrieved_at)
        records = []
        with closing(sqlite3.connect(source_db.as_uri() + '?mode=ro', uri=True)) as connection:
            connection.execute('PRAGMA query_only=ON')
            connection.row_factory = sqlite3.Row
            for session_id, version_id in versions.items():
                meta = connection.execute(
                    'SELECT v.*,s.event_id,s.session_code FROM session_versions v '
                    'JOIN sessions s USING(session_id) WHERE v.version_id=? AND v.session_id=? '
                    "AND v.status='published'", (version_id, session_id)).fetchone()
                if meta is None:
                    raise ValueError('Pinned version is not published for session: ' + session_id)
                sources = [dict(row) for row in connection.execute(
                    'SELECT DISTINCT o.source_id,o.provider publisher,o.canonical_url url,'
                    'o.payload_sha256 source_sha256,o.original_published_at_utc published_at,'
                    'o.first_fetched_at_utc first_verified_at FROM source_objects o '
                    'JOIN session_version_sources v USING(source_id) WHERE v.version_id=? '
                    'ORDER BY o.source_id', (version_id,))]
                for result in connection.execute('SELECT * FROM session_results WHERE version_id=?', (version_id,)):
                    row = dict(result)
                    status = classify_outcome(row['status_raw'], meta['session_code'],
                        laps_completed=row['laps_completed'], classified_position_raw=row['classified_position_raw'])
                    records.append(dict(event_id=meta['event_id'], session_id=session_id, version_id=version_id,
                        driver_id=row['driver_id'], status=status, raw_status=row['status_raw'],
                        laps_completed=row['laps_completed'], classified_position_raw=row['classified_position_raw'],
                        original_normalized_status=row['normalized_outcome_status'],
                        summary='Recorded result status; incident reason independently unverified.',
                        evidence_kind='archive_result', assertion_kind='fact', cause_qualification='unknown',
                        confidence='recorded', publisher='Local F1 archive', author=None,
                        url=source_db.as_uri().replace('file:', 'archive:', 1), source_sha256=None,
                        sources=sources, published_at=row['original_publication_utc'], retrieved_at=retrieved_at,
                        archive_published_at=meta['created_at_utc'], archive_path=str(source_db)))
            for session_id in dict.fromkeys(cancellation_session_ids):
                if session_id not in NOT_HELD:
                    raise ValueError('Not a known cancelled session: ' + session_id)
                session = connection.execute('SELECT * FROM sessions WHERE session_id=?', (session_id,)).fetchone()
                if session is None:
                    raise ValueError('Cancelled session is absent from this calendar: ' + session_id)
                summary, url = NOT_HELD[session_id]
                records.append(dict(event_id=session['event_id'], session_id=session_id, status='cancelled',
                    summary=summary, publisher='Local curated NOT_HELD references', url=url,
                    evidence_kind='cancellation', assertion_kind='fact', cause_qualification='stated',
                    confidence='curated', published_at=None, retrieved_at=retrieved_at,
                    source_sha256=None, provenance='data.archive.notebook_audit.NOT_HELD'))
        return self.import_records(records)
