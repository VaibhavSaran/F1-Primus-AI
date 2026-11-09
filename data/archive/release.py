"""Offline, version-pinned yearly publication preparation. Never uploads or fetches.

Dense processed Parquet is streamed once into stored round bundles. The catalogue
is rebuilt from selected relational tables; source caches are never copied.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing, contextmanager
from datetime import date, datetime, timezone
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import sqlite3
import stat
import time
import zipfile

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

COHORT = '87cf0ea1a028ed7d738409699e472741e251a9a4b3c952176fb73fab02535c83'
SOURCE_FIELDS = ('source_id', 'provider', 'endpoint_kind', 'payload_sha256',
    'first_fetched_at_utc', 'original_published_at_utc', 'original_issued_at_utc',
    'original_available_at_utc', 'availability_basis', 'source_schema_version')
TABLES = ('seasons', 'events', 'sessions', 'circuits', 'circuit_layouts', 'drivers',
    'teams', 'session_versions', 'session_entries', 'session_results', 'laps',
    'track_status_events', 'session_status_events', 'race_control_messages',
    'circuit_markers', 'weather_series', 'weather_observations', 'coverage')
EXCLUDED_COLUMNS = {'ingestion_run_id', 'metadata_json', 'coordinate_frame_metadata_json'}


def checked_path(root, relative):
    """Reject drive paths, traversal and filesystem links, including parent links."""
    root = Path(root).absolute()
    relative = str(relative)
    parts = PurePosixPath(relative).parts
    if (not parts or '\\' in relative or PurePosixPath(relative).is_absolute()
            or PureWindowsPath(relative).drive or any(p in ('.', '..') or ':' in p
                or p.endswith((' ', '.')) or '\x00' in p for p in parts)):
        raise ValueError('Unsafe release path')
    target = root.joinpath(*parts)
    for part in dict.fromkeys((root, *root.parents, *target.parents, target)):
        try:
            attributes = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(attributes.st_mode) or getattr(attributes, 'st_file_attributes', 0) & 0x400:
            raise ValueError('Linked release/archive path is forbidden')
    return target


def safe_source(record):
    return {key: record[key] for key in SOURCE_FIELDS if key in record}


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + '\n', encoding='utf-8')


def _hash(path):
    digest = sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _check_parquet_identity(parquet, item):
    for name in ('session_clock_ns', 'timestamp_utc_ns'):
        if name in parquet.schema_arrow.names and not pa.types.is_integer(parquet.schema_arrow.field(name).type):
            raise ValueError('Canonical nanosecond clock must be integer')
    for name in ('driver_id', 'version_id'):
        if item.get(name) is not None and name not in parquet.schema_arrow.names:
            raise ValueError('Dense publication/driver identity column missing')
    identities = {k: item[k] for k in ('driver_id', 'version_id')
                  if item.get(k) is not None and k in parquet.schema_arrow.names}
    for batch in parquet.iter_batches(batch_size=65536, columns=list(identities)):
        for name, expected in identities.items():
            column = batch.column(batch.schema.get_field_index(name))
            if column.null_count or pc.all(pc.equal(column, expected)).as_py() is False:
                raise ValueError('Dense publication/driver identity mismatch')


@contextmanager
def _read_db(path):
    path = Path(path).absolute()
    checked_path(path.parent, path.name)
    connection = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute('PRAGMA query_only=ON')
    connection.execute('BEGIN')
    try:
        yield connection
    finally:
        connection.close()


def write_bundle(source_root, target, items):
    """Validate metadata and streaming source hashes while writing fixed ZIP members."""
    records = []
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_STORED, allowZip64=True) as bundle:
        seen = set()
        for member, item in items:
            checked_path(Path(target).parent, member)
            if member in seen:
                raise ValueError('Duplicate bundle member')
            seen.add(member)
            source = checked_path(source_root, item['relative_path'])
            if source.suffix != '.parquet' or not source.is_file():
                raise ValueError('Only Parquet files can be bundled')
            before = source.stat()
            parquet = pq.ParquetFile(source)
            if parquet.metadata.num_rows != item['row_count']:
                raise ValueError('Parquet row count mismatch')
            _check_parquet_identity(parquet, item)
            schema = str(parquet.schema_arrow)
            info = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            digest = sha256()
            with source.open('rb') as incoming, bundle.open(info, 'w', force_zip64=True) as outgoing:
                for block in iter(lambda: incoming.read(4 * 1024 * 1024), b''):
                    digest.update(block)
                    outgoing.write(block)
            after = source.stat()
            if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
                raise ValueError('Source mutated during release build')
            if digest.hexdigest() != item['file_sha256']:
                raise ValueError('Parquet source hash mismatch')
            records.append({'bundle': Path(target).name, 'member': member,
                'sha256': digest.hexdigest(), 'bytes': before.st_size,
                'rows': item['row_count'], 'kind': item['kind'],
                'driver_id': item.get('driver_id'), 'schema': schema})
    return records


def _copy_table(source, target, name, query, parameters=(), *, columns=None, output_name=None):
    """Keep exact SQLite integers and declared keys; never round-trip through float."""
    output_name = output_name or name
    info = list(source.execute(f'PRAGMA table_info("{name}")'))
    selected = [r for r in info if r['name'] not in EXCLUDED_COLUMNS
                and (columns is None or r['name'] in columns)]
    names = [r['name'] for r in selected]
    key = sorted((r for r in selected if r['pk']), key=lambda r: r['pk'])
    definitions = [f'"{r["name"]}" {r["type"] or "TEXT"}' for r in selected]
    if key:
        definitions.append('PRIMARY KEY(' + ','.join(f'"{r["name"]}"' for r in key) + ')')
    target.execute(f'CREATE TABLE "{output_name}" (' + ','.join(definitions) + ')')
    cursor = source.execute('SELECT ' + ','.join('"' + n + '"' for n in names) + ' FROM ' + query, parameters)
    count = 0
    for batch in iter(lambda: cursor.fetchmany(5000), []):
        target.executemany(f'INSERT INTO "{output_name}" VALUES (' + ','.join('?' for _ in names) + ')', batch)
        count += len(batch)
    return count


def _catalogue(f1_root, weather_root, context_root, target, sessions, weather, year):
    versions = [s['version_id'] for s in sessions]
    marks = ','.join('?' for _ in versions) or 'NULL'
    counts = {}
    with _read_db(Path(f1_root) / 'f1.sqlite3') as source, closing(sqlite3.connect(target)) as output:
        for session in sessions:
            row = source.execute('SELECT session_id,status FROM session_versions WHERE version_id=?',
                                 (session['version_id'],)).fetchone()
            if not row or row['session_id'] != session['session_id'] or row['status'] != 'published':
                raise ValueError('Pinned session publication no longer matches')
        for table in TABLES:
            args = versions
            if table == 'seasons':
                condition, args = 'year=?', [year]
            elif table == 'events':
                condition, args = 'season=?', [year]
            elif table == 'sessions':
                condition, args = 'event_id IN (SELECT event_id FROM events WHERE season=?)', [year]
            elif table in ('circuits', 'circuit_layouts'):
                field = 'circuit_id' if table == 'circuits' else 'layout_id'
                condition, args = f'{field} IN (SELECT {field} FROM events WHERE season=?)', [year]
            elif table in ('drivers', 'teams'):
                field = 'driver_id' if table == 'drivers' else 'team_id'
                condition = f'{field} IN (SELECT {field} FROM session_entries WHERE version_id IN ({marks}))'
            elif table == 'weather_series':
                condition = f'session_version_id IN ({marks})'
            elif table == 'weather_observations':
                condition = f'series_id IN (SELECT series_id FROM weather_series WHERE session_version_id IN ({marks}))'
            else:
                condition = f'version_id IN ({marks})'
            counts[table] = _copy_table(source, output, table, f'{table} WHERE {condition}', args)
        for view in ('v_stints', 'v_pit_visits'):
            counts[view] = _copy_table(source, output, view, f'{view} WHERE version_id IN ({marks})', versions)
        series_ids = [w['series_id'] for w in weather if w.get('series_id')]
        wm = ','.join('?' for _ in series_ids) or 'NULL'
        with _read_db(Path(weather_root) / 'f1.sqlite3') as ws:
            for item in weather:
                if not item.get('series_id'):
                    continue
                path = checked_path(weather_root, item['relative_path'])
                if _hash(path) != item['file_sha256']:
                    raise ValueError('Pinned weather Parquet hash mismatch')
                expected = item['observation_count']
                if (pq.ParquetFile(path).metadata.num_rows != expected or ws.execute(
                        'SELECT count(*) FROM weather_observations WHERE series_id=?',
                        (item['series_id'],)).fetchone()[0] != expected):
                    raise ValueError('Pinned weather observation count mismatch')
            for table in ('weather_series', 'weather_observations'):
                counts['reanalysis_' + table] = _copy_table(ws, output, table,
                    f'{table} WHERE series_id IN ({wm})', series_ids, output_name='reanalysis_' + table)
        output.execute('CREATE TABLE session_series_links(session_id TEXT,series_id TEXT,event_id TEXT,PRIMARY KEY(session_id,series_id))')
        for w in weather:
            if w.get('series_id'):
                output.executemany('INSERT INTO session_series_links VALUES (?,?,?)',
                    [(s['session_id'], w['series_id'], w['event_id']) for s in sessions
                     if s['session_id'].rsplit(':', 1)[0] == w['event_id']])
        output.execute('CREATE TABLE curated_context(record_id TEXT PRIMARY KEY,record_sha256 TEXT,canonical_record TEXT)')
        if context_root and (Path(context_root) / 'context.sqlite3').is_file():
            with _read_db(Path(context_root) / 'context.sqlite3') as context:
                for record in context.execute('SELECT * FROM context_records WHERE event_id LIKE ? ORDER BY record_id', (str(year) + ':%',)):
                    value = json.loads(record['canonical_record'])
                    # Context's canonical hash refers to the private source record. Export a
                    # deliberately smaller public projection with a separate content hash.
                    allowed = ('event_id', 'session_id', 'version_id', 'driver_id', 'status',
                        'known_at', 'published_at', 'publisher', 'summary', 'assertion_kind',
                        'confidence', 'cause_qualification', 'evidence_kind', 'raw_status',
                        'original_normalized_status', 'classified_position_raw', 'laps_completed')
                    clean = {k: value[k] for k in allowed if k in value}
                    clean['known_at'] = record['known_at']
                    if clean.get('version_id') and clean['version_id'] not in versions:
                        continue
                    encoded = json.dumps(clean, sort_keys=True, ensure_ascii=False)
                    output.execute('INSERT INTO curated_context VALUES (?,?,?)',
                                   (record['record_id'], sha256(encoded.encode()).hexdigest(), encoded))
        output.execute('CREATE TABLE source_objects(source_id TEXT PRIMARY KEY,provider TEXT,endpoint_kind TEXT,payload_sha256 TEXT,first_fetched_at_utc TEXT,original_published_at_utc TEXT,original_issued_at_utc TEXT,original_available_at_utc TEXT,availability_basis TEXT,source_schema_version TEXT)')
        ids = set()
        for table in TABLES:
            for info in output.execute(f'PRAGMA table_info("{table}")'):
                if info[1] == 'source_id' or info[1].endswith('_source_id'):
                    ids.update(r[0] for r in output.execute(f'SELECT DISTINCT "{info[1]}" FROM "{table}"') if r[0])
        ids.update(link['source_id'] for s in sessions for link in s.get('source_links', []))
        ids.update(w['source_id'] for w in weather if w.get('source_id'))
        with _read_db(Path(weather_root) / 'f1.sqlite3') as ws:
            for src in (source, ws):
                for row in src.execute('SELECT ' + ','.join(SOURCE_FIELDS) + ' FROM source_objects ORDER BY source_id'):
                    if row['source_id'] in ids:
                        output.execute('INSERT OR IGNORE INTO source_objects VALUES (' + ','.join('?' for _ in SOURCE_FIELDS) + ')', tuple(row))
        output.execute('CREATE INDEX laps_session_driver ON laps(version_id,driver_id,lap_number)')
        output.commit()
        if output.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise ValueError('Export catalogue integrity failure')
    return counts


def measure_release(cohort, f1_root, years):
    result = {}
    for year in years:
        selected = [s for s in cohort['sessions'] if s['season'] == year]
        items = [f for s in selected for f in s['files'] if f['kind'] in ('car', 'positions')]
        result[year] = {'sessions': len(selected), 'files': len(items),
            'bytes': sum(checked_path(f1_root, f['relative_path']).stat().st_size for f in items)}
    return result


def _documents(root, year, summary):
    (root / 'README.md').write_text(f'''# Primus F1 {year}: prepared local release

Status: PREPARED LOCALLY, NOT PUBLISHED. Redistribution rights and Kaggle owner,
slug, licence and visibility require review. No blanket data licence is assigned.
{f"In-progress 2026 season; event-date scope through {summary['data_through']} America/Chicago. See quality.json for pending sessions and weather." if year == 2026 else "Historical season; indexed sessions can still have partial telemetry."}

Open catalogue.sqlite with Python sqlite3 (no project code needed). It contains
events/scheduled sessions, version-pinned entries/results/laps, derived v_stints
and v_pit_visits tables, measured track weather, independent reanalysis and a
session_series_links bridge. Dense car/position Parquet is inside bundles/round-*.zip.
manifest.json maps every member to a version/session and records rows, hashes,
schema and bytes. No extraction is required: PyArrow can read a member in memory.
The SQLite catalogue contains the small reference and analytical tables, avoiding
duplicate small Parquet copies. examples.ipynb demonstrates standalone access.

Coverage: {summary['sessions']} pinned sessions; {summary['strict_review_sessions']}
in the strict review subset. That subset is a data-review criterion, not permission
to redistribute or evidence that all channels are complete. See quality.json.
Nulls are unknown/unavailable, never zero-filled. Session-scoped driver numbers
are not permanent identities. Raw parser caches and raw/unmapped datasets are
excluded; unresolved source attribution remains declared in quality.json.

Run validation from the project with python -m data.archive.release --validate PATH.
All bundle source hashes and row counts were checked during streaming export;
validation rereads ZIP CRC, content hashes and Parquet row counts. Extract ZIPs
only after validating member paths and into a new directory. Never overwrite an
existing release. No network, credentials or API access is needed to read this data.
''', encoding='utf-8')
    (root / 'DATA_DICTIONARY.md').write_text('''# Dictionary and clocks

schema.json contains every SQLite column/type/key and dense Arrow schema is in
manifest.json. Join sessions.event_id to events.event_id; session_versions.session_id
to sessions; version_id to entries/results/laps/status/coverage; driver_id to drivers,
team_id to teams. Version IDs pin a publication, never "latest". Dense member paths
are release-local and manifest entries carry session_id/version_id/driver_id/kind.
sample_index orders original samples; repeated clocks are retained.

*_ns is nullable integer nanoseconds, kept exact. *_utc_ns is Unix UTC;
session_clock_ns and lap_*_clock_ns are original provider session clocks relative
to session_versions.t0_utc_ns, not elapsed time since the green flag. *_ns duration
fields are durations. *_kmh is km/h, *_mps is m/s, *_c Celsius, *_hpa hectopascals,
*_pct percent, *_mm millimetres, *_deg degrees; x/y coordinates retain raw provider
scale and circuit rotation/markers and are not geographic lat/lon. Boolean flags
are 0/1/null. Raw status text and normalized outcome status are separate; an outcome
does not establish an incident cause. DNF is not inferred from missing coordinates.

v_stints groups observed laps by version/driver/stint, gives first/last observed lap
and missing count; compound is null when mixed. v_pit_visits pairs an observed
pit-in to the next eligible observed pit-out, retains unmatched visits. Its duration
is pit-lane elapsed time, never stationary stop duration. These are derived tables.

weather_observations contains measured session track observations.
reanalysis_weather_observations is independent historical grid reanalysis, joined
via reanalysis_weather_series and session_series_links. Reanalysis is hindsight,
not a track sensor or an archived forecast. Spatial coordinates/provider/model,
units_metadata_json and source IDs remain explicit. No causal forecast claim is made.
For replay use observations at or before the playback clock and expire stale
measurements (example: 120 seconds). Do not treat carried parser fills or interpolation
as new measurements; retain quality flags. Timing gaps/ranks are not inferred from
XY or final results; this release offers observed lap_position checkpoints only.
''', encoding='utf-8')
    (root / 'ATTRIBUTION.md').write_text('''# Sources and redistribution review

F1 session observations: Formula 1 timing feeds processed by FastF1 and the pinned
local normalizer; source_objects/source_versions.json retain source hashes and
publication/fetch times where known. A software licence does not establish feed
redistribution rights. Review https://docs.fastf1.dev/ and
https://www.formula1.com/ before any publication. No F1 data licence is asserted.

Independent weather: Open-Meteo historical weather API and underlying reanalysis
model identified per series. Terms: https://open-meteo.com/en/terms ; documentation:
https://open-meteo.com/en/docs/historical-weather-api . Open-Meteo attribution and
applicable model acknowledgments must be reviewed before upload. Do not assign a
blanket licence to the combined F1/weather/context payload.

Curated context is a sanitized projection of locally reviewed assertions, retaining
known_at/publication evidence and uncertainty; no independently unverified incident
cause becomes a fact. Local archive paths, source URLs, author details and raw
payloads were removed from context. Original source record hashes are not claimed
to hash this transformed projection. No credentials or private notebook outputs.
''', encoding='utf-8')
    _json(root / 'dataset-metadata.template.json', {'title': f'Primus F1 {year} — Sessions, Telemetry and Weather' + (f" — in progress through {summary['data_through']}" if year == 2026 else ''),
        'id': 'REVIEW_REQUIRED_OWNER/REVIEW_REQUIRED_SLUG', 'licenses': [],
        'description': 'Local preparation only. Rights, licence, owner and visibility unresolved; do not upload this template.',
        'keywords': ['formula-one', 'telemetry', 'weather']})
    cells = [
        ('markdown', '# Read a yearly release offline\nRun from the release folder. Python 3.10+, pandas and PyArrow are required; matplotlib is optional. Reanalysis is hindsight, never a forecast.'),
        ('code', "from pathlib import Path\nimport sqlite3, json, zipfile\nimport pandas as pd\nimport pyarrow.parquet as pq\nroot = Path.cwd()\nmanifest = json.loads((root/'manifest.json').read_text())\ndb = sqlite3.connect((root/'catalogue.sqlite').as_uri() + '?mode=ro', uri=True)\nsessions = pd.read_sql_query('SELECT * FROM session_versions ORDER BY session_id', db)\nsessions.head()"),
        ('code', "version = sessions.iloc[0].version_id\nlaps = pd.read_sql_query('SELECT * FROM laps WHERE version_id=?', db, params=[version])\n# Compare observed, timed laps. Apply your own declared clean-lap filters.\nlaps.groupby('driver_id').lap_time_ns.min().div(1e9).sort_values().head(10)"),
        ('code', "members = [m for m in manifest['members'] if m['version_id']==version and m['kind']=='positions']\nif members:\n    item = members[0]\n    with zipfile.ZipFile(root/'bundles'/item['bundle']) as bundle:\n        with bundle.open(item['member']) as stream:\n            positions = pq.read_table(stream).to_pandas()\n    display(positions.head())\n    # Observed samples only: no inference of running ranks from XY.\n    clock = int(positions.session_clock_ns.dropna().iloc[0])\n    observed = positions[positions.session_clock_ns.le(clock)]\n    display(observed.tail(1))\n    weather = pd.read_sql_query('SELECT w.* FROM weather_observations w JOIN weather_series s USING(series_id) WHERE s.session_version_id=? AND w.session_clock_ns<=? AND w.session_clock_ns>=? ORDER BY w.session_clock_ns DESC LIMIT 1', db, params=[version, clock, clock-120_000_000_000])\n    display(weather)\nreanalysis = pd.read_sql_query('SELECT * FROM reanalysis_weather_series', db)\nreanalysis.head()")]
    _json(root / 'examples.ipynb', {'nbformat': 4, 'nbformat_minor': 5,
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
        'cells': [{'cell_type': kind, 'metadata': {}, 'source': text.splitlines(keepends=True),
                   **({'execution_count': None, 'outputs': []} if kind == 'code' else {})} for kind, text in cells]})
    (root / 'requirements.txt').write_text('pandas>=2.2,<3\npyarrow>=18,<24\n', encoding='utf-8')


def build_year_release(f1_root, weather_root, output_root, year, through_date='2026-11-02',
                       release_version='v1-local', *, cohort_path=None, context_root=None):
    from .notebook_audit import NOT_HELD
    if year not in range(2020, 2027) or date.fromisoformat(through_date).year != 2026:
        raise ValueError('Unsupported year/cutoff')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,63}', release_version):
        raise ValueError('Unsafe release version')
    f1_root, weather_root, output_root = map(lambda p: Path(p).absolute(), (f1_root, weather_root, output_root))
    roots = [f1_root, weather_root] + ([Path(context_root).absolute()] if context_root else [])
    for source in roots:
        if output_root.resolve().is_relative_to(source.resolve()) or source.resolve().is_relative_to(output_root.resolve()):
            raise ValueError('Output and source archives must be separate')
    final = checked_path(output_root, f'f1-{year}/{release_version}')
    staging = checked_path(output_root, f'f1-{year}/.{release_version}.building')
    if final.exists() or staging.exists():
        raise ValueError('Release exists; previous outputs are immutable (choose a new version)')
    cohort_path = Path(cohort_path or f'.local/review_cohorts/{COHORT}/manifest.json')
    cohort = json.loads(cohort_path.read_text(encoding='utf-8'))
    sessions = [s for s in cohort['sessions'] if s['season'] == year]
    if any(s['local_race_date'] > through_date for s in sessions):
        raise ValueError('Pinned session exceeds approved cutoff')
    weather = [w for w in cohort['weather_tasks'] if w['event_id'].startswith(str(year) + ':')]
    estimate = measure_release(cohort, f1_root, [year])[year]
    output_root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output_root).free < estimate['bytes'] + 1024**3:
        raise ValueError('Insufficient disk space for local export and safety reserve')
    staging.mkdir(parents=True)
    (staging / 'bundles').mkdir()
    members = []
    for round_number in sorted({s['round_number'] for s in sessions}):
        items, mapping = [], {}
        for session in sorted((s for s in sessions if s['round_number'] == round_number), key=lambda s: s['session_id']):
            for item in sorted(session['files'], key=lambda f: (f['kind'], f.get('driver_id') or '', f['relative_path'])):
                if item['kind'] not in ('car', 'positions'):
                    continue
                # Original hashed filenames are already safe and unique; the manifest
                # carries authoritative identities instead of trusting folder aliases.
                member = f"events/round={round_number:02d}/sessions/{session['session_code']}/{item['kind']}/{Path(item['relative_path']).name}"
                mapping[member] = session
                items.append((member, item))
        records = write_bundle(f1_root, staging / 'bundles' / f'round-{round_number:02d}.zip', items)
        for record in records:
            session = mapping[record['member']]
            record.update(session_id=session['session_id'], version_id=session['version_id'])
        members.extend(records)
        print(json.dumps({'year': year, 'round': round_number, 'bundled_files': len(members),
                          'bytes': sum(r['bytes'] for r in members)}), flush=True)
    counts = _catalogue(f1_root, weather_root, context_root, staging / 'catalogue.sqlite', sessions, weather, year)
    summary = {'year': year, 'sessions': len(sessions),
        'strict_review_sessions': sum(bool(s.get('strict_review_subset')) for s in sessions),
        'weather': dict(Counter(w['status'] for w in weather)), 'season_status': 'in_progress' if year == 2026 else 'historical',
        'data_through': through_date if year == 2026 else f'{year}-12-31',
        'table_rows': counts, 'dense_files': len(members), 'dense_bytes': sum(r['bytes'] for r in members)}
    quality = {'summary': summary, 'sessions': [{k: s[k] for k in ('session_id', 'version_id',
        'required_channel_status', 'quality_flags', 'strict_review_subset') if k in s} for s in sessions],
        'weather': [{k: w[k] for k in ('event_id', 'status', 'series_id', 'observation_count', 'window_start', 'window_end') if k in w} for w in weather],
        'identity_policy': 'Only processed source-attributed car/positions are exported. Raw/unmapped streams excluded; no automatic reassignment. Published partial flags retained.',
        'pending': ['weather:' + w['event_id'] for w in weather if w['status'] != 'complete']}
    with _read_db(staging / 'catalogue.sqlite') as catalog:
        # The calendar includes future/cancelled slots; a missing publication does
        # not imply an acquisition failure or that the scheduled session was held.
        quality['calendar_sessions_without_publication'] = [dict(r) for r in catalog.execute(
            'SELECT s.session_id,s.status,e.local_race_date FROM sessions s JOIN events e USING(event_id) '
            'LEFT JOIN session_versions v USING(session_id) WHERE v.version_id IS NULL ORDER BY s.session_id')]
        quality['pending'].extend(row['session_id'] for row in quality['calendar_sessions_without_publication']
                                  if row['local_race_date'] and row['local_race_date'] <= through_date
                                  and row['session_id'] not in NOT_HELD)
        schema = {r[0]: [dict(x) for x in catalog.execute(f'PRAGMA table_info("{r[0]}")')]
                  for r in catalog.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")}
        sources = [dict(r) for r in catalog.execute('SELECT * FROM source_objects ORDER BY source_id')]
    _json(staging / 'quality.json', quality)
    _json(staging / 'schema.json', schema)
    _json(staging / 'source_versions.json', {'cohort_id': cohort_path.parent.name,
        'cohort_manifest_sha256': _hash(cohort_path), 'sessions': [{k: s[k] for k in
        ('session_id', 'version_id', 'dataset_hash', 'normalizer_version', 'created_at_utc', 'observed_start_utc_ns', 'observed_end_utc_ns') if k in s} for s in sessions],
        'weather': [{k: w[k] for k in ('event_id', 'series_id', 'source_id', 'file_sha256') if k in w} for w in weather],
        'sources': sources})
    _documents(staging, year, summary)
    files = [{'path': p.relative_to(staging).as_posix(), 'bytes': p.stat().st_size, 'sha256': _hash(p)}
             for p in sorted(staging.rglob('*')) if p.is_file()]
    manifest = {'format_version': 1, 'state': 'prepared_local_not_published', 'release_version': release_version,
        'summary': summary, 'files': files, 'members': members,
        'lineage': 'Pinned processed Parquet copied byte-for-byte; selected SQL relational tables, observed-lap stint/pit views and sanitized context projection. No raw caches or inferred running order.'}
    _json(staging / 'manifest.json', manifest)
    (staging / 'checksums.sha256').write_text(''.join(f"{r['sha256']}  {r['path']}\n" for r in files) + f"{_hash(staging / 'manifest.json')}  manifest.json\n", encoding='utf-8')
    # A complete second read detects output corruption before an atomic promotion.
    validation = validate_release(staging)
    _json(staging / 'validation.json', validation)
    _json(staging / 'build_receipt.json', {'release_built_at': datetime.now(timezone.utc).isoformat(),
        'state': 'prepared_local_not_published', 'manifest_sha256': _hash(staging / 'manifest.json')})
    staging.rename(final)
    print(json.dumps({'completed': str(final), **summary}), flush=True)
    return final


def validate_release(root):
    root = Path(root).absolute()
    manifest = json.loads(checked_path(root, 'manifest.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        path = checked_path(root, item['path'])
        if path.stat().st_size != item['bytes'] or _hash(path) != item['sha256']:
            raise ValueError('Release file hash or size mismatch')
    members = manifest.get('members', [])
    for name in sorted({m['bundle'] for m in members}):
        expected = {m['member']: m for m in members if m['bundle'] == name}
        with zipfile.ZipFile(checked_path(root, 'bundles/' + name)) as bundle:
            if set(bundle.namelist()) != set(expected):
                raise ValueError('Unexpected or missing bundle member')
            for member, item in expected.items():
                checked_path(root, member)
                digest = sha256()
                with bundle.open(member) as stream:
                    for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                        digest.update(block)
                if digest.hexdigest() != item['sha256']:
                    raise ValueError('Bundled member hash mismatch')
                with bundle.open(member) as stream:
                    parquet = pq.ParquetFile(stream)
                    if parquet.metadata.num_rows != item['rows']:
                        raise ValueError('Bundled member row mismatch')
                    _check_parquet_identity(parquet, item)
                    _scan_text(str(parquet.schema_arrow))
    catalogue = root / 'catalogue.sqlite'
    if catalogue.exists():
        with _read_db(catalogue) as db:
            if db.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
                raise ValueError('Catalogue corruption')
            for table in ('session_entries', 'laps', 'session_results', 'coverage'):
                if db.execute(f'SELECT count(*) FROM {table} t LEFT JOIN session_versions v USING(version_id) WHERE v.version_id IS NULL').fetchone()[0]:
                    raise ValueError('Orphan publication relationship')
            if db.execute('SELECT count(*) FROM laps l LEFT JOIN session_entries e USING(version_id,driver_id) WHERE e.driver_id IS NULL').fetchone()[0]:
                raise ValueError('Orphan driver relationship')
            _scan_catalogue(db)
    for item in manifest['files']:
        if item['path'].endswith(('.json', '.md', '.ipynb', '.txt')):
            _scan_text(checked_path(root, item['path']).read_text(encoding='utf-8'))
    return {'passed': True, 'files': len(manifest['files']), 'dense_members': len(members),
            'checks': ['file SHA256/bytes', 'ZIP member CRC/SHA256', 'Parquet rows/driver/version identity/integer clocks', 'catalogue integrity', 'publication/driver joins', 'portable payload metadata leakage scan'],
            'publication': 'not_published', 'redistribution_review': 'required'}


_PRIVATE = re.compile(r'(?i)(?:(?<![A-Za-z0-9])[A-Z]:[\\/]|archive://|/Users/|/home/|\bBearer\s+[A-Za-z0-9._-]{12,}|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|(?:password|authorization|api[_-]?key)[\"\x27]?\s*[:=]\s*[\"\x27]?[A-Za-z0-9._-]{8,})')


def _scan_text(value):
    if _PRIVATE.search(value):
        raise ValueError('Private machine path, account or credential marker in payload')


def _scan_catalogue(db):
    for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'"):
        table = row[0]
        columns = [c[1] for c in db.execute(f'PRAGMA table_info("{table}")') if c[2].upper() == 'TEXT']
        if any(c in ('raw_path', 'canonical_url', 'request_fingerprint', 'checkpoint_json') for c in columns):
            raise ValueError('Private runtime field in release catalogue')
        if columns:
            query = 'SELECT ' + ','.join('"' + c + '"' for c in columns) + f' FROM "{table}"'
            for values in db.execute(query):
                for value in values:
                    if isinstance(value, str):
                        _scan_text(value)


def write_context_availability(output_root, context_root):
    """Restore exact local knowledge timestamps without changing immutable packages.

    known_at means when the local context store knew an assertion, never its event
    date and never evidence that an assertion was available before a historical race.
    """
    output_root = Path(output_root).absolute()
    as_of = datetime.now(timezone.utc).isoformat()
    with _read_db(Path(context_root) / 'context.sqlite3') as source:
        known = {r['record_id']: r['known_at'] for r in source.execute(
            'SELECT record_id,known_at FROM context_records ORDER BY record_id')}
    records = []
    for manifest_path in sorted(output_root.glob('f1-*/*/manifest.json')):
        if manifest_path.parent.name.startswith('.'):
            continue
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        year, version = manifest['summary']['year'], manifest['release_version']
        rows = []
        with _read_db(manifest_path.parent / 'catalogue.sqlite') as catalogue:
            for row in catalogue.execute('SELECT record_id,record_sha256 FROM curated_context ORDER BY record_id'):
                if row['record_id'] not in known:
                    raise ValueError('Exported context record absent from current context snapshot')
                rows.append({'record_id': row['record_id'], 'known_at': known[row['record_id']],
                             'exported_record_sha256': row['record_sha256']})
        relative = f'context-availability/f1-{year}-{version}.json'
        target = checked_path(output_root, relative)
        value = {'year': year, 'release_version': version,
            'manifest_sha256': _hash(manifest_path), 'context_snapshot_read_at_utc': as_of,
            'known_at_basis': 'local_context_store_knowledge_time_not_event_date', 'records': rows}
        _scan_text(json.dumps(value))
        if target.exists():
            previous = json.loads(target.read_text(encoding='utf-8'))
            if (previous['manifest_sha256'] != value['manifest_sha256']
                    or previous['records'] != value['records']):
                raise ValueError('Context availability sidecar exists with different content')
        else:
            _json(target, value)
        records.append({'year': year, 'release_version': version, 'path': relative,
                        'sha256': _hash(target), 'records': len(rows)})
    return records


def build_release_index(output_root):
    """Cross-year metadata only; never mutates the completed yearly releases."""
    from .notebook_audit import NOT_HELD
    output_root = Path(output_root).absolute()
    index = {'state': 'prepared_local_not_published', 'cohort_id': None,
             'rights_review': 'required', 'datasets': [], 'known_cancelled_sessions': []}
    for manifest_path in sorted(output_root.glob('f1-*/*/manifest.json')):
        if manifest_path.parent.name.startswith('.'):
            continue
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        summary = manifest['summary']
        lineage = json.loads((manifest_path.parent / 'source_versions.json').read_text(encoding='utf-8'))
        index['datasets'].append({'year': summary['year'], 'release_version': manifest['release_version'],
            'cohort_id': lineage['cohort_id'],
            'path': manifest_path.parent.relative_to(output_root).as_posix(),
            'manifest_sha256': _hash(manifest_path), 'summary': summary,
            'package_bytes': sum(p.stat().st_size for p in manifest_path.parent.rglob('*') if p.is_file()),
            'kaggle_url': None, 'publication': 'not_published'})
        sidecar = checked_path(output_root,
            f"context-availability/f1-{summary['year']}-{manifest['release_version']}.json")
        if sidecar.is_file():
            index['datasets'][-1]['context_availability'] = {
                'path': sidecar.relative_to(output_root).as_posix(), 'sha256': _hash(sidecar),
                'known_at_basis': 'local_context_store_knowledge_time_not_event_date'}
    index['cohort_ids'] = sorted({item['cohort_id'] for item in index['datasets']})
    if len(index['cohort_ids']) == 1:
        index['cohort_id'] = index['cohort_ids'][0]
    for session_id, (reason, source_url) in sorted(NOT_HELD.items()):
        index['known_cancelled_sessions'].append({'session_id': session_id, 'status': 'not_held',
                                                  'reason': reason, 'source_url': source_url})
    _json(output_root / 'release-index.json', index)
    (output_root / 'README.md').write_text('''# Prepared local yearly F1 releases

These releases have NOT been published. Redistribution rights, licence, Kaggle
owner/slug and visibility remain for review. release-index.json identifies every
completed yearly version, size, manifest hash and its paired context availability
sidecar. postbuild-validation.json records enhanced checks against each manifest.
standalone-example-validation.json records offline example execution.

Copy the selected f1-YYYY/VERSION directory and its indexed context-availability
sidecar when moving a year independently. Join sidecar record_id to the catalogue's
curated_context.record_id; verify exported_record_sha256 and manifest_sha256.
The sidecar restores the context store's exact known_at to early sanitized exports
without modifying immutable yearly packages. known_at is LOCAL KNOWLEDGE TIME,
never event date or evidence that a later assertion was known before a race.
Missing availability is unknown; do not infer historical knowledge from results.

Known cancelled sessions and their references are listed separately in the index.
Original calendar status may remain scheduled; curated_context cancellation records
state not-held evidence. A missing publication is not a claim of acquisition failure.
2026 is in progress. Each indexed summary declares its data-through date;
each package's quality.json records pending sessions and independent weather.
Each yearly README/dictionary explains clocks, joins, flags and the offline notebook.
''', encoding='utf-8')
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Build locally after disk measurement; no upload')
    parser.add_argument('--validate', type=Path)
    parser.add_argument('--years', nargs='+', type=int, default=list(range(2020, 2027)))
    parser.add_argument('--version', default='v1-local')
    parser.add_argument('--output', type=Path, default=Path('.local/release_packages'))
    parser.add_argument('--cohort', type=Path, default=Path(f'.local/review_cohorts/{COHORT}/manifest.json'))
    args = parser.parse_args()
    if args.validate:
        print(json.dumps(validate_release(args.validate), indent=2))
        return
    cohort = json.loads(args.cohort.read_text(encoding='utf-8'))
    f1, weather = Path('.local/f1_2025_2026'), Path('.local/f1_weather')
    measurements = measure_release(cohort, f1, args.years)
    print(json.dumps({'measurement': measurements, 'free_bytes': shutil.disk_usage('.').free}), flush=True)
    if args.execute:
        started = time.monotonic()
        for year in args.years:
            build_year_release(f1, weather, args.output, year, release_version=args.version,
                               cohort_path=args.cohort, context_root=Path('.local/event_context'))
        write_context_availability(args.output, Path('.local/event_context'))
        build_release_index(args.output)
        print(json.dumps({'elapsed_seconds': time.monotonic() - started}), flush=True)


if __name__ == '__main__':
    main()
