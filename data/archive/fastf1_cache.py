"""Reuse the user's trusted FastF1 cache and archived calendar without fetching."""

from contextlib import closing, contextmanager
from datetime import timedelta, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sqlite3
import stat
from uuid import uuid4

import pandas as pd

TRUSTED_CACHE = Path(__file__).resolve().parents[2] / '.fastf1_cache'


def _linked(path):
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, 'st_file_attributes', 0) & 0x400)


def _publish(temporary, target):
    try:
        if os.name == 'nt':
            os.rename(temporary, target)
        else:
            os.link(temporary, target)
    except FileExistsError:
        pass
    finally:
        temporary.unlink(missing_ok=True)


def seed_trusted_cache(source, destination):
    """Copy trusted parser files without unpickling or overwriting cache entries.

    Requests-cache SQLite is copied with the backup API, including committed WAL
    data. Symlinks and Windows junctions are excluded; source files stay untouched.
    """
    source, destination = Path(source).absolute(), Path(destination).absolute()
    result = dict(copied=0, existing=0, skipped_links=0, source_present=source.is_dir())
    if not source.exists():
        return result
    if _linked(source) or (destination.exists() and _linked(destination)):
        raise ValueError('FastF1 cache roots must not be linked paths')
    source, destination = source.resolve(), destination.resolve()
    if source == destination or destination.is_relative_to(source):
        raise ValueError('FastF1 seed destination must be outside the trusted source cache')
    destination.mkdir(parents=True, exist_ok=True)
    for directory, subdirectories, files in os.walk(source, followlinks=False):
        for name in list(subdirectories):
            if _linked(Path(directory) / name):
                subdirectories.remove(name)
                result['skipped_links'] += 1
        for name in sorted(files):
            original = Path(directory) / name
            relative = original.relative_to(source)
            parser_file = original.suffix == '.ff1pkl' and len(relative.parts) > 1 and relative.parts[0].isdigit()
            request_db = relative.as_posix() == 'fastf1_http_cache.sqlite'
            if not parser_file and not request_db:
                continue
            if _linked(original):
                result['skipped_links'] += 1
                continue
            target = destination / relative
            if not target.resolve().is_relative_to(destination):
                raise ValueError('FastF1 cache target escapes destination')
            if target.exists():
                if _linked(target):
                    raise ValueError('FastF1 cache target must not be linked')
                result['existing'] += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f'.{target.name}.{uuid4().hex}.seed')
            try:
                if request_db:
                    with closing(sqlite3.connect(original.as_uri() + '?mode=ro', uri=True)) as incoming:
                        with closing(sqlite3.connect(temporary)) as outgoing:
                            incoming.backup(outgoing)
                else:
                    with original.open('rb') as incoming, temporary.open('xb') as outgoing:
                        shutil.copyfileobj(incoming, outgoing)
                        outgoing.flush()
                        os.fsync(outgoing.fileno())
                _publish(temporary, target)
                result['copied'] += 1
            finally:
                temporary.unlink(missing_ok=True)
    return result


def archived_event(archive, event_id):
    """Construct a real FastF1 Event from the verified original schedule JSON."""
    from fastf1.events import EventSchedule

    with archive.connect() as conn:
        source = conn.execute('''SELECT so.* FROM events e JOIN source_objects so
            ON so.source_id=e.calendar_source_id WHERE e.event_id=?''', (event_id,)).fetchone()
    if source is None or source['provider'] != 'fastf1_schedule':
        raise ValueError('Original archived FastF1 schedule is unavailable')
    path = (archive.root / source['raw_path']).resolve()
    if not path.is_relative_to(archive.root.resolve()):
        raise ValueError('Archived schedule path escapes archive')
    original = path.read_bytes()
    if sha256(original).hexdigest() != source['payload_sha256']:
        raise ValueError('Archived FastF1 schedule hash mismatch')
    payload = json.loads(original)
    frame = pd.DataFrame(payload).reset_index(drop=True)
    frame = frame.rename(columns={name: ''.join(part.capitalize() for part in name.split('_')) for name in frame})
    for index in range(1, 6):
        column = f'Session{index}Date'
        local, utc = [], []
        for row in frame.to_dict('records'):
            value = row.get(column)
            if value is None or pd.isna(value):
                local.append(pd.NaT)
                utc.append(pd.NaT)
                continue
            stamp = pd.Timestamp(value)
            if stamp.tzinfo is None:
                offset = str(row.get('GmtOffset') or '+00:00')
                hours, minutes = map(int, offset.lstrip('+-').split(':')[:2])
                delta = timedelta(hours=hours, minutes=minutes)
                stamp = stamp.tz_localize(timezone(-delta if offset.startswith('-') else delta))
            local.append(stamp)
            utc.append(stamp.tz_convert('UTC').tz_localize(None))
        frame[column] = pd.Series(local, dtype=object)
        frame[f'{column}Utc'] = utc
    frame['EventDate'] = pd.to_datetime(frame['EventDate'])
    year, round_number = map(int, event_id.split(':'))
    return EventSchedule(frame, year=year, _force_default_cols=True).get_event_by_round(round_number)


@contextmanager
def configured_cache(path, *, offline):
    """Scope FastF1 cache globals and prevent bypassing offline mode."""
    import fastf1

    names = ['_CACHE_DIR', '_requests_session_cached', '_FORCE_RENEW', '_IGNORE_VERSION',
             '_tmp_disabled', '_http_tmp_disabled', '_func_tmp_disabled', '_ci_mode',
             '_default_cache_enabled', '_no_cached_warned']
    previous = {name: getattr(fastf1.Cache, name) for name in names if hasattr(fastf1.Cache, name)}
    Path(path).mkdir(parents=True, exist_ok=True)
    current = None
    try:
        if hasattr(fastf1.Cache, 'configure'):
            fastf1.Cache.configure(cache_dir=str(path))
        else:
            fastf1.Cache.enable_cache(str(path))
        current = fastf1.Cache._requests_session_cached
        fastf1.Cache.set_enabled()
        fastf1.Cache._ci_mode = False
        fastf1.Cache.offline_mode(offline)
        yield
    finally:
        if current is not None and current is not previous['_requests_session_cached']:
            current.close()
        for name, value in previous.items():
            setattr(fastf1.Cache, name, value)


def add_original_cached_streams(session, path, result, session_id):
    """Export cached parser frames before Session rounds dates/drops channels.

    These are FastF1 parser outputs, not wire originals. A missing original Time
    marks parser-created zero fills and must not count as an observed sample.
    """
    import fastf1

    cached = fastf1.Cache._requests_session_cached
    before = cached.settings.only_if_cached
    details = result['metadata'].setdefault('original_parser_streams', {})
    result['metadata']['processed_streams_basis'] = (
        'car/positions are FastF1 Session outputs and may round Date/filter channels; '
        'raw_car/raw_positions preserve earlier parser frames, not wire originals')
    entries = {str(row['driver_number']): row['driver_id']
               for row in result['tables']['session_entries'] if row.get('driver_number') is not None}
    try:
        cached.settings.only_if_cached = True
        for kind, function_name in [('car', 'car_data'), ('positions', 'position_data')]:
            cache_file = path / f'{function_name}.ff1pkl'
            if not cache_file.is_file():
                continue
            if _linked(cache_file):
                raise ValueError('Original FastF1 parser cache must not be linked')
            try:
                streams = getattr(fastf1._api, function_name)(session.api_path)
            except (Exception, SystemExit) as exc:
                details[kind] = {'error': f'{type(exc).__name__}: {exc}'}
                continue
            details[kind] = {}
            for number, original in streams.items():
                number = str(number)
                frame = pd.DataFrame(original).copy(deep=True)
                fill = frame['Time'].isna() if 'Time' in frame else pd.Series(False, index=frame.index)
                frame['fastf1_parser_missing_time_fill'] = fill
                driver_id = entries.get(number)
                if driver_id is None:
                    driver_id = f'fastf1:{session_id}:number:{number}'
                    result['tables']['drivers'].append(dict(driver_id=driver_id, source_driver_id=None))
                    result['tables']['session_entries'].append(dict(driver_id=driver_id, driver_number=number,
                                                                  source_driver_id=None))
                    entries[number] = driver_id
                result['datasets'].append(dict(kind=f'raw_{kind}', driver_id=driver_id, table=frame))
                dates = frame['Date'].dropna() if 'Date' in frame else []
                if len(dates):
                    first, last = int(pd.Timestamp(min(dates)).value), int(pd.Timestamp(max(dates)).value)
                    metadata = result['metadata']
                    previous_start, previous_end = metadata.get('observed_start_utc_ns'), metadata.get('observed_end_utc_ns')
                    metadata['observed_start_utc_ns'] = min(first, previous_start) if previous_start is not None else first
                    metadata['observed_end_utc_ns'] = max(last, previous_end) if previous_end is not None else last
                valid = int((~fill & frame['Date'].notna()).sum()) if 'Date' in frame else 0
                details[kind][number] = dict(rows=len(frame), parser_generated_fills=int(fill.sum()),
                    valid_observed_samples=valid, source_columns=list(original.columns),
                    basis='cached_FastF1_parser_output_before_Session_postprocessing')
                result['metadata']['coverage'].append(dict(dataset_kind=f'raw_{kind}/{driver_id}',
                    status='available' if valid == len(frame) and len(frame) else 'partial' if valid else 'unavailable',
                    rows=len(frame), valid_count=valid, invalid_count=len(frame) - valid,
                    reason='Parser missing-Time fills are not observed telemetry; raw JSON remains in HTTP cache'))
                for dataset in result['datasets']:
                    if dataset['kind'] != kind or dataset.get('driver_id') != driver_id or len(dataset['table']) != len(frame):
                        continue
                    table = dataset['table']
                    flags = table['quality_flags'].tolist()
                    for index, generated in enumerate(fill.to_numpy()):
                        if generated:
                            flags[index] = json.dumps(json.loads(flags[index]) + ['fastf1_parser_generated_fill'])
                    table['quality_flags'] = flags
                if fill.any():
                    completeness = result['metadata']['completeness'][kind]
                    driver_coverage = completeness.get('drivers', {}).get(driver_id)
                    # Supplemental raw vehicles retain their own coverage; they
                    # are not part of the processed entrant-channel denominator.
                    if driver_coverage is None:
                        continue
                    completeness['status'] = 'partial'
                    if driver_coverage is not None:
                        driver_coverage.update(status='partial', valid_timestamps=valid,
                                               parser_generated_fills=int(fill.sum()))
                        for column, count in driver_coverage.get('channel_valid_counts', {}).items():
                            generated_count = int((fill & frame[column].notna()).sum()) if column in frame else int(fill.sum())
                            driver_coverage['channel_valid_counts'][column] = max(0, count - generated_count)
                    for row in result['metadata']['coverage']:
                        if row['dataset_kind'] in (kind, f'{kind}/{driver_id}'):
                            row.update(status='partial', reason='FastF1 parser generated missing-Time sample fills')
                            if row['dataset_kind'] != kind:
                                row.update(valid_count=valid, invalid_count=len(frame) - valid)
    finally:
        cached.settings.only_if_cached = before
