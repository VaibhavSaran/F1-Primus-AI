"""Bounded historical FastF1 loading; cached observations are hindsight evidence.

The default loader uses a disposable copy of the existing cache in a child
process. No provider import, network request, or global cache change occurs on
import. Injected loaders return an already loaded session and own their timeout.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import isfinite
from pathlib import Path
import pickle
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from typing import Any, Callable

from data.contracts import EventKey, RunStatus, SourceRecord, require_utc
from data.sessions import session_cache_key

SESSION_IDS = {'FP1', 'FP2', 'FP3', 'Q', 'SQ', 'S', 'R'}


@dataclass(frozen=True)
class LoadedSession:
    event: EventKey
    session_id: str
    entries: tuple[dict[str, Any], ...]
    laps: tuple[dict[str, Any], ...]
    source: SourceRecord
    status: RunStatus
    errors: tuple[str, ...] = ()
    quality_flags: tuple[str, ...] = ()
    session_start_utc: datetime | None = None
    session_end_utc: datetime | None = None


def _present(value):
    import pandas as pd
    return value is not None and not bool(pd.isna(value))


def _text(value):
    return str(value) if _present(value) and str(value).strip() else None


def _number(value):
    if not _present(value) or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if isfinite(result) else None


def _seconds(value):
    return value.total_seconds() if _present(value) and hasattr(value, 'total_seconds') else None


def _utc(value):
    if not _present(value):
        return None
    if hasattr(value, 'to_pydatetime'):
        value = value.to_pydatetime()
    if not isinstance(value, datetime):
        return None
    # FastF1 t0_date/LapStartDate are explicitly UTC, often stored naive.
    return require_utc(value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value)


def _property(session, name, default=None):
    try:
        return getattr(session, name)
    except (AttributeError, RuntimeError):
        return default
    except Exception as error:
        # FastF1 properties raise this when optional telemetry was not loaded.
        if type(error).__name__ == 'DataNotLoadedError':
            return default
        raise


def _normalize(event, session_id, raw, source):
    entries, by_number, flags = [], {}, ['historical_availability_unverified', 'hindsight_reconstruction']
    results = _property(raw, 'results')
    for row in results.to_dict('records') if results is not None else ():
        number = _text(row.get('DriverNumber'))
        if number is None:
            flags.append('entry_without_driver_number')
            continue
        driver = _text(row.get('DriverId')) or f'fastf1:number:{number}'
        if driver.startswith('fastf1:number:'):
            flags.append('driver_identity_fallback')
        if number in by_number or any(item['driver_id'] == driver for item in entries):
            raise ValueError('duplicate entry identity')
        by_number[number] = driver
        position = _number(row.get('Position'))
        entries.append(dict(driver_id=driver, driver_number=number,
                            abbreviation=_text(row.get('Abbreviation')),
                            team_id=_text(row.get('TeamId')) or _text(row.get('TeamName')),
                            position=int(position) if position and position.is_integer() and position > 0 else None,
                            classified_position=_text(row.get('ClassifiedPosition')),
                            status=_text(row.get('Status')), grid_position=_number(row.get('GridPosition'))))
    laps, seen = [], set()
    t0 = _utc(_property(raw, 't0_date'))
    table = _property(raw, 'laps')
    for row in table.to_dict('records') if table is not None else ():
        number, lap_number = _text(row.get('DriverNumber')), _number(row.get('LapNumber'))
        if number not in by_number or lap_number is None or lap_number <= 0 or not lap_number.is_integer():
            flags.append('invalid_lap_identity_removed')
            continue
        key = (by_number[number], int(lap_number))
        if key in seen:
            flags.append('duplicate_laps_removed')
            continue
        seen.add(key)
        deleted_value, accurate_value = row.get('Deleted'), row.get('IsAccurate')
        deleted = bool(deleted_value) if _present(deleted_value) else None
        end_clock, start_clock = _seconds(row.get('Time')), _seconds(row.get('LapStartTime'))
        start_utc = _utc(row.get('LapStartDate'))
        if start_utc is None and t0 is not None and start_clock is not None:
            start_utc = t0 + timedelta(seconds=start_clock)
        end_utc = t0 + timedelta(seconds=end_clock) if t0 is not None and end_clock is not None else None
        # Missing deletion evidence cannot certify a clean lap.
        accurate = bool(accurate_value) if _present(accurate_value) and deleted is not None else False
        laps.append(dict(event_id=event.event_id, session_id=session_id, driver_id=key[0],
                         lap_number=key[1], lap_time_s=_seconds(row.get('LapTime')),
                         provider_is_accurate=bool(accurate_value) if _present(accurate_value) else None,
                         accurate=accurate, deleted=deleted, pit_in=_present(row.get('PitInTime')),
                         pit_out=_present(row.get('PitOutTime')), track_status=_text(row.get('TrackStatus')),
                         compound=_text(row.get('Compound')), tyre_age_laps=_number(row.get('TyreLife')),
                         stint=_number(row.get('Stint')), session_time_s=end_clock,
                         lap_start_session_time_s=start_clock, lap_start_utc=start_utc, lap_end_utc=end_utc,
                         source_id=source.source_id, available_at_utc=source.fetched_at_utc))
    if any(lap['deleted'] is None for lap in laps):
        flags.append('lap_deletion_evidence_unavailable')
    if any(lap['lap_end_utc'] is None for lap in laps):
        flags.append('absolute_lap_times_unavailable')
    start, end = None, None
    statuses = _property(raw, 'session_status')
    for row in statuses.to_dict('records') if statuses is not None and t0 is not None else ():
        seconds = _seconds(row.get('Time'))
        if seconds is None:
            continue
        at = t0 + timedelta(seconds=seconds)
        if row.get('Status') == 'Started' and start is None:
            start = at
        if row.get('Status') == 'Finished':
            end = at
    return LoadedSession(event, session_id, tuple(entries), tuple(laps), source,
                         RunStatus.DEGRADED if entries and laps else RunStatus.INSUFFICIENT_DATA,
                         (), tuple(sorted(set(flags))), start, end)


def load_session(event: EventKey, session_id: str, *, loader: Callable | None = None,
                 cache_path: str | Path = '.fastf1_cache', timeout_s: float = 45,
                 allow_fetch: bool = False, fetched_at_utc: datetime | None = None,
                 include_telemetry: bool = False) -> LoadedSession:
    """Return normalized records without asserting historical publication times.

    ``allow_fetch=True`` explicitly permits free FastF1 HTTP access in the worker.
    Downloaded values remain in the disposable cache; the original stays intact.
    A custom loader is called as ``loader(event, session_id)`` and must be bounded
    by its caller. It exists for already loaded offline fixtures/adapters.
    """
    if session_id not in SESSION_IDS:
        raise ValueError('unsupported session_id')
    if isinstance(timeout_s, bool) or not isfinite(timeout_s) or not 0 < timeout_s <= 300:
        raise ValueError('timeout_s must be finite and between 0 and 300 seconds')
    fetched = require_utc(fetched_at_utc or datetime.now(timezone.utc))
    source = SourceRecord('fastf1', session_cache_key(event, session_id), fetched,
                          url=f'https://www.formula1.com/en/results/{event.season}/races',
                          original_timestamps={'historical_availability': 'unverified',
                                               'cache_mode': 'explicit_fetch' if allow_fetch else 'cache_only'})
    def unavailable(code, status=RunStatus.ERROR):
        return LoadedSession(event, session_id, (), (), source, status, (code,),
                             ('historical_availability_unverified',))
    try:
        if loader is not None:
            return _normalize(event, session_id, loader(event, session_id), source)
        cache = Path(cache_path).resolve()
        if not cache.is_dir() and not allow_fetch:
            return unavailable('cache_unavailable', RunStatus.INSUFFICIENT_DATA)
        work = Path('.scratch/pilot-work')
        work.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=work) as temporary:
            destination = Path(temporary).resolve()
            output = destination / 'normalized.pkl'
            args = [sys.executable, '-c', 'from data.fastf1_source import _worker; _worker()', str(event.season), str(event.round_number),
                    session_id, str(cache), str(destination / 'cache'), str(output),
                    '1' if allow_fetch else '0', fetched.isoformat(), '1' if include_telemetry else '0']
            completed = subprocess.run(args, capture_output=True, timeout=timeout_s, check=False)
            if completed.returncode != 0 or not output.exists():
                return unavailable('provider_load_failed')
            # Only our isolated child writes this file; never accept a provider pickle here.
            with output.open('rb') as stream:
                result = pickle.load(stream)
            # The worker does not know optional caller-supplied circuit metadata.
            return LoadedSession(event, session_id, result.entries, result.laps, result.source,
                                 result.status, result.errors, result.quality_flags,
                                 result.session_start_utc, result.session_end_utc)
    except subprocess.TimeoutExpired:
        return unavailable('provider_timeout')
    except Exception:
        return unavailable('provider_load_failed')


def coverage_report(sessions) -> dict[str, Any]:
    """Coverage only; this does not estimate forecasting accuracy."""
    from ml.features import clean_laps
    sessions = list(sessions)
    laps = [lap for session in sessions for lap in session.laps]
    return dict(sessions_requested=len(sessions), sessions_loaded=sum(bool(item.entries and item.laps) for item in sessions),
                entries=sum(len(item.entries) for item in sessions), total_laps=len(laps),
                eligible_clean_laps=len(clean_laps(laps)),
                laps_with_deletion_evidence=sum(lap['deleted'] is not None for lap in laps),
                laps_with_absolute_utc=sum(lap['lap_end_utc'] is not None for lap in laps),
                strict_historical_eligible_sessions=0,
                unavailable_features=['archived_weather_issue_times', 'historical_publication_times',
                                      'verified_pre_cutoff_entry_lists', 'pit_loss_estimates'],
                sessions=[dict(event_id=item.event.event_id, session_id=item.session_id,
                               status=item.status.value, errors=item.errors, quality_flags=item.quality_flags,
                               historical_analysis_status='hindsight_only' if item.entries and item.laps else 'unavailable',
                               forecast_eligibility='excluded_historical_availability_unverified',
                               entries=len(item.entries), laps=len(item.laps),
                               eligible_clean_laps=len(clean_laps(item.laps))) for item in sessions])


def _worker():
    season, round_number, session_id, original, destination, output, fetch, fetched, telemetry = sys.argv[1:]
    old, cache = Path(original), Path(destination)
    if old.is_dir():
        shutil.copytree(old, cache, ignore=shutil.ignore_patterns('*.sqlite', '*.sqlite-wal', '*.sqlite-shm'))
        database = old / 'fastf1_http_cache.sqlite'
        if database.exists():
            # SQLite backup reads a consistent snapshot without writing the original.
            with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as source_db:
                with sqlite3.connect(cache / database.name) as target_db:
                    source_db.backup(target_db)
    else:
        cache.mkdir(parents=True)
    import fastf1
    fastf1.Cache.enable_cache(str(cache))
    fastf1.Cache.offline_mode(fetch != '1')
    event = EventKey(int(season), int(round_number))
    raw = fastf1.get_session(event.season, event.round_number, session_id)
    raw.load(laps=True, telemetry=telemetry == '1', weather=False, messages=True)
    loaded = load_session(event, session_id, loader=lambda *args: raw,
                          fetched_at_utc=datetime.fromisoformat(fetched), allow_fetch=fetch == '1')
    metadata = dict(loaded.source.original_timestamps,
                    api_path=raw.api_path, fastf1_version=fastf1.__version__,
                    telemetry_requested=telemetry == '1', race_control_messages_requested=True)
    source = SourceRecord('fastf1', loaded.source.source_id, loaded.source.fetched_at_utc,
                          url='https://livetiming.formula1.com' + raw.api_path, original_timestamps=metadata)
    loaded = LoadedSession(event, session_id, loaded.entries, loaded.laps, source, loaded.status,
                           loaded.errors, loaded.quality_flags, loaded.session_start_utc, loaded.session_end_utc)
    with Path(output).open('wb') as stream:
        pickle.dump(loaded, stream)


if __name__ == '__main__':
    _worker()
