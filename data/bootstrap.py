"""Explicit operator download of real calendars/weekends; never runs on UI render.

Run with primusEnv: python -m data.bootstrap --fetch --seasons 2024 2025 2026
Historical forecast snapshots are retrospective reconstructions, never backtests
with verified original publication times. Final race results are not their inputs.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from statistics import median, pstdev

from data.audit import persist_session
from data.contracts import DataSnapshot, DriverSessionRecord, EventKey, SourceRecord
from data.fastf1_source import LoadedSession, load_session, _property, _seconds, _number
from data.contracts import RunStatus
from data.sessions import resolve_event
from data.snapshots import save_snapshot

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / '.local'


def write_calendar(seasons, *, loader=None, path=None):
    if loader is None:
        import fastf1
        loader = fastf1.get_event_schedule
    events = []
    for season in seasons:
        schedule = loader(season, include_testing=False)
        for row in schedule.to_dict('records'):
            round_number = int(row['RoundNumber'])
            if round_number <= 0:
                continue
            info = resolve_event(EventKey(season, round_number), loader=lambda *args, row=row: row)
            race = next((s for s in info.sessions if s.session_id == 'R'), None)
            local_start = next((row.get(f'Session{i}Date') for i in range(1, 6)
                                if row.get(f'Session{i}') == 'Race'), None)
            if local_start is None or str(local_start) in {'NaT', 'nan', 'None'}:
                local_start = row.get('EventDate')
            date = (str(local_start)[:10] if local_start is not None and str(local_start) not in {'NaT','nan','None'}
                    else race.starts_at_utc.date().isoformat() if race else None)
            events.append(dict(event_id=info.event.event_id, season=season, round_number=round_number,
                               event_name=info.event_name, location=info.location,
                               race_date=date,
                               sessions=[dict(session_id=s.session_id, name=s.name,
                                              starts_at_utc=s.starts_at_utc.isoformat()) for s in info.sessions]))
    destination = Path(path or LOCAL / 'catalog/calendar.json')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(dict(schema_version='calendar-v1', fetched_at_utc=datetime.now(timezone.utc).isoformat(),
                                          provider='FastF1 official calendar', events=events), indent=2), encoding='utf-8')
    return events


def normalized_positions(raw, entries, *, interval_s=2):
    """Retain actual coordinate samples at most once per driver per interval."""
    positions = []
    by_number = {entry['driver_number']: entry['driver_id'] for entry in entries}
    for number, table in (_property(raw, 'pos_data', {}) or {}).items():
        driver = by_number.get(str(number))
        if driver is None:
            continue
        seen = set()
        for row in table.to_dict('records'):
            clock, x, y = _seconds(row.get('SessionTime')), _number(row.get('X')), _number(row.get('Y'))
            if clock is None or clock < 0 or x is None or y is None or row.get('Status') != 'OnTrack':
                continue
            bucket = int(clock // interval_s)
            if bucket in seen:
                continue
            seen.add(bucket)
            positions.append(dict(driver_id=driver, session_time_s=clock, x=x, y=y))
    return sorted(positions, key=lambda item: (item['session_time_s'], item['driver_id']))


def retrospective_snapshot(info, sessions):
    """Completed earlier sessions only; publication limitations remain explicit."""
    qualifying = next((s for s in sessions if s.session_id == 'Q' and s.entries), None)
    race = next((s for s in info.sessions if s.session_id == 'R'), None)
    if qualifying is None or race is None:
        raise ValueError('qualifying and scheduled race are required')
    cutoff = race.starts_at_utc - timedelta(minutes=5)
    records, ends = [], {}
    earlier = [s for s in sessions if s.session_id != 'R' and s.entries]
    for session in earlier:
        timestamps = [lap['lap_end_utc'] for lap in session.laps if lap.get('lap_end_utc') is not None]
        end = session.session_end_utc or (max(timestamps) if timestamps else None)
        if end is None and session.source.provider == 'jolpica':
            # Archived final classifications prove order, not publication time.
            # Retrospective only: assume finalized earlier sessions were known at cutoff.
            end = cutoff
        if end is None or end > cutoff:
            continue
        ends[session.session_id] = end
        medians = {}
        for entry in session.entries:
            laps = [lap for lap in session.laps if lap['driver_id'] == entry['driver_id']
                    and lap.get('lap_time_s') and lap.get('provider_is_accurate') is True
                    and not lap.get('deleted') and not lap.get('pit_in') and not lap.get('pit_out')
                    and lap.get('track_status') == '1' and lap['lap_number'] > 1]
            if laps:
                medians[entry['driver_id']] = median(lap['lap_time_s'] for lap in laps)
        center = median(medians.values()) if medians else None
        for entry in session.entries:
            driver = entry['driver_id']
            values = {}
            if session.session_id == 'Q':
                # Qualifying classification is a proxy; later grid penalties are unknown.
                values = dict(qualifying_position=entry['position'], grid_position=entry['position'])
            elif session.session_id.startswith('FP'):
                laps = [lap for lap in session.laps if lap['driver_id'] == driver and lap.get('lap_time_s')
                        and lap.get('provider_is_accurate') is True and not lap.get('deleted')
                        and not lap.get('pit_in') and not lap.get('pit_out') and lap.get('track_status') == '1'
                        and lap['lap_number'] > 1]
                if driver in medians:
                    values = dict(practice_pace_delta_s=medians[driver]-center,
                                  practice_variability_s=pstdev(lap['lap_time_s'] for lap in laps),
                                  practice_laps=len(laps))
            elif session.session_id == 'S':
                values = dict(sprint_position=entry['position'])
            records.append(DriverSessionRecord(info.event, session.session_id, driver, end, values, session.source))
    if 'Q' not in ends:
        raise ValueError('qualifying completion evidence is required')
    schedule = replace(info, sessions=tuple(replace(s, ends_at_utc=ends.get(s.session_id)) for s in info.sessions))
    entries = [{key: row.get(key) for key in ('driver_id', 'driver_number', 'abbreviation', 'team_id')}
               for row in qualifying.entries]
    flags = ['historical_reconstruction', 'historical_publication_times_unverified',
             'qualifying_order_used_as_grid_proxy', 'practice_lap_quality_not_certified',
             'weather_forecast_and_pit_loss_unavailable']
    if any(s.source.provider == 'jolpica' for s in earlier):
        flags.append('classification_availability_assumed_at_cutoff_not_observed')
    return DataSnapshot(f'historical-{info.event.season}-{info.event.round_number}-postq', info.event, cutoff,
                        dict(mode='post_qualifying', target_session_id='R', historical_reconstruction=True,
                             entries=entries, entries_source=qualifying.source, driver_sessions=records,
                             schedule=schedule, history=[],
                             observed_session_ends={s.session_id: ends[s.session_id] for s in earlier
                                                    if s.source.provider != 'jolpica' and s.session_id in ends},
                             reconstruction_session_ends=ends,
                             completion_assumptions={s.session_id: 'archived_final_classification_assumed_known_at_cutoff'
                                                     for s in earlier if s.source.provider == 'jolpica'}),
                        [s.source for s in earlier], flags)


def jolpica_classification(event, session_id):
    """Public official classification mirror, without invented session lap clocks."""
    import requests
    endpoint, key = ('qualifying', 'QualifyingResults') if session_id == 'Q' else ('sprint', 'SprintResults')
    url = f'https://api.jolpi.ca/ergast/f1/{event.season}/{event.round_number}/{endpoint}.json?limit=100'
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    races = response.json()['MRData']['RaceTable']['Races']
    rows = races[0].get(key, []) if races else []
    entries = []
    for row in rows:
        driver, team = row['Driver'], row['Constructor']
        entries.append(dict(driver_id=driver['driverId'], driver_number=row.get('number') or driver.get('permanentNumber'),
                            abbreviation=driver.get('code'), team_id=team['constructorId'],
                            position=int(row['position']), classified_position=row['position'],
                            status=row.get('status'), grid_position=None))
    source = SourceRecord('jolpica', f'jolpica:{event.event_id}:{session_id}', datetime.now(timezone.utc), url=url,
                          original_timestamps={'historical_publication': 'unverified', 'completion': 'unknown'})
    return LoadedSession(event, session_id, tuple(entries), (), source,
                         RunStatus.DEGRADED if entries else RunStatus.INSUFFICIENT_DATA,
                         quality_flags=('historical_availability_unverified', 'hindsight_reconstruction', 'lap_timing_unavailable'))


def download_weekend(season, round_number, *, telemetry=True):
    import fastf1
    event = EventKey(season, round_number)
    info = resolve_event(event)
    sessions, evidence = [], []
    for scheduled in info.sessions:
        raw = fastf1.get_session(season, round_number, scheduled.session_id)
        try:
            raw.load(laps=True, telemetry=telemetry and scheduled.session_id == 'R', weather=True, messages=True)
            loaded = load_session(event, scheduled.session_id, loader=lambda *args: raw)
        except Exception:
            loaded = load_session(event, scheduled.session_id, cache_path=ROOT / '.fastf1_cache')
        if not loaded.laps:
            cached = load_session(event, scheduled.session_id, cache_path=ROOT / '.fastf1_cache')
            if cached.laps:
                loaded = cached
        if not loaded.laps and scheduled.session_id in {'Q', 'S'}:
            loaded = jolpica_classification(event, scheduled.session_id)
        positions = normalized_positions(raw, loaded.entries) if scheduled.session_id == 'R' and telemetry else []
        path = persist_session(loaded, directory=LOCAL / 'sessions', positions=positions) if loaded.laps or (
            loaded.entries and scheduled.session_id in {'Q', 'S'}) else None
        sessions.append(loaded)
        item = dict(event_id=event.event_id, session_id=scheduled.session_id, entries=len(loaded.entries),
                    laps=len(loaded.laps), positions=len(positions), path=str(path) if path else None,
                    availability='available' if path else 'timing_archive_unavailable')
        evidence.append(item)
        print(json.dumps(item), flush=True)
    snapshot = retrospective_snapshot(info, sessions)
    path = save_snapshot(snapshot, LOCAL / 'snapshots')
    return dict(event_id=event.event_id, sessions=evidence, snapshot=str(path))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch', action='store_true', required=True, help='Explicit permission for free FastF1 HTTP')
    parser.add_argument('--seasons', type=int, nargs='+', default=[2024, 2025, 2026])
    parser.add_argument('--weekends', nargs='*', default=['2025:1'], help='season:round, default 2025:1')
    parser.add_argument('--no-telemetry', action='store_true')
    args = parser.parse_args(argv)
    import fastf1
    cache = LOCAL / 'fastf1-cache'
    cache.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache))
    events = write_calendar(args.seasons)
    print(json.dumps(dict(calendar_events=len(events))), flush=True)
    results = []
    for weekend in args.weekends:
        season, round_number = map(int, weekend.split(':'))
        results.append(download_weekend(season, round_number, telemetry=not args.no_telemetry))
    path = LOCAL / 'catalog/bootstrap-evidence.json'
    path.write_text(json.dumps(dict(calendar_events=len(events), weekends=results), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
