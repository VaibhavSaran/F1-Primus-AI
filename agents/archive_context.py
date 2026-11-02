"""Bounded read-only archive projections for local Q&A and season profiles.

Only the trusted application constructs this adapter. User input never supplies
SQL, paths, URLs or executable text. Publication AND first retrieval bound local
knowledge; old event dates do not imply the archive knew the facts then.
"""
from datetime import datetime, timezone
import re
from urllib.parse import urlsplit

import pandas as pd

from data.archive.event_context import classify_outcome
from data.profiles import build_profiles, _time


def text(value, limit=180):
    return re.sub(r'[<>\[\]`\r\n]', '', str(value or ''))[:limit]


def citation_url(value):
    """Display vetted public publisher links; never return local paths or secrets."""
    try:
        url = urlsplit(str(value or ''))
        host = url.hostname or ''
        if (url.scheme == 'https' and url.port in (None, 443) and not re.search(r'[\s<>"\x27]', str(value))
                and not url.username and not url.password and not url.query
                and not url.fragment and (host in {'fia.com', 'formula1.com', 'pirelli.com'}
                or host.endswith(('.fia.com', '.formula1.com', '.pirelli.com')))):
            return url.geturl()[:1200]
    except ValueError:
        pass
    return None


class ArchiveContext:
    def __init__(self, reader, context_store=None):
        self.reader, self.context_store = reader, context_store

    def _sources(self, connection, version, created, cutoff):
        rows = connection.execute('SELECT DISTINCT o.source_id,o.canonical_url,o.first_fetched_at_utc,'
            'o.original_published_at_utc,o.original_issued_at_utc,o.original_available_at_utc '
            'FROM source_objects o JOIN session_version_sources s USING(source_id) '
            'WHERE s.version_id=? ORDER BY o.source_id LIMIT 201', (version,)).fetchall()
        if not rows or len(rows) > 200 or _time(created) > cutoff:
            return None
        clocks = [_time(value) for row in rows for value in row[2:] if value is not None]
        if any(clock > cutoff for clock in clocks):
            return None
        known = max([_time(created)] + clocks)
        return dict(source_ids=[r[0] for r in rows],
                    source_urls=sorted({citation_url(r[1]) for r in rows if citation_url(r[1])})[:8],
                    known_at_utc=known.isoformat())

    def history(self, season, as_of_utc):
        if type(season) is not int or not 1950 <= season <= 2100:
            raise ValueError('invalid season')
        cutoff = _time(as_of_utc)
        sessions = [s for s in self.reader.available_sessions() if s['season'] == season and s['session_code'] == 'R']
        if len(sessions) > 40:
            raise ValueError('season session limit exceeded')
        history = []
        if not sessions:
            return history
        with self.reader._connect() as connection:
            for s in sessions:
                source = self._sources(connection, s['version_id'], s['created_at_utc'], cutoff)
                if source is None:
                    continue
                clock = s.get('observed_end_utc_ns')
                if clock is None or pd.isna(clock):
                    clock = s.get('scheduled_start_utc_ns')
                if clock is None or pd.isna(clock):
                    continue
                event_time = datetime.fromtimestamp(int(clock) / 1e9, timezone.utc)
                if event_time >= cutoff or _time(source['known_at_utc']) < event_time:
                    continue
                records = connection.execute('SELECT r.driver_id,e.team_id,d.display_name,e.team_name_at_event,'
                    'r.finishing_position,r.status_raw,r.classified_position_raw,v.circuit_id '
                    'FROM session_results r JOIN session_entries e USING(version_id,driver_id) '
                    'JOIN drivers d USING(driver_id) JOIN events v ON v.event_id=? '
                    'WHERE r.version_id=? ORDER BY r.driver_id LIMIT 101',
                    (s['session_id'].rsplit(':', 1)[0], s['version_id'])).fetchall()
                if len(records) > 100:
                    raise ValueError('season entry limit exceeded')
                for driver, team, name, team_name, position, raw, classified, circuit in records:
                    status = classify_outcome(raw, 'R', classified_position_raw=classified)
                    normalized = {'retired': 'DNF', 'dns': 'DNS', 'withdrawn': 'DNS',
                                  'disqualified': 'DSQ', 'finished': 'FINISHED', 'lapped': 'FINISHED'}.get(status)
                    if normalized is None:
                        # Unknown outcomes do not silently turn into classified finishes.
                        continue
                    history.append(dict(event_id=s['session_id'].rsplit(':', 1)[0], driver_id=driver,
                        team_id=team, display_name=text(name or driver), team_display_name=text(team_name or team),
                        event_time_utc=event_time.isoformat(), outcome_available_at_utc=source['known_at_utc'],
                        actual_position=position if position and position >= 1 else None, status=normalized,
                        circuit_id=circuit, source_ids=source['source_ids'], source_urls=source['source_urls']))
        return history

    def profiles(self, season, as_of_utc):
        history = self.history(season, as_of_utc)
        bundle = build_profiles(history, as_of_utc)
        bundle['season'] = season
        for kind, key, name in [('drivers', 'driver_id', 'display_name'), ('teams', 'team_id', 'team_display_name')]:
            for profile in bundle[kind]:
                records = [r for r in history if r[key] == profile[key]]
                profile.update(season=season, display_name=records[-1][name],
                    source_urls=sorted({u for r in records for u in r['source_urls']})[:8])
        return bundle

    def query(self, *, event_id, session_id, question, as_of_utc, driver_id=None, team_id=None):
        cutoff = _time(as_of_utc)
        if question not in {'results', 'laps', 'tyres', 'weather', 'context', 'profile'}:
            raise ValueError('unsupported archive question')
        if not session_id or session_id.rsplit(':', 1)[0] != event_id:
            return None
        if question == 'profile':
            bundle = self.profiles(int(event_id.split(':')[0]), cutoff)
            return dict(profiles=bundle, reference=bundle['snapshot_id'])
        sessions = [s for s in self.reader.available_sessions() if s['session_id'] == session_id]
        meta = sessions[0] if sessions else None
        if question == 'context':
            if team_id and not driver_id:
                return None
            if self.context_store is None:
                return None
            # Include event-wide curated statements and the selected publication only.
            records = self.context_store.read(event_id=event_id, as_of=cutoff.isoformat())
            records = [r for r in records if (r.get('session_id') in (None, session_id))
                       and (not r.get('version_id') or meta and r['version_id'] == meta['version_id'])
                       and (not driver_id or r.get('driver_id') == driver_id)]
            records.sort(key=lambda r: (r['evidence_kind'] == 'archive_result', r['record_id']))
            if not records:
                return None
            return dict(reference=session_id, records=[dict({k: r.get(k) for k in
                ('record_id', 'summary', 'publisher', 'url', 'status', 'raw_status', 'driver_id',
                 'evidence_kind', 'cause_qualification', 'known_at')}, url=citation_url(r.get('url')))
                 for r in records[:6]])
        if meta is None:
            return None
        with self.reader._connect() as connection:
            sources = self._sources(connection, meta['version_id'], meta['created_at_utc'], cutoff)
        if sources is None:
            return None
        tables = self.reader.tables(session_id)
        # A concurrent publication change cannot switch evidence under this query.
        if tables['meta']['version_id'] != meta['version_id']:
            return None
        entries = tables['entries'].to_dict('records')
        affiliations = {e['driver_id']: dict(team_id=e.get('team_id'),team_name=text(e.get('team_name_at_event'))) for e in entries}
        names = {e['driver_id']: text(e.get('display_name') or e['driver_id']) for e in entries}
        if driver_id and driver_id not in names:
            return None
        team_drivers = {e['driver_id'] for e in entries if e.get('team_id') == team_id} if team_id else set(names)
        if team_id and not team_drivers:
            return None
        if driver_id and driver_id not in team_drivers:
            return None
        payload = dict(reference=meta['version_id'], session_id=session_id,
                       event_name=text(meta['event_name']), **sources)
        if question == 'weather':
            weather = tables['weather']
            # Independently attached weather can have been retrieved after the
            # session publication, so its own provenance must pass the cutoff.
            with self.reader._connect() as connection:
                weather_sources = connection.execute('SELECT DISTINCT o.source_id,o.canonical_url,'
                    'o.first_fetched_at_utc,o.original_published_at_utc,o.original_available_at_utc '
                    'FROM weather_series s JOIN weather_observations w USING(series_id) '
                    'LEFT JOIN source_objects o ON o.source_id=coalesce(w.source_id,s.source_id,('
                    'SELECT max(v.source_id) FROM session_version_sources v JOIN source_objects x USING(source_id) '
                    "WHERE v.version_id=s.session_version_id AND x.endpoint_kind='weather_data' "
                    "AND s.provider='fastf1' AND s.kind='track_observation' "
                    'HAVING count(DISTINCT v.source_id)=1)) '
                    'WHERE s.session_version_id=? LIMIT 201', (meta['version_id'],)).fetchall()
            if (not weather_sources or len(weather_sources) > 200
                    or any(row[0] is None for row in weather_sources)):
                return None
            clocks = [_time(value) for row in weather_sources for value in row[2:] if value is not None]
            if any(clock > cutoff for clock in clocks):
                return None
            payload['known_at_utc'] = max([_time(payload['known_at_utc'])] + clocks).isoformat()
            payload['source_ids'] = sorted(set(payload['source_ids']) | {row[0] for row in weather_sources})
            payload['source_urls'] = sorted(set(payload['source_urls']) |
                {citation_url(row[1]) for row in weather_sources if citation_url(row[1])})[:8]
            fields = {'air_temperature_c': 'air temperature (C)', 'track_temperature_c': 'track temperature (C)',
                      'relative_humidity_pct': 'humidity (%)', 'wind_speed_mps': 'wind speed (m/s)',
                      'pressure_hpa': 'pressure (hPa)', 'rainfall_flag': 'rain indicator (0=no, 1=yes)'}
            payload['measurements'] = [{ 'label': label, 'minimum': float(weather[key].min()),
                'maximum': float(weather[key].max()), 'sample_count': int(weather[key].notna().sum())}
                for key, label in fields.items() if key in weather and weather[key].notna().any()]
            return payload if payload['measurements'] else None
        frame = tables['results'] if question == 'results' else tables['laps']
        if question == 'results' and not frame.empty:
            frame = frame[frame['original_publication_utc'].map(
                lambda value: pd.isna(value) or _time(value) <= cutoff)]
        if team_id:
            frame = frame[frame['driver_id'].isin(team_drivers)]
        if driver_id:
            frame = frame[frame['driver_id'] == driver_id]
        if frame.empty:
            return None
        if question == 'results':
            rows = frame.sort_values('finishing_position', na_position='last').head(24).to_dict('records')
            payload['rows'] = [dict(driver_id=r['driver_id'], display_name=names.get(r['driver_id'], r['driver_id']),
                position=None if pd.isna(r['finishing_position']) else r['finishing_position'],
                status=text(r['status_raw']), **affiliations.get(r['driver_id'], {}),
                **{k[:-3]+'_s':float(r[k])/1e9 for k in ('q1_ns','q2_ns','q3_ns') if k in r and pd.notna(r[k]) and r[k]>0}) for r in rows]
        else:
            rows = []
            for driver, group in frame.groupby('driver_id', sort=True):
                timed = group['lap_time_s'].dropna()
                timed = timed[timed > 0]
                rows.append(dict(driver_id=driver, display_name=names.get(driver, driver), lap_count=len(group),
                    timed_lap_count=len(timed), best_lap_s=float(timed.min()) if len(timed) else None,
                    mean_lap_s=float(timed.mean()) if len(timed) else None,
                    compounds={text(k): int(v) for k, v in group['compound_normalized'].dropna().value_counts().items()}))
            rows.sort(key=lambda r: (r['best_lap_s'] is None, r['best_lap_s'] or float('inf'), r['driver_id']))
            payload['rows'] = rows[:24]
        return payload

    def resolve_driver(self, message, session_id):
        """Prefer selected roster; absent drivers can have sourced withdrawal facts."""
        try:
            entries = self.reader.tables(session_id)['entries'].to_dict('records')
        except ValueError:
            entries = []
        words = set(re.findall(r'\w+', message.casefold()))
        matches = []
        for row in entries:
            abbreviation = row.get('abbreviation')
            names = [row['driver_id'].split(':')[-1], row.get('display_name')]
            if isinstance(abbreviation, str) and re.search(r'\b' + re.escape(abbreviation) + r'\b', message):
                names.append(abbreviation)
            if any({w for w in re.findall(r'\w+', str(n).casefold()) if len(w) >= 3} & words for n in names if n):
                matches.append(row['driver_id'])
        if matches:
            return matches[0] if len(matches) == 1 else None
        if self.reader.db_path.is_file():
            with self.reader._connect() as connection:
                global_names = connection.execute("SELECT driver_id,display_name FROM drivers "
                    "WHERE instr(driver_id,':number:')=0 ORDER BY driver_id LIMIT 1001").fetchall()
            if len(global_names) > 1000:
                return None
            for identifier, name in global_names:
                tokens = {w for w in re.findall(r'\w+', str(name or identifier.split(':')[-1]).casefold()) if len(w) >= 3}
                if tokens & words:
                    matches.append(identifier)
        return matches[0] if len(matches) == 1 else None

    def scope_matches(self, message, event_id, session_id):
        years = re.findall(r'\b(?:19|20)\d{2}\b', message)
        if years and any(y != event_id.split(':')[0] for y in years):
            return False
        code = session_id.rsplit(':', 1)[-1]
        for pattern, codes in [(r'\b(?:qualifying|qualification)\b', {'Q', 'SQ'}),
                               (r'\bsprint\b', {'S', 'SQ'}), (r'\bpractice\b', {'FP1', 'FP2', 'FP3'}),
                               (r'\brace\b', {'R'})]:
            if re.search(pattern, message) and code not in codes:
                return False
        explicit = re.findall(r'\bfp[123]\b', message)
        if explicit and any(value.upper() != code for value in explicit):
            return False
        # Explicit other Grand Prix names must not silently answer the selector's race.
        named = re.search(r'\b(?:the )?([\w -]+?) (?:grand prix|gp)\b', message)
        if named:
            phrase = named.group(1).casefold()
            if phrase.split()[-1] in {'the', 'this', 'that', 'selected', 'a', 'f1', 'in'}:
                return True
            selected_name = None
            if self.reader.db_path.is_file():
                with self.reader._connect() as connection:
                    selected = connection.execute('SELECT event_name FROM events WHERE event_id=?', (event_id,)).fetchone()
                selected_name = selected[0] if selected else None
            if selected_name and not any(w in phrase.split() for w in str(selected_name).casefold().split() if w not in {'grand', 'prix'}):
                return False
        return True

    def resolve_team(self, message, session_id):
        try:
            entries = self.reader.tables(session_id)['entries'].to_dict('records')
        except ValueError:
            return None
        words = set(re.findall(r'\w+', message.casefold()))
        matches = set()
        for row in entries:
            if not row.get('team_id') or pd.isna(row['team_id']):
                continue
            tokens = set(re.findall(r'\w+', str(row.get('team_name_at_event') or row['team_id'].split(':')[-1]).casefold()))
            tokens -= {'team', 'f1', 'formula', 'one', 'racing', 'scuderia'}
            if tokens & words:
                matches.add(row['team_id'])
        return next(iter(matches)) if len(matches) == 1 else None
