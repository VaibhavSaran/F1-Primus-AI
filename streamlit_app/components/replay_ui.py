"""Native committed snapshots with an optional bounded browser clock."""
from pathlib import Path
import json

import streamlit as st
import streamlit.components.v1 as components

from data.replay import load_replay, validate_commit
from streamlit_app import services
from streamlit_app.components.tables import display_table


def _clock_label(clock_ms):
    seconds = int(clock_ms // 1000)
    return f'{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}'


def _standings_table(timing_basis, drivers, labels):
    """Display only the clock projection's recorded timing, including absolute ranks."""
    official = timing_basis == 'timing_stream'
    checkpoint = timing_basis == 'lap_end_position'
    if official:
        title = 'Race standings'
        caption = ('Recorded timing snapshots carry the last reported values; missing values and '
                   'snapshots older than two minutes show —. Interval refers to the car ahead in the full field.')
    elif checkpoint:
        title = 'Lap-end classification checkpoints'
        caption = ('Position is the latest recorded lap-end checkpoint, not live race order. '
                   'Gaps and intervals are unavailable between checkpoints.')
    else:
        title = 'Session lap completion'
        caption = ('Ordered by positively timed lap records and their clock, not race order. '
                   'Missing durations do not certify completed laps; running positions and gaps are unavailable.')
    rows = []
    for order, driver in enumerate(drivers, 1):
        row = {}
        if official or checkpoint:
            row['Position'] = driver.get('position') if driver.get('position') is not None else '—'
        else:
            row['Order'] = order
        row['Driver'] = labels.get(driver['driver_id'], driver.get('abbreviation') or 'Unknown driver')
        if official:
            gap = driver.get('gap_label')
            if gap == 'Leader' and not (driver.get('position') == 1 and driver.get('position_fresh')):
                gap = None
            interval = driver.get('interval_label')
            row.update({'Gap to leader': gap or '—', 'Interval': interval if interval and interval != 'Leader' else '—'})
        elif not checkpoint:
            row.update({'Timed laps observed': driver['completed_laps'],
                        'Last timed lap': driver['last_lap_number']})
        row.update({'Last lap (s)': driver['last_lap_time_s'], 'Tyres': driver['compound'] or 'Unknown'})
        rows.append(row)
    return title, caption, rows


def _render_weather(weather):
    def value(key, unit):
        number = weather.get(key)
        return 'Unknown' if number is None else f'{number:.1f}{unit}'
    rain = weather.get('rainfall_flag')
    values = [('Air', value('air_temperature_c', ' °C')), ('Track', value('track_temperature_c', ' °C')),
              ('Humidity', value('relative_humidity_pct', '%')), ('Pressure', value('pressure_hpa', ' hPa')),
              ('Wind', value('wind_speed_mps', ' m/s')),
              ('Rain', 'Unknown' if rain is None else 'Yes' if rain else 'No')]
    for column, (label, reading) in zip(st.columns(6), values):
        column.metric(label, reading)
        if label == 'Wind':
            column.caption('Direction ' + value('wind_direction_deg', '°'))
    st.caption(f'Weather observed {_clock_label(weather["clock_ms"])}; fields without readings stay unknown.'
               if weather else 'No fresh weather observation (older than 2 min or unavailable).')


@st.cache_data(show_spinner=False, max_entries=12)
def _cached_source(path, modified_ns, size):
    return load_replay(path)


def _sources(context):
    matching, rejected, total_bytes = [], 0, 0
    directory = services.LOCAL / 'sessions'
    for index, path in enumerate(sorted(directory.glob('session-*.json'))):
        if index >= 64:
            rejected += 1
            break
        try:
            info = path.stat()
            total_bytes += info.st_size
            if path.is_symlink() or total_bytes > 50_000_000:
                rejected += 1
                continue
            source = _cached_source(str(path.resolve()), info.st_mtime_ns, info.st_size)
            if (source.event.season, source.event.round_number, source.session_id) == (
                    context['season'], context['round_number'], context['session']):
                matching.append(source)
        except (OSError, ValueError, TypeError):
            rejected += 1
    matching.sort(key=lambda session: (len(session.laps), len(session.positions)), reverse=True)
    return matching, rejected


def _commit(session, clock_ms, driver_ids, sequence=0):
    previous = st.session_state.get('replay_commit', {})
    st.session_state['replay_commit'] = dict(session_id=session.identity, frame_version=session.frame_version,
                                            clock_ms=clock_ms, driver_ids=driver_ids, sequence=sequence,
                                            revision=previous.get('revision', 0) + 1)
    st.session_state['_replay_pending_seconds'] = clock_ms / 1000


def render_replay(context):
    labels = {}
    if context.get('archive_session_id'):
        from streamlit_app import archive_services as archive
        try:
            tables = archive.tables(context)
            entries = tables['entries']
            observed = set(tables['laps'].driver_id)
            completeness = json.loads(tables['meta'].get('completeness_json') or '{}')
            observed.update(driver for driver, detail in completeness.get('positions', {}).get('drivers', {}).items()
                            if detail.get('valid_timestamps', 0) > 0)
            entries = entries[entries.driver_id.isin(observed)]
            labels = dict(zip(entries.driver_id, entries.display_name))
            if not labels:
                st.info('No driver has recorded laps or positions for this session yet.')
                return
            if 'pilot_replay' not in st.session_state:
                st.session_state['pilot_replay'] = list(labels)[:2]
            if st.button('Select all drivers'):
                st.session_state['pilot_replay'] = list(labels)
            drivers = st.multiselect('Replay drivers', list(labels),
                                     format_func=labels.get, max_selections=len(labels), key='pilot_replay')
            if not drivers:
                st.info('Select at least one driver to replay the session.')
                return
            with st.spinner('Preparing recorded session positions…'):
                sources = [archive.replay(str(archive.root().resolve()), context['archive_session_id'],
                                           context['archive_version_id'], tuple(drivers))]
            rejected = 0
        except (OSError, ValueError) as error:
            st.error(f'The recorded replay could not be loaded: {error}')
            return
        st.caption('Select any driver or the full grid. Markers follow recorded positions and disappear when observations are more than five seconds old.')
    else:
        sources, rejected = _sources(context)
    if rejected:
        st.warning('Some recorded sessions could not be opened and were skipped.')
    if not sources:
        st.info('No recorded replay is available for this event and session.')
        return
    source_ids = [source.frame_version for source in sources]
    if st.session_state.get('replay_source') not in source_ids:
        st.session_state.pop('replay_source', None)
    if len(sources) == 1:
        selected = source_ids[0]
    else:
        selected = st.selectbox('Recorded session', source_ids, key='replay_source',
                                format_func=lambda value: next(f'Saved {item.source["fetched_at_utc"][:10]}'
                                    for item in sources if item.frame_version == value))
    session = sources[source_ids.index(selected)]
    labels = {entry['driver_id']: labels.get(entry['driver_id']) or entry.get('display_name') or entry['abbreviation'] or 'Unknown driver' for entry in session.entries}
    current = st.session_state.get('replay_commit', {})
    if current.get('session_id') != session.identity or current.get('frame_version') != session.frame_version:
        timed = session.timed_laps
        initial = min(session.duration_ms, min((lap['clock_ms'] for lap in timed), default=0) + 120_000)
        if context.get('archive_session_id') and session.positions:
            target = min((lap['clock_ms'] for lap in timed), default=session.positions[0]['clock_ms'])
            initial = next((p['clock_ms'] for p in session.positions if p['clock_ms'] >= target), session.positions[0]['clock_ms'])
        _commit(session, initial, [entry['driver_id'] for entry in session.entries])
        for key in ('replay_native_seconds', 'replay_native_driver', 'replay_native_lap'):
            st.session_state.pop(key, None)
    current = st.session_state['replay_commit']
    if '_replay_pending_seconds' in st.session_state:
        st.session_state['replay_native_seconds'] = st.session_state.pop('_replay_pending_seconds')
    st.caption('Causal shows observations up to the clock; Hindsight includes all laps; reduced motion uses jump controls. Pause or jump to update analysis.')
    native_snapshot = session.frame_at(current['clock_ms'])
    if context.get('reduced_motion', st.session_state.get('reduced_motion', True)):
        _render_weather(native_snapshot.frame.weather)
        last_marker = native_snapshot.markers[-1] if native_snapshot.markers else None
        st.info(f'{_clock_label(last_marker["clock_ms"])} · {last_marker["message"]}' if last_marker else 'No observed event by this clock.')
    if session.track_unavailable_reason:
        st.info(session.track_unavailable_reason)
    driver_ids = [entry['driver_id'] for entry in session.entries]
    with st.expander('Jump to time or lap'):
        st.caption('Time is elapsed session time. A lap jump moves to its recorded completion.')
        driver = st.selectbox('Driver for lap jump', driver_ids, format_func=labels.get, key='replay_native_driver')
        with st.form('replay_native_seek'):
            st.number_input('Session clock (seconds)', min_value=0.0, max_value=session.duration_ms / 1000,
                            step=1.0, key='replay_native_seconds')
            lap_options = [lap for lap in session.timed_laps if lap['driver_id'] == driver]
            lap_index = st.selectbox('Completed lap', range(len(lap_options)),
                                     format_func=lambda index: f'Lap {lap_options[index]["lap_number"]} · {_clock_label(lap_options[index]["clock_ms"])}',
                                     key='replay_native_lap') if lap_options else None
            time_submit = st.form_submit_button('Jump to time')
            lap_submit = st.form_submit_button('Jump to lap', disabled=not lap_options)
    if time_submit:
        _commit(session, round(st.session_state['replay_native_seconds'] * 1000), current['driver_ids'], current['sequence'])
        st.rerun()
    if lap_submit:
        _commit(session, lap_options[lap_index]['clock_ms'], current['driver_ids'], current['sequence'])
        st.rerun()
    current = st.session_state['replay_commit']
    reduced_motion = context.get('reduced_motion', st.session_state.get('reduced_motion', True))
    if not reduced_motion:
        try:
            payload = session.browser_payload()
            payload['driver_labels'] = labels
        except ValueError:
            st.warning('This source exceeds the browser payload bound. Use the native clock controls and committed tables.')
        else:
            replay_component = components.declare_component('historical_replay', path=str(Path(__file__).with_name('replay_component').resolve()))
            event = replay_component(payload=payload, committed_clock_ms=current['clock_ms'],
                                     commit_revision=current['revision'], last_sequence=current['sequence'],
                                     selected_driver_ids=current['driver_ids'], key='replay_browser_' + session.frame_version,
                                     default=None)
            if event is not None:
                try:
                    committed = validate_commit(event, session, last_sequence=current['sequence'])
                except ValueError:
                    # Streamlit retains the last return value on ordinary reruns.
                    if not isinstance(event, dict) or event.get('sequence') != current['sequence']:
                        st.warning('An invalid or stale browser replay update was ignored.')
                else:
                    _commit(session, committed['clock_ms'], committed['driver_ids'], committed['sequence'])
                    st.rerun()
            st.caption('Sampled coordinates show recorded locations; they cannot certify overtakes.')
    current = st.session_state['replay_commit']
    hindsight = st.session_state.get('replay_mode') == 'Hindsight'
    snapshot = session.frame_at(current['clock_ms'], hindsight=hindsight)
    selected_laps = [lap for lap in snapshot.laps if lap['driver_id'] in current['driver_ids']]
    selected_stints = [row for row in snapshot.stints if row['driver_id'] in current['driver_ids']]
    st.caption(f'Analysis time: {_clock_label(current["clock_ms"])} · {"All recorded laps" if hindsight else "Observed laps up to this time"}')
    title, caption, tower = _standings_table(session.timing_basis,
        [dict(row, driver_id=driver) for driver, row in snapshot.frame.drivers.items()], labels)
    st.subheader(title)
    st.caption(caption)
    import pandas as pd
    tower_frame = pd.DataFrame(tower)
    if 'Position' in tower_frame:
        # Keep ranks and the unknown marker in one Arrow-compatible display type.
        tower_frame['Position'] = tower_frame.Position.astype('string')
    display_table(tower_frame)
    view = st.segmented_control('Analysis', ['Pace', 'Tyres', 'Sectors', 'Telemetry'],
                                default='Pace', key='replay_analysis') or 'Pace'
    if view == 'Pace':
        if selected_laps:
            import pandas as pd
            pace = pd.DataFrame([{'Driver': labels.get(lap['driver_id'], 'Unknown driver'), 'Lap': lap['lap_number'], 'Lap time (s)': lap['lap_time_s']}
                                 for lap in selected_laps if lap['lap_time_s'] is not None])
            if not pace.empty:
                st.line_chart(pace, x='Lap', y='Lap time (s)', color='Driver')
            display_table([{'Driver': labels.get(lap['driver_id']), 'Lap': lap['lap_number'],
                           'Lap time (s)': lap['lap_time_s'], 'Tyre': lap['compound'],
                           'Tyre age (laps)': lap['tyre_age_laps']} for lap in selected_laps], hide_index=True, width='stretch')
            st.caption('Descriptive observed lap times include flagged, pit and unverified-deletion laps. They do not certify clean-lap pace or tyre degradation.')
        else:
            st.info('No lap has completed by this clock.')
    elif view == 'Tyres':
        if selected_stints:
            display_table([{'Driver': labels.get(row['driver_id']), 'Stint': row['stint'],
                           'Tyre': row['compound'], 'First lap': row['first_lap'],
                           'Last lap': row['last_lap'], 'Observed laps': row['observed_laps']}
                          for row in selected_stints], hide_index=True, width='stretch')
        else:
            st.info('No observed stint/compound evidence is available in this scope.')
    elif context.get('archive_session_id') and view in ('Sectors', 'Telemetry'):
        visible = tables['laps']
        visible = visible[visible.driver_id.isin(current['driver_ids'])]
        if not hindsight:
            visible = visible[visible.lap_end_session_s.round(3) <= snapshot.frame.session_time_s]
        if view == 'Sectors':
            sectors = visible[['driver_id', 'lap_number', 'sector1_s', 'sector2_s', 'sector3_s']].copy()
            sectors['driver_id'] = sectors.driver_id.map(labels)
            sectors = sectors.rename(columns={'driver_id': 'Driver', 'lap_number': 'Lap',
                'sector1_s': 'Sector 1 (s)', 'sector2_s': 'Sector 2 (s)', 'sector3_s': 'Sector 3 (s)'})
            if sectors.empty:
                st.info('No completed sector times are available at this playback time.')
            else:
                display_table(sectors)
            st.caption('Recorded sectors for completed laps in the committed scope. Missing splits remain unknown.')
        elif visible.empty:
            st.info('No completed lap is available for the committed driver selection yet.')
        elif st.button('Load latest completed lap telemetry'):
            lap = visible.sort_values('lap_end_session_s').iloc[-1]
            car = archive.dense(str(archive.root().resolve()), context['archive_session_id'],
                                context['archive_version_id'], 'car', (lap.driver_id,))
            end = lap.lap_end_session_s if hindsight else min(lap.lap_end_session_s, snapshot.frame.session_time_s)
            from streamlit_app.archive_views import (
                segmented_car, telemetry_window, car_clock_unavailable, UNALIGNED_CAR_MESSAGE)
            unaligned = car_clock_unavailable(car, tables['meta'])
            start, _ = telemetry_window(lap)
            car = segmented_car(car[car.session_time_s.between(start, end)])
            if not car.empty and 'speed_kmh' in car:
                import plotly.graph_objects as go
                chart = go.Figure(go.Scatter(x=car.session_time_s, y=car.speed_kmh,
                    mode='lines', connectgaps=False, name='Recorded speed'))
                chart.update_layout(xaxis_title='Session time (s)', yaxis_title='Speed (km/h)')
                st.plotly_chart(chart, width='stretch', key='replay_lap_speed')
                st.caption(f'Recorded speed (km/h) · {labels.get(lap.driver_id)} · lap {lap.lap_number} · last committed scope.')
            else:
                st.info(UNALIGNED_CAR_MESSAGE if unaligned else 'No recorded car samples are available for this completed lap.')
    else:
        st.info(f'{view} samples are not present in this normalized source.')
    st.write('Observed pit / flag markers')
    if snapshot.markers:
        display_table([{'Session time': _clock_label(row['clock_ms']), 'Event': row['message']}
                      for row in snapshot.markers], hide_index=True, width='stretch')
        marker = st.selectbox('Jump to an observed marker', range(len(snapshot.markers)),
                              format_func=lambda index: f'{_clock_label(snapshot.markers[index]["clock_ms"])} · {snapshot.markers[index]["message"]}')
        st.caption('Jump to an event already observed by the playback clock.')
        if st.button('Jump to event'):
            _commit(session, snapshot.markers[marker]['clock_ms'], current['driver_ids'], current['sequence'])
            st.rerun()
    else:
        st.info('No pit / flag marker has been observed by this clock. Official race-control messages may be absent from the source.')
    with st.expander('About this replay'):
        st.write(f'Recorded session · Saved {session.source["fetched_at_utc"][:10]}.')
        st.write(f'{len(session.entries)} drivers · {len(session.timed_laps):,} timed laps.')
        st.caption('Track paths may include the pit lane. Missing observations stay unknown; a stopped coordinate does not prove retirement.')
