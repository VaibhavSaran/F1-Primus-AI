"""Readable analysis of published sessions; no forecast calls or provider fetches."""
import sqlite3

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_app import archive_services as archive
from streamlit_app.components.tables import display_table


LABELS = {
    'display_name': 'Driver', 'abbreviation': 'Code', 'team_name_at_event': 'Team',
    'driver_number': 'Number', 'finishing_position': 'Position', 'grid_position': 'Grid',
    'classified_position_raw': 'Classification', 'points': 'Points', 'laps_completed': 'Laps',
    'status_raw': 'Status', 'lap_number': 'Lap', 'lap_time_s': 'Lap time (s)',
    'sector1_s': 'Sector 1 (s)', 'sector2_s': 'Sector 2 (s)', 'sector3_s': 'Sector 3 (s)',
    'compound_raw': 'Tyre compound', 'tyre_age_laps': 'Tyre age (laps)',
    'stint_number': 'Stint', 'first_observed_lap': 'First lap', 'last_observed_lap': 'Last lap',
    'observed_lap_count': 'Recorded laps', 'start_tyre_age_laps': 'Starting tyre age (laps)',
    'end_tyre_age_laps': 'Final tyre age (laps)', 'missing_lap_count': 'Missing laps',
    'entry_lap': 'Entry lap', 'exit_lap': 'Exit lap', 'duration_s': 'Entry to exit (s)',
    'pit_in_session_s': 'Entry session time (s)', 'pit_out_session_s': 'Exit session time (s)',
    'air_temperature_c': 'Air temperature (°C)', 'track_temperature_c': 'Track temperature (°C)',
    'relative_humidity_pct': 'Humidity (%)', 'wind_speed_mps': 'Wind speed (m/s)',
    'wind_direction_deg': 'Wind direction (°)', 'pressure_hpa': 'Pressure (hPa)',
    'rainfall_flag': 'Rain observed', 'speed_kmh': 'Speed (km/h)',
    'throttle_pct_raw': 'Throttle (%)', 'rpm': 'Engine speed (rpm)', 'gear': 'Gear',
    'brake_applied': 'Brake applied', 'drs_code': 'DRS status code',
    'session_time_s': 'Session time (s)', 'message_raw': 'Track condition',
    'message_text': 'Message', 'category_raw': 'Category', 'flag_raw': 'Flag',
    'scope_raw': 'Scope', 'sector': 'Sector', 'driver_number_raw': 'Car number',
}
TYRES = {'SOFT': '#ef5350', 'MEDIUM': '#ffcf40', 'HARD': '#d4d9e2',
         'INTERMEDIATE': '#4caf80', 'WET': '#4e9eff', 'UNKNOWN': '#8793a5'}
UNALIGNED_CAR_MESSAGE = ('Car telemetry is saved, but it cannot be aligned to laps because '
                         'the session timing reference is unavailable.')


def car_clock_unavailable(car, meta):
    return (not car.empty and pd.isna(meta.get('t0_utc_ns'))
            and not car.session_time_s.notna().any())


def load(context):
    try:
        data = archive.tables(context)
        data = {**data, 'entries': participant_entries(data)}
        data['results'] = data['results'][data['results'].driver_id.isin(data['entries'].driver_id)].copy()
        return data
    except (OSError, ValueError, sqlite3.Error) as error:
        st.error(f'The saved session could not be read: {error}')
        return None


def participant_entries(data):
    results = data['results']
    meaningful = pd.Series(False, index=results.index)
    for key in ['finishing_position', 'q1_ns', 'q2_ns', 'q3_ns', 'reported_time_ns']:
        if key in results:
            meaningful |= results[key].gt(0).fillna(False)
    if 'laps_completed' in results:
        meaningful |= results.laps_completed.notna()
    for key in ['classified_position_raw', 'status_raw', 'normalized_outcome_status']:
        if key in results:
            meaningful |= results[key].astype('string').str.strip().replace('', pd.NA).notna()
    # Empty provider result rows can describe unused placeholders with real-looking names.
    known = set(data['laps'].driver_id) | set(results.loc[meaningful, 'driver_id'])
    return data['entries'][data['entries'].driver_id.isin(known)].copy()


def named(frame, entries):
    return frame.merge(entries[['driver_id', 'abbreviation', 'display_name', 'team_name_at_event']],
                       on='driver_id', how='left')


def pit_visit_label(row):
    if row.unmatched_exit:
        return 'Initial pit departure' if pd.notna(row.exit_lap) and row.exit_lap == 1 else 'Entry not recorded'
    return 'Exit not recorded' if row.unmatched_entry else 'Entry and exit recorded'


def table(frame, columns):
    display_table(frame[[key for key in columns if key in frame]].rename(columns=LABELS))


def chart(figure):
    figure.update_layout(legend_title_text='', margin=dict(l=12, r=12, t=24, b=12),
                         hovermode='closest')
    st.plotly_chart(figure, width='stretch')


def lap_time(seconds):
    if pd.isna(seconds) or seconds <= 0:
        return '—'
    milliseconds = round(float(seconds) * 1000)
    minutes, remainder = divmod(milliseconds, 60000)
    return f'{minutes}:{remainder / 1000:06.3f}'


def session_clock(seconds):
    if pd.isna(seconds):
        return '—'
    hours, remainder = divmod(round(float(seconds)), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02}:{minutes:02}:{seconds:02}'


def telemetry_lap_label(lap):
    status = []
    if pd.notna(lap.get('pit_out_session_s')):
        status.append('out lap')
    if pd.notna(lap.get('pit_in_session_s')):
        status.append('in lap')
    timing = lap_time(lap.lap_time_s) if pd.notna(lap.lap_time_s) and lap.lap_time_s > 0 else 'untimed · lap time unavailable'
    return f'Lap {int(lap.lap_number)} · {timing}' + (f" · {' / '.join(status)}" if status else '')


def telemetry_window(lap):
    # An out-lap boundary can start at the previous lap's finish, including garage time.
    start = lap.lap_start_session_s
    pit_out = lap.get('pit_out_session_s')
    if pd.notna(pit_out) and start <= pit_out <= lap.lap_end_session_s:
        start = pit_out
    return start, lap.lap_end_session_s


def throttle_brake_figure(car):
    figure = go.Figure()
    if 'throttle_pct_raw' in car and car.throttle_pct_raw.notna().any():
        figure.add_trace(go.Scatter(x=car['Lap elapsed (s)'], y=car.throttle_pct_raw,
                                   name='Throttle (%)', line=dict(color='#27ae60'), connectgaps=False))
    if 'brake_applied' in car and car.brake_applied.notna().any():
        figure.add_trace(go.Scatter(x=car['Lap elapsed (s)'], y=car.brake_applied.astype('Float64') * 100,
                                   name='Brake applied (binary)', line=dict(color='#ef5350', shape='hv'), connectgaps=False))
    figure.update_xaxes(title_text='Lap elapsed (s)')
    figure.update_yaxes(title_text='Throttle (%) / brake state (0 or 100)', range=[-3, 103])
    return figure


def clean_laps(laps):
    mask = ((laps['lap_time_s'] > 0) & (laps['lap_number'] > 1)
            & laps['provider_is_accurate'].eq(1) & laps['deleted'].eq(0)
            & laps['fastf1_generated'].eq(0) & laps['track_status_raw'].eq('1')
            & laps['pit_in_clock_ns'].isna() & laps['pit_out_clock_ns'].isna())
    return laps[mask.fillna(False)].copy()


def observed_car(car):
    """FastF1 missing-Time rows are parser fills, not observed zero-speed samples."""
    mask = car.session_time_s.notna()
    if 'quality_flags' in car:
        mask &= ~car.quality_flags.astype('string').str.contains('fastf1_parser_generated_fill', regex=False, na=False)
    if 'slice_time_ns' in car:
        mask &= car.slice_time_ns.notna()
    if 'Time' in car:
        mask &= car.Time.notna()
    return car[mask].sort_values('session_time_s').copy()


def segmented_car(car):
    """Keep missing observations and gaps over two seconds as breaks in plotted lines."""
    car = car.sort_values('session_time_s').reset_index(drop=True).copy()
    channels = [key for key in ['speed_kmh', 'throttle_pct_raw', 'brake_applied', 'gear', 'rpm', 'drs_code'] if key in car]
    valid = observed_car(car)
    gaps = valid.session_time_s.diff().gt(2)
    for key in channels:
        car.loc[~car.index.isin(valid.index), key] = pd.NA
    breaks = {index - .5: valid.loc[index, 'session_time_s'] for index in valid.index[gaps]}
    car = car.reindex(sorted([*car.index, *breaks]))
    for index, timestamp in breaks.items():
        car.loc[index, 'session_time_s'] = timestamp
    return car


def pace_summary(data):
    laps = data['laps']
    timed = laps[(laps.lap_time_s > 0) & ~laps.deleted.eq(1) & ~laps.fastf1_generated.eq(1)]
    best = timed.groupby('driver_id').lap_time_s.min().rename('Best lap (s)')
    count = laps.groupby('driver_id').size().rename('Recorded laps')
    summary = data['entries'].merge(best, on='driver_id', how='left').merge(count, on='driver_id', how='left')
    summary['Recorded laps'] = summary['Recorded laps'].fillna(0).astype(int)
    summary = summary.sort_values('Best lap (s)', na_position='last')
    summary['Best timed lap'] = summary['Best lap (s)'].map(lap_time)
    return summary


def overview(context):
    data = load(context)
    if data is None:
        return
    st.subheader('Explore the session')
    st.write('Compare Drivers for lap, sector and car telemetry. Race Replay for recorded positions. Strategy Lab for tyres and pit visits. Weather & Context for conditions and race control.')
    for column, label, value in zip(st.columns(4), ['Drivers', 'Recorded laps', 'Weather observations', 'Observed stints'],
                                   [len(data['entries']), len(data['laps']), len(data['weather']), len(data['stints'])]):
        column.metric(label, value)
    code = data['meta']['session_code']
    results = named(data['results'], data['entries']).sort_values('finishing_position')
    if code in ('R', 'S') and not results.empty:
        st.subheader('Race classification' if code == 'R' else 'Sprint classification')
        table(results, ['finishing_position', 'display_name', 'team_name_at_event', 'grid_position',
                        'laps_completed', 'status_raw', 'points'])
        st.caption('Saved provider classification, including retirements and penalties recorded in that result.')
    elif code in ('Q', 'SQ', 'SS') and not results.empty:
        st.subheader('Qualifying classification')
        for number in (1, 2, 3):
            results[f'Q{number}'] = (results[f'q{number}_ns'] / 1e9).map(lap_time)
        table(results, ['finishing_position', 'display_name', 'team_name_at_event', 'Q1', 'Q2', 'Q3'])
    else:
        st.subheader('Practice times' if code.startswith('FP') else 'Recorded session pace')
        st.caption('Fastest recorded non-deleted, non-generated timed lap. Fuel loads and engine modes are unknown; this is not a race classification.')
        table(pace_summary(data), ['display_name', 'team_name_at_event', 'Best timed lap', 'Recorded laps'])
    with st.expander('Drivers and teams in this session'):
        table(data['entries'], ['driver_number', 'display_name', 'team_name_at_event'])


def compare(context):
    data = load(context)
    if data is None:
        return
    entries = data['entries']
    labels = dict(zip(entries.driver_id, entries.display_name))
    selected = st.multiselect('Drivers', list(labels), default=list(labels)[:2],
                              format_func=labels.get, key='pilot_compare')
    clean = st.checkbox('Clean laps only', value=True)
    st.caption('After lap 1, explicitly accurate, not deleted or generated, green track and no pit entry or exit; unknown quality is excluded.')
    selected_laps = data['laps'][data['laps'].driver_id.isin(selected)].copy()
    compounds = sorted(selected_laps.compound_raw.fillna('Unknown').unique())
    laps = clean_laps(data['laps']) if clean else data['laps'].copy()
    laps = laps[laps.driver_id.isin(selected)]
    laps['compound_raw'] = laps.compound_raw.fillna('Unknown')
    chosen = st.multiselect('Tyre compounds', compounds, default=compounds)
    st.caption('Choices include only compounds recorded for the selected drivers, including laps excluded by Clean laps only.')
    laps = named(laps[laps.compound_raw.isin(chosen)], entries).sort_values(['driver_id', 'lap_number'])
    if not laps.empty:
        st.subheader('Lap times')
        chart(px.line(laps, x='lap_number', y='lap_time_s', color='display_name', markers=True,
                      labels=LABELS, hover_data=['compound_raw', 'tyre_age_laps']))
        st.subheader('Sector Times')
        st.caption('Each box covers the middle 50% of sector times; its line is the median. Whiskers extend to values within 1.5× the box height; dots show recorded sectors.')
        sectors = laps.melt(id_vars=['display_name', 'lap_number'],
                            value_vars=['sector1_s', 'sector2_s', 'sector3_s'],
                            var_name='Sector', value_name='Time (s)').dropna()
        sectors['Sector'] = sectors.Sector.map({f'sector{i}_s': f'Sector {i}' for i in (1, 2, 3)})
        if not sectors.empty:
            chart(px.box(sectors, x='Sector', y='Time (s)', color='display_name', points='all', labels=LABELS))
        with st.expander('Lap and sector details'):
            table(laps, ['display_name', 'lap_number', 'lap_time_s', 'sector1_s', 'sector2_s', 'sector3_s', 'compound_raw', 'tyre_age_laps'])
    else:
        st.info('No laps match these filters. Try other drivers, compounds or turn off Clean laps only.')
    telemetry(context, data, selected)


def telemetry(context, data, selected):
    st.subheader('Car Telemetry')
    if not selected:
        st.info('Select a driver to inspect telemetry.')
        return
    names = dict(zip(data['entries'].driver_id, data['entries'].display_name))
    driver = st.selectbox('Telemetry driver', selected, format_func=names.get, key='pilot_telemetry_driver')
    laps = data['laps']
    eligible = laps[(laps.driver_id == driver) & laps.lap_start_session_s.notna() & laps.lap_end_session_s.notna()]
    options = eligible.lap_number.astype(int).tolist()
    if not options:
        st.info('Lap boundaries are unavailable for this driver.')
        return
    if st.session_state.get('pilot_telemetry_lap') not in options:
        st.session_state['pilot_telemetry_lap'] = options[0]
    lap_labels = {int(row.lap_number): telemetry_lap_label(row) for _, row in eligible.iterrows()}
    number = st.selectbox('Telemetry lap', options, format_func=lap_labels.get, key='pilot_telemetry_lap')
    try:
        car = archive.dense(str(archive.root().resolve()), context['archive_session_id'],
                            context['archive_version_id'], 'car', (driver,))
    except (OSError, ValueError) as error:
        st.error(f'Telemetry could not be loaded: {error}')
        return
    unaligned = car_clock_unavailable(car, data['meta'])
    lap = eligible[eligible.lap_number == number].iloc[0]
    window_start, window_end = telemetry_window(lap)
    car = car[car.session_time_s.between(window_start, window_end)].copy()
    if window_start > lap.lap_start_session_s:
        st.caption('Out lap: showing samples from recorded pit exit, excluding the preceding garage interval. Elapsed time starts at pit exit; this is not an official lap time.')
    if pd.isna(lap.lap_time_s) or lap.lap_time_s <= 0:
        st.caption('Official lap time is unavailable for this lap; recorded telemetry and available sector times remain inspectable.')
    samples = observed_car(car)
    car = segmented_car(car)
    if samples.empty:
        st.info(UNALIGNED_CAR_MESSAGE if unaligned else 'No observed car samples are available inside this lap.')
        return
    car['Lap elapsed (s)'] = car.session_time_s - window_start
    channels = [key for key in ['speed_kmh', 'throttle_pct_raw', 'brake_applied', 'gear', 'rpm', 'drs_code']
                if key in car and car[key].notna().any()]
    groups = [('Speed', ['speed_kmh']), ('Throttle & brake', ['throttle_pct_raw', 'brake_applied']),
              ('Engine & gear', ['rpm', 'gear']), ('DRS', ['drs_code'])]
    groups = [(title, [key for key in keys if key in channels]) for title, keys in groups]
    groups = [(title, keys) for title, keys in groups if keys]
    if not groups:
        st.info('No car telemetry channels were recorded for this lap.')
        return
    for tab, (title, group_channels) in zip(st.tabs([title for title, _ in groups]), groups):
        with tab:
            if title == 'Throttle & brake':
                chart(throttle_brake_figure(car))
                st.caption('Green: throttle percentage. Red: binary brake state, 100 = applied and 0 = released; brake pressure is unavailable.')
                continue
            for channel in group_channels:
                display = car.copy()
                if channel == 'brake_applied':
                    display[channel] = display[channel].astype('Float64')
                figure = px.line(display, x='Lap elapsed (s)', y=channel, labels=LABELS)
                figure.update_traces(connectgaps=False)
                if channel in ('gear', 'brake_applied', 'drs_code'):
                    figure.update_traces(line_shape='hv')
                if channel == 'brake_applied':
                    figure.update_yaxes(tickvals=[0, 1], ticktext=['Released', 'Applied'])
                chart(figure)
    st.caption(f'{len(samples):,} recorded samples · elapsed time from the displayed interval start. Line breaks mark missing-Time parser fills or gaps over two seconds. Unknown sample quality remains unknown. DRS is the provider status code; tyre temperature is not supplied.')
    with st.expander('Telemetry sample details'):
        samples = samples.copy()
        samples['Lap elapsed (s)'] = samples.session_time_s - window_start
        table(samples, ['Lap elapsed (s)'] + channels)


def strategy(context):
    data = load(context)
    if data is None:
        return
    st.subheader('Observed tyre stints')
    stints = named(data['stints'], data['entries'])
    if not stints.empty:
        plotted = stints.copy()
        plotted['Lap span'] = plotted.last_observed_lap - plotted.first_observed_lap + 1
        plotted['compound_raw'] = plotted.compound_raw.fillna('UNKNOWN')
        chart(px.bar(plotted, x='Lap span', y='display_name', color='compound_raw',
                     base='first_observed_lap', orientation='h', color_discrete_map=TYRES,
                     labels={**LABELS, 'Lap span': 'Lap'},
                     hover_data=['first_observed_lap', 'last_observed_lap', 'observed_lap_count', 'missing_lap_count']))
        with st.expander('Stint details'):
            table(stints, ['display_name', 'stint_number', 'compound_raw', 'first_observed_lap', 'last_observed_lap', 'observed_lap_count', 'start_tyre_age_laps', 'end_tyre_age_laps', 'missing_lap_count'])
        st.caption('Bars span the first to last observed lap; gaps in recorded laps are listed in the details.')
    else:
        st.info('No tyre stints were recorded for this session.')
    st.subheader('Observed pit visits')
    visits = named(data['pit_visits'], data['entries'])
    if not visits.empty:
        visits['Visit record'] = visits.apply(pit_visit_label, axis=1)
        for field in ('entry_lap', 'exit_lap', 'duration_s'):
            visits[field] = visits[field].map(lambda value: '—' if pd.isna(value) else f'{value:g}')
        table(visits, ['display_name', 'entry_lap', 'exit_lap', 'duration_s', 'Visit record'])
    else:
        st.info('No pit entries or exits were recorded for this session.')
    st.caption('Entry-to-exit duration includes pit-lane travel and any garage time. It is not stationary stop time or race pit loss. A missing entry or exit leaves duration unknown.')
    st.caption('An initial pit departure can be a normal first exit from the garage, with no earlier entry to pair. Other unmatched visits may reflect recording gaps or a session ending in the pits; a partial-coverage flag alone cannot establish the cause.')
    laps = named(clean_laps(data['laps']), data['entries'])
    if not laps.empty:
        st.subheader('Lap time and tyre age')
        drivers = laps[['driver_id', 'display_name']].drop_duplicates().set_index('driver_id').display_name.to_dict()
        focus = st.selectbox('Tyre-age driver', list(drivers), format_func=drivers.get, key='pilot_tyre_age_driver')
        focused = laps[laps.driver_id == focus].copy()
        focused['compound_raw'] = focused.compound_raw.fillna('UNKNOWN')
        chart(px.scatter(focused, x='tyre_age_laps', y='lap_time_s', color='compound_raw',
                         color_discrete_map=TYRES, labels=LABELS,
                         hover_data=['display_name', 'lap_number', 'stint_number']))
        st.caption('One driver’s clean laps, coloured by recorded compound. Tyre age is laps of use, including prior use; fuel, traffic, conditions and run programme also affect pace. This does not isolate tyre degradation.')


def weather(context):
    data = load(context)
    if data is None:
        return
    observations = data['weather'].copy()
    if not observations.empty:
        observations['Session minute'] = observations.session_time_s / 60
        observations = observations.sort_values('Session minute')
        st.subheader('Session conditions')
        temperature_tab, wind_tab, air_tab, rain_tab = st.tabs(['Temperature', 'Wind', 'Humidity & pressure', 'Rain'])
        with temperature_tab:
            temperatures = [key for key in ['air_temperature_c', 'track_temperature_c'] if observations[key].notna().any()]
            if temperatures:
                melted = observations.melt(id_vars=['Session minute'], value_vars=temperatures,
                                           var_name='Measurement', value_name='Temperature (°C)')
                melted['Measurement'] = melted.Measurement.map(LABELS)
                chart(px.line(melted, x='Session minute', y='Temperature (°C)', color='Measurement'))
            else:
                st.info('No temperature measurements were recorded.')
        for tab, channels in [(wind_tab, ['wind_speed_mps', 'wind_direction_deg']),
                              (air_tab, ['relative_humidity_pct', 'pressure_hpa']),
                              (rain_tab, ['rainfall_flag'])]:
            with tab:
                available = [key for key in channels if observations[key].notna().any()]
                if not available:
                    st.info('No measurements were recorded for this view.')
                for channel in available:
                    figure = px.line(observations, x='Session minute', y=channel, labels=LABELS)
                    if channel == 'rainfall_flag':
                        figure.update_traces(line_shape='hv')
                        figure.update_yaxes(tickvals=[0, 1], ticktext=['No rain', 'Rain'])
                    chart(figure)
        with st.expander('Weather observations'):
            observations['rainfall_flag'] = observations.rainfall_flag.map({0: 'No rain', 1: 'Rain'})
            observations['Session time (HH:MM:SS)'] = observations.session_time_s.map(session_clock)
            table(observations, ['Session time (HH:MM:SS)', 'air_temperature_c', 'track_temperature_c', 'relative_humidity_pct',
                                 'wind_speed_mps', 'wind_direction_deg', 'pressure_hpa', 'rainfall_flag'])
        st.caption('Recorded conditions. Wind direction is in compass degrees. Pressure reference is unspecified by the provider. Rain is a yes/no observation, not rainfall in millimetres. Missing measurements remain unknown.')
    else:
        st.info('No weather observations were published for this session.')
    st.subheader('Track conditions')
    status = data['track_status'].copy()
    if status.empty:
        st.info('No track-condition changes were recorded.')
    else:
        status['Session time (HH:MM:SS)'] = status.session_time_s.map(session_clock)
        table(status.sort_values('session_time_s'), ['Session time (HH:MM:SS)', 'message_raw'])
    st.subheader('Race control')
    control = data['race_control'].copy()
    if control.empty:
        st.info('No race-control messages were recorded.')
    else:
        names = data['entries'].set_index('driver_id').display_name
        control['display_name'] = control.driver_id.map(names)
        control['Session time (HH:MM:SS)'] = control.session_time_s.map(session_clock)
        table(control.sort_values('session_time_s'), ['Session time (HH:MM:SS)', 'category_raw', 'message_text', 'flag_raw', 'display_name', 'lap_number', 'sector'])


