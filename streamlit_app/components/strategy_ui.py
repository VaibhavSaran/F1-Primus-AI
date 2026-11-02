"""Observed clock-prefix stints and explicitly submitted, uncalibrated scenarios."""
from dataclasses import asdict, replace
from uuid import uuid4

import streamlit as st

from data.replay import load_replay
from ml.features import clean_laps
from ml.simulator import CompoundPace, DriverScenario, PitLoss, PitStop, SimulationConfig, simulate_race
from security.access import AccessDenied
from security.limits import LimitRejected, SQLiteLimiter
from streamlit_app import services


@st.cache_data(show_spinner=False, max_entries=12)
def _load(path, modified_ns, size):
    return load_replay(path)


def _sources(context):
    matches, skipped, total_bytes = [], 0, 0
    for index, path in enumerate(services.local_files(services.LOCAL / 'sessions', 'session-*.json')):
        if index >= 64:
            skipped += 1
            break
        try:
            info = path.stat()
            total_bytes += info.st_size
            if total_bytes > 50_000_000:
                skipped += 1
                continue
            session = _load(str(path), info.st_mtime_ns, info.st_size)
            if (session.event.season, session.event.round_number, session.session_id) == (
                    context['season'], context['round_number'], context['session']):
                matches.append(session)
        except (OSError, ValueError, TypeError):
            skipped += 1
    return matches, skipped


def _show_result(result):
    st.warning(result.interpretation)
    st.dataframe([{'Driver': row.driver_id, 'Expected position (scenario)': row.expected_finish_position,
                   'Finish after decision leader boundary (s)': row.finish_time_median_s,
                   '5% time (s)': row.finish_time_lower_s, '95% time (s)': row.finish_time_upper_s}
                  for row in result.drivers], hide_index=True, width='stretch')
    frequencies = [{'Driver': row.driver_id, 'Position': index + 1, 'Scenario finish frequency': value}
                   for row in result.drivers for index, value in enumerate(row.finish_frequencies)]
    st.dataframe(frequencies, hide_index=True, width='stretch')
    trajectory = [{'Driver': row.driver_id, 'Future laps completed': index,
                   'Time since decision leader boundary (s)': value}
                  for row in result.drivers for index, value in enumerate(row.trajectory_median_s)]
    st.line_chart(trajectory, x='Future laps completed', y='Time since decision leader boundary (s)', color='Driver')
    with st.expander('Scenario assumptions and modeling limits'):
        st.json(result.assumptions)
        for limitation in result.limits:
            st.write(limitation)
    st.download_button('Download scenario JSON', services.json_export(asdict(result)),
                        file_name='strategy-scenario.json', mime='application/json')


def _show_comparison(alternative, reference):
    st.write('Alternative versus assumed no-further-stop reference')
    st.caption('Reference uses the same manually assumed current tyre, pace, uncertainty and weather path with no further stops. '
               'Both are counterfactual scenarios; differences are not measured strategy benefits.')
    reference_rows = {row.driver_id: row for row in reference.drivers}
    differences = [{'Driver': row.driver_id, 'Alternative minus reference median finish time (s)':
                    row.finish_time_median_s - reference_rows[row.driver_id].finish_time_median_s}
                   for row in alternative.drivers]
    st.dataframe(differences, hide_index=True, width='stretch')
    gap_path = [{'Driver': row.driver_id, 'Future laps completed': index,
                 'Alternative minus reference cumulative median time (s)': value - base}
                for row in alternative.drivers
                for index, (value, base) in enumerate(zip(row.trajectory_median_s,
                                                        reference_rows[row.driver_id].trajectory_median_s))]
    st.line_chart(gap_path, x='Future laps completed',
                  y='Alternative minus reference cumulative median time (s)', color='Driver')
    st.download_button('Download reference JSON', services.json_export(asdict(reference)),
                        file_name='strategy-reference.json', mime='application/json')


def _manual_scenario(session, snapshot, scope):
    st.warning('Manual scenario: every pace, gap, uncertainty, future stop and weather value is a user assumption. '
               'Observed completion order does not supply an instantaneous race gap. No fitted model is implied.')
    selected = st.multiselect('Scenario participants (up to 4)', [entry['driver_id'] for entry in session.entries],
                              max_selections=4, key='strategy_participants')
    with st.form('strategy_scenario_form'):
        completed = st.number_input('Shared completed race lap at decision boundary', min_value=0, max_value=100,
                                     value=None, step=1)
        remaining = st.number_input('Remaining race laps', min_value=1, max_value=100, value=10, step=1)
        samples = st.number_input('Simulation samples', min_value=1, max_value=2000, value=500, step=100)
        seed = st.number_input('Reproducible seed', min_value=0, max_value=2**32 - 1, value=42, step=1)
        weather_path = st.text_input('Weather pace delta per future lap (seconds, comma separated; required)')
        evidence = st.text_input('Explain pace/gap/uncertainty/weather assumptions (required)', max_chars=1000)
        inputs = []
        for driver in selected:
            st.write(f'{driver} · manual inputs')
            gap = st.number_input(f'{driver} gap to decision leader (s; one participant must be 0)',
                                   min_value=0., max_value=10_000., value=None)
            compound = st.selectbox(f'{driver} current compound', sorted({'SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET'}), index=None)
            age = st.number_input(f'{driver} current tyre age (laps)', min_value=0., max_value=300., value=None)
            pace = st.number_input(f'{driver} expected pace at tyre age zero (s)', min_value=.001, max_value=1200., value=None)
            slope = st.number_input(f'{driver} pace change per tyre-age lap (s/lap)', min_value=-10., max_value=10., value=None)
            sigma = st.number_input(f'{driver} lap variability standard deviation (s)', min_value=0., max_value=120., value=None)
            pit_lap = st.number_input(f'{driver} optional stop after absolute race lap (blank = no stop)',
                                       min_value=1, max_value=100, value=None, step=1)
            pit_compound = st.selectbox(f'{driver} post-stop compound (required with stop)',
                                         ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET'], index=None)
            pit_pace = st.number_input(f'{driver} post-stop age-zero pace (s)', min_value=.001, max_value=1200., value=None)
            pit_slope = st.number_input(f'{driver} post-stop pace change (s/lap)', min_value=-10., max_value=10., value=None)
            pit_sigma = st.number_input(f'{driver} post-stop variability (s)', min_value=0., max_value=120., value=None)
            pit_loss = st.number_input(f'{driver} net pit loss (s; includes stationary and lane delta)',
                                        min_value=0., max_value=600., value=None)
            inputs.append((driver, gap, compound, age, pace, slope, sigma,
                           pit_lap, pit_compound, pit_pace, pit_slope, pit_sigma, pit_loss))
        compare = st.checkbox('Compare with an assumed no-further-stop reference (two separately admitted simulations).')
        confirmed = st.checkbox('I explicitly assume these future pit/compound and weather paths; this is an uncalibrated scenario.')
        submitted = st.form_submit_button('Run scenario', disabled=not selected)
    if submitted:
        try:
            principal = services.local_identity()
            if not confirmed or completed is None:
                raise ValueError('Confirm the assumptions and supply a shared completed-lap decision boundary.')
            deltas = tuple(float(value.strip()) for value in weather_path.split(','))
            drivers = []
            for driver, gap, compound, age, pace, slope, sigma, pit_lap, pit_compound, pit_pace, pit_slope, pit_sigma, pit_loss in inputs:
                paces = [CompoundPace(compound, pace, slope, sigma, 'user_assumption', evidence)]
                pits = ()
                if pit_lap is not None:
                    if pit_compound == compound:
                        if (pit_pace, pit_slope, pit_sigma) != (pace, slope, sigma):
                            raise ValueError('A repeated compound uses the same explicit age-zero pace, slope and variability.')
                    else:
                        paces.append(CompoundPace(pit_compound, pit_pace, pit_slope, pit_sigma, 'user_assumption', evidence))
                    pits = (PitStop(pit_lap, pit_compound, PitLoss(net_loss_s=pit_loss)),)
                elif any(value is not None for value in (pit_compound, pit_pace, pit_slope, pit_sigma, pit_loss)):
                    raise ValueError('Supply a future pit lap with pit inputs, or leave every optional pit field blank.')
                drivers.append(DriverScenario(driver, completed, gap, compound, age, tuple(paces), pits))
            config = SimulationConfig(remaining, samples, seed, deltas, evidence)
            limiter = SQLiteLimiter(services.LOCAL / 'limits.sqlite3')
            result = simulate_race(tuple(drivers), config, principal=principal, limiter=limiter,
                                    request_id='scenario-' + uuid4().hex)
            st.session_state['strategy_result'] = dict(scope=scope, result=result,
                                                      observed_clock_ms=snapshot.provenance['committed_clock_ms'])
            if compare:
                reference_drivers = tuple(replace(driver, pits=(),
                                                  paces=tuple(pace for pace in driver.paces if pace.compound == driver.compound))
                                          for driver in drivers)
                reference = simulate_race(reference_drivers, config, principal=principal, limiter=limiter,
                                           request_id='scenario-reference-' + uuid4().hex)
                st.session_state['strategy_result']['reference'] = reference
        except (ValueError, AccessDenied, LimitRejected) as error:
            st.error(f'Scenario unavailable: {error}')
    saved = st.session_state.get('strategy_result')
    if saved and saved['scope'] == scope:
        st.caption(f'Saved submitted scenario · source decision clock {saved["observed_clock_ms"] / 1000:.3f} s. '
                   'Editing controls retains the prior result until Run scenario is submitted again.')
        _show_result(saved['result'])
        if saved.get('reference'):
            _show_comparison(saved['result'], saved['reference'])


def render_strategy(context):
    sources, skipped = _sources(context)
    if skipped:
        st.warning('Some local sources were invalid or exceeded the bounded source scan and were skipped.')
    if not sources:
        st.info('No normalized historical source matches this event/session. Observed stints and source-linked scenarios need a trusted local session.')
        return
    versions = [session.frame_version for session in sources]
    selected = st.selectbox('Strategy historical source', versions,
                             format_func=lambda value: next(f'{s.identity} · fetched {s.source["fetched_at_utc"]}'
                                                            for s in sources if s.frame_version == value))
    session = sources[versions.index(selected)]
    scope = (session.identity, session.frame_version)
    if st.session_state.get('strategy_scope') != scope:
        for key in ('strategy_clock', 'strategy_result', 'strategy_participants'):
            st.session_state.pop(key, None)
        st.session_state['strategy_scope'] = scope
    replay = st.session_state.get('replay_commit', {})
    initial = replay.get('clock_ms', 0) / 1000 if (replay.get('session_id'), replay.get('frame_version')) == scope else 0.
    clock_s = st.number_input('Strategy observed clock (session-relative seconds)', min_value=0.,
                               max_value=session.duration_ms / 1000, value=float(initial), step=1., key='strategy_clock')
    snapshot = session.frame_at(round(clock_s * 1000))
    st.caption(f'Historical observed prefix through {clock_s:.3f} s · source availability at historical UTC cutoff is unverified.')
    mode = st.segmented_control('Strategy view', ['Observed', 'Scenario'], default='Observed')
    if mode == 'Scenario':
        _manual_scenario(session, snapshot, scope)
        return
    if snapshot.stints:
        st.dataframe(list(snapshot.stints), hide_index=True, width='stretch')
        import plotly.express as px
        bars = [dict(row, compound=row['compound'] or 'Unknown compound',
                     start_lap=row['first_lap'] - 1, observed_span=row['last_lap'] - row['first_lap'] + 1,
                     label=f'{row["driver_id"]} / stint {row["stint"]}') for row in snapshot.stints]
        figure = px.bar(bars, x='observed_span', y='label', base='start_lap', color='compound', orientation='h',
                         labels={'observed_span': 'Observed race-lap span', 'start_lap': 'First lap boundary', 'label': 'Driver / stint'})
        st.plotly_chart(figure, width='stretch')
        st.caption('Bars span completed observed laps; gaps inside a stint are not imputed. No tyre-health percentage is inferred.')
    else:
        st.info('No observed stint has completed by this clock.')
    if snapshot.laps:
        st.line_chart([{'Driver': lap['driver_id'], 'Race lap': lap['lap_number'], 'Observed lap time (s)': lap['lap_time_s']}
                       for lap in snapshot.laps if lap['lap_time_s'] is not None],
                      x='Race lap', y='Observed lap time (s)', color='Driver')
    if snapshot.markers:
        st.write('Observed pit and flag timeline')
        st.dataframe(list(snapshot.markers), hide_index=True, width='stretch')
    strict = [lap for lap in snapshot.laps if all(lap.get(key) is False for key in ('deleted', 'pit_in', 'pit_out'))]
    count = len(clean_laps(strict))
    st.info(f'{count} strict clean laps in this clock prefix. Automatic fitted scenarios require validated clean-lap '
            'provenance and a held-out evaluation; no eligible calibrated model is available. Manual assumptions are available in Scenario.')
    st.caption('Observed times include pit, flagged and unknown-deletion laps. Degradation/fuel effects and counterfactual strategy accuracy are not established.')
    with st.expander('Strategy source quality and provenance'):
        st.json(snapshot.provenance)
