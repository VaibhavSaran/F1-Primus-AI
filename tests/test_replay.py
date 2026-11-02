"""Replay must disclose only observed prefixes, never final classifications."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess

import pytest


def session_data():
    return {
        'schema_version': 'session-v1', 'event_id': '2025:1', 'season': 2025,
        'round_number': 1, 'session_id': 'R',
        'source': {'provider': 'fixture', 'source_id': 'test:R',
                   'fetched_at_utc': '2026-10-23T00:00:00+00:00'},
        'quality_flags': ['historical_availability_unverified'],
        'entries': [{'driver_id': 'a', 'abbreviation': 'AAA', 'position': 2, 'status': 'Retired'},
                    {'driver_id': 'b', 'abbreviation': 'BBB', 'position': 1}],
        'laps': [
            {'driver_id': 'a', 'lap_number': 1, 'session_time_s': 60, 'lap_time_s': 60,
             'compound': 'MEDIUM', 'stint': 1, 'track_status': '1'},
            {'driver_id': 'b', 'lap_number': 1, 'session_time_s': 62, 'lap_time_s': 62},
            {'driver_id': 'a', 'lap_number': 2, 'session_time_s': 125, 'lap_time_s': 65,
             'compound': 'MEDIUM', 'stint': 1, 'pit_in': True, 'track_status': '4'},
            {'driver_id': 'b', 'lap_number': 2, 'session_time_s': 126, 'lap_time_s': 64},
        ],
        'positions': [{'driver_id': 'a', 'session_time_s': 60, 'x': 10, 'y': 20},
                      {'driver_id': 'a', 'session_time_s': 65, 'x': 30, 'y': 40}],
        'race_control': [{'session_time_s': 100, 'message': 'Safety car deployed', 'category': 'Flag'}],
    }


def replay(data=None):
    from data.replay import ReplaySession
    return ReplaySession.from_mapping(data or session_data())


def test_seek_rewind_uses_completed_laps_without_final_classification():
    session = replay()
    first = session.frame_at(61000)
    assert first.frame.session_time_s == 61
    assert first.frame.drivers['a']['completed_laps'] == 1
    assert first.frame.drivers['b']['completed_laps'] == 0
    assert first.frame.drivers['a']['position'] is None
    assert 'status' not in first.frame.drivers['a']
    assert len(session.frame_at(126000).laps) == 4
    assert session.frame_at(61000) == first


def test_future_perturbation_cannot_change_prefix_analytics_or_positions():
    original = replay().frame_at(62000)
    changed = session_data()
    changed['laps'][2]['lap_time_s'] = 999
    changed['laps'][2]['compound'] = 'SOFT'
    changed['positions'][1]['x'] = 99999
    changed['entries'][0]['position'] = 1
    other = replay(changed).frame_at(62000)
    assert other.frame == original.frame
    assert other.laps == original.laps
    assert other.stints == original.stints
    assert other.markers == original.markers


def test_hindsight_is_explicit_and_does_not_change_the_tower_clock():
    session = replay()
    causal, hindsight = session.frame_at(61000), session.frame_at(61000, hindsight=True)
    assert len(causal.laps) == 1 and len(hindsight.laps) == 4
    assert causal.frame == hindsight.frame
    assert hindsight.provenance['analysis_scope'] == 'full_session_hindsight'
    assert causal.provenance['forecast_eligible'] is False


def test_track_and_tower_share_clock_and_stale_positions_disappear():
    session = replay()
    at = session.frame_at(62000).frame.drivers['a']
    assert at['coordinates'] == {'x': 10.0, 'y': 20.0, 'sample_clock_ms': 60000}
    assert at['completed_laps'] == 1
    assert session.frame_at(71000).frame.drivers['a']['coordinates'] is None


def test_pit_and_flag_markers_only_appear_after_observation_clock():
    session = replay()
    assert session.frame_at(99000).markers == ()
    assert [m['kind'] for m in session.frame_at(100000).markers] == ['race_control']
    markers = session.frame_at(125000).markers
    assert {m['kind'] for m in markers} == {'race_control', 'pit_in', 'track_status'}
    assert all(m['clock_ms'] <= 125000 for m in markers)


def test_missing_fields_remain_unknown_and_no_retirement_is_inferred():
    data = session_data()
    data['positions'] = []
    session = replay(data)
    driver = session.frame_at(126000).frame.drivers['b']
    assert driver['compound'] is None and driver['coordinates'] is None
    assert driver['position'] is None and driver['gap_s'] is None
    assert session.track_unavailable_reason
    assert session.frame_at(126000).frame.weather == {}


def test_unknown_duration_provider_rows_do_not_certify_lap_crossings_or_tyres():
    data = session_data()
    data['laps'][0]['lap_time_s'] = None
    session = replay(data)
    at = session.frame_at(62000)
    driver = at.frame.drivers['a']
    assert driver['completed_laps'] == 0
    assert driver['last_lap_time_s'] is None and driver['compound'] is None
    assert list(at.frame.drivers)[0] == 'b'
    assert len(at.laps) == 1 and not at.stints
    assert at.provenance['untimed_lap_records_excluded'] == 1
    # A later timed lap certifies one observed crossing, not the unknown earlier row.
    assert session.frame_at(125000).frame.drivers['a']['completed_laps'] == 1
    assert len(session.laps) == 4


@pytest.mark.parametrize('duration', [None, 0.0])
def test_unknown_or_zero_duration_laps_are_not_native_completed_lap_seek_options(tmp_path, monkeypatch, duration):
    from streamlit.testing.v1 import AppTest
    from streamlit_app import services
    root = tmp_path / 'sessions'
    root.mkdir()
    monkeypatch.setattr(services, 'LOCAL', tmp_path)
    data = session_data()
    data['laps'][0]['lap_time_s'] = duration
    (root / ('session-' + '4' * 64 + '.json')).write_text(json.dumps(data), encoding='utf-8')
    script = tmp_path / 'app.py'
    script.write_text('''
from streamlit_app.components.replay_ui import render_replay
render_replay({'season': 2025, 'round_number': 1, 'session': 'R', 'reduced_motion': True})
''', encoding='utf-8')
    app = AppTest.from_file(script).run()
    assert not app.exception
    assert app.selectbox(key='replay_native_lap').options == ['Lap 2 · 0:02:05']


@pytest.mark.parametrize('mutation', [
    lambda d: d.update(schema_version='untrusted'),
    lambda d: d.update(event_id='2025:2'),
    lambda d: d['laps'][0].update(driver_id='unknown'),
    lambda d: d['laps'][0].update(session_time_s=float('nan')),
    lambda d: d['positions'][0].update(x=float('inf')),
    lambda d: d['laps'].append(deepcopy(d['laps'][0])),
    lambda d: d['entries'].append(deepcopy(d['entries'][0])),
    lambda d: d['source'].update(fetched_at_utc='2026-10-23'),
    lambda d: d['laps'][0].update(driver_id=[]),
    lambda d: d['laps'][0].update(session_time_s=1e308),
    lambda d: d['positions'][0].update(driver_id={}),
])
def test_invalid_sources_fail_closed(mutation):
    data = session_data()
    mutation(data)
    with pytest.raises(ValueError):
        replay(data)


@pytest.mark.parametrize('clock', [-1, True, float('nan'), 126001])
def test_invalid_seek_is_rejected(clock):
    with pytest.raises(ValueError):
        replay().frame_at(clock)


def test_bounded_payload_decimates_positions_but_preserves_laps_and_events():
    data = session_data()
    data['positions'] = [{'driver_id': 'a', 'session_time_s': i, 'x': i, 'y': i / 2}
                         for i in range(127)]
    session = replay(data)
    payload = session.browser_payload(max_position_samples=8)
    assert len(payload['positions']) == 8
    assert payload['positions'][0]['clock_ms'] == 0
    assert payload['positions'][-1]['clock_ms'] == 126000
    assert len(payload['laps']) == 4
    assert len(payload['markers']) == 3
    assert payload['position_samples_original'] == 127
    with pytest.raises(ValueError, match='payload'):
        session.browser_payload(max_bytes=100)


def test_full_race_sized_fixture_preserves_prefix_and_bounded_sample_count():
    data = session_data()
    data['entries'] = [{'driver_id': f'd{i:02}'} for i in range(20)]
    data['laps'] = [{'driver_id': e['driver_id'], 'lap_number': lap,
                     'session_time_s': lap * 90 + i, 'lap_time_s': 90.0}
                    for i, e in enumerate(data['entries']) for lap in range(1, 61)]
    data['positions'] = [{'driver_id': e['driver_id'], 'session_time_s': sample * 5,
                          'x': sample, 'y': i}
                         for i, e in enumerate(data['entries']) for sample in range(1200)]
    data['race_control'] = []
    session = replay(data)
    at = session.frame_at(4519000)
    assert len(at.laps) == 1000
    assert {driver['completed_laps'] for driver in at.frame.drivers.values()} == {50}
    payload = session.browser_payload(max_position_samples=4000)
    assert len(payload['positions']) == 4000
    assert len(payload['laps']) == 1200
    assert len(json.dumps(payload).encode()) < 1_500_000


def test_component_commit_rejects_stale_session_version_sequence_and_unknown_driver():
    from data.replay import validate_commit
    session = replay()
    good = {'session_id': session.identity, 'frame_version': session.frame_version,
            'sequence': 2, 'clock_ms': 62000, 'driver_ids': ['a'], 'event': 'pause'}
    assert validate_commit(good, session, last_sequence=1)['clock_ms'] == 62000
    for update in ({'session_id': 'other'}, {'frame_version': 'old'}, {'sequence': 1},
                   {'clock_ms': 999999}, {'driver_ids': ['unknown']}, {'event': 'play'},
                   {'sequence': True}, {'extra': 'unbounded'}):
        with pytest.raises(ValueError):
            validate_commit(dict(good, **update), session, last_sequence=1)
    with pytest.raises(ValueError):
        validate_commit(dict(good, driver_ids=[{}]), session, last_sequence=1)


def test_loading_json_is_bounded_and_does_not_accept_nonfinite_json(tmp_path):
    from data.replay import load_replay
    path = tmp_path / 'session.json'
    path.write_text(json.dumps(session_data()), encoding='utf-8')
    assert load_replay(path).identity == '2025:1:R'
    with pytest.raises(ValueError):
        load_replay(path, max_bytes=10)
    data = session_data()
    data['positions'][0]['x'] = float('nan')
    path.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(ValueError):
        load_replay(path)


def test_native_replay_empty_state_and_committed_seek(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from streamlit_app import services
    root = tmp_path / '.local' / 'sessions'
    root.mkdir(parents=True)
    monkeypatch.setattr(services, 'LOCAL', tmp_path / '.local')
    script = tmp_path / 'app.py'
    script.write_text('''
from streamlit_app.components.replay_ui import render_replay
render_replay({'season': 2025, 'round_number': 1, 'session': 'R', 'reduced_motion': True})
''', encoding='utf-8')
    app = AppTest.from_file(script)
    app.run()
    assert not app.exception
    assert any('No recorded replay' in item.value for item in app.info)
    (root / ('session-' + '1' * 64 + '.json')).write_text(json.dumps(session_data()), encoding='utf-8')
    app.run()
    assert not app.exception
    app.number_input(key='replay_native_seconds').set_value(62.0)
    next(button for button in app.button if button.label == 'Jump to time').click()
    app.run()
    assert not app.exception
    assert app.session_state['replay_commit']['clock_ms'] == 62000
    assert any('0:01:02' in item.value for item in app.caption)
    tower = app.dataframe[0].value
    assert list(tower['Timed laps observed']) == [1, 1]
    next(button for button in app.button if button.label == 'Jump to lap').click()
    app.run()
    assert app.session_state['replay_commit']['clock_ms'] == 60000
    assert app.number_input(key='replay_native_seconds').value == 60.0


def test_native_replay_does_not_open_another_event(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from streamlit_app import services
    root = tmp_path / '.local' / 'sessions'
    root.mkdir(parents=True)
    monkeypatch.setattr(services, 'LOCAL', tmp_path / '.local')
    (root / ('session-' + '2' * 64 + '.json')).write_text(json.dumps(session_data()), encoding='utf-8')
    script = tmp_path / 'app.py'
    script.write_text('''
from streamlit_app.components.replay_ui import render_replay
render_replay({'season': 2026, 'round_number': 1, 'session': 'R', 'reduced_motion': True})
''', encoding='utf-8')
    app = AppTest.from_file(script).run()
    assert not app.exception
    assert not app.dataframe


def test_replay_commits_do_not_duplicate_widget_defaults(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from streamlit.elements.lib import policies
    from streamlit_app import services
    from streamlit_app.components import replay_ui
    import streamlit as st
    root = tmp_path / '.local' / 'sessions'
    root.mkdir(parents=True)
    monkeypatch.setattr(services, 'LOCAL', tmp_path / '.local')
    (root / ('session-' + '3' * 64 + '.json')).write_text(json.dumps(session_data()), encoding='utf-8')
    # Inject at the component return boundary; Python validation and reruns remain real.
    def browser_return(**kwargs):
        return st.session_state.pop('_test_browser_event', None)
    monkeypatch.setattr(replay_ui.components, 'declare_component', lambda *args, **kwargs: browser_return)
    script = tmp_path / 'app.py'
    script.write_text('''
from streamlit_app.components.replay_ui import render_replay
render_replay({'season': 2025, 'round_number': 1, 'session': 'R', 'reduced_motion': False})
''', encoding='utf-8')
    app = AppTest.from_file(script)
    def run_without_duplicate_warning():
        # Streamlit displays this warning only once per process; reset for each transition.
        monkeypatch.setattr(policies, '_shown_default_value_warning', False)
        app.run()
        assert not app.exception
        assert not any('default value' in warning.value and 'Session State API' in warning.value
                       for warning in app.warning)
    run_without_duplicate_warning()
    # Start at an observed preview so a populated source is useful immediately.
    assert app.number_input(key='replay_native_seconds').value == 126.0
    next(button for button in app.button if button.label == 'Jump to lap').click()
    run_without_duplicate_warning()
    assert app.number_input(key='replay_native_seconds').value == 60.0
    session = replay()
    app.session_state['_test_browser_event'] = dict(session_id=session.identity, frame_version=session.frame_version,
        sequence=1, clock_ms=62000, driver_ids=['a'], event='pause')
    run_without_duplicate_warning()
    assert app.session_state['replay_commit']['clock_ms'] == 62000
    assert app.number_input(key='replay_native_seconds').value == 62.0
    assert set(app.dataframe[1].value['Driver']) == {'AAA'}
    app.session_state['replay_mode'] = 'Hindsight'
    run_without_duplicate_warning()
    assert set(app.dataframe[1].value['Driver']) == {'AAA'}
    app.session_state['_test_browser_event'] = dict(session_id=session.identity, frame_version=session.frame_version,
        sequence=2, clock_ms=126000, driver_ids=['b'], event='selection')
    run_without_duplicate_warning()
    assert app.session_state['replay_commit']['driver_ids'] == ['b']
    app.session_state['replay_analysis'] = 'Tyres'
    run_without_duplicate_warning()
    assert any('No observed stint' in item.value for item in app.info)


def test_browser_clock_prefix_and_message_origin_with_real_javascript(tmp_path):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node.js unavailable; real-browser gate still required')
    payload_file = tmp_path / 'payload.json'
    payload_file.write_text(json.dumps(replay().browser_payload()), encoding='utf-8')
    core = Path(__file__).resolve().parents[1] / 'streamlit_app/components/replay_component/clock.js'
    script = tmp_path / 'check.cjs'
    script.write_text('''
const assert = require('node:assert/strict');
const fs = require('node:fs');
const replay = require(process.argv[2]);
const payload = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
assert.equal(replay.validatePayload(payload), true);
assert.equal(replay.validatePayload({...payload, schema_version: 'bad'}), false);
assert.deepEqual(replay.trackBounds([]), null);
assert.deepEqual(replay.trackBounds([{x: 10, y: 20}]), {xMin: 9, xMax: 11, yMin: 19, yMax: 21});
assert.deepEqual(replay.trackBounds(payload.positions), {xMin: 10, xMax: 30, yMin: 20, yMax: 40});
assert.equal(replay.formatClock(3661000), '1:01:01');
const early = replay.frameAt(payload, 62000);
assert.deepEqual(early.drivers.map(d => d.completed_laps), [1, 1]);
assert.deepEqual(early.drivers[0].coordinates, {x: 10, y: 20, sample_clock_ms: 60000});
assert.equal(early.markers.length, 0);
assert.equal(replay.frameAt(payload, 126000).markers.length, 3);
assert.deepEqual(replay.frameAt(payload, 62000), early);
const changed = structuredClone(payload);
changed.laps[2].lap_time_s = 999;
changed.positions[1].x = 9999;
assert.deepEqual(replay.frameAt(changed, 62000), early);
const untimed = structuredClone(payload);
untimed.laps[0].lap_time_s = null;
const verified = replay.frameAt(untimed, 62000);
assert.equal(verified.drivers[0].driver_id, 'b');
assert.equal(verified.drivers.find(d => d.driver_id === 'a').completed_laps, 0);
assert.equal(verified.drivers.find(d => d.driver_id === 'a').compound, null);
assert.equal(replay.frameAt(untimed, 125000).drivers.find(d => d.driver_id === 'a').completed_laps, 1);
const parent = {};
const event = {source: parent, origin: 'https://allowed.test', data: {type: 'streamlit:render', args: {payload}}};
assert.equal(replay.acceptRender(event, parent, 'https://allowed.test'), true);
assert.equal(replay.acceptRender({...event, origin: 'https://attacker.test'}, parent, 'https://allowed.test'), false);
assert.equal(replay.acceptRender({...event, source: {}}, parent, 'https://allowed.test'), false);
console.log('Browser prefix, shared clock, deterministic rewind and origin checks passed');
''', encoding='utf-8')
    result = subprocess.run([node, str(script), str(core), str(payload_file)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_weather_clock_contract_matches_browser_and_ignores_future(tmp_path):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node.js unavailable')
    data = session_data()
    data['weather'] = [
        {'session_time_s': 10, 'air_temperature_c': 21, 'relative_humidity_pct': 75, 'rainfall_flag': 0},
        {'session_time_s': 60, 'air_temperature_c': 22, 'rainfall_flag': 1},
        {'session_time_s': 240, 'air_temperature_c': 99, 'rainfall_flag': 0}]
    session = replay(data)
    clocks = [0, 9999, 10000, 59999, 60000, 180000, 180001, 239999, 240000]
    expected = [session.frame_at(clock).frame.weather for clock in clocks]
    changed = deepcopy(data)
    changed['weather'][2]['air_temperature_c'] = -99
    assert replay(changed).frame_at(60000).frame.weather == expected[4]
    assert session.frame_at(60000, hindsight=True).frame.weather == expected[4]
    assert expected[4]['relative_humidity_pct'] is None  # Never carry an older field into a newer row.
    assert expected[6] == {}
    payload = tmp_path / 'weather-payload.json'
    payload.write_text(json.dumps(session.browser_payload()), encoding='utf-8')
    core = Path(__file__).resolve().parents[1] / 'streamlit_app/components/replay_component/clock.js'
    script = "const r=require(process.argv[1]),p=require(process.argv[2]);if(!r.validatePayload(p))process.exit(1);console.log(JSON.stringify(" + json.dumps(clocks) + ".map(t=>r.frameAt(p,t).weather)));"
    result = subprocess.run([node, '-e', script, str(core), str(payload)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == expected


