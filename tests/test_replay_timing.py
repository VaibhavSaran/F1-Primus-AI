import json
from pathlib import Path
import shutil
import subprocess

import pytest

from test_replay import session_data, replay


def timing_data():
    data = session_data()
    data['timing_basis'] = 'timing_stream'
    data['timing'] = [
        {'driver_id': 'a', 'session_time_s': 10, 'position': 2, 'gap_to_leader_raw': '+1.234', 'interval_to_ahead_raw': '+0.500'},
        {'driver_id': 'b', 'session_time_s': 10, 'position': 1, 'gap_to_leader_raw': 'LAP 1', 'interval_to_ahead_raw': 'LAP 1'},
        {'driver_id': 'a', 'session_time_s': 120, 'position': 1},
        {'driver_id': 'b', 'session_time_s': 140, 'position': 2, 'gap_to_leader_raw': '+ 1 Lap', 'interval_to_ahead_raw': '+2 Laps'},
        {'driver_id': 'a', 'session_time_s': 300, 'position': 2, 'gap_to_leader_raw': '+99.999'},
    ]
    return data


def test_recorded_rank_overrules_completion_and_fields_expire_independently():
    session = replay(timing_data())
    early = session.frame_at(62000).frame.drivers
    assert list(early) == ['b', 'a']
    assert early['a']['gap_s'] == 1.234
    assert early['a']['interval_s'] == 0.5
    assert early['b']['gap_label'] == 'Leader'
    later = session.frame_at(131000).frame.drivers['a']
    assert later['position'] == 1 and later['position_fresh']
    assert later['gap_s'] is None and not later['gap_fresh']
    assert later['interval_label'] is None
    stale = session.frame_at(261000).frame.drivers
    assert all(d['position'] is None for d in stale.values())
    assert all('status' not in d for d in stale.values())


def test_lap_deficits_and_leader_counter_are_distinct_and_clock_causal():
    session = replay(timing_data())
    driver = session.frame_at(140000).frame.drivers['b']
    assert driver['gap_laps'] == 1 and driver['interval_laps'] == 2
    assert driver['gap_label'] == '+1 lap' and driver['interval_label'] == '+2 laps'
    assert session.frame_at(62000).frame == session.frame_at(62000, hindsight=True).frame
    changed = timing_data()
    changed['timing'][-1]['gap_to_leader_raw'] = '+999'
    assert replay(changed).frame_at(62000).frame == session.frame_at(62000).frame
    changed['timing'][0]['gap_to_leader_raw'] = 'LAP 9'
    assert replay(changed).frame_at(62000).frame.drivers['a']['gap_laps'] is None


def test_duplicate_observed_ranks_are_preserved_and_practice_never_gets_race_rank():
    data = timing_data()
    data['timing'][0]['position'] = 1
    assert [d['position'] for d in replay(data).frame_at(62000).frame.drivers.values()] == [1, 1]
    data['session_id'] = 'FP1'
    session = replay(data)
    assert session.timing_basis == 'derived_timed_lap_record_order'
    assert all(d['position'] is None for d in session.frame_at(62000).frame.drivers.values())


@pytest.mark.parametrize('raw', ['', 'PIT', 'STOP', '+1:02.345', 'LAP 9'])
def test_unrecognized_or_unavailable_gap_update_clears_old_values(raw):
    data = timing_data()
    data['timing'][2]['gap_to_leader_raw'] = raw
    data['timing'][2]['position'] = 2
    driver = replay(data).frame_at(120000).frame.drivers['a']
    assert driver['gap_s'] is None and driver['gap_laps'] is None and driver['gap_label'] is None
    assert driver['interval_s'] == 0.5  # Independent source update ages.


@pytest.mark.parametrize('raw,laps', [('1 L', 1), ('2 L', 2), ('1L', 1), ('2L', 2), ('28L', 28)])
def test_archived_lap_deficit_spellings_do_not_imply_retirement(raw, laps):
    data = timing_data()
    data['timing'][0]['gap_to_leader_raw'] = raw
    driver = replay(data).frame_at(62000).frame.drivers['a']
    assert driver['gap_laps'] == laps and driver['gap_s'] is None
    assert 'status' not in driver


def test_lap_end_checkpoint_basis_does_not_invent_gaps_and_timing_is_lossless():
    data = timing_data()
    data['timing_basis'] = 'lap_end_position'
    data['timing'] = [{'driver_id': 'b', 'session_time_s': 62, 'position': 1}]
    session = replay(data)
    at = session.frame_at(62000).frame.drivers
    assert list(at) == ['b', 'a'] and at['b']['gap_label'] is None
    assert at['a']['position'] is None
    assert session.browser_payload(max_position_samples=2)['timing'] == session.timing


@pytest.mark.parametrize('update', [{'position': True}, {'position': 65},
                                  {'position': 1.5}, {'gap_to_leader_raw': 1.234}, {'driver_id': 'unknown'}])
def test_invalid_timing_sources_fail_closed(update):
    data = timing_data()
    data['timing'][0].update(update)
    with pytest.raises(ValueError):
        replay(data)


def test_explicit_unavailable_rank_clears_previous_rank():
    data = timing_data()
    data['timing'][2]['position'] = 0
    session = replay(data)
    assert session.timing[2]['position'] is None
    driver = session.frame_at(120000).frame.drivers['a']
    assert driver['position'] is None and driver['position_fresh']


@pytest.mark.parametrize('position, gap', [(None, None), (0, ''), (None, 'PIT')])
def test_provider_state_snapshot_unknown_fields_clear_previous_values(position, gap, tmp_path):
    data = timing_data()
    data['timing'] = [
        {'driver_id': 'a', 'session_time_s': 10, 'position': 1,
         'gap_to_leader_raw': '+1', 'interval_to_ahead_raw': '+0.5'},
        {'driver_id': 'a', 'session_time_s': 20, 'state_snapshot': True,
         'position': position, 'gap_to_leader_raw': gap, 'interval_to_ahead_raw': None}]
    session = replay(data)
    driver = session.frame_at(20000).frame.drivers['a']
    assert driver['position'] is None
    assert driver['gap_s'] is None and driver['gap_label'] is None
    assert driver['interval_s'] is None and driver['interval_label'] is None
    node = shutil.which('node')
    if not node:
        pytest.skip('Node.js unavailable')
    payload = tmp_path / 'snapshot.json'
    payload.write_text(json.dumps(session.browser_payload()), encoding='utf8')
    core = Path(__file__).resolve().parents[1] / 'streamlit_app/components/replay_component/clock.js'
    result = subprocess.run([node, '-e',
        "const r=require(process.argv[1]),p=require(process.argv[2]);if(!r.validatePayload(p))process.exit(1);console.log(JSON.stringify(r.frameAt(p,20000).drivers.find(d=>d.driver_id==='a')));",
        str(core), str(payload)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == driver


def test_leader_counter_does_not_label_interval_and_snapshot_flag_is_boolean():
    data = timing_data()
    data['timing'][1]['interval_to_ahead_raw'] = 'LAP 1'
    driver = replay(data).frame_at(10000).frame.drivers['b']
    assert driver['gap_label'] == 'Leader' and driver['interval_label'] is None
    data['timing'][1]['state_snapshot'] = 'yes'
    with pytest.raises(ValueError):
        replay(data)


def test_timing_python_browser_parity_and_strict_payload_validation(tmp_path):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node.js unavailable')
    session = replay(timing_data())
    clocks = [0, 10000, 62000, 120000, 130000, 130001, 140000, 240001, 260001, 300000]
    expected = [list(session.frame_at(t).frame.drivers.values()) for t in clocks]
    payload = tmp_path / 'timing.json'
    payload.write_text(json.dumps(session.browser_payload()), encoding='utf8')
    core = Path(__file__).resolve().parents[1] / 'streamlit_app/components/replay_component/clock.js'
    script = '''const assert=require('node:assert/strict'),r=require(process.argv[1]),p=require(process.argv[2]);
assert(r.validatePayload(p));
const bad=structuredClone(p);bad.timing.reverse();assert(!r.validatePayload(bad));
const invalid=structuredClone(p);invalid.timing[0].position=0;assert(!r.validatePayload(invalid));
console.log(JSON.stringify(CLOCKS.map(t=>r.frameAt(p,t).drivers)));'''.replace('CLOCKS', json.dumps(clocks))
    result = subprocess.run([node, '-e', script, str(core), str(payload)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == expected
