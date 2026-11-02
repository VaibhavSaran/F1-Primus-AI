"""Presentation preserves official timing values rather than deriving race order."""
from pathlib import Path
import json
import shutil
import subprocess

import pytest

from streamlit_app.components.replay_ui import _standings_table


def driver(**changes):
    return dict(driver_id='b', position=7, position_fresh=True, gap_label='+1 lap',
                interval_label='+2.300 s', completed_laps=2, last_lap_number=3,
                last_lap_time_s=88.123, compound='HARD', **changes)


def test_official_subset_keeps_absolute_rank_and_official_interval():
    title, caption, rows = _standings_table('timing_stream', [driver()], {'b': 'Full Driver Name'})
    assert title == 'Race standings'
    assert rows == [{'Position': 7, 'Driver': 'Full Driver Name', 'Gap to leader': '+1 lap',
                     'Interval': '+2.300 s', 'Last lap (s)': 88.123, 'Tyres': 'HARD'}]
    assert 'last reported values' in caption
    assert 'snapshots older than two minutes show —' in caption
    assert 'car ahead in the full field' in caption


def test_missing_timing_stays_unknown_and_unconfirmed_leader_is_not_labelled():
    row = driver()
    row.update(position=None, position_fresh=False, gap_label='Leader', interval_label=None)
    _, _, rows = _standings_table('timing_stream', [row], {})
    assert rows[0]['Position'] == '—'
    assert rows[0]['Gap to leader'] == '—'
    assert rows[0]['Interval'] == '—'
    assert 'DNF' not in str(rows)
    row.update(gap_label='+3.100 s')
    assert _standings_table('timing_stream', [row], {})[2][0]['Gap to leader'] == '+3.100 s'


def test_leader_label_never_appears_as_an_interval():
    row = driver()
    row.update(position=1, gap_label='Leader', interval_label='Leader')
    result = _standings_table('timing_stream', [row], {})[2][0]
    assert result['Gap to leader'] == 'Leader'
    assert result['Interval'] == '—'


def test_checkpoint_has_position_but_no_claimed_live_gaps():
    title, caption, rows = _standings_table('lap_end_position', [driver()], {})
    assert title == 'Lap-end classification checkpoints'
    assert rows[0]['Position'] == 7
    assert 'Gap to leader' not in rows[0]
    assert 'checkpoint' in caption


def test_unrelated_drs_and_final_status_do_not_change_standings():
    row = driver()
    expected = _standings_table('timing_stream', [row], {})
    row.update(drs=12, drs_available=True, status='Retired', final_position=1)
    assert _standings_table('timing_stream', [row], {}) == expected


def test_practice_and_sources_without_timing_remain_lap_completion():
    title, caption, rows = _standings_table('derived_timed_lap_record_order', [driver()], {})
    assert title == 'Session lap completion'
    assert 'race order' in caption
    assert rows[0]['Timed laps observed'] == 2
    assert 'Position' not in rows[0]
    assert 'Gap to leader' not in rows[0]
    assert rows[0]['Order'] == 1
    assert _standings_table('derived_timed_lap_record_order', [driver(),driver()], {})[2][1]['Order'] == 2


@pytest.mark.parametrize('session_id, basis, expected', [
    ('R', 'timing_stream', 'Race standings'),
    ('S', 'timing_stream', 'Race standings'),
    ('R', 'lap_end_position', 'Lap-end classification checkpoints'),
    ('FP1', 'timing_stream', 'Session lap completion'),
    ('Q', 'timing_stream', 'Session lap completion'),
])
def test_native_replay_uses_observed_timing_without_final_classification(
        tmp_path, monkeypatch, session_id, basis, expected):
    from streamlit.testing.v1 import AppTest
    from streamlit_app import services
    from test_replay import session_data
    data = session_data()
    data.update(session_id=session_id, timing_basis=basis, timing=[
        {'driver_id': 'a', 'session_time_s': 10, 'position': 7,
         'gap_to_leader_raw': '+1 Lap', 'interval_to_ahead_raw': '+2.300'},
        {'driver_id': 'b', 'session_time_s': 10, 'position': 1, 'gap_to_leader_raw': 'LAP 1'},
        {'driver_id': 'a', 'session_time_s': 300, 'position': 1},
    ])
    directory = tmp_path / 'sessions'
    directory.mkdir()
    (directory / ('session-' + '5' * 64 + '.json')).write_text(json.dumps(data), encoding='utf-8')
    monkeypatch.setattr(services, 'LOCAL', tmp_path)
    script = tmp_path / 'app.py'
    script.write_text('from streamlit_app.components.replay_ui import render_replay\n'
                      f'render_replay({{"season":2025,"round_number":1,"session":"{session_id}","reduced_motion":True}})',
                      encoding='utf-8')
    app = AppTest.from_file(script, default_timeout=10).run()
    app.number_input(key='replay_native_seconds').set_value(62.0)
    next(button for button in app.button if button.label == 'Jump to time').click()
    app.run()
    assert not app.exception
    assert expected in [item.value for item in app.subheader]
    table = app.dataframe[0].value
    if expected == 'Race standings':
        assert list(table['Position']) == ['1', '7']
        assert list(table['Gap to leader']) == ['Leader', '+1 lap']
        assert table.loc[1, 'Interval'] == '+2.300s'
        assert not any('running positions and gaps are unavailable' in item.value for item in app.caption)
    elif basis == 'lap_end_position':
        assert list(table['Position']) == ['1', '7']
        assert 'Gap to leader' not in table
    else:
        assert 'Timed laps observed' in table and 'Position' not in table
    assert 'Retired' not in table.to_string() and 'DNF' not in table.to_string()


def test_browser_standings_columns_selection_and_version_reset(tmp_path):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node.js unavailable; real-browser gate still required')
    script = tmp_path / 'standings.cjs'
    script.write_text(r'''
const assert = require('node:assert/strict');
const fs = require('node:fs'), vm = require('node:vm');
function element() { return {textContent:'', children:[], value:'1', width:640, height:320,
  clientWidth:960, clientHeight:480,
  append(...nodes) {this.children.push(...nodes)}, replaceChildren(...nodes) {this.children=nodes},
  setAttribute(k,v) {this[k]=v}, getContext() {return {clearRect(){}, fillText(){}}}}; }
const nodes = new Map(), $ = id => {if(!nodes.has(id))nodes.set(id,element());return nodes.get(id)};
const events = {}, sent = [], parent = {postMessage(m){sent.push(m)}};
global.document = {getElementById:$, createElement:element, referrer:'https://host.test',
  documentElement:{scrollHeight:700}, body:{}, addEventListener(){}};
global.location = {href:'https://host.test/replay'};
global.window = {parent, addEventListener(name, fn){events[name]=fn}};
global.matchMedia = () => ({matches:false, addEventListener(){}});
let resize;
global.ResizeObserver = class {constructor(fn){resize=fn} observe(){}};
global.requestAnimationFrame = () => {};
let drivers = [{driver_id:'b', abbreviation:'BBB', position:7, position_fresh:true,
  gap_label:'+1 lap', interval_label:'+2.300 s', completed_laps:2, last_lap_number:3,
  last_lap_time_s:88.123, compound:'HARD', coordinates:null}];
global.ReplayClock = {acceptRender:()=>true, validatePayload:()=>true, trackBounds:()=>null,
  formatClock:()=> '0:01:00', frameAt:()=>({clock_ms:60000, drivers, markers:[],weather:{}})};
vm.runInThisContext(fs.readFileSync(process.argv[2],'utf8'));
let payload = {session_id:'race', frame_version:'v1', duration_ms:60000, entries:drivers,
  positions:[], driver_labels:{b:'Full Driver Name'}, timing_basis:'timing_stream'};
function render(revision=1, selected=['b']) {events.message({data:{args:{payload,
  committed_clock_ms:60000, last_sequence:0,commit_revision:revision,selected_driver_ids:selected}}});}
render();
assert.equal($('tower-title').textContent,'Race standings');
assert($('tower-note').textContent.includes('snapshots older than two minutes show —'));
assert($('tower-note').textContent.includes('car ahead in the full field'));
assert.deepEqual($('tower-head').children.map(n=>n.textContent),
  ['Position','Driver','Gap to leader','Interval','Last lap (s)','Tyres']);
let row=$('tower').children[0];
assert.deepEqual(row.children.map(n=>n.children.length?n.children[0].textContent:n.textContent),
  ['7','Full Driver Name','+1 lap','+2.300 s','88.123','HARD']);
row.children[1].children[0].onclick();
assert.deepEqual(sent.at(-1).value.driver_ids,['b']);
assert.equal(sent.at(-1).value.event,'selection');
assert.equal(sent.at(-1).value.frame_version,'v1');
drivers=[{...drivers[0],position:1,gap_label:'Leader',interval_label:'Leader'}];render(2);
assert.equal($('tower').children[0].children[3].textContent,'—');
drivers=[{...drivers[0],position:null,position_fresh:false,gap_label:'Leader',interval_label:null}];
payload={...payload,frame_version:'v2'}; render(2);
row=$('tower').children[0]; assert.equal(row.children[0].textContent,'—');
assert.equal(row.children[2].textContent,'—');
row.children[1].children[0].onclick(); assert.equal(sent.at(-1).value.frame_version,'v2');
payload={...payload,timing_basis:'lap_end_position'};render(3);
assert.equal($('tower-title').textContent,'Lap-end classification checkpoints');
assert(!$('tower-head').children.some(n=>n.textContent==='Gap to leader'));
payload={...payload,timing_basis:'derived_timed_lap_record_order'};render(4);
assert.equal($('tower-title').textContent,'Session lap completion');
assert(!$('tower-head').children.some(n=>n.textContent==='Position'));
assert.equal($('tower-head').children[0].textContent,'Order');
assert.equal($('tower').children[0].children[0].textContent,'1');
drivers=[{...drivers[0],driver_id:'a',position:1},{...drivers[0],position:2}];
payload={...payload,timing_basis:'timing_stream',entries:drivers};render(5);
let names=()=> $('tower').children.map(row=>row.children[1].children[0].textContent);
assert.deepEqual(names(),['BBB','Full Driver Name']);
drivers=[{...drivers[1],position:1},{...drivers[0],position:2}];render(6);
assert.deepEqual(names(),['Full Driver Name','BBB']);
assert.deepEqual($('tower').children.map(row=>row.children[0].textContent),['1','2']);
assert.equal($('tower').children[0],row);
assert.equal($('track').width,960);
assert.equal($('track').height,480);
$('track').clientWidth=720;$('track').clientHeight=360;resize();
assert.equal($('track').width,720);
assert.equal($('track').height,360);
''', encoding='utf-8')
    component = Path(__file__).resolve().parents[1] / 'streamlit_app/components/replay_component/replay.js'
    result = subprocess.run([node, str(script), str(component)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
