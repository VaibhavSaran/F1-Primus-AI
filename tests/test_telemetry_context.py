"""Downloaded-weekend selectors must follow the actual saved session inventory."""
from streamlit.testing.v1 import AppTest
from streamlit_app import archive_services
from test_streamlit import offline_ui


def test_published_weekend_selectors_and_context_reset(monkeypatch):
    def row(number, code):
        return dict(season=2026, round_number=number, event_name='Chinese Grand Prix' if number == 2 else 'Japanese Grand Prix',
                    local_race_date='2026-03-15' if number == 2 else '2026-03-29', session_code=code,
                    full_name={'FP1': 'Practice 1', 'S': 'Sprint', 'SQ': 'Sprint Qualifying', 'R': 'Race'}[code],
                    session_id=f'2026:{number}:{code}', version_id=f'version-{number}-{code}', scheduled_start_utc_ns=1_700_000_000_000_000_000)
    monkeypatch.setattr(archive_services, 'inventory', lambda: [row(2, c) for c in ('FP1', 'SQ', 'S', 'R')] +
                        [row(3, c) for c in ('FP1', 'R')])
    app = AppTest.from_string('from streamlit_app.components.context import render_context\nimport streamlit as st\nst.session_state["result"] = render_context()').run()
    assert not app.exception
    assert app.selectbox(key='session').options == ['Grand Prix Race', 'Sprint Qualifying', 'Sprint', 'Free Practice 1']
    app.selectbox(key='session').select('S').run()
    app.session_state['pilot_telemetry_loaded'] = True
    app.selectbox(key='round_number').select(3).run()
    assert not app.exception
    assert app.selectbox(key='session').options == ['Grand Prix Race', 'Free Practice 1']
    assert app.session_state['result']['session'] is None
    app.selectbox(key='session').select('R').run()
    assert app.session_state['result']['session'] == 'R'
    assert 'pilot_telemetry_loaded' not in app.session_state
    assert all('Grand Prix' in name and '2026-' in name for name in app.selectbox(key='round_number').options)


def test_season_change_clears_race_and_session(monkeypatch):
    rows = [dict(season=year, round_number=1, event_name='Test Grand Prix', local_race_date=f'{year}-03-01',
                 session_code='R', full_name='Race', session_id=f'{year}:1:R', version_id=f'v{year}',
                 scheduled_start_utc_ns=1_700_000_000_000_000_000) for year in (2020, 2026)]
    monkeypatch.setattr(archive_services, 'inventory', lambda: rows)
    app = AppTest.from_string('from streamlit_app.components.context import render_context\nimport streamlit as st\nst.session_state["result"] = render_context()').run()
    app.selectbox(key='season').select(2020).run()
    assert not app.exception
    assert app.selectbox(key='round_number').value is None
    assert app.selectbox(key='session').value is None
    assert app.selectbox(key='session').disabled
    app.selectbox(key='round_number').select(1).run()
    assert app.selectbox(key='session').value is None
    assert not app.selectbox(key='session').disabled
    app.selectbox(key='session').select('R').run()
    assert app.session_state['result']['archive_session_id'] == '2020:1:R'
