from dataclasses import replace
import pytest


def scenario():
    from ml.simulator import CompoundPace, DriverScenario, SimulationConfig
    pace = CompoundPace('MEDIUM', 90, .2, .5, 'user_assumption', 'explicit test assumption')
    drivers = (DriverScenario('A', 10, 0, 'MEDIUM', 10, (pace,)),
               DriverScenario('B', 10, 3, 'MEDIUM', 10, (replace(pace, pace_s=91),)))
    config = SimulationConfig(remaining_laps=5, samples=300, seed=42,
                              weather_delta_s=(0.,) * 5, weather_assumption='Dry constant pace assumed')
    return drivers, config


def test_seed_reproducible_frequencies_and_absolute_clock_units():
    from ml.simulator import _simulate_race
    drivers, config = scenario()
    result = _simulate_race(drivers, config)
    assert result == _simulate_race(drivers, config)
    assert result.calibrated is False
    assert all(sum(row.finish_frequencies) == pytest.approx(1) for row in result.drivers)
    assert all(sum(row.finish_frequencies[i] for row in result.drivers) == pytest.approx(1) for i in range(2))
    assert result.drivers[0].finish_time_median_s > 450
    assert result.drivers[0].trajectory_median_s[0] == pytest.approx(0)
    assert result.drivers[1].trajectory_median_s[0] == pytest.approx(3)
    assert 'uncalibrated' in result.interpretation.lower()
    assert result.limits


def test_pit_loss_applied_once_and_new_tyres_reset_after_pit_lap():
    from ml.simulator import _simulate_race, CompoundPace, PitLoss, PitStop
    drivers, config = scenario()
    medium = replace(drivers[0].paces[0], sigma_s=0)
    hard = CompoundPace('HARD', 95, .1, 0, 'user_assumption', 'explicit')
    stop = PitStop(11, 'HARD', PitLoss(net_loss_s=20))
    a = replace(drivers[0], paces=(medium, hard), pits=(stop,))
    result = _simulate_race((a,), replace(config, remaining_laps=2, weather_delta_s=(0., 0.)))
    assert result.drivers[0].finish_time_median_s == pytest.approx(90 + .2 * 11 + 20 + 95 + .1)
    assert PitLoss(stationary_s=3, lane_travel_delta_s=17).total_s == 20
    with pytest.raises(ValueError):
        PitLoss(net_loss_s=20, stationary_s=3, lane_travel_delta_s=17)


def test_rain_delta_is_explicit_and_positive_laps():
    from ml.simulator import _simulate_race
    drivers, config = scenario()
    drivers = tuple(replace(d, paces=tuple(replace(p, sigma_s=0) for p in d.paces)) for d in drivers)
    dry = _simulate_race(drivers, config)
    wet = _simulate_race(drivers, replace(config, weather_delta_s=(1, 2, 3, 4, 5)))
    assert wet.drivers[0].finish_time_median_s - dry.drivers[0].finish_time_median_s == pytest.approx(15)
    with pytest.raises(ValueError):
        _simulate_race(drivers, replace(config, weather_delta_s=(-100.,) * 5))


def test_missing_or_invalid_inputs_rejected():
    from ml.simulator import _simulate_race, PitStop, PitLoss
    drivers, config = scenario()
    bad_drivers = [(), drivers * 12, (replace(drivers[0], paces=()),),
                   (replace(drivers[0], gap_s=None),),
                   (replace(drivers[0], paces=(replace(drivers[0].paces[0], quality='unknown'),)),),
                   (replace(drivers[0], pits=(PitStop(11, 'SOFT', PitLoss(net_loss_s=20)),)),),
                   (drivers[0], replace(drivers[1], completed_lap=9)),
                   (replace(drivers[0], completed_lap=99),)]
    for invalid in bad_drivers:
        with pytest.raises(ValueError):
            _simulate_race(invalid, config)
    for invalid in [replace(config, samples=2001), replace(config, remaining_laps=101),
                    replace(config, weather_delta_s=()), replace(config, seed=True)]:
        with pytest.raises(ValueError):
            _simulate_race(drivers, invalid)


def test_guard_rejects_unauthorized_and_duplicate_dispatch(tmp_path, monkeypatch):
    import ml.simulator as simulator
    from security.access import Principal, AccessDenied
    from security.limits import SQLiteLimiter, LimitRejected
    drivers, config = scenario()
    limiter = SQLiteLimiter(tmp_path / 'limits.sqlite3')
    with pytest.raises(AccessDenied):
        simulator.simulate_race(drivers, config, principal=Principal('v', 'viewer'), limiter=limiter, request_id='v1')
    principal = Principal('a', 'ai_user')
    simulator.simulate_race(drivers, config, principal=principal, limiter=limiter, request_id='a1')
    monkeypatch.setattr(simulator, '_simulate_race', lambda *a, **kw: pytest.fail('duplicate dispatch'))
    with pytest.raises(LimitRejected, match='request_already_submitted'):
        simulator.simulate_race(drivers, config, principal=principal, limiter=limiter, request_id='a1')
    with pytest.raises(LimitRejected, match='request_conflict'):
        simulator.simulate_race(drivers, replace(config, seed=3), principal=principal, limiter=limiter, request_id='a1')


def test_public_mode_cannot_use_local_identity(tmp_path):
    from ml.simulator import simulate_race
    from security.access import Principal
    from security.limits import SQLiteLimiter, LimitRejected
    drivers, config = scenario()
    with pytest.raises(LimitRejected, match='public_disabled'):
        simulate_race(drivers, config, principal=Principal('a', 'ai_user'),
                      limiter=SQLiteLimiter(tmp_path / 'limits.sqlite3'), request_id='p', public=True)


def test_bootstrap_parameter_uncertainty_is_shared_across_future_laps():
    from ml.simulator import _simulate_race
    drivers, config = scenario()
    pace = replace(drivers[0].paces[0], sigma_s=0, parameter_draws=((89., .2), (91., .2)))
    result = _simulate_race((replace(drivers[0], paces=(pace,)),), config)
    row = result.drivers[0]
    assert row.finish_time_upper_s - row.finish_time_lower_s == pytest.approx(10)


def test_queued_request_does_not_dispatch_and_deadline_is_enforced(tmp_path, monkeypatch):
    import ml.simulator as simulator
    from security.access import Principal
    from security.limits import SQLiteLimiter, LimitPolicy, LimitRejected
    drivers, config = scenario()
    now = [1000.]
    limiter = SQLiteLimiter(tmp_path / 'limits.sqlite3', clock=lambda: now[0],
                            policy=LimitPolicy(global_active=1))
    limiter.reserve(Principal('other', 'ai_user'), 'other', 'other', action='scenario')
    monkeypatch.setattr(simulator, '_simulate_race', lambda *a, **kw: pytest.fail('queued dispatch'))
    with pytest.raises(LimitRejected, match='busy'):
        simulator.simulate_race(drivers, config, principal=Principal('a', 'ai_user'), limiter=limiter, request_id='queued')
    now[0] += 50
    def expired(*a, checkpoint, **kw):
        now[0] += 50
        checkpoint()
    monkeypatch.setattr(simulator, '_simulate_race', expired)
    with pytest.raises(LimitRejected, match='deadline_exceeded'):
        simulator.simulate_race(drivers, config, principal=Principal('b', 'ai_user'), limiter=limiter, request_id='expired')


def test_strategy_ui_prefix_missing_quality_and_manual_submit(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from streamlit_app import services
    from data.replay import ReplaySession
    from test_replay import session_data
    import streamlit_app.components.strategy_ui as ui
    from security.access import Principal
    session = ReplaySession.from_mapping(session_data())
    monkeypatch.setattr(ui, '_sources', lambda context: ([session], 0))
    monkeypatch.setattr(services, 'LOCAL', tmp_path)
    monkeypatch.setattr(services, 'local_identity', lambda: Principal('test', 'ai_user'))
    app = AppTest.from_string("from streamlit_app.components.strategy_ui import render_strategy\nrender_strategy({'season':2025,'round_number':1,'session':'R'})").run()
    assert not app.exception
    assert not app.dataframe
    app.number_input(key='strategy_clock').set_value(61.).run()
    assert not app.exception
    assert app.dataframe[0].value.iloc[0]['observed_laps'] == 1
    assert any('0 strict clean laps' in info.value for info in app.info)
    app.button_group[0].set_value('Scenario').run()
    app.multiselect(key='strategy_participants').set_value(['a']).run()
    assert not app.exception
    assert not (tmp_path / 'limits.sqlite3').exists()
    values = {'Shared completed race lap at decision boundary': 1, 'Remaining race laps': 2,
              'Simulation samples': 10, 'a gap to decision leader (s; one participant must be 0)': 0.,
              'a current tyre age (laps)': 1., 'a expected pace at tyre age zero (s)': 90.,
              'a pace change per tyre-age lap (s/lap)': .2, 'a lap variability standard deviation (s)': .5}
    for element in app.number_input:
        if element.label in values:
            element.set_value(values[element.label])
    next(element for element in app.selectbox if element.label == 'a current compound').select('MEDIUM')
    app.text_input[0].set_value('0,0')
    app.text_input[1].set_value('Explicit dry pace and gap assumptions for a synthetic test')
    next(element for element in app.checkbox if element.label.startswith('I explicitly assume')).check()
    next(element for element in app.checkbox if element.label.startswith('Compare with an assumed')).check()
    next(button for button in app.button if button.label == 'Run scenario').click().run()
    assert not app.exception
    assert not app.error
    saved = app.session_state['strategy_result']['result']
    assert saved.calibrated is False
    assert app.session_state['strategy_result']['reference'].calibrated is False
    import ml.simulator as simulator
    monkeypatch.setattr(simulator, '_simulate_race', lambda *a, **kw: pytest.fail('render repeated simulation'))
    app.run()
    assert not app.exception
    assert app.session_state['strategy_result']['result'] == saved


def test_strategy_ui_handles_unknown_observed_compound(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from data.replay import ReplaySession
    from test_replay import session_data
    import streamlit_app.components.strategy_ui as ui
    data = session_data()
    data['laps'][0]['compound'] = None
    session = ReplaySession.from_mapping(data)
    monkeypatch.setattr(ui, '_sources', lambda context: ([session], 0))
    app = AppTest.from_string("from streamlit_app.components.strategy_ui import render_strategy\nrender_strategy({'season':2025,'round_number':1,'session':'R'})").run()
    app.number_input(key='strategy_clock').set_value(61.).run()
    assert not app.exception
    assert app.dataframe[0].value.iloc[0]['compound'] is None
