"""Readable view and telemetry quality regressions, with no provider calls."""
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from streamlit_app.archive_views import lap_time, observed_car, pace_summary
from test_archive_reader import saved


def test_telemetry_discards_parser_fills_but_preserves_observed_zero_speed():
    car = pd.DataFrame({'session_time_s': [3., 1., 2., None],
                        'slice_time_ns': [3, 1, None, 4],
                        'Time': [pd.Timedelta(seconds=3), pd.Timedelta(seconds=1), pd.NaT, pd.Timedelta(seconds=4)],
                        'speed_kmh': [200., 0., 0., 250.]})
    actual = observed_car(car)
    assert actual.session_time_s.tolist() == [1., 3.]
    assert actual.speed_kmh.tolist() == [0., 200.]


def test_lap_display_rounding_and_unknown():
    assert lap_time(89.1234) == '1:29.123'
    assert lap_time(59.9999) == '1:00.000'
    assert lap_time(None) == '—'


def test_practice_pace_includes_first_timed_lap_and_zero_running_driver():
    data = {'entries': pd.DataFrame({'driver_id': ['a', 'b']}),
            'laps': pd.DataFrame({'driver_id': ['a', 'a', 'a'], 'lap_time_s': [80., 70., 60.],
                                  'deleted': [0, 1, 0], 'fastf1_generated': [0, 0, 1]})}
    summary = pace_summary(data).set_index('driver_id')
    assert summary.loc['a', 'Best timed lap'] == '1:20.000'
    assert summary.loc['b', 'Recorded laps'] == 0
    assert summary.loc['b', 'Best timed lap'] == '—'


def render_saved_view(view, data, monkeypatch):
    from streamlit_app import archive_services
    monkeypatch.setattr(archive_services, 'tables', lambda context: data)
    monkeypatch.setattr(archive_services, 'dense', lambda *args: pd.DataFrame({
        'session_time_s': [20., 30.], 'slice_time_ns': [1, None],
        'speed_kmh': [100., 0.], 'brake_applied': pd.array([True, False], dtype='boolean')}))
    return AppTest.from_string(f'from streamlit_app.archive_views import {view}\n{view}({{"archive_session_id": "fixture", "archive_version_id": "v"}})').run()


@pytest.mark.parametrize('code', ['R', 'Q', 'FP1'])
def test_classification_uses_session_type_and_readable_names(saved, monkeypatch, code):
    from data.archive.reader import ArchiveReader
    data = ArchiveReader(saved[0].root).tables('2025:1:FP1')
    data['meta'].update(session_code=code, full_name='Fixture')
    result = {key: None for key in data['results'].columns}
    data['results'] = pd.DataFrame([{**result, 'driver_id': 'a', 'finishing_position': 1, 'q1_ns': 80_000_000_000}])
    app = render_saved_view('overview', data, monkeypatch)
    assert not app.exception
    assert 'Driver' in app.dataframe[0].value
    assert app.dataframe[0].value.iloc[0]['Driver'] == 'A Driver'
    if code == 'Q':
        assert app.dataframe[0].value.iloc[0]['Q1'] == '1:20.000'


def test_telemetry_persists_across_filter_rerun(saved, monkeypatch):
    from data.archive.reader import ArchiveReader
    data = ArchiveReader(saved[0].root).tables('2025:1:FP1')
    app = render_saved_view('compare', data, monkeypatch)
    assert not app.exception
    assert any('1 recorded samples' in item.value for item in app.caption)
    app.checkbox[0].uncheck().run()
    assert not app.exception
    assert any('1 recorded samples' in item.value for item in app.caption)


def test_weather_complete_channels_and_control_use_readable_columns(saved, monkeypatch):
    from data.archive.reader import ArchiveReader
    data = ArchiveReader(saved[0].root).tables('2025:1:FP1')
    weather = data['weather']
    for key, value in {'air_temperature_c': 20., 'track_temperature_c': 30.,
                       'wind_speed_mps': 3., 'wind_direction_deg': 180.,
                       'pressure_hpa': 1010., 'rainfall_flag': 0}.items():
        weather[key] = value
    data['track_status'] = pd.DataFrame({'session_time_s': [10.], 'message_raw': ['All clear']})
    data['race_control'] = pd.DataFrame({'session_time_s': [12.], 'message_text': ['GREEN LIGHT'],
                                         'driver_id': ['a'], 'lap_number': [1], 'sector': [1]})
    app = render_saved_view('weather', data, monkeypatch)
    assert not app.exception
    assert len(app.get('plotly_chart')) == 6
    assert any('Message' in frame.value.columns for frame in app.dataframe)
    assert not any('_json' in str(column) for frame in app.dataframe for column in frame.value.columns)


def test_telemetry_excludes_explicit_parser_generated_quality_flag():
    car = pd.DataFrame({'session_time_s': [1., 2.], 'slice_time_ns': [1, 2],
                        'quality_flags': ['["fastf1_parser_generated_fill"]', '[]']})
    assert observed_car(car).session_time_s.tolist() == [2.]


def test_roster_excludes_placeholders_but_retains_unknown_real_participants():
    from streamlit_app.archive_views import participant_entries
    data = {'entries': pd.DataFrame({'driver_id': ['known', 'unknown-real', 'unused-placeholder']}),
            'laps': pd.DataFrame({'driver_id': ['known', 'unknown-real']}),
            'results': pd.DataFrame({'driver_id': ['known']})}
    assert participant_entries(data).driver_id.tolist() == ['known', 'unknown-real']


def test_roster_ignores_empty_placeholder_result_and_keeps_named_nonstarter():
    from streamlit_app.archive_views import participant_entries
    data = {'entries': pd.DataFrame({'driver_id': ['racer', 'nonstarter', 'placeholder'],
                                     'display_name': ['A Driver', 'B Driver', 'B Driver']}),
            'laps': pd.DataFrame({'driver_id': ['racer']}),
            'results': pd.DataFrame({'driver_id': ['racer', 'nonstarter', 'placeholder'],
                                     'finishing_position': [1., None, None],
                                     'laps_completed': [10., 0., None],
                                     'status_raw': ['Finished', 'Did not start', None]})}
    assert participant_entries(data).driver_id.tolist() == ['racer', 'nonstarter']


def test_segmented_chart_keeps_parser_gaps_and_observed_zero_speed():
    from streamlit_app.archive_views import segmented_car
    car = pd.DataFrame({'session_time_s': [1., 1.2, 1.4, 10.],
                        'slice_time_ns': [1, None, 3, 4], 'speed_kmh': [0., 0., 20., 100.]})
    plotted = segmented_car(car)
    assert plotted.session_time_s.tolist() == [1., 1.2, 1.4, 10., 10.]
    assert plotted.speed_kmh.iloc[0] == 0.
    assert pd.isna(plotted.speed_kmh.iloc[1])
    assert pd.isna(plotted.speed_kmh.iloc[3])
    assert plotted.speed_kmh.iloc[-1] == 100.


def test_compare_all_drivers_and_recorded_wet_compounds_survive_clean_filter(saved, monkeypatch):
    from data.archive.reader import ArchiveReader
    data = ArchiveReader(saved[0].root).tables('2025:1:FP1')
    entries, laps = data['entries'], data['laps']
    data['entries'] = pd.concat([entries.assign(driver_id=f'd{i}', display_name=f'Driver {i}') for i in range(6)], ignore_index=True)
    data['laps'] = pd.concat([laps.assign(driver_id=f'd{i}') for i in range(6)], ignore_index=True)
    wet = data['laps'].iloc[[0]].assign(driver_id='d0', compound_raw='WET', provider_is_accurate=0)
    intermediate = wet.assign(compound_raw='INTERMEDIATE')
    data['laps'] = pd.concat([data['laps'], wet, intermediate], ignore_index=True)
    app = render_saved_view('compare', data, monkeypatch)
    assert not app.exception
    assert {'WET', 'INTERMEDIATE'} <= set(app.multiselect[1].options)
    app.multiselect[0].set_value([f'd{i}' for i in range(6)]).run()
    assert not app.exception
    assert len(app.multiselect[0].value) == 6
    assert {'WET', 'INTERMEDIATE'} <= set(app.multiselect[1].options)


def test_untimed_outlap_uses_recorded_pit_exit_and_never_invents_lap_time():
    from streamlit_app.archive_views import telemetry_lap_label, telemetry_window
    lap = pd.Series({'lap_number': 2, 'lap_time_s': pd.NA,
                     'lap_start_session_s': 1138.180, 'lap_end_session_s': 2362.099,
                     'pit_out_session_s': 2258.358, 'pit_in_session_s': pd.NA})
    assert telemetry_window(lap) == (2258.358, 2362.099)
    label = telemetry_lap_label(lap)
    assert 'untimed' in label and 'out lap' in label
    assert '1:43' not in label
    lap['pit_out_session_s'] = pd.NA
    assert telemetry_window(lap) == (1138.180, 2362.099)


def test_combined_controls_have_explicit_colours_binary_scale_and_gaps():
    from streamlit_app.archive_views import throttle_brake_figure
    car = pd.DataFrame({'Lap elapsed (s)': [0., 1., 2.],
                        'throttle_pct_raw': [0., None, 100.],
                        'brake_applied': pd.array([True, None, False], dtype='boolean')})
    figure = throttle_brake_figure(car)
    throttle, brake = figure.data
    assert throttle.line.color == '#27ae60'
    assert brake.line.color == '#ef5350'
    assert brake.y[0] == 100 and brake.y[2] == 0
    assert pd.isna(brake.y[1]) and pd.isna(throttle.y[1])
    assert brake.line.shape == 'hv'
    assert throttle.connectgaps is False and brake.connectgaps is False


def test_session_clock_handles_hours_zero_and_unknown():
    from streamlit_app.archive_views import session_clock
    assert session_clock(0) == '00:00:00'
    assert session_clock(3661.2) == '01:01:01'
    assert session_clock(None) == '—'


def test_tyre_age_chart_focuses_one_driver(saved, monkeypatch):
    import json
    from data.archive.reader import ArchiveReader
    data = ArchiveReader(saved[0].root).tables('2025:1:FP1')
    clean = data['laps'].assign(lap_number=2, provider_is_accurate=1, deleted=0,
                              fastf1_generated=0, track_status_raw='1',
                              pit_in_clock_ns=pd.NA, pit_out_clock_ns=pd.NA)
    data['laps'] = pd.concat([clean.assign(driver_id='a', compound_raw='SOFT'),
                              clean.assign(driver_id='b', compound_raw='WET')], ignore_index=True)
    app = render_saved_view('strategy', data, monkeypatch)
    assert not app.exception
    focused = app.selectbox(key='pilot_tyre_age_driver')
    assert focused.options == ['A Driver', 'B Driver']
    assert [trace['name'] for trace in json.loads(app.get('plotly_chart')[-1].proto.spec)['data']] == ['SOFT']
    focused.set_value('b').run()
    assert not app.exception
    assert [trace['name'] for trace in json.loads(app.get('plotly_chart')[-1].proto.spec)['data']] == ['WET']
