"""Published archive screens render without provider access."""
import pytest
from test_archive_reader import saved
from test_streamlit import offline_ui, page


@pytest.fixture
def pilot(saved):
    archive, _, _, _ = saved
    archive.root.rename(archive.root.with_name('f1_2025_2026'))


@pytest.mark.parametrize('slug', ['overview', 'compare', 'replay', 'strategy', 'weather', 'settings'])
def test_published_archive_pages_offline(pilot, slug):
    app = page(slug)
    assert not app.exception
    assert app.session_state['workspace_context']['archive_session_id'] == '2025:1:FP1'
    assert app.selectbox(key='season').options == ['2025']
    assert len(app.selectbox(key='session').options) == 1
    if slug == 'compare':
        app.checkbox[0].uncheck().run()
        assert not app.exception
    assert app.dataframe


def test_clean_laps_excludes_unknown_and_bad_evidence():
    import pandas as pd
    from streamlit_app.archive_views import clean_laps
    row = dict(lap_time_s=80., lap_number=2, provider_is_accurate=1, deleted=0,
               fastf1_generated=0, track_status_raw='1', pit_in_clock_ns=None, pit_out_clock_ns=None)
    rows = [row, {**row, 'deleted': None}, {**row, 'fastf1_generated': 1},
            {**row, 'pit_in_clock_ns': 1}, {**row, 'provider_is_accurate': 0},
            {**row, 'track_status_raw': '2'}, {**row, 'lap_time_s': None}]
    assert clean_laps(pd.DataFrame(rows)).index.tolist() == [0]


def test_replay_analysis_selection_and_deselection(pilot):
    app = page('replay')
    app.session_state['replay_analysis'] = 'Sectors'
    app.run()
    assert not app.exception
    assert any('Sector 1 (s)' in frame.value.columns for frame in app.dataframe)
    app.session_state['replay_analysis'] = None
    app.run()
    assert not app.exception
    assert not any('None samples' in item.value for item in app.info)


def test_replay_uses_names_and_hides_entries_without_observed_evidence(pilot):
    app = page('replay')
    assert not app.exception
    assert app.multiselect(key='pilot_replay').options == ['A Driver']
    assert app.selectbox(key='replay_native_driver').options == ['A Driver']
    assert not app.json
    assert any(expander.label == 'Jump to time or lap' for expander in app.expander)


def test_replay_telemetry_breaks_lines_at_parser_fills(pilot, monkeypatch):
    import pandas as pd
    from streamlit_app import archive_services
    from streamlit_app.components import replay_ui
    car = pd.DataFrame({'session_time_s': [20., 21., 22.], 'speed_kmh': [100., 0., 110.],
                        'quality_flags': ['[]', '["fastf1_parser_generated_fill"]', '[]']})
    monkeypatch.setattr(archive_services, 'dense', lambda *args, **kwargs: car)
    charts = []
    monkeypatch.setattr(replay_ui.st, 'plotly_chart', lambda figure, **kwargs: charts.append(figure))
    app = page('replay')
    app.session_state['replay_analysis'] = 'Telemetry'
    app.run()
    next(button for button in app.button if button.label == 'Load latest completed lap telemetry').click().run()
    assert not app.exception
    assert charts and charts[-1].data[0].connectgaps is False
    assert list(charts[-1].data[0].x) == [20., 21., 22.]
    assert pd.isna(charts[-1].data[0].y[1])
