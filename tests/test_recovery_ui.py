"""UTC-only retained telemetry must not be presented as absent observations."""
import pandas as pd
import pytest
from test_archive_reader import saved
from test_archive_ui import pilot
from test_streamlit import offline_ui, page

MESSAGE = ('Car telemetry is saved, but it cannot be aligned to laps because '
           'the session timing reference is unavailable.')


@pytest.mark.parametrize('slug', ['compare', 'replay'])
@pytest.mark.parametrize('state', ['utc_only', 'empty', 'known_reference'])
def test_unaligned_saved_car_message(pilot, monkeypatch, slug, state):
    from streamlit_app import archive_services
    original = archive_services.tables
    def tables(context):
        data = original(context)
        data = {**data, 'meta': {**data['meta'], 't0_utc_ns': None if state != 'known_reference' else 1}}
        return data
    monkeypatch.setattr(archive_services, 'tables', tables)
    car = pd.DataFrame({'session_time_s': pd.array([None], dtype='Float64'),
                        'speed_kmh': [100.], 'timestamp_utc_ns': [1700000000000000000]})
    if state == 'empty':
        car = car.iloc[:0]
    monkeypatch.setattr(archive_services, 'dense', lambda *args, **kwargs: car)
    app = page(slug)
    if slug == 'replay':
        app.session_state['replay_analysis'] = 'Telemetry'
        app.run()
        next(button for button in app.button if button.label == 'Load latest completed lap telemetry').click().run()
    assert not app.exception
    assert not app.error
    assert not app.get('code') and not app.get('json')
    messages = [item.value for item in app.info]
    if state == 'utc_only':
        assert MESSAGE in messages
        assert not any('car samples are available' in message for message in messages)
    else:
        assert MESSAGE not in messages
        assert any('car samples are available' in message for message in messages)
