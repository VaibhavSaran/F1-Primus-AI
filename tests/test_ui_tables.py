import pandas as pd
from streamlit_app.components import tables


def test_tables_center_cells_without_stretching_columns(monkeypatch):
    captured = {}
    def render(frame, **kwargs):
        captured.update(frame=frame, **kwargs)
    monkeypatch.setattr(tables.st, 'dataframe', render)
    tables.display_table(pd.DataFrame({'Driver': ['Roy Nissany'], 'Laps': [25]}))
    assert captured['width'] == 'content'
    assert all(config['alignment'] == 'center' for config in captured['column_config'].values())
    assert captured['column_config']['Laps']['width'] < captured['column_config']['Driver']['width']
    assert captured['frame'].iloc[0]['Laps'] == 25
    assert captured['column_config']['Laps']['type_config'] == {'type': 'text'}
