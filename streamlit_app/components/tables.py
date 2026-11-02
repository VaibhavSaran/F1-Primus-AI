"""Consistent compact read-only tables using the installed native grid."""
import pandas as pd
import streamlit as st


def display_table(data, **kwargs):
    frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    config = {}
    for name in frame.columns:
        values = frame[name].dropna().astype(str)
        longest = max(len(str(name)), int(values.str.len().max()) if not values.empty else 0)
        config[str(name)] = {'alignment': 'center', 'width': min(420, max(90, longest * 8 + 30))}
        # 1.55's number renderer forces left/right alignment; text rendering centers
        # the display while the original numeric frame remains available for CSV.
        if pd.api.types.is_numeric_dtype(frame[name].dtype):
            config[str(name)]['type_config'] = {'type': 'text'}
    config.update(kwargs.pop('column_config', {}))
    kwargs.pop('width', None)
    kwargs.setdefault('hide_index', True)
    return st.dataframe(frame, width='content', column_config=config, **kwargs)
