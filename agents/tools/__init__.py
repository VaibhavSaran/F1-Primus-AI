"""Lazy compatibility tools; the forecast graph does not bind unrestricted tools."""
from importlib import import_module

_EXPORTS={'get_weather_forecast':'weather_tool','get_practice_session_data':'fastf1_tool',
          'get_news_and_penalties':'news_tool','run_race_prediction':'prediction_tool'}


def __getattr__(name):
    if name not in _EXPORTS:raise AttributeError(name)
    return getattr(import_module(f'{__name__}.{_EXPORTS[name]}'),name)
