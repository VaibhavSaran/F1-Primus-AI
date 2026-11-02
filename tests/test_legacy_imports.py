import importlib
import sys
from unittest.mock import patch


def test_importing_tools_constructs_no_provider_or_cache_client(monkeypatch):
    monkeypatch.delenv('TAVILY_API_KEY',raising=False)
    for name in list(sys.modules):
        if name=='agents.tools' or name.startswith('agents.tools.'):
            sys.modules.pop(name)
    with patch('langchain_tavily.TavilySearch',side_effect=AssertionError('import created Tavily client')), patch('fastf1.Cache.enable_cache',side_effect=AssertionError('import changed FastF1 cache')):
        package=importlib.import_module('agents.tools')
        for module in ('fastf1_tool','news_tool','weather_tool','prediction_tool'):
            importlib.import_module('agents.tools.'+module)
