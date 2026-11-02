"""Compatibility boundary. Paid external search is not enabled by tool import."""
from langchain_core.tools import tool


@tool
def get_news_and_penalties(season:int,event_name:str)->dict:
    """Explain external search eligibility; news never becomes a verified penalty."""
    return dict(status='unavailable',articles=[],quality_flags=['external_search_not_enabled'],
                errors=['Use the guarded Ask F1 service; external Tavily search requires an explicit budgeted provider adapter.'],
                event_name=event_name,season=season)
