"""
Agent News Tool For F1 Races
Uses Tavily search (via LangChain) to find recent F1 news,
grid penalties, and driver/team updates before a race weekend.
"""

from __future__ import annotations
import os
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from config import TAVILY_API_KEY, F1_CALENDAR

os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY or ""

_search = TavilySearch(max_results=5)


@tool
def get_news_and_penalties(round_number: int) -> dict:
    """
    Search for the latest F1 news, grid penalties, and team updates
    relevant to the upcoming race weekend.

    Args:
        round_number: Race round number on the 2026 calendar (1-24).

    Returns:
        dict with penalty flags, relevant headlines, and a plain-English
        summary of anything that could affect the race outcome.
    """
    race_info = F1_CALENDAR.get(round_number)
    if not race_info:
        return {"error": f"Round {round_number} not found in 2026 calendar."}

    race_name, location, country, race_date = race_info

    queries = [
        f"F1 2026 {race_name} grid penalty",
        f"F1 2026 {race_name} latest news team updates",
        f"Formula 1 2026 {race_name} driver incident retirement",
    ]

    all_results = []
    for query in queries:
        try:
            print(f"  Searching: {query}")
            results = _search.invoke(query)
            if isinstance(results, list):
                all_results.extend(results)
        except Exception as exc:
            all_results.append({"error": str(exc), "query": query})

    # Deduplicate by URL 
    seen_urls = set()
    unique_results = []
    for r in all_results:
        url = r.get("url", "")
        if url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(r)

    # Scan for penalty keywords 
    penalty_keywords = [
        "penalty", "grid drop", "reprimand", "disqualified",
        "dsq", "pit lane start", "power unit", "gearbox change",
    ]
    penalty_flags = []
    for r in unique_results:
        content  = (r.get("content") or r.get("snippet") or "").lower()
        title    = (r.get("title") or "").lower()
        combined = content + " " + title
        for kw in penalty_keywords:
            if kw in combined:
                penalty_flags.append({
                    "keyword":  kw,
                    "headline": r.get("title", "No title"),
                    "url":      r.get("url", ""),
                    "snippet":  (r.get("content") or "")[:200],
                })
                break

    # Build clean headlines list 
    headlines = [
        {
            "title":   r.get("title", "No title"),
            "url":     r.get("url", ""),
            "snippet": (r.get("content") or "")[:300],
        }
        for r in unique_results[:6]
        if "error" not in r
    ]

    return {
        "round":         round_number,
        "race":          race_name,
        "location":      location,
        "race_date":     race_date,
        "penalty_flags": penalty_flags,
        "headlines":     headlines,
        "total_articles": len(unique_results),
        "summary":       _summarise(race_name, penalty_flags, headlines),
    }


# helper functions

def _summarise(race: str, penalties: list, headlines: list) -> str:
    if penalties:
        penalty_summary = (
            f"{len(penalties)} potential penalty/incident(s) detected "
            f"heading into {race}. Review flagged articles carefully."
        )
    else:
        penalty_summary = f"No grid penalties detected for {race}."

    top_news = ""
    if headlines:
        top_news = " Latest: " + headlines[0]["title"]

    return penalty_summary + top_news