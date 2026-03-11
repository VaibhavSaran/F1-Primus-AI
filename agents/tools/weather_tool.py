"""
Agent Weather Tool
Fetches race-weekend weather forecast for a given F1 round.
Uses Open-Meteo API — free and no API key required.
Automatically uses the correct race weekend dates from config.
"""

from __future__ import annotations
import requests
from datetime import datetime, timedelta
from langchain_core.tools import tool
from config import F1_CALENDAR


# Approximate coordinates for every 2026 circuit
CIRCUIT_COORDS = {
    "Melbourne":    (-37.8497, 144.9680),
    "Shanghai":     (31.3389,  121.2198),
    "Suzuka":       (34.8431,  136.5407),
    "Sakhir":       (26.0325,   50.5106),
    "Jeddah":       (21.6319,   39.1044),
    "Miami":        (25.9581,  -80.2389),
    "Montreal":     (45.5000,  -73.5228),
    "Monaco":       (43.7347,    7.4205),
    "Barcelona":    (41.5700,    2.2611),
    "Spielberg":    (47.2197,   14.7647),
    "Silverstone":  (52.0786,   -1.0169),
    "Spa":          (50.4372,    5.9714),
    "Budapest":     (47.5789,   19.2486),
    "Zandvoort":    (52.3888,    4.5409),
    "Monza":        (45.6156,    9.2811),
    "Madrid":       (40.4168,   -3.7038),
    "Baku":         (40.3725,   49.8533),
    "Singapore":    ( 1.2914,  103.8639),
    "Austin":       (30.1328,  -97.6411),
    "Mexico City":  (19.4042,  -99.0907),
    "São Paulo":    (-23.7036, -46.6997),
    "Las Vegas":    (36.1699, -115.1398),
    "Lusail":       (25.4900,   51.4542),
    "Abu Dhabi":    (24.4672,   54.6031),
}


@tool
def get_weather_forecast(round_number: int) -> dict:
    """
    Fetch the race-weekend weather forecast for a 2026 F1 round.
    Automatically uses the correct Friday/Saturday/Sunday dates
    for that round from the F1 calendar.

    Args:
        round_number: Race round number on the 2026 calendar (1-24).

    Returns:
        dict with daily forecasts for Friday (Practice), Saturday
        (Qualifying) and Sunday (Race) — temperature, precipitation
        probability, wind speed, and a plain-English race day summary.
    """
    # Lookup race info from calendar 
    race_info = F1_CALENDAR.get(round_number)
    if not race_info:
        return {"error": f"Round {round_number} not found in 2026 calendar."}

    race_name, location, country, race_date_str = race_info

    coords = CIRCUIT_COORDS.get(location)
    if not coords:
        return {"error": f"No coordinates found for location: '{location}'."}

    # Calculate Friday and Saturday from race (Sunday) date 
    race_date = datetime.strptime(race_date_str, "%Y-%m-%d")

    # Azerbaijan and Las Vegas race on Saturday — adjust accordingly
    saturday_races = ["Baku", "Las Vegas"]
    if location in saturday_races:
        saturday = race_date
        friday   = race_date - timedelta(days=1)
        sunday   = race_date + timedelta(days=1)  # no race, use for forecast completeness
    else:
        friday   = race_date - timedelta(days=2)
        saturday = race_date - timedelta(days=1)
        sunday   = race_date

    start_date = friday.strftime("%Y-%m-%d")
    end_date   = sunday.strftime("%Y-%m-%d")

    # Check if date is within Open-Meteo forecast range 
    today     = datetime.today()
    days_away = (race_date - today).days

    if days_away > 16:
        return {
            "circuit":    location,
            "race":       race_name,
            "race_date":  race_date_str,
            "days_away":  days_away,
            "forecast":   {},
            "summary":    (
                f"{race_name} is {days_away} days away — too far ahead for a "
                f"weather forecast. Open-Meteo forecasts up to 16 days ahead. "
                f"Re-run this tool closer to the race weekend."
            ),
        }

    lat, lon = coords

    # Historical or forecast endpoint
    if days_away < 0:
        # Race already happened — use historical archive
        base_url = "https://archive-api.open-meteo.com/v1/archive"
    else:
        # Upcoming race — use forecast API
        base_url = "https://api.open-meteo.com/v1/forecast"

    url = (
        f"{base_url}"
        f"?latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_max,temperature_2m_min,"
        f"precipitation_probability_max,windspeed_10m_max"
        f"&start_date={start_date}"
        f"&end_date={end_date}"
        f"&timezone=auto"
    )

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        daily  = data["daily"]
        dates  = daily["time"]
        t_max  = daily["temperature_2m_max"]
        t_min  = daily["temperature_2m_min"]
        rain   = daily["precipitation_probability_max"]
        wind   = daily["windspeed_10m_max"]

        if location in saturday_races:
            labels = ["Friday (Practice)", "Saturday (Race)", "Sunday (Post-Race)"]
        else:
            labels = ["Friday (Practice)", "Saturday (Qualifying)", "Sunday (Race)"]

        forecast = {}
        for i, label in enumerate(labels):
            if i < len(dates):
                forecast[label] = {
                    "date":                   dates[i],
                    "temp_max_c":             t_max[i],
                    "temp_min_c":             t_min[i],
                    "precipitation_prob_pct": rain[i],
                    "wind_speed_kmh":         wind[i],
                    "conditions":             _describe(rain[i], wind[i]),
                }

        return {
            "circuit":   location,
            "race":      race_name,
            "country":   country,
            "race_date": race_date_str,
            "days_away": days_away,
            "forecast":  forecast,
            "summary":   _summarise(forecast, race_name, location, saturday_races),
        }

    except requests.RequestException as exc:
        return {"error": f"Weather API request failed: {exc}"}


# helper Functions to convert raw forecast data into plain-English descriptions and summaries 

def _describe(rain_prob: float, wind: float) -> str:
    parts = []
    if rain_prob >= 70:
        parts.append("high rain risk")
    elif rain_prob >= 40:
        parts.append("possible rain")
    else:
        parts.append("dry conditions expected")

    if wind >= 40:
        parts.append("strong winds")
    elif wind >= 20:
        parts.append("moderate winds")

    return ", ".join(parts)


def _summarise(forecast: dict, race: str, location: str, saturday_races: list) -> str:
    race_key = "Saturday (Race)" if location in saturday_races else "Sunday (Race)"
    race_day = forecast.get(race_key, {})

    rain = race_day.get("precipitation_prob_pct", 0)
    temp = race_day.get("temp_max_c", "N/A")
    wind = race_day.get("wind_speed_kmh", 0)

    if rain >= 60:
        outlook = "WET race very likely — expect strategy chaos."
    elif rain >= 30:
        outlook = "Mixed conditions possible — teams may split strategies."
    else:
        outlook = "Dry race expected — standard strategy likely."

    return (
        f"Race day forecast for {race}: {temp}°C, "
        f"{rain}% rain probability, {wind} km/h winds. {outlook}"
    )