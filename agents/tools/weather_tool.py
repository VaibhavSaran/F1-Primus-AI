"""
Agent Weather tool
Fetches race-weekend weather forecast for a given F1 round.
Uses Open-Meteo API — completely free, no API key required.
Automatically uses the correct race weekend dates from config.
Handles both future forecasts (up to 16 days) and historical data.
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

# Circuits that hold their race on Saturday
SATURDAY_RACES = ["Baku", "Las Vegas"]


@tool
def get_weather_forecast(round_number: int) -> dict:
    """
    Fetch the race-weekend weather for a 2026 F1 round.
    Automatically uses correct Friday/Saturday/Sunday dates.
    Handles historical data for past races and forecasts
    for upcoming races (up to 16 days ahead).

    Args:
        round_number: Race round number on the 2026 calendar (1-24).

    Returns:
        dict with daily weather for each session, plus a race day
        summary with strategic implications.
    """
    # Lookup race info 
    race_info = F1_CALENDAR.get(round_number)
    if not race_info:
        return {"error": f"Round {round_number} not found in 2026 calendar."}

    race_name, location, country, race_date_str = race_info

    coords = CIRCUIT_COORDS.get(location)
    if not coords:
        return {"error": f"No coordinates found for: '{location}'."}

    lat, lon = coords

    # Calculate session dates 
    race_date = datetime.strptime(race_date_str, "%Y-%m-%d")
    today     = datetime.today()
    days_away = (race_date - today).days

    if location in SATURDAY_RACES:
        # Race is on Saturday
        friday   = race_date - timedelta(days=1)
        saturday = race_date
        sunday   = race_date + timedelta(days=1)
        labels   = [
            "Friday (Practice)",
            "Saturday (Race)",
            "Sunday (Post-Race)",
        ]
    else:
        # Standard Sunday race weekend
        friday   = race_date - timedelta(days=2)
        saturday = race_date - timedelta(days=1)
        sunday   = race_date
        labels   = [
            "Friday (Practice)",
            "Saturday (Qualifying)",
            "Sunday (Race)",
        ]

    start_date = friday.strftime("%Y-%m-%d")
    end_date   = sunday.strftime("%Y-%m-%d")

    # Too far ahead for any forecast 
    if days_away > 16:
        return {
            "circuit":   location,
            "race":      race_name,
            "race_date": race_date_str,
            "days_away": days_away,
            "forecast":  {},
            "is_historical": False,
            "summary": (
                f"⏳ {race_name} is {days_away} days away — too far ahead "
                f"for a weather forecast. Open-Meteo forecasts up to 16 days "
                f"ahead. Re-run closer to the race weekend."
            ),
        }

    # Choose correct API and fields 
    is_historical = days_away < 0

    if is_historical:
        # Past race — use archive API (no precipitation probability)
        base_url    = "https://archive-api.open-meteo.com/v1/archive"
        daily_fields = (
            "temperature_2m_max,"
            "temperature_2m_min,"
            "precipitation_sum,"        # actual rainfall in mm
            "windspeed_10m_max"
        )
    else:
        # Upcoming race — use forecast API
        base_url    = "https://api.open-meteo.com/v1/forecast"
        daily_fields = (
            "temperature_2m_max,"
            "temperature_2m_min,"
            "precipitation_probability_max,"
            "windspeed_10m_max"
        )

    url = (
        f"{base_url}"
        f"?latitude={lat}&longitude={lon}"
        f"&daily={daily_fields}"
        f"&start_date={start_date}"
        f"&end_date={end_date}"
        f"&timezone=auto"
    )

    # Fetch data 
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        daily  = data.get("daily", {})
        dates  = daily.get("time", [])
        t_max  = daily.get("temperature_2m_max", [])
        t_min  = daily.get("temperature_2m_min", [])
        wind   = daily.get("windspeed_10m_max", [])

        # Rain field differs between historical and forecast
        if is_historical:
            # precipitation_sum = actual mm of rain that fell
            rain_raw  = daily.get("precipitation_sum", [0] * len(dates))
            rain_type = "mm_actual"
        else:
            # precipitation_probability_max = % chance of rain
            rain_raw  = daily.get("precipitation_probability_max", [0] * len(dates))
            rain_type = "probability_pct"

        # Build forecast dict 
        forecast = {}
        for i, label in enumerate(labels):
            if i < len(dates):
                rain_val = rain_raw[i] if rain_raw and i < len(rain_raw) else 0
                wind_val = wind[i]     if wind     and i < len(wind)     else 0
                tmax_val = t_max[i]    if t_max    and i < len(t_max)    else None
                tmin_val = t_min[i]    if t_min    and i < len(t_min)    else None

                forecast[label] = {
                    "date":        dates[i],
                    "temp_max_c":  tmax_val,
                    "temp_min_c":  tmin_val,
                    "wind_speed_kmh": wind_val,
                    "rain_type":   rain_type,
                    "rain_value":  rain_val,
                    "conditions":  _describe(
                                       rain_val, wind_val,
                                       is_historical=is_historical
                                   ),
                }

                # Keep consistent key name for the agent
                if is_historical:
                    forecast[label]["precipitation_mm"]      = rain_val
                    forecast[label]["precipitation_prob_pct"] = None
                else:
                    forecast[label]["precipitation_prob_pct"] = rain_val
                    forecast[label]["precipitation_mm"]       = None

        return {
            "circuit":        location,
            "race":           race_name,
            "country":        country,
            "race_date":      race_date_str,
            "days_away":      days_away,
            "is_historical":  is_historical,
            "forecast":       forecast,
            "summary":        _summarise(
                                  forecast, race_name,
                                  location, is_historical
                              ),
        }

    except requests.RequestException as exc:
        return {"error": f"Weather API request failed: {exc}"}


# Helper functions

def _describe(rain_val: float, wind: float, is_historical: bool = False) -> str:
    """Generate a plain-English conditions description."""
    rain_val = rain_val or 0
    wind     = wind     or 0

    parts = []

    if is_historical:
        # rain_val is actual mm of rainfall
        if rain_val >= 10:
            parts.append("heavy rain recorded")
        elif rain_val >= 2:
            parts.append("light rain recorded")
        elif rain_val > 0:
            parts.append("trace rainfall recorded")
        else:
            parts.append("dry conditions")
    else:
        # rain_val is probability percentage
        if rain_val >= 70:
            parts.append("high rain risk")
        elif rain_val >= 40:
            parts.append("possible rain")
        else:
            parts.append("dry conditions expected")

    if wind >= 40:
        parts.append("strong winds")
    elif wind >= 20:
        parts.append("moderate winds")

    return ", ".join(parts)


def _summarise(
    forecast: dict,
    race: str,
    location: str,
    is_historical: bool,
) -> str:
    """Generate a race-day weather summary with strategic insight."""

    # Pick the race day key
    if location in SATURDAY_RACES:
        race_key = "Saturday (Race)"
    else:
        race_key = "Sunday (Race)" if not is_historical else "Sunday (Race)"

    race_day = forecast.get(race_key, {})

    temp = race_day.get("temp_max_c") or "N/A"
    wind = race_day.get("wind_speed_kmh") or 0

    if is_historical:
        rain_mm = race_day.get("precipitation_mm") or 0
        if rain_mm >= 10:
            outlook = "WET race — significant rainfall was recorded."
        elif rain_mm >= 2:
            outlook = "Damp conditions — some rainfall affected the race."
        elif rain_mm > 0:
            outlook = "Mostly dry with trace rainfall recorded."
        else:
            outlook = "Dry race conditions."

        return (
            f"Historical weather for {race}: {temp}°C, "
            f"{rain_mm}mm rainfall, {wind} km/h winds. {outlook}"
        )
    else:
        rain_prob = race_day.get("precipitation_prob_pct") or 0
        if rain_prob >= 60:
            outlook = "WET race very likely — expect strategy chaos."
        elif rain_prob >= 30:
            outlook = "Mixed conditions possible — teams may split strategies."
        else:
            outlook = "Dry race expected — standard strategy likely."

        return (
            f"Race day forecast for {race}: {temp}°C, "
            f"{rain_prob}% rain probability, {wind} km/h winds. {outlook}"
        )