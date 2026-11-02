"""Explicit-date hourly Open-Meteo adapter with unknown issue times preserved."""
from dataclasses import asdict
from datetime import date,datetime,timezone
from math import isfinite
from langchain_core.tools import tool
from data.weather import fetch_weather


@tool
def get_weather_forecast(latitude:float,longitude:float,start_date:str,end_date:str)->dict:
    """Fetch hourly weather for verified circuit coordinates and schedule dates."""
    for value,limit in ((latitude,90),(longitude,180)):
        if isinstance(value,bool) or not isfinite(value) or abs(value)>limit:raise ValueError('invalid circuit coordinates')
    first,last=date.fromisoformat(start_date),date.fromisoformat(end_date)
    now=datetime.now(timezone.utc)
    if first<now.date() or last<first or (last-now.date()).days>15:
        raise ValueError('Forecast dates must be upcoming and within the provider horizon; historical observations are separate')
    result=fetch_weather('https://api.open-meteo.com/v1/forecast',
        dict(latitude=latitude,longitude=longitude,start_date=start_date,end_date=end_date,
             hourly='temperature_2m,precipitation_probability,wind_speed_10m'),fetched_at_utc=now,timeout=10)
    return asdict(result)
