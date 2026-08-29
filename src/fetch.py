"""
Functions for getting air pollution and weather data
from the OpenWeather API.

We use current pollution, historical pollution,
and current weather data for our AQI project.
"""

import requests
from src.config import (
    OPENWEATHER_API_KEY,
    AIR_POLLUTION_URL,
    AIR_POLLUTION_HISTORY_URL,
    WEATHER_URL,
)


def get_current_pollution(lat, lon):
    params = {
        "lat": lat,
        "lon": lon,
        "appid": OPENWEATHER_API_KEY
    }

    resp = requests.get(
        AIR_POLLUTION_URL,
        params=params,
        timeout=15
    )

    resp.raise_for_status()

    return resp.json()


def get_historical_pollution(lat, lon, start_unix, end_unix):
    params = {
        "lat": lat,
        "lon": lon,
        "start": start_unix,
        "end": end_unix,
        "appid": OPENWEATHER_API_KEY,
    }

    resp = requests.get(
        AIR_POLLUTION_HISTORY_URL,
        params=params,
        timeout=15
    )

    resp.raise_for_status()

    return resp.json()


def get_current_weather(lat, lon):
    params = {
        "lat": lat,
        "lon": lon,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"
    }

    resp = requests.get(
        WEATHER_URL,
        params=params,
        timeout=15
    )

    resp.raise_for_status()

    return resp.json()