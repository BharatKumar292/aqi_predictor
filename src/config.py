"""
Basic settings for the AQI Predictor project.

Cities, API keys, and file paths are kept here so they
do not need to be hardcoded in other files.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# API keys
OPENWEATHER_API_KEY = os.getenv(
    "OPENWEATHER_API_KEY",
    ""
)

HOPSWORKS_API_KEY = os.getenv(
    "HOPSWORKS_API_KEY",
    ""
)

# Cities
# (city name, latitude, longitude)
CITIES = [
    ("Sukkur", 27.7052, 68.8574),
    ("Karachi", 24.8607, 67.0011),
    ("Lahore", 31.5497, 74.3436),
]

# Local storage
# This is only a local backup/cache.
# Hopsworks will be the main persistent
# storage for the final project.

DATA_DIR = os.path.join(
    os.path.dirname(
        os.path.dirname(__file__)
    ),
    "data"
)

FEATURES_FILE = os.path.join(
    DATA_DIR,
    "aqi_features.csv"
)

MODELS_DIR = os.path.join(
    os.path.dirname(DATA_DIR),
    "models"
)

# OpenWeather endpoints
AIR_POLLUTION_URL = (
    "http://api.openweathermap.org/data/2.5/air_pollution"
)

AIR_POLLUTION_HISTORY_URL = (
    "http://api.openweathermap.org/data/2.5/air_pollution/history"
)

WEATHER_URL = (
    "https://api.openweathermap.org/data/2.5/weather"
)