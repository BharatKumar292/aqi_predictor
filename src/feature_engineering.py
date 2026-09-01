"""
This file prepares the data collected from the OpenWeather API.
It calculates the AQI from the pollutant values and also creates
some extra features that we can use when training our models.

There are two sets of features:
1. Features for predicting the current AQI.
2. Features for predicting AQI for the next 1 to 3 days.

"""

from datetime import datetime, timezone
import pandas as pd

# AQI breakpoints for PM2.5
# Format:
# (lowest concentration, highest concentration, lowest AQI, highest AQI)

PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]

# AQI breakpoints for PM10

PM10_BREAKPOINTS = [
    (0, 54, 0, 50),
    (55, 154, 51, 100),
    (155, 254, 101, 150),
    (255, 354, 151, 200),
    (355, 424, 201, 300),
    (425, 504, 301, 400),
    (505, 604, 401, 500),
]

def linear_aqi(concentration, breakpoints):
    """Calculate AQI using the given concentration breakpoints."""

    if concentration is None:
        return None

    for c_lo, c_hi, aqi_lo, aqi_hi in breakpoints:

        if c_lo <= concentration <= c_hi:

            return (
                ((aqi_hi - aqi_lo) / (c_hi - c_lo))
                * (concentration - c_lo)
                + aqi_lo
            )

    # Keep AQI within the 0-500 range
    return 500.0

def compute_aqi(components):
    """
    Calculate the overall AQI using PM2.5 and PM10.

    The higher pollutant sub-index is used as the final AQI.
    """

    pm25_aqi = linear_aqi(
        components.get("pm2_5"),
        PM25_BREAKPOINTS
    )

    pm10_aqi = linear_aqi(
        components.get("pm10"),
        PM10_BREAKPOINTS
    )

    values = [
        value
        for value in [pm25_aqi, pm10_aqi]
        if value is not None
    ]

    if not values:
        return None

    return round(max(values), 1)

def build_feature_row(
    city,
    lat,
    lon,
    pollution_data,
    weather_data=None
):
    """Create one feature row from the current API data."""

    item = pollution_data["list"][0]

    components = item["components"]

    ts = datetime.fromtimestamp(
        item["dt"],
        tz=timezone.utc
    )

    row = {
        "city": city,
        "lat": lat,
        "lon": lon,
        "timestamp": ts,
        "aqi": compute_aqi(components),

        "co": components.get("co"),
        "no": components.get("no"),
        "no2": components.get("no2"),
        "o3": components.get("o3"),
        "so2": components.get("so2"),
        "pm2_5": components.get("pm2_5"),
        "pm10": components.get("pm10"),
        "nh3": components.get("nh3"),
    }

    # Weather data is added when the current weather API is used
    if weather_data:

        row["temperature"] = weather_data.get(
            "main", {}
        ).get("temp")

        row["humidity"] = weather_data.get(
            "main", {}
        ).get("humidity")

        row["pressure"] = weather_data.get(
            "main", {}
        ).get("pressure")

        row["wind_speed"] = weather_data.get(
            "wind", {}
        ).get("speed")

        row["wind_deg"] = weather_data.get(
            "wind", {}
        ).get("deg")

        row["cloudiness"] = weather_data.get(
            "clouds", {}
        ).get("all")

        row["rain_1h"] = weather_data.get(
            "rain", {}
        ).get("1h", 0)

    return row

def build_feature_rows_from_history(
    city,
    lat,
    lon,
    pollution_history
):
    """
    Create feature rows from historical pollution data.

    The historical API returns multiple readings, so we create
    one row for each reading.
    """

    rows = []

    for item in pollution_history.get("list", []):

        components = item["components"]

        ts = datetime.fromtimestamp(
            item["dt"],
            tz=timezone.utc
        )

        rows.append({
            "city": city,
            "lat": lat,
            "lon": lon,
            "timestamp": ts,
            "aqi": compute_aqi(components),

            "co": components.get("co"),
            "no": components.get("no"),
            "no2": components.get("no2"),
            "o3": components.get("o3"),
            "so2": components.get("so2"),
            "pm2_5": components.get("pm2_5"),
            "pm10": components.get("pm10"),
            "nh3": components.get("nh3"),
        })

    return rows

def add_time_features(df, timestamp_col="timestamp"):
    """Add basic time-related features to the data."""

    df = df.copy()

    dt = pd.to_datetime(
        df[timestamp_col],
        utc=True
    )

    df["hour"] = dt.dt.hour
    df["day"] = dt.dt.day
    df["month"] = dt.dt.month
    df["day_of_week"] = dt.dt.dayofweek

    return df


def add_derived_features(df, group_col="city"):
    """
    Add previous AQI and rolling features.

    These features are mainly used by the 3-day forecast model.
    The calculations are done separately for each city.
    """

    df = df.copy()

    df = df.sort_values(
        [group_col, "timestamp"]
    )

    grouped = df.groupby(group_col)["aqi"]

    df["aqi_lag_1"] = grouped.shift(1)

    df["aqi_lag_3"] = grouped.shift(3)

    df["aqi_lag_24"] = grouped.shift(24)

    df["aqi_rolling_mean_6"] = grouped.transform(
        lambda s: s.rolling(
            6,
            min_periods=1
        ).mean()
    )

    # Avoid division by zero when the previous AQI is 0
    df["aqi_change_rate"] = (
        (df["aqi"] - df["aqi_lag_1"])
        / df["aqi_lag_1"].replace(0, pd.NA)
    )

    return df

# Features used for the current AQI prediction model.
#
# These are readings available at the current time.
# Previous AQI values are not used here.

CURRENT_PREDICTION_FEATURES = [
    "pm2_5",
    "pm10",
    "co",
    "no",
    "no2",
    "o3",
    "so2",
    "nh3",
    "temperature",
    "humidity",
    "pressure",
    "wind_speed",
]

# Features used for the 3-day forecast model.
#
# Previous AQI values can be used here because they come
# from earlier times than the value we are trying to predict.

FORECAST_FEATURES = [
    "aqi",
    "hour",
    "day",
    "month",
    "day_of_week",
    "co",
    "no",
    "no2",
    "o3",
    "so2",
    "pm2_5",
    "pm10",
    "nh3",
    "aqi_lag_1",
    "aqi_lag_3",
    "aqi_lag_24",
    "aqi_rolling_mean_6",
    "aqi_change_rate",
]