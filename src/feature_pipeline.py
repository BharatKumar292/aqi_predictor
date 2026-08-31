"""
Feature Pipeline
================

Runs periodically to collect the latest AQI and weather data.

What it does:
1. Calls OpenWeather for Sukkur, Karachi and Lahore.
2. Converts API responses into feature rows.
3. Updates the local CSV backup.
4. Pushes only newly collected rows to Hopsworks.

Run:
    python -m src.feature_pipeline
"""

import os
import sys
import time
import pandas as pd

from src.config import (
    CITIES,
    OPENWEATHER_API_KEY,
    FEATURES_FILE,
    DATA_DIR,
)

from src.fetch import (
    get_current_pollution,
    get_current_weather,
)

from src.feature_engineering import (
    build_feature_row,
    add_time_features,
    add_derived_features,
)


def fetch_all_cities():
    rows = []

    for city, lat, lon in CITIES:
        try:
            pollution = get_current_pollution(lat, lon)

            weather = get_current_weather(lat, lon)

            row = build_feature_row(
                city,
                lat,
                lon,
                pollution,
                weather
            )

            rows.append(row)

            print(
                f"[ok] {city}: AQI={row['aqi']}"
            )

        except Exception as e:
            print(f"[FAILED] {city}: {e}")

        time.sleep(1)

    return pd.DataFrame(rows)


def push_to_hopsworks(df):
    """
    Push only newly collected rows to Hopsworks.

    The Hopsworks feature group's schema includes weather columns
    (temperature, humidity, etc) because the live hourly pipeline
    collects them. Backfilled historical rows don't have weather data
    (see the note at the top of backfill.py) - so if those columns
    are missing from the dataframe, we add them as empty (null)
    values before pushing, so the row still matches the schema.
    """

    try:
        from src.hopsworks_client import get_feature_group

        fg = get_feature_group()

        if fg is None:
            print("[hopsworks] not configured")
            return

        hw_df = df.copy()

        weather_columns = [
            "temperature", "humidity", "pressure",
            "wind_speed", "wind_deg", "cloudiness", "rain_1h",
        ]
        for col in weather_columns:
            if col not in hw_df.columns:
                hw_df[col] = None

        hw_df["timestamp_unix"] = (
            hw_df["timestamp"].astype("int64")
            // 10**9
        )

        fg.insert(hw_df)

        print(
            f"[hopsworks] pushed "
            f"{len(hw_df)} new rows"
        )

    except Exception as e:
        print(
            f"[hopsworks] could not push data: {e}"
        )


def save_features(df):
    """
    Save new rows to the local CSV backup.

    The complete local history is used to calculate
    time and lag features.

    Only the newly collected rows are pushed to Hopsworks.
    """

    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(FEATURES_FILE):
        existing = pd.read_csv(
            FEATURES_FILE,
            parse_dates=["timestamp"]
        )

        combined = pd.concat(
            [existing, df],
            ignore_index=True
        )

        combined = combined.drop_duplicates(
            subset=["city", "timestamp"]
        )

    else:
        combined = df.copy()

    # Make sure the weather columns always exist, even if this batch of
    # data doesn't have any (e.g. backfilled historical rows). Without
    # this, if the local CSV ever gets rebuilt from backfill data alone,
    # these columns would be missing entirely instead of just empty -
    # and train_current.py would crash looking for them.
    weather_columns = [
        "temperature", "humidity", "pressure",
        "wind_speed", "wind_deg", "cloudiness", "rain_1h",
    ]
    for col in weather_columns:
        if col not in combined.columns:
            combined[col] = None

    # Calculate time and lag features using
    # the complete local history.
    combined = add_time_features(
        combined
    )

    combined = add_derived_features(
        combined
    )

    combined = combined.sort_values(
        ["city", "timestamp"]
    )

    combined.to_csv(
        FEATURES_FILE,
        index=False
    )

    print(
        f"Saved {len(combined)} total rows "
        f"to {FEATURES_FILE}"
    )

    # Push only the newly collected rows,
    # but include their calculated features.
    new_rows = combined.merge(
        df[["city", "timestamp"]],
        on=["city", "timestamp"],
        how="inner"
    )

    push_to_hopsworks(new_rows)


def main():
    if not OPENWEATHER_API_KEY:
        print("ERROR: OPENWEATHER_API_KEY is not set.")
        sys.exit(1)

    df = fetch_all_cities()

    if df.empty:
        print(
            "No data fetched - "
            "check API key or internet connection."
        )
        sys.exit(1)

    save_features(df)


if __name__ == "__main__":
    main()