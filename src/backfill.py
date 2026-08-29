"""
This file is used to collect old pollution data from OpenWeather.

We run this script to get enough historical data for training our
AQI prediction model.

Run this file using:

    python -m src.backfill

This script collects historical pollution data for our cities.
The weather data is collected separately by the current weather
pipeline.

We are trying to collect around 2 years of historical data.
The amount of old data we can get depends on our OpenWeather API plan.
"""
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.config import CITIES, OPENWEATHER_API_KEY
from src.fetch import get_historical_pollution
from src.feature_engineering import build_feature_rows_from_history
from src.feature_pipeline import save_features

DAYS_TO_BACKFILL = 730      # about 2 years
CHUNK_DAYS = 30              # request data in manageable chunks
REQUEST_DELAY_SECONDS = 1


def daterange_chunks(start, end, chunk_days):
    current = start

    while current < end:
        chunk_end = min(
            current + timedelta(days=chunk_days),
            end
        )

        yield current, chunk_end
        current = chunk_end


def backfill_city(city, lat, lon, start, end):
    all_rows = []

    for chunk_start, chunk_end in daterange_chunks(
        start,
        end,
        CHUNK_DAYS
    ):

        start_unix = int(
            chunk_start.timestamp()
        )

        end_unix = int(
            chunk_end.timestamp()
        )

        try:

            history = get_historical_pollution(
                lat,
                lon,
                start_unix,
                end_unix
            )

            rows = build_feature_rows_from_history(
                city,
                lat,
                lon,
                history
            )

            all_rows.extend(rows)

            print(
                f"[ok] {city}: "
                f"{chunk_start.date()} to "
                f"{chunk_end.date()} "
                f"({len(rows)} rows)"
            )

        except Exception as e:

            print(
                f"[FAILED] {city}: "
                f"{chunk_start.date()} to "
                f"{chunk_end.date()}: "
                f"{e}"
            )

        time.sleep(
            REQUEST_DELAY_SECONDS
        )

    return pd.DataFrame(
        all_rows
    )


def main():

    if not OPENWEATHER_API_KEY:

        print(
            "ERROR: OPENWEATHER_API_KEY is not set."
        )

        sys.exit(1)

    end = datetime.now(
        timezone.utc
    )

    start = end - timedelta(
        days=DAYS_TO_BACKFILL
    )

    print(
        f"Backfilling {DAYS_TO_BACKFILL} days "
        f"for {len(CITIES)} cities "
        f"({start.date()} to {end.date()})...\n"
    )

    all_dfs = []

    for city, lat, lon in CITIES:

        df = backfill_city(
            city,
            lat,
            lon,
            start,
            end
        )

        if not df.empty:

            all_dfs.append(
                df
            )

    if not all_dfs:

        print(
            "No data was fetched for any city."
        )

        sys.exit(1)

    combined = pd.concat(
        all_dfs,
        ignore_index=True
    )

    # Remove accidental duplicate city/timestamp rows
    if "city" in combined.columns and "timestamp" in combined.columns:

        combined = combined.drop_duplicates(
            subset=["city", "timestamp"]
        )

    print(
        f"\nTotal rows fetched: "
        f"{len(combined)}"
    )

    print(
        f"Cities collected: "
        f"{combined['city'].nunique()}"
    )

    save_features(
        combined
    )


if __name__ == "__main__":
    main()