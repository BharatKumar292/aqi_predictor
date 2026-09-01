"""
This file is used to collect old pollution data from OpenWeather.

trying to collect around 2 years of historical data.
The amount of old data we can get depends on our OpenWeather API plan.

IMPORTANT: this script pushes data to Hopsworks in small chunks
(one chunk at a time, roughly 700 rows) instead of collecting
everything and pushing it all at once. We do this because Hopsworks'
free-tier compute struggled to process one giant 50,000+ row batch in
a single materialization job (it kept failing). Smaller batches match
what already works fine for the hourly pipeline.
"""
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.config import CITIES, OPENWEATHER_API_KEY
from src.fetch import get_historical_pollution
from src.feature_engineering import build_feature_rows_from_history
from src.feature_pipeline import save_features

DAYS_TO_BACKFILL = 730       # about 2 years
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
    """
    Fetch historical data for one city, chunk by chunk, and push each
    chunk to Hopsworks right away instead of waiting until everything
    is collected. This keeps each Hopsworks upload small enough for
    the free-tier compute to handle.
    """
    total_rows = 0

    for chunk_start, chunk_end in daterange_chunks(
        start,
        end,
        CHUNK_DAYS
    ):

        start_unix = int(chunk_start.timestamp())
        end_unix = int(chunk_end.timestamp())

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

            print(
                f"[ok] {city}: "
                f"{chunk_start.date()} to "
                f"{chunk_end.date()} "
                f"({len(rows)} rows) - saving this chunk now..."
            )

            if rows:
                chunk_df = pd.DataFrame(rows)
                # save_features() updates the local CSV AND pushes this
                # chunk to Hopsworks - one small batch at a time.
                save_features(chunk_df)
                total_rows += len(rows)

        except Exception as e:
            print(
                f"[FAILED] {city}: "
                f"{chunk_start.date()} to "
                f"{chunk_end.date()}: "
                f"{e}"
            )

        time.sleep(REQUEST_DELAY_SECONDS)

    return total_rows


def main():

    if not OPENWEATHER_API_KEY:
        print("ERROR: OPENWEATHER_API_KEY is not set.")
        sys.exit(1)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=DAYS_TO_BACKFILL)

    print(
        f"Backfilling {DAYS_TO_BACKFILL} days "
        f"for {len(CITIES)} cities "
        f"({start.date()} to {end.date()})...\n"
    )

    grand_total = 0

    for city, lat, lon in CITIES:
        rows_for_city = backfill_city(city, lat, lon, start, end)
        print(f"\n{city}: {rows_for_city} rows collected and pushed.\n")
        grand_total += rows_for_city

    if grand_total == 0:
        print("No data was fetched for any city.")
        sys.exit(1)

    print(f"\nBackfill finished. Total rows collected across all cities: {grand_total}")


if __name__ == "__main__":
    main()