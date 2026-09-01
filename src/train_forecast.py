"""
Trains the "3-Day Forecast" models.

This is different from train_current.py. Here we predict AQI
24, 48 and 72 hours into the future.

The forecast model can use the current AQI and previous AQI values
because these values are already known when making a forecast.

"""

import os
import sys
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from src.config import FEATURES_FILE, MODELS_DIR
from src.feature_engineering import FORECAST_FEATURES


# We predict AQI 1, 2 and 3 days into the future.
HORIZONS_HOURS = [24, 48, 72]

# Keep the last 20% of the data for testing.
TEST_SIZE_FRACTION = 0.2


def load_features():
    """
    Load feature data from Hopsworks when available.

    If Hopsworks is not available, use the local CSV file instead.
    """

    # Try Hopsworks first.
    try:
        from src.hopsworks_client import get_feature_group

        fg = get_feature_group()

        if fg is not None:
            df = fg.read()

            # Keep timestamps consistent with the rest of the project.
            df["timestamp"] = pd.to_datetime(
                df["timestamp"],
                utc=True
            )

            print(
                f"[hopsworks] loaded {len(df)} rows "
                "from the feature store"
            )

            return df

    except Exception as e:
        print(
            f"[hopsworks] could not read feature store ({e}), "
            "using local CSV instead"
        )

    # Fall back to the local CSV.
    if not os.path.exists(FEATURES_FILE):
        print(
            f"ERROR: {FEATURES_FILE} not found. "
            "Run feature_pipeline.py or backfill.py first."
        )
        sys.exit(1)

    df = pd.read_csv(
        FEATURES_FILE,
        parse_dates=["timestamp"]
    )

    # Make sure local timestamps are also UTC-aware.
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    print(
        f"Loaded {len(df)} rows from local CSV"
    )

    return df


def build_target(df, horizon_hours):
    """
    Create the future AQI target for each city.

    For example, with a 24-hour horizon, the AQI from 24 hours
    later becomes the target for the current row.
    """

    df = df.sort_values(
        ["city", "timestamp"]
    ).copy()

    df["aqi_target"] = (
        df.groupby("city")["aqi"]
        .shift(-horizon_hours)
    )

    return df


def prepare_dataset(df):
    """
    Prepare the data for model training.

    City is converted into dummy columns so the model can
    distinguish between Sukkur, Karachi and Lahore.
    """

    df = df.dropna(
        subset=FORECAST_FEATURES + ["aqi_target"]
    ).copy()

    city_dummies = pd.get_dummies(
        df["city"],
        prefix="city"
    )

    X = pd.concat(
        [
            df[FORECAST_FEATURES].reset_index(drop=True),
            city_dummies.reset_index(drop=True),
        ],
        axis=1,
    )

    y = df["aqi_target"].reset_index(
        drop=True
    )

    meta = df[
        ["city", "timestamp"]
    ].reset_index(drop=True)

    return X, y, meta


def time_based_split(X, y, meta):
    """
    Split the data by time instead of randomly.

    This is important for forecasting because the model should
    not train on future data and then test on older data.
    """

    cutoff = meta["timestamp"].quantile(
        1 - TEST_SIZE_FRACTION
    )

    train_mask = meta["timestamp"] < cutoff

    X_train = X[train_mask]
    X_test = X[~train_mask]

    y_train = y[train_mask]
    y_test = y[~train_mask]

    print(
        f"Train rows: {len(X_train)} | "
        f"Test rows: {len(X_test)} | "
        f"Split cutoff: {cutoff}"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


def evaluate(name, model, X_test, y_test):
    """
    Calculate RMSE, MAE and R² for a model.
    """

    predictions = model.predict(X_test)

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    print(
        f"  [{name}] "
        f"RMSE={rmse:.2f}  "
        f"MAE={mae:.2f}  "
        f"R2={r2:.3f}"
    )

    return {
        "name": name,
        "model": model,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def evaluate_naive_baseline(
    y_test,
    current_aqi_test
):
    """
    Use the current AQI as a simple baseline.

    This represents a situation where we assume the AQI
    will stay the same in the future.
    """

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            current_aqi_test
        )
    )

    mae = mean_absolute_error(
        y_test,
        current_aqi_test
    )

    r2 = r2_score(
        y_test,
        current_aqi_test
    )

    print(
        f"  [Naive baseline] "
        f"RMSE={rmse:.2f}  "
        f"MAE={mae:.2f}  "
        f"R2={r2:.3f}"
    )


def save_model(
    result,
    feature_columns,
    horizon_hours
):
    """
    Save the trained model locally and also register it
    in the Hopsworks Model Registry when available.
    """

    os.makedirs(
        MODELS_DIR,
        exist_ok=True
    )

    model_file = os.path.join(
        MODELS_DIR,
        f"model_{horizon_hours}h.pkl"
    )

    payload = {
        "model": result["model"],
        "model_name": result["name"],
        "feature_columns": feature_columns,
        "horizon_hours": horizon_hours,
        "trained_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "metrics": {
            "rmse": result["rmse"],
            "mae": result["mae"],
            "r2": result["r2"],
        },
    }

    with open(
        model_file,
        "wb"
    ) as f:
        pickle.dump(
            payload,
            f
        )

    print(
        f"  Saved to {model_file}"
    )

    # Also register the model in Hopsworks.
    try:
        from src.hopsworks_client import (
            get_model_registry
        )

        import tempfile

        mr = get_model_registry()

        if mr is not None:
            with tempfile.TemporaryDirectory() as tmpdir:

                model_path = os.path.join(
                    tmpdir,
                    "model.pkl"
                )

                with open(
                    model_path,
                    "wb"
                ) as f:
                    # Same reasoning as train_current.py - saving the full
                    # payload here, not just the bare model, so the
                    # dashboard can load it the same way whether it comes
                    # from Hopsworks or the local file.
                    pickle.dump(
                        payload,
                        f
                    )

                # Hopsworks stores metrics as JSON, which doesn't allow
                # NaN values - replace NaN with 0.0 just for the upload.
                def clean_metric(value):
                    if value is None:
                        return 0.0
                    if isinstance(value, float) and value != value:  # NaN check
                        return 0.0
                    return value

                py_model = mr.python.create_model(
                    name=f"aqi_model_{horizon_hours}h",
                    metrics={
                        "rmse": clean_metric(result["rmse"]),
                        "mae": clean_metric(result["mae"]),
                        "r2": clean_metric(result["r2"]),
                    },
                    description=(
                        f"AQI forecast model, "
                        f"{horizon_hours}h ahead "
                        f"({result['name']})."
                    ),
                )

                py_model.save(
                    tmpdir
                )

            print(
                f"  [hopsworks] registered "
                f"aqi_model_{horizon_hours}h"
            )

    except Exception as e:
        print(
            f"  [hopsworks] could not register "
            f"model: {e}"
        )


def train_for_horizon(
    raw_df,
    horizon_hours
):
    """
    Train and evaluate the models for one forecast horizon.
    """

    print(
        f"\n=== Horizon: "
        f"{horizon_hours}h ahead ==="
    )

    df = build_target(
        raw_df,
        horizon_hours
    )

    X, y, meta = prepare_dataset(
        df
    )

    if len(X) < 50:
        print(
            f"  WARNING: only {len(X)} rows "
            "available - results may not be "
            "reliable yet."
        )

    X_train, X_test, y_train, y_test = (
        time_based_split(
            X,
            y,
            meta
        )
    )

    # Compare our models with the simple
    # "AQI stays the same" baseline.
    evaluate_naive_baseline(
        y_test,
        X_test["aqi"]
    )

    results = []

    # Ridge Regression
    ridge = Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),
            (
                "ridge",
                Ridge(alpha=1.0)
            ),
        ]
    )

    ridge.fit(
        X_train,
        y_train
    )

    results.append(
        evaluate(
            "Ridge Regression",
            ridge,
            X_test,
            y_test
        )
    )

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=5,
        random_state=42,
    )

    rf.fit(
        X_train,
        y_train
    )

    results.append(
        evaluate(
            "Random Forest",
            rf,
            X_test,
            y_test
        )
    )

    # Select the model with the lowest RMSE.
    best = min(
        results,
        key=lambda result: result["rmse"]
    )

    print(
        f"  Best model for "
        f"{horizon_hours}h: "
        f"{best['name']}"
    )

    save_model(
        best,
        list(X.columns),
        horizon_hours
    )


def main():
    """
    Load the feature data and train models for
    all three forecast horizons.
    """

    raw_df = load_features()

    for horizon in HORIZONS_HOURS:
        train_for_horizon(
            raw_df,
            horizon
        )

    print(
        "\nDone. Trained models for: "
        + ", ".join(
            f"{h}h"
            for h in HORIZONS_HOURS
        )
    )


if __name__ == "__main__":
    main()