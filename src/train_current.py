"""
Trains the Current AQI Prediction model.

This model predicts AQI using the environmental readings
from the same time, such as PM2.5, PM10, CO, NO2,
temperature, humidity and other weather values.

"""

import os
import sys
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
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
from src.feature_engineering import CURRENT_PREDICTION_FEATURES


def load_data():
    """
    Load training data from Hopsworks first.

    If Hopsworks is not available, use the local
    CSV backup instead.
    """

    try:
        from src.hopsworks_client import get_feature_group

        fg = get_feature_group()

        if fg is not None:
            df = fg.read()

            print(
                f"[hopsworks] loaded "
                f"{len(df)} rows from feature store"
            )

            return df

    except Exception as e:
        print(
            f"[hopsworks] could not load data: {e}"
        )

    # Local CSV is used as a backup.
    if not os.path.exists(FEATURES_FILE):
        print(
            f"ERROR: {FEATURES_FILE} not found."
        )
        print(
            "Run backfill.py or feature_pipeline.py first."
        )
        sys.exit(1)

    df = pd.read_csv(
        FEATURES_FILE,
        parse_dates=["timestamp"]
    )

    print(
        f"Loaded {len(df)} rows from local CSV"
    )

    return df


def main():

    df = load_data()

    # Backfilled rows may not have weather data.
    # Only rows with all required features are used.
    df = df.dropna(
        subset=CURRENT_PREDICTION_FEATURES + ["aqi"]
    )

    print(
        f"Rows usable for current AQI model: "
        f"{len(df)}"
    )

    if len(df) < 30:
        print(
            "WARNING: very few rows are available "
            "for training."
        )

    X = df[
        CURRENT_PREDICTION_FEATURES
    ]

    y = df["aqi"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print(
        f"Train rows: {len(X_train)} | "
        f"Test rows: {len(X_test)}"
    )

    results = []

    # Ridge Regression
    ridge = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "ridge",
            Ridge(alpha=1.0)
        )
    ])

    ridge.fit(
        X_train,
        y_train
    )

    ridge_preds = ridge.predict(
        X_test
    )

    ridge_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            ridge_preds
        )
    )

    ridge_mae = mean_absolute_error(
        y_test,
        ridge_preds
    )

    ridge_r2 = r2_score(
        y_test,
        ridge_preds
    )

    print(
        f"[Ridge Regression] "
        f"RMSE={ridge_rmse:.2f} "
        f"MAE={ridge_mae:.2f} "
        f"R2={ridge_r2:.3f}"
    )

    results.append(
        (
            "Ridge Regression",
            ridge,
            ridge_rmse,
            ridge_mae,
            ridge_r2
        )
    )

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42
    )

    rf.fit(
        X_train,
        y_train
    )

    rf_preds = rf.predict(
        X_test
    )

    rf_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            rf_preds
        )
    )

    rf_mae = mean_absolute_error(
        y_test,
        rf_preds
    )

    rf_r2 = r2_score(
        y_test,
        rf_preds
    )

    print(
        f"[Random Forest] "
        f"RMSE={rf_rmse:.2f} "
        f"MAE={rf_mae:.2f} "
        f"R2={rf_r2:.3f}"
    )

    results.append(
        (
            "Random Forest",
            rf,
            rf_rmse,
            rf_mae,
            rf_r2
        )
    )

    # Select the model with the lowest RMSE.
    best_name, best_model, best_rmse, best_mae, best_r2 = min(
        results,
        key=lambda result: result[2]
    )

    print(
        f"\nBest model: {best_name}"
    )

    # Save the model locally as a backup.
    os.makedirs(
        MODELS_DIR,
        exist_ok=True
    )

    model_file = os.path.join(
        MODELS_DIR,
        "model_current.pkl"
    )

    payload = {
        "model": best_model,
        "model_name": best_name,
        "feature_columns": CURRENT_PREDICTION_FEATURES,
        "trained_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "metrics": {
            "rmse": best_rmse,
            "mae": best_mae,
            "r2": best_r2
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
        f"Saved to {model_file}"
    )

    # Register the trained model in Hopsworks.
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
                    # Saving the full payload (not just the bare model)
                    # loads Hopsworks models the exact same way it loads
                    pickle.dump(
                        payload,
                        f
                    )

                
                def clean_metric(value):
                    if value is None:
                        return 0.0
                    if isinstance(value, float) and value != value:  # NaN check
                        return 0.0
                    return value

                py_model = mr.python.create_model(
                    name="aqi_model_current",
                    metrics={
                        "rmse": clean_metric(best_rmse),
                        "mae": clean_metric(best_mae),
                        "r2": clean_metric(best_r2)
                    },
                    description=(
                        "Current AQI prediction model "
                        f"using {best_name}."
                    )
                )

                py_model.save(
                    tmpdir
                )

            print(
                "[hopsworks] registered "
                "aqi_model_current"
            )

    except Exception as e:
        print(
            f"[hopsworks] could not register "
            f"model: {e}"
        )


if __name__ == "__main__":
    main()