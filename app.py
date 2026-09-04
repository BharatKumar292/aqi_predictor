"""
AQI Predictor Dashboard
========================
Shows current AQI, historical trends, and 3-day forecasts for Sukkur,
Karachi, and Lahore.

Data and models are loaded from Hopsworks (feature store + model
registry)-
"""

import os
import pickle

import pandas as pd
import streamlit as st

try:
    import shap
    import matplotlib.pyplot as plt
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from src.config import FEATURES_FILE, MODELS_DIR, CITIES
from src.feature_engineering import CURRENT_PREDICTION_FEATURES

CITY_NAMES = [c[0] for c in CITIES]
HORIZONS_HOURS = [24, 48, 72]
HORIZON_LABELS = {24: "Tomorrow (24h)", 48: "Day 2 (48h)", 72: "Day 3 (72h)"}

AQI_LEVELS = [
    (0, 50, "Good", "#16A34A"),
    (50, 100, "Moderate", "#CA8A04"),
    (100, 150, "Unhealthy for Sensitive Groups", "#EA580C"),
    (150, 200, "Unhealthy", "#DC2626"),
    (200, 300, "Very Unhealthy", "#9333EA"),
    (300, 500, "Hazardous", "#7F1D1D"),
]


def aqi_level(aqi):
    for lo, hi, label, color in AQI_LEVELS:
        if lo <= aqi <= hi:
            return label, color
    return "Unknown", "#888888"


# ------------------------------------------------------------------
# Data / model loading - Hopsworks first, local file as backup only
# ------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_data():
    try:
        from src.hopsworks_client import get_feature_group
        fg = get_feature_group()
        if fg is not None:
            df = fg.read()
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df.sort_values(["city", "timestamp"])
    except Exception as e:
        st.warning(f"Could not reach Hopsworks feature store ({e}). Using local backup data instead.")

    if not os.path.exists(FEATURES_FILE):
        return pd.DataFrame()
    df = pd.read_csv(FEATURES_FILE, parse_dates=["timestamp"])
    return df.sort_values(["city", "timestamp"])


@st.cache_resource
def load_model(hopsworks_model_name, local_filename):
    try:
        from src.hopsworks_client import get_model_registry
        mr = get_model_registry()
        if mr is not None:
            all_versions = mr.get_models(hopsworks_model_name)
            if all_versions:
                model_entry = max(all_versions, key=lambda m: m.version)
                model_dir = model_entry.download()
                model_path = os.path.join(model_dir, "model.pkl")
                with open(model_path, "rb") as f:
                    loaded = pickle.load(f)
                if isinstance(loaded, dict) and "model" in loaded:
                    return loaded
                print(f"[hopsworks] model '{hopsworks_model_name}' is in an old format, using local file instead")
    except Exception as e:
        print(f"[hopsworks] could not load model '{hopsworks_model_name}' from registry: {e}")

    path = os.path.join(MODELS_DIR, local_filename)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_current_aqi(model_payload, latest_row):
    """Uses the current-prediction model - raw pollutant/weather readings only."""
    X = pd.DataFrame([latest_row[CURRENT_PREDICTION_FEATURES].to_dict()])
    return float(model_payload["model"].predict(X)[0])


def predict_forecast_aqi(model_payload, latest_row, city):
    """Uses one of the forecast models (24h/48h/72h)."""
    feature_columns = model_payload["feature_columns"]
    model = model_payload["model"]

    base_cols = [c for c in feature_columns if not c.startswith("city_")]
    row = {col: latest_row.get(col, 0) for col in base_cols}

    for col in feature_columns:
        if col.startswith("city_"):
            city_name = col.replace("city_", "")
            row[col] = 1 if city_name == city else 0

    X = pd.DataFrame([row])[feature_columns]
    return float(model.predict(X)[0])


def current_aqi_for_row(current_model, row):
    """Predicted current AQI if the model + weather data are available, else the computed AQI value."""
    if current_model is not None and row[CURRENT_PREDICTION_FEATURES].notna().all():
        return predict_current_aqi(current_model, row)
    return row["aqi"]


def daily_average(frame, value_cols):
    """
    Charting 50,000+ raw hourly points freezes the browser tab, so we
    resample to a daily average before plotting. Still shows the same
    overall trend, just far fewer points to render.
    """
    resampled = frame.set_index("timestamp")[value_cols].resample("D").mean()
    return resampled


# ---------------- Page setup ----------------
st.set_page_config(page_title="Pearls AQI Predictor", page_icon="\U0001F32B\uFE0F", layout="wide")
st.title("\U0001F32B\uFE0F Pearls AQI Predictor")
st.caption("Live air quality tracking and 3-day forecasts for Sukkur, Karachi, and Lahore")

df = load_data()
current_model = load_model("aqi_model_current", "model_current.pkl")
forecast_models = {h: load_model(f"aqi_model_{h}h", f"model_{h}h.pkl") for h in HORIZONS_HOURS}

if df.empty:
    st.error(
        "No data found yet. Run `python -m src.feature_pipeline` and "
        "`python -m src.backfill` first to generate the dataset."
    )
    st.stop()

# ---------------- Overview: all cities ----------------
st.subheader("Current AQI \u2014 all tracked cities")

latest_per_city = df.sort_values("timestamp").groupby("city").tail(1)
overview_rows = []
for _, row in latest_per_city.iterrows():
    overview_rows.append((row["city"], current_aqi_for_row(current_model, row)))
overview_rows.sort(key=lambda r: r[1], reverse=True)

cols = st.columns(len(overview_rows))
for i, (city_name, aqi_value) in enumerate(overview_rows):
    label, color = aqi_level(aqi_value)
    city_latest_row = latest_per_city[latest_per_city["city"] == city_name].iloc[0]
    with cols[i]:
        st.markdown(
            f"""
            <div style="border-radius:10px;padding:14px;margin-bottom:10px;
                        background-color:{color}22;border:1px solid {color};">
                <div style="font-size:14px;color:#666;">{city_name}</div>
                <div style="font-size:28px;font-weight:700;">{aqi_value:.0f}</div>
                <div style="font-size:13px;color:{color};font-weight:600;">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(f"As of {city_latest_row['timestamp']}")

hazardous = [c for c, v in overview_rows if v > 150]
if hazardous:
    st.warning(
        f"\u26A0\uFE0F Hazardous or unhealthy AQI currently reported in: **{', '.join(hazardous)}**. "
        "Consider limiting outdoor exposure in these areas."
    )

st.divider()

# ---------------- City detail ----------------
st.subheader("City detail")
selected_city = st.selectbox("Select a city", CITY_NAMES)

city_df = df[df["city"] == selected_city].sort_values("timestamp")
latest_row = city_df.iloc[-1]
predicted_aqi = current_aqi_for_row(current_model, latest_row)

col1, col2, col3 = st.columns(3)
label, color = aqi_level(predicted_aqi)
col1.metric("Current AQI", f"{predicted_aqi:.0f}", label)
col2.metric("PM2.5 (\u00b5g/m\u00b3)", f"{latest_row['pm2_5']:.1f}")
col3.metric("PM10 (\u00b5g/m\u00b3)", f"{latest_row['pm10']:.1f}")
st.caption(f"Reading as of {latest_row['timestamp']}")

st.line_chart(daily_average(city_df, ["aqi"]), height=300)

# ---------------- Forecast ----------------
st.subheader("3-day forecast")

fcols = st.columns(4)
with fcols[0]:
    label, color = aqi_level(predicted_aqi)
    st.markdown(
        f"""
        <div style="border-radius:10px;padding:16px;text-align:center;
                    background-color:{color}22;border:1px solid {color};">
            <div style="font-size:13px;color:#666;">Today</div>
            <div style="font-size:32px;font-weight:700;">{predicted_aqi:.0f}</div>
            <div style="font-size:13px;color:{color};font-weight:600;">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

predictions = {}
available_horizons = [h for h in HORIZONS_HOURS if forecast_models.get(h) is not None]

for i, h in enumerate(available_horizons):
    payload = forecast_models[h]
    pred = predict_forecast_aqi(payload, latest_row, selected_city)
    predictions[h] = pred
    label, color = aqi_level(pred)
    with fcols[i + 1]:
        st.markdown(
            f"""
            <div style="border-radius:10px;padding:16px;text-align:center;
                        background-color:{color}22;border:1px solid {color};">
                <div style="font-size:13px;color:#666;">{HORIZON_LABELS[h]}</div>
                <div style="font-size:32px;font-weight:700;">{pred:.0f}</div>
                <div style="font-size:13px;color:{color};font-weight:600;">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

if not available_horizons:
    st.info("No trained forecast models found yet. Run `python -m src.train_forecast` to enable forecasts.")
else:
    worst_horizon = max(predictions, key=predictions.get) if predictions else None
    if worst_horizon and predictions[worst_horizon] > 150:
        st.warning(
            f"\u26A0\uFE0F Forecast indicates unhealthy air quality in {selected_city} "
            f"around {HORIZON_LABELS[worst_horizon].lower()}."
        )

    with st.expander("Model details", expanded=True):
        rows = []
        if current_model is not None:
            m = current_model["metrics"]
            rows.append({
                "Forecast": "Current AQI",
                "Algorithm": current_model["model_name"],
                "RMSE": round(m["rmse"], 1),
                "MAE": round(m["mae"], 1),
                "R\u00b2": round(m["r2"], 3),
                "Trained": current_model["trained_at"][:19].replace("T", " "),
            })
        for h in available_horizons:
            payload = forecast_models[h]
            m = payload["metrics"]
            rows.append({
                "Forecast": HORIZON_LABELS[h],
                "Algorithm": payload["model_name"],
                "RMSE": round(m["rmse"], 1),
                "MAE": round(m["mae"], 1),
                "R\u00b2": round(m["r2"], 3),
                "Trained": payload["trained_at"][:19].replace("T", " "),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # SHAP explainability for the 24h forecast model
    if 24 in forecast_models and forecast_models[24] is not None:
        with st.expander("Why this prediction? (SHAP explainability, 24h model)"):
            model = forecast_models[24]["model"]
            feature_columns = forecast_models[24]["feature_columns"]

            if not SHAP_AVAILABLE:
                st.caption("Install the `shap` package (`pip install shap`) to see this.")
            else:
                try:
                    # Build a sample feature matrix the same way predict_forecast_aqi does,
                    # using recent rows across all cities so SHAP has real, varied examples.
                    base_cols = [c for c in feature_columns if not c.startswith("city_")]
                    sample_source = df.dropna(subset=base_cols).tail(200)

                    sample_rows = []
                    for _, r in sample_source.iterrows():
                        row = {col: r.get(col, 0) for col in base_cols}
                        for col in feature_columns:
                            if col.startswith("city_"):
                                city_name = col.replace("city_", "")
                                row[col] = 1 if city_name == r["city"] else 0
                        sample_rows.append(row)
                    X_sample = pd.DataFrame(sample_rows)[feature_columns]

                    if hasattr(model, "feature_importances_"):
                        # Tree-based model (Random Forest) - use the fast, exact TreeExplainer
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(X_sample)

                        fig, ax = plt.subplots()
                        shap.summary_plot(shap_values, X_sample, show=False, max_display=10)
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                        st.caption(
                            "Each point is one historical reading. Red = high value for that "
                            "feature, blue = low. Position left/right shows whether that value "
                            "pushed the AQI prediction down or up for that particular reading."
                        )
                    else:
                        # Linear model (Ridge) - fall back to coefficient-based importance
                        importances = abs(model.named_steps["ridge"].coef_) if hasattr(model, "named_steps") else abs(model.coef_)
                        imp_df = pd.DataFrame({"feature": feature_columns, "importance": importances})
                        imp_df = imp_df.sort_values("importance", ascending=False).head(10)
                        st.bar_chart(imp_df.set_index("feature"))
                        st.caption("This model is linear, so SHAP falls back to coefficient magnitude here.")
                except Exception as e:
                    st.caption(f"Could not compute SHAP values: {e}")

st.divider()
st.subheader("Correlation between pollutants and AQI")
corr_cols = ["aqi", "pm2_5", "pm10", "co", "no", "no2", "o3", "so2", "nh3"]
corr_matrix = df[corr_cols].corr(numeric_only=True)
st.dataframe(
    corr_matrix.style.background_gradient(cmap="RdYlGn_r", vmin=-1, vmax=1).format("{:.2f}"),
    use_container_width=True,
)
aqi_corr = corr_matrix["aqi"].drop("aqi").sort_values(ascending=False)
st.caption(
    f"Strongest correlation with AQI: **{aqi_corr.index[0]}** ({aqi_corr.iloc[0]:.2f}). "
    f"Computed from all {len(df):,} collected readings across the three cities."
)

st.divider()
st.subheader("Pollutant trend")
pollutant_cols = ["pm2_5", "pm10", "co", "no", "no2", "o3", "so2", "nh3"]
chosen_pollutant = st.selectbox("Pick a pollutant", pollutant_cols, index=0)
st.line_chart(daily_average(city_df, [chosen_pollutant]))

st.divider()
st.subheader("City comparison")
comparison_daily = df.copy()
comparison_daily["date"] = comparison_daily["timestamp"].dt.floor("D")
pivot = comparison_daily.pivot_table(index="date", columns="city", values="aqi", aggfunc="mean")
st.line_chart(pivot)