"""
Streamlit dashboard for the AQI Predictor project.

Tabs:
- Overview: pick a city, see current AQI and pollutant readings
- Pollutants: closer look at individual pollutant levels
- Trends: AQI history over time for the picked city
- 3-Day Forecast: today / tomorrow / day 2 / day 3 predictions
- City Comparison: compare all 3 cities side by side
- About Project: what this project is and how it works  

Run with:
    streamlit run app.py
"""

import os
import pickle
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from src.config import FEATURES_FILE, MODELS_DIR, CITIES
from src.feature_engineering import CURRENT_PREDICTION_FEATURES, FORECAST_FEATURES

CITY_NAMES = [c[0] for c in CITIES]

# EPA AQI categories - used everywhere we show a color/label for a number
AQI_LEVELS = [
    (0, 50, "Good", "#2e7d32"),
    (51, 100, "Moderate", "#c9a227"),
    (101, 150, "Unhealthy for Sensitive Groups", "#e08a2b"),
    (151, 200, "Unhealthy", "#c0392b"),
    (201, 300, "Very Unhealthy", "#8e44ad"),
    (301, 500, "Hazardous", "#6b2737"),
]

HEALTH_ADVICE = {
    "Good": "Air quality is satisfactory. Enjoy normal outdoor activities.",
    "Moderate": "Acceptable air quality. Unusually sensitive people should consider reducing prolonged outdoor exertion.",
    "Unhealthy for Sensitive Groups": "Children, elderly people, and those with respiratory conditions should limit prolonged outdoor activity.",
    "Unhealthy": "Everyone may begin to notice health effects. Sensitive groups should avoid outdoor exertion.",
    "Very Unhealthy": "Health warning. Avoid outdoor activity where possible, especially sensitive groups.",
    "Hazardous": "Serious health risk for everyone. Stay indoors and avoid outdoor exertion.",
}


def aqi_level(aqi):
    for lo, hi, label, color in AQI_LEVELS:
        if lo <= aqi <= hi:
            return label, color
    return "Unknown", "#888888"


@st.cache_data(ttl=600)
def load_data():
    if not os.path.exists(FEATURES_FILE):
        return pd.DataFrame()
    df = pd.read_csv(FEATURES_FILE, parse_dates=["timestamp"])
    return df.sort_values(["city", "timestamp"])


@st.cache_resource
def load_model(filename):
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_current_aqi(model_payload, latest_row):
    """Uses the current-prediction model - raw pollutant/weather readings only."""
    X = pd.DataFrame([latest_row[CURRENT_PREDICTION_FEATURES].to_dict()])
    return float(model_payload["model"].predict(X)[0])


def predict_forecast_aqi(model_payload, latest_row, city):
    """Uses one of the forecast models (24h/48h/72h) - includes lag features and city dummy columns."""
    feature_columns = model_payload["feature_columns"]
    base_cols = [c for c in feature_columns if not c.startswith("city_")]
    row = {col: latest_row.get(col, 0) for col in base_cols}
    for col in feature_columns:
        if col.startswith("city_"):
            row[col] = 1 if col == f"city_{city}" else 0
    X = pd.DataFrame([row])[feature_columns]
    return float(model_payload["model"].predict(X)[0])


def metric_card(label, value, unit=""):
    st.metric(label, f"{value:.1f}{unit}" if pd.notna(value) else "N/A")


def big_aqi_card(aqi, label, color):
    st.markdown(
        f"""
        <div style="border-radius:12px;padding:24px;text-align:center;
                    background-color:{color}18;border:1px solid {color};">
            <div style="font-size:15px;color:#555;">Predicted AQI</div>
            <div style="font-size:52px;font-weight:700;color:{color};">{aqi:.0f}</div>
            <div style="font-size:16px;font-weight:600;color:{color};">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------
st.set_page_config(page_title="AQI Prediction & Air Quality Monitoring", page_icon="🌍", layout="wide")

st.title("AQI Prediction & Air Quality Monitoring")
st.caption(
    "Predicts current and 3-day-ahead Air Quality Index for Sukkur, Karachi, and Lahore, "
    "using pollution and weather data collected from OpenWeather."
)

df = load_data()
if df.empty:
    st.error("No data yet. Run `python -m src.feature_pipeline` and `python -m src.backfill` first.")
    st.stop()

current_model = load_model("model_current.pkl")
forecast_models = {h: load_model(f"model_{h}h.pkl") for h in [24, 48, 72]}

with st.sidebar:
    st.header("City")
    selected_city = st.radio("Choose a city", CITY_NAMES)

city_df = df[df["city"] == selected_city].sort_values("timestamp")
latest_row = city_df.iloc[-1]

tab_overview, tab_pollutants, tab_trends, tab_forecast, tab_compare, tab_about = st.tabs(
    ["Overview", "Pollutants", "Trends", "3-Day Forecast", "City Comparison", "About Project"]
)

# ------------------------------------------------------------------
# Overview tab
# ------------------------------------------------------------------
with tab_overview:
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader(selected_city)
        if current_model is not None and latest_row[CURRENT_PREDICTION_FEATURES].notna().all():
            predicted_aqi = predict_current_aqi(current_model, latest_row)
        else:
            # not enough weather data yet for this row - fall back to the
            # already-computed AQI value so the dashboard still shows something
            predicted_aqi = latest_row["aqi"]
        label, color = aqi_level(predicted_aqi)
        big_aqi_card(predicted_aqi, label, color)
        st.caption(f"Last updated: {latest_row['timestamp']}")

    with col_right:
        st.subheader("Pollutant readings")
        p1, p2, p3 = st.columns(3)
        with p1:
            metric_card("PM2.5", latest_row["pm2_5"], " \u00b5g/m\u00b3")
            metric_card("NO2", latest_row["no2"], " \u00b5g/m\u00b3")
        with p2:
            metric_card("PM10", latest_row["pm10"], " \u00b5g/m\u00b3")
            metric_card("SO2", latest_row["so2"], " \u00b5g/m\u00b3")
        with p3:
            metric_card("CO", latest_row["co"], " \u00b5g/m\u00b3")
            metric_card("O3", latest_row["o3"], " \u00b5g/m\u00b3")

    st.divider()
    st.subheader("Health guidance")
    st.info(HEALTH_ADVICE.get(label, ""))

# ------------------------------------------------------------------
# Pollutants tab
# ------------------------------------------------------------------
with tab_pollutants:
    st.subheader(f"Pollutant levels - {selected_city}")
    pollutant_cols = ["pm2_5", "pm10", "co", "no", "no2", "o3", "so2", "nh3"]
    latest_pollutants = latest_row[pollutant_cols]
    st.bar_chart(latest_pollutants)

    st.subheader("Pollutant trend over time")
    chosen_pollutant = st.selectbox("Pick a pollutant", pollutant_cols, index=0)
    st.line_chart(city_df.set_index("timestamp")[[chosen_pollutant]])

# ------------------------------------------------------------------
# Trends tab
# ------------------------------------------------------------------
with tab_trends:
    st.subheader(f"AQI trend - {selected_city}")
    st.line_chart(city_df.set_index("timestamp")[["aqi"]])
    st.caption(f"Based on {len(city_df)} recorded readings.")

# ------------------------------------------------------------------
# 3-Day Forecast tab
# ------------------------------------------------------------------
with tab_forecast:
    st.subheader(f"3-Day AQI Forecast - {selected_city}")
    st.caption(
        "Today's value comes from the current-AQI model using live readings. "
        "Tomorrow / Day 2 / Day 3 come from separate forecast models trained "
        "on how AQI has moved in the past - not a copy of today's number."
    )

    today = datetime.now()
    day_labels = ["Today", "Tomorrow", "Day 2", "Day 3"]
    day_dates = [today, today + timedelta(days=1), today + timedelta(days=2), today + timedelta(days=3)]

    forecast_values = []

    # Today: current-prediction model
    if current_model is not None and latest_row[CURRENT_PREDICTION_FEATURES].notna().all():
        forecast_values.append(predict_current_aqi(current_model, latest_row))
    else:
        forecast_values.append(latest_row["aqi"])

    # Tomorrow / Day 2 / Day 3: forecast models
    for h in [24, 48, 72]:
        model_payload = forecast_models[h]
        if model_payload is not None:
            forecast_values.append(predict_forecast_aqi(model_payload, latest_row, selected_city))
        else:
            forecast_values.append(None)

    cols = st.columns(4)
    for col, day_label, day_date, value in zip(cols, day_labels, day_dates, forecast_values):
        with col:
            if value is None:
                st.markdown(f"**{day_label}**")
                st.caption(day_date.strftime("%b %d"))
                st.write("Model not trained yet")
            else:
                lvl_label, lvl_color = aqi_level(value)
                st.markdown(
                    f"""
                    <div style="border-radius:10px;padding:16px;text-align:center;
                                background-color:{lvl_color}18;border:1px solid {lvl_color};">
                        <div style="font-size:13px;color:#666;">{day_label}</div>
                        <div style="font-size:12px;color:#888;">{day_date.strftime('%b %d')}</div>
                        <div style="font-size:30px;font-weight:700;">{value:.0f}</div>
                        <div style="font-size:13px;color:{lvl_color};font-weight:600;">{lvl_label}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    worst_value = max([v for v in forecast_values if v is not None], default=None)
    if worst_value is not None and worst_value > 150:
        st.warning(f"AQI is expected to reach unhealthy levels in {selected_city} within the next 3 days.")

# ------------------------------------------------------------------
# City Comparison tab
# ------------------------------------------------------------------
with tab_compare:
    st.subheader("Current AQI across all cities")
    latest_per_city = df.sort_values("timestamp").groupby("city").tail(1).set_index("city")["aqi"]
    st.bar_chart(latest_per_city)

    st.subheader("AQI trend comparison")
    pivot = df.pivot_table(index="timestamp", columns="city", values="aqi")
    st.line_chart(pivot)

# ------------------------------------------------------------------
# About Project tab
# ------------------------------------------------------------------
with tab_about:
    st.subheader("About this project")
    st.markdown(
        """
This project predicts Air Quality Index (AQI) for **Sukkur, Karachi, and Lahore**
using data collected from the OpenWeather Air Pollution and Weather APIs.

**Two separate models are used:**
- **Current AQI model** - predicts AQI from live pollutant and weather readings
  (PM2.5, PM10, CO, NO2, SO2, O3, NH3, temperature, humidity, pressure, wind speed).
  This model never sees AQI itself as an input, since AQI is calculated from
  these same pollutant readings - using it as a feature would be circular.
- **3-day forecast models** - three separate models predict AQI 24h, 48h, and
  72h ahead, using how AQI has moved in the recent past (lag features, rolling
  averages, time-of-day patterns) rather than a copy of the current reading.

**Data pipeline:**
1. A feature pipeline fetches fresh data every hour and stores it in Hopsworks
   (feature store) and a local CSV.
2. A backfill script pulled about 2 months of historical pollution data to
   build the initial training set.
3. Training scripts pull data from Hopsworks, train Ridge Regression and
   Random Forest models, and save the best one to Hopsworks Model Registry.
4. GitHub Actions runs the feature pipeline hourly and the training pipeline
   daily, so the whole thing keeps itself up to date.

**Known limitations:**
- Free-tier historical weather data is limited to about a week, so backfilled
  rows only have pollutant readings, not historical weather. Weather columns
  fill in going forward as the hourly pipeline runs.
- The 3-day forecast doesn't have access to real future weather forecasts,
  so accuracy naturally gets less certain further into the future.
"""
    )

st.divider()
st.caption(
    f"Data Source: OpenWeather API  |  "
    f"Model: {current_model['model_name'] if current_model else 'not trained yet'}  |  "
    f"Cities: {', '.join(CITY_NAMES)}"
)
