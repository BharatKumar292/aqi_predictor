# 🌫️ AQI Predictor — Sukkur, Karachi & Lahore

An end-to-end **Air Quality Index (AQI) prediction and forecasting system** for
Sukkur, Karachi, and Lahore.

The project collects real-time air pollution and weather data from the
**OpenWeather API**, stores features in **Hopsworks Feature Store**, trains
machine learning models, and provides current AQI predictions and
**1–3 day AQI forecasts** through a deployed Streamlit dashboard.

Built as part of the **10Pearls Shine Internship — Data Science Track**.

---

## 🚀 Project Overview

The system performs the complete machine learning pipeline:

**OpenWeather API → Feature Engineering → Hopsworks Feature Store →
Model Training → Hopsworks Model Registry → Streamlit Dashboard**

### Cities

- 🇵🇰 Sukkur
- 🇵🇰 Karachi
- 🇵🇰 Lahore

### Main capabilities

- Real-time pollution data collection
- Historical pollution data collection
- AQI calculation from pollutant concentrations
- Feature engineering
- Current AQI prediction
- 24-hour AQI forecasting
- 48-hour AQI forecasting
- 72-hour AQI forecasting
- Automated hourly feature pipeline
- Model versioning with Hopsworks
- Interactive Streamlit dashboard

---

## 🏗️ Project Architecture

```text
                OpenWeather API
                       │
                       ▼
              ┌─────────────────┐
              │  Data Collection │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Feature         │
              │ Engineering     │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Hopsworks       │
              │ Feature Store   │
              └────────┬────────┘
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
     Current AQI Model     Forecast Models
        Ridge / RF          24h / 48h / 72h
             │                   │
             └─────────┬─────────┘
                       ▼
              ┌─────────────────┐
              │ Hopsworks Model │
              │ Registry        │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Streamlit       │
              │ Dashboard       │
              └─────────────────┘
📂 Project Structure
AQI_Project/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── aqi_features.csv
│
├── models/
│   ├── model_current.pkl
│   ├── model_24h.pkl
│   ├── model_48h.pkl
│   └── model_72h.pkl
│
├── src/
│   ├── config.py
│   ├── fetch.py
│   ├── feature_engineering.py
│   ├── feature_pipeline.py
│   ├── backfill.py
│   ├── train_current.py
│   ├── train_forecast.py
│   └── hopsworks_client.py
│
└── .github/
    └── workflows/
        ├── hourly_feature_pipeline.yml
        ├── train_current.yml
        └── train_forecast.yml
📊 Data & Features

The project uses pollution and weather information collected from
the OpenWeather API.

Pollution features
PM2.5
PM10
CO
NO
NO₂
O₃
SO₂
NH₃
Weather features
Temperature
Humidity
Pressure
Wind speed
Wind direction
Cloudiness
Rainfall
Time features
Hour
Day
Month
Day of week
Forecast features

The forecasting models additionally use historical AQI patterns:

Previous AQI
1-step AQI lag
3-step AQI lag
24-step AQI lag
6-point rolling AQI mean
AQI change rate
🤖 Machine Learning Models
Current AQI Prediction

The current model predicts AQI using environmental readings available
at the current time.

It does not use previous AQI values as input.

Models evaluated:

Ridge Regression
Random Forest Regressor

The model with the lowest RMSE is selected and registered in Hopsworks.

AQI Forecasting

Three separate models are used for future predictions:

Model	Forecast
aqi_model_24h	24 hours ahead
aqi_model_48h	48 hours ahead
aqi_model_72h	72 hours ahead

The forecast models use recent AQI history and engineered features to
predict future AQI.

🔒 Avoiding Data Leakage

Data leakage is an important consideration in this project.

The current AQI model does not use AQI itself as an input feature
because AQI is calculated from pollutant concentrations.

Using AQI to predict the same AQI value would allow the model to see
information that is essentially the target itself.

For forecasting, previous AQI values are allowed because they are known
before the future prediction time.

For example:

Past AQI ───────► Current time ───────► Future AQI
  70                  80                    ?

The model can use the 70 and 80 values because they were already
available before the future AQI occurred.

☁️ Hopsworks

Hopsworks is used as the project's cloud ML infrastructure.

Feature Store

The AQI feature group stores:

Pollution features
Weather features
Time features
AQI values
Forecast features
City information
Timestamps
Model Registry

The trained models are versioned in Hopsworks:

aqi_model_current
aqi_model_24h
aqi_model_48h
aqi_model_72h

This allows the dashboard to load the latest registered model versions.

⚙️ Automated Hourly Pipeline

GitHub Actions automatically runs the feature pipeline on an hourly
schedule.

GitHub Actions
      │
      ▼
OpenWeather API
      │
      ▼
Feature Engineering
      │
      ▼
Hopsworks Feature Store
      │
      ▼
Updated AQI Data

This allows the feature store to continuously receive new readings
without manually running the pipeline.

📈 Streamlit Dashboard

The project includes an interactive Streamlit dashboard showing:

Current AQI for all three cities
Current pollutant levels
Historical AQI trends
Pollutant trends
24-hour forecast
48-hour forecast
72-hour forecast
Model performance metrics
Feature importance
City comparison

The dashboard loads data and models from Hopsworks, with local files
available as a fallback.

🛠️ Technologies Used
Programming
Python
Data Science & Machine Learning
Pandas
NumPy
Scikit-learn
Matplotlib
APIs & Data
OpenWeather API
MLOps & Cloud
Hopsworks
Feature Store
Model Registry
GitHub Actions
Dashboard
Streamlit
Development
Git
GitHub
⚙️ Setup

Clone the repository and install the dependencies:

pip install -r requirements.txt

Set your API keys:

Windows CMD
set OPENWEATHER_API_KEY=your_openweather_key
set HOPSWORKS_API_KEY=your_hopsworks_key

Run the feature pipeline:

python -m src.feature_pipeline

Backfill historical pollution data:

python -m src.backfill

Train the current AQI model:

python -m src.train_current

Train the forecasting models:

python -m src.train_forecast

Run the dashboard:

streamlit run app.py
⚠️ Limitations
Historical weather availability depends on the OpenWeather API plan.
Historical pollution data and historical weather data have different
availability.
Forecast accuracy generally decreases as the prediction horizon
increases.
The forecast models do not use actual future weather observations.
AQI values from different services may differ because different
providers can use different data sources, update times, AQI standards,
or calculation methods.

🔮 Future Improvements
Add more Pakistani cities
Incorporate weather forecast data
Improve forecasting models
Add LSTM/GRU-based time-series models
Add model monitoring
Add automated model retraining
Improve prediction uncertainty estimates
Add more advanced feature engineering

👨‍💻 Author

Bharat Kumar

BSCS Student | Data Science & Machine Learning
