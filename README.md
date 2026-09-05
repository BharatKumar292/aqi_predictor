# 🌫️ Pearls AQI Predictor

**An end-to-end air quality prediction system** that fetches live pollution and weather data for three Pakistani cities, stores it in a feature store, trains two separate machine learning models (one for right-now predictions, one for 3-day forecasts), and serves both through a live public dashboard — fully automated on a schedule via GitHub Actions.

> Built as part of the **10Pearls Shine Internship Program** (Data Science track) · Aug 2026

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io)
[![Hopsworks](https://img.shields.io/badge/feature%20store-Hopsworks-1EB182.svg)](https://www.hopsworks.ai)

---

## 🌐 Live Demo

**🔗 [pearlsaqipredictorbk.streamlit.app](https://pearlsaqipredictorbk.streamlit.app)**

Shows current AQI and a 3-day forecast for Sukkur, Karachi, and Lahore, pulling live from the Hopsworks feature store and model registry on every load — not a static demo or cached screenshot.

---

## 📊 Dashboard

<img width="1797" height="827" alt="dashboard" src="https://github.com/user-attachments/assets/a5fa5385-f067-4e52-a169-8856bd87fbe5" />

---

## 🏗️ Architecture

### Data Flow

```
OpenWeather API (pollution + weather)
            │
            ▼
    Feature Pipeline (hourly)  ──┐
    Backfill (one-time, ~2yrs)   │
            │                    │
            ▼                    │
   Hopsworks Feature Store  ◄────┘
   (feature group: aqi_features)
            │
    ┌───────┴────────┐
    ▼                ▼
train_current.py  train_forecast.py
(no AQI as input)  (24h/48h/72h models)
    │                │
    └───────┬────────┘
            ▼
  Hopsworks Model Registry
  (aqi_model_current, aqi_model_24h/48h/72h)
            │
            ▼
    Streamlit Dashboard
   (fetches data + models
    from Hopsworks live)
            │
            ▼
      🌐 Public Internet
  pearlsaqipredictorbk.streamlit.app
```

GitHub Actions runs the feature pipeline every hour and the training pipeline once a day, so the whole system keeps itself current without manual intervention.

### How It Works

1. **Feature pipeline** (`src/feature_pipeline.py`) calls the OpenWeather Air Pollution and Weather APIs every hour for Sukkur, Karachi, and Lahore, computes AQI using the US EPA breakpoint formula, and pushes the result to the Hopsworks feature store.
2. **Backfill** (`src/backfill.py`) pulled about 2 years of historical pollution data in 30-day chunks (pushed to Hopsworks chunk-by-chunk, not as one giant batch, because the free-tier compute couldn't materialize 50,000+ rows in a single job).
3. **Two training scripts** read from the Hopsworks feature store:
   - `train_current.py` trains a model that predicts AQI from live pollutant/weather readings only — it never sees AQI as an input, since AQI is itself derived from those readings.
   - `train_forecast.py` trains three separate models (24h/48h/72h ahead), using past AQI values (lag features, rolling averages) as legitimate time-series inputs, with a **time-based** train/test split so the model never trains on data from after the test period.
4. Both scripts save the best model (Ridge Regression or Random Forest, whichever scores lower RMSE) to the **Hopsworks Model Registry**, alongside a local `.pkl` backup.
5. **GitHub Actions** runs the feature pipeline hourly and the training pipeline daily.
6. **Streamlit dashboard** (`app.py`) loads the latest data and models directly from Hopsworks on every page load (falling back to the local backup only if Hopsworks is briefly unreachable), and renders current AQI, pollutant breakdowns, trends, and the 3-day forecast.

---

## ✅ Model Evaluation

Every forecast model is compared against a **naive baseline** (assume AQI doesn't change from right now) — if a trained model can't beat that, it isn't adding real value.

| Horizon | Naive RMSE | Best model | Best RMSE | Improvement vs. naive |
|---|---|---|---|---|
| 24h | 69.1 | Random Forest | 51.2 | ~26% |
| 48h | 72.2 | Random Forest | 52.2 | ~28% |
| 72h | 71.8 | Random Forest | 52.4 | ~27% |

A consistent ~26-28% error reduction across all three horizons, on a real held-out test set split by time (not randomly), is the evidence that these models generalize rather than overfit to one lucky split.

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Data source** | OpenWeather Air Pollution API + Weather API | Live pollutant and weather readings |
| **Feature store** | Hopsworks | Central, versioned source of truth for features |
| **Model registry** | Hopsworks Model Registry | Stores trained models with metrics |
| **Model training** | scikit-learn (Ridge Regression, Random Forest) | Two model families compared per task |
| **Automation** | GitHub Actions | Hourly feature pipeline, daily training pipeline |
| **Dashboard** | Streamlit + Plotly | Live public dashboard |
| **Deployment** | Streamlit Community Cloud | Free hosting, auto-redeploys on push |

---

## 🚀 Quick Start (Local Development)

### Prerequisites

- Python 3.11+ (3.13 has a known incompatibility with one Hopsworks dependency on some hosts — see [Engineering Decisions](#-engineering-decisions))
- Free API keys: [OpenWeather](https://openweathermap.org/api), [Hopsworks](https://www.hopsworks.ai)

### Setup

```bash
git clone https://github.com/BharatKumar292/aqi_predictor.git
cd aqi_predictor
pip install -r requirements.txt
```

Create a `.env` file:
```
OPENWEATHER_API_KEY=your_key_here
HOPSWORKS_API_KEY=your_key_here
```

### Run the pipeline

```bash
python -m src.feature_pipeline    # fetch current data
python -m src.backfill            # pull ~2 years of history (one-time)
python -m src.train_current       # train the current-AQI model
python -m src.train_forecast      # train the 3 forecast models
streamlit run app.py              # launch the dashboard
```

---

## 🔄 Automation (GitHub Actions)

Two scheduled workflows in `.github/workflows/`:

| Workflow | Schedule | What it does |
|---|---|---|
| `feature_pipeline.yml` | Every hour | Fetches fresh data, commits it back to the repo, pushes to Hopsworks |
| `train_pipeline.yml` | Once a day | Retrains all 4 models, commits updated `.pkl` files, registers new versions in Hopsworks |

Both use `OPENWEATHER_API_KEY` and `HOPSWORKS_API_KEY` stored as GitHub Secrets — never committed to the repository.

---

## 📁 Project Structure

```
aqi_predictor/
├── .github/
│   └── workflows/
│       ├── feature_pipeline.yml     # hourly automation
│       └── train_pipeline.yml       # daily automation
├── .streamlit/
│   └── config.toml                  # dashboard theme
├── src/
│   ├── config.py                    # cities, API keys, file paths
│   ├── fetch.py                     # OpenWeather API calls
│   ├── feature_engineering.py       # AQI formula, feature lists, leakage-safe split
│   ├── feature_pipeline.py          # hourly data collection
│   ├── backfill.py                  # historical data collection
│   ├── hopsworks_client.py          # feature store / model registry connection
│   ├── train_current.py             # current-AQI model
│   └── train_forecast.py            # 24h/48h/72h forecast models
├── data/
│   └── aqi_features.csv             # local backup of the feature store
├── models/
│   └── *.pkl                        # local backup of trained models
├── app.py                           # Streamlit dashboard
├── requirements.txt
├── screenshots/
│   └── *.png                        # images of dashboard and aqi model training  
├── AQI_Project_Report.pdf           # final report of project           
└── README.md
```

---

## 🔧 Engineering Decisions

### Why two separate models instead of one?

AQI is calculated FROM pollutant readings (see `compute_aqi()` in `feature_engineering.py`), so a model that predicts AQI from live readings must never see AQI itself as an input — that would be circular. But a model forecasting AQI days ahead legitimately needs past AQI values as features, since that's normal time-series forecasting. Trying to force one model to do both jobs either introduces leakage or throws away useful signal, so the project uses two: `train_current.py` (no AQI input) and `train_forecast.py` (AQI lag features allowed).

### Why a time-based split for forecasting but a random split for the current-AQI model?

The forecast models predict the future, so training on data from after the test period (which a random split can accidentally do) lets the model "see" outcomes it shouldn't have access to yet. The current-AQI model isn't predicting anything in time — each row is an independent snapshot — so a random split is fine there.

### Why HUDI instead of the Hopsworks default (Delta) for the feature group?

The default `time_travel_format` needed the `delta` Python library, which isn't installed by default and adds a heavy dependency. Explicitly setting `time_travel_format="HUDI"` avoided that extra install without losing any functionality this project needs.

### Why does the backfill script push data in small chunks instead of one batch?

The first attempt pushed all ~50,000 backfilled rows to Hopsworks in a single insert. The free-tier compute couldn't materialize a batch that large — the background job kept failing. Splitting the backfill into 30-day chunks (~700 rows each) and pushing incrementally matched what already worked reliably for the hourly pipeline's small pushes.

---

## 🧠 What I Learned

### Real debugging stories from building this

**Windows couldn't find `/tmp`.** The Hopsworks client library assumes a Unix-style `/tmp` directory exists for storing connection certificates. On Windows, that path doesn't exist by default, so login failed with `WinError 3`. Fixed by creating the folder ahead of time on Windows specifically, while leaving Linux/Mac (and GitHub Actions) untouched since `/tmp` already exists there.

**A cryptic `AttributeError` was actually a permissions problem.** Login failed with `'Response' object has no attribute 'error_code'` — a confusing internal library bug. The real cause, found by reading the full traceback, was an API key created without the "Serving" scope enabled, which triggered a 403 that the library's own error-handling code didn't parse correctly. Recreating the key with full scopes fixed it immediately — the visible error and the actual cause were two different things.

**A single NaN value can silently kill a whole upload.** Registering a model with only a handful of training rows meant R² was mathematically undefined (`NaN`) for one horizon. Hopsworks stores metrics as JSON, and JSON doesn't support NaN — the model registration failed with an obscure Jackson parser error. Fixed by sanitizing metrics (replacing NaN with 0.0) before sending them, while keeping the real value in the local backup file.

**A boundary gap in a lookup table caused an "Unknown" label.** The AQI category ranges were defined as `(0,50), (51,100), (101,150)...` — any fractional prediction between 50 and 51 fell into neither range and defaulted to "Unknown." Fixed by making the ranges continuous: `(0,50), (50,100), (100,150)...`.

**A stale model version, not a stale dataset, caused a dashboard crash.** After fixing the training scripts to save a full payload (model + feature list) instead of a bare model object, the dashboard still crashed — because it was hardcoded to load Hopsworks model **version 1**, which had been created by the old code before the fix. Loading the latest version dynamically (`max(versions, key=lambda m: m.version)`) instead of a hardcoded number solved it.

**Weather columns existing in one place but not another caused a schema mismatch.** Historical backfilled rows never had weather data (only live/hourly rows do), but after a local file reset, the local CSV briefly lost the weather columns entirely rather than having them empty — causing a `KeyError` at training time. Fixed by explicitly ensuring those columns always exist (filled with `None` when missing) before saving locally, not just before pushing to Hopsworks.

**Python 3.13 broke a Hopsworks dependency on deployment.** The dashboard deployed fine locally but failed on Streamlit Community Cloud with `ModuleNotFoundError: No module named 'imp'` — a module removed in Python 3.13 that an old Hopsworks dependency (`avro`) still imports. Pinning the deployment to Python 3.11 in Streamlit Cloud's advanced settings fixed it, since the local machine had happened to have a prebuilt wheel available that the cloud build server didn't.

### Concepts internalized

- **Data leakage isn't just about train/test splitting** — using a feature that's derived from (or too similar to) the target is leakage too, even with a perfect split.
- **A naive baseline is the real judge of a forecasting model** — RMSE alone means nothing without something to compare it to.
- **Free-tier cloud services have real, undocumented limits** — batch size, compute timeouts, and dependency mismatches all showed up in practice and needed workarounds, not just retries.
- **Reading the full traceback matters more than the top-line error message** — several of the bugs above looked unrelated to their actual cause until the full stack trace was read carefully.

---

## 📈 Future Improvements

- [ ] Add a real weather **forecast** API (not just past weather) as a model input for the 3-day predictions
- [ ] Try XGBoost or a simple LSTM and compare against Ridge/Random Forest
- [ ] Let the hourly pipeline keep growing the dataset and retrain on a larger history
- [ ] Add SHAP-based explanations for individual predictions
- [ ] Add automated tests for the feature engineering and prediction functions

---

## 👤 Author

**Bharat Kumar**
10Pearls Shine Internship Program — Data Science Track

[![GitHub](https://img.shields.io/badge/GitHub-BharatKumar292-181717?style=flat&logo=github)](https://github.com/BharatKumar292)
