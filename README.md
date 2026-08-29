# AQI Predictor - Sukkur, Karachi, Lahore

Predicts current and 3-day-ahead Air Quality Index for three Pakistani
cities, using data from the OpenWeather API. Built as part of the
10Pearls Shine Internship (Data Science track).

## What's in this project

| Part | File | What it does |
|---|---|---|
| Feature pipeline | `src/feature_pipeline.py` | Fetches live pollution + weather data every hour |
| Backfill | `src/backfill.py` | Pulls ~2 months of historical pollution data (one-time) |
| Current AQI model | `src/train_current.py` | Predicts AQI from live pollutant/weather readings |
| Forecast models | `src/train_forecast.py` | Predicts AQI 24h / 48h / 72h ahead |
| Automation | `.github/workflows/*.yml` | Runs the above on a schedule via GitHub Actions |
| Dashboard | `app.py` | Streamlit app showing everything |

## Why two separate models?

This is the most important design decision in the project, so it's
worth explaining clearly:

**`train_current.py`** answers "what is the AQI right now, given these
pollutant readings?" It only uses raw sensor-style features (PM2.5,
PM10, CO, NO2, temperature, etc). It is NOT allowed to use AQI itself
as an input, because AQI is basically calculated FROM these pollutant
values (see `feature_engineering.py`, `compute_aqi()`). Using AQI to
predict AQI would just be circular - the model would be "cheating" by
seeing something close to the answer.

**`train_forecast.py`** answers "what will AQI be in 1-3 days?" Nobody
has real pollutant readings from the future, so this model instead
looks at how AQI moved recently - the current AQI, AQI from a few hours
ago, AQI from a day ago, and a rolling average. Using PAST AQI values
to predict a FUTURE AQI value is normal time-series forecasting, not
leakage, because the model never actually sees the value it's trying
to predict.

## Setup

```cmd
pip install -r requirements.txt
set OPENWEATHER_API_KEY=your_key_here
set HOPSWORKS_API_KEY=your_hopsworks_key_here

python -m src.feature_pipeline    # fetch current data
python -m src.backfill            # pull ~2 months of history (run once)
python -m src.train_current       # train the current-AQI model
python -m src.train_forecast      # train the 3 forecast models
streamlit run app.py              # launch the dashboard
```

## Known limitations

- OpenWeather's free plan only gives historical WEATHER data for about
  a week back (historical POLLUTION data goes back much further, to
  Nov 2020, so that part is fine). This means backfilled rows only have
  pollutant columns filled in - temperature, humidity, wind, etc. are
  empty for old rows and only fill in going forward as the hourly
  pipeline runs. Because of this, the current-AQI model can only train
  on the smaller set of rows that do have weather data.
- The 3-day forecast doesn't use real future weather forecasts (that
  would need a paid API tier), so it relies on recent AQI patterns
  instead. Accuracy naturally drops the further out you forecast.
- Wind direction, cloudiness, and rainfall are only available for
  live/current readings, not historical ones, so they're shown on the
  dashboard but not used as model training features.

## Setting up GitHub Actions automation

1. Push this project to a GitHub repo
2. Repo Settings -> Secrets and variables -> Actions -> add:
   - `OPENWEATHER_API_KEY`
   - `HOPSWORKS_API_KEY`
3. Go to the Actions tab, run each workflow once manually to check it works

## Setting up Hopsworks

1. Sign up free at hopsworks.ai, create a project
2. Account Settings -> API Keys -> create one, copy it
3. Set it as `HOPSWORKS_API_KEY` locally and as a GitHub secret
4. Run any of the pipeline scripts - you'll see `[hopsworks] ...` log
   lines confirming it connected

If `HOPSWORKS_API_KEY` isn't set, everything still works using the
local CSV/pickle files - Hopsworks is additive, not required for the
scripts to run.
