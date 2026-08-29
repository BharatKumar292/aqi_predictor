# Project Explanation - for presenting to my mentor

This is my own plain-English walkthrough of the project, so I can explain
it without reading from a script.

## What problem are we solving?

Air pollution in Pakistani cities is often bad enough to affect health,
but people don't always know how bad it is right now or how it's going
to change over the next few days. This project builds a system that
tells you the current AQI and forecasts it 3 days ahead, for three
cities: Sukkur, Karachi, and Lahore.

## Why AQI prediction?

It's a good project for learning the full ML pipeline because it has
real, freely available data, a clear number to predict, and a natural
reason to build both a "right now" model and a "future" model - which
forces you to actually think about data leakage and forecasting
properly instead of just calling `model.fit()` once.

## Where does the data come from?

OpenWeather's Air Pollution API and Weather API. The pollution API gives
pollutant concentrations (PM2.5, PM10, CO, NO, NO2, O3, SO2, NH3). The
weather API gives temperature, humidity, pressure, wind speed and
direction, cloudiness, and rain.

## Why these features?

I kept the pollutants because they're always returned and are directly
tied to air quality. I kept temperature/humidity/pressure/wind because
they affect how pollution disperses and they're always available too.
Wind direction, cloudiness, and rain are shown on the dashboard but not
used to train the forecast models, because OpenWeather's free plan
doesn't give historical weather data far enough back - so I'd have almost
no training rows if I required those columns to be filled in.

## Why these three cities?

The project brief specifically asked for Sukkur, Karachi, and Lahore.
They're also a reasonable mix - Sukkur is smaller and inland, Karachi is
a large coastal city, Lahore is a large inland city - so the AQI
patterns aren't identical across them, which makes the city-comparison
part of the dashboard actually meaningful.

## How is the data stored in Hopsworks?

The feature pipeline and backfill script both call a small helper
(`src/hopsworks_client.py`) that logs into Hopsworks and pushes the
data into a feature group called `aqi_features`. Hopsworks is the
"single source of truth" - the training scripts read from it (falling
back to the local CSV if Hopsworks isn't reachable, so the project
doesn't break if Hopsworks has a hiccup).

## How is the model trained?

Two separate training scripts:
- `train_current.py` trains on rows that have both pollutant AND
  weather data, using a normal random train/test split, since each row
  is an independent snapshot (not a future prediction).
- `train_forecast.py` trains three models (24h, 48h, 72h ahead), using
  a TIME-based train/test split (train on older data, test on the most
  recent chunk) - this matters because a random split would let the
  model accidentally train on data from after the test period, which
  would make the results look better than they really are.

Both scripts try Ridge Regression and Random Forest, and keep whichever
one gets the lower RMSE on the test set.

## Why was the final model selected?

Whichever model has the lowest RMSE on the held-out test data. I also
compare against a "naive baseline" (assume AQI doesn't change) for the
forecast models - if a trained model can't beat that, it isn't actually
useful.

## How is the 3-day forecast generated?

By training on how AQI has moved in the past relative to itself
(yesterday's AQI, AQI 3 hours ago, a 6-hour rolling average, etc), not
by copying today's number. Because I don't have real future weather
data, this is a fair and honest limitation - it means forecast accuracy
gets less certain the further ahead you look, which the dashboard and
report both mention directly instead of hiding it.

## How does Streamlit talk to the model?

The dashboard loads the saved `.pkl` model files directly with Python's
`pickle` module and calls `.predict()` on them - no separate API server,
since Streamlit runs the same Python process that can just import the
model.

## What are the limitations?

- Limited historical weather data (explained above)
- Training data currently covers about 2 months - more history would
  help the model learn seasonal patterns
- No real future weather forecast used, so 3-day accuracy is naturally
  lower than 24h accuracy
- Only Ridge Regression and Random Forest were tried - more advanced
  models (XGBoost, LSTM) were mentioned as optional in the brief but
  not required for this version

## What could be improved in the future?

- Add a real weather forecast API as an input to the forecast models
- Try XGBoost or a simple LSTM and compare against Ridge/Random Forest
- Collect more historical data over time (the hourly pipeline keeps
  adding to it automatically)
- Add SHAP explanations for individual predictions, not just basic
  feature importance
