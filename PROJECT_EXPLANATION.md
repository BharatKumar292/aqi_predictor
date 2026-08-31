# Project Explanation - notes for talking to my mentor

Writing this out so I actually understand my own project well enough to
explain it without reading off a script.

## What's the point of this project?

Air quality in a lot of Pakistani cities gets pretty bad, and most people
don't have an easy way to check how bad it is right now, let alone what
it'll look like in a few days. So the idea here is: build something that
tells you both. Current AQI, plus a 3-day forecast, for Sukkur, Karachi,
and Lahore.

## Why pick AQI prediction as a project?

Honestly, it's a good excuse to actually go through the full ML pipeline
properly - not just load a CSV and call `.fit()`. The data is free and
public, there's a clear number to predict, and because "predict AQI now"
and "predict AQI in 3 days" are genuinely different problems, it forces
you to think about things like data leakage instead of just skipping past
them.

## Where's the data coming from?

OpenWeather - specifically their Air Pollution API and Weather API. The
pollution one gives PM2.5, PM10, CO, NO, NO2, O3, SO2, and NH3. The
weather one gives temperature, humidity, pressure, wind (speed and
direction), cloud cover, and rain.

## Why these particular features?

The pollutants were an easy call - they're always returned, no missing
data, and they're directly what AQI is based on. Temperature, humidity,
pressure and wind speed I kept too, since they affect how pollution
spreads out and they're available on every live API call.

Wind direction, cloudiness and rain didn't make the cut for training,
though they do show up on the dashboard. The issue is OpenWeather's free
tier doesn't give you historical weather that far back, so if I required
those columns to be filled in, I'd barely have any training data left.

## Why Sukkur, Karachi, and Lahore specifically?

That's what the project brief asked for. They also happen to be a decent
spread geographically - Sukkur's smaller and inland, Karachi's a big
coastal city, Lahore's a big inland one - so their AQI patterns don't all
look the same, which is what makes the city-comparison tab on the
dashboard actually useful instead of three overlapping lines.

## How does Hopsworks fit in?

There's a small helper file, `src/hopsworks_client.py`, that both the
feature pipeline and the backfill script call to push data into a feature
group called `aqi_features`. It's meant to be the single source of truth
- the training scripts pull from there. If Hopsworks happens to be down
or unreachable for some reason, things fall back to the local CSV instead
so the project doesn't just stop working.

## How's the model actually trained?

Two scripts, two different approaches.

`train_current.py` only uses rows where BOTH pollutant and weather data
are present, and splits train/test randomly, because each row here is
its own independent snapshot - there's no "future" to accidentally leak.

`train_forecast.py` is trickier. It splits by TIME instead - train on
older data, test on the most recent chunk. If I split randomly here, some
training rows could end up being from after some of the test rows, and
the model would essentially get to peek at the future during training.
That's a mistake I want to avoid, not something to hide after the fact.

Both scripts try Ridge Regression and Random Forest, and whichever scores
lower RMSE on the test set gets saved.

## Why that particular model, though?

Lowest RMSE on data the model hasn't seen. For the forecast models I also
check against a naive baseline - basically "assume AQI stays the same" -
because if a trained model can't beat that, it's not really adding
anything.

## How does the 3-day forecast actually work?

By looking at how AQI has been trending - AQI a few hours ago, AQI a day
ago, a short rolling average - not by copying today's reading three
times. I don't have real future weather data to feed in, so this is a
genuine limitation, and I'd rather say that plainly than pretend the
forecast is more solid than it is.

## How does the dashboard talk to the models?

Pretty directly, actually - Streamlit just loads the saved `.pkl` files
with Python's `pickle` module and calls `.predict()`. No separate API
layer needed since it's all the same Python process.

## What doesn't work great right now

- Limited historical weather (mentioned above)
- Training data only covers a couple months so far - more history would
  help it learn seasonal stuff
- No real future weather forecast, so the further out the prediction, the
  shakier it gets
- Only tried Ridge and Random Forest - the brief mentioned XGBoost/LSTM
  as optional extras, didn't get to those this time around

## If I had more time

- Feed in an actual weather forecast, not just past weather
- Try XGBoost, maybe a basic LSTM, see if either beats what I have now
- Let the hourly pipeline keep running and retrain on the bigger dataset
  later
- Proper SHAP explanations instead of the basic feature importance I'm
  showing now